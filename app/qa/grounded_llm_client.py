from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from app.qa.text_utils import normalize_text, split_sentences, token_set


ReasoningMode = Literal["text", "table", "formula", "figure", "multi_span"]

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai-compatible", "openai_compatible"}
OLLAMA_PROVIDERS = {"ollama"}
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_API_KEY = "ollama"
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b-instruct"


@dataclass(slots=True)
class EvidencePacket:
    evidence_id: str
    chunk_id: str
    modality: ReasoningMode
    text: str
    source_name: str | None = None
    doc_id: str | None = None
    page: int | None = None
    section: str | None = None
    heading_path: list[str] = field(default_factory=list)
    score: float = 0.0
    table_text: str | None = None
    table_rows: list[Any] = field(default_factory=list)
    table_json: Any | None = None
    formula_text: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = _prompt_safe_metadata(self.metadata)
        return payload


@dataclass(slots=True)
class GroundedLLMRequest:
    question: str
    query_type: str
    expected_answer_shape: str
    reasoning_mode: ReasoningMode
    evidence_packets: list[EvidencePacket]

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "query_type": self.query_type,
            "expected_answer_shape": self.expected_answer_shape,
            "reasoning_mode": self.reasoning_mode,
            "evidence_packets": [packet.prompt_dict() for packet in self.evidence_packets],
        }


@dataclass(slots=True)
class GroundedLLMResponse:
    decision: Literal["answer", "insufficient_evidence"]
    answer: str
    used_evidence_ids: list[str]
    reasoning_mode: ReasoningMode = "text"
    confidence: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)


class BaseGroundedLLMClient(Protocol):
    provider_name: str

    def generate(self, request: GroundedLLMRequest) -> GroundedLLMResponse:
        ...


class DummyGroundedLLMClient:
    """Small deterministic client for offline smoke tests."""

    provider_name = "dummy"

    def generate(self, request: GroundedLLMRequest) -> GroundedLLMResponse:
        for packet in request.evidence_packets:
            sentence = _first_relevant_sentence(request.question, packet.text)
            if sentence:
                return GroundedLLMResponse(
                    decision="answer",
                    answer=sentence,
                    used_evidence_ids=[packet.evidence_id],
                    reasoning_mode=request.reasoning_mode,
                    confidence=0.50,
                    raw_response={"provider": "dummy"},
                )
        return GroundedLLMResponse(
            decision="insufficient_evidence",
            answer="",
            used_evidence_ids=[],
            reasoning_mode=request.reasoning_mode,
            confidence=0.0,
            raw_response={"provider": "dummy", "reason": "no_relevant_sentence"},
        )


class OpenAICompatibleGroundedLLMClient:
    """Minimal OpenAI-compatible chat-completions client used by Ollama/OpenAI."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        provider_name: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        if provider_name is not None:
            self.provider_name = provider_name
        self.base_url = (base_url or os.getenv("BOXTALK_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("BOXTALK_LLM_API_KEY")
        self.model = model or os.getenv("BOXTALK_LLM_MODEL") or "gpt-4o-mini"
        self.timeout_s = timeout_s

    def generate(self, request: GroundedLLMRequest) -> GroundedLLMResponse:
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_grounded_messages(request),
        }
        if self.provider_name == "ollama":
            request_payload["temperature"] = 0
        payload = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM provider HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM provider request failed: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM provider returned no choices")
        content = str((choices[0].get("message") or {}).get("content") or "")
        return response_from_payload(parse_llm_json(content), request.reasoning_mode)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def build_grounded_messages(request: GroundedLLMRequest) -> list[dict[str, str]]:
    system = (
        "Bạn là bộ sinh câu trả lời dựa trên bằng chứng.\n\n"
        "Quy tắc:\n"
        "1. Chỉ sử dụng thông tin trong evidence_packets được cung cấp.\n"
        "2. Không dùng kiến thức bên ngoài và không suy đoán giá trị không có trong bằng chứng.\n"
        "3. Nếu bằng chứng không đủ để trả lời, đặt abstain=true.\n"
        "4. Chỉ dùng evidence_id đã được cung cấp trong used_evidence_ids.\n"
        "5. Với bảng, giữ đúng quan hệ hàng, cột và ô; không ghép dữ liệu từ hàng/cột khác nhau.\n"
        "6. Trả lời bằng tiếng Việt, thành câu đầy đủ, ngắn gọn nhưng đủ ý.\n"
        "7. Không chỉ trả lời bằng một số hoặc một cụm ngắn nếu bằng chứng có điều kiện, ngoại lệ, thời hạn, ngưỡng hoặc hệ quả liên quan.\n"
        "8. Với câu hỏi liệt kê, chính sách hoặc so sánh, phải giữ đủ các trường hợp, điều kiện, ngoại lệ và mốc số có trong bằng chứng liên quan.\n"
        "9. Với câu hỏi Có/Không, bắt đầu bằng Có hoặc Không rồi nêu căn cứ ngắn gọn.\n"
        "10. Trả về JSON đúng schema, không thêm chữ ngoài JSON.\n\n"
        "Schema:\n"
        "{\n"
        '  "answer": "string",\n'
        '  "used_evidence_ids": ["E1"],\n'
        '  "abstain": false,\n'
        '  "reason": null,\n'
        '  "confidence": 1.0\n'
        "}"
    )
    user = {
        "task": "Grounded QA synthesis",
        "instructions": [
            "Use only the supplied evidence_packets.",
            "If a claim cannot be tied to at least one evidence_id, do not include it.",
            "If the evidence is insufficient, set abstain=true.",
            "Answer in Vietnamese with complete sentences.",
            "Keep the answer concise but preserve all relevant conditions, exceptions, numeric thresholds, time limits, and consequences from the evidence.",
            "For list, policy, or comparison questions, include every relevant case found in the supplied evidence.",
            "Do not answer with only a bare number or short phrase when the evidence contains additional required context.",
        ],
        "input": request.prompt_payload(),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_llm_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def response_from_payload(payload: dict[str, Any], default_mode: ReasoningMode) -> GroundedLLMResponse:
    used_raw = payload.get("used_evidence_ids") or []
    used_ids = [used_raw] if isinstance(used_raw, str) else [str(item) for item in used_raw if str(item).strip()]
    answer = str(payload.get("answer") or "").strip()
    decision = str(payload.get("decision") or "").strip().lower()
    if isinstance(payload.get("abstain"), bool):
        decision = "insufficient_evidence" if payload["abstain"] else "answer"
    if decision not in {"answer", "insufficient_evidence"}:
        decision = "answer" if answer and used_ids else "insufficient_evidence"

    mode = str(payload.get("reasoning_mode") or default_mode).strip().lower()
    if mode not in {"text", "table", "formula", "figure", "multi_span"}:
        mode = default_mode
    confidence = _coerce_confidence(payload.get("confidence"), default=1.0 if answer and used_ids else 0.0)
    return GroundedLLMResponse(
        decision=decision,  # type: ignore[arg-type]
        answer=answer,
        used_evidence_ids=used_ids,
        reasoning_mode=mode,  # type: ignore[arg-type]
        confidence=confidence,
        raw_response=dict(payload),
    )


def normalize_llm_provider_name(provider: str | None = None) -> str:
    selected = (provider or os.getenv("BOXTALK_LLM_PROVIDER") or "dummy").strip().lower()
    if selected in OPENAI_COMPATIBLE_PROVIDERS:
        return "openai-compatible"
    if selected in OLLAMA_PROVIDERS:
        return "ollama"
    return "dummy"


def make_grounded_llm_client(provider: str | None = None, *, timeout_s: float = 30.0) -> BaseGroundedLLMClient:
    normalized = normalize_llm_provider_name(provider)
    if normalized == "openai-compatible":
        return OpenAICompatibleGroundedLLMClient(timeout_s=timeout_s)
    if normalized == "ollama":
        return OpenAICompatibleGroundedLLMClient(
            base_url=os.getenv("BOXTALK_LLM_BASE_URL") or OLLAMA_DEFAULT_BASE_URL,
            api_key=os.getenv("BOXTALK_LLM_API_KEY") or OLLAMA_DEFAULT_API_KEY,
            model=os.getenv("BOXTALK_LLM_MODEL") or OLLAMA_DEFAULT_MODEL,
            provider_name="ollama",
            timeout_s=timeout_s,
        )
    return DummyGroundedLLMClient()


def provider_runtime_info(provider: str | None = None) -> dict[str, Any]:
    normalized = normalize_llm_provider_name(provider)
    base_url = os.getenv("BOXTALK_LLM_BASE_URL")
    model = os.getenv("BOXTALK_LLM_MODEL")
    api_key = os.getenv("BOXTALK_LLM_API_KEY")
    if normalized == "ollama":
        base_url = base_url or OLLAMA_DEFAULT_BASE_URL
        model = model or OLLAMA_DEFAULT_MODEL
        api_key = api_key or OLLAMA_DEFAULT_API_KEY
    missing_envs: list[str] = []
    if normalized == "openai-compatible":
        for name in ("BOXTALK_LLM_BASE_URL", "BOXTALK_LLM_API_KEY", "BOXTALK_LLM_MODEL"):
            if not os.getenv(name):
                missing_envs.append(name)
    return {
        "provider": normalized,
        "ready": not missing_envs,
        "missing_envs": missing_envs,
        "base_url": base_url if normalized in {"openai-compatible", "ollama"} else None,
        "model": model if normalized in {"openai-compatible", "ollama"} else None,
        "api_key_present": bool(api_key) if normalized in {"openai-compatible", "ollama"} else False,
        "env": {
            "BOXTALK_LLM_PROVIDER": os.getenv("BOXTALK_LLM_PROVIDER"),
            "BOXTALK_LLM_BASE_URL": os.getenv("BOXTALK_LLM_BASE_URL"),
            "BOXTALK_LLM_MODEL": os.getenv("BOXTALK_LLM_MODEL"),
            "BOXTALK_LLM_API_KEY_PRESENT": str(bool(api_key)),
        },
    }


def _prompt_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "block_type",
        "citation_target",
        "heading_path",
        "section",
        "table_id",
        "table_backend",
        "table_shape",
        "row_header",
        "col_header",
        "cell_text",
        "caption",
        "formula_text",
        "equation",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


def _first_relevant_sentence(question: str, text: str) -> str | None:
    question_terms = token_set(question)
    candidates: list[tuple[int, str]] = []
    for sentence in split_sentences(text):
        overlap = len(question_terms & token_set(sentence))
        if overlap:
            candidates.append((overlap, normalize_text(sentence)))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    sentences = split_sentences(text)
    return normalize_text(sentences[0]) if sentences else None


def _coerce_confidence(value: Any, *, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))
