from __future__ import annotations

import re


class QueryRouter:
    """Small deterministic query-type router for routed PDF QA."""

    POLICY_TERMS = {
        "policy",
        "regulation",
        "rule",
        "requirement",
        "must",
        "may",
        "eligible",
        "quy định",
        "quy chế",
        "điều kiện",
        "bắt buộc",
        "được phép",
        "không được",
        "có được",
        "cần đáp ứng",
        "hậu quả",
        "trường hợp",
        "bị hạn chế",
        "xét công nhận",
        "nội quy",
    }
    STRONG_POLICY_TERMS = {
        "điều kiện",
        "được phép",
        "có được",
        "cần đáp ứng",
        "hậu quả",
        "trường hợp",
        "bị",
        "phải",
        "không được",
        "xét công nhận",
    }
    PROCEDURAL_TERMS = {
        "how",
        "procedure",
        "process",
        "steps",
        "apply",
        "submit",
        "cách",
        "làm sao",
        "quy trình",
        "thủ tục",
        "hướng dẫn",
        "bước",
        "giai đoạn",
        "quá trình",
        "đăng ký học tập",
        "cộng hoặc trừ",
    }
    COMPARISON_TERMS = {
        "compare",
        "difference",
        "different",
        "differ",
        "versus",
        "vs",
        "than",
        "so sánh",
        "khác nhau",
        "khác biệt",
        "giữa",
    }
    DEFINITION_TERMS = {
        "what is",
        "meaning",
        "definition",
        "là gì",
        "định nghĩa",
        "khái niệm",
        "mô tả",
        "được gọi",
        "đặc điểm",
    }
    FACTOID_TERMS = {
        "bao nhiêu",
        "bao lâu",
        "khi nào",
        "từ khi nào",
        "ai",
        "gì",
        "nào",
        "mấy",
        "ở đâu",
        "tối đa",
        "tối thiểu",
    }

    def route(self, question: str) -> str:
        q = question.lower().strip()
        if any(term in q for term in self.COMPARISON_TERMS):
            return "comparison"
        if any(term in q for term in self.PROCEDURAL_TERMS):
            return "procedural"
        if self._is_definition(q):
            return "definition"
        if self._is_factoid(q):
            return "factoid"
        if any(term in q for term in self.POLICY_TERMS):
            return "policy"
        if any(term in q for term in {"prerequisite", "đồng thời", "vừa", "cùng lúc"}) and len(q.split()) > 10:
            return "multi_hop"
        if len(q.split()) <= 6:
            return "factoid"
        return "ambiguous"

    def _is_definition(self, question: str) -> bool:
        if question.startswith("điều kiện") or "điều kiện để" in question:
            return False
        return any(term in question for term in self.DEFINITION_TERMS)

    def _is_factoid(self, question: str) -> bool:
        if not any(term in question for term in self.FACTOID_TERMS):
            return False
        if any(term in question for term in self.STRONG_POLICY_TERMS):
            return False
        return True
