from __future__ import annotations

import re
import unicodedata


WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "do",
    "does",
    "did",
    "must",
    "have",
    "has",
    "with",
    "cua",
    "của",
    "la",
    "là",
    "gi",
    "gì",
    "nhung",
    "những",
    "cac",
    "các",
    "cho",
    "trong",
    "theo",
    "nhu",
    "như",
    "của",
    "là",
    "gì",
    "những",
    "các",
    "được",
    "và",
    "với",
    "khi",
    "nào",
    "bao",
    "nhiêu",
    "lâu",
    "người",
    "học",
}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = re.sub(r"\bTC\b", "tín chỉ", normalized, flags=re.I)
    normalized = re.sub(r"\bCTĐT\b", "chương trình đào tạo", normalized, flags=re.I)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_for_match(text: str) -> str:
    return normalize_text(text).casefold()


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = [token.casefold() for token in WORD_RE.findall(normalize_text(text))]
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def split_sentences(text: str) -> list[str]:
    compact = normalize_text(text)
    if not compact:
        return []
    parts = [part.strip(" -•\t") for part in SENTENCE_SPLIT_RE.split(compact) if part.strip()]
    if len(parts) <= 1 and len(compact) > 260:
        parts = [part.strip() for part in re.split(r";\s+|,\s+(?=[A-Z0-9])", compact) if part.strip()]
    return parts or [compact]


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction, keep_stopwords=False)
    gold_tokens = tokenize(gold, keep_stopwords=False)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(token, 0), count) for token, count in gold_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_text(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return normalize_for_match(needle) in normalize_for_match(haystack)
