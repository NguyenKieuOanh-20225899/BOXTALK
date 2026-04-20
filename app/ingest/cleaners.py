from __future__ import annotations

import copy
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable

from app.ingest.schemas import BlockNode


NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S+")
LEGAL_HEADING_RE = re.compile(
    r"^(chương|phần|mục|điều|khoản|điểm)\s+[0-9ivxlcdmA-Za-z]+[\.\:\-]?\s*",
    re.IGNORECASE,
)
LIST_ITEM_RE = re.compile(
    r"""^(
        \d+[\.\)] |
        [a-zA-Z][\.\)] |
        [ivxlcdmIVXLCDM]+[\.\)] |
        [•●▪■\-*]
    )\s+\S+""",
    re.VERBOSE,
)
METADATA_LINE_RE = re.compile(
    r"^[^:\n]{1,80}:\s+\S+"
)
PAGE_ONLY_RE = re.compile(r"^\s*\d{1,3}\s*$")


def clean_blocks(
    blocks: list[BlockNode],
    repeated_min_pages: int = 3,
    repeated_ratio_threshold: float = 0.6,
    merge_paragraph_lines: bool = True,
) -> list[BlockNode]:
    """
    Main cleaning pipeline:
    1) remove repeated headers/footers
    2) normalize block type and markdown
    3) remove obvious noise
    4) merge adjacent paragraph lines when appropriate
    """
    if not blocks:
        return []

    working = [copy.deepcopy(b) for b in blocks]
    working = sorted(working, key=lambda b: (b.page_index, b.reading_order))

    repeated_texts = detect_repeated_header_footer_candidates(
        working,
        min_pages=repeated_min_pages,
        repeated_ratio_threshold=repeated_ratio_threshold,
    )
    working = remove_repeated_blocks(working, repeated_texts)
    working = [normalize_block(b) for b in working]
    working = remove_obvious_noise(working)

    if merge_paragraph_lines:
        working = merge_adjacent_blocks(working)

    working = reindex_reading_order_per_page(working)
    return working


def detect_repeated_header_footer_candidates(
    blocks: list[BlockNode],
    min_pages: int = 3,
    repeated_ratio_threshold: float = 0.6,
) -> set[str]:
    """
    Detect repeated texts that appear across many pages, especially near
    top or bottom of page, which are likely headers/footers.
    """
    if not blocks:
        return set()

    pages = sorted({b.page_index for b in blocks})
    total_pages = len(pages)
    if total_pages == 0:
        return set()

    page_hits: dict[str, set[int]] = defaultdict(set)
    top_bottom_hits: dict[str, int] = defaultdict(int)

    for b in blocks:
        norm = normalize_text_for_matching(b.text)
        if not norm:
            continue

        # chỉ quan tâm block tương đối ngắn
        if len(norm) > 220:
            continue

        page_hits[norm].add(b.page_index)

        if is_top_or_bottom_block(b):
            top_bottom_hits[norm] += 1

    repeated: set[str] = set()
    for text, seen_pages in page_hits.items():
        seen_count = len(seen_pages)
        ratio = seen_count / total_pages if total_pages else 0.0

        if seen_count >= min_pages and ratio >= repeated_ratio_threshold:
            repeated.add(text)
            continue

        # Ưu tiên header/footer nếu nó xuất hiện nhiều lần gần mép trang
        if seen_count >= min_pages and top_bottom_hits.get(text, 0) >= max(2, min_pages - 1):
            repeated.add(text)

    return repeated


def remove_repeated_blocks(blocks: list[BlockNode], repeated_texts: set[str]) -> list[BlockNode]:
    out: list[BlockNode] = []
    for b in blocks:
        norm = normalize_text_for_matching(b.text)
        if norm in repeated_texts:
            continue
        out.append(b)
    return out


def normalize_block(block: BlockNode) -> BlockNode:
    text = normalize_inline_text(block.text)

    block = replace(block, text=text)

    inferred_type = infer_block_type(block)
    item_number = extract_item_number(text)

    md = to_markdown_from_type(text, inferred_type)

    meta = dict(block.meta or {})
    if item_number and not block.item_number:
        meta["detected_item_number"] = item_number

    return replace(
        block,
        block_type=inferred_type,
        markdown=md,
        item_number=block.item_number or item_number,
        meta=meta,
    )


def infer_block_type(block: BlockNode) -> str:
    text = (block.text or "").strip()
    if not text:
        return "paragraph"

    # giữ table/image nếu extractor sau này có set sẵn
    if block.block_type in {"table", "image", "figure"}:
        return block.block_type

    if looks_like_heading(text, block):
        return "heading"

    if looks_like_list_item(text):
        return "list_item"

    if looks_like_metadata_line(text):
        return "metadata"

    return "paragraph"


def looks_like_heading(text: str, block: BlockNode | None = None) -> bool:
    t = text.strip()

    if len(t) > 180:
        return False

    if LEGAL_HEADING_RE.match(t):
        return True

    if NUMBERED_HEADING_RE.match(t):
        # ví dụ:
        # 1. Thông tin học phần
        # 2.1 Các bước thực hiện
        # 1.2 Mục tiêu học phần
        # nhưng tránh nhầm với câu mô tả ngắn bắt đầu bằng số
        if looks_like_sentence_after_number(t):
            return False
        return True

    # uppercase ngắn
    alpha_chars = [c for c in t if c.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio > 0.85 and len(t.split()) <= 12:
            return True

    # title-like ngắn nằm riêng một dòng
    if len(t.split()) <= 10 and not t.endswith((".", ";")) and ":" not in t:
        if block and is_visually_heading_like(block):
            return True

    return False


def looks_like_sentence_after_number(text: str) -> bool:
    """
    Tránh nhầm các list item kiểu:
    1 Bước 1: ...
    2 Hiểu quy trình...
    """
    t = text.strip()

    # "1 Bước 1: ..." và "1 Hiểu ..."
    if re.match(r"^\d+\s+\S+", t):
        return True

    return False


def looks_like_list_item(text: str) -> bool:
    t = text.strip()

    if LIST_ITEM_RE.match(t):
        return True

    # hỗ trợ case OCR / extract lỗi:
    # "1 Hiểu quy trình..." hoặc "2 Bước 2: ..."
    if re.match(r"^\d+\s+\S+", t):
        return True

    return False


def looks_like_metadata_line(text: str) -> bool:
    t = text.strip()
    if not t:
        return False

    if METADATA_LINE_RE.match(t):
        return True

    common_keys = (
        "tên học phần",
        "mã học phần",
        "giảng viên",
        "email",
        "thời gian",
        "địa chỉ",
        "số điện thoại",
        "hạn nộp hồ sơ",
        "ngôn ngữ",
        "phiên bản",
    )
    lowered = t.lower()
    if any(lowered.startswith(k + ":") for k in common_keys):
        return True

    return False


def extract_item_number(text: str) -> str | None:
    t = text.strip()

    m = re.match(r"^(\d+(?:\.\d+)*\.?)\s+\S+", t)
    if m:
        return m.group(1)

    m = re.match(r"^([a-zA-Z][\.\)])\s+\S+", t)
    if m:
        return m.group(1)

    m = re.match(r"^([ivxlcdmIVXLCDM]+[\.\)])\s+\S+", t)
    if m:
        return m.group(1)

    m = re.match(r"^(•|●|▪|■|\-|\*)\s+\S+", t)
    if m:
        return m.group(1)

    m = re.match(r"^(khoản|điểm)\s+([0-9A-Za-z]+)\b", t, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    return None


def to_markdown_from_type(text: str, block_type: str) -> str:
    t = normalize_inline_text(text)

    if not t:
        return ""

    if block_type == "heading":
        return f"## {t}"

    if block_type == "list_item":
        stripped = strip_list_marker_keep_content(t)
        return f"- {stripped}"

    return t


def strip_list_marker_keep_content(text: str) -> str:
    t = text.strip()

    patterns = [
        r"^\d+[\.\)]\s+",
        r"^[a-zA-Z][\.\)]\s+",
        r"^[ivxlcdmIVXLCDM]+[\.\)]\s+",
        r"^[•●▪■\-*]\s+",
        r"^\d+\s+",
    ]
    for p in patterns:
        if re.match(p, t):
            return re.sub(p, "", t, count=1)

    return t


def remove_obvious_noise(blocks: list[BlockNode]) -> list[BlockNode]:
    out: list[BlockNode] = []

    for b in blocks:
        text = (b.text or "").strip()
        if not text:
            continue

        if PAGE_ONLY_RE.match(text):
            continue

        # artifact kiểu một ký tự lẻ
        if len(text) == 1 and not text.isalnum():
            continue

        # dòng rất ngắn toàn uppercase không có nhiều ý nghĩa
        if is_short_uppercase_noise(text):
            continue

        out.append(b)

    return out


def is_short_uppercase_noise(text: str) -> bool:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False

    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    if upper_ratio < 0.95:
        return False

    word_count = len(text.split())
    if word_count > 3:
        return False

    # giữ lại các heading kiểu "MỤC I" nếu hợp lệ
    if LEGAL_HEADING_RE.match(text):
        return False

    return len(text) <= 12


def merge_adjacent_blocks(blocks: list[BlockNode]) -> list[BlockNode]:
    """
    Merge mainly:
    - paragraph + paragraph
    - metadata + metadata? -> không merge
    - list_item + list_item? -> không merge
    - heading giữ riêng
    """
    if not blocks:
        return []

    grouped: dict[int, list[BlockNode]] = defaultdict(list)
    for b in blocks:
        grouped[b.page_index].append(b)

    merged_all: list[BlockNode] = []

    for page_index in sorted(grouped):
        page_blocks = sorted(grouped[page_index], key=lambda b: b.reading_order)
        merged_page = merge_blocks_in_page(page_blocks)
        merged_all.extend(merged_page)

    return merged_all


def merge_blocks_in_page(blocks: list[BlockNode]) -> list[BlockNode]:
    if not blocks:
        return []

    out: list[BlockNode] = []
    buffer: BlockNode | None = None

    for b in blocks:
        if buffer is None:
            buffer = copy.deepcopy(b)
            continue

        if should_merge(buffer, b):
            buffer = merge_two_blocks(buffer, b)
        else:
            out.append(buffer)
            buffer = copy.deepcopy(b)

    if buffer is not None:
        out.append(buffer)

    return out


def should_merge(left: BlockNode, right: BlockNode) -> bool:
    if left.page_index != right.page_index:
        return False

    if left.block_type != "paragraph" or right.block_type != "paragraph":
        return False

    lt = left.text.strip()
    rt = right.text.strip()

    if not lt or not rt:
        return False

    if looks_like_metadata_line(lt) or looks_like_metadata_line(rt):
        return False

    if looks_like_heading(lt, left) or looks_like_heading(rt, right):
        return False

    if looks_like_list_item(lt) or looks_like_list_item(rt):
        return False

    # nối nếu dòng trước chưa kết thúc câu
    if not lt.endswith((".", "!", "?", ";", ":")):
        return True

    # nối nếu dòng sau bắt đầu bằng chữ thường
    if rt and rt[0].islower():
        return True

    return False


def merge_two_blocks(left: BlockNode, right: BlockNode) -> BlockNode:
    joined_text = smart_join_text(left.text, right.text)
    joined_md = joined_text

    meta = dict(left.meta or {})
    right_ids = meta.get("merged_from_block_ids", [])
    if left.block_id not in right_ids:
        base_ids = [left.block_id]
    else:
        base_ids = right_ids

    merged_ids = list(dict.fromkeys(base_ids + [right.block_id]))
    meta["merged_from_block_ids"] = merged_ids

    bbox = merge_bbox(left.bbox, right.bbox)

    return replace(
        left,
        text=joined_text,
        markdown=joined_md,
        bbox=bbox,
        meta=meta,
    )


def smart_join_text(left: str, right: str) -> str:
    l = normalize_inline_text(left)
    r = normalize_inline_text(right)

    if not l:
        return r
    if not r:
        return l

    # bỏ gạch nối xuống dòng kiểu "thông-\ntin"
    if l.endswith("-") and r and r[0].islower():
        return l[:-1] + r

    return f"{l} {r}"


def merge_bbox(
    b1: tuple[float, float, float, float] | None,
    b2: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if not b1:
        return b2
    if not b2:
        return b1

    x0 = min(b1[0], b2[0])
    y0 = min(b1[1], b2[1])
    x1 = max(b1[2], b2[2])
    y1 = max(b1[3], b2[3])
    return (x0, y0, x1, y1)


def reindex_reading_order_per_page(blocks: list[BlockNode]) -> list[BlockNode]:
    grouped: dict[int, list[BlockNode]] = defaultdict(list)
    for b in blocks:
        grouped[b.page_index].append(b)

    out: list[BlockNode] = []
    for page_index in sorted(grouped):
        page_blocks = sorted(grouped[page_index], key=lambda b: b.reading_order)
        for i, b in enumerate(page_blocks):
            out.append(replace(b, reading_order=i))
    return out


def is_visually_heading_like(block: BlockNode) -> bool:
    """
    Placeholder nhẹ cho giai đoạn hiện tại.
    Nếu sau này có font size / span info thì nâng cấp tại đây.
    """
    text = (block.text or "").strip()

    if len(text.split()) <= 10 and len(text) <= 100:
        return True

    return False


def is_top_or_bottom_block(block: BlockNode) -> bool:
    if not block.bbox:
        return False

    _, y0, _, y1 = block.bbox
    # heuristic an toàn, vì hiện chưa có page height chuẩn ở schema
    # giả định vùng gần mép trên / dưới của PDF thường có y nhỏ hoặc rất lớn
    return y0 < 90 or y1 > 720


def normalize_inline_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\xad", "")  # soft hyphen
    text = text.replace("•", "• ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_for_matching(text: str) -> str:
    text = normalize_inline_text(text).lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def block_stats_by_page(blocks: list[BlockNode]) -> dict[int, Counter]:
    stats: dict[int, Counter] = {}
    grouped: dict[int, list[BlockNode]] = defaultdict(list)
    for b in blocks:
        grouped[b.page_index].append(b)

    for page_index, page_blocks in grouped.items():
        c = Counter()
        for b in page_blocks:
            c[b.block_type] += 1
        stats[page_index] = c

    return stats
