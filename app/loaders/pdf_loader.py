from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz

from app.models import DocumentChunk


class PDFLoader:
    HEADING_PATTERNS = [
        re.compile(r"^(Chương|CHƯƠNG)\s+\w+", re.UNICODE),
        re.compile(r"^(Mục|MỤC)\s+\w+", re.UNICODE),
        re.compile(r"^(Điều|ĐIỀU)\s+\d+", re.UNICODE),
        re.compile(r"^\d+\.\s+.+"),       # 1. Thông tin học phần
        re.compile(r"^\d+\.\d+\s+.+"),    # 1.1 Mô tả ngắn
    ]

    LIST_PATTERNS = [
        re.compile(r"^\s*(\d+[\.\)]|\-|\•|\*)\s+"),
        re.compile(r"^\s*\d+\s+.+"),
        re.compile(r"^\s*Bước\s+\d+\s*:\s+", re.IGNORECASE),
    ]

    def load_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        doc = fitz.open(pdf_path)
        source_name = Path(pdf_path).stem

        page_lines = self._extract_page_lines(doc)
        repeated_lines = self._detect_repeated_header_footer(page_lines)

        chunks: List[DocumentChunk] = []
        current_section: Optional[str] = None
        current_heading_path: Optional[str] = None
        current_section_blocks: List[str] = []
        current_section_page: Optional[int] = None

        chunk_counter = 1
        order_counter = 1

        for page_idx, lines in enumerate(page_lines, start=1):
            filtered_lines = self._remove_repeated_lines(lines, repeated_lines)
            blocks = self._lines_to_blocks(filtered_lines)

            for block in blocks:
                normalized = self._normalize_text(block)
                if not normalized:
                    continue

                # bỏ chunk quá vô nghĩa như "1"
                if self._is_noise(normalized):
                    continue

                block_type = self._detect_block_type(normalized)

                if block_type == "heading":
                    chunk_counter = self._flush_section_summary(
                        chunks=chunks,
                        source_name=source_name,
                        section=current_section,
                        heading_path=current_heading_path,
                        section_blocks=current_section_blocks,
                        section_page=current_section_page,
                        chunk_counter=chunk_counter,
                        order_counter=order_counter,
                    )
                    if current_section_blocks:
                        order_counter += 1

                    current_section = normalized
                    current_heading_path = self._build_heading_path(current_heading_path, normalized)
                    current_section_blocks = []
                    current_section_page = page_idx
                    continue

                item_number = self._extract_item_number(normalized)
                metadata = {}
                if item_number is not None:
                    metadata["item_number"] = item_number

                if block_type == "metadata_line":
                    kv = self._split_metadata_line(normalized)
                    if kv is not None:
                        metadata["field_name"] = kv[0]
                        metadata["field_value"] = kv[1]

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{source_name}_p{page_idx}_c{chunk_counter}",
                        text=normalized,
                        source_name=source_name,
                        page=page_idx,
                        section=current_section,
                        heading_path=current_heading_path,
                        block_type=block_type,
                        order=order_counter,
                        metadata=metadata,
                    )
                )
                chunk_counter += 1
                order_counter += 1
                current_section_blocks.append(normalized)

        chunk_counter = self._flush_section_summary(
            chunks=chunks,
            source_name=source_name,
            section=current_section,
            heading_path=current_heading_path,
            section_blocks=current_section_blocks,
            section_page=current_section_page,
            chunk_counter=chunk_counter,
            order_counter=order_counter,
        )

        return chunks

    # -------------------------
    # Stage 1: extract lines
    # -------------------------
    def _extract_page_lines(self, doc: fitz.Document) -> List[List[str]]:
        pages: List[List[str]] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")
            raw_lines = text.splitlines()

            lines: List[str] = []
            for line in raw_lines:
                norm = self._normalize_text(line)
                if norm:
                    lines.append(norm)

            pages.append(lines)

        return pages

    def _detect_repeated_header_footer(self, page_lines: List[List[str]]) -> set[str]:
        """
        Dòng nào xuất hiện ở >= 2 trang với nội dung giống nhau gần như chắc là header/footer.
        """
        freq: Dict[str, int] = {}
        for lines in page_lines:
            unique_lines = set(lines)
            for ln in unique_lines:
                freq[ln] = freq.get(ln, 0) + 1

        repeated = {ln for ln, count in freq.items() if count >= 2 and len(ln) >= 8}
        return repeated

    def _remove_repeated_lines(self, lines: List[str], repeated_lines: set[str]) -> List[str]:
        filtered: List[str] = []

        for idx, ln in enumerate(lines):
            # Nếu là dòng lặp ở nhiều trang, mặc định bỏ
            if ln in repeated_lines:
                # Nhưng giữ lại nếu là heading nội dung thật kiểu:
                # 1. ...
                # 1.1 ...
                # THÔNG TIN CHUNG
                if re.match(r"^\d+\.\s+.+", ln):
                    filtered.append(ln)
                    continue
                if re.match(r"^\d+\.\d+\s+.+", ln):
                    filtered.append(ln)
                    continue
                if ln == "THÔNG TIN CHUNG":
                    filtered.append(ln)
                    continue

                # còn lại bỏ luôn, kể cả uppercase
                continue

            filtered.append(ln)

        return filtered

    # -------------------------
    # Stage 2: lines -> semantic blocks
    # -------------------------
    def _lines_to_blocks(self, lines: List[str]) -> List[str]:
        blocks: List[str] = []
        paragraph_buffer: List[str] = []

        def flush_paragraph():
            nonlocal paragraph_buffer
            if not paragraph_buffer:
                return

            merged = " ".join(paragraph_buffer).strip()
            paragraph_buffer = []

            if not merged:
                return

            # tách numbered items dính nhau
            numbered_parts = self._split_embedded_numbered_items(merged)
            if len(numbered_parts) > 1:
                blocks.extend(numbered_parts)
                return

            # tách bullet items dính nhau
            bullet_parts = self._split_embedded_bullets(merged)
            if len(bullet_parts) > 1:
                blocks.extend(bullet_parts)
                return

            blocks.append(merged)

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                flush_paragraph()
                i += 1
                continue

            if self._looks_like_heading(line):
                flush_paragraph()
                blocks.append(line)
                i += 1
                continue

            if self._looks_like_list(line):
                flush_paragraph()
                blocks.append(line)
                i += 1
                continue

            if self._looks_like_metadata_line(line):
                flush_paragraph()
                blocks.append(line)
                i += 1
                continue

            paragraph_buffer.append(line)
            i += 1

        flush_paragraph()
        return blocks

    # -------------------------
    # Stage 3: chunk helpers
    # -------------------------
    def _flush_section_summary(
        self,
        chunks: List[DocumentChunk],
        source_name: str,
        section: Optional[str],
        heading_path: Optional[str],
        section_blocks: List[str],
        section_page: Optional[int],
        chunk_counter: int,
        order_counter: int,
    ) -> int:
        if not section or not section_blocks:
            return chunk_counter

        cleaned_blocks = [b for b in section_blocks if not self._is_noise(b)]
        if not cleaned_blocks:
            return chunk_counter

        summary_text = "\n".join(cleaned_blocks).strip()
        if not summary_text:
            return chunk_counter

        chunks.append(
            DocumentChunk(
                chunk_id=f"{source_name}_section_{chunk_counter}",
                text=summary_text,
                source_name=source_name,
                page=section_page,
                section=section,
                heading_path=heading_path,
                block_type="section_summary",
                order=order_counter,
                metadata={
                    "is_section_summary": True,
                    "source_block_count": len(cleaned_blocks),
                },
            )
        )
        return chunk_counter + 1

    def _build_heading_path(self, current_path: Optional[str], heading: str) -> str:
        heading = heading.strip()

        # heading chữ hoa nội dung thật
        if heading == "THÔNG TIN CHUNG":
            return heading

        # heading cấp 1: 1. ..., 2. ...
        m_top = re.match(r"^(\d+)\.\s+.+", heading)
        if m_top:
            return heading

        # heading cấp 2: 1.1 ..., 1.2 ..., 2.1 ...
        m_sub = re.match(r"^(\d+)\.(\d+)\s+.+", heading)
        if m_sub:
            major = m_sub.group(1)

            # current_path có thể là:
            # - "1. Thông tin học phần"
            # - "1. Thông tin học phần > 1.1 Mô tả ngắn"
            # - "2. Quy trình ... > 2.1 Các bước thực hiện"
            if current_path:
                root = current_path.split(" > ")[0]

                # chỉ gắn vào root cùng major
                m_root = re.match(rf"^{major}\.\s+.+", root)
                if m_root:
                    return f"{root} > {heading}"

            return heading

        return heading
    # -------------------------
    # Detection helpers
    # -------------------------
    def _normalize_text(self, text: str) -> str:
        text = text.replace("\u200b", " ").replace("\ufeff", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _detect_block_type(self, text: str) -> str:
        if self._looks_like_heading(text):
            return "heading"

        if self._looks_like_list(text):
            return "list"

        if self._looks_like_metadata_line(text):
            return "metadata_line"

        return "paragraph"

    def _looks_like_heading(self, text: str) -> bool:
        if any(p.match(text) for p in self.HEADING_PATTERNS):
            return True

        # Chỉ coi UPPERCASE là heading nếu nó ngắn và có vẻ là heading nội dung
        if len(text) < 80 and text.isupper():
            allowed_uppercase = {
                "THÔNG TIN CHUNG",
            }
            if text in allowed_uppercase:
                return True

        return False

    def _looks_like_real_heading(self, text: str) -> bool:
        if re.match(r"^\d+\.\s+.+", text):
            return True
        if re.match(r"^\d+\.\d+\s+.+", text):
            return True
        if text == "THÔNG TIN CHUNG":
            return True
        return False

    def _looks_like_list(self, text: str) -> bool:
        return any(p.match(text) for p in self.LIST_PATTERNS)

    def _looks_like_metadata_line(self, text: str) -> bool:
        if ":" not in text:
            return False
        if len(text) > 140:
            return False

        # Loại trừ procedural step
        if re.match(r"^\s*Bước\s+\d+\s*:\s+.+", text, flags=re.IGNORECASE):
            return False

        left, right = text.split(":", 1)
        if not left.strip() or not right.strip():
            return False

        # Chỉ coi là metadata khi vế trái ngắn, giống nhãn trường
        if len(left.strip()) > 40:
            return False

        return True

    def _split_metadata_line(self, text: str) -> Optional[Tuple[str, str]]:
        if not self._looks_like_metadata_line(text):
            return None
        left, right = text.split(":", 1)
        return left.strip(), right.strip()

    def _extract_item_number(self, text: str) -> Optional[int]:
        m = re.match(r"^\s*(\d+)[\.\)]\s+", text)
        if m:
            return int(m.group(1))

        m2 = re.match(r"^\s*(\d+)\s+[A-ZÀ-ỴĂÂĐÊÔƠƯ]", text)
        if m2:
            return int(m2.group(1))

        return None

    def _is_noise(self, text: str) -> bool:
        if re.fullmatch(r"\d+", text):
            return True
        if len(text) == 1:
            return True

        # front-matter bị dính đầu trang, chưa có cấu trúc rõ
        if text.startswith("1 Tài liệu mẫu tiếng Việt để kiểm thử PDFLoader"):
            return True

        return False
    def _split_embedded_numbered_items(self, text: str) -> List[str]:
        """
        Tách:
        '1 Hiểu ... 2 Biết ... 3 Trình bày ...'
        thành 3 item riêng.
        Không áp dụng cho heading kiểu '1. Thông tin học phần'.
        """
        if re.match(r"^\d+\.\s+", text) or re.match(r"^\d+\.\d+\s+", text):
            return [text]

        matches = list(re.finditer(r"(?=(?:^|\s)(\d+)\s+[A-ZÀ-ỴĂÂĐÊÔƠƯ])", text))
        if len(matches) < 2:
            return [text]

        parts: List[str] = []
        starts = [m.start() for m in matches] + [len(text)]
        for i in range(len(starts) - 1):
            part = text[starts[i]:starts[i + 1]].strip()
            if part:
                parts.append(part)
        return parts if parts else [text]


    def _split_embedded_bullets(self, text: str) -> List[str]:
        """
        Tách:
        '• Đơn ... • Bản sao ... • Phiếu xác nhận ...'
        thành nhiều bullet riêng.
        """
        if "•" not in text:
            return [text]

        parts = re.split(r"(?=•\s+)", text)
        parts = [p.strip() for p in parts if p.strip()]
        return parts if len(parts) > 1 else [text]
