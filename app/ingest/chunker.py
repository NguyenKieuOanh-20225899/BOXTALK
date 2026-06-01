from app.ingest.schemas import BlockNode, ChunkNode
from app.ingest.table_chunking import build_table_chunks, table_aware_chunking_enabled


def build_chunks(blocks: list[BlockNode], max_chars: int = 1200) -> list[ChunkNode]:
    """Build retrieval chunks from structured ingest blocks.

    The chunker is intentionally conservative for policy/procedure PDFs:
    headings are kept as retrieval context and short enumerated list items are
    allowed to overflow the soft limit a little so an a/b/c clause set is less
    likely to be split at a page boundary.
    """

    chunks: list[ChunkNode] = []

    current_blocks: list[BlockNode] = []
    current_text_parts: list[str] = []
    current_len = 0
    chunk_index = 0

    def flush():
        nonlocal current_blocks, current_text_parts, current_len, chunk_index
        if not current_blocks:
            return

        first = current_blocks[0]
        heading_path = _best_heading_path(current_blocks)
        text = _compose_chunk_text(current_text_parts, heading_path)

        chunks.append(
            ChunkNode(
                chunk_id=f"chunk_{chunk_index:05d}",
                chunk_index=chunk_index,
                text=text,
                markdown=text,
                heading_path=heading_path,
                page_start=first.page_index,
                page_end=current_blocks[-1].page_index,
                page_indices=sorted({b.page_index for b in current_blocks}),
                block_ids=[b.block_id for b in current_blocks],
                block_types=[b.block_type for b in current_blocks],
                source_mode=first.source_mode,
                meta={"chunking_strategy": "structure_aware_v2"},
            )
        )

        chunk_index += 1
        current_blocks = []
        current_text_parts = []
        current_len = 0

    for block in blocks:
        text = (block.text or "").strip()
        if not text:
            continue

        if block.block_type == "heading":
            flush()

        if block.block_type == "table":
            flush()
            if table_aware_chunking_enabled():
                table_chunks = build_table_chunks(block, start_index=chunk_index)
                chunks.extend(table_chunks)
                chunk_index += len(table_chunks)
                continue
            chunks.append(
                ChunkNode(
                    chunk_id=f"chunk_{chunk_index:05d}",
                    chunk_index=chunk_index,
                    text=text,
                    markdown=text,
                    heading_path=block.heading_path,
                    page_start=block.page_index,
                    page_end=block.page_index,
                    page_indices=[block.page_index],
                    block_ids=[block.block_id],
                    block_types=[block.block_type],
                    source_mode=block.source_mode,
                    meta={"is_table_chunk": True},
                )
            )
            chunk_index += 1
            continue

        if (
            current_len + len(text) > max_chars
            and current_blocks
            and not _can_soft_keep_list_item(current_blocks, block, current_len, len(text), max_chars)
        ):
            flush()

        current_blocks.append(block)
        current_text_parts.append(text)
        current_len += len(text)

    flush()
    return chunks


def _best_heading_path(blocks: list[BlockNode]) -> list[str]:
    for block in blocks:
        if block.heading_path:
            return list(block.heading_path)
    return []


def _compose_chunk_text(text_parts: list[str], heading_path: list[str]) -> str:
    body = "\n".join(part for part in text_parts if part).strip()
    if not body or not heading_path:
        return body

    heading_prefix = "\n".join(dict.fromkeys(part.strip() for part in heading_path if part.strip()))
    if not heading_prefix or body.startswith(heading_prefix):
        return body
    prefix_parts = [
        heading
        for heading in heading_prefix.splitlines()
        if heading and heading not in body[: max(240, len(heading) + 20)]
    ]
    if not prefix_parts:
        return body
    prefix = "\n".join(prefix_parts)
    return f"{prefix}\n{body}".strip()


def _can_soft_keep_list_item(
    current_blocks: list[BlockNode],
    next_block: BlockNode,
    current_len: int,
    next_len: int,
    max_chars: int,
) -> bool:
    if next_block.block_type not in {"list_item", "paragraph", "metadata"}:
        return False
    if current_len + next_len > int(max_chars * 1.45):
        return False
    if not _same_heading(current_blocks[-1], next_block):
        return False
    if next_block.block_type == "list_item":
        return True
    return _looks_like_clause_continuation(next_block.text)


def _same_heading(left: BlockNode, right: BlockNode) -> bool:
    return list(left.heading_path or []) == list(right.heading_path or [])


def _looks_like_clause_continuation(text: str) -> bool:
    stripped = (text or "").strip().lower()
    return stripped.startswith(
        (
            "a)",
            "b)",
            "c)",
            "d)",
            "đ)",
            "e)",
            "g)",
            "h)",
            "i)",
            "k)",
            "l)",
            "m)",
            "a.",
            "b.",
            "c.",
            "d.",
            "đ.",
        )
    )
