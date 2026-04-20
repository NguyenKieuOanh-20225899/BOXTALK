from app.ingest.schemas import BlockNode, ChunkNode


def build_chunks(blocks: list[BlockNode], max_chars: int = 1200) -> list[ChunkNode]:
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

        chunks.append(
            ChunkNode(
                chunk_id=f"chunk_{chunk_index:05d}",
                chunk_index=chunk_index,
                text="\n".join(current_text_parts).strip(),
                markdown="\n".join(current_text_parts).strip(),
                heading_path=first.heading_path,
                page_start=first.page_index,
                page_end=current_blocks[-1].page_index,
                page_indices=list({b.page_index for b in current_blocks}),
                block_ids=[b.block_id for b in current_blocks],
                block_types=[b.block_type for b in current_blocks],
                source_mode=first.source_mode,
                meta={},
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

        if current_len + len(text) > max_chars and current_blocks:
            flush()

        current_blocks.append(block)
        current_text_parts.append(text)
        current_len += len(text)

    flush()
    return chunks
