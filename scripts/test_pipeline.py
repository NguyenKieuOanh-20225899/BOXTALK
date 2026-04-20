from app.ingest.pipeline import ingest_pdf

pdf_path = "data/sample_layout.pdf"
out = ingest_pdf(pdf_path)

print("MODE:", out["probe"]["probe_detected_mode"])
print("PAGES:", len(out["pages"]))
print("BLOCKS:", len(out["blocks"]))
print("CHUNKS:", len(out["chunks"]))

if out["blocks"]:
    b = out["blocks"][0]
    print("\nFIRST BLOCK")
    print("id:", b.block_id)
    print("type:", b.block_type)
    print("page:", b.page_index)
    print("heading_path:", b.heading_path)
    print("text:", b.text[:200])

if out["chunks"]:
    c = out["chunks"][0]
    print("\nFIRST CHUNK")
    print("id:", c.chunk_id)
    print("pages:", c.page_indices)
    print("heading_path:", c.heading_path)
    print("block_types:", c.block_types)
    print("text:", c.text[:300])
