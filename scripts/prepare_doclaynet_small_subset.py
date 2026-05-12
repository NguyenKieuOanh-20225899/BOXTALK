from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "pierreguillou/DocLayNet-small"
DEFAULT_FILENAME = "data/dataset_small.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DocLayNet-small from Hugging Face as local COCO files")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "doclaynet")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out
    raw_dir = out_root / "raw"
    image_dir = out_root / "images" / args.split
    ann_dir = out_root / "annotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    archive = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
            local_dir=raw_dir,
        )
    )

    if args.force:
        for child in image_dir.glob("*"):
            if child.is_file():
                child.unlink()

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    category_name_to_id: dict[str, int] = {}
    ann_id = 1

    with zipfile.ZipFile(archive) as zf:
        annotation_names = sorted(
            name
            for name in zf.namelist()
            if name.startswith(f"small_dataset/{args.split}/annotations/") and name.endswith(".json")
        )
        if args.limit > 0:
            annotation_names = annotation_names[: args.limit]

        for image_index, annotation_name in enumerate(annotation_names, start=1):
            payload = json.load(zf.open(annotation_name))
            metadata = payload.get("metadata", {}) or {}
            page_hash = str(metadata.get("page_hash") or Path(annotation_name).stem)
            image_name = page_hash + ".png"
            zip_image_name = f"small_dataset/{args.split}/images/{image_name}"
            if zip_image_name not in zf.namelist():
                continue

            image_path = image_dir / image_name
            with zf.open(zip_image_name) as src, image_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            image_id = image_index
            images.append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "width": int(metadata.get("coco_width") or 0),
                    "height": int(metadata.get("coco_height") or 0),
                    "doc_category": metadata.get("doc_category"),
                    "collection": metadata.get("collection"),
                    "doc_name": metadata.get("original_filename"),
                    "page_no": metadata.get("page_no"),
                }
            )

            seen_boxes: set[tuple[Any, ...]] = set()
            for item in payload.get("form", []) or []:
                category_name = str(item.get("category") or "unknown").strip()
                bbox = item.get("box") or [0, 0, 0, 0]
                box_key = (item.get("id_box"), category_name, *(round(float(v), 3) for v in bbox))
                if box_key in seen_boxes:
                    continue
                seen_boxes.add(box_key)

                if category_name not in category_name_to_id:
                    category_name_to_id[category_name] = len(category_name_to_id) + 1
                category_id = category_name_to_id[category_name]
                x, y, w, h = [float(v) for v in bbox]
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, w, h],
                        "area": max(0.0, w) * max(0.0, h),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    categories = [
        {"id": category_id, "name": name}
        for name, category_id in sorted(category_name_to_id.items(), key=lambda item: item[1])
    ]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    ann_path = ann_dir / f"{args.split}.json"
    ann_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset": "doclaynet_small",
        "source": {"type": "huggingface_zip", "repo_id": args.repo_id, "filename": args.filename},
        "root": str(out_root),
        "split": args.split,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "categories": categories,
        "annotation_path": str(ann_path),
        "image_dir": str(image_dir),
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
