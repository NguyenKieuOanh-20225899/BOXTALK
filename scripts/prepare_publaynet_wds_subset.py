from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "lhoestq/small-publaynet-wds"
DEFAULT_FILENAME = "publaynet-train-000000.tar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small real PubLayNet WebDataset subset as local COCO files")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "publaynet")
    parser.add_argument("--split", default="test")
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
    seen = 0

    with tarfile.open(archive, "r") as tf:
        json_members = [member for member in tf.getmembers() if member.isfile() and member.name.endswith(".json")]
        for json_member in json_members:
            if args.limit > 0 and seen >= args.limit:
                break
            stem = Path(json_member.name).stem
            image_member = _find_image_member(tf, stem)
            if image_member is None:
                continue

            json_file = tf.extractfile(json_member)
            image_file = tf.extractfile(image_member)
            if json_file is None or image_file is None:
                continue

            payload = json.load(json_file)
            image_name = Path(image_member.name).name
            image_path = image_dir / image_name
            with image_file, image_path.open("wb") as f:
                shutil.copyfileobj(image_file, f)

            image_id = int(payload.get("id") or seen + 1)
            images.append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "width": int(payload.get("width") or 0),
                    "height": int(payload.get("height") or 0),
                }
            )

            for ann in payload.get("annotations", []) or []:
                category_name = str(ann.get("category_name") or ann.get("category_id") or "unknown").strip().lower()
                if category_name not in category_name_to_id:
                    category_name_to_id[category_name] = len(category_name_to_id) + 1
                category_id = category_name_to_id[category_name]
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(v) for v in ann.get("bbox", [0, 0, 0, 0])],
                        "area": float(ann.get("area") or 0.0),
                        "iscrowd": int(ann.get("iscrowd") or 0),
                    }
                )
                ann_id += 1
            seen += 1

    categories = [
        {"id": category_id, "name": name}
        for name, category_id in sorted(category_name_to_id.items(), key=lambda item: item[1])
    ]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    ann_path = ann_dir / f"{args.split}.json"
    ann_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset": "publaynet_subset",
        "source": {"type": "huggingface_webdataset", "repo_id": args.repo_id, "filename": args.filename},
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


def _find_image_member(tf: tarfile.TarFile, stem: str) -> tarfile.TarInfo | None:
    for suffix in (".png", ".jpg", ".jpeg"):
        try:
            member = tf.getmember(stem + suffix)
        except KeyError:
            continue
        if member.isfile():
            return member
    return None


if __name__ == "__main__":
    main()
