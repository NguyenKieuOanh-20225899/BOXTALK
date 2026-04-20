from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
from huggingface_hub import hf_hub_download


BENCHMARKS_ROOT = ROOT / "data" / "benchmarks"

DOCLAYNET_URL = (
    "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/"
    "dax-doclaynet/1.0.0/DocLayNet_core.zip"
)
PUBTABLES_REPO_ID = "bsmock/pubtables-1m"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DOCLAYNET_EXPECTED_JSONS = ("train.json", "val.json", "test.json")

PUBTABLES_SPLIT_FILES = {
    "train": {
        "images": [
            "PubTables-1M-Detection_Images_Train_Part1.tar.gz",
            "PubTables-1M-Detection_Images_Train_Part2.tar.gz",
        ],
        "annotations": [
            "PubTables-1M-Detection_Annotations_Train.tar.gz",
        ],
    },
    "val": {
        "images": ["PubTables-1M-Detection_Images_Val.tar.gz"],
        "annotations": ["PubTables-1M-Detection_Annotations_Val.tar.gz"],
    },
    "test": {
        "images": ["PubTables-1M-Detection_Images_Test.tar.gz"],
        "annotations": ["PubTables-1M-Detection_Annotations_Test.tar.gz"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract scientific benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["doclaynet", "pubtables", "all"],
        default="all",
        help="Dataset(s) to prepare",
    )
    parser.add_argument(
        "--benchmarks-root",
        type=Path,
        default=BENCHMARKS_ROOT,
        help="Root directory for benchmark datasets",
    )
    parser.add_argument(
        "--pubtables-splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["test"],
        help="PubTables detection splits to download/extract",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for PubTables downloads",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload archives even if they already exist",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract even if the target layout already exists",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archives in raw/ (default behavior). Included for explicitness.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not download or extract. Only inspect the standardized local layout and write manifests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without downloading or extracting",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_files(root: Path, *, suffixes: set[str] | None = None) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes)
    )


def inspect_doclaynet_layout(root: Path) -> dict[str, Any]:
    raw_dir = root / "raw"
    extracted_root = root / "extracted" / "DocLayNet_core"
    coco_dir = extracted_root / "COCO"
    png_dir = extracted_root / "PNG"

    archive_present = (raw_dir / "DocLayNet_core.zip").exists()
    available_jsons = [name for name in DOCLAYNET_EXPECTED_JSONS if (coco_dir / name).exists()]
    png_count = _count_files(png_dir, suffixes={".png"})
    missing_items: list[str] = []
    missing_optional_items: list[str] = []
    if not archive_present:
        missing_optional_items.append("raw/DocLayNet_core.zip")
    if not extracted_root.exists():
        missing_items.append("extracted/DocLayNet_core")
    if not coco_dir.exists():
        missing_items.append("extracted/DocLayNet_core/COCO")
    if not png_dir.exists():
        missing_items.append("extracted/DocLayNet_core/PNG")
    for json_name in DOCLAYNET_EXPECTED_JSONS:
        if json_name not in available_jsons:
            missing_items.append(f"extracted/DocLayNet_core/COCO/{json_name}")
    if png_count <= 0:
        missing_items.append("extracted/DocLayNet_core/PNG/*.png")

    return {
        "dataset": "doclaynet",
        "root": str(root),
        "archive": str(raw_dir / "DocLayNet_core.zip"),
        "archive_present": archive_present,
        "extracted_root": str(extracted_root),
        "source": {
            "type": "url",
            "url": DOCLAYNET_URL,
        },
        "expected_layout": {
            "raw": ["DocLayNet_core.zip"],
            "extracted": {
                "DocLayNet_core": {
                    "COCO": list(DOCLAYNET_EXPECTED_JSONS),
                    "PNG": ["*.png"],
                }
            },
        },
        "stats": {
            "coco_json_files": available_jsons,
            "png_pages": png_count,
        },
        "missing_items": missing_items,
        "missing_optional_items": missing_optional_items,
        "ready": not missing_items,
        "layout": "standardized",
    }


def inspect_pubtables_layout(root: Path, *, splits: list[str]) -> dict[str, Any]:
    raw_dir = root / "raw"
    extracted_root = root / "extracted"
    images_root = extracted_root / "images"
    annotations_root = extracted_root / "annotations"

    split_stats: dict[str, Any] = {}
    missing_items: list[str] = []
    missing_optional_items: list[str] = []
    archives: dict[str, dict[str, list[str]]] = {}
    for split in splits:
        config = PUBTABLES_SPLIT_FILES[split]
        image_archives = [str(raw_dir / filename) for filename in config["images"]]
        annotation_archives = [str(raw_dir / filename) for filename in config["annotations"]]
        archives[split] = {
            "images": image_archives,
            "annotations": annotation_archives,
        }

        image_dir = images_root / split
        annotation_dir = annotations_root / split
        image_count = _count_files(image_dir, suffixes=IMAGE_EXTENSIONS)
        annotation_count = _count_files(annotation_dir, suffixes={".xml"})
        split_missing: list[str] = []
        split_missing_optional: list[str] = []
        for archive in image_archives + annotation_archives:
            if not Path(archive).exists():
                split_missing_optional.append(str(Path(archive).relative_to(root)))
        if image_count <= 0:
            split_missing.append(f"extracted/images/{split}/*")
        if annotation_count <= 0:
            split_missing.append(f"extracted/annotations/{split}/*.xml")
        missing_items.extend(split_missing)
        missing_optional_items.extend(split_missing_optional)
        split_stats[split] = {
            "image_dir": str(image_dir),
            "annotation_dir": str(annotation_dir),
            "image_count": image_count,
            "annotation_count": annotation_count,
            "missing_items": split_missing,
            "missing_optional_items": split_missing_optional,
            "ready": not split_missing,
        }

    return {
        "dataset": "pubtables_detection",
        "root": str(root),
        "repo_id": PUBTABLES_REPO_ID,
        "source": {
            "type": "huggingface_dataset",
            "repo_id": PUBTABLES_REPO_ID,
        },
        "splits": splits,
        "archives": archives,
        "images_root": str(images_root),
        "annotations_root": str(annotations_root),
        "archives_present": {
            split: {
                kind: [Path(path).exists() for path in paths]
                for kind, paths in split_archives.items()
            }
            for split, split_archives in archives.items()
        },
        "expected_layout": {
            "raw": ["PubTables-1M-Detection_*.tar.gz"],
            "extracted": {
                "images": {split: ["*.jpg|*.jpeg|*.png"] for split in splits},
                "annotations": {split: ["*.xml"] for split in splits},
            },
        },
        "split_stats": split_stats,
        "missing_items": missing_items,
        "missing_optional_items": missing_optional_items,
        "ready": not missing_items,
        "layout": "standardized",
    }


def download_via_requests(url: str, destination: Path, *, force: bool, dry_run: bool) -> None:
    if destination.exists() and not force:
        return

    if dry_run:
        print(f"[dry-run] download {url} -> {destination}")
        return

    ensure_dir(destination.parent)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_via_hf(
    *,
    repo_id: str,
    filename: str,
    destination_dir: Path,
    token: str | None,
    force: bool,
    dry_run: bool,
) -> Path:
    destination = destination_dir / filename
    if destination.exists() and not force:
        return destination

    if dry_run:
        print(f"[dry-run] hf download {repo_id}:{filename} -> {destination}")
        return destination

    ensure_dir(destination_dir)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        force_download=force,
        local_dir=destination_dir,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != destination and downloaded_path.exists():
        ensure_dir(destination.parent)
        shutil.copy2(downloaded_path, destination)
    return destination


def extract_doclaynet(
    root: Path,
    *,
    force_download: bool,
    force_extract: bool,
    dry_run: bool,
) -> dict[str, Any]:
    raw_dir = ensure_dir(root / "raw")
    extracted_dir = root / "extracted"
    archive_path = raw_dir / "DocLayNet_core.zip"
    extracted_root = extracted_dir / "DocLayNet_core"

    download_via_requests(DOCLAYNET_URL, archive_path, force=force_download, dry_run=dry_run)

    if extracted_root.exists() and not force_extract:
        return inspect_doclaynet_layout(root)

    if dry_run:
        print(f"[dry-run] extract {archive_path} -> {extracted_dir}")
    else:
        ensure_dir(extracted_dir)
        if extracted_root.exists():
            shutil.rmtree(extracted_root)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extracted_dir)

    if dry_run:
        return {
            **inspect_doclaynet_layout(root),
            "ready": False,
        }
    return inspect_doclaynet_layout(root)


def extract_pubtables_detection(
    root: Path,
    *,
    splits: list[str],
    token: str | None,
    force_download: bool,
    force_extract: bool,
    dry_run: bool,
) -> dict[str, Any]:
    raw_dir = ensure_dir(root / "raw")
    extracted_root = root / "extracted"
    images_root = extracted_root / "images"
    annotations_root = extracted_root / "annotations"

    archives: dict[str, dict[str, list[str]]] = {}
    for split in splits:
        config = PUBTABLES_SPLIT_FILES[split]
        split_archives: dict[str, list[str]] = {"images": [], "annotations": []}
        for kind in ("images", "annotations"):
            for filename in config[kind]:
                archive_path = download_via_hf(
                    repo_id=PUBTABLES_REPO_ID,
                    filename=filename,
                    destination_dir=raw_dir,
                    token=token,
                    force=force_download,
                    dry_run=dry_run,
                )
                split_archives[kind].append(str(archive_path))
                target_dir = (images_root if kind == "images" else annotations_root) / split
                _extract_pubtables_archive(
                    archive_path=archive_path,
                    kind=kind,
                    target_dir=target_dir,
                    force_extract=force_extract,
                    dry_run=dry_run,
                )
        archives[split] = split_archives

    summary = inspect_pubtables_layout(root, splits=splits)
    return {
        **summary,
        "archives": archives,
        "ready": False if dry_run else summary["ready"],
    }


def _extract_pubtables_archive(
    *,
    archive_path: Path,
    kind: str,
    target_dir: Path,
    force_extract: bool,
    dry_run: bool,
) -> None:
    if target_dir.exists() and any(target_dir.iterdir()) and not force_extract:
        return

    if dry_run:
        print(f"[dry-run] extract {archive_path} -> {target_dir}")
        return

    if force_extract and target_dir.exists():
        shutil.rmtree(target_dir)
    ensure_dir(target_dir)

    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue

            suffix = Path(member.name).suffix.lower()
            if kind == "images" and suffix not in IMAGE_EXTENSIONS:
                continue
            if kind == "annotations" and suffix != ".xml":
                continue

            filename = Path(member.name).name
            if not filename:
                continue

            destination = target_dir / filename
            if destination.exists() and not force_extract:
                continue

            extracted = tf.extractfile(member)
            if extracted is None:
                continue

            with extracted, destination.open("wb") as f:
                shutil.copyfileobj(extracted, f)


def write_manifest(root: Path, payload: dict[str, Any]) -> None:
    manifest = {
        **payload,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_json(root / "manifest.json", manifest)


def main() -> None:
    args = parse_args()
    benchmarks_root = args.benchmarks_root
    ensure_dir(benchmarks_root)

    selected = (
        ["doclaynet", "pubtables"]
        if args.dataset == "all"
        else [args.dataset]
    )
    token = args.hf_token or os.getenv("HF_TOKEN")

    prepared: list[dict[str, Any]] = []
    if "doclaynet" in selected:
        root = benchmarks_root / "doclaynet"
        ensure_dir(root)
        info = (
            inspect_doclaynet_layout(root)
            if args.validate_only
            else extract_doclaynet(
                root,
                force_download=args.force_download,
                force_extract=args.force_extract,
                dry_run=args.dry_run,
            )
        )
        write_manifest(root, info) if not args.dry_run else None
        prepared.append(info)

    if "pubtables" in selected:
        root = benchmarks_root / "pubtables_detection"
        ensure_dir(root)
        info = (
            inspect_pubtables_layout(root, splits=args.pubtables_splits)
            if args.validate_only
            else extract_pubtables_detection(
                root,
                splits=args.pubtables_splits,
                token=token,
                force_download=args.force_download,
                force_extract=args.force_extract,
                dry_run=args.dry_run,
            )
        )
        write_manifest(root, info) if not args.dry_run else None
        prepared.append(info)

    summary = {
        "benchmarks_root": str(benchmarks_root),
        "dry_run": args.dry_run,
        "validate_only": args.validate_only,
        "standard_layout": {
            "doclaynet": "doclaynet/raw/DocLayNet_core.zip + doclaynet/extracted/DocLayNet_core/{COCO,PNG}",
            "pubtables_detection": "pubtables_detection/raw/*.tar.gz + pubtables_detection/extracted/{images,annotations}/{split}",
        },
        "prepared": prepared,
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    save_json(benchmarks_root / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
