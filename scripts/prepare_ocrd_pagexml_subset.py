from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OCR-D PAGE-XML images as BOXTALK OCR JSONL manifest")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more cloned OCR-D/PAGE-XML roots.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "ocrd_pagexml")
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.out / "images"
    manifest_dir = args.out / "ocr"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for raw_dir in args.raw_dir:
        raw_dir = raw_dir.resolve()
        if not raw_dir.exists():
            continue
        for xml_path in sorted(raw_dir.rglob("*.xml")):
            if xml_path.name.lower().startswith("mets"):
                continue
            sample = _pagexml_to_sample(xml_path, raw_dir, image_dir)
            if sample is None:
                continue
            rows.append(sample)
            if args.limit > 0 and len(rows) >= args.limit:
                break
        if args.limit > 0 and len(rows) >= args.limit:
            break

    manifest_path = manifest_dir / "ocr_samples.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "ocrd_pagexml",
        "raw_dirs": [str(path.resolve()) for path in args.raw_dir],
        "num_samples": len(rows),
        "manifest": str(manifest_path),
        "image_dir": str(image_dir),
        "note": "PAGE-XML line text is used as OCR text ground truth; form/layout fields are not evaluated.",
    }
    (args.out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _pagexml_to_sample(xml_path: Path, raw_dir: Path, image_dir: Path) -> dict | None:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    page = _first_by_local_name(root, "Page")
    if page is None:
        return None

    image_name = page.attrib.get("imageFilename")
    image_path = _find_image(xml_path.parent, raw_dir, image_name, xml_path.stem)
    if image_path is None:
        return None

    line_texts = _extract_line_texts(page)
    if not line_texts:
        return None

    source_id = _safe_id(raw_dir.name)
    doc_id = f"ocrd_{source_id}_{xml_path.stem}"
    output_image = image_dir / f"{doc_id}.png"
    _write_png(image_path, output_image)

    return {
        "doc_id": doc_id,
        "image_path": str(output_image.resolve()),
        "ground_truth": {
            "text": "\n".join(line_texts),
            "ordered_text": line_texts,
        },
        "metadata": {
            "source": "OCR-D PAGE-XML",
            "raw_dir": str(raw_dir),
            "pagexml_path": str(xml_path),
            "source_image": str(image_path),
            "note": "Historical OCR-D samples may contain Fraktur/old glyphs; OCR scores depend strongly on language/model support.",
        },
    }


def _first_by_local_name(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if _local_name(element.tag) == name:
            return element
    return None


def _extract_line_texts(page: ET.Element) -> list[str]:
    texts: list[str] = []
    for element in page.iter():
        if _local_name(element.tag) != "TextLine":
            continue
        text = _text_equiv_unicode(element)
        if text:
            texts.append(text)

    if texts:
        return texts

    for element in page.iter():
        if _local_name(element.tag).endswith("Region"):
            text = _text_equiv_unicode(element)
            if text:
                texts.append(text)
    return texts


def _text_equiv_unicode(element: ET.Element) -> str:
    parts: list[str] = []
    for child in element.iter():
        if _local_name(child.tag) == "Unicode" and child.text:
            normalized = _normalize_space(child.text)
            if normalized:
                parts.append(normalized)
    return " ".join(parts).strip()


def _find_image(xml_dir: Path, raw_dir: Path, image_name: str | None, stem: str) -> Path | None:
    candidates: list[Path] = []
    if image_name:
        candidates.extend([xml_dir / image_name, raw_dir / image_name])
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidates.append(xml_dir / f"{stem}{suffix}")
        candidates.append(xml_dir / f"{stem}_B{suffix}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    names = []
    if image_name:
        names.append(Path(image_name).name)
    names.extend([f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
    names.extend([f"{stem}_B{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
    for name in names:
        matches = list(raw_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _write_png(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(target)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "source"


if __name__ == "__main__":
    main()
