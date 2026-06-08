from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "benchmarks" / "pubtables_detection" / "extracted" / "images" / "test" / "PMC1064082_1.jpg"
XML_PATH = ROOT / "data" / "benchmarks" / "pubtables_detection" / "extracted" / "annotations" / "test" / "PMC1064082_1.xml"
PRED_PATH = ROOT / "docs" / "chapter5" / "figures" / "pubtables_detection_pmc1064082_1_run" / "predictions" / "PMC1064082_1.json"
OUT_DIR = ROOT / "docs" / "chapter5" / "figures" / "pubtables_detection_pmc1064082_1"
FIG_DIR = ROOT / "docs" / "chapter5" / "figures"


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _gt_boxes() -> list[list[float]]:
    root = ET.parse(XML_PATH).getroot()
    boxes: list[list[float]] = []
    for obj in root.findall("object"):
        if (obj.findtext("name") or "").strip().lower() != "table":
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        boxes.append(
            [
                float(bnd.findtext("xmin") or 0),
                float(bnd.findtext("ymin") or 0),
                float(bnd.findtext("xmax") or 0),
                float(bnd.findtext("ymax") or 0),
            ]
        )
    return boxes


def _pred_boxes() -> list[list[float]]:
    data = json.loads(PRED_PATH.read_text(encoding="utf-8"))
    boxes: list[list[float]] = []
    for region in data.get("table_regions") or []:
        bbox = region.get("bbox") or []
        if len(bbox) >= 4:
            boxes.append([float(value) for value in bbox[:4]])
    return boxes


def _draw_boxes(
    output_path: Path,
    *,
    gt_boxes: list[list[float]] | None = None,
    pred_boxes: list[list[float]] | None = None,
    title: str | None = None,
) -> None:
    img = Image.open(IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)
    label_font = _font(22)
    title_font = _font(28)

    def draw_one(box: list[float], color: tuple[int, int, int], label: str) -> None:
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
        label_pos = (x0 + 4, max(4, y0 - 28))
        text_bbox = draw.textbbox(label_pos, label, font=label_font)
        draw.rectangle(text_bbox, fill=(255, 255, 255))
        draw.text(label_pos, label, fill=color, font=label_font)

    if gt_boxes:
        for box in gt_boxes:
            draw_one(box, (0, 150, 80), "Ground truth")
    if pred_boxes:
        for box in pred_boxes:
            draw_one(box, (220, 40, 40), "Prediction")

    if title:
        pos = (18, 18)
        text_bbox = draw.textbbox(pos, title, font=title_font)
        draw.rectangle(text_bbox, fill=(255, 255, 255))
        draw.text(pos, title, fill=(20, 20, 20), font=title_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = _gt_boxes()
    pred = _pred_boxes()

    original = OUT_DIR / "pubtables_detection_original.png"
    shutil.copyfile(IMAGE_PATH, original)
    _draw_boxes(OUT_DIR / "pubtables_detection_groundtruth.png", gt_boxes=gt)
    _draw_boxes(OUT_DIR / "pubtables_detection_overlay.png", gt_boxes=gt, pred_boxes=pred)

    for name in (
        "pubtables_detection_original.png",
        "pubtables_detection_groundtruth.png",
        "pubtables_detection_overlay.png",
    ):
        shutil.copyfile(OUT_DIR / name, FIG_DIR / name)

    readme = f"""# PubTables Detection Overlay: PMC1064082_1

This directory contains figure assets for the PubTables table-region detection subsection.

## Files

| File | Meaning |
|---|---|
| `pubtables_detection_original.png` | Original PubTables page image. |
| `pubtables_detection_groundtruth.png` | Ground-truth table bbox from Pascal VOC XML, shown in green. |
| `pubtables_detection_overlay.png` | Ground-truth bbox in green and model prediction in red. |

## Source

- Image: `{IMAGE_PATH.relative_to(ROOT)}`
- Ground truth XML: `{XML_PATH.relative_to(ROOT)}`
- Prediction JSON: `{PRED_PATH.relative_to(ROOT)}`

## Notes

This sample is useful for explaining IoU thresholds: it is counted as a match at IoU=0.50 but not at IoU=0.75, because the predicted bbox includes extra area around the table.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")
    print("Copied aliases to docs/chapter5/figures/")


if __name__ == "__main__":
    main()
