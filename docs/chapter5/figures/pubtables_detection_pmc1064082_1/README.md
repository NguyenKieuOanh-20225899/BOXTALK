# PubTables Detection Overlay: PMC1064082_1

This directory contains figure assets for the PubTables table-region detection subsection.

## Files

| File | Meaning |
|---|---|
| `pubtables_detection_original.png` | Original PubTables page image. |
| `pubtables_detection_groundtruth.png` | Ground-truth table bbox from Pascal VOC XML, shown in green. |
| `pubtables_detection_overlay.png` | Ground-truth bbox in green and model prediction in red. |

## Source

- Image: `data\benchmarks\pubtables_detection\extracted\images\test\PMC1064082_1.jpg`
- Ground truth XML: `data\benchmarks\pubtables_detection\extracted\annotations\test\PMC1064082_1.xml`
- Prediction JSON: `docs\chapter5\figures\pubtables_detection_pmc1064082_1_run\predictions\PMC1064082_1.json`

## Reproduce Commands

Run from repo root.

Create a one-sample PubTables detection dataset for this figure:

```powershell
$sample = "PMC1064082_1"
$tmp = "docs\chapter5\figures\pubtables_detection_pmc1064082_1_dataset"
New-Item -ItemType Directory -Force "$tmp\extracted\images\test" | Out-Null
New-Item -ItemType Directory -Force "$tmp\extracted\annotations\test" | Out-Null
Copy-Item "data\benchmarks\pubtables_detection\extracted\images\test\$sample.jpg" "$tmp\extracted\images\test\$sample.jpg" -Force
Copy-Item "data\benchmarks\pubtables_detection\extracted\annotations\test\$sample.xml" "$tmp\extracted\annotations\test\$sample.xml" -Force
```

Run the table-region detection benchmark and save the prediction bbox:

```powershell
$env:BOXBIIBOO_LAYOUT_MODEL_NAME = "Aryn/deformable-detr-DocLayNet"
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py `
  --dataset pubtables `
  --data-dir docs\chapter5\figures\pubtables_detection_pmc1064082_1_dataset `
  --limit 1 `
  --out docs\chapter5\figures\pubtables_detection_pmc1064082_1_run `
  --mode table `
  --device cuda `
  --save-predictions
```

Render the original, ground-truth bbox, and overlay figures:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\create_pubtables_detection_overlay.py
```

The render script writes this directory and also copies figure aliases to:

```text
docs\chapter5\figures\pubtables_detection_original.png
docs\chapter5\figures\pubtables_detection_groundtruth.png
docs\chapter5\figures\pubtables_detection_overlay.png
```

## Notes

This sample is useful for explaining IoU thresholds: it is counted as a match at IoU=0.50 but not at IoU=0.75, because the predicted bbox includes extra area around the table.
