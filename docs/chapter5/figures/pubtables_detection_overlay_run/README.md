# Ingest Benchmark: pubtables

- Mode: `table`
- Samples: 1
- Success rate: 1.000
- Latency mean/p50/p95: 19.873s / 19.873s / 19.873s
- Error count: 0
- Backend counts: `{"model_layout_direct": 1}`

## Key Metrics

- spanning_cell_count: mean=0.000, p50=0.000, p95=0.000
- non_empty_pred_cell_rate: mean=0.000, p50=0.000, p95=0.000
- text_source_missing_count: mean=0.000, p50=0.000, p95=0.000
- table_detection_iou50: `{"macro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_precision": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_recall": {"mean": 1.0, "p50": 1.0, "p95": 1.0}}`
- table_detection_iou75: `{"macro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_precision": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_recall": {"mean": 1.0, "p50": 1.0, "p95": 1.0}}`

## Limitations

- This runner evaluates local subsets and does not download large datasets automatically.
- Some adapters run in detection-only mode when cell/text ground truth is unavailable.
- OCR/form metrics are reported only when ground-truth text or form fields are provided.