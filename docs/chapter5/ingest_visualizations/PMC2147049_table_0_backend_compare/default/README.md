# Ingest Benchmark: pubtables_structure

- Mode: `table`
- Samples: 1
- Success rate: 1.000
- Latency mean/p50/p95: 38.830s / 38.830s / 38.830s
- Error count: 0
- Backend counts: `{"ocr": 1}`

## Key Metrics

- char_accuracy: mean=0.443, p50=0.443, p95=0.443
- token_f1: mean=0.841, p50=0.841, p95=0.841
- normalized_text_similarity: mean=0.153, p50=0.153, p95=0.153
- cer: mean=0.557, p50=0.557, p95=0.557
- wer: mean=0.670, p50=0.670, p95=0.670
- reading_order_score: mean=0.575, p50=0.575, p95=0.575
- table_exact_csv: mean=0.000, p50=0.000, p95=0.000
- table_exact_html: mean=0.000, p50=0.000, p95=0.000
- cell_precision_iou50: mean=0.841, p50=0.841, p95=0.841
- cell_recall_iou50: mean=0.716, p50=0.716, p95=0.716
- cell_f1_iou50: mean=0.774, p50=0.774, p95=0.774
- text_assignment_f1: mean=0.972, p50=0.972, p95=0.972
- row_count_error: mean=6.000, p50=6.000, p95=6.000
- col_count_error: mean=2.000, p50=2.000, p95=2.000
- row_count_mae: mean=6.000, p50=6.000, p95=6.000
- col_count_mae: mean=2.000, p50=2.000, p95=2.000
- row_oversegmentation_count: mean=0.000, p50=0.000, p95=0.000
- row_undersegmentation_count: mean=1.000, p50=1.000, p95=1.000
- col_oversegmentation_count: mean=1.000, p50=1.000, p95=1.000
- col_undersegmentation_count: mean=0.000, p50=0.000, p95=0.000
- spanning_cell_count: mean=0.000, p50=0.000, p95=0.000
- non_empty_pred_cell_rate: mean=1.000, p50=1.000, p95=1.000
- text_source_missing_count: mean=0.000, p50=0.000, p95=0.000
- grits_top_like: mean=0.786, p50=0.786, p95=0.786
- grits_loc_like: mean=0.000, p50=0.000, p95=0.000
- grits_con_like: mean=0.000, p50=0.000, p95=0.000
- empty_cell_rate: mean=0.000, p50=0.000, p95=0.000
- matched_cell_count: mean=53.000, p50=53.000, p95=53.000
- unmatched_pred_count: mean=35.000, p50=35.000, p95=35.000
- unmatched_gt_count: mean=21.000, p50=21.000, p95=21.000
- table_detection_iou50: `{"macro_f1": {"mean": 0.5, "p50": 0.5, "p95": 0.5}, "micro_f1": {"mean": 0.5, "p50": 0.5, "p95": 0.5}, "micro_precision": {"mean": 0.3333333333333333, "p50": 0.3333333333333333, "p95": 0.3333333333333333}, "micro_recall": {"mean": 1.0, "p50": 1.0, "p95": 1.0}}`
- table_detection_iou75: `{"macro_f1": {"mean": 0.5, "p50": 0.5, "p95": 0.5}, "micro_f1": {"mean": 0.5, "p50": 0.5, "p95": 0.5}, "micro_precision": {"mean": 0.3333333333333333, "p50": 0.3333333333333333, "p95": 0.3333333333333333}, "micro_recall": {"mean": 1.0, "p50": 1.0, "p95": 1.0}}`
- table_structure: `{"f1": {"mean": 0.0, "p50": 0.0, "p95": 0.0}, "precision": {"mean": 0.0, "p50": 0.0, "p95": 0.0}, "recall": {"mean": 0.0, "p50": 0.0, "p95": 0.0}}`
- table_cell_iou50: `{"f1": {"mean": 0.7737226277372262, "p50": 0.7737226277372262, "p95": 0.7737226277372262}, "precision": {"mean": 0.8412698412698413, "p50": 0.8412698412698413, "p95": 0.8412698412698413}, "recall": {"mean": 0.7162162162162162, "p50": 0.7162162162162162, "p95": 0.7162162162162162}}`
- table_cell_iou75: `{"f1": {"mean": 0.014598540145985401, "p50": 0.014598540145985401, "p95": 0.014598540145985401}, "precision": {"mean": 0.015873015873015872, "p50": 0.015873015873015872, "p95": 0.015873015873015872}, "recall": {"mean": 0.013513513513513514, "p50": 0.013513513513513514, "p95": 0.013513513513513514}}`

## Limitations

- This runner evaluates local subsets and does not download large datasets automatically.
- Some adapters run in detection-only mode when cell/text ground truth is unavailable.
- OCR/form metrics are reported only when ground-truth text or form fields are provided.