# Ingest Benchmark: pubtables_structure

- Mode: `table`
- Samples: 1
- Success rate: 1.000
- Latency mean/p50/p95: 17.362s / 17.362s / 17.362s
- Error count: 0
- Backend counts: `{"tatr": 1}`

## Key Metrics

- char_accuracy: mean=0.000, p50=0.000, p95=0.000
- token_f1: mean=0.000, p50=0.000, p95=0.000
- normalized_text_similarity: mean=0.000, p50=0.000, p95=0.000
- cer: mean=1.000, p50=1.000, p95=1.000
- wer: mean=1.000, p50=1.000, p95=1.000
- reading_order_score: mean=0.000, p50=0.000, p95=0.000
- table_exact_csv: mean=0.000, p50=0.000, p95=0.000
- table_exact_html: mean=0.000, p50=0.000, p95=0.000
- cell_precision_iou50: mean=0.315, p50=0.315, p95=0.315
- cell_recall_iou50: mean=0.392, p50=0.392, p95=0.392
- cell_f1_iou50: mean=0.349, p50=0.349, p95=0.349
- text_assignment_f1: mean=0.000, p50=0.000, p95=0.000
- row_count_error: mean=6.000, p50=6.000, p95=6.000
- col_count_error: mean=0.000, p50=0.000, p95=0.000
- row_count_mae: mean=6.000, p50=6.000, p95=6.000
- col_count_mae: mean=0.000, p50=0.000, p95=0.000
- row_oversegmentation_count: mean=1.000, p50=1.000, p95=1.000
- row_undersegmentation_count: mean=0.000, p50=0.000, p95=0.000
- col_oversegmentation_count: mean=0.000, p50=0.000, p95=0.000
- col_undersegmentation_count: mean=0.000, p50=0.000, p95=0.000
- spanning_cell_count: mean=0.000, p50=0.000, p95=0.000
- non_empty_pred_cell_rate: mean=0.000, p50=0.000, p95=0.000
- text_source_missing_count: mean=0.000, p50=0.000, p95=0.000
- grits_top_like: mean=0.892, p50=0.892, p95=0.892
- grits_loc_like: mean=0.217, p50=0.217, p95=0.217
- grits_con_like: mean=0.000, p50=0.000, p95=0.000
- empty_cell_rate: mean=1.000, p50=1.000, p95=1.000
- matched_cell_count: mean=29.000, p50=29.000, p95=29.000
- unmatched_pred_count: mean=63.000, p50=63.000, p95=63.000
- unmatched_gt_count: mean=45.000, p50=45.000, p95=45.000
- table_detection_iou50: `{"macro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_precision": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_recall": {"mean": 1.0, "p50": 1.0, "p95": 1.0}}`
- table_detection_iou75: `{"macro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_precision": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_recall": {"mean": 1.0, "p50": 1.0, "p95": 1.0}}`
- table_structure: `{"f1": {"mean": 0.0, "p50": 0.0, "p95": 0.0}, "precision": {"mean": 0.0, "p50": 0.0, "p95": 0.0}, "recall": {"mean": 0.0, "p50": 0.0, "p95": 0.0}}`
- table_cell_iou50: `{"f1": {"mean": 0.34939759036144574, "p50": 0.34939759036144574, "p95": 0.34939759036144574}, "precision": {"mean": 0.31521739130434784, "p50": 0.31521739130434784, "p95": 0.31521739130434784}, "recall": {"mean": 0.3918918918918919, "p50": 0.3918918918918919, "p95": 0.3918918918918919}}`
- table_cell_iou75: `{"f1": {"mean": 0.0, "p50": 0.0, "p95": 0.0}, "precision": {"mean": 0.0, "p50": 0.0, "p95": 0.0}, "recall": {"mean": 0.0, "p50": 0.0, "p95": 0.0}}`

## Limitations

- This runner evaluates local subsets and does not download large datasets automatically.
- Some adapters run in detection-only mode when cell/text ground truth is unavailable.
- OCR/form metrics are reported only when ground-truth text or form fields are provided.