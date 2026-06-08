# Ingest Benchmark: pubtables_structure

- Mode: `table`
- Samples: 1
- Success rate: 1.000
- Latency mean/p50/p95: 17.035s / 17.035s / 17.035s
- Error count: 0
- Backend counts: `{"hybrid_tatr": 1}`

## Key Metrics

- char_accuracy: mean=0.910, p50=0.910, p95=0.910
- token_f1: mean=0.820, p50=0.820, p95=0.820
- normalized_text_similarity: mean=0.864, p50=0.864, p95=0.864
- cer: mean=0.090, p50=0.090, p95=0.090
- wer: mean=0.381, p50=0.381, p95=0.381
- reading_order_score: mean=0.887, p50=0.887, p95=0.887
- table_exact_csv: mean=0.000, p50=0.000, p95=0.000
- table_exact_html: mean=0.000, p50=0.000, p95=0.000
- cell_precision_iou50: mean=0.087, p50=0.087, p95=0.087
- cell_recall_iou50: mean=0.108, p50=0.108, p95=0.108
- cell_f1_iou50: mean=0.096, p50=0.096, p95=0.096
- text_assignment_f1: mean=0.923, p50=0.923, p95=0.923
- row_count_error: mean=6.000, p50=6.000, p95=6.000
- col_count_error: mean=0.000, p50=0.000, p95=0.000
- row_count_mae: mean=6.000, p50=6.000, p95=6.000
- col_count_mae: mean=0.000, p50=0.000, p95=0.000
- row_oversegmentation_count: mean=1.000, p50=1.000, p95=1.000
- row_undersegmentation_count: mean=0.000, p50=0.000, p95=0.000
- col_oversegmentation_count: mean=0.000, p50=0.000, p95=0.000
- col_undersegmentation_count: mean=0.000, p50=0.000, p95=0.000
- spanning_cell_count: mean=0.000, p50=0.000, p95=0.000
- non_empty_pred_cell_rate: mean=0.946, p50=0.946, p95=0.946
- text_source_missing_count: mean=0.000, p50=0.000, p95=0.000
- grits_top_like: mean=0.892, p50=0.892, p95=0.892
- grits_loc_like: mean=0.126, p50=0.126, p95=0.126
- grits_con_like: mean=0.122, p50=0.122, p95=0.122
- empty_cell_rate: mean=0.054, p50=0.054, p95=0.054
- matched_cell_count: mean=8.000, p50=8.000, p95=8.000
- unmatched_pred_count: mean=84.000, p50=84.000, p95=84.000
- unmatched_gt_count: mean=66.000, p50=66.000, p95=66.000
- table_detection_iou50: `{"macro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_precision": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_recall": {"mean": 1.0, "p50": 1.0, "p95": 1.0}}`
- table_detection_iou75: `{"macro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_f1": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_precision": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "micro_recall": {"mean": 1.0, "p50": 1.0, "p95": 1.0}}`
- table_structure: `{"f1": {"mean": 0.4096385542168674, "p50": 0.4096385542168674, "p95": 0.4096385542168674}, "precision": {"mean": 0.3695652173913043, "p50": 0.3695652173913043, "p95": 0.3695652173913043}, "recall": {"mean": 0.4594594594594595, "p50": 0.4594594594594595, "p95": 0.4594594594594595}}`
- table_cell_iou50: `{"f1": {"mean": 0.09638554216867469, "p50": 0.09638554216867469, "p95": 0.09638554216867469}, "precision": {"mean": 0.08695652173913043, "p50": 0.08695652173913043, "p95": 0.08695652173913043}, "recall": {"mean": 0.10810810810810811, "p50": 0.10810810810810811, "p95": 0.10810810810810811}}`
- table_cell_iou75: `{"f1": {"mean": 0.0, "p50": 0.0, "p95": 0.0}, "precision": {"mean": 0.0, "p50": 0.0, "p95": 0.0}, "recall": {"mean": 0.0, "p50": 0.0, "p95": 0.0}}`

## Limitations

- This runner evaluates local subsets and does not download large datasets automatically.
- Some adapters run in detection-only mode when cell/text ground truth is unavailable.
- OCR/form metrics are reported only when ground-truth text or form fields are provided.