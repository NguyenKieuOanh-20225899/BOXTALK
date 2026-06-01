# Phase 8 - Final Validation

## 1. Commands đã chạy
```powershell
python -m compileall app scripts
.\.venv-gpu\Scripts\python.exe -m compileall app scripts
.\.venv-gpu\Scripts\python.exe -m pytest -q
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_after_table_aware_safe --mode all
.\.venv-gpu\Scripts\python.exe scripts\benchmark_table_qa.py --queries data\benchmarks\table_qa_vi\queries.jsonl --out results\table_qa_vi\final_safe
```

## 2. Kết quả test
- `python -m compileall app scripts`: pass.
- `python -m pytest -q` bằng Python hệ thống: không chạy vì interpreter hệ thống thiếu `pytest`.
- `.\.venv-gpu\Scripts\python.exe -m pytest -q`: `76 passed`.

## 3. Kết quả benchmark
- Mock ingest benchmark: success_rate `1.000`, error_count `0`, table_structure F1 `1.000`, output `results/ingest/mock_after_table_aware_safe/summary.json`.
- Table QA mock-safe benchmark: 8 queries, best variant đạt answer/evidence/cell-citation `1.000`, output `results/table_qa_vi/final_safe/summary.json`.

## 4. Regression check
- Default table chunking vẫn sinh một table chunk khi flag tắt.
- Table-aware chunking/retrieval chỉ bật qua env flag.
- `routed_grounded` answer generator vẫn dùng citation cũ cho paragraph; table citation chỉ thêm metadata khi evidence là table.
- QA smoke/QCDT/Operations full rerun không chạy trong phase này vì không có command smoke riêng được xác định trong workspace hiện tại; unit tests của QA path vẫn pass trong full pytest.

## 5. Kết luận bật/tắt default
Không bật `hybrid_tatr`, table-aware chunking, hoặc table-aware retrieval làm default ở phase này.
