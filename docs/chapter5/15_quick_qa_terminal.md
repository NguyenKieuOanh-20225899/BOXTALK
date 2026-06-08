# Quick QA Terminal

Script này tạo giao diện terminal để hỏi đáp nhanh trên retrieval index đã build.
Nó dùng pipeline có sẵn trong repo:

`retrieval index -> routed retrieval -> evidence checker -> grounded answer -> citation`

## Chạy một câu hỏi

```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv-gpu\Scripts\python.exe scripts\quick_qa_terminal.py `
  --index-dir results\retrieval_index\qcdt_2025_5445_constraint_table_reconstruction `
  --question "Vắng 3-4 buổi thì điểm quá trình bị trừ bao nhiêu?" `
  --strategy hybrid_rerank `
  --table-aware-retrieval `
  --top-k 8 `
  --show-evidence 8
```

## Mở chế độ hỏi đáp liên tục

```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv-gpu\Scripts\python.exe scripts\quick_qa_terminal.py `
  --index-dir results\retrieval_index\qcdt_2025_5445_constraint_table_reconstruction `
  --strategy hybrid_rerank `
  --table-aware-retrieval `
  --show-evidence 5
```

Sau đó nhập câu hỏi ở prompt `Q>`. Thoát bằng `exit`, `quit` hoặc `Ctrl+C`.

## Ghi chú cấu hình

- `--strategy auto`: để router tự chọn strategy theo loại câu hỏi.
- `--strategy bm25`: chạy nhanh, phù hợp demo lexical retrieval.
- `--strategy hybrid_rerank`: phù hợp demo evidence/citation vì có reranking heuristic.
- `--table-aware-retrieval`: bật boost metadata bảng khi truy vấn dạng tra cứu bảng.
- `--load-dense`: chỉ bật nếu index có dense embeddings và muốn dùng dense thật.
- Mặc định CLI ưu tiên in câu trả lời trực tiếp từ `table_cell` evidence nếu tìm được cell rõ ràng. Dùng `--no-cell-answer` nếu muốn xem nguyên câu trả lời standard của pipeline.

## Lưu ý khi demo trước hội đồng

CLI này là công cụ demo/debug, không phải benchmark chính. Dòng `Answer` có thể dùng direct
cell answer khi retrieval đã lấy được cell rõ ràng. Script vẫn in thêm `Pipeline answer` để phân
biệt với câu trả lời standard của pipeline QA lõi. Vì vậy khi trình bày nên nói rõ:

> Terminal demo dùng cell-level evidence để hiển thị nhanh câu trả lời ở mức ô. Benchmark chính
> vẫn được báo cáo bằng các metric Table Hit@k, Row Match@k, Column Match@k và Cell Match@k.

Nếu muốn chụp màn hình pipeline standard không dùng direct cell answer:

```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv-gpu\Scripts\python.exe scripts\quick_qa_terminal.py `
  --index-dir results\retrieval_index\qcdt_2025_5445_constraint_table_reconstruction `
  --question "Vắng 3-4 buổi thì điểm quá trình bị trừ bao nhiêu?" `
  --strategy hybrid_rerank `
  --table-aware-retrieval `
  --top-k 8 `
  --show-evidence 8 `
  --no-cell-answer
```

## Bật LLM explanation để demo

Flag `--llm-explain` bật lớp `llm_explainer`. Lớp này chỉ giải thích câu trả lời dựa trên
answer và citation đã có, không thay thế câu trả lời lõi của pipeline. Đây là phần phù hợp
để demo cho hội đồng thấy: hệ thống có thể dùng LLM để diễn giải mượt hơn, nhưng vẫn giữ
nguyên cơ chế grounded answer và citation.

Chạy với provider giả lập `dummy`, không cần gọi model ngoài:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\quick_qa_terminal.py `
  --index-dir results\retrieval_index\qcdt_2025_5445_constraint_table_reconstruction `
  --question "B tương ứng bao nhiêu điểm thang 4?" `
  --strategy bm25 `
  --table-aware-retrieval `
  --top-k 8 `
  --show-evidence 3 `
  --llm-explain `
  --llm-provider dummy
```

Nếu muốn dùng Ollama local:

```powershell
$env:BOXTALK_LLM_PROVIDER="ollama"
$env:BOXTALK_LLM_BASE_URL="http://localhost:11434/v1"
$env:BOXTALK_LLM_MODEL="qwen2.5:7b-instruct"

.\.venv-gpu\Scripts\python.exe scripts\quick_qa_terminal.py `
  --index-dir results\retrieval_index\qcdt_2025_5445_constraint_table_reconstruction `
  --question "B tương ứng bao nhiêu điểm thang 4?" `
  --strategy bm25 `
  --table-aware-retrieval `
  --top-k 8 `
  --show-evidence 3 `
  --llm-explain `
  --llm-provider ollama
```

Khi giải thích được tạo thành công, terminal sẽ in thêm mục `LLM explanation`. Nếu không
đủ điều kiện gọi LLM, terminal sẽ in `LLM explanation trace` kèm lý do, ví dụ
`answer_is_not_grounded_or_has_no_citations` hoặc `evidence_decision_is_not_answer`.
