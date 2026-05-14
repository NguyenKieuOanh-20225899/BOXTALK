# Demo Checklist

## 1. Chuẩn bị trước khi demo

- Đảm bảo đang ở repo `BOXTALK`.
- Chạy validation nhanh nếu còn thời gian:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_demo_check --mode all
```

- Mở sẵn tài liệu PDF demo.
- Mở sẵn file kết quả benchmark hoặc slide chứa bảng kết quả.

## 2. Cách chạy server

Entrypoint UI hiện tại là FastAPI app trong `app.routed_rag_starter:app`:

```powershell
.\.venv-gpu\Scripts\python.exe -m uvicorn app.routed_rag_starter:app --host 127.0.0.1 --port 8000
```

Nếu dùng virtual environment khác, thay `.\.venv-gpu\Scripts\python.exe` bằng Python của môi trường đó.

## 3. Port nên dùng

- Port mặc định: `8000`.
- Nếu port 8000 busy:

```powershell
.\.venv-gpu\Scripts\python.exe -m uvicorn app.routed_rag_starter:app --host 127.0.0.1 --port 8001
```

- Truy cập: `http://127.0.0.1:8000` hoặc `http://127.0.0.1:8001`.

## 4. PDF demo nên dùng

Ưu tiên chọn PDF đã chạy ổn trong benchmark:

- QCDT real PDF trong `data/real_pdfs` nếu demo câu hỏi tiếng Việt/quy chế.
- `1706.03762v7.pdf` nếu demo paper khoa học Attention.

Không nên chọn PDF mới hoàn toàn ngay trong buổi bảo vệ nếu chưa chạy thử trước.

## 5. Câu hỏi demo an toàn

Chọn câu hỏi có câu trả lời nằm rõ trong tài liệu:

1. “Tài liệu này nói về chủ đề gì?”
2. “Mục tiêu chính của tài liệu là gì?”
3. “Điều kiện hoặc quy định chính được nêu trong phần này là gì?”
4. “Theo tài liệu, thành phần/khái niệm X được mô tả như thế nào?”
5. “Tóm tắt ngắn nội dung của phần Y.”

Với paper Attention:

- “What is the main idea of the Transformer model?”
- “What role does self-attention play in the model?”
- “What is positional encoding used for?”

## 6. Câu hỏi bảng

Chọn câu hỏi bảng đã biết chắc có trong QCDT:

- “Chương trình có thời gian đào tạo bao lâu và bao nhiêu tín chỉ?”
- “Mức/giá trị tương ứng trong bảng là gì?”

Khi demo, chỉ ra citation hoặc đoạn bảng được hệ thống dùng.

## 7. Câu hỏi không có trong tài liệu

Dùng một câu hỏi ngoài phạm vi để minh họa kiểm soát evidence:

- “Tài liệu này có nói về học phí năm 2030 không?”
- “Tác giả có khuyến nghị dùng mô hình GPT-5 không?”

Kỳ vọng: hệ thống nên từ chối hoặc trả lời rằng không có đủ thông tin trong tài liệu. Nếu hệ thống vẫn trả lời, cần giải thích đây là limitation liên quan abstention.

## 8. Screenshot cần chụp

- Màn hình upload/chọn PDF.
- Màn hình indexing hoặc ingest thành công.
- Câu trả lời văn bản có citation.
- Câu trả lời bảng có citation.
- Một ví dụ hệ thống không đủ evidence.
- Bảng kết quả benchmark trong docs/slide.

## 9. Lưu ý khi demo

- Không bật LLM thật nếu đang trình bày pipeline chính.
- Không claim `hybrid_tatr` là backend production chính.
- Không demo trên PDF scan xấu nếu chưa kiểm tra trước.
- Chuẩn bị sẵn screenshot phòng trường hợp server hoặc GPU có vấn đề.
- Nếu cần nói về QASPER, nhấn mạnh đó là benchmark khó để phân tích hạn chế, không phải kết quả thất bại của pipeline chính.
