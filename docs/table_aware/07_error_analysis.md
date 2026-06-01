# Phase 7 - Error Analysis

| Nhóm lỗi | Mô tả | Nguyên nhân | Ảnh hưởng QA/citation | Hướng cải thiện |
|---|---|---|---|---|
| merged cell | Một ô trải nhiều hàng/cột | Detector structure chưa ổn định | Sai row/col citation | Cải thiện span resolver, benchmark cell IoU |
| multi-row header | Header gồm nhiều tầng | Header inference đang đơn giản | Sai `col_header` | Ghép header theo cây cột |
| caption/footnote lẫn bảng | Text ngoài bảng bị gán vào cell | Region bbox hoặc word assignment rộng | Retrieval nhiễu | Tách caption/footnote trước assign |
| bảng không đường kẻ | Grid khó phát hiện | PDF thiếu vector lines | Thiếu cells | Dựa vào TATR + word alignment |
| scan/OCR lỗi | Text cell sai | OCR nhận dạng kém | Answer/citation sai | OCR confidence + review queue |
| exact CSV/HTML sai | Structure export lệch | Span/header phức tạp | Benchmark CSV fail | Dùng cell graph thay vì row text đơn giản |
| row/column split sai | Số hàng/cột sai | Detection threshold | Lookup nhầm | Calibrate threshold theo validation |
| word assignment sai | Word gán nhầm cell | Bbox overlap/center sai | Cell text sai | Kết hợp overlap, reading order, confidence |
| reverse lookup sai | Hỏi từ value ra label | Value lặp nhiều nơi | Trả nhầm hàng | Boost thêm row/col context |
| citation thiếu row/col | Metadata không đủ | Extractor không có header | Fallback table-level | Không sinh header giả, cải thiện metadata upstream |
