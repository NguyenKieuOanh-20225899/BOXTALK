# Tóm tắt đồ án để trình bày với giảng viên hướng dẫn

## 1. Đề tài

**Nghiên cứu các kĩ thuật truy xuất và hỏi đáp thông tin trên tài liệu PDF**

Mục tiêu của đồ án là nghiên cứu và xây dựng một pipeline hỏi đáp trên tài liệu PDF theo hướng **grounded QA có dẫn chứng**. Hệ thống không chỉ trả lời câu hỏi, mà còn truy xuất các đoạn bằng chứng liên quan và gắn citation để giảm hallucination.

## 2. Phạm vi tài liệu phù hợp

Hệ thống hiện phù hợp nhất với các PDF dạng text có cấu trúc bán hình thức:

- Quy chế, quy định, thông tư.
- Hướng dẫn nghiệp vụ.
- Tài liệu vận hành, sổ tay quy trình.
- Tài liệu policy nội bộ.
- Tài liệu đào tạo dạng text PDF.

Hệ thống có hỗ trợ bảng, layout và OCR, nhưng các trường hợp scan mờ, bảng quá phức tạp, nhiều merged cell hoặc tài liệu khoa học dài vẫn được xem là phạm vi mở rộng.

## 3. Pipeline ingest chi tiết

Phần ingest là bước biến PDF đầu vào thành các khối nội dung có cấu trúc để các bước sau có thể chunk, index, retrieval và trả lời có dẫn chứng. Đây là tầng quan trọng nhất vì nếu ingest sai thứ tự đọc, mất bảng hoặc xóa nhầm tiêu đề thì retrieval và QA phía sau sẽ dễ trả lời sai.

### 3.1 Sơ đồ ingest

```text
PDF input
-> probe PDF
-> build extractor plan
   -> ưu tiên region-level routing nếu đang bật
      -> text region    -> text extractor
      -> table region   -> table extractor / conditional hybrid_tatr
      -> image region   -> OCR nếu cần
      -> caption/header/footer -> gắn nhãn hoặc lọc tùy trường hợp
   -> nếu region routing yếu/fail thì fallback theo probe:
      -> text PDF        -> text extraction
      -> scanned PDF     -> OCR extraction
      -> layout/mixed PDF -> layout/text/OCR theo thứ tự phù hợp
-> gom về BlockNode
-> clean / normalize
-> enrich structure: Chương / Mục / Điều / khoản / danh sách
-> structure-aware chunking
-> chunks có metadata, page, section, citation context
```

Lưu ý đúng theo code hiện tại: region-level routing được ưu tiên trước trong extractor plan khi biến môi trường `BOXBIIBOO_ENABLE_REGION_ROUTING` đang bật. Giá trị mặc định là bật. Vì vậy với PDF text thông thường, hệ thống vẫn thử route theo vùng trước; nếu kết quả yếu hoặc lỗi thì mới fallback sang text extraction, layout backend hoặc OCR tùy mode probe.

### 3.2 Probe PDF

Trước khi trích xuất, hệ thống chạy bước probe để ước lượng loại PDF. Probe không trực tiếp trích xuất nội dung cuối cùng, mà dùng để sắp xếp thứ tự fallback giữa các backend. Nói cách khác, probe vẫn có ích, nhưng khi region routing đang bật thì region được thử trước; probe quyết định hệ thống sẽ fallback sang text, layout hay OCR theo thứ tự nào nếu region không đủ tốt.

Các tín hiệu chính:

- `pages_with_text`: số trang có text layer.
- `pages_without_text`: số trang không có hoặc rất yếu text layer.
- `text_layer_ratio`: tỷ lệ trang có thể trích text trực tiếp.
- `empty_text_ratio`: tỷ lệ trang trích ra text rỗng hoặc quá ít.
- `likely_scanned_ratio`: tỷ lệ trang có dấu hiệu là ảnh scan.
- `image_heavy_ratio`: tỷ lệ trang nặng ảnh.
- `avg_text_quality`: chất lượng text trích xuất được.
- `probe_detected_mode`: mode gợi ý, ví dụ `text`, `layout`, `ocr`, `mixed`.

Ý nghĩa:

- Nếu PDF có text layer tốt, hệ thống ưu tiên text extraction vì nhanh và ít lỗi OCR.
- Nếu PDF gần như scan, hệ thống ưu tiên OCR.
- Nếu PDF vừa có text vừa có nhiều ảnh/bảng/layout phức tạp, hệ thống dùng layout hoặc region-level routing.
- Nếu tài liệu có nhiều block, cột, bảng hoặc caption, hệ thống tránh coi toàn bộ trang là text tuyến tính đơn giản.

### 3.3 Region-level routing

Region-level routing là cải tiến để không xử lý cả trang PDF bằng một backend duy nhất. Thay vào đó, mỗi trang được chia thành các vùng nội dung:

- Vùng văn bản: đưa vào text extractor.
- Vùng bảng: đưa vào table extractor, có thể bật `hybrid_tatr` khi cần cải thiện cấu trúc bảng.
- Vùng ảnh hoặc scan: đưa vào OCR nếu cần lấy chữ.
- Header/footer: gắn nhãn hoặc lọc để không làm nhiễu nội dung chính.
- Caption: giữ lại nếu có liên quan đến bảng/hình.

Điểm quan trọng là dù cả tài liệu được probe là text PDF, nếu region routing đang bật thì hệ thống vẫn có thể phát hiện một vùng bảng riêng trong một trang và route vùng đó sang table extractor. Vì vậy trường hợp một PDF 18 trang text tốt nhưng có 1 trang chứa bảng sẽ không nhất thiết bị xử lý hoàn toàn như text thường; bảng vẫn có cơ hội được xử lý bằng nhánh table.

### 3.4 Text extraction

Với vùng text hoặc PDF có text layer tốt, hệ thống trích xuất trực tiếp text từ PDF. Cách này phù hợp với quy chế, quy định, thông tư, hướng dẫn nghiệp vụ và tài liệu policy nội bộ.

Sau khi trích xuất, text được chuẩn hóa:

- Bỏ nhiễu header/footer lặp lại.
- Giữ lại các nhãn cấu trúc quan trọng như `Điều`, `Khoản`, `1. Thành phần:`.
- Sửa thứ tự đọc để tránh nhầm văn bản một cột thành nhiều cột.
- Gắn metadata trang, section, loại block để phục vụ citation.

### 3.5 Table extraction và hybrid_tatr

Khi gặp bảng, hệ thống không chỉ lấy text thô mà cố gắng giữ cấu trúc hàng/cột/cell.

Có hai mức xử lý:

- **Table extractor mặc định**: dùng text/OCR boxes và thuật toán gom hàng/cột để tạo `table_cells`.
- **Hybrid TATR có điều kiện**: dùng Table Transformer để lấy hình học bảng, hàng, cột; sau đó dùng word boxes từ PDF/OCR để gán text vào từng cell.

Nói ngắn gọn:

```text
TATR = nhận diện hình học bảng
word boxes = chữ + bbox của từng từ
hybrid_tatr = TATR geometry + word boxes + text assignment
```

Hybrid TATR không thay thế toàn bộ ingest. Nó chỉ chạy khi hệ thống phát hiện vùng bảng và backend bảng được bật. Nếu TATR không khả dụng hoặc không có word boxes phù hợp, hệ thống fallback về table extractor mặc định để tránh làm vỡ pipeline.

### 3.6 OCR extraction

Với PDF scan hoặc vùng ảnh cần lấy chữ, hệ thống render trang/vùng thành ảnh rồi OCR. OCR phù hợp với scan rõ, nhưng chất lượng phụ thuộc mạnh vào:

- độ phân giải ảnh;
- độ nghiêng;
- font chữ;
- nhiễu scan;
- ngôn ngữ;
- chất lượng preprocessing.

Vì vậy OCR được xem là nhánh cần thiết để mở rộng phạm vi PDF, nhưng với nhóm tài liệu mục tiêu hiện tại, text-layer PDF vẫn là phạm vi ổn định nhất.

### 3.7 BlockNode và chunking

Sau khi từng backend xử lý xong, hệ thống gom kết quả về dạng `BlockNode`. Đây là biểu diễn thống nhất cho các loại nội dung:

- paragraph;
- heading;
- list item;
- table;
- figure/caption;
- metadata/header/footer.

Từ các `BlockNode`, hệ thống chạy clean, normalize, nhận diện cấu trúc và chunking. Chunk cuối cùng không chỉ có text, mà còn có metadata:

- `doc_id`;
- `page`;
- `section`;
- `block_type`;
- `heading context`;
- thông tin phục vụ citation.

Nhờ vậy, khi QA trả lời, hệ thống có thể dẫn lại trang và mục liên quan thay vì chỉ trả lời một đoạn text không rõ nguồn.

## 4. Cải tiến mới nhất

Khi thử PDF quy chế thi, câu hỏi:

```text
Ban coi thi gồm những thành phần nào?
```

ban đầu trả sai vì retrieval kéo nhầm đoạn và chunk Điều 13 bị tách thiếu nội dung.

Đã cải tiến:

- Sửa **reading order** để không nhận nhầm trang văn bản một cột thành nhiều cột.
- Sửa **cleaner** để không xóa nhầm các nhãn cấu trúc như `1. Thành phần:`.
- Sửa **heading hierarchy** theo cấu trúc pháp quy: `Chương > Mục > Điều > khoản`.
- Sửa **chunking** để giữ heading context và danh sách `a), b), c)` tốt hơn.

Sau re-index, hệ thống trả đúng:

```text
Ban Coi thi gồm:
a) Trưởng ban do lãnh đạo Hội đồng thi kiêm nhiệm;
b) Phó Trưởng ban là lãnh đạo sở GDĐT và/hoặc lãnh đạo cấp phòng thuộc sở GDĐT
   và/hoặc lãnh đạo trường phổ thông và/hoặc lãnh đạo trường THCS;
c) Ủy viên, thư ký là lãnh đạo, chuyên viên các phòng của sở GDĐT và/hoặc
   lãnh đạo, giáo viên trường phổ thông và/hoặc lãnh đạo, giáo viên trường THCS.
```

Citation đúng: **Điều 13. Ban Coi thi, mục 1. Thành phần, trang 12**.

## 5. Kết quả kiểm thử gần nhất

| Nhóm kiểm thử | Kết quả |
|---|---:|
| `python -m compileall app scripts` | Passed |
| `pytest -q` | 64 passed |
| Mock ingest benchmark | success rate 1.000 |
| Bast-Korzen proxy token F1 | 0.998 |
| PubTables structure OCR words F1 | 0.638 |
| PubTables exact CSV | 0.040 |
| DocLayNet layout F1@0.50 | 0.879 trên subset 25 |
| PubLayNet layout F1@0.50 | 0.778 trên subset 25 |
| QA thực tế câu “Ban coi thi...” | Trả đúng, grounded, có citation |

Các benchmark không cho thấy regression sau khi cải tiến chunking/reading order.

## 6. Ý nghĩa của các benchmark

- **DocLayNet / PubLayNet**: đánh giá layout detection.
- **PubTables**: đánh giá phát hiện và tái tạo cấu trúc bảng.
- **Bast-Korzen proxy**: đánh giá text extraction và reading order.
- **OCR-D / FUNSD**: đánh giá OCR scan.
- **SciFact**: đánh giá evidence/citation trên claim khoa học.
- **QASPER**: đánh giá natural scientific QA khó hơn, dùng để phân tích hạn chế.
- **QCDT / Operations / PDF quy chế thật**: đánh giá sát miền ứng dụng của đồ án.

Các benchmark công khai dùng để đánh giá từng tầng kỹ thuật. Các bộ QCDT/Operations/PDF quy chế thật dùng để chứng minh hệ thống hoạt động trong phạm vi tài liệu mục tiêu.

## 7. Hạn chế cần trình bày rõ

- Chưa nên khẳng định xử lý tốt mọi loại PDF.
- Tài liệu scan mờ hoặc OCR kém vẫn ảnh hưởng retrieval và QA.
- Bảng phức tạp, nhiều merged cell hoặc footnote vẫn khó.
- QASPER còn thấp vì câu hỏi khoa học tự nhiên dài, free-form và cần synthesis tốt hơn.
- Một số benchmark đang chạy subset nhỏ, chưa phải đánh giá quy mô rất lớn.
- Hybrid TATR là cải tiến có điều kiện cho bảng, chưa nên xem là backend duy nhất.

## 8. Kết luận trình bày

Đồ án đã xây dựng được một pipeline PDF QA có dẫn chứng, gồm các tầng ingest, chunk/index, retrieval, grounded QA và citation. Hệ thống phù hợp nhất với tài liệu PDF text có cấu trúc như quy chế, quy định, hướng dẫn nghiệp vụ và policy nội bộ. Các cải tiến mới giúp hệ thống xử lý tốt hơn cấu trúc điều/khoản/danh sách trong văn bản pháp quy, đồng thời không làm giảm kết quả benchmark đại diện.

Thông điệp nên trình bày với thầy:

```text
Em không đặt mục tiêu đạt SOTA cho mọi PDF, mà tập trung nghiên cứu và xây dựng
một pipeline hỏi đáp có dẫn chứng cho nhóm PDF text có cấu trúc. Hệ thống đã có
benchmark nhiều tầng để đánh giá ingest, layout, OCR, table, retrieval và QA.
Các lỗi thực tế như sai reading order/chunking đã được phát hiện qua demo và
cải tiến theo hướng tổng quát, không hardcode theo một tài liệu cụ thể.
```
