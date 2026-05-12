# Ingest Real Benchmark Run - 2026-05-12

Run này dùng dữ liệu thật/local subset để kiểm tra framework benchmark ingest PDF.

## Dataset đã chuẩn bị

| Dataset | Nguồn | Local path | Ghi chú |
|---|---|---|---|
| PubTables-1M Detection | `bsmock/pubtables-1m` | `data/benchmarks/pubtables_detection` | Official test split đã có sẵn: 57,125 ảnh và 57,125 XML annotation |
| DocLayNet-small | `pierreguillou/DocLayNet-small` | `data/benchmarks/doclaynet` | Subset thật từ Hugging Face, chuẩn hóa sang COCO local |
| PubLayNet subset | `lhoestq/small-publaynet-wds` | `data/benchmarks/publaynet` | Subset thật từ WebDataset, chuẩn hóa sang COCO local |
| Text JSONL manifest | local PDF | `data/benchmarks/text_extraction_smoke` | Manifest từ `data/retrieval_smoke/employee_handbook_smoke.pdf` |
| OCR JSONL manifest | local PDF | `data/benchmarks/ocr_smoke` | Cùng local PDF, dùng để kiểm tra adapter/metric OCR text; chưa phải scan OCR dataset |

## Commands

```powershell
python scripts/prepare_publaynet_wds_subset.py --limit 25
python scripts/prepare_doclaynet_small_subset.py --limit 25
python scripts/create_local_text_ocr_manifests.py --pdf-dir data\retrieval_smoke --text-out data\benchmarks\text_extraction_smoke --ocr-out data\benchmarks\ocr_smoke --limit 1

$env:BOXBIIBOO_LAYOUT_MODEL_NAME='default'; $env:BOXBIIBOO_LAYOUT_DEVICE='cpu'
python scripts/benchmark_ingest_suite.py --dataset pubtables --data-dir data\benchmarks\pubtables_detection --limit 5 --out results\ingest\pubtables_real_model_5 --mode table --save-predictions
python scripts/benchmark_ingest_suite.py --dataset doclaynet --data-dir data\benchmarks\doclaynet --limit 5 --out results\ingest\doclaynet_real_model_5 --mode layout --save-predictions
python scripts/benchmark_ingest_suite.py --dataset publaynet --data-dir data\benchmarks\publaynet --limit 5 --out results\ingest\publaynet_real_model_5 --mode layout --save-predictions

$env:BOXBIIBOO_ENABLE_REGION_ROUTING='0'; $env:BOXBIIBOO_LAYOUT_MODEL_NAME='0'
python scripts/benchmark_ingest_suite.py --dataset bastkorzen --data-dir data\benchmarks\text_extraction_smoke --limit 1 --out results\ingest\text_manifest_smoke_real_1 --mode text --save-predictions
python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data\benchmarks\ocr_smoke --limit 1 --out results\ingest\ocr_manifest_smoke_real_1 --mode ocr --save-predictions
```

## Results

| Run | Samples | Success | Backend | Key result | Latency mean |
|---|---:|---:|---|---|---:|
| `pubtables_real_model_5` | 5 | 1.00 | `model_layout` | table detection F1@0.50 = 1.00, F1@0.75 = 0.80 | 50.16s |
| `doclaynet_real_model_5` | 5 | 0.60 | `model_layout` | layout micro F1@0.50 = 0.137, macro F1@0.50 = 0.139 | 92.53s |
| `publaynet_real_model_5` | 5 | 1.00 | `model_layout` | layout micro F1@0.50 = 0.134, macro F1@0.50 = 0.147 | 63.55s |
| `text_manifest_smoke_real_1` | 1 | 1.00 | `text` | char accuracy = 1.00, token F1 = 1.00, reading order = 1.00 | 0.013s |
| `ocr_manifest_smoke_real_1` | 1 | 1.00 | `text` | OCR CER = 0.00, OCR WER = 0.00, OCR token F1 = 1.00 | 0.013s |

## GPU rerun

Environment:

```text
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
PyTorch: 2.6.0+cu124
CUDA available: true
```

Commands:

```powershell
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='default'
$env:BOXBIIBOO_LAYOUT_DEVICE='cuda'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'

python scripts/benchmark_ingest_suite.py --dataset pubtables --data-dir data\benchmarks\pubtables_detection --limit 5 --out results\ingest\pubtables_real_cuda_5 --mode table --device cuda --save-predictions
python scripts/benchmark_ingest_suite.py --dataset doclaynet --data-dir data\benchmarks\doclaynet --limit 5 --out results\ingest\doclaynet_real_cuda_5 --mode layout --device cuda --save-predictions
python scripts/benchmark_ingest_suite.py --dataset publaynet --data-dir data\benchmarks\publaynet --limit 5 --out results\ingest\publaynet_real_cuda_5 --mode layout --device cuda --save-predictions
```

| Dataset | CPU mean | GPU mean | CPU p50 | GPU p50 | Success | Key metric |
|---|---:|---:|---:|---:|---:|---|
| PubTables | 50.16s | 9.52s | 45.05s | 0.80s | 1.00 | table F1@0.50 = 1.00, F1@0.75 = 0.80 |
| PubLayNet | 63.55s | 18.41s | 59.93s | 2.05s | 1.00 | layout micro F1@0.50 = 0.134 |
| DocLayNet | 92.53s | 29.45s | 52.55s | 2.30s | 0.60 | layout micro F1@0.50 = 0.137 |

GPU làm giảm latency rõ rệt, còn metric giữ nguyên vì cùng model và cùng sample.

## GPU 25-sample rerun after benchmark-wrapper improvement

`--limit 5` nghĩa là chỉ lấy 5 sample. Sau đó runner được cải tiến để các benchmark image-layout/table gọi trực tiếp component layout detector (`model_layout_direct`) thay vì ép đi qua toàn bộ `ingest_pdf`. Điều này đúng hơn cho mục tiêu đánh giá DocLayNet/PubLayNet/PubTables và tránh lỗi OCR/Paddle làm fail sample layout.

Commands:

```powershell
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='default'
$env:BOXBIIBOO_LAYOUT_DEVICE='cuda'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'

python scripts/benchmark_ingest_suite.py --dataset pubtables --data-dir data\benchmarks\pubtables_detection --limit 25 --out results\ingest\pubtables_real_cuda_25 --mode table --device cuda --save-predictions
python scripts/benchmark_ingest_suite.py --dataset doclaynet --data-dir data\benchmarks\doclaynet --limit 25 --out results\ingest\doclaynet_real_cuda_25 --mode layout --device cuda --save-predictions
python scripts/benchmark_ingest_suite.py --dataset publaynet --data-dir data\benchmarks\publaynet --limit 25 --out results\ingest\publaynet_real_cuda_25 --mode layout --device cuda --save-predictions

$env:BOXBIIBOO_ENABLE_REGION_ROUTING='0'
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='0'
python scripts\create_text_ocr_manifest_25.py --count 25
python scripts/benchmark_ingest_suite.py --dataset bastkorzen --data-dir data\benchmarks\text_ocr_manifest_25\text_extraction --limit 25 --out results\ingest\text_manifest_cuda_25 --mode text --save-predictions
python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data\benchmarks\text_ocr_manifest_25\ocr --limit 25 --out results\ingest\ocr_manifest_cuda_25 --mode ocr --save-predictions
```

| Dataset | Samples | Success | Backend | Key metric | Latency mean / p50 |
|---|---:|---:|---|---|---:|
| PubTables | 25 | 1.00 | `model_layout_direct` | table F1@0.50 = 0.987, F1@0.75 = 0.887 | 1.004s / 0.411s |
| DocLayNet | 25 | 1.00 | `model_layout_direct` | layout micro F1@0.50 = 0.815, F1@0.75 = 0.772 | 0.939s / 0.347s |
| PubLayNet | 25 | 1.00 | `model_layout_direct` | layout micro F1@0.50 = 0.771, F1@0.75 = 0.743 | 0.977s / 0.395s |
| Text manifest | 25 | 1.00 | `text` | char accuracy = 1.00, token F1 = 1.00 | 0.006s / 0.006s |
| OCR manifest | 25 | 1.00 | `text` | OCR CER = 0.00, OCR WER = 0.00, OCR token F1 = 1.00 | 0.006s / 0.006s |

Các run 25 sample cho layout/table ổn định: PubTables, DocLayNet và PubLayNet đều không có lỗi. Text/OCR manifest cũng ổn định, nhưng OCR manifest ở đây vẫn là bộ PDF có text layer để regression adapter/metric, chưa phải benchmark scan OCR-D/FUNSD/Nougat thật.

## OCR scan fix and real scan run

Lỗi OCR scan ban đầu đến từ `paddlepaddle-gpu` trên Windows: PaddleOCR/PaddleX nạp cuDNN DLL xung đột với PyTorch CUDA trong cùng virtualenv. Cách sửa đã áp dụng:

```powershell
python -m pip uninstall -y paddlepaddle-gpu
python -m pip install paddlepaddle==3.2.2
```

Sau khi sửa:

```text
PyTorch CUDA: true, NVIDIA GeForce RTX 3050 6GB Laptop GPU
Paddle CUDA: false
```

Nghĩa là layout model vẫn chạy GPU bằng PyTorch, còn PaddleOCR chạy CPU để tránh xung đột DLL.

OCR backend cũng được chỉnh:

- default `BOXBIIBOO_OCR_PAGE_SCALE`: `1.5`
- default `BOXBIIBOO_OCR_USE_TEXTLINE_ORIENTATION`: `0`
- default `BOXBIIBOO_OCR_USE_DOC_ORIENTATION`: `0`
- default `BOXBIIBOO_OCR_USE_DOC_UNWARPING`: `0`

True scan benchmark local:

```powershell
python scripts\create_ocr_scan_manifest_25.py --count 25

$env:BOXBIIBOO_ENABLE_REGION_ROUTING='0'
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='0'
$env:BOXBIIBOO_OCR_LANG='en'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'
python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data\benchmarks\ocr_scan_25\ocr --limit 25 --out results\ingest\ocr_scan_real_25_fixed --mode ocr --save-predictions
```

| Dataset | Samples | Success | Backend | Key metric | Latency mean / p50 |
|---|---:|---:|---|---|---:|
| OCR scan synthetic | 25 | 1.00 | `ocr` | OCR CER = 0.00, OCR WER = 0.00, OCR token F1 = 1.00 | 34.95s / 6.93s |

Smoke run sau khi tắt textline orientation mặc định:

| Dataset | Samples | Success | Backend | Key metric | Latency mean / p50 |
|---|---:|---:|---|---|---:|
| OCR scan synthetic | 5 | 1.00 | `ocr` | OCR CER = 0.00, OCR WER = 0.00, OCR token F1 = 1.00 | 8.62s / 6.82s |

## OCR GPU fix: isolated PaddleOCR virtualenv

Root cause: on Windows, `paddlepaddle-gpu` and PyTorch CUDA can load conflicting CUDA/cuDNN DLLs when they are installed in the same `.venv-gpu` and imported in the same process. The stable fix used for benchmark runs is to isolate PaddleOCR GPU in a separate virtualenv and run OCR benchmarks in a separate process.

Main env policy:

- `.venv-gpu`: PyTorch CUDA for layout/model benchmarks.
- `.venv-ocr-gpu`: PaddleOCR GPU only for OCR benchmarks.

Install commands:

```powershell
py -3.12 -m venv .venv-ocr-gpu
.\.venv-ocr-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ocr-gpu\Scripts\python.exe -m pip install "paddlepaddle-gpu==3.2.2" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ --extra-index-url https://pypi.org/simple
.\.venv-ocr-gpu\Scripts\python.exe -m pip install pymupdf pillow huggingface-hub pyarrow pandas "paddleocr>=3.4.1"
```

Equivalent shortcut after this update:

```powershell
.\.venv-ocr-gpu\Scripts\python.exe -m pip install -r requirements_ocr_gpu.txt
```

Verification:

```text
Paddle compiled with CUDA: true
Paddle device: gpu:0
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
PaddleOCR(lang="en", device="gpu:0") initializes successfully
```

OCR benchmark environment:

```powershell
$env:BOXBIIBOO_ENABLE_REGION_ROUTING='0'
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='0'
$env:BOXBIIBOO_OCR_LANG='en'
$env:BOXBIIBOO_OCR_DEVICE='gpu:0'
$env:BOXBIIBOO_OCR_PAGE_SCALE='1.5'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'
```

## External OCR / Nougat runs on GPU

Prepared datasets:

| Dataset | Local path | Prep script | Notes |
|---|---|---|---|
| Synthetic scan OCR | `data/benchmarks/ocr_scan_25/ocr` | `scripts/create_ocr_scan_manifest_25.py` | Image-only PDF sanity check |
| FUNSD OCR | `data/benchmarks/funsd/ocr` | `scripts/prepare_funsd_ocr_subset.py` | Real external form images; OCR text only, not form-field extraction |
| OCR-D PAGE-XML | `data/benchmarks/ocrd_pagexml/ocr` | `scripts/prepare_ocrd_pagexml_subset.py` | Real OCR-D/PAGE-XML samples from small public repos; 19 valid pages available |
| Nougat/arXiv proxy | `data/benchmarks/nougat_arxiv_small/text` | `scripts/prepare_nougat_arxiv_subset.py` | Real arXiv PDFs with markdown-like Nougat content; text extraction proxy, not full Nougat markup decoding |

Commands:

```powershell
.\.venv-ocr-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset ocr --data-dir data\benchmarks\ocr_scan_25\ocr --limit 25 --out results\ingest\ocr_scan_gpu_25 --mode ocr --save-predictions

.\.venv-ocr-gpu\Scripts\python.exe scripts\prepare_funsd_ocr_subset.py --limit 25 --split test
.\.venv-ocr-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset ocr --data-dir data\benchmarks\funsd\ocr --limit 25 --out results\ingest\funsd_ocr_gpu_25 --mode ocr --save-predictions

git clone --depth 1 https://github.com/tboenig/gt-guideline-examples.git data\benchmarks\ocrd_gt_guideline_examples\raw
git clone --depth 1 https://github.com/tboenig/16_frak_simple.git data\benchmarks\ocrd_16_frak_simple\raw
.\.venv-ocr-gpu\Scripts\python.exe scripts\prepare_ocrd_pagexml_subset.py --raw-dir data\benchmarks\ocrd_gt_guideline_examples\raw data\benchmarks\ocrd_16_frak_simple\raw --limit 25
$env:BOXBIIBOO_OCR_LANG='german'
.\.venv-ocr-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset ocr --data-dir data\benchmarks\ocrd_pagexml\ocr --limit 19 --out results\ingest\ocrd_pagexml_gpu_19 --mode ocr --save-predictions

.\.venv-ocr-gpu\Scripts\python.exe scripts\prepare_nougat_arxiv_subset.py --limit 25 --timeout 90
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset nougat --data-dir data\benchmarks\nougat_arxiv_small\text --limit 25 --out results\ingest\nougat_arxiv_text_direct_25 --mode text --save-predictions
```

Results:

| Run | Samples | Success | Backend | Key metric | Latency mean / p50 |
|---|---:|---:|---|---|---:|
| `ocr_scan_gpu_25` | 25 | 1.00 | `ocr` | OCR CER = 0.000, WER = 0.000, token F1 = 1.000 | 0.638s / 0.359s |
| `funsd_ocr_gpu_25` | 25 | 1.00 | `ocr` | OCR token F1 = 0.749, CER = 0.728, WER = 0.857 | 1.084s / 0.755s |
| `ocrd_pagexml_gpu_19` | 19 | 1.00 | `ocr` | OCR token F1 = 0.657, CER = 0.847, WER = 0.976 | 4.776s / 4.663s |
| `nougat_arxiv_text_direct_25` | 25 | 1.00 | `text_direct` | token F1 = 0.628, approx CER = 0.612, reading order = 0.570 | 1.078s / 0.489s |

Important interpretation:

- FUNSD ground truth is word annotation, not exact OCR line text. Token F1 and reading-order score are more informative than full-string CER/WER.
- OCR-D samples are historical German/Latin/Fraktur-like pages. PaddleOCR German/Latin models run on GPU, but the dataset is harder than modern English scans, so CER/WER are expectedly high.
- Nougat/arXiv is evaluated as text extraction against markdown-like academic content. This repo does not yet implement full Nougat-style PDF-to-markup generation.
- Long-document edit-distance metrics are capped in `app/eval/ingest_metrics.py` to avoid O(n*m) timeouts. For long academic PDFs, token F1 is the main stable text metric; CER/WER are approximate unless the cap is raised with `BOXBIIBOO_BENCHMARK_MAX_CHAR_EDIT_LENGTH` and `BOXBIIBOO_BENCHMARK_MAX_TOKEN_EDIT_LENGTH`.

## Notes

- PubTables là detection-only vì ground truth hiện có là table bbox Pascal VOC XML, chưa có cell/row/column structure ground truth.
- DocLayNet/PubLayNet dùng subset thật để tránh tải toàn bộ dataset rất lớn. Full DocLayNet core có thể tải bằng `scripts/setup_benchmark_datasets.py --dataset doclaynet`.
- OCR text-layer manifest dùng để kiểm tra adapter/metric regression. OCR scan synthetic dùng image-only PDF để kiểm tra OCR backend thật, nhưng vẫn chưa phải dataset ngoài như OCR-D/FUNSD/Nougat.
- Hai PDF lớn trong `data/real_pdfs` bị timeout khi chạy text/OCR manifest trong giới hạn 10 phút; local smoke PDF nhỏ chạy ổn.
- Bảng CPU phía trên giữ lại để so sánh lịch sử. Kết quả chính nên dùng bảng GPU 25-sample sau cải tiến.
