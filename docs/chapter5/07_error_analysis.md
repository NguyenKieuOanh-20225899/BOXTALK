# Error Analysis

## 1. Retrieval lech voi cau hoi rong

- Mo ta: cau hoi tong quat hoac can tong hop nhieu dieu khoan thuong lay dung trang nhung sai tieu muc.
- Vi du: nhom comparison/ambiguous trong QCDT co answer_match thap hon factoid.
- Nguyen nhan: scoring theo chunk ngan uu tien cau co keyword gan nhat.
- Tac dong: evidence_match co the dung nhung answer synthesis thieu y.
- Cai thien: section-aware retrieval, parent-child chunk, multi-hop evidence aggregation.

## 2. Dense retrieval yeu voi tieng Viet phap quy

- Mo ta: dense MiniLM kem BM25 tren QCDT.
- Nguyen nhan: cum tu phap quy co keyword dac thu; embedding da ngon ngu khong toi uu domain Viet.
- Tac dong: dense-only khong nen lam mac dinh.
- Cai thien: Vietnamese/domain embedding, hard-negative tuning ve sau, nhung khong lam trong giai doan nay.

## 3. Answer synthesis lay thieu y

- Mo ta: SciFact answer_match 0.220, QASPER answer_match 0.084 du grounded_rate 1.0.
- Nguyen nhan: module answer hien uu tien grounded extractive/rule-safe, khong phai long-form synthesis.
- Tac dong: citation dung nhung cau tra loi chua du/noi dung khong khop gold.
- Cai thien: evidence planning, sentence selection, LLM fallback co kiem chung.

## 4. Cau hoi phu dinh/absence_check

- Mo ta: absence/ambiguous can biet khi nao khong du bang chung.
- Nguyen nhan: retrieval co the luon tra ve chunk gan dung, lam he thong de tra loi qua muc.
- Tac dong: false answer neu bo evidence checker.
- Cai thien: sufficiency threshold, contradiction/absence classifier, calibration tren unanswerable queries.

## 5. Section ambiguity

- Mo ta: dung trang nhung sai dieu/khoan, dac biet muc luc, chuong/dieu lap tu.
- Nguyen nhan: heading extraction va page-level expected label chua du min.
- Tac dong: answer co the dung mot phan nhung citation kem chinh xac.
- Cai thien: hierarchical citation, heading-path scoring, metadata filter theo dieu/khoan.

## 6. Table extraction

- Mo ta: merged cell, multi-row header, exact CSV/HTML thap.
- Vi du: bang QCDT trang 6 can constraint-aware reconstruction moi tach duoc row Tien si va cot thoi gian/tin chi.
- Nguyen nhan: TATR box geometry khong tu gan text; rule grid de merge token sai cot.
- Tac dong: QA bang va citation cell sai neu chunk tu markdown loi.
- Cai thien: graph reconstruction, domain constraints, confidence-based hypothesis scoring, fine-tune TATR structure o buoc sau.

## 7. QASPER scientific QA

- Mo ta: cau hoi dai, answer free-form, can nhieu evidence trong paper.
- Nguyen nhan: retrieval top-k va answer synthesis deu chua du; gold answer khong phai exact span ngan.
- Tac dong: end-to-end success chi 0.050 trong rerun chapter5.
- Cai thien: multi-evidence retrieval, abstractive grounded generation, evaluation bang semantic metrics.

## 8. OCR scan tieng Viet thuc te

- Mo ta: OCR scan synthetic dat cao nhung OCR-D/FUNSD that con loi.
- Nguyen nhan: font, layout, ngon ngu, noise, historical text va bbox alignment.
- Tac dong: chunking/retrieval giam chat luong khi source la scan.
- Cai thien: bo benchmark scan tieng Viet that, OCR confidence propagation, human-labeled page samples.

## Ket luan

Loi chinh hien nam o ba noi: retrieval cho cau hoi rong, answer synthesis tren scientific QA, va reconstruction bang phuc tap. Day la cac gioi han nen ghi ro trong bao cao.
