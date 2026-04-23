# User PDF Benchmark Suite

This suite is the project-level benchmark for answering questions over
user-uploaded PDFs. It is intentionally local and lightweight: public academic
benchmarks define the evaluation ideas, while the suite tests the actual routed
RAG pipeline on the PDFs you expect users to upload.

## What It Measures

Use this as the main "is my PDF QA system ready?" benchmark:

- retrieval strength across BM25, dense, hybrid, routed, and adaptive route-retry
- grounded answer quality with citations
- hallucination and abstention behavior
- latency and route retry cost
- robustness by document type and query type

The default manifest is:

```text
data/user_pdf_benchmark_suite.json
```

It currently includes:

- `qcdt_policy_vi`: Vietnamese policy/regulation PDF
- `attention_scientific_en`: scientific paper PDF
- `operations_handbook_en`: controlled handbook/manual PDF

## Benchmark Ideas Used

The suite maps the project to known evaluation families:

- BEIR-style retrieval comparison: BM25 vs dense vs hybrid vs rerank.
- MIRACL-style multilingual retrieval check: Vietnamese policy documents should
  be evaluated separately from English PDFs.
- QASPER-style scientific-paper QA: long paper sections, methods, results,
  tables, and insufficient-evidence questions.
- DocVQA-style grounded document QA: answers must point back to page/chunk
  evidence, not just match text.
- TAT-QA-style text/table QA: questions may require numeric values from tables
  or mixed table-text evidence.
- MultiDoc2Dial-style policy/manual QA: procedural and policy questions should
  be checked separately.

For ingest quality, keep using DocLayNet and PubTables-1M through the existing
ingest benchmark scripts. This suite focuses on final QA behavior.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_user_pdf_suite.py `
  --manifest data\user_pdf_benchmark_suite.json `
  --output-dir results\user_pdf_benchmark_suite\current
```

The runner builds a missing index unless `--skip-build` is passed. Existing
indexes are reused unless `--rebuild-index` is passed.

Run only one document:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_user_pdf_suite.py `
  --manifest data\user_pdf_benchmark_suite.json `
  --output-dir results\user_pdf_benchmark_suite\attention_only `
  --doc-id attention_scientific_en
```

Override configs:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_user_pdf_suite.py `
  --manifest data\user_pdf_benchmark_suite.json `
  --output-dir results\user_pdf_benchmark_suite\routed_vs_adaptive `
  --config routed_grounded,adaptive_route_retry
```

## Outputs

Each run writes:

- `suite_summary.json`: aggregate metrics by config, document type, and language
- `per_question.json`: full QA trace with document metadata
- `per_question.csv`: spreadsheet-friendly trace
- `README.md`: short markdown report table
- `documents/<doc_id>/`: original `benchmark_qa.py` output per document

Key fields to inspect:

- `end_to_end_success_rate`: answer match + evidence match + grounded answer
- `answer_match_rate`: answer quality against `gold_answer` or `match_text`
- `evidence_match_rate`: whether retrieved/cited evidence hits gold evidence
- `grounded_rate`: whether answers have citation support
- `hallucination_rate`: wrong answer or unsupported answer behavior
- `abstention_precision` and `abstention_recall`: insufficient-evidence handling
- `route_retry_rate` and `avg_route_attempt_count`: adaptive controller cost

## Add A New User PDF

1. Put the PDF under `data/real_pdfs/`.
2. Create a matching JSONL query file.
3. Add a document entry to `data/user_pdf_benchmark_suite.json`.
4. Run the suite.

Minimal query row:

```json
{"id":"q01","question":"What is the policy deadline?","query_type":"factoid","gold_answer":"The deadline is 30 days.","gold_pages":[4],"gold_sections":["Deadlines"],"match_text":"30 days","should_answer":true}
```

Recommended query fields:

- `id`
- `question`
- `query_type`: `factoid`, `definition`, `policy`, `procedural`,
  `comparison`, `multi_hop`, or `ambiguous_or_insufficient`
- `gold_answer`
- `expected_chunk_ids` when known
- `gold_pages` or `gold_sections`
- `match_text` for robust exact matching
- `should_answer`
- `expected_decision` for insufficient-evidence questions

Recommended minimum per real document:

- 8-12 factoid / definition questions
- 5-8 policy or procedural questions when relevant
- 3-5 comparison or multi-hop questions
- 3-5 insufficient-evidence questions
- at least a few table/numeric questions when the PDF has tables

## Interpretation

Do not expect one method to win on every PDF type.

- Policy/regulation PDFs often favor BM25 because exact terms, article numbers,
  and legal phrases matter.
- Scientific papers often benefit from hybrid retrieval because questions mix
  semantic paraphrases with exact numeric/table evidence.
- Manuals and handbooks are useful for testing procedural grounding and
  abstention.
- If `no_citation_grounding` has high answer match but zero grounded rate, the
  citation layer is doing useful anti-hallucination work.
- If adaptive route-retry improves success but latency rises sharply, report the
  tradeoff rather than treating it as a free improvement.
