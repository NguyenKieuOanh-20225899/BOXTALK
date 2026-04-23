from __future__ import annotations

import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "retrieval_smoke"


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    pdf_path = DATA_ROOT / "employee_handbook_smoke.pdf"
    queries_path = DATA_ROOT / "queries.jsonl"

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    lines = [
        ("BOXTALK Employee Handbook (Smoke)", 20, (72, 72)),
        ("Attendance Policy", 16, (72, 120)),
        ("Employees must badge in before 09:00 on workdays.", 11, (72, 150)),
        ("Late arrivals after 09:15 require manager approval.", 11, (72, 168)),
        ("", 11, (72, 186)),
        ("Expense Reimbursement", 16, (72, 220)),
        ("Staff must submit receipts within 30 days of the purchase.", 11, (72, 250)),
        ("Finance reviews reimbursement requests every Friday.", 11, (72, 268)),
        ("", 11, (72, 286)),
        ("Security Contacts", 16, (72, 320)),
        ("Security incidents must be reported to security@boxtalk.local.", 11, (72, 350)),
        ("Urgent access issues should be escalated to the IT duty line.", 11, (72, 368)),
        ("", 11, (72, 386)),
        ("Benefits Table", 16, (72, 420)),
        ("Benefit  Waiting period  Owner", 11, (72, 450)),
        ("Health plan  30 days  HR Ops", 11, (72, 468)),
        ("VPN access  Same day  IT Support", 11, (72, 486)),
    ]

    for text, size, point in lines:
        if not text:
            continue
        page.insert_text(point, text, fontsize=size)

    doc.save(pdf_path)
    doc.close()

    queries = [
        {
            "query_id": "q1",
            "query": "When must employees badge in on workdays?",
            "source_name": pdf_path.name,
            "expected_section": "Attendance Policy",
            "match_text": "before 09:00",
            "gold_answer": "Employees must badge in before 09:00 on workdays.",
        },
        {
            "query_id": "q2",
            "query": "Who approves late arrivals after 09:15?",
            "source_name": pdf_path.name,
            "expected_section": "Attendance Policy",
            "match_text": "manager approval",
            "gold_answer": "Late arrivals after 09:15 require manager approval.",
        },
        {
            "query_id": "q3",
            "query": "How long do staff have to submit receipts?",
            "source_name": pdf_path.name,
            "expected_section": "Expense Reimbursement",
            "match_text": "within 30 days",
            "gold_answer": "Staff must submit receipts within 30 days of the purchase.",
        },
        {
            "query_id": "q4",
            "query": "Which email handles security incidents?",
            "source_name": pdf_path.name,
            "expected_section": "Security Contacts",
            "match_text": "security@boxtalk.local",
            "gold_answer": "Security incidents must be reported to security@boxtalk.local.",
        },
        {
            "query_id": "q5",
            "query": "Who owns VPN access in the benefits table?",
            "source_name": pdf_path.name,
            "match_text": "IT Support",
            "gold_answer": "VPN access is owned by IT Support.",
        },
    ]

    with queries_path.open("w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(DATA_ROOT)


if __name__ == "__main__":
    main()
