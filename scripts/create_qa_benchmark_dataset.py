from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "qa_benchmark"


SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Attendance Policy",
        [
            "Employees must check in by 09:00 on regular workdays.",
            "Late arrivals after 09:15 require manager approval before the end of the day.",
            "Three unapproved late arrivals in one calendar month trigger an attendance review.",
        ],
    ),
    (
        "Expense Reimbursement Policy",
        [
            "Staff must submit itemized receipts within 30 days of the purchase date.",
            "Finance reviews reimbursement requests every Friday and pays approved claims in the next payroll cycle.",
            "Meals above 75 USD require a business purpose and the attendee list.",
        ],
    ),
    (
        "Security Incident Response",
        [
            "Security incidents must be reported to security@boxtalk.local within two hours of discovery.",
            "The security lead classifies incidents as low, medium, or high severity.",
            "High severity incidents require an executive briefing within one business day.",
        ],
    ),
    (
        "Access Management Policy",
        [
            "Standard application access is provisioned by IT Support on the same day after manager approval.",
            "Privileged access requires security approval, manager approval, and automatic expiration after 90 days.",
            "Contractor access must end on the contract end date unless the sponsor renews it.",
        ],
    ),
    (
        "Equipment Loan Procedure",
        [
            "Employees may borrow loaner laptops for up to 14 days.",
            "To request equipment, submit an IT ticket, include the asset type, and confirm manager approval.",
            "Returned equipment must be inspected by IT before the ticket is closed.",
        ],
    ),
    (
        "Travel Approval Rules",
        [
            "Domestic travel under 2,000 USD may be approved by the department manager.",
            "Travel above 2,000 USD or any international travel requires director approval.",
            "Travel requests must include destination, dates, estimated cost, and business purpose.",
        ],
    ),
    (
        "Vendor Onboarding Procedure",
        [
            "New vendors require procurement review, security screening, and finance setup before work begins.",
            "The requester must attach the statement of work and data processing description.",
            "Vendors that handle customer data require a completed security questionnaire.",
        ],
    ),
    (
        "Records Retention",
        [
            "Visitor logs are retained for 18 months.",
            "Signed customer contracts are retained for seven years after expiration.",
            "Routine support tickets are retained for two years unless tied to a legal hold.",
        ],
    ),
    (
        "Remote Work Policy",
        [
            "Employees may work remotely up to three days per week with manager approval.",
            "Remote work from another country requires People Ops review and tax approval before travel.",
            "Employees must use the company VPN when accessing internal systems remotely.",
        ],
    ),
    (
        "Support Channels and Escalation",
        [
            "Urgent production outages must be posted in the #ops-urgent channel.",
            "Non-urgent IT requests should be submitted through the helpdesk portal.",
            "Customer-impacting incidents require a status update every 30 minutes until resolved.",
        ],
    ),
    (
        "Benefit Eligibility",
        [
            "Full-time employees become eligible for the health plan after 30 days of employment.",
            "Part-time employees working at least 20 hours per week are eligible for wellness reimbursement.",
            "Contractors are not eligible for employee health benefits.",
        ],
    ),
    (
        "Report Review Cadence",
        [
            "The finance forecast is reviewed monthly by the CFO and department leads.",
            "Security risk reports are reviewed quarterly by the risk committee.",
            "The customer health dashboard is reviewed every Monday by Customer Success.",
        ],
    ),
    (
        "Standard Access Versus Privileged Access",
        [
            "Standard access grants normal application permissions after manager approval.",
            "Privileged access grants administrative permissions and requires both security and manager approval.",
            "Privileged access expires after 90 days, while standard access remains active until role change or termination.",
        ],
    ),
    (
        "Policy Exceptions",
        [
            "Policy exceptions must document the business reason, risk owner, expiration date, and compensating control.",
            "Exceptions longer than 60 days require director approval.",
            "Expired exceptions must be closed or renewed before the expiration date.",
        ],
    ),
]


QUERIES: list[dict[str, Any]] = [
    {
        "query_id": "factoid_001",
        "query_type": "factoid",
        "query": "When must employees check in on regular workdays?",
        "expected_section": "Attendance Policy",
        "match_text": "09:00 on regular workdays",
        "gold_answer": "Employees must check in by 09:00 on regular workdays.",
        "should_answer": True,
    },
    {
        "query_id": "factoid_002",
        "query_type": "factoid",
        "query": "Which email receives security incident reports?",
        "expected_section": "Security Incident Response",
        "match_text": "security@boxtalk.local",
        "gold_answer": "Security incidents must be reported to security@boxtalk.local within two hours of discovery.",
        "should_answer": True,
    },
    {
        "query_id": "factoid_003",
        "query_type": "factoid",
        "query": "Who provisions standard application access?",
        "expected_section": "Access Management Policy",
        "match_text": "IT Support",
        "gold_answer": "Standard application access is provisioned by IT Support on the same day after manager approval.",
        "should_answer": True,
    },
    {
        "query_id": "factoid_004",
        "query_type": "factoid",
        "query": "How long may employees borrow loaner laptops?",
        "expected_section": "Equipment Loan Procedure",
        "match_text": "up to 14 days",
        "gold_answer": "Employees may borrow loaner laptops for up to 14 days.",
        "should_answer": True,
    },
    {
        "query_id": "factoid_005",
        "query_type": "factoid",
        "query": "How long are visitor logs retained?",
        "expected_section": "Records Retention",
        "match_text": "18 months",
        "gold_answer": "Visitor logs are retained for 18 months.",
        "should_answer": True,
    },
    {
        "query_id": "factoid_006",
        "query_type": "factoid",
        "query": "Who approves international travel?",
        "expected_section": "Travel Approval Rules",
        "match_text": "director approval",
        "gold_answer": "Travel above 2,000 USD or any international travel requires director approval.",
        "should_answer": True,
    },
    {
        "query_id": "factoid_007",
        "query_type": "factoid",
        "query": "Which team reviews reimbursement requests every Friday?",
        "expected_section": "Expense Reimbursement Policy",
        "match_text": "Finance reviews reimbursement requests every Friday",
        "gold_answer": "Finance reviews reimbursement requests every Friday.",
        "should_answer": True,
    },
    {
        "query_id": "factoid_008",
        "query_type": "factoid",
        "query": "Which channel is used for urgent production outages?",
        "expected_section": "Support Channels and Escalation",
        "match_text": "#ops-urgent",
        "gold_answer": "Urgent production outages must be posted in the #ops-urgent channel.",
        "should_answer": True,
    },
    {
        "query_id": "policy_001",
        "query_type": "policy",
        "query": "What is the policy for late arrivals after 09:15?",
        "expected_section": "Attendance Policy",
        "match_text": "manager approval before the end of the day",
        "gold_answer": "Late arrivals after 09:15 require manager approval before the end of the day.",
        "should_answer": True,
    },
    {
        "query_id": "policy_002",
        "query_type": "policy",
        "query": "What receipts are required for expense reimbursement?",
        "expected_section": "Expense Reimbursement Policy",
        "match_text": "itemized receipts within 30 days",
        "gold_answer": "Staff must submit itemized receipts within 30 days of the purchase date.",
        "should_answer": True,
    },
    {
        "query_id": "policy_003",
        "query_type": "policy",
        "query": "What policy applies to meals above 75 USD?",
        "expected_section": "Expense Reimbursement Policy",
        "match_text": "business purpose and the attendee list",
        "gold_answer": "Meals above 75 USD require a business purpose and the attendee list.",
        "should_answer": True,
    },
    {
        "query_id": "policy_004",
        "query_type": "policy",
        "query": "What is required for privileged access?",
        "expected_section": "Access Management Policy",
        "match_text": "security approval, manager approval, and automatic expiration after 90 days",
        "gold_answer": "Privileged access requires security approval, manager approval, and automatic expiration after 90 days.",
        "should_answer": True,
    },
    {
        "query_id": "policy_005",
        "query_type": "policy",
        "query": "What is the policy for contractor access end dates?",
        "expected_section": "Access Management Policy",
        "match_text": "Contractor access must end on the contract end date",
        "gold_answer": "Contractor access must end on the contract end date unless the sponsor renews it.",
        "should_answer": True,
    },
    {
        "query_id": "policy_006",
        "query_type": "policy",
        "query": "What approval is required for travel above 2,000 USD?",
        "expected_section": "Travel Approval Rules",
        "match_text": "requires director approval",
        "gold_answer": "Travel above 2,000 USD or any international travel requires director approval.",
        "should_answer": True,
    },
    {
        "query_id": "policy_007",
        "query_type": "policy",
        "query": "What is the remote work policy per week?",
        "expected_section": "Remote Work Policy",
        "match_text": "up to three days per week",
        "gold_answer": "Employees may work remotely up to three days per week with manager approval.",
        "should_answer": True,
    },
    {
        "query_id": "policy_008",
        "query_type": "policy",
        "query": "What must policy exceptions document?",
        "expected_section": "Policy Exceptions",
        "match_text": "business reason, risk owner, expiration date, and compensating control",
        "gold_answer": "Policy exceptions must document the business reason, risk owner, expiration date, and compensating control.",
        "should_answer": True,
    },
    {
        "query_id": "policy_009",
        "query_type": "policy",
        "query": "What is required for remote work from another country?",
        "expected_section": "Remote Work Policy",
        "match_text": "People Ops review and tax approval before travel",
        "gold_answer": "Remote work from another country requires People Ops review and tax approval before travel.",
        "should_answer": True,
    },
    {
        "query_id": "policy_010",
        "query_type": "policy",
        "query": "What rule applies to customer-impacting incident updates?",
        "expected_section": "Support Channels and Escalation",
        "match_text": "status update every 30 minutes",
        "gold_answer": "Customer-impacting incidents require a status update every 30 minutes until resolved.",
        "should_answer": True,
    },
    {
        "query_id": "procedural_001",
        "query_type": "procedural",
        "query": "What steps are needed to request equipment?",
        "expected_section": "Equipment Loan Procedure",
        "match_text": "submit an IT ticket, include the asset type, and confirm manager approval",
        "gold_answer": "To request equipment, submit an IT ticket, include the asset type, and confirm manager approval.",
        "should_answer": True,
    },
    {
        "query_id": "procedural_002",
        "query_type": "procedural",
        "query": "How should a new vendor be onboarded before work begins?",
        "expected_section": "Vendor Onboarding Procedure",
        "match_text": "procurement review, security screening, and finance setup",
        "gold_answer": "New vendors require procurement review, security screening, and finance setup before work begins.",
        "should_answer": True,
    },
    {
        "query_id": "procedural_003",
        "query_type": "procedural",
        "query": "What should the requester attach during vendor onboarding?",
        "expected_section": "Vendor Onboarding Procedure",
        "match_text": "statement of work and data processing description",
        "gold_answer": "The requester must attach the statement of work and data processing description.",
        "should_answer": True,
    },
    {
        "query_id": "procedural_004",
        "query_type": "procedural",
        "query": "How should returned equipment be handled?",
        "expected_section": "Equipment Loan Procedure",
        "match_text": "inspected by IT before the ticket is closed",
        "gold_answer": "Returned equipment must be inspected by IT before the ticket is closed.",
        "should_answer": True,
    },
    {
        "query_id": "procedural_005",
        "query_type": "procedural",
        "query": "What process is used for security incident classification?",
        "expected_section": "Security Incident Response",
        "match_text": "classifies incidents as low, medium, or high severity",
        "gold_answer": "The security lead classifies incidents as low, medium, or high severity.",
        "should_answer": True,
    },
    {
        "query_id": "procedural_006",
        "query_type": "procedural",
        "query": "How should non-urgent IT requests be submitted?",
        "expected_section": "Support Channels and Escalation",
        "match_text": "submitted through the helpdesk portal",
        "gold_answer": "Non-urgent IT requests should be submitted through the helpdesk portal.",
        "should_answer": True,
    },
    {
        "query_id": "procedural_007",
        "query_type": "procedural",
        "query": "What steps are needed for travel requests?",
        "expected_section": "Travel Approval Rules",
        "match_text": "destination, dates, estimated cost, and business purpose",
        "gold_answer": "Travel requests must include destination, dates, estimated cost, and business purpose.",
        "should_answer": True,
    },
    {
        "query_id": "procedural_008",
        "query_type": "procedural",
        "query": "How are expired policy exceptions handled?",
        "expected_section": "Policy Exceptions",
        "match_text": "closed or renewed before the expiration date",
        "gold_answer": "Expired exceptions must be closed or renewed before the expiration date.",
        "should_answer": True,
    },
    {
        "query_id": "comparison_001",
        "query_type": "comparison",
        "query": "Compare standard access and privileged access approval requirements.",
        "expected_section": "Standard Access Versus Privileged Access",
        "match_text": "Privileged access grants administrative permissions and requires both security and manager approval",
        "gold_answer": "Standard access needs manager approval, while privileged access requires both security and manager approval.",
        "should_answer": True,
    },
    {
        "query_id": "comparison_002",
        "query_type": "comparison",
        "query": "What is the difference between standard access duration and privileged access duration?",
        "expected_section": "Standard Access Versus Privileged Access",
        "match_text": "Privileged access expires after 90 days",
        "gold_answer": "Privileged access expires after 90 days, while standard access remains active until role change or termination.",
        "should_answer": True,
    },
    {
        "query_id": "comparison_003",
        "query_type": "comparison",
        "query": "Compare health plan eligibility for full-time employees and contractors.",
        "expected_section": "Benefit Eligibility",
        "match_text": "Contractors are not eligible for employee health benefits",
        "gold_answer": "Full-time employees become eligible for the health plan after 30 days, but contractors are not eligible.",
        "should_answer": True,
    },
    {
        "query_id": "comparison_004",
        "query_type": "comparison",
        "query": "Compare domestic travel under 2,000 USD with travel above 2,000 USD.",
        "expected_section": "Travel Approval Rules",
        "match_text": "Domestic travel under 2,000 USD may be approved by the department manager",
        "gold_answer": "Domestic travel under 2,000 USD may be approved by the department manager; travel above 2,000 USD requires director approval.",
        "should_answer": True,
    },
    {
        "query_id": "comparison_005",
        "query_type": "comparison",
        "query": "Compare finance forecast review and security risk report review cadence.",
        "expected_section": "Report Review Cadence",
        "match_text": "Security risk reports are reviewed quarterly",
        "gold_answer": "The finance forecast is reviewed monthly, while security risk reports are reviewed quarterly.",
        "should_answer": True,
    },
    {
        "query_id": "comparison_006",
        "query_type": "comparison",
        "query": "What is different between urgent outages and non-urgent IT requests?",
        "expected_section": "Support Channels and Escalation",
        "match_text": "Urgent production outages must be posted in the #ops-urgent channel",
        "gold_answer": "Urgent production outages go to #ops-urgent, while non-urgent IT requests go through the helpdesk portal.",
        "should_answer": True,
    },
    {
        "query_id": "comparison_007",
        "query_type": "comparison",
        "query": "Compare visitor log retention with customer contract retention.",
        "expected_section": "Records Retention",
        "match_text": "Signed customer contracts are retained for seven years after expiration",
        "gold_answer": "Visitor logs are retained for 18 months, while signed customer contracts are retained for seven years after expiration.",
        "should_answer": True,
    },
    {
        "query_id": "comparison_008",
        "query_type": "comparison",
        "query": "Compare full-time and part-time benefit eligibility.",
        "expected_section": "Benefit Eligibility",
        "match_text": "Part-time employees working at least 20 hours per week are eligible for wellness reimbursement",
        "gold_answer": "Full-time employees become eligible for the health plan after 30 days; part-time employees working at least 20 hours per week are eligible for wellness reimbursement.",
        "should_answer": True,
    },
    {
        "query_id": "insufficient_001",
        "query_type": "ambiguous",
        "query": "What is the CEO's home address?",
        "gold_answer": "",
        "should_answer": False,
        "expected_decision": "abstain",
    },
    {
        "query_id": "insufficient_002",
        "query_type": "ambiguous",
        "query": "What is the 2028 salary band for senior engineers?",
        "gold_answer": "",
        "should_answer": False,
        "expected_decision": "abstain",
    },
    {
        "query_id": "insufficient_003",
        "query_type": "ambiguous",
        "query": "Which vendor won the cloud migration contract?",
        "gold_answer": "",
        "should_answer": False,
        "expected_decision": "abstain",
    },
    {
        "query_id": "insufficient_004",
        "query_type": "ambiguous",
        "query": "What medical diagnosis did Alice Nguyen report?",
        "gold_answer": "",
        "should_answer": False,
        "expected_decision": "abstain",
    },
    {
        "query_id": "insufficient_005",
        "query_type": "ambiguous",
        "query": "How many visitor parking spaces are reserved at headquarters?",
        "gold_answer": "",
        "should_answer": False,
        "expected_decision": "abstain",
    },
    {
        "query_id": "insufficient_006",
        "query_type": "ambiguous",
        "query": "What was the Q4 2030 revenue?",
        "gold_answer": "",
        "should_answer": False,
        "expected_decision": "abstain",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a richer local QA benchmark PDF and query set.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf-name", default="operations_handbook.pdf")
    parser.add_argument("--queries-name", default="queries.jsonl")
    return parser.parse_args()


def insert_wrapped(page: fitz.Page, text: str, *, x: float, y: float, font_size: int, width: int) -> float:
    for line in textwrap.wrap(text, width=width):
        page.insert_text((x, y), line, fontsize=font_size)
        y += font_size + 5
    return y


def write_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 64.0
    page.insert_text((72, y), "BOXTALK Operations Handbook", fontsize=20)
    y += 42

    for section, paragraphs in SECTIONS:
        if y > 700:
            page = doc.new_page(width=595, height=842)
            y = 64.0
        page.insert_text((72, y), section, fontsize=16)
        y += 28
        for paragraph in paragraphs:
            y = insert_wrapped(page, paragraph, x=90, y=y, font_size=11, width=84)
            y += 5
        y += 14

    doc.save(path)
    doc.close()


def write_queries(path: Path, pdf_name: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in QUERIES:
            payload = {"source_name": pdf_name, **row}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_manifest(path: Path, pdf_path: Path, queries_path: Path) -> None:
    counts = Counter(str(row["query_type"]) for row in QUERIES)
    manifest = {
        "name": "qa_benchmark_operations_handbook",
        "description": "Controlled local QA benchmark for routed grounded PDF QA baseline and ablation.",
        "pdf": str(pdf_path),
        "queries": str(queries_path),
        "query_count": len(QUERIES),
        "query_type_counts": dict(sorted(counts.items())),
        "answerable_count": sum(1 for row in QUERIES if row["should_answer"]),
        "unanswerable_count": sum(1 for row in QUERIES if not row["should_answer"]),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / args.pdf_name
    queries_path = args.output_dir / args.queries_name

    write_pdf(pdf_path)
    write_queries(queries_path, args.pdf_name)
    write_manifest(args.output_dir / "manifest.json", pdf_path, queries_path)
    print(args.output_dir)


if __name__ == "__main__":
    main()
