from __future__ import annotations

import json
import sys
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

console = Console()


def _display_text(value: Any, default: str = "-") -> str:
    if value in (None, "", []):
        return default
    return str(value)


def _bool_status(value: bool | None, *, true_label: str = "Hợp lệ", false_label: str = "Không hợp lệ") -> Text:
    if value is True:
        return Text(true_label, style="bold green")
    if value is False:
        return Text(false_label, style="bold red")
    return Text("-", style="dim")


def _latency_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.1f} ms"


def _score(hit: Any) -> float:
    final_score = getattr(hit, "final_score", None)
    raw_score = getattr(hit, "score", 0.0)
    return float(final_score if final_score is not None else raw_score)


def _hit_metadata(hit: Any) -> dict[str, Any]:
    chunk = getattr(hit, "chunk", None)
    chunk_metadata = getattr(chunk, "metadata", None) or {}
    hit_metadata = getattr(hit, "metadata", None) or {}
    return {**dict(chunk_metadata), **dict(hit_metadata)}


def _decision_text(decision: Any) -> Text:
    text = _display_text(decision)
    normalized = text.strip().lower()
    if normalized == "answer":
        return Text(text, style="bold green")
    if normalized in {"abstain", "switch_strategy"}:
        return Text(text, style="bold red")
    if normalized in {"expand_retrieval"}:
        return Text(text, style="bold yellow")
    return Text(text, style="bold")


def _evidence_location(metadata: dict[str, Any]) -> str:
    target = str(metadata.get("citation_target") or "").strip().lower()
    if target == "cell":
        return (
            f"Hàng: {_display_text(metadata.get('row_header'))}\n"
            f"Cột: {_display_text(metadata.get('col_header'))}\n"
            f"Ô: {_display_text(metadata.get('cell_text'))}"
        )
    if target == "row":
        return f"Hàng: {_display_text(metadata.get('row_header'))}"
    if target:
        return f"Đích dẫn chứng: {target}"
    return "-"


def print_result(result: Any, *, show_evidence: int = 5, show_context: bool = False) -> None:
    retrieved_hits = list(getattr(result, "retrieved_hits", []) or [])
    citations = list(getattr(result, "citations", []) or [])
    hit_text_by_chunk_id = {str(hit.chunk_id): hit.text for hit in retrieved_hits}

    console.print()
    console.print(
        Panel(
            Text(_display_text(getattr(result, "question", None)), style="bold white"),
            title="[bold cyan]KẾT QUẢ HỎI ĐÁP[/bold cyan]",
            subtitle="[dim]Grounded PDF QA terminal[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    overview = Table(
        title="Tổng quan phiên chạy",
        box=box.SQUARE,
        expand=True,
        show_lines=True,
        header_style="bold black on cyan",
        border_style="cyan",
        padding=(0, 1),
    )
    overview.add_column("Mục", style="bold cyan", width=18, no_wrap=True)
    overview.add_column("Giá trị", ratio=1, overflow="fold")
    overview.add_column("Mục", style="bold cyan", width=18, no_wrap=True)
    overview.add_column("Giá trị", ratio=1, overflow="fold")
    overview.add_row(
        "Loại câu hỏi",
        _display_text(getattr(result, "query_type", None)),
        "Chiến lược",
        _display_text(getattr(result, "retrieval_strategy", None)),
    )
    overview.add_row(
        "Quyết định",
        _decision_text(getattr(result, "decision", None)),
        "Nguồn trả lời",
        _display_text(getattr(result, "final_answer_source", None)),
    )
    overview.add_row(
        "Có căn cứ",
        _bool_status(getattr(result, "grounded", None), true_label="Có", false_label="Không"),
        "Kiểm tra",
        _bool_status(getattr(result, "validation_passed", None)),
    )
    console.print(overview)

    latency_table = Table(
        title="Thời gian xử lý",
        box=box.SQUARE,
        expand=True,
        show_lines=True,
        header_style="bold black on magenta",
        border_style="magenta",
        padding=(0, 1),
    )
    latency_table.add_column("Giai đoạn", style="bold magenta", width=24, no_wrap=True)
    latency_table.add_column("Độ trễ", justify="right", style="bold white", no_wrap=True)
    latency_table.add_column("Ghi chú", ratio=1, overflow="fold")
    latency_table.add_row(
        "Truy xuất",
        _latency_text(getattr(result, "retrieval_latency_ms", None)),
        "Tìm và xếp hạng bằng chứng",
    )
    latency_table.add_row(
        "Sinh câu trả lời",
        _latency_text(getattr(result, "answer_latency_ms", None)),
        "Từ context đến câu trả lời cuối",
    )
    latency_table.add_row(
        "LLM",
        _latency_text(getattr(result, "llm_latency_ms", None)),
        "Thời gian gọi mô hình",
    )
    latency_table.add_row(
        "Tổng",
        Text(_latency_text(getattr(result, "total_latency_ms", None)), style="bold magenta"),
        "Toàn bộ lượt hỏi đáp",
    )
    console.print(latency_table)

    console.print(Rule("[bold yellow]BẰNG CHỨNG[/bold yellow]", style="yellow"))
    evidence_table = Table(
        title="Tóm tắt bằng chứng",
        box=box.SQUARE,
        expand=True,
        show_lines=True,
        header_style="bold black on yellow",
        border_style="yellow",
        row_styles=["", "dim"],
        padding=(0, 1),
    )
    evidence_table.add_column("Đã truy xuất", justify="center")
    evidence_table.add_column("Đã chọn", justify="center")
    evidence_table.add_column("Đủ bằng chứng", justify="center")
    evidence_table.add_column("Lý do", ratio=3)
    evidence_table.add_row(
        str(len(retrieved_hits)),
        str(getattr(result, "selected_evidence_count", 0)),
        _bool_status(getattr(result, "evidence_sufficient", None), true_label="Đủ", false_label="Thiếu"),
        _display_text(getattr(result, "evidence_reason", None)),
    )
    console.print(evidence_table)

    missing_constraints = list(getattr(result, "missing_constraints", []) or [])
    if missing_constraints:
        console.print(
            Panel(
                "\n".join(f"- {item}" for item in missing_constraints),
                title="[bold red]THIẾU ĐIỀU KIỆN[/bold red]",
                border_style="red",
            )
        )

    context_token_count = getattr(result, "context_token_count", None)
    if context_token_count is not None:
        selected_ids = list(getattr(result, "selected_evidence_ids", []) or [])
        context_table = Table.grid(expand=True, padding=(0, 1))
        context_table.add_column(style="bold cyan", width=22)
        context_table.add_column()
        context_table.add_row("Evidence IDs", ", ".join(map(str, selected_ids)) if selected_ids else "-")
        context_table.add_row("Context tokens", f"{context_token_count:,}")
        console.print(
            Panel(
                context_table,
                title="[bold cyan]NGỮ CẢNH[/bold cyan]",
                border_style="cyan",
            )
        )
        if show_context and getattr(result, "context_evidence", None):
            console.print(Rule("[bold cyan]NỘI DUNG NGỮ CẢNH[/bold cyan]", style="cyan"))
            print_context_evidence(result.context_evidence)

    generator_type = _display_text(getattr(result, "generator_type", None))
    if bool(getattr(result, "evidence_sufficient", False)) or generator_type != "llm_grounded":
        generator_table = Table.grid(expand=True, padding=(0, 1))
        generator_table.add_column(style="bold magenta", width=20)
        generator_table.add_column()
        generator_table.add_row("Type", generator_type)
        generator_table.add_row("Provider", _display_text(getattr(result, "generator_provider", None)))
        generator_table.add_row("Model", _display_text(getattr(result, "generator_model", None)))
        generator_table.add_row("LLM latency", _latency_text(getattr(result, "llm_latency_ms", None)))
        console.print(
            Panel(
                generator_table,
                title="[bold magenta]BỘ SINH[/bold magenta]",
                border_style="magenta",
            )
        )

    validation_passed = getattr(result, "validation_passed", None)
    validation_table = Table.grid(expand=True, padding=(0, 1))
    validation_table.add_column(style="bold blue", width=16)
    validation_table.add_column()
    validation_table.add_row("Passed", _bool_status(validation_passed))
    if getattr(result, "validation_reason", None):
        validation_table.add_row("Reason", str(result.validation_reason))
    console.print(
        Panel(
            validation_table,
            title="[bold blue]KIỂM TRA[/bold blue]",
            border_style="green" if validation_passed else "red",
        )
    )

    answer_border = "green" if getattr(result, "grounded", False) and validation_passed else "red"
    console.print(
        Panel(
            Text(_display_text(getattr(result, "answer", None)), style="bold white"),
            title="[bold green]CÂU TRẢ LỜI[/bold green]",
            border_style=answer_border,
            padding=(1, 2),
        )
    )

    if citations:
        console.print(Rule("[bold green]DẪN CHỨNG[/bold green]", style="green"))
        for idx, citation in enumerate(citations, start=1):
            chunk_id = str(citation.get("chunk_id") or "")
            chunk_text = hit_text_by_chunk_id.get(chunk_id)
            citation_body = Table.grid(expand=True)
            citation_body.add_row(Text(format_citation(citation), style="bold white"))
            if chunk_text:
                citation_body.add_row("")
                citation_body.add_row(Text("Đoạn nguồn:", style="bold cyan"))
                citation_body.add_row(Text(format_chunk_text(chunk_text), overflow="fold"))
            console.print(
                Panel(
                    citation_body,
                    title=f"[bold green]Dẫn chứng {idx}[/bold green]",
                    subtitle=f"[dim]chunk_id={chunk_id or '-'}[/dim]",
                    border_style="green",
                )
            )

    hits = retrieved_hits[: max(0, show_evidence)]
    if hits:
        console.print(
            Rule(f"[bold yellow]TOP {len(hits)} ĐOẠN TRUY XUẤT HÀNG ĐẦU[/bold yellow]", style="yellow")
        )
        for hit in hits:
            metadata = _hit_metadata(hit)
            detail_table = Table(
                box=box.SQUARE,
                expand=True,
                show_header=False,
                show_lines=True,
                border_style="yellow",
                padding=(0, 1),
            )
            detail_table.add_column("Field", style="bold cyan", width=18, no_wrap=True)
            detail_table.add_column("Value", ratio=1, overflow="fold")
            detail_table.add_row("Rank", f"#{_display_text(getattr(hit, 'rank', None))}")
            detail_table.add_row("Score", f"{_score(hit):.3f}")
            detail_table.add_row(
                "Page",
                _display_text(getattr(hit, "page", None)),
            )
            detail_table.add_row("Block type", _display_text(getattr(hit.chunk, "block_type", None)))
            detail_table.add_row(
                "Chunking",
                _display_text(metadata.get("chunking_strategy")),
            )
            detail_table.add_row("Chunk ID", _display_text(getattr(hit, "chunk_id", None)))
            location = _evidence_location(metadata)
            if location != "-":
                detail_table.add_row("Table/cell", location)

            evidence_body = Table.grid(expand=True)
            evidence_body.add_row(detail_table)
            evidence_body.add_row("")
            evidence_body.add_row(Text(format_chunk_text(getattr(hit, "text", "")), overflow="fold"))
            console.print(
                Panel(
                    evidence_body,
                    title=f"[bold yellow]Bằng chứng #{_display_text(getattr(hit, 'rank', None))}[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 1),
                )
            )

    console.print(Rule("[bold cyan]HOÀN TẤT[/bold cyan]", style="cyan"))
    console.print()


def format_citation(citation: dict[str, Any]) -> str:
    parts = []
    for key in ("source_name", "doc_id", "page", "section", "chunk_id", "citation_target"):
        value = citation.get(key)
        if value not in (None, "", []):
            parts.append(f"{key}={value}")
    metadata = citation.get("metadata") if isinstance(citation.get("metadata"), dict) else {}
    for key in ("table_id", "row_header", "col_header", "cell_text"):
        value = metadata.get(key) or citation.get(key)
        if value not in (None, "", []):
            parts.append(f"{key}={value}")
    return " | ".join(parts) if parts else json.dumps(citation, ensure_ascii=False)


def print_context_evidence(context_evidence: list[dict[str, Any]]) -> None:
    for item in context_evidence:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        detail_table = Table(
            box=box.SQUARE,
            expand=True,
            show_header=False,
            show_lines=True,
            border_style="yellow",
            padding=(0, 1),
        )
        detail_table.add_column("Field", style="bold cyan", width=18, no_wrap=True)
        detail_table.add_column("Value", ratio=1, overflow="fold")
        detail_table.add_row("Evidence ID", _display_text(item.get("evidence_id")))
        detail_table.add_row("Page", _display_text(item.get("page")))
        detail_table.add_row("Chunk ID", _display_text(item.get("chunk_id")))
        detail_table.add_row("Target", _display_text(item.get("citation_target")))
        for key, label in (
            ("table_id", "Table ID"),
            ("row_header", "Row"),
            ("col_header", "Column"),
            ("cell_text", "Cell"),
        ):
            value = metadata.get(key)
            if value not in (None, "", []):
                detail_table.add_row(label, str(value))

        body = Table.grid(expand=True)
        body.add_row(detail_table)
        body.add_row("")
        body.add_row(Text(format_chunk_text(str(item.get("text") or "")), overflow="fold"))
        console.print(
            Panel(
                body,
                title=f"[bold yellow]Context {_display_text(item.get('evidence_id'))}[/bold yellow]",
                border_style="yellow",
                padding=(1, 1),
            )
        )


def format_chunk_text(text: str) -> str:
    return " ".join((text or "").split())
