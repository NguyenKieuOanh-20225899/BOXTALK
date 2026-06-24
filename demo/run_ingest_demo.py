from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

BOOTSTRAP_PATH = Path(__file__).resolve().parent / "bootstrap.py"
BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("demo_bootstrap", BOOTSTRAP_PATH)
if BOOTSTRAP_SPEC is None or BOOTSTRAP_SPEC.loader is None:
    raise RuntimeError(f"Cannot load demo bootstrap: {BOOTSTRAP_PATH}")
BOOTSTRAP = importlib.util.module_from_spec(BOOTSTRAP_SPEC)
BOOTSTRAP_SPEC.loader.exec_module(BOOTSTRAP)
DEMO_ROOT = BOOTSTRAP.DEMO_ROOT
BOOTSTRAP.ensure_repo_on_path()

from demo.adapters.ingest_adapter import DemoError, run_ingest_page, run_ingest_page_region_off
from demo.adapters.output_adapter import write_outputs
from demo.adapters.terminal_adapter import print_error, print_result


CONFIG_PATH = DEMO_ROOT / "config" / "demo_config.yaml"
OUTPUT_ROOT = DEMO_ROOT / "output"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terminal demo for one-page PDF ingest and extraction."
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Duong dan toi tep PDF.")
    parser.add_argument("--page", type=int, required=True, help="So trang tinh tu 1.")
    parser.add_argument("--output", type=Path, required=True, help="Thu muc luu ket qua.")
    parser.add_argument("--save-overlay", action="store_true", default=None, help="Luu lop phu truc quan.")
    parser.add_argument("--region-routing", choices=["on", "off"], default="on")
    parser.add_argument("--ocr-mode", choices=["auto", "always", "never"], default=None)
    parser.add_argument(
        "--table-extractor",
        choices=["configured", "default", "tatr", "hybrid_tatr"],
        default=None,
    )
    parser.add_argument("--open-result", action="store_true", help="Mo lop phu hoac thu muc output.")
    parser.add_argument("--verbose", action="store_true", help="Hien log loi chi tiet.")
    parser.add_argument("--no-color", action="store_true", help="Tat mau terminal.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_demo_config(CONFIG_PATH)

    save_overlay = (
        args.save_overlay
        if args.save_overlay is not None
        else bool(config.get("demo", {}).get("save_overlay", False))
    )
    ocr_mode = args.ocr_mode or str(config.get("ingest", {}).get("ocr_mode", "auto"))
    table_extractor = args.table_extractor or str(
        config.get("ingest", {}).get("table_extractor", "configured")
    )
    show_library_logs = bool(config.get("terminal", {}).get("show_library_logs", False)) or args.verbose

    try:
        output_dir = _validate_output_dir(args.output)
        if args.region_routing == "off":
            result = run_ingest_page_region_off(
                pdf_path=args.pdf,
                page_number=args.page,
                show_library_logs=show_library_logs,
            )
        else:
            result = run_ingest_page(
                pdf_path=args.pdf,
                page_number=args.page,
                ocr_mode=ocr_mode,
                table_extractor=table_extractor,
                show_library_logs=show_library_logs,
            )
        files = write_outputs(result, output_dir=output_dir, save_overlay=save_overlay)
        print_result(result, files=files, color=not args.no_color, verbose=args.verbose)
        if args.open_result:
            _open_result(files.get("overlay_png") or output_dir)
        return 0
    except DemoError as exc:
        print_error(str(exc), verbose=args.verbose, exc=exc)
        return 2
    except Exception as exc:
        print_error(f"Loi khong mong doi: {exc}", verbose=args.verbose, exc=exc)
        return 1


def load_demo_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            current_section = {}
            root[section] = current_section
            continue
        if current_section is None or ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        current_section[key.strip()] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    try:
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _validate_output_dir(path: Path) -> Path:
    output = path.resolve()
    root = OUTPUT_ROOT.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise DemoError(f"--output phai nam trong {OUTPUT_ROOT}") from exc
    return output


def _open_result(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            print(f"Mo thu cong: {path}")
    except Exception as exc:
        print(f"[CANH BAO] Khong mo duoc ket qua: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
