#!/usr/bin/env python3
"""REPORT-11 CLI: audit one raw Opta match through the complete report pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.audit import (  # noqa: E402
    MATCH_JSON_ENV_VAR,
    print_audit_result,
    resolve_match_json_path,
    run_match_report_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process a raw Opta JSON, build the neutral bundle, generate/export "
            "all report plots, create the Match Analysis Pack, then reopen and "
            "validate ZIP/PDF/JSON/CSV artifacts."
        )
    )
    parser.add_argument(
        "match_json",
        nargs="?",
        help=(
            "Path to raw Opta match JSON. If omitted, use "
            f"{MATCH_JSON_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory where PNG diagnostics and the generated pack are kept. "
            "If omitted, a temporary directory is used."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = resolve_match_json_path(args.match_json)
    except Exception as exc:
        print(f"REPORT-11 input error: {exc}", file=sys.stderr)
        return 2

    result = run_match_report_audit(source, output_dir=args.output_dir)
    print_audit_result(result)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
