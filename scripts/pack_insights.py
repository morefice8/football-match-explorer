#!/usr/bin/env python3
"""Write a markdown insight sheet from a Match Analysis Pack.

Usage:
    py -3.11 scripts/pack_insights.py pack.zip [--team "Napoli"] [--out sheet.md]

Reads the window-analysis tables (pack schema 1.12+) and lays out the phases,
the ranked insight candidates with their numbers, caveats and the questions to
settle before writing them, the pressure and window tables and the droughts.
Without ``--out`` the sheet is printed. Packs built before schema 1.12 are
reported: rebuild them with ``scripts/audit_match_report.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting import insight_sheet  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pack", type=Path, help="Match Analysis Pack zip")
    parser.add_argument("--team", help="Limit the per-team sections and the candidates to this team")
    parser.add_argument("--out", type=Path, help="Write the sheet to this markdown file instead of printing it")
    args = parser.parse_args(argv)

    try:
        tables, summary = insight_sheet.load_pack(args.pack)
        sheet = insight_sheet.render_insight_sheet(tables, summary, team=args.team)
    except insight_sheet.PackTooOldError as error:
        print(f"{args.pack}: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"{args.pack}: {error}", file=sys.stderr)
        return 1

    if args.out:
        args.out.write_text(sheet, encoding="utf-8")
        print(f"Written: {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
