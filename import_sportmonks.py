r"""Download and normalise a Sportmonks season.

Run from the project root, for example:

    python .\import_sportmonks.py --season 2025-2026 --league serie-a

Rebuild processed datasets without making API requests:

    python .\import_sportmonks.py --season 2025-2026 --normalize-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from src.providers.sportmonks import SportmonksClient
from src.providers.sportmonks.importer import SportmonksImporter, select_leagues


class _ApiDisabledClient:
    """Safety net: cache-only normalization must never reach the API client."""

    def get(self, *_args, **_kwargs):
        raise RuntimeError("API access is disabled in --normalize-only mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import Sportmonks data with a resumable local cache."
    )
    parser.add_argument("--season", default="2025-2026", help="Season in YYYY-YYYY format")
    parser.add_argument(
        "--league",
        action="append",
        help="League slug/name; repeat it or use comma-separated values. Default: all five.",
    )
    parser.add_argument(
        "--max-fixtures",
        type=int,
        help="Limit completed fixtures per league (useful for a smoke test).",
    )
    parser.add_argument(
        "--skip-xg",
        action="store_true",
        help="Skip team and player expected-goal relationships.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached JSON and download the selected scope again.",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help=(
            "Read the existing raw JSON cache and rebuild processed datasets without "
            "using an API token or making any Sportmonks requests."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_fixtures is not None and args.max_fixtures < 1:
        print("ERROR: --max-fixtures must be at least 1.", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parent
    if args.normalize_only:
        incompatible = []
        if args.refresh:
            incompatible.append("--refresh")
        if args.skip_xg:
            incompatible.append("--skip-xg")
        if args.max_fixtures is not None:
            incompatible.append("--max-fixtures")
        if args.league:
            incompatible.append("--league")
        if incompatible:
            print(
                "ERROR: --normalize-only rebuilds the complete cached season and cannot be "
                f"combined with {', '.join(incompatible)}.",
                file=sys.stderr,
            )
            return 2
        try:
            print("Cache-only normalization: no Sportmonks API requests will be made.")
            importer = SportmonksImporter(
                _ApiDisabledClient(),
                project_root,
                args.season,
                cache_only=True,
            )
            importer.run(select_leagues(None))
        except (ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    # The project-local file is authoritative even when Conda/base defines a
    # stale variable with the same name.
    load_dotenv(project_root / ".env", override=True)
    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    if not token:
        print(
            "ERROR: SPORTMONKS_API_TOKEN is missing. Copy .env.example to .env and add the token.",
            file=sys.stderr,
        )
        return 2

    try:
        leagues = select_leagues(args.league)
        with SportmonksClient(token) as client:
            importer = SportmonksImporter(
                client,
                project_root,
                args.season,
                refresh=args.refresh,
                skip_xg=args.skip_xg,
                max_fixtures=args.max_fixtures,
            )
            importer.run(leagues)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
