#!/usr/bin/env python3
"""Aggregate the phase profile of one team across several Match Analysis Packs.

Usage:
    py -3.11 scripts/aggregate_phase_profile.py "Napoli" pack1.zip pack2.zip ... [--out profile.csv]

Reads ``tables/phase-sequences.csv`` (pack schema 1.11+) from every zip, keeps the
rows of the requested team and summarises them with the same functions the pack
uses for a single match, so every sequence is weighted once. Packs generated
before schema 1.11 are reported and skipped.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import phase_profile as pp  # noqa: E402

_BOOL_COLUMNS = (
    "first_pass_forward", "reached_final_third", "produced_shot", "regained",
    "opponent_reached_final_third", "opponent_shot",
)


def _read_bool(series: pd.Series) -> pd.Series:
    mapping = {"True": True, "False": False, True: True, False: False}
    return series.map(mapping).astype("object").where(series.notna(), None)


def load_team_sequences(team: str, packs: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames, skipped = [], []
    for path in packs:
        with zipfile.ZipFile(path) as archive:
            name = "tables/phase-sequences.csv"
            if name not in archive.namelist():
                skipped.append(str(path))
                continue
            frame = pd.read_csv(io.BytesIO(archive.read(name)))
        frame = frame[frame["team_name"] == team].copy()
        frame["pack"] = Path(path).name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), skipped
    table = pd.concat(frames, ignore_index=True)
    for column in _BOOL_COLUMNS:
        if column in table.columns:
            table[column] = _read_bool(table[column])
    return table, skipped


def summarise(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = table[table["phase"] == "recovery"].rename(columns={"zone": "start_zone", "x": "start_x", "y": "start_y"})
    loss = table[table["phase"] == "loss"].rename(columns={"zone": "loss_zone", "x": "loss_x", "y": "loss_y"})
    rec = rec.reindex(columns=pp.RECOVERY_SEQUENCE_COLUMNS)
    loss = loss.reindex(columns=pp.LOSS_SEQUENCE_COLUMNS)
    return pp.summarize_recoveries(rec), pp.summarize_losses(loss)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("team")
    parser.add_argument("packs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="Write both summaries to this CSV (phase column added)")
    args = parser.parse_args(argv)

    table, skipped = load_team_sequences(args.team, args.packs)
    for path in skipped:
        print(f"skipped (no phase tables, rebuild the pack): {path}", file=sys.stderr)
    if table.empty:
        print("No phase sequences found.", file=sys.stderr)
        return 2
    matches = table["pack"].nunique()
    recoveries, losses = summarise(table)
    print(f"{args.team}: {matches} match(es)")
    print("\nWHERE THE TEAM WINS THE BALL -> how it attacks")
    print(recoveries.to_string(index=False))
    print("\nWHERE THE TEAM LOSES THE BALL -> how it wins it back")
    print(losses.to_string(index=False))
    if args.out:
        out = pd.concat([recoveries.assign(phase="recovery"), losses.assign(phase="loss")], ignore_index=True)
        out.to_csv(args.out, index=False)
        print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
