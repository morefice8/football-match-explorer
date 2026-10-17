"""REPORT-26: the markdown insight sheet read from a Match Analysis Pack."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting import insight_sheet as sheet
from src.reporting import window_analysis as wa
from test_report25_window_profile import _frame, _skeleton, _tackle, _touch

ROOT = Path(__file__).resolve().parents[1]

SUMMARY = {
    "match": {"home_team": "A", "away_team": "B", "competition": "Test Cup", "date": "2026-01-02T20:00:00Z"},
    "score_and_goals": {
        "home_score": 1,
        "away_score": 0,
        "goals": [{"team": "A", "scorer": "wing", "minute": 30, "attack_type": "Open Play", "creator": None}],
    },
}


def _match():
    """A's wide player is always followed by a B tackle, A's other players never; B only touches."""
    rows = []
    for i in range(30):
        minute = 1 + i
        rows.append(_touch(minute, "A", x=60, y=10, player="wing", pid="w", jersey=11, starter=True))
        rows.append(_tackle(minute, 2, "B", x=40, y=90))
        rows.append(_touch(minute, "A", x=50, y=50, player="mid", pid="m", jersey=8, starter=True))
        rows.append(_touch(minute, "A", x=30, y=50, player="back", pid="k", jersey=4, starter=True))
        rows.append(_touch(minute, "B", x=50, y=50, player="b1", pid="b1", jersey=9, starter=True))
    rows.append(dict(period=1, m=30, s=0, team="A", type="Goal", x=95, y=50, player="wing", pid="w", jersey=11, starter=True))
    return _frame(_skeleton() + rows)


def _pack_tables():
    out = wa.compute_window_analysis(_match(), ("A", "B"))
    return {
        "match-phases": out.phases,
        "window-profile": out.window_profile,
        "pressure-profile": out.pressure_profile,
        "pressure-touches": out.pressure_touches,
        "window-positions": out.positions,
        "shot-droughts": out.droughts,
        "insight-candidates": out.insights,
    }


def _write_pack(path, tables, summary=SUMMARY):
    """Write the tables the way a pack stores them (CSV in a zip) so the sheet reads real round-tripped data."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, frame in tables.items():
            archive.writestr(f"tables/{name}.csv", frame.to_csv(index=False))
        archive.writestr("analysis-summary.json", json.dumps(summary))


def _from_csv(tables):
    return {name: pd.read_csv(io.StringIO(frame.to_csv(index=False))) for name, frame in tables.items()}


class InsightSheetTests(unittest.TestCase):
    def setUp(self):
        self.tables = _from_csv(_pack_tables())

    def test_a_pack_on_disk_renders_every_section(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pack.zip"
            _write_pack(path, _pack_tables())
            tables, summary = sheet.load_pack(path)
        text = sheet.render_insight_sheet(tables, summary)
        self.assertTrue(text.startswith("# A 1–0 B — insight sheet"))
        self.assertIn("Test Cup · 2026-01-02\n", text)
        self.assertNotIn("T20:00", text)
        for heading in (
            "## Goals", "## How the match split", "## Insight candidates", "## Each team, phase by phase",
            "## Pressure", "## Droughts", "## Using the other tables", "## Definitions",
        ):
            self.assertIn(heading, text)
        self.assertIn("30' wing (A)", text)

    def test_players_are_shown_with_their_shirt_number(self):
        text = sheet.render_insight_sheet(self.tables, SUMMARY)
        self.assertIn("Players: wing (#11)", text)
        self.assertIn("| wing (#11) |", text)
        self.assertIn("| mid (#8) |", text)

    def test_substitutes_get_a_number_too_and_unknown_names_stay_plain(self):
        phases = self.tables["match-phases"].astype({"home_subs_on": object})
        phases.loc[phases.index[-1], "home_subs_on"] = "wing; ghost"
        self.tables["match-phases"] = phases
        text = sheet.render_insight_sheet(self.tables, SUMMARY)
        self.assertIn("A: wing (#11), ghost", text)

    def test_candidates_carry_the_check_and_a_numbered_caveat_listed_once(self):
        text = sheet.render_insight_sheet(self.tables, SUMMARY)
        self.assertIn("Settle before writing:", text)
        listed = text.split("Caveats:", 1)[1].split("## ", 1)[0]
        numbered = [line for line in listed.splitlines() if line[:1].isdigit()]
        self.assertEqual(len(numbered), self.tables["insight-candidates"]["caveat"].nunique())
        self.assertEqual(text.count("\nCaveat ["), len(self.tables["insight-candidates"]))

    def test_one_team_only(self):
        text = sheet.render_insight_sheet(self.tables, SUMMARY, team="B")
        self.assertIn("\n### B\n", text)
        self.assertNotIn("\n### A\n", text)
        self.assertNotIn("Players: wing", text)

    def test_unknown_team_is_refused(self):
        with self.assertRaises(ValueError):
            sheet.render_insight_sheet(self.tables, SUMMARY, team="C")

    def test_players_with_few_touches_are_left_out_and_counted(self):
        pressure = self.tables["pressure-profile"]
        extra = pressure[(pressure["team_name"] == "A") & (pressure["group_type"] == "player")].iloc[:1].copy()
        extra["group"], extra["touches"], extra["pressed"], extra["low_sample"] = "tiny", 5, 3, True
        self.tables["pressure-profile"] = pd.concat([pressure, extra], ignore_index=True)
        text = sheet.render_insight_sheet(self.tables, SUMMARY)
        self.assertNotIn("tiny", text)
        self.assertIn("1 player(s) with fewer than 20 touches are left out.", text)

    def test_text_booleans_from_a_csv_are_read_as_booleans(self):
        self.assertFalse(sheet._flag("False"))
        self.assertTrue(sheet._flag("True"))
        self.assertFalse(sheet._flag(np.nan))
        self.assertTrue(sheet._flag(True))
        phases = self.tables["match-phases"].astype({"short": object})
        phases["short"] = ["False"] * len(phases)
        self.tables["match-phases"] = phases
        self.assertNotIn("(short)", sheet.render_insight_sheet(self.tables, SUMMARY))

    def test_without_insights_it_says_so(self):
        self.tables["insight-candidates"] = self.tables["insight-candidates"].iloc[0:0]
        self.assertIn("The rules found nothing worth flagging.", sheet.render_insight_sheet(self.tables, SUMMARY))

    def test_a_pack_without_the_window_tables_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.zip"
            _write_pack(path, {"phase-sequences": pd.DataFrame({"a": [1]})})
            with self.assertRaises(sheet.PackTooOldError):
                sheet.load_pack(path)
        with self.assertRaises(sheet.PackTooOldError):
            sheet.render_insight_sheet({}, SUMMARY)


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("pack_insights_script", ROOT / "scripts" / "pack_insights.py")
        cls.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.script)

    def _run(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self.script.main([str(arg) for arg in args])

    def test_writes_the_sheet_to_a_file(self):
        with tempfile.TemporaryDirectory() as folder:
            pack, out = Path(folder) / "pack.zip", Path(folder) / "sheet.md"
            _write_pack(pack, _pack_tables())
            self.assertEqual(self._run(pack, "--out", out), 0)
            self.assertTrue(out.read_text(encoding="utf-8").startswith("# A 1–0 B"))

    def test_an_old_pack_exits_with_2(self):
        with tempfile.TemporaryDirectory() as folder:
            pack = Path(folder) / "old.zip"
            _write_pack(pack, {"phase-sequences": pd.DataFrame({"a": [1]})})
            self.assertEqual(self._run(pack), 2)

    def test_an_unknown_team_exits_with_1(self):
        with tempfile.TemporaryDirectory() as folder:
            pack = Path(folder) / "pack.zip"
            _write_pack(pack, _pack_tables())
            self.assertEqual(self._run(pack, "--team", "C"), 1)


if __name__ == "__main__":
    unittest.main()
