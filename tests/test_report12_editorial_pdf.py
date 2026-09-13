from __future__ import annotations

from io import BytesIO
import re
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from src.reporting.audit import FORBIDDEN_PDF_TEXT, searchable_pdf_text
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import RendererRegistry, build_report_figure_catalog
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.pdf_renderer import (
    MatchReportPdfConfig,
    _PDF_TABLE_COLUMNS,
    _format_table_cell,
    _formation_spells_rows,
    _paired_restart_takers_frame,
    _prepare_pdf_table_frame,
    _styles,
    _table_payloads,
    _table_story_for_spec,
    render_match_report_pdf,
)
from tests.test_report11_real_shaped_fixture import _processed_fixture


def _page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def _tiny_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 18), "white").save(output, format="PNG")
    return output.getvalue()


def _dummy_renderer(*args, **kwargs):
    return go.Figure()


def _table_spec(table_id: str):
    return next(
        table
        for section in REPORT_MANIFEST.sections
        for table in section.tables
        if table.id == table_id
    )


def _is_nested(value) -> bool:
    return isinstance(value, (pd.DataFrame, pd.Series, dict, list, tuple, set))


class Report12EditorialPdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.match_info = _processed_fixture()
        cls.bundle = build_match_report_data_bundle(cls.frame, cls.match_info)

    def test_manifest_selection_limits_match_editorial_contract(self):
        expected = {
            "cross-top-routes": 8,
            "build-up-sequences": 10,
            "defensive-transitions-sequences": 10,
            "offensive-transitions-sequences": 10,
            "restart-takers": 6,
            "player-highlights-table": 3,
        }
        for table_id, limit in expected.items():
            with self.subTest(table=table_id):
                self.assertEqual(_table_spec(table_id).selection.limit, limit)

        top_sequence_figures = [
            figure
            for section in REPORT_MANIFEST.sections
            for figure in section.figures
            if figure.selection.rule == "top-sequence-by-outcome-priority"
        ]
        top_player_figures = [
            figure
            for section in REPORT_MANIFEST.sections
            for figure in section.figures
            if figure.selection.rule == "top-player-by-metric-family"
        ]
        self.assertTrue(top_sequence_figures)
        self.assertTrue(top_player_figures)
        self.assertTrue(all(figure.selection.limit == 1 for figure in top_sequence_figures))
        self.assertTrue(all(figure.selection.limit == 1 for figure in top_player_figures))

    def test_selected_pdf_tables_respect_manifest_limits(self):
        for table_id in (
            "cross-top-routes",
            "build-up-sequences",
            "defensive-transitions-sequences",
            "offensive-transitions-sequences",
            "restart-takers",
        ):
            spec = _table_spec(table_id)
            payloads = _table_payloads(
                self.bundle,
                table_id,
                selection_limit=spec.selection.limit,
            )
            for title, frame in payloads:
                with self.subTest(table=table_id, payload=title):
                    self.assertLessEqual(len(frame), spec.selection.limit)

    def test_player_table_is_at_most_three_per_team_and_family(self):
        spec = _table_spec("player-highlights-table")
        payloads = _table_payloads(
            self.bundle,
            spec.id,
            selection_limit=spec.selection.limit,
        )
        self.assertTrue(payloads)
        for title, frame in payloads:
            team_column = next(
                (name for name in ("team_name", "teamName", "Team") if name in frame.columns),
                None,
            )
            self.assertIsNotNone(team_column, title)
            counts = frame.groupby(team_column, dropna=False).size()
            for team, count in counts.items():
                with self.subTest(family=title, team=team):
                    self.assertLessEqual(int(count), 3)

    def test_pdf_table_contract_is_explicit_scalar_and_a4_readable(self):
        for section in REPORT_MANIFEST.sections:
            for table in section.tables:
                payloads = _table_payloads(
                    self.bundle,
                    table.id,
                    selection_limit=table.selection.limit,
                )
                for title, frame in payloads:
                    prepared = _prepare_pdf_table_frame(table.id, frame)
                    with self.subTest(table=table.id, payload=title):
                        self.assertLessEqual(len(prepared.columns), 8)
                        if table.id in _PDF_TABLE_COLUMNS:
                            # The policy is alias based; after preparation every
                            # surviving column was explicitly requested by it.
                            self.assertTrue(set(prepared.columns).issubset(set(frame.columns)))
                        for value in prepared.to_numpy().ravel().tolist():
                            self.assertFalse(_is_nested(value), repr(value)[:160])

    def test_formation_spells_are_flattened_into_meaningful_editorial_fields(self):
        moments = [
            {
                "time_label": "67'",
                "score": "1 - 0",
                "home_formation_name": "4-3-3",
                "away_formation_name": "3-5-2",
                "events": [
                    {"kind": "substitution", "team": "Napoli"},
                    {"kind": "formation_change", "team": "Napoli"},
                ],
                "home_state": {"nested": "must not leak"},
            }
        ]
        frame = _formation_spells_rows(moments)
        prepared = _prepare_pdf_table_frame("formation-spells", frame)

        self.assertEqual(
            list(prepared.columns),
            ["time", "score", "home_formation", "away_formation", "change"],
        )
        self.assertEqual(prepared.iloc[0]["home_formation"], "4-3-3")
        self.assertIn("Substitution", prepared.iloc[0]["change"])
        self.assertIn("Formation change", prepared.iloc[0]["change"])
        self.assertNotIn("home_state", prepared.columns)

    def test_restart_takers_render_as_one_compact_comparison_table(self):
        left = pd.DataFrame(
            {
                "player_name": ["A", "B"],
                "restart_count": [6, 4],
                "primary_restart": ["Throw-in", "Corner"],
                "shots": [0, 1],
            }
        )
        right = pd.DataFrame(
            {
                "player_name": ["C", "D"],
                "restart_count": [5, 3],
                "primary_restart": ["Goal Kick", "Free Kick"],
                "shots": [0, 0],
            }
        )
        comparison = _paired_restart_takers_frame(
            [("Restart takers - Napoli", left), ("Restart takers - Udinese", right)]
        )
        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison), 2)
        self.assertEqual(len(comparison.columns), 8)
        self.assertIn("Napoli player", comparison.columns)
        self.assertIn("Udinese player", comparison.columns)

    def test_unsupported_latin_player_names_do_not_render_as_black_squares(self):
        self.assertEqual(_format_table_cell("B. Mlačić"), "B. Mlacic")

    def test_typographic_match_clock_marks_render_as_ascii(self):
        self.assertEqual(_format_table_cell("31′ 05″"), "31' 05\"")

    def test_real_shaped_fixture_renders_in_editorial_page_budget(self):
        png = _tiny_png()
        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            catalog = build_report_figure_catalog(self.bundle)
            pdf = render_match_report_pdf(self.bundle, catalog)

        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreaterEqual(_page_count(pdf), 24)
        self.assertLessEqual(_page_count(pdf), 32)

        text = searchable_pdf_text(pdf).casefold()
        self.assertNotIn("selection reason:", text)
        self.assertNotIn("np.int64", text)
        for phrase in FORBIDDEN_PDF_TEXT:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase.casefold(), text)


if __name__ == "__main__":
    unittest.main()
