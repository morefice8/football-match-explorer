from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from src.reporting.audit import searchable_pdf_text
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import RendererRegistry, build_report_figure_catalog
from src.reporting.pdf_renderer import (
    _player_highlight_payloads,
    _table_payloads,
    render_match_report_pdf,
)
from tests.test_report11_real_shaped_fixture import _processed_fixture


def _tiny_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 18), "white").save(output, format="PNG")
    return output.getvalue()


def _dummy_renderer(*args, **kwargs):
    return go.Figure()


class _FakeBundle:
    def __init__(self, data, teams=("Home FC", "Away FC")):
        self.teams = tuple(teams)
        self._section = SimpleNamespace(data=data)

    def section(self, section_id: str):
        if section_id != "player-highlights":
            raise KeyError(section_id)
        return self._section


class Report13PlayerHighlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, match_info = _processed_fixture()
        cls.bundle = build_match_report_data_bundle(frame, match_info)

    def test_multiindex_identity_is_materialized_before_per_team_selection(self):
        players = [
            ("Home FC", "Home A", 9),
            ("Home FC", "Home B", 8),
            ("Home FC", "Home C", 7),
            ("Home FC", "Home D", 6),
            ("Away FC", "Away A", 5),
            ("Away FC", "Away B", 4),
            ("Away FC", "Away C", 3),
            ("Away FC", "Away D", 2),
        ]
        passing_index = pd.MultiIndex.from_tuples(
            [(team, player) for team, player, _ in players],
            names=["team_name", "playerName"],
        )
        passing = pd.DataFrame(
            {
                "Offensive Pass Contributions": [score for _, _, score in players],
                "Progressive Passes": [score for _, _, score in players],
                "Passes into Box": [1] * len(players),
                "Key Passes": [1] * len(players),
                "Assists": [0] * len(players),
            },
            index=passing_index,
        )

        # Reverse the MultiIndex level order to ensure identity normalisation is
        # based on names rather than index position.
        shooting_index = pd.MultiIndex.from_tuples(
            [(player, team) for team, player, _ in players],
            names=["playerName", "team_name"],
        )
        shooting = pd.DataFrame(
            {
                "Shot Sequence Involvements": [score for _, _, score in players],
                "Shot Sequence Shots": [score for _, _, score in players],
                "Shot Sequence Shot Assists": [1] * len(players),
                "Shot Sequence Pre-Assists": [0] * len(players),
            },
            index=shooting_index,
        )

        defending = pd.DataFrame(
            {
                "unique": [score for _, _, score in players],
                "tackles_won": [1] * len(players),
                "interceptions": [1] * len(players),
                "recoveries": [1] * len(players),
                "clearances": [1] * len(players),
                "blocks": [1] * len(players),
            },
            index=passing_index,
        )

        bundle = _FakeBundle(
            {
                "player_stats": passing,
                "shot_sequence_ranking": shooting,
                "defensive_ranking": defending,
                "player_options": {},
            }
        )
        payloads = dict(_player_highlight_payloads(bundle, limit=3))

        expected_names = {
            "Home FC": ["Home A", "Home B", "Home C"],
            "Away FC": ["Away A", "Away B", "Away C"],
        }
        expected_columns = {
            "Player highlights - Passing": [
                "team_name",
                "playerName",
                "Offensive Pass Contributions",
                "Progressive Passes",
                "Passes into Box",
                "Key Passes",
                "Assists",
            ],
            "Player highlights - Shooting": [
                "team_name",
                "playerName",
                "Shot Sequence Shots",
                "Shot Sequence Shot Assists",
                "Shot Sequence Pre-Assists",
                "Shot Sequence Involvements",
            ],
            "Player highlights - Defending": [
                "team_name",
                "playerName",
                "unique",
                "tackles_won",
                "interceptions",
                "recoveries",
                "clearances",
                "blocks",
            ],
        }

        self.assertEqual(set(payloads), set(expected_columns))
        for title, frame in payloads.items():
            with self.subTest(family=title):
                self.assertEqual(list(frame.columns), expected_columns[title])
                counts = frame.groupby("team_name").size().to_dict()
                self.assertEqual(counts, {"Home FC": 3, "Away FC": 3})
                for team, names in expected_names.items():
                    actual = frame.loc[frame["team_name"].eq(team), "playerName"].tolist()
                    self.assertEqual(actual, names)

    def test_family_tables_match_report13_column_contract(self):
        payloads = dict(
            _table_payloads(
                self.bundle,
                "player-highlights-table",
                selection_limit=3,
            )
        )
        self.assertEqual(
            list(payloads["Player highlights - Passing"].columns),
            [
                "team_name",
                "playerName",
                "Offensive Pass Contributions",
                "Progressive Passes",
                "Passes into Box",
                "Key Passes",
                "Assists",
            ],
        )
        self.assertEqual(
            list(payloads["Player highlights - Shooting"].columns),
            [
                "team_name",
                "playerName",
                "Shot Sequence Shots",
                "Shot Sequence Shot Assists",
                "Shot Sequence Pre-Assists",
                "Shot Sequence Involvements",
            ],
        )
        self.assertEqual(
            list(payloads["Player highlights - Defending"].columns),
            [
                "team_name",
                "playerName",
                "unique",
                "tackles_won",
                "interceptions",
                "recoveries",
                "clearances",
                "blocks",
            ],
        )

    def test_pdf_text_contains_every_selected_player_name(self):
        payloads = _table_payloads(
            self.bundle,
            "player-highlights-table",
            selection_limit=3,
        )
        selected_names = {
            str(name)
            for _, frame in payloads
            for name in frame["playerName"].dropna().tolist()
        }
        self.assertTrue(selected_names)

        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=_tiny_png(),
        ):
            catalog = build_report_figure_catalog(self.bundle)
            pdf = render_match_report_pdf(self.bundle, catalog)

        text = searchable_pdf_text(pdf)
        for player_name in sorted(selected_names):
            with self.subTest(player=player_name):
                self.assertIn(player_name, text)

    def test_player_highlight_catalog_keeps_one_plot_per_team_and_family(self):
        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer):
            catalog = build_report_figure_catalog(self.bundle)

        artifacts = [
            artifact
            for artifact in catalog.figures
            if artifact.section_id == "player-highlights"
        ]
        self.assertEqual(len(artifacts), 6)
        for figure_id in (
            "player-highlight-passing",
            "player-highlight-shooting",
            "player-highlight-defending",
        ):
            scoped = [artifact for artifact in artifacts if artifact.id == figure_id]
            with self.subTest(figure=figure_id):
                self.assertEqual(len(scoped), 2)
                self.assertEqual(
                    {artifact.team_name for artifact in scoped},
                    set(self.bundle.teams),
                )


if __name__ == "__main__":
    unittest.main()
