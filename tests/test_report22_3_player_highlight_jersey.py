from __future__ import annotations

import unittest
from unittest.mock import patch

import plotly.graph_objects as go

from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import (
    RendererRegistry,
    build_report_figure_catalog,
)
from tests.test_report11_real_shaped_fixture import _processed_fixture


class Report223PlayerHighlightJerseyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, match_info = _processed_fixture()
        cls.bundle = build_match_report_data_bundle(
            frame,
            match_info,
        )

    def test_selected_player_highlights_receive_numeric_jerseys(self):
        calls = []

        def resolve(_self, renderer_id):
            def renderer(*args, **kwargs):
                if renderer_id in {
                    "player-pass-map",
                    "player-reception-map",
                    "player-defensive-map",
                }:
                    calls.append((renderer_id, args, kwargs))
                return go.Figure()
            return renderer

        with patch.object(
            RendererRegistry,
            "resolve",
            new=resolve,
        ):
            catalog = build_report_figure_catalog(self.bundle)

        artifacts = [
            artifact
            for artifact in catalog.figures
            if artifact.section_id == "player-highlights"
        ]
        self.assertEqual(len(artifacts), 6)
        self.assertEqual(len(calls), 6)

        for renderer_id, _args, kwargs in calls:
            jersey = (
                kwargs.get("player_jersey")
                if renderer_id == "player-pass-map"
                else kwargs.get("jersey")
            )
            jersey = str(jersey or "").strip()
            self.assertNotEqual(
                jersey,
                "?",
                msg=f"{renderer_id} received '?'",
            )
            self.assertRegex(
                jersey,
                r"^\d+$",
                msg=f"{renderer_id} received jersey={jersey!r}",
            )


if __name__ == "__main__":
    unittest.main()
