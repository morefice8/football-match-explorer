from __future__ import annotations

from dataclasses import replace
import unittest

import pandas as pd
import plotly.graph_objects as go

from src.reporting.bundle import (
    MatchReportDataBundle,
    ReportSectionBundle,
    ReportSectionStatus,
)
from src.reporting.figure_catalog import (
    ReportFigureStatus,
    build_figure_plans,
    build_report_figure_catalog,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportManifest, ReportScope
from src.reporting.renderer_registry import VALIDATED_RENDERERS, RendererRegistry


def _shots_frame():
    return pd.DataFrame(
        [
            {
                "team_name": "Home",
                "x": 92.0,
                "y": 48.0,
                "shot_outcome": "goal",
                "shot_counts_as_shot": True,
                "playerName": "H9",
                "timeMin": 23,
                "Goal mouth y co-ordinate": 51.0,
                "Goal mouth z co-ordinate": 40.0,
            },
            {
                "team_name": "Away",
                "x": 90.0,
                "y": 52.0,
                "shot_outcome": "saved",
                "shot_counts_as_shot": True,
                "playerName": "A7",
                "timeMin": 61,
                "Goal mouth y co-ordinate": 49.0,
                "Goal mouth z co-ordinate": 55.0,
            },
        ]
    )


def _bundle(*, empty=False):
    data = {"shots": pd.DataFrame() if empty else _shots_frame()}
    section = ReportSectionBundle(
        id="shooting",
        status=(
            ReportSectionStatus.EMPTY
            if empty
            else ReportSectionStatus.GENERATED
        ),
        data=data,
    )
    return MatchReportDataBundle(
        manifest_id=REPORT_MANIFEST.id,
        manifest_version=REPORT_MANIFEST.schema_version,
        source_signature="report23-shooting-test",
        scope=ReportScope.FULL_MATCH,
        teams=("Home", "Away"),
        match_info={"hteamName": "Home", "ateamName": "Away"},
        sections=(section,),
    )


def _shooting_manifest():
    source = next(s for s in REPORT_MANIFEST.sections if s.id == "shooting")
    section = replace(source, order=1)
    return ReportManifest(
        id="report23-shooting-only",
        schema_version="1.0",
        sections=(section,),
    )


class _RecordingRegistry:
    def __init__(self):
        self.calls = []

    def resolve(self, renderer_id):
        def render(shots_df, *, home_team, away_team, hcol, acol):
            self.calls.append(
                (renderer_id, home_team, away_team, hcol, acol, len(shots_df))
            )
            return go.Figure()

        return render


class Report23ShotMapPlacementTests(unittest.TestCase):
    def test_manifest_exposes_shooting_section_between_pass_locations_and_cross_flow(self):
        sections = {s.id: s for s in REPORT_MANIFEST.sections}
        self.assertIn("shooting", sections)
        shooting = sections["shooting"]
        self.assertEqual(shooting.title, "Shooting")
        self.assertEqual(
            [f.id for f in shooting.figures],
            ["shot-map-figure", "shot-placement-figure"],
        )
        self.assertEqual(
            sections["pass-locations"].order + 1,
            shooting.order,
        )
        self.assertEqual(
            shooting.order + 1,
            sections["cross-flow"].order,
        )

    def test_renderers_are_registered_and_resolve(self):
        self.assertIn("shot-map", VALIDATED_RENDERERS)
        self.assertIn("shot-placement", VALIDATED_RENDERERS)
        resolved = RendererRegistry().validate()
        self.assertIn("shot-map", resolved)
        self.assertIn("shot-placement", resolved)

    def test_catalog_plans_one_combined_figure_per_chart(self):
        plans = build_figure_plans(_bundle(), _shooting_manifest())
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertIn(plan.figure_id, ("shot-map-figure", "shot-placement-figure"))
            self.assertIsNone(plan.team_name)
            self.assertIsNone(plan.team_index)
            self.assertEqual(plan.variant, "summary")

    def test_generated_figures_pass_both_teams_and_colors_to_the_live_renderer(self):
        registry = _RecordingRegistry()
        catalog = build_report_figure_catalog(
            _bundle(),
            _shooting_manifest(),
            registry=registry,
        )

        statuses = {item.id: item.status for item in catalog.figures}
        self.assertEqual(statuses["shot-map-figure"], ReportFigureStatus.GENERATED)
        self.assertEqual(statuses["shot-placement-figure"], ReportFigureStatus.GENERATED)

        by_renderer = {call[0]: call for call in registry.calls}
        self.assertIn("shot-map", by_renderer)
        self.assertIn("shot-placement", by_renderer)
        for renderer_id, home_team, away_team, hcol, acol, shot_count in registry.calls:
            self.assertEqual(home_team, "Home")
            self.assertEqual(away_team, "Away")
            self.assertTrue(hcol)
            self.assertTrue(acol)
            self.assertEqual(shot_count, 2)

    def test_empty_shots_become_placeholders_not_errors(self):
        registry = _RecordingRegistry()
        catalog = build_report_figure_catalog(
            _bundle(empty=True),
            _shooting_manifest(),
            registry=registry,
        )
        statuses = {item.id: item.status for item in catalog.figures}
        self.assertEqual(statuses["shot-map-figure"], ReportFigureStatus.EMPTY)
        self.assertEqual(statuses["shot-placement-figure"], ReportFigureStatus.EMPTY)
        self.assertEqual(registry.calls, [])


if __name__ == "__main__":
    unittest.main()
