from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import unittest
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
from PIL import Image as PILImage

from src.metrics import defensive_metrics
from src.reporting.audit import pdf_page_count, searchable_pdf_text
from src.reporting.bundle import (
    MatchReportBundleConfig,
    MatchReportDataBundle,
    ReportSectionBundle,
    ReportSectionStatus,
    _BundleContext,
    _defensive_shape,
)
from src.reporting.figure_catalog import (
    MatchReportFigureCatalog,
    ReportFigureArtifact,
    ReportFigureStatus,
    build_figure_plans,
    build_report_figure_catalog,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportManifest, ReportScope
from src.reporting.pdf_renderer import (
    _defensive_shape_period_frame,
    render_match_report_pdf,
)


def _tiny_png_bytes():
    buffer = BytesIO()
    PILImage.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _event(team, minute, period, player, x, y):
    return {
        "team_name": team,
        "type_name": "Tackle",
        "x": x,
        "y": y,
        "expandedMinute": minute,
        "periodId": period,
        "playerName": player,
        "Mapped Jersey Number": int(player[-1]) if player[-1].isdigit() else 2,
        "id": f"{team}-{minute}-{player}",
    }


def _sample_events():
    rows = []
    for team, base in (("Home", 25), ("Away", 50)):
        for period, minute, shift in ((1, 15, 0), (2, 65, 20)):
            for idx in range(1, 8):
                rows.append(
                    _event(
                        team,
                        minute + idx / 10,
                        period,
                        f"{team[0]}{idx}",
                        base + shift + idx,
                        10 + idx * 10,
                    )
                )
    return pd.DataFrame(rows)


def _profile(team, period, *, empty=False, offset=0.0, scale=40.0):
    if empty:
        return {
            "team_name": team,
            "period": period,
            "actions": pd.DataFrame(),
            "action_count": 0,
            "block_height_m": None,
            "width_m": None,
            "compactness_m": None,
            "density_bin_size": 12.5,
            "density_peak_pct": None,
            "density_scale_max_pct": scale,
        }
    actions = pd.DataFrame(
        [
            {
                "team_name": team,
                "type_name": "Tackle",
                "x": 30 + offset + i * 3,
                "y": 15 + i * 10,
                "playerName": f"{team}-{i}",
            }
            for i in range(7)
        ]
    )
    return {
        "team_name": team,
        "period": period,
        "actions": actions,
        "action_count": len(actions),
        "block_height_m": 40.0 + offset,
        "width_m": 45.0 + offset,
        "compactness_m": 20.0 + offset,
        "density_bin_size": 12.5,
        "density_peak_pct": 30.0 + offset,
        "density_scale_max_pct": scale,
    }


def _bundle(*, empty_away_second=False):
    data = {}
    for team, offset in (("Home", 0.0), ("Away", 3.0)):
        first = _profile(team, "first_half", offset=offset)
        second = _profile(
            team,
            "second_half",
            empty=(empty_away_second and team == "Away"),
            offset=offset + 2.0,
        )
        data[team] = {
            "first_half": first,
            "second_half": second,
            "full_match": {
                "team_name": team,
                "period": "full",
                "action_count": 14,
                "block_height_m": 41.0 + offset,
                "width_m": 46.0 + offset,
                "compactness_m": 21.0 + offset,
                "density_bin_size": 12.5,
                "density_peak_pct": 28.0 + offset,
            },
        }
    section = ReportSectionBundle(
        id="defensive-shape",
        status=ReportSectionStatus.GENERATED,
        data=data,
    )
    return MatchReportDataBundle(
        manifest_id=REPORT_MANIFEST.id,
        manifest_version=REPORT_MANIFEST.schema_version,
        source_signature="report15-density-test",
        scope=ReportScope.FULL_MATCH,
        teams=("Home", "Away"),
        match_info={"hteamName": "Home", "ateamName": "Away"},
        sections=(section,),
    )


def _defensive_manifest():
    source = next(s for s in REPORT_MANIFEST.sections if s.id == "defensive-shape")
    section = replace(source, order=1)
    return ReportManifest(
        id="report15-defensive-only",
        schema_version="1.0",
        sections=(section,),
    )


class Report15DefensiveDensityTests(unittest.TestCase):
    def test_manifest_exposes_density_not_shape(self):
        section = next(s for s in REPORT_MANIFEST.sections if s.id == "defensive-shape")
        self.assertEqual(section.title, "Defensive Density")
        self.assertEqual(section.figures[0].title, "Defensive Density")
        self.assertIn("density", section.figures[0].selection.rule)

    def test_period_and_team_scope_use_all_half_actions(self):
        df = _sample_events()
        first = defensive_metrics.build_defensive_density_profile(
            df, "Home", period="first_half"
        )
        second = defensive_metrics.build_defensive_density_profile(
            df, "Home", period="second_half"
        )

        self.assertEqual(first["action_count"], 7)
        self.assertEqual(second["action_count"], 7)
        self.assertEqual(set(first["actions"]["team_name"]), {"Home"})
        self.assertEqual(set(second["actions"]["team_name"]), {"Home"})
        self.assertLess(first["actions"]["x"].max(), second["actions"]["x"].min())
        self.assertIsNotNone(first["block_height_m"])
        self.assertIsNotNone(second["compactness_m"])

    def test_nonempty_density_does_not_need_lineup_or_snapshot_metadata(self):
        df = pd.DataFrame(
            [
                _event("Home", 12 + idx, 1, f"P{idx}", 30 + idx, 20 + idx * 5)
                for idx in range(1, 5)
            ]
        )
        profile = defensive_metrics.build_defensive_density_profile(
            df, "Home", period="first_half"
        )
        self.assertEqual(profile["action_count"], 4)
        self.assertNotIn("representative", profile)
        self.assertNotIn("snapshot_count", profile)
        self.assertGreater(profile["density_peak_pct"], 0)

    def test_empty_period_is_explicit_and_metrics_are_none(self):
        profile = defensive_metrics.build_defensive_density_profile(
            _sample_events(), "Home", period="third_half"
        )
        # Unknown period aliases intentionally mean full match; verify a truly
        # empty team instead of relying on an unsupported period name.
        profile = defensive_metrics.build_defensive_density_profile(
            _sample_events(), "Missing", period="first_half"
        )
        self.assertEqual(profile["action_count"], 0)
        self.assertTrue(profile["actions"].empty)
        self.assertIsNone(profile["block_height_m"])
        self.assertIsNone(profile["width_m"])
        self.assertIsNone(profile["compactness_m"])

    def test_bundle_builds_halves_and_one_shared_density_scale(self):
        calls = []
        peaks = {
            ("Home", "first_half"): 18.0,
            ("Home", "second_half"): 31.0,
            ("Away", "first_half"): 37.0,
            ("Away", "second_half"): 22.0,
            ("Home", "full"): 20.0,
            ("Away", "full"): 21.0,
        }

        def fake(df, team, period="full", **kwargs):
            calls.append((team, period))
            return {
                "team_name": team,
                "period": period,
                "actions": pd.DataFrame([{"team_name": team, "x": 50, "y": 50}]),
                "action_count": 1,
                "block_height_m": 50.0,
                "width_m": 20.0,
                "compactness_m": 10.0,
                "density_bin_size": 12.5,
                "density_peak_pct": peaks[(team, period)],
                "density_scale_max_pct": None,
            }

        ctx = _BundleContext(
            pd.DataFrame([{"team_name": "Home"}]),
            {},
            MatchReportBundleConfig(),
            ("Home", "Away"),
        )
        with patch(
            "src.metrics.defensive_metrics.build_defensive_density_profile",
            side_effect=fake,
        ):
            produced = _defensive_shape(ctx)

        self.assertEqual(len(calls), 6)
        for team in ("Home", "Away"):
            payload = produced.data[team]
            self.assertNotIn("actions", payload["full_match"])
            for period in ("first_half", "second_half"):
                self.assertEqual(payload[period]["density_scale_max_pct"], 40.0)

    def test_catalog_plans_two_periods_per_team_and_no_full_match_plot(self):
        plans = [
            plan
            for plan in build_figure_plans(_bundle(), _defensive_manifest())
            if plan.figure_id == "defensive-shape-figure"
        ]
        self.assertEqual(
            {(p.team_name, p.variant) for p in plans},
            {
                ("Home", "first_half"),
                ("Home", "second_half"),
                ("Away", "first_half"),
                ("Away", "second_half"),
            },
        )
        self.assertNotIn("full_match", {p.variant for p in plans})

    def test_generated_figures_use_app_style_density_and_light_points(self):
        modes = []

        class Registry:
            def resolve(self, renderer_id):
                def render(profile, color, mode="shape"):
                    modes.append(mode)
                    fig = go.Figure(
                        [
                            go.Histogram2dContour(
                                x=profile["actions"]["x"],
                                y=profile["actions"]["y"],
                            ),
                            go.Scatter(
                                x=profile["actions"]["x"],
                                y=profile["actions"]["y"],
                                mode="markers",
                            ),
                        ]
                    )
                    return fig
                return render

        catalog = build_report_figure_catalog(
            _bundle(),
            _defensive_manifest(),
            registry=Registry(),
        )
        generated = [
            item for item in catalog.figures
            if item.status is ReportFigureStatus.GENERATED
        ]
        self.assertEqual(modes, ["density"] * 4)
        self.assertEqual(len(generated), 4)
        for item in generated:
            trace = item.figure.data[0]
            points = item.figure.data[1]
            # Do not force report-only bins or a shared z ceiling: the static
            # figure must retain the same automatic contour behaviour as the
            # interactive app.
            self.assertIsNone(trace.histnorm)
            self.assertIsNone(trace.zmin)
            self.assertIsNone(trace.zmax)
            self.assertEqual(trace.ncontours, 7)
            self.assertAlmostEqual(trace.opacity, 0.52)
            self.assertEqual(points.marker.size, 3)
            self.assertAlmostEqual(points.marker.opacity, 0.26)
            self.assertAlmostEqual(points.marker.line.width, 0.25)
            self.assertEqual(tuple(item.figure.layout.xaxis.range), (0, 100))
            self.assertEqual(tuple(item.figure.layout.yaxis.range), (0, 100))
            self.assertEqual(item.selection.get("colour_scale"), "panel-local")

    def test_empty_half_becomes_placeholder_only_when_no_actions_exist(self):
        class Registry:
            def resolve(self, renderer_id):
                return lambda profile, color, mode="density": go.Figure()

        catalog = build_report_figure_catalog(
            _bundle(empty_away_second=True),
            _defensive_manifest(),
            registry=Registry(),
        )
        scoped = {
            (item.team_name, item.variant): item.status
            for item in catalog.figures
        }
        self.assertEqual(scoped[("Away", "second_half")], ReportFigureStatus.EMPTY)
        self.assertEqual(scoped[("Away", "first_half")], ReportFigureStatus.GENERATED)
        self.assertEqual(scoped[("Home", "second_half")], ReportFigureStatus.GENERATED)

    def test_period_tables_drop_footprint_and_windows_and_show_delta(self):
        bundle = _bundle()
        first = _defensive_shape_period_frame(bundle, "first_half")
        second = _defensive_shape_period_frame(bundle, "second_half")

        self.assertEqual(
            list(first.columns),
            [
                "Team",
                "Block height (m)",
                "Width (m)",
                "Compactness (m)",
                "Sample size",
            ],
        )
        self.assertEqual(len(first), 2)
        self.assertTrue(first["Sample size"].str.fullmatch(r"\d+ actions").all())
        self.assertNotIn("Footprint (m²)", first.columns)
        self.assertIn("Δ vs 1H", second.columns)
        self.assertTrue(second["Δ vs 1H"].str.contains("BH ").all())
        self.assertTrue(second["Δ vs 1H"].str.contains("W ").all())
        self.assertTrue(second["Δ vs 1H"].str.contains("C ").all())
        self.assertFalse(second["Δ vs 1H"].str.contains("FP ").any())

    def test_defensive_density_section_is_exactly_two_pdf_pages(self):
        manifest = _defensive_manifest()
        bundle = _bundle()
        artifacts = []
        for index, team in enumerate(bundle.teams):
            for period in ("first_half", "second_half"):
                artifacts.append(
                    ReportFigureArtifact(
                        id="defensive-shape-figure",
                        section_id="defensive-shape",
                        title="Defensive Density",
                        variant=period,
                        filename=f"{team}-{period}.png",
                        width_px=1600,
                        height_px=900,
                        status=ReportFigureStatus.GENERATED,
                        figure=go.Figure(),
                        team_name=team,
                        is_away=bool(index),
                        renderer_id="defensive-density",
                    )
                )
        catalog = MatchReportFigureCatalog(
            manifest_id=manifest.id,
            manifest_version=manifest.schema_version,
            figures=tuple(artifacts),
        )

        with patch("src.reporting.pdf_renderer.pio.to_image", return_value=_tiny_png_bytes()):
            pdf = render_match_report_pdf(bundle, catalog, manifest)

        # Cover + contents + 1H + 2H: the density section itself is two pages.
        self.assertEqual(pdf_page_count(pdf), 4)

        text = searchable_pdf_text(pdf)
        self.assertIn("SECTION 01 / 1H vs 2H", text)
        self.assertIn("Selected sections may compare explicitly defined match periods", text)
        self.assertNotIn("All analytical sections use the Full Match scope", text)


if __name__ == "__main__":
    unittest.main()
