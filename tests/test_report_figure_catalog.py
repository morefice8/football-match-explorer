from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
from src.reporting.figure_export import (
    FigureExportStatus,
    export_report_figure_catalog,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportScope
from src.reporting.renderer_registry import (
    RendererRegistry,
    VALIDATED_RENDERERS,
)


ROOT = Path(__file__).resolve().parents[1]


def _bundle(
    *,
    section_error: str | None = None,
) -> MatchReportDataBundle:
    sections = []
    for spec in REPORT_MANIFEST.sections:
        if spec.id == section_error:
            sections.append(
                ReportSectionBundle(
                    id=spec.id,
                    status=ReportSectionStatus.ERROR,
                    error_type="RuntimeError",
                    error_message="synthetic section failure",
                )
            )
        else:
            sections.append(
                ReportSectionBundle(
                    id=spec.id,
                    status=ReportSectionStatus.EMPTY,
                    data={},
                )
            )

    return MatchReportDataBundle(
        manifest_id=REPORT_MANIFEST.id,
        manifest_version=REPORT_MANIFEST.schema_version,
        source_signature="report-04-test",
        scope=ReportScope.FULL_MATCH,
        teams=("Home FC", "Away FC"),
        match_info={
            "hteamName": "Home FC",
            "ateamName": "Away FC",
        },
        sections=tuple(sections),
    )


class ReportFigureCatalogTests(unittest.TestCase):
    def test_every_manifest_figure_has_a_catalog_plan(self):
        expected = {
            figure.id
            for section in REPORT_MANIFEST.sections
            for figure in section.figures
        }
        actual = {
            plan.figure_id
            for plan in build_figure_plans(_bundle())
        }
        self.assertEqual(actual, expected)

    def test_starting_and_final_formation_are_planned_for_both_teams(self):
        plans = [
            plan
            for plan in build_figure_plans(_bundle())
            if plan.figure_id == "formation-timeline-figure"
        ]
        self.assertEqual(
            {(plan.team_name, plan.variant) for plan in plans},
            {
                ("Home FC", "starting"),
                ("Home FC", "final"),
                ("Away FC", "starting"),
                ("Away FC", "final"),
            },
        )

    def test_ppda_has_timeline_and_full_half_summary(self):
        plans = [
            plan
            for plan in build_figure_plans(_bundle())
            if plan.figure_id == "ppda-figure"
        ]
        self.assertEqual(
            {plan.variant for plan in plans},
            {"timeline", "summary"},
        )

    def test_home_away_orientation_is_stable(self):
        plans = [
            plan
            for plan in build_figure_plans(_bundle())
            if plan.team_name is not None
        ]
        self.assertTrue(plans)

        for plan in plans:
            if plan.team_name == "Home FC":
                self.assertFalse(plan.is_away)
            else:
                self.assertTrue(plan.is_away)

    def test_empty_samples_create_plotly_placeholders(self):
        catalog = build_report_figure_catalog(_bundle())
        self.assertTrue(catalog.figures)
        self.assertTrue(
            all(
                isinstance(item.figure, go.Figure)
                for item in catalog.figures
            )
        )
        self.assertTrue(
            any(
                item.status is ReportFigureStatus.EMPTY
                for item in catalog.figures
            )
        )

    def test_empty_mean_positions_fixture_does_not_require_team_column(self):
        catalog = build_report_figure_catalog(_bundle())
        items = [
            item
            for item in catalog.figures
            if item.section_id == "mean-positions"
        ]
        self.assertTrue(items)
        self.assertTrue(
            all(
                item.status is ReportFigureStatus.EMPTY
                for item in items
            )
        )

    def test_source_section_error_propagates_only_to_its_figures(self):
        catalog = build_report_figure_catalog(
            _bundle(section_error="pass-network")
        )
        failed = [
            item
            for item in catalog.figures
            if item.section_id == "pass-network"
        ]
        others = [
            item
            for item in catalog.figures
            if item.section_id != "pass-network"
        ]

        self.assertTrue(failed)
        self.assertTrue(
            all(
                item.status is ReportFigureStatus.ERROR
                for item in failed
            )
        )
        self.assertTrue(
            any(
                item.status is not ReportFigureStatus.ERROR
                for item in others
            )
        )

    def test_renderer_exception_is_isolated_to_one_figure(self):
        from src.reporting import figure_catalog as module

        real_render = module._render_plan
        calls = {"count": 0}

        def flaky(bundle, plan, registry, config):
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("isolated figure failure")
            return real_render(bundle, plan, registry, config)

        with patch.object(
            module,
            "_render_plan",
            side_effect=flaky,
        ):
            catalog = build_report_figure_catalog(_bundle())

        errors = [
            item
            for item in catalog.figures
            if item.status is ReportFigureStatus.ERROR
        ]
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "isolated figure failure",
            errors[0].error_message,
        )

    def test_filenames_are_unique_and_deterministic(self):
        first = build_report_figure_catalog(_bundle())
        second = build_report_figure_catalog(_bundle())

        first_names = [item.filename for item in first.figures]
        second_names = [item.filename for item in second.figures]

        self.assertEqual(first_names, second_names)
        self.assertEqual(
            len(first_names),
            len(set(first_names)),
        )
        self.assertTrue(
            all(name.endswith(".png") for name in first_names)
        )

    def test_dimensions_follow_manifest_specs(self):
        catalog = build_report_figure_catalog(_bundle())
        dimensions = {
            figure.id: (
                figure.export.width_px,
                figure.export.height_px,
            )
            for section in REPORT_MANIFEST.sections
            for figure in section.figures
        }

        for item in catalog.figures:
            self.assertEqual(
                (item.width_px, item.height_px),
                dimensions[item.id],
            )
            self.assertEqual(
                item.figure.layout.width,
                item.width_px,
            )
            self.assertEqual(
                item.figure.layout.height,
                item.height_px,
            )

    def test_validated_renderer_registry_matches_inventory(self):
        self.assertEqual(
            set(VALIDATED_RENDERERS),
            {
                "formation-state",
                "mean-positions",
                "pass-network",
                "progressive-passes",
                "final-third-entries",
                "pass-locations",
                "cross-heatmap",
                "build-up-sequence",
                "defensive-shape",
                "ppda-timeline",
                "sequence-explorer",
                "restart-map",
                "player-pass-map",
                "player-reception-map",
                "player-defensive-map",
            },
        )

    def test_all_registered_renderers_resolve(self):
        resolved = RendererRegistry().validate()
        self.assertEqual(
            set(resolved),
            set(VALIDATED_RENDERERS),
        )

    def test_static_reporting_modules_do_not_import_app_dash_components(self):
        forbidden = []

        for relative in (
            "src/reporting/renderer_registry.py",
            "src/reporting/figure_catalog.py",
            "src/reporting/figure_export.py",
        ):
            path = ROOT / relative
            tree = ast.parse(path.read_text(encoding="utf-8"))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue

                for module in modules:
                    if (
                        module == "app"
                        or module.startswith("app.")
                        or module == "dash"
                        or module.startswith("dash.")
                        or module == "dash_bootstrap_components"
                        or module.startswith("dash_bootstrap_components.")
                        or module == "src.components"
                        or module.startswith("src.components.")
                    ):
                        forbidden.append((relative, module))

        self.assertEqual(forbidden, [])

    def test_export_prefers_batch_write_images(self):
        catalog = build_report_figure_catalog(_bundle())

        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "src.reporting.figure_export._static_export_diagnostic",
                return_value=(True, None),
            ):
                with patch(
                    "src.reporting.figure_export.pio.write_images"
                ) as batch:
                    result = export_report_figure_catalog(
                        catalog,
                        output_dir,
                        batch=True,
                    )

        self.assertTrue(batch.called)
        self.assertTrue(
            all(
                item.status is FigureExportStatus.EXPORTED
                for item in result.items
            )
        )

    def test_batch_failure_falls_back_to_isolated_writes(self):
        catalog = build_report_figure_catalog(_bundle())

        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "src.reporting.figure_export._static_export_diagnostic",
                return_value=(True, None),
            ):
                with patch(
                    "src.reporting.figure_export.pio.write_images",
                    side_effect=RuntimeError("batch failed"),
                ):
                    with patch(
                        "src.reporting.figure_export.pio.write_image"
                    ) as single:
                        result = export_report_figure_catalog(
                            catalog,
                            output_dir,
                            batch=True,
                        )

        self.assertTrue(single.called)
        self.assertTrue(
            all(
                item.status is FigureExportStatus.EXPORTED
                for item in result.items
            )
        )

    def test_missing_kaleido_returns_clear_error_in_safe_temp_dir(self):
        catalog = build_report_figure_catalog(_bundle())

        with patch(
            "src.reporting.figure_export._static_export_diagnostic",
            return_value=(
                False,
                "Static report export requires Kaleido.",
            ),
        ):
            result = export_report_figure_catalog(catalog)

        output_dir = Path(result.output_dir)
        try:
            self.assertTrue(result.temporary)
            self.assertTrue(output_dir.exists())
            self.assertTrue(
                all(
                    item.status is FigureExportStatus.ERROR
                    for item in result.items
                )
            )
            self.assertTrue(
                all(
                    "Kaleido" in (item.error_message or "")
                    for item in result.items
                )
            )
        finally:
            result.cleanup()

        self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
