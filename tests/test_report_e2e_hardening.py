from __future__ import annotations

from collections.abc import Mapping
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
    build_report_figure_catalog,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportScope
from src.reporting.pdf_renderer import _cover_context


def _bundle(
    section_id: str,
    data,
    *,
    match_info: dict | None = None,
) -> MatchReportDataBundle:
    sections = []
    for spec in REPORT_MANIFEST.sections:
        if spec.id == section_id:
            sections.append(
                ReportSectionBundle(
                    id=spec.id,
                    status=ReportSectionStatus.GENERATED,
                    data=data,
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
        source_signature="report-08-real-shape-test",
        scope=ReportScope.FULL_MATCH,
        teams=("Home FC", "Away FC"),
        match_info=dict(
            match_info
            or {
                "hteamName": "Home FC",
                "ateamName": "Away FC",
            }
        ),
        sections=tuple(sections),
    )


class _RecordingRegistry:
    def __init__(self, renderer_id: str):
        self.renderer_id = renderer_id
        self.calls = []

    def resolve(self, renderer_id: str):
        if renderer_id != self.renderer_id:
            raise AssertionError(
                f"Unexpected renderer {renderer_id!r}; "
                f"expected {self.renderer_id!r}."
            )

        def renderer(*args, **kwargs):
            self.calls.append((args, kwargs))
            return go.Figure()

        return renderer


def _home_artifact(catalog, section_id: str):
    return next(
        item
        for item in catalog.figures
        if item.section_id == section_id
        and item.team_name == "Home FC"
    )


class ReportE2EHardeningTests(unittest.TestCase):
    def test_cover_uses_canonical_date_formatted_metadata(self):
        bundle = _bundle(
            "overview",
            {
                "metadata": {
                    "competitionName": "Serie A",
                    "date_formatted": "07 February 2026",
                },
                "result": {
                    "home_score": 2,
                    "away_score": 3,
                },
            },
            match_info={
                "hteamName": "Genoa",
                "ateamName": "Napoli",
                "competitionName": "Serie A",
                "home_score": 2,
                "away_score": 3,
                "date_formatted": "07 February 2026",
                "date_iso": "2026-02-07T17:01:08Z",
            },
        )

        context = _cover_context(bundle)

        self.assertEqual(context["date"], "07 February 2026")
        self.assertEqual(context["competition"], "Serie A")
        self.assertEqual(context["score"], "2 - 3")

    def test_real_shaped_flat_buildup_rows_render_selected_sequence(self):
        sequence_id = 7
        events = [
            {
                "trigger_sequence_id": sequence_id,
                "team_name": "Home FC",
                "type_name": "Pass",
                "outcome": "Successful",
                "x": 20.0,
                "y": 50.0,
                "end_x": 55.0,
                "end_y": 45.0,
                "playerName": "Player A",
                "Mapped Jersey Number": 1,
                "sequence_outcome_type": "Possession Consolidated",
                "buildup_pass_count": 2,
            },
            {
                "trigger_sequence_id": sequence_id,
                "team_name": "Home FC",
                "type_name": "Pass",
                "outcome": "Successful",
                "x": 55.0,
                "y": 45.0,
                "end_x": 86.0,
                "end_y": 40.0,
                "playerName": "Player B",
                "Mapped Jersey Number": 4,
                "sequence_outcome_type": "Possession Consolidated",
                "buildup_pass_count": 2,
            },
        ]
        data = {
            "teams": {
                "Home FC": {
                    "summary": [
                        {
                            "sequence_id": sequence_id,
                            "team_name": "Home FC",
                            "milestone_box": True,
                            "max_controlled_x": 86.0,
                            "event_count": 2,
                            "duration_seconds": 4.0,
                        }
                    ],
                    "sequences": events,
                },
                "Away FC": {
                    "summary": [],
                    "sequences": [],
                },
            }
        }
        registry = _RecordingRegistry("build-up-sequence")

        catalog = build_report_figure_catalog(
            _bundle("build-up", data),
            registry=registry,
        )

        artifact = _home_artifact(catalog, "build-up")
        self.assertIs(artifact.status, ReportFigureStatus.GENERATED)
        self.assertEqual(len(registry.calls), 1)

        sequence = registry.calls[0][0][0]
        self.assertIsInstance(sequence, pd.DataFrame)
        self.assertEqual(len(sequence), 2)
        self.assertEqual(
            sequence["trigger_sequence_id"].astype(str).unique().tolist(),
            [str(sequence_id)],
        )

    def test_real_shaped_nested_transitions_are_normalized_before_renderer(self):
        for section_id, figure_id, sequence_type in (
            (
                "defensive-transitions",
                "defensive-transitions-top-sequence",
                "defensive_transition",
            ),
            (
                "offensive-transitions",
                "offensive-transitions-top-sequence",
                "offensive_transition",
            ),
        ):
            with self.subTest(section_id=section_id):
                sequence_id = 9
                sequence_rows = [
                    {
                        "loss_sequence_id": sequence_id,
                        "team_name": "Home FC",
                        "type_name": "Ball recovery",
                        "outcome": "Successful",
                        "x": 35.0,
                        "y": 42.0,
                        "end_x": None,
                        "end_y": None,
                        "playerName": "Player A",
                        "timeMin": 52,
                        "timeSec": 1,
                        "timeMin_at_loss": 52,
                        "timeSec_at_loss": 0,
                        "sequence_outcome_type": "Shot",
                        "terminal_outcome": "shot",
                        "loss_x": 35.0,
                        "loss_y": 42.0,
                    },
                    {
                        "loss_sequence_id": sequence_id,
                        "team_name": "Home FC",
                        "type_name": "Pass",
                        "outcome": "Successful",
                        "x": 35.0,
                        "y": 42.0,
                        "end_x": 77.0,
                        "end_y": 48.0,
                        "playerName": "Player B",
                        "timeMin": 52,
                        "timeSec": 3,
                        "timeMin_at_loss": 52,
                        "timeSec_at_loss": 0,
                        "sequence_outcome_type": "Shot",
                        "terminal_outcome": "shot",
                    },
                ]
                data = {
                    "Home FC": {
                        "combined": [
                            {
                                "loss_sequence_id": sequence_id,
                                "team_name": "Home FC",
                                "sequence_outcome_type": "Shot",
                                "max_controlled_x": 77.0,
                                "event_count": 2,
                                "duration_seconds": 3.0,
                            }
                        ],
                        "sequences": [sequence_rows],
                        "stats": {},
                    },
                    "Away FC": {
                        "combined": [],
                        "sequences": [],
                        "stats": {},
                    },
                }
                registry = _RecordingRegistry("sequence-explorer")

                catalog = build_report_figure_catalog(
                    _bundle(section_id, data),
                    registry=registry,
                )

                artifact = next(
                    item
                    for item in catalog.figures
                    if item.id == figure_id
                    and item.team_name == "Home FC"
                )
                self.assertIs(
                    artifact.status,
                    ReportFigureStatus.GENERATED,
                )
                self.assertEqual(len(registry.calls), 1)

                normalized = registry.calls[0][0][0]
                self.assertIsInstance(normalized, Mapping)
                self.assertNotIsInstance(normalized, pd.DataFrame)
                self.assertEqual(
                    normalized["sequence_type"],
                    sequence_type,
                )
                self.assertTrue(normalized["events"])

    def test_restart_renderer_receives_records_not_dataframe(self):
        data = {
            "Home FC": {
                "records": [
                    {
                        "sequence_id": "restart-26-1",
                        "restart_type": "Throw-in",
                        "side": "Left",
                        "delivery": "Medium Throw",
                        "destination": "Final Third",
                        "outcome": "Successful Delivery",
                        "development_outcome": "Possession Retained",
                        "player_name": "Player A",
                        "jersey_number": 3,
                        "start_x": 50.2,
                        "start_y": 100.0,
                        "end_x": 69.8,
                        "end_y": 93.3,
                        "match_second": 280.0,
                    }
                ]
            },
            "Away FC": {
                "records": [],
            },
        }
        registry = _RecordingRegistry("restart-map")

        catalog = build_report_figure_catalog(
            _bundle("restarts", data),
            registry=registry,
        )

        generated = [
            item
            for item in catalog.figures
            if item.section_id == "restarts"
            and item.team_name == "Home FC"
        ]
        self.assertTrue(generated)
        self.assertTrue(
            all(
                item.status is ReportFigureStatus.GENERATED
                for item in generated
            )
        )
        self.assertEqual(len(registry.calls), len(generated))

        for args, _ in registry.calls:
            records = args[0]
            self.assertIsInstance(records, list)
            self.assertNotIsInstance(records, pd.DataFrame)
            self.assertIsInstance(records[0], dict)
            self.assertEqual(
                records[0]["sequence_id"],
                "restart-26-1",
            )



    def test_defensive_transition_payload_is_scoped_by_payload_team_not_event_team(self):
        sequence_id = 12
        sequence_rows = [
            {
                "loss_sequence_id": sequence_id,
                "team_name": "Away FC",
                "type_name": "Pass",
                "outcome": "Successful",
                "x": 48.0,
                "y": 45.0,
                "end_x": 66.0,
                "end_y": 50.0,
                "playerName": "Opponent A",
                "timeMin": 61,
                "timeSec": 2,
                "timeMin_at_loss": 61,
                "timeSec_at_loss": 0,
                "sequence_outcome_type": "Opponent Possession Consolidated",
                "terminal_outcome": "consolidated",
                "loss_x": 48.0,
                "loss_y": 45.0,
            },
            {
                "loss_sequence_id": sequence_id,
                "team_name": "Away FC",
                "type_name": "Pass",
                "outcome": "Successful",
                "x": 66.0,
                "y": 50.0,
                "end_x": 73.0,
                "end_y": 46.0,
                "playerName": "Opponent B",
                "timeMin": 61,
                "timeSec": 5,
                "timeMin_at_loss": 61,
                "timeSec_at_loss": 0,
                "sequence_outcome_type": "Opponent Possession Consolidated",
                "terminal_outcome": "consolidated",
            },
        ]
        data = {
            "Home FC": {
                "combined": [
                    {
                        **sequence_rows[0],
                        "event_count": 2,
                        "duration_seconds": 5.0,
                        "max_controlled_x": 73.0,
                    }
                ],
                "sequences": [sequence_rows],
                "stats": {},
            },
            "Away FC": {
                "combined": [],
                "sequences": [],
                "stats": {},
            },
        }
        registry = _RecordingRegistry("sequence-explorer")

        catalog = build_report_figure_catalog(
            _bundle("defensive-transitions", data),
            registry=registry,
        )

        artifact = next(
            item
            for item in catalog.figures
            if item.id == "defensive-transitions-top-sequence"
            and item.team_name == "Home FC"
        )
        self.assertIs(
            artifact.status,
            ReportFigureStatus.GENERATED,
        )
        self.assertEqual(len(registry.calls), 1)

        normalized = registry.calls[0][0][0]
        self.assertIsInstance(normalized, Mapping)
        self.assertEqual(
            normalized["sequence_type"],
            "defensive_transition",
        )
        self.assertTrue(normalized["events"])

    def test_pdf_section_boundaries_do_not_force_blank_page_at_page_top(self):
        import inspect
        from src.reporting.pdf_renderer import render_match_report_pdf

        source = inspect.getsource(render_match_report_pdf)

        self.assertIn(
            "story.append(PageBreakIfNotEmpty())",
            source,
        )
        self.assertEqual(
            source.count("story.append(PageBreak())"),
            1,
        )


    def test_transition_selector_ranks_sequence_outcome_not_first_event_outcome(self):
        data = {
            "Home FC": {
                "combined": [],
                "sequences": [
                    [
                        {
                            "loss_sequence_id": 0,
                            "team_name": "Home FC",
                            "outcome": "Unsuccessful",
                            "sequence_outcome_type": "Lost Possessions",
                            "terminal_outcome": "turnover",
                            "type_name": "Pass",
                            "x": 30.0,
                            "y": 40.0,
                            "end_x": 45.0,
                            "end_y": 42.0,
                            "timeMin": 3,
                            "timeSec": 1,
                            "total_seconds": 181.0,
                            "playerName": "Player A",
                        }
                    ],
                    [
                        {
                            "loss_sequence_id": 53,
                            "team_name": "Home FC",
                            "outcome": "Successful",
                            "sequence_outcome_type": "Goals",
                            "terminal_outcome": "goal",
                            "type_name": "Pass",
                            "x": 75.0,
                            "y": 44.0,
                            "end_x": 86.0,
                            "end_y": 46.0,
                            "timeMin": 56,
                            "timeSec": 2,
                            "total_seconds": 3362.0,
                            "playerName": "Player B",
                        },
                        {
                            "loss_sequence_id": 53,
                            "team_name": "Home FC",
                            "outcome": "Successful",
                            "sequence_outcome_type": "Goals",
                            "terminal_outcome": "goal",
                            "type_name": "Goal",
                            "x": 86.0,
                            "y": 46.0,
                            "end_x": None,
                            "end_y": None,
                            "timeMin": 56,
                            "timeSec": 4,
                            "total_seconds": 3364.0,
                            "playerName": "Player C",
                        },
                    ],
                ],
                "stats": {},
            },
            "Away FC": {
                "combined": [],
                "sequences": [],
                "stats": {},
            },
        }
        registry = _RecordingRegistry("sequence-explorer")

        catalog = build_report_figure_catalog(
            _bundle("offensive-transitions", data),
            registry=registry,
        )

        artifact = next(
            item
            for item in catalog.figures
            if item.id == "offensive-transitions-top-sequence"
            and item.team_name == "Home FC"
        )

        self.assertIs(
            artifact.status,
            ReportFigureStatus.GENERATED,
        )
        self.assertIn("milestone='goal'", artifact.selection_reason)
        self.assertIn("stable_id='53'", artifact.selection_reason)

        normalized = registry.calls[0][0][0]
        self.assertEqual(str(normalized["sequence_id"]), "53")

        # Transition normalization intentionally inserts one synthetic
        # possession-change trigger when the raw sequence does not already
        # contain a turnover event. It may also infer carries between events,
        # so the presentation event count must not be hard-coded to raw rows.
        events = normalized["events"]
        synthetic_triggers = [
            event
            for event in events
            if (event.get("metadata") or {}).get("synthetic_trigger") is True
        ]
        self.assertEqual(len(synthetic_triggers), 1)
        self.assertIn(
            "Goal",
            {event.get("raw_event_type") for event in events},
        )


    def test_table_subheading_does_not_keep_entire_long_table_together(self):
        from src.reporting.pdf_renderer import MatchReportPdfConfig, _styles

        styles = _styles(MatchReportPdfConfig())

        self.assertIn("table_subheading", styles)
        self.assertFalse(styles["table_subheading"].keepWithNext)
        self.assertTrue(styles["subheading"].keepWithNext)

if __name__ == "__main__":
    unittest.main()
