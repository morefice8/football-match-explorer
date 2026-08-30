from __future__ import annotations

import ast
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from src.reporting.bundle import (
    MatchReportDataBundle,
    ReportSectionBundle,
    ReportSectionStatus,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportScope
from src.reporting.pack_builder import (
    MAX_RECOMMENDED_PACK_BYTES,
    TABLE_PATHS,
    build_match_analysis_pack,
    build_match_analysis_pack_buffer,
    match_report_pdf_filename,
)


ROOT = Path(__file__).resolve().parents[1]
PDF_BYTES = b"%PDF-1.4\nREPORT-06 synthetic PDF\n%%EOF\n"


def _section_payload(section_id: str, home: str, away: str):
    if section_id == "overview":
        return {
            "game_profile": {
                home: {
                    "passes": 412,
                    "shots": 11,
                },
                away: {
                    "passes": 361,
                    "shots": 8,
                },
            },
            "data_coverage": {
                "event_rows": 1287,
                "receiver": {
                    "coverage_pct": np.nan,
                },
            },
            "shots": pd.DataFrame(
                [
                    {
                        "id": 50,
                        "team_name": home,
                        "type_name": "Shot",
                        "x": 82.0,
                        "y": np.nan,
                    }
                ]
            ),
        }

    if section_id == "pass-network":
        return {
            "passes": pd.DataFrame(
                [
                    {
                        "id": 1,
                        "team_name": home,
                        "playerName": "A",
                    }
                ]
            ),
            "teams": {
                home: {
                    "edges": pd.DataFrame(
                        [
                            {
                                "passer": "A",
                                "receiver": "B",
                                "count": 8,
                            }
                        ]
                    )
                },
                away: {
                    "edges": pd.DataFrame(
                        [
                            {
                                "passer": "C",
                                "receiver": "D",
                                "count": 6,
                            }
                        ]
                    )
                },
            },
        }

    if section_id == "progressive-passes":
        return {
            "teams": {
                home: {
                    "player_ranking": pd.DataFrame(
                        [{"playerName": "A", "progressive": 7}]
                    )
                },
                away: {
                    "player_ranking": pd.DataFrame(
                        [{"playerName": "C", "progressive": 5}]
                    )
                },
            }
        }

    if section_id == "final-third-entries":
        return {
            "carries": pd.DataFrame(
                [
                    {
                        "id": "carry-1",
                        "team_name": home,
                        "type_name": "Carry",
                    }
                ]
            ),
            "teams": {
                home: {
                    "entries": pd.DataFrame(
                        [{"id": 10, "entry_type": "Pass"}]
                    )
                },
                away: {
                    "entries": pd.DataFrame(
                        [{"id": 11, "entry_type": "Carry"}]
                    )
                },
            },
        }

    if section_id == "pass-locations":
        return {
            "passes": pd.DataFrame(
                [
                    {
                        "id": 1,
                        "team_name": home,
                        "type_name": "Pass",
                    },
                    {
                        "id": 2,
                        "team_name": away,
                        "type_name": "Pass",
                    },
                ]
            )
        }

    if section_id == "cross-flow":
        return {
            home: {
                "routes": pd.DataFrame(
                    [
                        {
                            "Origin Zone": "Right",
                            "Destination Zone": "Center Box",
                            "Crosses": 4,
                        }
                    ]
                )
            },
            away: {
                "routes": pd.DataFrame(
                    [
                        {
                            "Origin Zone": "Left",
                            "Destination Zone": "Far Post",
                            "Crosses": 3,
                        }
                    ]
                )
            },
        }

    if section_id == "build-up":
        return {
            "teams": {
                home: {
                    "summary": pd.DataFrame(
                        [
                            {
                                "sequence_id": "bu-1",
                                "produced_shot": True,
                            }
                        ]
                    )
                },
                away: {
                    "summary": pd.DataFrame(
                        [
                            {
                                "sequence_id": "bu-2",
                                "produced_shot": False,
                            }
                        ]
                    )
                },
            }
        }

    if section_id == "defensive-shape":
        return {
            home: {
                "actions": pd.DataFrame(
                    [
                        {
                            "id": 90,
                            "team_name": home,
                            "type_name": "Tackle",
                        }
                    ]
                )
            },
            away: {
                "actions": pd.DataFrame(
                    [
                        {
                            "id": 91,
                            "team_name": away,
                            "type_name": "Interception",
                        }
                    ]
                )
            },
        }

    if section_id in {
        "defensive-transitions",
        "offensive-transitions",
    }:
        prefix = (
            "dt"
            if section_id == "defensive-transitions"
            else "ot"
        )
        return {
            home: {
                "combined": pd.DataFrame(
                    [
                        {
                            "loss_sequence_id": f"{prefix}-1",
                            "team_name": home,
                            "type_name": "Pass",
                        }
                    ]
                ),
                "stats": {"sequences": 1},
            },
            away: {
                "combined": pd.DataFrame(
                    [
                        {
                            "loss_sequence_id": f"{prefix}-2",
                            "team_name": away,
                            "type_name": "Ball recovery",
                        }
                    ]
                ),
                "stats": {"sequences": 1},
            },
        }

    if section_id == "restarts":
        return {
            home: {
                "records": pd.DataFrame(
                    [
                        {
                            "sequence_id": "r-1",
                            "restart_type": "Corner",
                        }
                    ]
                )
            },
            away: {
                "records": pd.DataFrame(
                    [
                        {
                            "sequence_id": "r-2",
                            "restart_type": "Free Kick",
                        }
                    ]
                )
            },
        }

    if section_id == "player-highlights":
        passing = pd.DataFrame(
            [
                {
                    "Offensive Pass Contributions": 9,
                    "Progressive Passes": 7,
                }
            ],
            index=pd.Index(["A"], name="playerName"),
        )
        shooting = pd.DataFrame(
            [
                {
                    "playerName": "B",
                    "Shot Sequence Involvements": 6,
                }
            ]
        )
        defending = pd.DataFrame(
            [
                {
                    "playerName": "C",
                    "unique": 8,
                }
            ]
        )
        return {
            "player_stats": passing,
            "shot_sequence_ranking": shooting,
            "defensive_ranking": defending,
        }

    return {}


def _bundle(
    *,
    teams=("Home FC", "Away FC"),
    match_info=None,
    overrides=None,
):
    home, away = teams
    overrides = dict(overrides or {})

    sections = []
    for spec in REPORT_MANIFEST.sections:
        status = overrides.get(
            spec.id,
            ReportSectionStatus.GENERATED,
        )
        sections.append(
            ReportSectionBundle(
                id=spec.id,
                status=status,
                data=(
                    None
                    if status is ReportSectionStatus.ERROR
                    else _section_payload(spec.id, home, away)
                ),
                error_type=(
                    "RuntimeError"
                    if status is ReportSectionStatus.ERROR
                    else None
                ),
                error_message=(
                    "synthetic failure"
                    if status is ReportSectionStatus.ERROR
                    else None
                ),
            )
        )

    return MatchReportDataBundle(
        manifest_id=REPORT_MANIFEST.id,
        manifest_version=REPORT_MANIFEST.schema_version,
        source_signature="report-06-source",
        scope=ReportScope.FULL_MATCH,
        teams=tuple(teams),
        match_info=dict(match_info or {}),
        sections=tuple(sections),
    )


def _build(bundle):
    with patch(
        "src.reporting.pack_builder.build_report_figure_catalog",
        return_value=object(),
    ) as catalog:
        with patch(
            "src.reporting.pack_builder.render_match_report_pdf",
            return_value=PDF_BYTES,
        ) as pdf:
            pack = build_match_analysis_pack(bundle)
    return pack, catalog, pdf


class MatchAnalysisPackTests(unittest.TestCase):
    def test_zip_contains_exact_structure_and_valid_pdf(self):
        bundle = _bundle()
        payload, _, _ = _build(bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            expected = [
                match_report_pdf_filename(bundle),
                "report-data.json",
                "report-manifest.json",
                *TABLE_PATHS,
            ]
            self.assertEqual(archive.namelist(), expected)
            self.assertTrue(
                archive.read(expected[0]).startswith(b"%PDF")
            )

    def test_report_data_json_is_strict_and_has_no_invalid_nan(self):
        bundle = _bundle()
        payload, _, _ = _build(bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            text = archive.read(
                "report-data.json"
            ).decode("utf-8")
            parsed = json.loads(text)

        self.assertNotIn("NaN", text)
        self.assertEqual(
            parsed["sections"][0]["data"]["data_coverage"]
            ["receiver"]["coverage_pct"],
            None,
        )

    def test_manifest_contains_generation_status_for_every_section(self):
        bundle = _bundle(
            overrides={
                "cross-flow": ReportSectionStatus.EMPTY,
                "pass-network": ReportSectionStatus.ERROR,
                "player-highlights": ReportSectionStatus.SKIPPED,
            }
        )
        payload, _, _ = _build(bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            manifest = json.loads(
                archive.read("report-manifest.json")
            )

        statuses = manifest["generation"]["section_statuses"]
        self.assertEqual(
            set(statuses),
            {section.id for section in REPORT_MANIFEST.sections},
        )
        self.assertEqual(statuses["cross-flow"], "empty")
        self.assertEqual(statuses["pass-network"], "error")
        self.assertEqual(statuses["player-highlights"], "skipped")
        self.assertEqual(statuses["overview"], "generated")

        section_map = {
            section["id"]: section
            for section in manifest["sections"]
        }
        self.assertEqual(
            section_map["pass-network"]["generation"]["error_message"],
            "synthetic failure",
        )

    def test_every_csv_is_utf8_and_parseable(self):
        bundle = _bundle()
        payload, _, _ = _build(bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for path in TABLE_PATHS:
                text = archive.read(path).decode("utf-8")
                rows = list(csv.reader(StringIO(text)))
                self.assertGreaterEqual(
                    len(rows),
                    1,
                    msg=path,
                )
                self.assertGreaterEqual(
                    len(rows[0]),
                    1,
                    msg=path,
                )

    def test_special_team_names_have_sanitized_deterministic_pdf_name(self):
        bundle = _bundle(
            teams=(
                "Atlético / Club",
                "Paris & Côte d'Azur",
            )
        )

        first = match_report_pdf_filename(bundle)
        second = match_report_pdf_filename(bundle)

        self.assertEqual(first, second)
        self.assertEqual(
            first,
            "atletico-club-vs-paris-cote-d-azur-report.pdf",
        )
        self.assertNotIn("/", first)
        self.assertNotIn("\\", first)

        payload, _, _ = _build(bundle)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertIn(first, archive.namelist())

    def test_missing_metadata_does_not_break_pack(self):
        bundle = _bundle(match_info={})
        payload, _, _ = _build(bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertIn(
                "home-fc-vs-away-fc-report.pdf",
                archive.namelist(),
            )
            report_data = json.loads(
                archive.read("report-data.json")
            )
        self.assertEqual(report_data["match_info"], {})

    def test_same_bundle_drives_catalog_pdf_json_and_tables_without_metrics(self):
        bundle = _bundle()
        payload, catalog, pdf = _build(bundle)

        catalog_args = catalog.call_args.args
        pdf_args = pdf.call_args.args
        self.assertIs(catalog_args[0], bundle)
        self.assertIs(pdf_args[0], bundle)
        self.assertIs(pdf_args[1], catalog.return_value)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            report_data = json.loads(
                archive.read("report-data.json")
            )
        self.assertEqual(
            report_data["source_signature"],
            bundle.source_signature,
        )

        source = (
            ROOT / "src" / "reporting" / "pack_builder.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")

        self.assertFalse(
            any(
                module == "src.metrics"
                or module.startswith("src.metrics.")
                or module == "app"
                or module.startswith("app.")
                or module == "dash"
                or module.startswith("dash.")
                for module in imports
            )
        )
        self.assertNotIn(
            "build_match_report_data_bundle",
            source,
        )
        self.assertNotIn("tempfile", source)

    def test_zip_is_deflated_and_contains_no_raw_json_entry(self):
        bundle = _bundle()
        payload, _, _ = _build(bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for info in archive.infolist():
                self.assertEqual(
                    info.compress_type,
                    zipfile.ZIP_DEFLATED,
                    msg=info.filename,
                )
                self.assertNotIn(
                    "raw",
                    info.filename.casefold(),
                )

    def test_empty_sections_still_emit_all_csv_files_with_headers(self):
        overrides = {
            section.id: ReportSectionStatus.EMPTY
            for section in REPORT_MANIFEST.sections
        }
        bundle = _bundle(overrides=overrides)
        payload, _, _ = _build(bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for path in TABLE_PATHS:
                text = archive.read(path).decode("utf-8")
                header = next(csv.reader(StringIO(text)))
                self.assertTrue(header, msg=path)

    def test_buffer_api_is_rewound_and_pack_stays_under_recommended_size(self):
        bundle = _bundle()

        with patch(
            "src.reporting.pack_builder.build_report_figure_catalog",
            return_value=object(),
        ):
            with patch(
                "src.reporting.pack_builder.render_match_report_pdf",
                return_value=PDF_BYTES,
            ):
                first = build_match_analysis_pack(bundle)
                second = build_match_analysis_pack(bundle)
                buffer = build_match_analysis_pack_buffer(bundle)

        self.assertEqual(first, second)
        self.assertEqual(buffer.tell(), 0)
        self.assertEqual(buffer.read(), first)
        self.assertLess(len(first), MAX_RECOMMENDED_PACK_BYTES)


if __name__ == "__main__":
    unittest.main()
