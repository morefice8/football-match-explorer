from __future__ import annotations

from io import BytesIO
import csv
import json
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.data_exports import (
    EVENTS_CORE_COLUMNS,
    EVENTS_CORE_PATH,
    build_events_core,
)
from src.reporting.figure_catalog import RendererRegistry
from src.reporting.pack_builder import (
    DEBUG_TABLE_PATHS,
    TABLE_PATHS,
    build_match_analysis_pack,
)
from tests.test_report11_real_shaped_fixture import _processed_fixture


def _tiny_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (48, 27), "white").save(output, format="PNG")
    return output.getvalue()


def _dummy_renderer(*args, **kwargs):
    return go.Figure()


class Report20LightweightPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, match_info = _processed_fixture()
        cls.frame = frame
        cls.bundle = build_match_report_data_bundle(frame, match_info)

    def test_events_core_is_one_row_per_event_and_bounded_to_38_columns(self):
        core = build_events_core(self.bundle)

        self.assertEqual(list(core.columns), list(EVENTS_CORE_COLUMNS))
        self.assertLessEqual(len(core.columns), 38)
        self.assertEqual(len(core), len(self.frame))
        self.assertEqual(core["id"].nunique(), len(core))

        for column in (
            "receiver",
            "receiver_confidence",
            "buildup_sequence_id",
            "defensive_transition_sequence_id",
            "offensive_transition_sequence_id",
            "restart_sequence_id",
            "is_progressive",
            "is_shot",
            "shot_outcome",
            "shot_on_target",
            "shot_blocked",
            "shot_blocked_qualifier",
            "shot_keeper_saved_off_target",
            "shot_hit_woodwork",
            "shot_own_goal_qualifier",
            "shot_classification_issue",
            "is_defensive_action",
            "is_restart",
        ):
            self.assertIn(column, core.columns)

    def test_standard_pack_excludes_bulk_debug_dumps(self):
        png = _tiny_png()

        with patch.object(
            RendererRegistry,
            "resolve",
            return_value=_dummy_renderer,
        ), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            standard = build_match_analysis_pack(self.bundle)
            debug = build_match_analysis_pack(
                self.bundle,
                debug=True,
            )

        self.assertLess(len(standard), len(debug))

        with zipfile.ZipFile(BytesIO(standard)) as archive:
            names = archive.namelist()
            self.assertIn(EVENTS_CORE_PATH, names)
            self.assertNotIn("report-data.json", names)
            self.assertNotIn("tables/event-explorer.csv", names)

        with zipfile.ZipFile(BytesIO(debug)) as archive:
            names = archive.namelist()
            self.assertIn("report-data.json", names)
            for path in DEBUG_TABLE_PATHS:
                self.assertIn(path, names)

    def test_standard_csvs_are_utf8_parseable_and_manifest_documented(self):
        png = _tiny_png()

        with patch.object(
            RendererRegistry,
            "resolve",
            return_value=_dummy_renderer,
        ), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            payload = build_match_analysis_pack(self.bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            manifest = json.loads(
                archive.read("report-manifest.json").decode("utf-8")
            )
            docs = manifest["generation"]["csv_exports"]
            documented = {item["path"] for item in docs}

            self.assertEqual(documented, set(TABLE_PATHS))
            self.assertTrue(
                all(item.get("description") for item in docs)
            )
            self.assertTrue(
                all(item.get("granularity") for item in docs)
            )

            for path in TABLE_PATHS:
                raw = archive.read(path)
                text = raw.decode("utf-8")
                rows = list(csv.reader(text.splitlines()))
                self.assertTrue(rows, msg=path)
                self.assertTrue(rows[0], msg=path)
                self.assertEqual(
                    len(rows[0]),
                    len(set(rows[0])),
                    msg=path,
                )

    def test_events_core_csv_is_lean_and_unique(self):
        png = _tiny_png()

        with patch.object(
            RendererRegistry,
            "resolve",
            return_value=_dummy_renderer,
        ), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            payload = build_match_analysis_pack(self.bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            core = pd.read_csv(
                BytesIO(archive.read(EVENTS_CORE_PATH))
            )

        self.assertLessEqual(len(core.columns), 38)
        self.assertEqual(list(core.columns), list(EVENTS_CORE_COLUMNS))
        self.assertEqual(core["id"].nunique(), len(core))

    def test_specialized_standard_csvs_do_not_repeat_event_frames(self):
        png = _tiny_png()

        with patch.object(
            RendererRegistry,
            "resolve",
            return_value=_dummy_renderer,
        ), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            payload = build_match_analysis_pack(self.bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            final_third = pd.read_csv(
                BytesIO(
                    archive.read("tables/final-third-entries.csv")
                )
            )
            defensive = pd.read_csv(
                BytesIO(
                    archive.read("tables/defensive-transitions.csv")
                )
            )
            offensive = pd.read_csv(
                BytesIO(
                    archive.read("tables/offensive-transitions.csv")
                )
            )

        event_level_columns = {
            "id",
            "eventId",
            "playerName",
            "x",
            "y",
            "end_x",
            "end_y",
        }

        self.assertFalse(
            event_level_columns.intersection(final_third.columns)
        )
        self.assertFalse(
            event_level_columns.intersection(defensive.columns)
        )
        self.assertFalse(
            event_level_columns.intersection(offensive.columns)
        )


if __name__ == "__main__":
    unittest.main()
