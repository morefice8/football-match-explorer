from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from src import config
from src.data_processing import pass_processing, preprocess
from src.reporting.audit import FORBIDDEN_PDF_TEXT, searchable_pdf_text
from src.reporting.bundle import build_match_report_data_bundle
from src.reporting.figure_catalog import RendererRegistry
from src.reporting.pack_builder import TABLE_PATHS, build_match_analysis_pack
from src.utils import mapping_loader


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "report11_real_shaped_match.json"


def _processed_fixture():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    event_map = mapping_loader.load_opta_event_mapping(
        ROOT / config.OPTA_EVENTS_XLSX
    )
    qualifier_map = mapping_loader.load_opta_qualifier_mapping(
        ROOT / config.OPTA_QUALIFIERS_JSON
    )
    match_info = config.extract_match_info(raw)
    frame, _, _, _ = preprocess.process_opta_events(
        raw,
        event_map,
        qualifier_map,
        match_info,
    )
    return frame, match_info


def _tiny_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (24, 16), "white").save(output, format="PNG")
    return output.getvalue()


def _dummy_renderer(*args, **kwargs):
    return go.Figure()


class Report11RealShapedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.match_info = _processed_fixture()

    def test_fixture_covers_raw_pipeline_edge_cases(self):
        frame = self.frame
        self.assertFalse(frame.empty)
        self.assertTrue(frame.columns.is_unique)

        # Historical collision: Q145 and Q292 must both survive canonically.
        player_on = frame.loc[frame["typeId"].eq(19)].iloc[0]
        self.assertEqual(player_on["Formation slot"], "11")
        self.assertEqual(player_on["Substitution position code"], "6")

        # Substitution + later team setup represent an in-match formation change.
        self.assertTrue(frame["typeId"].eq(18).any())
        self.assertTrue(frame["typeId"].eq(19).any())
        self.assertGreaterEqual(frame["typeId"].eq(34).sum(), 3)

        passes = pass_processing.get_passes_df(frame.copy())
        self.assertIn("receiver", passes.columns)
        self.assertGreater(int(passes["receiver_is_reliable"].sum()), 0)

        carries = pass_processing.infer_carries(frame.copy())
        self.assertFalse(carries.empty)
        self.assertTrue(carries["carry_is_inferred"].all())

        shots = frame[frame["type_name"].isin(config.SHOT_TYPES)]
        self.assertGreaterEqual(len(shots), 4)

        for flag in ("ThrowIn", "Free kick taken", "Corner taken", "Goal Kick"):
            self.assertIn(flag, frame.columns)
            self.assertTrue(pd.to_numeric(frame[flag], errors="coerce").fillna(0).gt(0).any())

    def test_fixture_builds_transition_and_restart_report_sections(self):
        bundle = build_match_report_data_bundle(self.frame, self.match_info)
        for section_id in (
            "defensive-transitions",
            "offensive-transitions",
            "restarts",
        ):
            with self.subTest(section=section_id):
                self.assertEqual(bundle.section(section_id).status.value, "generated")

    def test_pack_pdf_contains_no_error_fallback_or_traceback(self):
        """Exercise raw fixture -> bundle -> catalog -> PDF -> ZIP deterministically.

        The renderer registry is replaced with a neutral Plotly renderer and the
        PNG conversion is replaced with a valid in-memory image. REPORT-11's CLI
        separately exercises the real Kaleido/Chrome stack on an external match.
        This test therefore validates report composition without making unit CI
        depend on a system browser.
        """

        bundle = build_match_report_data_bundle(self.frame, self.match_info)
        png = _tiny_png()

        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            payload = build_match_analysis_pack(bundle)

        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertIsNone(archive.testzip())
            names = archive.namelist()
            for table in TABLE_PATHS:
                self.assertIn(table, names)
            pdf_name = next(name for name in names if name.endswith(".pdf"))
            pdf_text = searchable_pdf_text(archive.read(pdf_name))

        lowered = pdf_text.casefold()
        for phrase in FORBIDDEN_PDF_TEXT:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase.casefold(), lowered)


if __name__ == "__main__":
    unittest.main()
