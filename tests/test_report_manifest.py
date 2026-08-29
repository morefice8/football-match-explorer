from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from src.reporting import (
    REPORT_MANIFEST,
    REQUIRED_SECTION_IDS,
    ReportScope,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SECTION_IDS = (
    "overview",
    "formation-timeline",
    "mean-positions",
    "pass-network",
    "progressive-passes",
    "final-third-entries",
    "pass-locations",
    "cross-flow",
    "build-up",
    "defensive-shape",
    "ppda",
    "defensive-transitions",
    "offensive-transitions",
    "restarts",
    "player-highlights",
    "methodology-appendix",
)

EXPECTED_REQUIRED_SECTION_IDS = (
    "overview",
    "formation-timeline",
    "mean-positions",
    "pass-network",
    "progressive-passes",
    "final-third-entries",
    "pass-locations",
    "cross-flow",
    "build-up",
    "defensive-shape",
    "ppda",
    "defensive-transitions",
    "offensive-transitions",
    "restarts",
    "methodology-appendix",
)


class ReportManifestTests(unittest.TestCase):
    def test_all_ids_are_globally_unique(self):
        ids = REPORT_MANIFEST.all_ids()
        self.assertEqual(len(ids), len(set(ids)))

    def test_sections_have_canonical_order(self):
        self.assertEqual(
            tuple(section.id for section in REPORT_MANIFEST.sections),
            EXPECTED_SECTION_IDS,
        )
        self.assertEqual(
            tuple(section.order for section in REPORT_MANIFEST.sections),
            tuple(range(1, 17)),
        )

    def test_manifest_serializes_to_json_safe_structure(self):
        payload = REPORT_MANIFEST.to_dict()
        encoded = REPORT_MANIFEST.to_json()
        decoded = json.loads(encoded)

        self.assertEqual(decoded, payload)
        self.assertEqual(decoded["id"], "match-analysis-report")
        self.assertEqual(decoded["schema_version"], "1.0")
        self.assertEqual(len(decoded["sections"]), 16)

    def test_required_sections_are_present(self):
        self.assertEqual(
            REQUIRED_SECTION_IDS,
            EXPECTED_REQUIRED_SECTION_IDS,
        )
        self.assertEqual(
            REPORT_MANIFEST.required_section_ids(),
            EXPECTED_REQUIRED_SECTION_IDS,
        )
        present = {section.id for section in REPORT_MANIFEST.sections}
        self.assertTrue(set(EXPECTED_REQUIRED_SECTION_IDS).issubset(present))

    def test_every_section_is_full_match_scope(self):
        self.assertTrue(
            all(
                section.scope is ReportScope.FULL_MATCH
                for section in REPORT_MANIFEST.sections
            )
        )

    def test_each_section_declares_content_and_export_contract(self):
        for section in REPORT_MANIFEST.sections:
            with self.subTest(section=section.id):
                self.assertTrue(section.figures or section.tables)
                self.assertGreater(section.export.width_px, 0)
                self.assertGreater(section.export.height_px, 0)
                self.assertTrue(section.selection.rule)
                self.assertIsNotNone(section.missing_data)

                for item in (*section.figures, *section.tables):
                    self.assertTrue(item.id)
                    self.assertTrue(item.title)
                    self.assertTrue(item.selection.rule)
                    self.assertGreater(item.export.width_px, 0)
                    self.assertGreater(item.export.height_px, 0)

    def test_reporting_package_has_no_dash_or_app_imports(self):
        reporting_dir = ROOT / "src" / "reporting"

        forbidden = []
        for path in reporting_dir.glob("*.py"):
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
                    ):
                        forbidden.append((path.name, module))

        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
