from __future__ import annotations

import base64
import tempfile
from pathlib import Path
import unittest
import zlib

from src.reporting.audit import (
    MATCH_JSON_ENV_VAR,
    resolve_match_json_path,
    searchable_pdf_text,
    validate_pdf_bytes,
)


class ReportAuditUtilityTests(unittest.TestCase):
    def test_input_path_falls_back_to_environment_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "match.json"
            path.write_text("{}", encoding="utf-8")
            resolved = resolve_match_json_path(
                None,
                environ={MATCH_JSON_ENV_VAR: str(path)},
            )
        self.assertEqual(resolved, path.resolve())

    def test_cli_argument_takes_precedence_over_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")
            resolved = resolve_match_json_path(
                first,
                environ={MATCH_JSON_ENV_VAR: str(second)},
            )
        self.assertEqual(resolved, first.resolve())

    def test_pdf_scanner_reads_ascii85_flate_reportlab_style_stream(self):
        visible = b"BT (Figure unavailable: synthetic) Tj ET"
        encoded = base64.a85encode(zlib.compress(visible), adobe=True)
        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Filter [/ASCII85Decode /FlateDecode] >>\n"
            b"stream\n" + encoded + b"\nendstream\nendobj\n%%EOF\n"
        )
        self.assertIn("Figure unavailable", searchable_pdf_text(pdf))
        issues = validate_pdf_bytes(pdf)
        self.assertTrue(any(i.code == "pdf-forbidden-text" for i in issues))

    def test_clean_pdf_has_no_forbidden_text_issue(self):
        pdf = b"%PDF-1.4\n1 0 obj <<>> endobj\n%%EOF\n"
        issues = validate_pdf_bytes(pdf)
        self.assertFalse(any(i.code == "pdf-forbidden-text" for i in issues))


if __name__ == "__main__":
    unittest.main()
