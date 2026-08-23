import ast
import logging
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.logging_config import (
    configure_logging,
)


ROOT = Path(__file__).resolve().parents[1]


def runtime_python_files():
    yield ROOT / "app.py"

    for relative in (
        "src/data_processing",
        "src/data_preparation_for_plots",
        "src/metrics",
        "src/visualization",
        "src/components",
    ):
        folder = ROOT / relative

        if not folder.exists():
            continue

        yield from folder.rglob("*.py")


class CleanLoggingTests(unittest.TestCase):

    def test_normal_runtime_defaults_to_warning(self):
        with patch.dict(
            os.environ,
            {
                "MATCH_ANALYSIS_LOG_LEVEL":
                    "WARNING",
            },
            clear=False,
        ):
            level = configure_logging()

        self.assertEqual(
            level,
            logging.WARNING,
        )
        self.assertEqual(
            logging.getLogger(
                "src.metrics"
            ).getEffectiveLevel(),
            logging.WARNING,
        )

    def test_debug_can_be_enabled_explicitly(self):
        level = configure_logging(
            "DEBUG"
        )

        self.assertEqual(
            level,
            logging.DEBUG,
        )
        self.assertEqual(
            logging.getLogger(
                "src.metrics"
            ).getEffectiveLevel(),
            logging.DEBUG,
        )

        # Restore the normal test/runtime contract.
        configure_logging(
            "WARNING"
        )

    def test_runtime_tree_has_no_active_print_calls(self):
        offenders = []

        for path in runtime_python_files():
            source = path.read_text(
                encoding="utf-8",
            )

            tree = ast.parse(
                source,
                filename=str(path),
            )

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(
                        node.func,
                        ast.Name,
                    )
                    and node.func.id == "print"
                ):
                    offenders.append(
                        (
                            str(
                                path.relative_to(
                                    ROOT
                                )
                            ),
                            node.lineno,
                        )
                    )

        self.assertEqual(
            offenders,
            [],
            msg=(
                "Active print() calls remain in runtime "
                f"code: {offenders}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
