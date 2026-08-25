import ast
import unittest
from pathlib import Path


class TransitionHeatmapInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("app.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _function_source(self, name):
        lines = self.source.splitlines(keepends=True)
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                start = min(
                    [node.lineno]
                    + [decorator.lineno for decorator in node.decorator_list]
                )
                return "".join(lines[start - 1:node.end_lineno])
        self.fail(f"Missing app function {name}")

    def test_both_heatmap_panels_have_kpi_and_cell_stores(self):
        for needle in (
            'id="loss-heatmap-kpis"',
            'id="recovery-heatmap-kpis"',
            'id="def-transition-heatmap-cell"',
            'id="off-transition-heatmap-cell"',
        ):
            self.assertIn(needle, self.source)

    def test_defensive_explorer_filters_by_loss_cell(self):
        block = self._function_source("update_def_transition_plot")
        self.assertIn('Input("def-transition-heatmap-cell", "data")', block)
        self.assertIn('filter_sequences_by_cell(', block)
        self.assertIn('location_kind="loss"', block)

    def test_offensive_explorer_filters_by_recovery_cell(self):
        block = self._function_source("update_off_transition_plot")
        self.assertIn('Input("off-transition-heatmap-cell", "data")', block)
        self.assertIn('filter_sequences_by_cell(', block)
        self.assertIn('location_kind="recovery"', block)

    def test_cell_callbacks_reset_controller_counts(self):
        for name in (
            "select_def_transition_heatmap_cell",
            "select_off_transition_heatmap_cell",
        ):
            block = self._function_source(name)
            self.assertIn("make_carousel_controller", block)
            self.assertIn("cell_from_click", block)


    def test_defensive_cell_filters_use_transition_team_frame(self):
        explorer = self._function_source(
            "update_def_transition_plot"
        )
        selector = self._function_source(
            "select_def_transition_heatmap_cell"
        )

        self.assertIn(
            "loss_to_transition_frame=True",
            explorer,
        )
        self.assertIn(
            "loss_to_transition_frame=True",
            selector,
        )

if __name__ == "__main__":
    unittest.main()
