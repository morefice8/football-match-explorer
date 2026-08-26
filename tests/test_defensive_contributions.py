import ast
import unittest
from pathlib import Path

import pandas as pd

from src.components import defensive_contribution_view
from src.metrics import defensive_contribution_metrics


class DefensiveContributionMetricTests(unittest.TestCase):
    def test_unique_excludes_fouls_and_failed_tackles(self):
        frame = pd.DataFrame([
            {"id": 1, "playerName": "Defender", "team_name": "Home", "Mapped Jersey Number": 4, "type_name": "Tackle", "outcome": "Successful"},
            {"id": 2, "playerName": "Defender", "team_name": "Home", "Mapped Jersey Number": 4, "type_name": "Tackle", "outcome": "Unsuccessful"},
            {"id": 3, "playerName": "Defender", "team_name": "Home", "Mapped Jersey Number": 4, "type_name": "Interception", "outcome": "Successful"},
            {"id": 4, "playerName": "Defender", "team_name": "Home", "Mapped Jersey Number": 4, "type_name": "Ball recovery", "outcome": "Successful"},
            {"id": 5, "playerName": "Defender", "team_name": "Home", "Mapped Jersey Number": 4, "type_name": "Clearance", "outcome": "Successful"},
            {"id": 6, "playerName": "Defender", "team_name": "Home", "Mapped Jersey Number": 4, "type_name": "Blocked pass", "outcome": "Successful"},
            {"id": 7, "playerName": "Defender", "team_name": "Home", "Mapped Jersey Number": 4, "type_name": "Foul", "outcome": "Unsuccessful"},
        ])

        ranking = defensive_contribution_metrics.build_defensive_ranking(frame)
        row = ranking.iloc[0]

        self.assertEqual(int(row["unique"]), 5)
        self.assertEqual(int(row["tackles_won"]), 1)
        self.assertEqual(int(row["tackles_attempted"]), 2)
        self.assertEqual(int(row["fouls"]), 1)

    def test_ranking_is_unweighted(self):
        frame = pd.DataFrame([
            {"id": index, "playerName": "Volume", "team_name": "Home", "Mapped Jersey Number": 5, "type_name": "Clearance", "outcome": "Successful"}
            for index in range(1, 6)
        ] + [
            {"id": 20, "playerName": "Tackler", "team_name": "Away", "Mapped Jersey Number": 6, "type_name": "Tackle", "outcome": "Successful"},
            {"id": 21, "playerName": "Tackler", "team_name": "Away", "Mapped Jersey Number": 6, "type_name": "Interception", "outcome": "Successful"},
        ])

        ranking = defensive_contribution_metrics.build_defensive_ranking(frame)
        self.assertEqual(ranking.iloc[0]["playerName"], "Volume")
        self.assertNotIn("Weighted Score", ranking.columns)


class DefensiveContributionViewTests(unittest.TestCase):
    def test_view_uses_tackle_fraction(self):
        ranking = pd.DataFrame([{
            "playerName": "Defender",
            "team_name": "Home",
            "jersey": "4",
            "unique": 5,
            "tackles_won": 2,
            "tackles_attempted": 3,
            "interceptions": 1,
            "recoveries": 1,
            "clearances": 1,
            "blocks": 0,
            "fouls": 2,
        }])

        component = defensive_contribution_view.panel(
            ranking,
            home_team_name="Home",
            hcol="#e96a4a",
            acol="#1597c2",
        )
        text = str(component)
        self.assertIn("Defensive contributions", text)
        self.assertIn("2/3", text)
        self.assertIn("fouls", text.lower())


class DefensiveContributionAppTests(unittest.TestCase):
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

    def test_defender_stats_branch_uses_new_panel(self):
        block = self._function_source("render_defending_analysis_content")
        start = block.index('if active_tab == "pa_defender_stats":')
        end = block.index('elif active_tab == "pa_home_defender_map":')
        branch = block[start:end]
        self.assertIn("defensive_contribution_metrics", branch)
        self.assertIn("defensive_contribution_view", branch)
        self.assertNotIn("plot_defender_stats_bar_plotly", branch)
        self.assertNotIn("Weighted Defensive Score", branch)

    def test_public_tab_uses_contribution_language(self):
        self.assertIn('label="Defensive Contributions"', self.source)


if __name__ == "__main__":
    unittest.main()
