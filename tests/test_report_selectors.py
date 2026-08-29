from __future__ import annotations

import ast
from pathlib import Path
import random
import unittest

import pandas as pd

from src.reporting.selectors import (
    select_representative_restart,
    select_representative_sequence,
    select_top_defender,
    select_top_passer,
    select_top_shooting_contributor,
)


ROOT = Path(__file__).resolve().parents[1]


class ReportSelectorTests(unittest.TestCase):
    def test_top_passer_uses_required_tie_break_order(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "playerName": "Zulu",
                "Offensive Pass Contributions": 8,
                "Progressive completed": 3,
                "Passes into Box": 2,
                "Key Passes": 1,
                "Completed Passes": 30,
            },
            {
                "team_name": "A",
                "playerName": "Alpha",
                "Offensive Pass Contributions": 8,
                "Progressive completed": 4,
                "Passes into Box": 0,
                "Key Passes": 0,
                "Completed Passes": 5,
            },
        ])

        selected = select_top_passer(frame, "A")
        self.assertIsNotNone(selected)
        self.assertEqual(selected.selected_name, "Alpha")
        self.assertIn(
            "Offensive Pass Contributions",
            selected.selection_reason,
        )

    def test_top_passer_final_tie_break_is_name(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "playerName": "Zeta",
                "Offensive Pass Contributions": 5,
                "Progressive completed": 2,
                "Passes into Box": 1,
                "Key Passes": 1,
                "Completed Passes": 20,
            },
            {
                "team_name": "A",
                "playerName": "Alfa",
                "Offensive Pass Contributions": 5,
                "Progressive completed": 2,
                "Passes into Box": 1,
                "Key Passes": 1,
                "Completed Passes": 20,
            },
        ])
        self.assertEqual(
            select_top_passer(frame, "A").selected_name,
            "Alfa",
        )

    def test_shooting_contributor_uses_shots_before_assists(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "playerName": "Player One",
                "Shot Sequence Involvements": 6,
                "Shots": 1,
                "Shot Assists": 5,
                "Pre-Assists": 2,
            },
            {
                "team_name": "A",
                "playerName": "Player Two",
                "Shot Sequence Involvements": 6,
                "Shots": 2,
                "Shot Assists": 0,
                "Pre-Assists": 0,
            },
        ])

        selected = select_top_shooting_contributor(frame, "A")
        self.assertEqual(selected.selected_name, "Player Two")

    def test_defender_prefers_fewer_fouls_after_all_positive_ties(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "playerName": "More Fouls",
                "Unique Defensive Contributions": 7,
                "Interceptions": 2,
                "Tackles Won": 2,
                "Recoveries": 2,
                "Clearances": 1,
                "Blocks": 1,
                "Fouls": 3,
            },
            {
                "team_name": "A",
                "playerName": "Fewer Fouls",
                "Unique Defensive Contributions": 7,
                "Interceptions": 2,
                "Tackles Won": 2,
                "Recoveries": 2,
                "Clearances": 1,
                "Blocks": 1,
                "Fouls": 1,
            },
        ])

        selected = select_top_defender(frame, "A")
        self.assertEqual(selected.selected_name, "Fewer Fouls")

    def test_empty_candidates_return_none(self):
        empty = pd.DataFrame()
        self.assertIsNone(select_top_passer(empty, "A"))
        self.assertIsNone(
            select_top_shooting_contributor(empty, "A")
        )
        self.assertIsNone(select_top_defender(empty, "A"))
        self.assertIsNone(
            select_representative_sequence(
                empty,
                "A",
                category="build-up",
            )
        )
        self.assertIsNone(
            select_representative_restart(empty, "A")
        )

    def test_missing_primary_value_loses_to_present_value(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "playerName": "Missing",
                "Offensive Pass Contributions": None,
                "Progressive completed": 99,
            },
            {
                "team_name": "A",
                "playerName": "Present",
                "Offensive Pass Contributions": 1,
                "Progressive completed": 0,
            },
        ])
        self.assertEqual(
            select_top_passer(frame, "A").selected_name,
            "Present",
        )

    def test_selection_is_scoped_per_team(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "playerName": "A Player",
                "Offensive Pass Contributions": 1,
            },
            {
                "team_name": "B",
                "playerName": "B Player",
                "Offensive Pass Contributions": 99,
            },
        ])

        self.assertEqual(
            select_top_passer(frame, "A").selected_name,
            "A Player",
        )
        self.assertEqual(
            select_top_passer(frame, "B").selected_name,
            "B Player",
        )

    def test_sequence_priority_beats_territorial_progression(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "sequence_id": "seq-shot",
                "outcome": "shot",
                "territorial_progression": 10,
                "action_count": 2,
                "duration": 3,
            },
            {
                "team_name": "A",
                "sequence_id": "seq-box",
                "outcome": "box entry",
                "territorial_progression": 90,
                "action_count": 20,
                "duration": 30,
            },
        ])

        selected = select_representative_sequence(
            frame,
            "A",
            category="offensive-transition",
        )
        self.assertEqual(selected.selected_id, "seq-shot")

    def test_sequence_ties_follow_progression_actions_duration_id(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "sequence_id": "seq-z",
                "outcome": "final third",
                "territorial_progression": 30,
                "action_count": 5,
                "duration": 8,
            },
            {
                "team_name": "A",
                "sequence_id": "seq-a",
                "outcome": "final third",
                "territorial_progression": 30,
                "action_count": 5,
                "duration": 8,
            },
        ])

        selected = select_representative_sequence(
            frame,
            "A",
            category="build-up",
        )
        self.assertEqual(selected.selected_id, "seq-a")
        self.assertIn("stable_id='seq-a'", selected.selection_reason)

    def test_restart_priority_precedes_destination_quality(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "restart_id": "shot",
                "development_outcome": "shot",
                "outcome": "Unsuccessful Delivery",
                "destination": "Own Third",
            },
            {
                "team_name": "A",
                "restart_id": "delivery",
                "development_outcome": "Possession Retained",
                "outcome": "Successful Delivery",
                "destination": "Center of 6-Yard Box",
            },
        ])

        selected = select_representative_restart(frame, "A")
        self.assertEqual(selected.selected_id, "shot")

    def test_restart_uses_destination_quality_then_id(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "restart_id": "r-z",
                "development_outcome": "retained development",
                "destination": "Final Third",
            },
            {
                "team_name": "A",
                "restart_id": "r-b",
                "development_outcome": "retained development",
                "destination": "Center Box",
            },
            {
                "team_name": "A",
                "restart_id": "r-a",
                "development_outcome": "retained development",
                "destination": "Center Box",
            },
        ])

        selected = select_representative_restart(frame, "A")
        self.assertEqual(selected.selected_id, "r-a")

    def test_determinism_is_independent_from_row_order(self):
        base = [
            {
                "team_name": "A",
                "playerName": "Bravo",
                "Offensive Pass Contributions": 10,
                "Progressive completed": 3,
                "Passes into Box": 1,
                "Key Passes": 2,
                "Completed Passes": 40,
            },
            {
                "team_name": "A",
                "playerName": "Alpha",
                "Offensive Pass Contributions": 10,
                "Progressive completed": 3,
                "Passes into Box": 1,
                "Key Passes": 2,
                "Completed Passes": 40,
            },
            {
                "team_name": "A",
                "playerName": "Charlie",
                "Offensive Pass Contributions": 9,
                "Progressive completed": 99,
                "Passes into Box": 99,
                "Key Passes": 99,
                "Completed Passes": 99,
            },
        ]

        winners = set()
        for seed in range(10):
            rows = list(base)
            random.Random(seed).shuffle(rows)
            winners.add(
                select_top_passer(
                    pd.DataFrame(rows),
                    "A",
                ).selected_name
            )

        self.assertEqual(winners, {"Alpha"})

    def test_sequence_determinism_is_independent_from_row_order(self):
        base = [
            {
                "team_name": "A",
                "sequence_id": "seq-b",
                "outcome": "retained",
                "territorial_progression": 20,
                "action_count": 4,
                "duration": 6,
            },
            {
                "team_name": "A",
                "sequence_id": "seq-a",
                "outcome": "retained",
                "territorial_progression": 20,
                "action_count": 4,
                "duration": 6,
            },
        ]

        winners = set()
        for seed in range(10):
            rows = list(base)
            random.Random(seed).shuffle(rows)
            winners.add(
                select_representative_sequence(
                    pd.DataFrame(rows),
                    "A",
                    category="defensive-transition",
                ).selected_id
            )

        self.assertEqual(winners, {"seq-a"})


    def test_top_passer_accepts_canonical_player_metrics_schema(self):
        frame = pd.DataFrame([
            {
                "playerName": "Zulu",
                "Offensive Pass Contributions": 8,
                "Progressive Passes": 3,
                "Passes into Box": 2,
                "Key Passes": 1,
                "Successful Passes": 30,
            },
            {
                "playerName": "Alpha",
                "Offensive Pass Contributions": 8,
                "Progressive Passes": 4,
                "Passes into Box": 0,
                "Key Passes": 0,
                "Successful Passes": 5,
            },
        ]).set_index("playerName")

        selected = select_top_passer(frame, "A")
        self.assertEqual(selected.selected_name, "Alpha")

    def test_shooting_accepts_canonical_rel08_schema(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "playerName": "Creator",
                "Shot Sequence Involvements": 6,
                "Shot Sequence Shots": 1,
                "Shot Sequence Shot Assists": 5,
                "Shot Sequence Pre-Assists": 2,
            },
            {
                "team_name": "A",
                "playerName": "Shooter",
                "Shot Sequence Involvements": 6,
                "Shot Sequence Shots": 2,
                "Shot Sequence Shot Assists": 0,
                "Shot Sequence Pre-Assists": 0,
            },
        ])

        selected = select_top_shooting_contributor(frame, "A")
        self.assertEqual(selected.selected_name, "Shooter")

    def test_defender_accepts_canonical_defensive_ranking_schema(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "playerName": "Second",
                "unique": 5,
                "interceptions": 1,
                "tackles_won": 2,
                "recoveries": 1,
                "clearances": 1,
                "blocks": 0,
                "fouls": 0,
            },
            {
                "team_name": "A",
                "playerName": "First",
                "unique": 5,
                "interceptions": 2,
                "tackles_won": 0,
                "recoveries": 0,
                "clearances": 0,
                "blocks": 0,
                "fouls": 9,
            },
        ])

        selected = select_top_defender(frame, "A")
        self.assertEqual(selected.selected_name, "First")

    def test_sequence_accepts_canonical_summary_schema(self):
        frame = pd.DataFrame([
            {
                "team_name": "A",
                "sequence_id": "seq-box",
                "terminal_outcome": "retained",
                "milestone_box": True,
                "milestone_shot": False,
                "milestone_goal": False,
                "max_controlled_x": 99,
                "event_count": 20,
                "duration_seconds": 30,
            },
            {
                "team_name": "A",
                "sequence_id": "seq-shot",
                "terminal_outcome": "shot",
                "milestone_box": True,
                "milestone_shot": True,
                "milestone_goal": False,
                "max_controlled_x": 80,
                "event_count": 2,
                "duration_seconds": 3,
            },
        ])

        selected = select_representative_sequence(
            frame,
            "A",
            category="offensive-transition",
        )
        self.assertEqual(selected.selected_id, "seq-shot")

    def test_restart_accepts_canonical_execution_development_destination(self):
        frame = pd.DataFrame([
            {
                "sequence_id": "r-final-third",
                "outcome": "Successful Delivery",
                "development_outcome": "Possession Retained",
                "destination": "Final Third",
            },
            {
                "sequence_id": "r-box",
                "outcome": "Successful Delivery",
                "development_outcome": "Possession Retained",
                "destination": "Center Box",
            },
            {
                "sequence_id": "r-shot",
                "outcome": "Unsuccessful Delivery",
                "development_outcome": "Shot",
                "destination": "Own Third",
            },
        ])

        selected = select_representative_restart(frame, "A")
        self.assertEqual(selected.selected_id, "r-shot")
        self.assertIn("priority='shot'", selected.selection_reason)

    def test_restart_destination_taxonomy_breaks_equal_priority_ties(self):
        frame = pd.DataFrame([
            {
                "sequence_id": "r-final",
                "outcome": "Successful Delivery",
                "development_outcome": "Possession Retained",
                "destination": "Final Third",
            },
            {
                "sequence_id": "r-six",
                "outcome": "Successful Delivery",
                "development_outcome": "Possession Retained",
                "destination": "Center of 6-Yard Box",
            },
        ])

        selected = select_representative_restart(frame, "A")
        self.assertEqual(selected.selected_id, "r-six")

    def test_no_dash_app_or_rendering_dependencies(self):
        source = (
            ROOT / "src" / "reporting" / "selectors.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)

        forbidden = []
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
                    or module.startswith("src.components")
                    or module.startswith("src.visualization")
                ):
                    forbidden.append(module)

        self.assertEqual(forbidden, [])

    def test_invalid_sequence_category_is_rejected(self):
        with self.assertRaises(ValueError):
            select_representative_sequence(
                [{"sequence_id": "x"}],
                "A",
                category="restart",
            )


if __name__ == "__main__":
    unittest.main()
