import unittest

import pandas as pd

from src.metrics import defensive_metrics
from test_report11_real_shaped_fixture import _processed_fixture


def _frame(rows):
    base = {
        "team_name": "Home",
        "x": 30.0,
        "y": 50.0,
        "playerName": "A",
        "Mapped Jersey Number": 4,
        "outcome": "Successful",
    }
    frame = pd.DataFrame([{**base, **row} for row in rows])
    frame["id"] = range(len(frame))
    return frame


class DefensiveActionFilterTests(unittest.TestCase):
    def test_blocked_pass_counts_whatever_its_case(self):
        frame = _frame(
            [
                {"type_name": "Blocked Pass"},
                {"type_name": "Blocked pass"},
                {"type_name": " blocked pass "},
                {"type_name": "Pass"},
            ]
        )
        actions = defensive_metrics.get_defensive_actions(frame)
        self.assertEqual(len(actions), 3)

    def test_explicit_type_list_is_case_insensitive_too(self):
        frame = _frame([{"type_name": "Blocked Pass"}, {"type_name": "Tackle"}])
        actions = defensive_metrics.get_defensive_actions(frame, ["blocked pass"])
        self.assertEqual(actions["type_name"].tolist(), ["Blocked Pass"])

    def test_only_fouls_committed_count(self):
        frame = _frame(
            [
                {"type_name": "Foul", "outcome": "Unsuccessful"},
                {"type_name": "Foul", "outcome": "Unsuccessful"},
                {"type_name": "Foul", "outcome": "Successful"},
            ]
        )
        actions = defensive_metrics.get_defensive_actions(frame)
        self.assertEqual(len(actions), 2)
        self.assertTrue(actions["outcome"].eq("Unsuccessful").all())

    def test_fouls_stay_when_outcome_is_missing(self):
        frame = _frame([{"type_name": "Foul"}, {"type_name": "Foul"}]).drop(columns=["outcome"])
        self.assertEqual(len(defensive_metrics.get_defensive_actions(frame)), 2)

    def test_other_actions_keep_their_outcome_untouched(self):
        # Only fouls are filtered on outcome: an unsuccessful tackle is still a tackle.
        frame = _frame(
            [
                {"type_name": "Tackle", "outcome": "Unsuccessful"},
                {"type_name": "Tackle", "outcome": "Successful"},
                {"type_name": "Clearance", "outcome": "Successful"},
            ]
        )
        self.assertEqual(len(defensive_metrics.get_defensive_actions(frame)), 3)

    def test_aerial_is_still_limited_to_the_defensive_third(self):
        frame = _frame(
            [
                {"type_name": "Aerial", "x": 20.0},
                {"type_name": "Aerial", "x": 60.0},
            ]
        )
        self.assertEqual(len(defensive_metrics.get_defensive_actions(frame)), 1)

    def test_real_shaped_match_counts_what_the_data_holds(self):
        frame, _ = _processed_fixture()
        actions = defensive_metrics.get_defensive_actions(frame)
        for team, group in frame.groupby("team_name"):
            mine = actions[actions["team_name"] == team]
            fouls = group[group["type_name"] == "Foul"]
            blocked = group[group["type_name"] == "Blocked Pass"]
            self.assertEqual(
                int((mine["type_name"] == "Foul").sum()),
                int((fouls["outcome"] == "Unsuccessful").sum()),
                team,
            )
            self.assertEqual(int((mine["type_name"] == "Blocked Pass").sum()), len(blocked), team)


if __name__ == "__main__":
    unittest.main()
