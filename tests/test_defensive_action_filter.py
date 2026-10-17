import unittest

import pandas as pd

from src.metrics import defensive_metrics, player_metrics
from src.visualization import player_plots
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


class PlayerLevelDefensiveTests(unittest.TestCase):
    """The per-player views share the same rule: any case for Blocked Pass, committed fouls only."""

    def _actions(self):
        return _frame(
            [
                {"type_name": "Blocked Pass"},
                {"type_name": "Blocked Pass"},
                {"type_name": "Foul", "outcome": "Unsuccessful"},
                {"type_name": "Foul", "outcome": "Successful"},
                {"type_name": "Tackle", "outcome": "Successful"},
                {"type_name": "Tackle", "outcome": "Unsuccessful"},
                {"type_name": "Pass"},
            ]
        )

    def test_mask_ignores_case_and_keeps_committed_fouls_only(self):
        frame = self._actions()
        mask = defensive_metrics.defensive_type_mask(frame, defensive_metrics.PLAYER_DEFENSIVE_TYPES)
        self.assertEqual(frame.loc[mask, "type_name"].value_counts().to_dict(), {"Blocked Pass": 2, "Tackle": 2, "Foul": 1})

    def test_action_table_counts_blocked_passes_and_committed_fouls(self):
        table = player_metrics.calculate_defensive_action_rates(self._actions()).set_index("Action")
        self.assertEqual(int(table.loc["Blocked Pass", "Total"]), 2)
        self.assertEqual(table.loc["Blocked Pass", "Success Rate"], "100.0%")
        self.assertEqual(int(table.loc["Foul", "Total"]), 1)
        self.assertEqual((int(table.loc["Tackle", "Successful"]), int(table.loc["Tackle", "Total"])), (1, 2))

    def test_action_map_draws_blocked_passes_and_no_foul_suffered(self):
        frame = self._actions().assign(timeMin=10, timeSec=0)
        figure = player_plots.plot_player_defensive_actions_plotly(frame, "Player", "#e96a4a", False)
        names = [trace.name for trace in figure.data]
        self.assertIn("Blocked Pass (Successful)", names)
        self.assertIn("Foul (Unsuccessful)", names)
        self.assertNotIn("Foul (Successful)", names)

    def test_defensive_block_data_counts_blocked_passes(self):
        frame = self._actions().assign(eventId=lambda f: f["id"], **{"Is Starter": True})
        actions, per_player = defensive_metrics.get_defensive_block_data(frame, "Home")
        self.assertEqual(len(actions), 5)  # 2 blocked passes, 1 committed foul, 2 tackles
        self.assertEqual(int(per_player["action_count"].iloc[0]), 5)


if __name__ == "__main__":
    unittest.main()
