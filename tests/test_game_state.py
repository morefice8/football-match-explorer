import unittest

import pandas as pd

from src.metrics import game_state
from src.metrics.shot_classification import classify_shots


def raw_events_frame():
    return pd.DataFrame([
        {
            # Home scores first, minute 10.
            "id": 1,
            "playerName": "Home Striker",
            "team_name": "Home",
            "type_name": "Goal",
            "x": 95.0,
            "y": 50.0,
            "timeMin": 10,
            "timeSec": 0,
        },
        {
            # Home shot 5 seconds after their own goal -- by now the goal
            # has already happened, so this reflects the new 1-0 state.
            "id": 2,
            "playerName": "Home Striker",
            "team_name": "Home",
            "type_name": "Miss",
            "x": 90.0,
            "y": 50.0,
            "timeMin": 10,
            "timeSec": 5,
        },
        {
            # Away shot at minute 20, while trailing 0-1.
            "id": 3,
            "playerName": "Away Striker",
            "team_name": "Away",
            "type_name": "Attempt Saved",
            "x": 88.0,
            "y": 47.0,
            "timeMin": 20,
            "timeSec": 0,
        },
        {
            # Away own goal at minute 30 -> Home now leads 2-0.
            "id": 4,
            "playerName": "Away Defender",
            "team_name": "Away",
            "type_name": "Goal",
            "Own goal": True,
            "x": 5.0,
            "y": 50.0,
            "timeMin": 30,
            "timeSec": 0,
        },
        {
            # Home shot at minute 40, while leading 2-0.
            "id": 5,
            "playerName": "Home Winger",
            "team_name": "Home",
            "type_name": "Miss",
            "x": 82.0,
            "y": 60.0,
            "timeMin": 40,
            "timeSec": 0,
        },
    ])


class GameStateTests(unittest.TestCase):
    def test_scoring_shot_reflects_state_before_its_own_goal(self):
        # The goal event itself (id 1) should not count its own goal toward
        # its own prior state -- it's evaluated at 0-0, not 1-0.
        shots_df = classify_shots(raw_events_frame())
        tagged = game_state.add_game_state_column(shots_df, "Home", "Away")

        goal_shot = tagged[tagged["id"] == 1].iloc[0]
        self.assertEqual(goal_shot["game_state"], game_state.GAME_STATE_DRAWING)

    def test_shot_shortly_after_teammates_goal_reflects_new_state(self):
        shots_df = classify_shots(raw_events_frame())
        tagged = game_state.add_game_state_column(shots_df, "Home", "Away")

        home_shot_after_goal = tagged[tagged["id"] == 2].iloc[0]
        self.assertEqual(home_shot_after_goal["game_state"], game_state.GAME_STATE_LEADING)

    def test_trailing_team_shot(self):
        shots_df = classify_shots(raw_events_frame())
        tagged = game_state.add_game_state_column(shots_df, "Home", "Away")

        away_shot = tagged[tagged["id"] == 3].iloc[0]
        self.assertEqual(away_shot["game_state"], game_state.GAME_STATE_TRAILING)

    def test_own_goal_benefits_the_opponent(self):
        shots_df = classify_shots(raw_events_frame())
        tagged = game_state.add_game_state_column(shots_df, "Home", "Away")

        home_shot_after_own_goal = tagged[tagged["id"] == 5].iloc[0]
        self.assertEqual(home_shot_after_own_goal["game_state"], game_state.GAME_STATE_LEADING)

    def test_empty_input_returns_empty_frame_with_column(self):
        result = game_state.add_game_state_column(pd.DataFrame(), "Home", "Away")
        self.assertIn("game_state", result.columns)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
