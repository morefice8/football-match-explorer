import unittest

import pandas as pd

from src.metrics.shot_classification import classify_shots
from src.visualization import shot_map_plotly


def raw_events_frame():
    return pd.DataFrame([
        {
            "id": 1,
            "playerName": "Home Striker",
            "team_name": "Home",
            "type_name": "Goal",
            "x": 95.0,
            "y": 48.0,
            "timeMin": 12,
        },
        {
            "id": 2,
            "playerName": "Home Winger",
            "team_name": "Home",
            "type_name": "Miss",
            "x": 80.0,
            "y": 60.0,
            "timeMin": 30,
        },
        {
            "id": 3,
            "playerName": "Away Striker",
            "team_name": "Away",
            "type_name": "Attempt Saved",
            "Blocked": True,
            "x": 88.0,
            "y": 52.0,
            "timeMin": 45,
        },
        {
            "id": 4,
            "playerName": "Away Midfielder",
            "team_name": "Away",
            "type_name": "Attempt Saved",
            "x": 70.0,
            "y": 40.0,
            "timeMin": 60,
        },
        {
            "id": 5,
            "playerName": "Own Goal Scorer",
            "team_name": "Home",
            "type_name": "Goal",
            "Own goal": True,
            "x": 5.0,
            "y": 50.0,
            "timeMin": 70,
        },
        {
            "id": 6,
            "playerName": "Home Midfielder",
            "team_name": "Home",
            "type_name": "Pass",
            "x": 40.0,
            "y": 40.0,
            "timeMin": 20,
        },
    ])


class ShotMapPlotlyTests(unittest.TestCase):
    def test_plots_only_counted_shots_per_team(self):
        shots_df = classify_shots(raw_events_frame())

        fig = shot_map_plotly.plot_shot_map(
            shots_df,
            home_team="Home",
            away_team="Away",
            hcol="#e96a4a",
            acol="#1597c2",
        )

        total_points = sum(
            len(trace.x)
            for trace in fig.data
            if "·" in (trace.name or "")
        )
        # 1 home goal + 1 home miss + 1 away saved (on target) + 1 away
        # blocked = 4 counted shots. The own goal and the plain pass must
        # not appear.
        self.assertEqual(total_points, 4)

        names = {trace.name for trace in fig.data}
        self.assertIn("Home", names)
        self.assertIn("Away", names)
        self.assertIn("Home · Goal", names)
        self.assertIn("Away · Blocked", names)

    def test_unresolved_outcomes_are_shown_not_hidden(self):
        # Contradictory evidence: Attempt Saved marked both blocked and
        # off-target-saved, which classify_shot_event resolves to "unknown"
        # rather than guessing.
        ambiguous_row = pd.DataFrame([{
            "id": 7,
            "playerName": "Ambiguous Shooter",
            "team_name": "Home",
            "type_name": "Attempt Saved",
            "Blocked": True,
            "Keeper Saved": True,
            "x": 85.0,
            "y": 50.0,
            "timeMin": 80,
        }])
        events = pd.concat([raw_events_frame(), ambiguous_row], ignore_index=True)
        shots_df = classify_shots(events)

        fig = shot_map_plotly.plot_shot_map(
            shots_df,
            home_team="Home",
            away_team="Away",
            hcol="#e96a4a",
            acol="#1597c2",
        )

        unresolved = next(
            trace for trace in fig.data if trace.name == "Home · Unclear"
        )
        self.assertEqual(len(unresolved.x), 1)
        self.assertEqual(unresolved.marker.color, shot_map_plotly.UNKNOWN_COLOR)

    def test_empty_input_renders_zero_state_without_error(self):
        fig = shot_map_plotly.plot_shot_map(
            pd.DataFrame(),
            home_team="Home",
            away_team="Away",
            hcol="#e96a4a",
            acol="#1597c2",
        )

        annotations = [ann.text for ann in fig.layout.annotations]
        self.assertTrue(
            any("No shots recorded" in text for text in annotations)
        )


if __name__ == "__main__":
    unittest.main()
