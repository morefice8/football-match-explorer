import unittest

import pandas as pd

from src.metrics.shot_classification import classify_shots
from src.visualization import shot_placement_plotly


def raw_events_frame():
    return pd.DataFrame([
        {
            # Goal, dead center, mid-height.
            "id": 1,
            "playerName": "Home Striker",
            "team_name": "Home",
            "type_name": "Goal",
            "x": 95.0,
            "y": 50.0,
            "timeMin": 12,
            "Goal mouth y co-ordinate": 50.0,
            "Goal mouth z co-ordinate": 50.0,
        },
        {
            # On-target save, high, toward the left post.
            "id": 2,
            "playerName": "Away Striker",
            "team_name": "Away",
            "type_name": "Attempt Saved",
            "x": 88.0,
            "y": 47.0,
            "timeMin": 30,
            "Goal mouth y co-ordinate": 46.0,
            "Goal mouth z co-ordinate": 90.0,
        },
        {
            # Wide miss - still carries a goal-mouth crossing point per Opta.
            "id": 3,
            "playerName": "Home Winger",
            "team_name": "Home",
            "type_name": "Miss",
            "x": 82.0,
            "y": 60.0,
            "timeMin": 40,
            "Goal mouth y co-ordinate": 40.0,
            "Goal mouth z co-ordinate": 50.0,
        },
        {
            # Blocked shot: ball never reached the goal line, no placement data.
            "id": 4,
            "playerName": "Away Midfielder",
            "team_name": "Away",
            "type_name": "Attempt Saved",
            "Blocked": True,
            "x": 78.0,
            "y": 52.0,
            "timeMin": 55,
        },
    ])


class ShotPlacementPlotlyTests(unittest.TestCase):
    def test_only_shots_with_placement_data_are_plotted(self):
        shots_df = classify_shots(raw_events_frame())

        fig = shot_placement_plotly.plot_shot_placement(
            shots_df,
            home_team="Home",
            away_team="Away",
            hcol="#e96a4a",
            acol="#1597c2",
        )

        placed_points = sum(
            len(trace.x)
            for trace in fig.data
            if "·" in (trace.name or "")
        )
        # The blocked shot (no goal-mouth data) must not appear; the other
        # three (goal, saved, wide miss) all carry a crossing point.
        self.assertEqual(placed_points, 3)

    def test_goal_mouth_scale_converts_to_metres_correctly(self):
        # Left post (45.2) and crossbar (100) are the documented Opta
        # reference points; check the conversion lands on the real frame.
        left_post_x = shot_placement_plotly._y_to_metres(
            shot_placement_plotly.GOAL_Y_LEFT_POST
        )
        crossbar_z = shot_placement_plotly._z_to_metres(100.0)

        self.assertAlmostEqual(left_post_x, -shot_placement_plotly.GOAL_WIDTH_M / 2, places=6)
        self.assertAlmostEqual(crossbar_z, shot_placement_plotly.GOAL_HEIGHT_M, places=6)

    def test_wide_miss_lands_outside_the_frame(self):
        shots_df = classify_shots(raw_events_frame())

        fig = shot_placement_plotly.plot_shot_placement(
            shots_df,
            home_team="Home",
            away_team="Away",
            hcol="#e96a4a",
            acol="#1597c2",
        )

        off_target_trace = next(
            trace for trace in fig.data if trace.name == "Home · Off Target"
        )
        self.assertLess(off_target_trace.x[0], -shot_placement_plotly.GOAL_WIDTH_M / 2)

    def test_high_qualifier_mismatch_is_flagged_in_hover_text(self):
        # Real-world Opta quirk: a header tagged "Miss" + qualifier 74
        # ("High" - hit crossbar or missed over) but whose tracked
        # goal-mouth z still lands inside the frame. The categorical outcome
        # (off target) should win, and the hover text should call out the
        # mismatch instead of silently presenting the point as trustworthy.
        events = pd.concat(
            [
                raw_events_frame(),
                pd.DataFrame(
                    [
                        {
                            "id": 5,
                            "playerName": "J. Enem",
                            "team_name": "Away",
                            "type_name": "Miss",
                            "x": 92.5,
                            "y": 40.3,
                            "timeMin": 86,
                            "Goal mouth y co-ordinate": 49.6,
                            "Goal mouth z co-ordinate": 65.3,
                            "High": 1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        shots_df = classify_shots(events)

        fig = shot_placement_plotly.plot_shot_placement(
            shots_df,
            home_team="Home",
            away_team="Away",
            hcol="#e96a4a",
            acol="#1597c2",
        )

        off_target_trace = next(
            trace for trace in fig.data if trace.name == "Away · Off Target"
        )
        enem_text = next(text for text in off_target_trace.text if "Enem" in text)
        self.assertIn("missed high", enem_text)

    def test_no_placement_data_renders_zero_state_without_error(self):
        fig = shot_placement_plotly.plot_shot_placement(
            pd.DataFrame(),
            home_team="Home",
            away_team="Away",
            hcol="#e96a4a",
            acol="#1597c2",
        )

        annotations = [ann.text for ann in fig.layout.annotations]
        self.assertTrue(
            any("No shot placement data" in text for text in annotations)
        )


if __name__ == "__main__":
    unittest.main()
