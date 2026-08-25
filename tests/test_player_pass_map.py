import unittest
from pathlib import Path

import pandas as pd

from src.metrics import player_pass_map_metrics
from src.visualization import player_plots

def pass_event(
    event_id,
    *,
    outcome="Successful",
    progressive=False,
    key_pass=False,
    assist=False,
    x=20.0,
    y=30.0,
    end_x=60.0,
    end_y=40.0,
):
    return {
        "id": event_id,
        "x": x,
        "y": y,
        "end_x": end_x,
        "end_y": end_y,
        "outcome": outcome,
        "is_progressive": progressive,
        "is_key_pass": key_pass,
        "is_assist": assist,
        "is_into_box": False,
        "timeMin": 12,
        "timeSec": event_id,
        "receiver": "Receiver",
    }

class PlayerPassMapTests(unittest.TestCase):
    def setUp(self):
        self.passes = pd.DataFrame(
            [
                pass_event(1, outcome="Unsuccessful"),
                pass_event(2),
                pass_event(3, progressive=True),
                pass_event(4, progressive=True, key_pass=True),
                pass_event(
                    5,
                    progressive=True,
                    key_pass=True,
                    assist=True,
                ),
            ]
        )

    def test_each_pass_receives_exactly_one_visual_category(self):
        classified = player_pass_map_metrics.classify_player_passes(
            self.passes
        )
        self.assertEqual(
            classified["visual_category"].tolist(),
            [
                "Incomplete",
                "Completed",
                "Progressive",
                "Key Pass",
                "Assist",
            ],
        )
        self.assertEqual(len(classified), len(self.passes))

    def test_kpis_keep_overlap_out_of_chance_creation_count(self):
        profile = player_pass_map_metrics.player_pass_profile(
            self.passes
        )
        self.assertEqual(profile["volume"], 5)
        self.assertEqual(profile["completed"], 4)
        self.assertAlmostEqual(profile["completion_pct"], 80.0)
        self.assertEqual(profile["progressive_completions"], 3)
        self.assertEqual(profile["key_passes"], 2)
        self.assertEqual(profile["assists"], 1)
        self.assertEqual(profile["chance_creation"], 2)

    def test_plot_uses_non_overlapping_category_traces(self):
        fig = player_plots.plot_player_pass_map_plotly(
            self.passes,
            "Player",
            "#1597c2",
            player_jersey="8",
            is_away_team=False,
        )
        line_traces = [
            trace
            for trace in fig.data
            if trace.type == "scattergl" and trace.showlegend
        ]
        self.assertEqual(
            [trace.name for trace in line_traces],
            [
                "Incomplete",
                "Completed",
                "Progressive",
                "Key Pass",
                "Assist",
            ],
        )
        segment_count = sum(
            list(trace.x).count(None)
            for trace in line_traces
        )
        self.assertEqual(segment_count, 5)

    def test_away_player_pass_map_keeps_rel03_coordinates(self):
        passes = pd.DataFrame(
            [
                pass_event(
                    1,
                    x=20.0,
                    y=30.0,
                    end_x=70.0,
                    end_y=60.0,
                )
            ]
        )
        fig = player_plots.plot_player_pass_map_plotly(
            passes,
            "Away Player",
            "#1597c2",
            player_jersey="8",
            is_away_team=True,
        )
        completed = next(
            trace
            for trace in fig.data
            if trace.name == "Completed"
        )
        self.assertEqual(list(completed.x[:2]), [20.0, 70.0])
        self.assertEqual(list(completed.y[:2]), [30.0, 60.0])
        self.assertEqual(list(fig.layout.xaxis.range), [0, 100])
        self.assertEqual(list(fig.layout.yaxis.range), [0, 100])

    def test_callbacks_use_shared_metrics_and_component(self):
        source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn(
            "player_pass_map_metrics.player_pass_profile",
            source,
        )
        self.assertGreaterEqual(
            source.count("player_pass_map_view.panel"),
            2,
        )

    def test_baseline_passes_are_visually_subordinate(self):
        fig = (
            player_plots
            .plot_player_pass_map_plotly(
                self.passes,
                "Player",
                "#1597c2",
                player_jersey="8",
            )
        )

        traces = {
            trace.name: trace
            for trace in fig.data
            if trace.showlegend
        }

        self.assertLess(
            traces["Incomplete"].opacity,
            traces["Completed"].opacity,
        )
        self.assertLess(
            traces["Completed"].opacity,
            traces["Progressive"].opacity,
        )
        self.assertLess(
            traces["Completed"].line.width,
            traces["Key Pass"].line.width,
        )

    def test_player_name_is_not_duplicated_inside_pitch(self):
        fig = (
            player_plots
            .plot_player_pass_map_plotly(
                self.passes,
                "Player",
                "#1597c2",
                player_jersey="8",
            )
        )

        annotation_text = " ".join(
            str(annotation.text)
            for annotation in (
                fig.layout.annotations
                or []
            )
        )

        self.assertNotIn(
            "<b>Player</b>",
            annotation_text,
        )

if __name__ == "__main__":
    unittest.main()
