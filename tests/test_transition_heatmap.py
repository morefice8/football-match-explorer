import unittest

import pandas as pd

from src.metrics import transition_heatmap_metrics
from src.visualization import (
    defensive_transitions_plotly,
    offensive_transitions_plotly,
)


def _sequence(
    *,
    start_x,
    start_y,
    loss_x,
    loss_y,
    end_x,
    duration,
    terminal_outcome="retained",
):
    return pd.DataFrame([
        {
            "x": start_x,
            "y": start_y,
            "end_x": end_x,
            "end_y": start_y,
            "loss_x": loss_x,
            "loss_y": loss_y,
            "timeMin_at_loss": 10,
            "timeSec_at_loss": 0,
            "timeMin": 10,
            "timeSec": 0,
            "type_name": "Pass",
            "type_of_initial_loss": "Pass",
            "loss_zone": "Middle third",
            "sequence_outcome_type": "Retained",
            "terminal_outcome": terminal_outcome,
        },
        {
            "x": end_x,
            "y": start_y,
            "end_x": end_x,
            "end_y": start_y,
            "loss_x": loss_x,
            "loss_y": loss_y,
            "timeMin_at_loss": 10,
            "timeSec_at_loss": 0,
            "timeMin": 10,
            "timeSec": duration,
            "type_name": (
                "Miss"
                if terminal_outcome == "shot"
                else "Pass"
            ),
            "type_of_initial_loss": "Pass",
            "loss_zone": "Middle third",
            "sequence_outcome_type": (
                "Shots"
                if terminal_outcome == "shot"
                else "Retained"
            ),
            "terminal_outcome": terminal_outcome,
        },
    ])


class TransitionHeatmapTests(unittest.TestCase):
    def setUp(self):
        self.sequences = [
            _sequence(
                start_x=20,
                start_y=20,
                loss_x=15,
                loss_y=20,
                end_x=72,
                duration=8,
            ),
            _sequence(
                start_x=22,
                start_y=22,
                loss_x=16,
                loss_y=22,
                end_x=50,
                duration=12,
                terminal_outcome="shot",
            ),
            _sequence(
                start_x=70,
                start_y=70,
                loss_x=72,
                loss_y=70,
                end_x=75,
                duration=10,
            ),
        ]

    def test_loss_and_recovery_share_grid_palette_and_scale(self):
        loss = defensive_transitions_plotly.plot_loss_heatmap_on_pitch(
            self.sequences,
            grid_size=6,
        )
        recovery = offensive_transitions_plotly.plot_recovery_heatmap_on_pitch(
            self.sequences,
            grid_size=6,
        )

        loss_heatmap = next(trace for trace in loss.data if trace.type == "heatmap")
        recovery_heatmap = next(trace for trace in recovery.data if trace.type == "heatmap")

        self.assertEqual(list(loss_heatmap.x), list(recovery_heatmap.x))
        self.assertEqual(list(loss_heatmap.y), list(recovery_heatmap.y))
        self.assertEqual(loss_heatmap.zmin, recovery_heatmap.zmin)
        self.assertEqual(loss_heatmap.zmax, recovery_heatmap.zmax)
        self.assertEqual(loss_heatmap.colorscale, recovery_heatmap.colorscale)

    def test_kpis_count_duration_and_final_third_or_shot(self):
        metrics = transition_heatmap_metrics.transition_kpis(self.sequences)
        self.assertEqual(metrics["transition_count"], 3)
        self.assertAlmostEqual(metrics["median_duration_seconds"], 10.0)
        self.assertAlmostEqual(metrics["final_third_or_shot_pct"], 100.0)

    def test_click_cell_filters_sequence_set(self):
        click_data = {
            "points": [{
                "customdata": [
                    0.0,
                    16.6667,
                    16.6667,
                    33.3333,
                    2,
                ]
            }]
        }
        cell = transition_heatmap_metrics.cell_from_click(click_data)
        selected = transition_heatmap_metrics.filter_sequences_by_cell(
            self.sequences,
            cell,
            location_kind="loss",
        )
        self.assertEqual(len(selected), 2)

    def test_heatmap_cells_expose_click_customdata(self):
        fig = offensive_transitions_plotly.plot_recovery_heatmap_on_pitch(
            self.sequences
        )
        heatmap = next(trace for trace in fig.data if trace.type == "heatmap")
        self.assertIsNotNone(heatmap.customdata)
        self.assertEqual(len(heatmap.customdata[0][0]), 5)

    def test_point_trace_names_remain_compatible(self):
        loss = defensive_transitions_plotly.plot_loss_heatmap_on_pitch(self.sequences)
        recovery = offensive_transitions_plotly.plot_recovery_heatmap_on_pitch(self.sequences)
        self.assertTrue(any(trace.name == "Possession loss" for trace in loss.data))
        self.assertTrue(any(trace.name == "Recovery" for trace in recovery.data))


    def test_defensive_loss_heatmap_uses_transition_team_frame(self):
        sequence = _sequence(
            start_x=80,
            start_y=70,
            loss_x=20,
            loss_y=30,
            end_x=85,
            duration=6,
        )

        fig = (
            defensive_transitions_plotly
            .plot_loss_heatmap_on_pitch(
                [sequence]
            )
        )

        points = next(
            trace
            for trace in fig.data
            if trace.name
            == "Possession loss"
        )

        self.assertAlmostEqual(
            float(points.x[0]),
            80.0,
        )
        self.assertAlmostEqual(
            float(points.y[0]),
            70.0,
        )

    def test_defensive_loss_cell_filter_uses_same_transition_frame(self):
        sequence = _sequence(
            start_x=80,
            start_y=70,
            loss_x=20,
            loss_y=30,
            end_x=85,
            duration=6,
        )

        cell = {
            "x0": 66.6667,
            "x1": 83.3334,
            "y0": 66.6667,
            "y1": 83.3334,
        }

        selected = (
            transition_heatmap_metrics
            .filter_sequences_by_cell(
                [sequence],
                cell,
                location_kind="loss",
                loss_to_transition_frame=True,
            )
        )

        self.assertEqual(
            len(selected),
            1,
        )

if __name__ == "__main__":
    unittest.main()
