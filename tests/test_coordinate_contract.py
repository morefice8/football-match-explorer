import math
import unittest

import pandas as pd

from src.visualization.coordinate_contract import (
    ATTACKING_DIRECTION_LABEL,
    orient_point,
    orient_segment,
)
from src.visualization import (
    buildup_plotly,
    defensive_transitions_plotly,
    formation_plotly,
    offensive_transitions_plotly,
    pass_plotly,
    player_plots,
)


class CoordinateContractTests(unittest.TestCase):
    def assert_attacking_direction(self, fig):
        texts = [str(annotation.text) for annotation in (fig.layout.annotations or [])]
        self.assertTrue(
            any(ATTACKING_DIRECTION_LABEL in text for text in texts),
            texts,
        )

    def test_point_orientation_is_identity_for_home_and_away(self):
        self.assertEqual(orient_point(20.0, 30.0, is_away=False), (20.0, 30.0))
        self.assertEqual(orient_point(20.0, 30.0, is_away=True), (20.0, 30.0))

    def test_segment_orientation_is_identity_for_home_and_away(self):
        expected = (20.0, 30.0, 70.0, 60.0)
        self.assertEqual(
            orient_segment(20.0, 30.0, 70.0, 60.0, is_away=False),
            expected,
        )
        self.assertEqual(
            orient_segment(20.0, 30.0, 70.0, 60.0, is_away=True),
            expected,
        )

    def test_nan_coordinates_are_not_rewritten(self):
        x, y = orient_point(float('nan'), 30.0, is_away=True)
        self.assertTrue(math.isnan(x))
        self.assertEqual(y, 30.0)

    def test_final_third_entries_do_not_reverse_away_axes(self):
        entries = pd.DataFrame([
            {
                'entry_type': 'Pass',
                'x': 60.0,
                'y': 25.0,
                'end_x': 72.0,
                'end_y': 35.0,
                'playerName': 'Away Player',
                'timeMin': 10,
                'receiver': 'Receiver',
                'receiver_confidence': 'high',
                'final_third_channel': 'Right',
                'destination_zone': 'Final Third',
            }
        ])
        stats = {
            'total_final_third': 1,
            'pass_entries': 1,
            'carry_entries': 0,
            'zone14': 0,
            'hs_left': 0,
            'hs_right': 0,
        }

        fig = pass_plotly.plot_final_third_entries_plotly(
            entries,
            stats,
            'Away',
            '#0b88a8',
            is_away=True,
        )

        self.assertEqual(list(fig.layout.xaxis.range), [-2, 102])
        self.assertEqual(list(fig.layout.yaxis.range), [-2, 102])
        self.assertNotEqual(fig.layout.xaxis.autorange, 'reversed')
        self.assertNotEqual(fig.layout.yaxis.autorange, 'reversed')
        self.assertEqual(list(fig.data[0].x[:2]), [60.0, 72.0])
        self.assertEqual(list(fig.data[0].y[:2]), [25.0, 35.0])
        self.assert_attacking_direction(fig)

    def test_mean_positions_keep_away_geometry(self):
        touches = pd.DataFrame([
            {'x': 24.0, 'y': 35.0},
            {'x': 28.0, 'y': 40.0},
        ])
        players = pd.DataFrame([
            {
                'median_x': 26.0,
                'median_y': 38.0,
                'Mapped Jersey Number': 8,
                'playerName': 'Midfielder',
                'action_count': 12,
                'Is Starter': True,
            }
        ])

        fig = formation_plotly.plot_mean_positions_plotly(
            touches,
            players,
            '#0b88a8',
            is_away=True,
        )

        self.assertEqual(list(fig.layout.xaxis.range), [0, 100])
        self.assertEqual(list(fig.layout.yaxis.range), [0, 100])
        player_trace = next(trace for trace in fig.data if trace.name == 'Midfielder')
        self.assertEqual(list(player_trace.x), [26.0])
        self.assertEqual(list(player_trace.y), [38.0])
        self.assert_attacking_direction(fig)

    def test_offensive_recovery_heatmap_keeps_away_recovery_location(self):
        sequence = pd.DataFrame([
            {'x': 20.0, 'y': 30.0, 'sequence_outcome_type': 'Shot'}
        ])
        fig = offensive_transitions_plotly.plot_recovery_heatmap_on_pitch(
            [sequence],
            is_away=True,
        )

        recovery_trace = next(trace for trace in fig.data if trace.name == 'Recovery')
        self.assertEqual(list(recovery_trace.x), [20.0])
        self.assertEqual(list(recovery_trace.y), [30.0])
        self.assert_attacking_direction(fig)

    def test_defensive_loss_heatmap_maps_loss_into_transition_team_frame(self):
            import pandas as pd

            from src.visualization import (
                defensive_transitions_plotly,
            )

            sequence = pd.DataFrame(
                [
                    {
                        "loss_x": 18.0,
                        "loss_y": 74.0,
                        "x": 82.0,
                        "y": 26.0,
                        "type_of_initial_loss":
                            "Unsuccessful Pass",
                        "loss_zone":
                            "Middle third",
                        "sequence_outcome_type":
                            "Retained",
                    }
                ]
            )

            fig = (
                defensive_transitions_plotly
                .plot_loss_heatmap_on_pitch(
                    [sequence],
                    losing_team_is_away=True,
                )
            )

            loss_trace = next(
                trace
                for trace in fig.data
                if trace.name
                == "Possession loss"
            )

            # loss_x/loss_y belong to the team that lost possession.
            # Defensive-transition analysis is displayed in the opponent /
            # transition-team frame, so the loss point is rotated once:
            # (18, 74) -> (82, 26).
            self.assertEqual(
                list(loss_trace.x),
                [82.0],
            )
            self.assertEqual(
                list(loss_trace.y),
                [26.0],
            )

    def test_player_pass_map_does_not_mirror_away_segment(self):
        passes = pd.DataFrame([
            {
                'x': 20.0,
                'y': 30.0,
                'end_x': 70.0,
                'end_y': 60.0,
                'outcome': 'Successful',
                'is_assist': False,
                'is_key_pass': False,
                'is_into_box': False,
                'is_progressive': False,
                'timeMin': 12,
            }
        ])

        fig = player_plots.plot_player_pass_map_plotly(
            passes,
            'Away Player',
            '#0b88a8',
            player_jersey='8',
            is_away_team=True,
        )

        pass_trace = next(trace for trace in fig.data if trace.name == 'Completed')
        self.assertEqual(list(pass_trace.x[:2]), [20.0, 70.0])
        self.assertEqual(list(pass_trace.y[:2]), [30.0, 60.0])
        self.assert_attacking_direction(fig)

    def test_buildup_sequence_away_axes_remain_left_to_right(self):
        sequence = pd.DataFrame([
            {
                'x': 15.0,
                'y': 25.0,
                'end_x': 55.0,
                'end_y': 45.0,
                'type_name': 'Pass',
                'outcome': 'Successful',
                'playerName': 'Player',
                'Mapped Jersey Number': 4,
                'sequence_outcome_type': 'Progressed',
                'buildup_pass_count': 1,
            }
        ])

        fig = buildup_plotly.plot_buildup_sequence_plotly(
            sequence,
            '#0b88a8',
            is_away=True,
        )

        self.assertEqual(list(fig.layout.xaxis.range), [0, 100])
        self.assertEqual(list(fig.layout.yaxis.range), [0, 100])
        self.assert_attacking_direction(fig)


if __name__ == '__main__':
    unittest.main()
