import unittest

import pandas as pd

from src.metrics import buildup_metrics
from src.metrics.restart_metrics import (
    classify_restart_delivery,
    extract_restart_sequences,
)


def row(
    event_id,
    *,
    team="Home",
    event_type="Pass",
    outcome="Successful",
    second=0.0,
    x=30.0,
    y=50.0,
    end_x=40.0,
    end_y=50.0,
    length=10.0,
    corner=0,
    free_kick=0,
    throw_in=0,
    goal_kick=0,
    cross=0,
):
    return {
        "id": event_id,
        "eventId": event_id,
        "team_name": team,
        "type_name": event_type,
        "outcome": outcome,
        "x": x,
        "y": y,
        "end_x": end_x,
        "end_y": end_y,
        "playerName": f"{team} {event_id}",
        "Mapped Jersey Number": event_id,
        "timeMin": 81,
        "timeSec": second,
        "periodId": 2,
        "lb": 0,
        "Length": length,
        "cross": cross,
        "Corner taken": corner,
        "Free kick taken": free_kick,
        "Freekick taken": 0,
        "ThrowIn": throw_in,
        "Goal kick": goal_kick,
        "Goal kick taken": 0,
        "Penalty": 0,
        "Own goal": 0,
        "Blocked": 0,
        "Goal mouth y co-ordinate": 50.0,
        "Right footed": 1,
        "Left footed": 0,
        "In-swinger": 0,
        "Out-swinger": 0,
        "Straight": 0,
        "positional_role": "DF",
        "receiver": None,
        "receiver_jersey_number": None,
    }


class RestartExecutionSemanticsTests(unittest.TestCase):

    def test_throw_in_delivery_has_distance_semantics(self):
        labels = []
        for event_id, length in enumerate((10.0, 20.0, 35.0), start=1):
            event = pd.Series(row(event_id, throw_in=1, length=length))
            labels.append(classify_restart_delivery(event, "Throw-in"))

        self.assertEqual(
            labels,
            ["Short Throw", "Medium Throw", "Long Throw"],
        )

    def test_goal_kick_delivery_has_distance_semantics(self):
        event = pd.Series(row(1, goal_kick=1, length=42.0))
        self.assertEqual(
            classify_restart_delivery(event, "Goal Kick"),
            "Long Goal Kick",
        )

    def test_corner_delivery_distinguishes_cross_and_short(self):
        direct = pd.Series(row(1, corner=1, cross=1, length=28.0))
        short = pd.Series(row(2, corner=1, cross=0, length=7.0))

        self.assertEqual(
            classify_restart_delivery(direct, "Corner"),
            "Direct Cross",
        )
        self.assertEqual(
            classify_restart_delivery(short, "Corner"),
            "Short Corner",
        )

    def test_restart_sequence_stops_at_delivery(self):
        df = pd.DataFrame([
            row(1, throw_in=1, second=14.0, x=41.0, end_x=43.0),
            row(2, second=18.0, x=43.0, end_x=58.0),
            row(
                3,
                event_type="Goal",
                second=45.0,
                x=90.0,
                end_x=100.0,
                length=0.0,
            ),
        ])

        sequence = extract_restart_sequences(df, "Home")[0]

        self.assertEqual(sequence["eventId"].tolist(), [1])
        self.assertEqual(
            sequence.iloc[0]["restart_execution_outcome"],
            "Successful Delivery",
        )

    def test_generic_out_is_not_a_canonical_buildup_origin(self):
        df = pd.DataFrame([
            row(
                1,
                team="Away",
                event_type="Out",
                outcome="Unsuccessful",
                second=10.0,
                x=70.0,
                end_x=70.0,
                length=0.0,
            ),
            row(
                2,
                team="Home",
                second=12.0,
                x=30.0,
                end_x=40.0,
            ),
        ])

        result = buildup_metrics.find_buildup_sequences(
            df,
            attacking_team="Home",
            defending_team="Away",
            metric_to_analyze="buildup_phase",
            triggers_buildups=(
                buildup_metrics.DEFAULT_FIRST_PHASE_BUILDUP_TRIGGERS
            ),
        )

        self.assertTrue(result.empty)

    def test_de_bruyne_style_throw_in_starts_buildup_but_goal_is_outside_first_phase(self):
        df = pd.DataFrame([
            row(
                1,
                team="Away",
                event_type="Out",
                outcome="Unsuccessful",
                second=12.0,
                x=59.0,
                end_x=59.0,
                length=0.0,
            ),
            row(
                2,
                throw_in=1,
                second=14.0,
                x=41.0,
                end_x=43.0,
                length=8.0,
            ),
            row(
                3,
                second=18.0,
                x=43.0,
                end_x=58.0,
                length=18.0,
            ),
            row(
                4,
                event_type="Goal",
                second=45.0,
                x=90.0,
                end_x=100.0,
                length=0.0,
            ),
        ])

        result = buildup_metrics.find_buildup_sequences(
            df,
            attacking_team="Home",
            defending_team="Away",
            metric_to_analyze="buildup_phase",
            triggers_buildups=(
                buildup_metrics.DEFAULT_FIRST_PHASE_BUILDUP_TRIGGERS
            ),
        )

        self.assertFalse(result.empty)
        self.assertEqual(
            result.iloc[0]["type_of_initial_trigger"],
            "Throw-in",
        )
        self.assertEqual(result["eventId"].tolist(), [2, 3])
        self.assertFalse(result["type_name"].eq("Goal").any())
        self.assertEqual(
            result.iloc[-1]["termination_reason"],
            "opposition_half_reached",
        )

    def test_free_kick_and_goal_kick_are_valid_first_phase_origins(self):
        for flag_name, expected in (
            ("free_kick", "Free Kick"),
            ("goal_kick", "Goal Kick"),
        ):
            with self.subTest(restart=expected):
                kwargs = {flag_name: 1}
                df = pd.DataFrame([
                    row(
                        1,
                        second=1.0,
                        x=25.0,
                        end_x=35.0,
                        **kwargs,
                    ),
                    row(
                        2,
                        second=4.0,
                        x=35.0,
                        end_x=55.0,
                    ),
                ])

                result = buildup_metrics.find_buildup_sequences(
                    df,
                    attacking_team="Home",
                    defending_team="Away",
                    metric_to_analyze="buildup_phase",
                    triggers_buildups=(
                        buildup_metrics.DEFAULT_FIRST_PHASE_BUILDUP_TRIGGERS
                    ),
                )

                self.assertFalse(result.empty)
                self.assertEqual(
                    result.iloc[0]["type_of_initial_trigger"],
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
