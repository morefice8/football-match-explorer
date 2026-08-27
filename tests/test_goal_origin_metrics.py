import unittest

import pandas as pd

from src.metrics.goal_origin_metrics import classify_goal_origins


def event(
    event_id,
    team,
    event_type,
    outcome="Successful",
    *,
    minute=10,
    second=0.0,
    player=None,
    period=1,
    throw_in=0,
    corner=0,
    goal_kick=0,
    free_kick=0,
    penalty=0,
    key_pass=0,
    assist=0,
):
    return {
        "id": event_id,
        "eventId": event_id,
        "team_name": team,
        "type_name": event_type,
        "outcome": outcome,
        "playerName": player or f"{team} {event_id}",
        "periodId": period,
        "timeMin": minute,
        "timeSec": float(second),
        "x": 50.0,
        "y": 50.0,
        "end_x": 60.0,
        "end_y": 50.0,
        "ThrowIn": throw_in,
        "Corner taken": corner,
        "Goal kick": goal_kick,
        "Free kick taken": free_kick,
        "Penalty": penalty,
        "Own goal": 0,
        "is_key_pass": key_pass,
        "is_assist": assist,
        "cross": 0,
    }


class GoalOriginMetricTests(unittest.TestCase):

    def test_failed_control_starts_offensive_transition_goal(self):
        df = pd.DataFrame([
            event(1, "Como", "Pass", second=50, player="M. Baturina"),
            event(2, "Como", "Pass", second=52, player="M. Baturina"),
            event(
                3,
                "Como",
                "Ball touch",
                "Unsuccessful",
                second=54,
                player="N. Paz",
            ),
            event(4, "Udinese", "Pass", second=55, player="J. Abankwah"),
            event(5, "Udinese", "Ball recovery", second=56, player="I. Gueye"),
            event(6, "Udinese", "Pass", second=57, player="I. Gueye"),
            event(7, "Udinese", "Pass", second=58, player="J. Ekkelenkamp"),
            event(8, "Udinese", "Goal", minute=11, second=2, player="H. Kamara"),
        ])

        goals = classify_goal_origins(
            df,
            home_team="Udinese",
            away_team="Como",
        )

        self.assertEqual(len(goals), 1)
        goal = goals[0]
        self.assertEqual(goal["possession_origin"], "Opponent turnover")
        self.assertEqual(goal["origin_detail"], "Failed control")
        self.assertEqual(goal["attack_type"], "Offensive Transition")
        self.assertEqual(goal["analysis_tab"], "offensive-transition")
        self.assertEqual(goal["scorer"], "H. Kamara")

    def test_corner_saved_attempt_rebound_goal_is_set_piece(self):
        df = pd.DataFrame([
            event(
                10,
                "Como",
                "Pass",
                minute=53,
                second=8,
                player="Luis Milla",
                corner=1,
                key_pass=1,
            ),
            event(
                11,
                "Como",
                "Attempt Saved",
                minute=53,
                second=9,
                player="T. Chalobah",
            ),
            event(
                12,
                "Udinese",
                "Save",
                minute=53,
                second=9.2,
                player="M. Okoye",
            ),
            event(
                13,
                "Como",
                "Goal",
                minute=53,
                second=10,
                player="T. Douvikas",
            ),
        ])

        goal = classify_goal_origins(
            df,
            home_team="Udinese",
            away_team="Como",
        )[0]

        self.assertEqual(goal["possession_origin"], "Corner")
        self.assertEqual(goal["attack_type"], "Set Piece")
        self.assertEqual(goal["decisive_mechanism"], "Rebound")
        self.assertEqual(goal["analysis_tab"], "set-piece")
        self.assertEqual(goal["shot_creating_pass"], "Luis Milla")

    def test_long_throw_in_possession_keeps_defensive_error_as_mechanism(self):
        rows = [
            event(
                100,
                "Napoli",
                "Pass",
                minute=81,
                second=14,
                player="L. Spinazzola",
                throw_in=1,
            ),
        ]

        passers = [
            "K. De Bruyne",
            "M. Olivera",
            "A. Rrahmani",
            "G. Di Lorenzo",
            "S. McTominay",
            "G. Di Lorenzo",
            "S. McTominay",
            "G. Di Lorenzo",
            "K. De Bruyne",
            "A. Vergara",
        ]

        seconds = [20, 24, 26, 29, 34, 37, 39, 41, 42, 43]
        for offset, (player, second) in enumerate(zip(passers, seconds), start=1):
            rows.append(
                event(
                    100 + offset,
                    "Napoli",
                    "Pass",
                    minute=81,
                    second=second,
                    player=player,
                    key_pass=1 if player == "A. Vergara" else 0,
                )
            )

        rows.extend([
            event(
                200,
                "Genoa",
                "Error",
                minute=81,
                second=43.2,
                player="J. Vasquez",
            ),
            event(
                201,
                "Napoli",
                "Goal",
                minute=81,
                second=45,
                player="K. De Bruyne",
            ),
        ])

        goal = classify_goal_origins(
            pd.DataFrame(rows),
            home_team="Genoa",
            away_team="Napoli",
        )[0]

        self.assertEqual(goal["possession_origin"], "Throw-in")
        self.assertEqual(goal["attack_type"], "Positional Attack")
        self.assertEqual(goal["decisive_mechanism"], "Defensive error")
        self.assertEqual(goal["decisive_player"], "J. Vasquez")
        self.assertEqual(goal["pass_count"], 11)
        self.assertAlmostEqual(goal["possession_duration_seconds"], 31.0)
        self.assertEqual(goal["shot_creating_pass"], "A. Vergara")
        self.assertEqual(goal["analysis_module"], "Possession development")
        self.assertIsNone(goal["analysis_tab"])


if __name__ == "__main__":
    unittest.main()
