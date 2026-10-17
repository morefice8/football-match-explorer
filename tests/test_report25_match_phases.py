"""REPORT-25: automatic match phases."""

from __future__ import annotations

import unittest

import pandas as pd

from src.metrics import match_clock as mc
from src.metrics import match_phases as mp


def _frame(extra_rows=(), end=(2, 93, 0)):
    """Skeleton: both periods start/end; rows are (period, min, sec, team, type, player, qualifiers)."""
    rows = [
        (1, 0, 0, "H", "Start", None), (1, 47, 0, "H", "End", None),
        (2, 45, 0, "H", "Start", None), (end[0], end[1], end[2], "H", "End", None),
    ]
    rows += list(extra_rows)
    frame = pd.DataFrame(rows, columns=["periodId", "timeMin", "timeSec", "team_name", "type_name", "playerName"])
    return frame


def _goals(*items):
    """items: (clock_minute, clock_second, team, period)."""
    frame = _frame()
    clock = mc.PlayClock.from_events(frame)
    rows = []
    for minute, second, team, period in items:
        rows.append({"play_seconds": clock.play_seconds(period, minute * 60 + second), "clock": mc.format_clock(minute * 60 + second), "team": team, "player": "x"})
    return pd.DataFrame(rows, columns=["play_seconds", "clock", "team", "player"])


def _sub(period, minute, second, team, on, off):
    return [(period, minute, second, team, "Player off", off), (period, minute, second, team, "Player on", on)]


class MatchPhaseTests(unittest.TestCase):
    def test_a_match_without_events_is_a_single_phase(self):
        phases = mp.build_match_phases(_frame(), "H", "A", goals=_goals())
        self.assertEqual(len(phases), 1)
        row = phases.iloc[0]
        self.assertEqual(row["trigger"], "kickoff")
        self.assertEqual((row["home_state"], row["away_state"]), ("level", "level"))
        self.assertAlmostEqual(row["play_minutes"], 47 + 48)
        self.assertEqual(row["label"], "0'-93'")

    def test_goals_cut_phases_and_set_the_score_state(self):
        phases = mp.build_match_phases(_frame(), "H", "A", goals=_goals((10, 0, "H", 1), (30, 0, "A", 1), (60, 0, "A", 2)))
        self.assertEqual(list(phases["trigger"]), ["kickoff", "goal", "goal", "goal"])
        self.assertEqual(list(zip(phases["home_score"], phases["away_score"])), [(0, 0), (1, 0), (1, 1), (1, 2)])
        self.assertEqual(list(phases["home_state"]), ["level", "leading", "level", "trailing"])
        self.assertEqual(list(phases["away_state"]), ["level", "trailing", "level", "leading"])
        self.assertEqual(int(phases["goals_inside"].sum()), 0)

    def test_two_goals_closer_than_the_minimum_merge_and_count_as_inside(self):
        phases = mp.build_match_phases(_frame(), "H", "A", goals=_goals((10, 0, "H", 1), (11, 0, "A", 1)))
        self.assertEqual(len(phases), 2)  # the second goal is too close to the first cut
        self.assertEqual(int(phases["goals_inside"].iloc[1]), 1)
        self.assertEqual((phases["home_score_end"].iloc[1], phases["away_score_end"].iloc[1]), (1, 1))

    def test_substitution_clusters_cut_once_and_list_who_came_on(self):
        subs = _sub(2, 60, 0, "H", "Lang", "Giovane") + _sub(2, 60, 20, "H", "Neres", "Politano")
        phases = mp.build_match_phases(_frame(subs), "H", "A", goals=_goals())
        self.assertEqual(list(phases["trigger"]), ["kickoff", "substitution"])
        self.assertEqual(phases["home_subs_on"].iloc[1], "Lang; Neres")
        self.assertEqual(phases["home_subs_on"].iloc[0], "")

    def test_nearby_substitutions_of_the_other_team_do_not_fragment_the_match(self):
        subs = _sub(2, 60, 0, "H", "a", "b") + _sub(2, 63, 0, "A", "c", "d") + _sub(2, 68, 0, "A", "e", "f")
        phases = mp.build_match_phases(_frame(subs), "H", "A", goals=_goals())
        self.assertEqual(len(phases), 2)  # 63' and 68' are within 10 minutes of the 60' cut
        self.assertEqual(phases["away_subs_on"].iloc[1], "c; e")

    def test_goal_replaces_a_substitution_just_before_it(self):
        subs = _sub(2, 83, 0, "A", "a", "b")
        phases = mp.build_match_phases(_frame(subs), "H", "A", goals=_goals((85, 48, "A", 2)))
        self.assertEqual(list(phases["trigger"]), ["kickoff", "goal"])

    def test_late_substitution_that_would_leave_a_sliver_is_dropped(self):
        subs = _sub(2, 88, 0, "H", "a", "b")
        phases = mp.build_match_phases(_frame(subs), "H", "A", goals=_goals())
        self.assertEqual(len(phases), 1)

    def test_interval_can_be_a_cut_and_phases_never_overlap(self):
        phases = mp.build_match_phases(_frame(), "H", "A", goals=_goals((20, 0, "H", 1), (70, 0, "A", 2)), split_at_half_time=True)
        self.assertIn("half_time", set(phases["trigger"]))
        self.assertTrue((phases["start_play_s"].iloc[1:].to_numpy() == phases["end_play_s"].iloc[:-1].to_numpy()).all())
        self.assertAlmostEqual(phases["play_minutes"].sum(), 95.0, places=0)

    def test_stoppage_time_does_not_shift_the_second_half_labels(self):
        phases = mp.build_match_phases(_frame(), "H", "A", goals=_goals((50, 0, "A", 2)))
        self.assertEqual(phases["start_clock"].iloc[1], "50:00")
        # first half 47:00 of play, so 50:00 on the clock is 47:00 + 5:00 of play
        self.assertAlmostEqual(phases["start_play_s"].iloc[1], 47 * 60 + 5 * 60)

    def test_dismissal_is_a_cut(self):
        frame = _frame([(2, 55, 0, "H", "Card", "p")])
        frame["Red card"] = [None] * (len(frame) - 1) + [1]
        phases = mp.build_match_phases(frame, "H", "A", goals=_goals())
        self.assertEqual(list(phases["trigger"]), ["kickoff", "dismissal"])

    def test_empty_input(self):
        self.assertTrue(mp.build_match_phases(pd.DataFrame(), "H", "A").empty)

    def test_goal_events_credit_own_goals_to_the_opponent(self):
        frame = _frame()
        shots = pd.DataFrame({
            "periodId": [1, 2], "timeMin": [10, 60], "timeSec": [0, 0], "team_name": ["H", "H"],
            "shot_outcome": ["goal", "own_goal"], "playerName": ["a", "b"],
        })
        goals = mp.goal_events(frame, "H", "A", shots_df=shots)
        self.assertEqual(list(goals["team"]), ["H", "A"])


if __name__ == "__main__":
    unittest.main()
