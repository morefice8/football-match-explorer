"""REPORT-25: match clock (stoppage-time safe) and shared pitch zones."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.metrics import match_clock as mc
from src.metrics import pitch_zones as pz


def _events(rows):
    """rows: (period, minute, second, team)."""
    frame = pd.DataFrame(rows, columns=["periodId", "timeMin", "timeSec", "team_name"])
    frame["type_name"] = "Pass"
    return frame


def _two_halves():
    # first half runs to 47:05 (2 minutes of stoppage), second half restarts at 45:00
    return _events([
        (1, 0, 0, "A"), (1, 20, 30, "A"), (1, 45, 10, "A"), (1, 47, 5, "B"),
        (2, 45, 0, "A"), (2, 45, 4, "B"), (2, 60, 0, "A"), (2, 93, 6, "B"),
        (14, 0, 0, "A"),  # pre-match entry: no play time
    ])


class PlayClockTests(unittest.TestCase):
    def test_second_half_continues_after_first_half_stoppage(self):
        frame = _two_halves()
        clock = mc.PlayClock.from_events(frame)
        play = clock.series(frame)
        # first half = 47:05 of play; the second half restarts right after it
        self.assertAlmostEqual(play.iloc[3], 47 * 60 + 5)
        self.assertAlmostEqual(play.iloc[4], 47 * 60 + 5)
        self.assertAlmostEqual(play.iloc[6], 47 * 60 + 5 + 15 * 60)
        self.assertTrue(math.isnan(play.iloc[8]))

    def test_play_seconds_is_monotonic_with_the_event_order(self):
        frame = _two_halves().iloc[:8]
        play = mc.PlayClock.from_events(frame).series(frame)
        self.assertTrue(play.is_monotonic_increasing)
        # the plain printed clock is NOT: first-half stoppage > second-half start
        self.assertFalse(mc.clock_seconds(frame).is_monotonic_increasing)

    def test_total_minutes_and_back_mapping(self):
        frame = _two_halves().iloc[:8]
        clock = mc.PlayClock.from_events(frame)
        self.assertAlmostEqual(clock.total_minutes, (47 * 60 + 5 + (93 * 60 + 6 - 45 * 60)) / 60)
        self.assertEqual(clock.label(20 * 60 + 30), "20:30")
        self.assertEqual(clock.label(47 * 60 + 5 + 15 * 60), "60:00")
        self.assertEqual(clock.label(float("nan")), "")

    def test_extra_time_is_chained(self):
        frame = _events([(1, 0, 0, "A"), (1, 45, 0, "A"), (2, 45, 0, "A"), (2, 90, 0, "A"), (3, 90, 0, "A"), (3, 105, 0, "A")])
        clock = mc.PlayClock.from_events(frame)
        self.assertAlmostEqual(clock.total_minutes, 45 + 45 + 15)
        self.assertEqual([p[0] for p in clock.periods], [1, 2, 3])

    def test_same_period_followers_ignore_the_other_half(self):
        frame = mc.add_play_clock(_two_halves().iloc[:8])
        first_half_end = float(frame["play_seconds"].iloc[3])
        within = mc.same_period_followers(frame, first_half_end, 1, 6.0)
        self.assertEqual(set(within["periodId"]), {1})  # second-half kick-off is 0 s later in play time
        sloppy = frame[(frame["play_seconds"] >= first_half_end) & (frame["play_seconds"] <= first_half_end + 6)]
        self.assertIn(2, set(sloppy["periodId"]))  # what a naive comparison would have paired

    def test_empty_and_missing_columns(self):
        self.assertEqual(mc.PlayClock.from_events(pd.DataFrame()).total_seconds, 0.0)
        frame = pd.DataFrame({"periodId": [1], "type_name": ["Pass"]})
        self.assertTrue(math.isnan(mc.PlayClock.from_events(frame).series(frame).iloc[0]))
        self.assertEqual(mc.format_clock(None), "")
        self.assertEqual(mc.format_clock(5405), "90:05")


class PitchZoneTests(unittest.TestCase):
    def test_lanes_follow_the_app_orientation(self):
        self.assertEqual(pz.lane_of(5), "right_wide")
        self.assertEqual(pz.lane_of(30), "right_half")
        self.assertEqual(pz.lane_of(50), "centre")
        self.assertEqual(pz.lane_of(70), "left_half")
        self.assertEqual(pz.lane_of(95), "left_wide")
        self.assertIsNone(pz.lane_of(float("nan")))

    def test_lane_edges_belong_to_the_next_lane(self):
        self.assertEqual(pz.lane_of(21.1), "right_half")
        self.assertEqual(pz.lane_of(21.0999), "right_wide")
        self.assertEqual(pz.lane_of(78.9), "left_wide")

    def test_thirds_and_series_variants_agree(self):
        xs = pd.Series([0.0, 33.0, 33.4, 66.6, 66.7, 100.0, np.nan])
        self.assertEqual(list(pz.third_series(xs).iloc[:6]), ["defensive", "defensive", "middle", "middle", "final", "final"])
        self.assertIsNone(pz.third_series(xs).iloc[6])
        ys = pd.Series([0.0, 25.0, 50.0, 75.0, 100.0, np.nan])
        self.assertEqual(list(pz.lane_series(ys).iloc[:5]), [pz.lane_of(v) for v in ys.iloc[:5]])
        self.assertIsNone(pz.lane_series(ys).iloc[5])

    def test_penalty_area_uses_true_metres(self):
        self.assertTrue(pz.in_penalty_area(90.0, 50.0))
        self.assertFalse(pz.in_penalty_area(80.0, 50.0))      # 21 m from the line
        self.assertFalse(pz.in_penalty_area(95.0, 85.0))      # wide of the area
        edge = 100 - 16.5 / 1.05
        self.assertTrue(pz.in_penalty_area(edge + 0.01, 50.0))
        self.assertFalse(pz.in_penalty_area(edge - 0.01, 50.0))
        series = pz.in_penalty_area(pd.Series([90.0, 80.0]), pd.Series([50.0, 50.0]))
        self.assertEqual(list(series), [True, False])

    def test_distance_scales_each_axis(self):
        self.assertAlmostEqual(float(pz.distance_m(100, 0, 100, 100)), 68.0)
        self.assertAlmostEqual(float(pz.distance_m(0, 50, 100, 50)), 105.0)

    def test_shared_frame_rotates_the_other_team(self):
        frame = pd.DataFrame({"team_name": ["A", "B"], "x": [80.0, 80.0], "y": [10.0, 10.0], "end_x": [90.0, 90.0], "end_y": [20.0, 20.0]})
        shared = pz.add_shared_frame(frame, "A")
        self.assertEqual((shared["x_shared"].iloc[0], shared["y_shared"].iloc[0]), (80.0, 10.0))
        # B attacks the other way: its right (y=10) is the reference team's left (y=90)
        self.assertEqual((shared["x_shared"].iloc[1], shared["y_shared"].iloc[1]), (20.0, 90.0))
        self.assertEqual((shared["end_x_shared"].iloc[1], shared["end_y_shared"].iloc[1]), (10.0, 80.0))
        self.assertEqual(pz.lane_of(shared["y_shared"].iloc[1]), "left_wide")


if __name__ == "__main__":
    unittest.main()
