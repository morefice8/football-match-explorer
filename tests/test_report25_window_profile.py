"""REPORT-25: window profile (territory, pressure, positions, droughts) and proportion statistics."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.metrics import match_clock as mc
from src.metrics import proportion_stats as ps
from src.metrics import window_profile as wp


def _frame(rows, players=None):
    """rows: dicts with period, m, s, team, type and optional outcome/x/y/end_x/end_y/player/role/qualifiers."""
    base = {"outcome": "Successful", "x": np.nan, "y": np.nan, "end_x": np.nan, "end_y": np.nan, "player": "p", "role": "CM", "jersey": 1}
    records = []
    for r in rows:
        d = {**base, **r}
        records.append({
            "periodId": d["period"], "timeMin": d["m"], "timeSec": d["s"], "team_name": d["team"], "type_name": d["type"],
            "outcome": d["outcome"], "x": d["x"], "y": d["y"], "end_x": d["end_x"], "end_y": d["end_y"],
            "playerName": d["player"], "playerId": d.get("pid", d["player"]), "positional_role": d["role"],
            "Mapped Jersey Number": d["jersey"], "Is Starter": d.get("starter", False),
            "cross": d.get("cross", np.nan), "Own goal": d.get("own_goal", np.nan),
        })
    return pd.DataFrame(records)


def _skeleton():
    return [
        dict(period=1, m=0, s=0, team="A", type="Start"), dict(period=1, m=45, s=0, team="A", type="End"),
        dict(period=2, m=45, s=0, team="A", type="Start"), dict(period=2, m=90, s=0, team="A", type="End"),
    ]


def _view(rows):
    frame = _frame(_skeleton() + rows)
    clock = mc.PlayClock.from_events(frame)
    return frame, clock, wp.prepare_events(frame, clock)


def _touch(minute, team="A", x=50.0, y=50.0, period=1, **kw):
    return dict(period=period, m=minute, s=0, team=team, type="Pass", x=x, y=y, **kw)


def _tackle(minute, second=0, team="B", x=50.0, y=50.0, period=1, **kw):
    return dict(period=period, m=minute, s=second, team=team, type="Tackle", x=x, y=y, **kw)


class ViewTests(unittest.TestCase):
    def test_flags(self):
        frame, clock, v = _view([
            _touch(5), _tackle(6), dict(period=1, m=7, s=0, team="A", type="Miss", x=90, y=50),
            dict(period=1, m=8, s=0, team="A", type="Goal", x=95, y=50, own_goal=1),
            dict(period=1, m=9, s=0, team="B", type="Foul", outcome="Unsuccessful", x=40, y=40),
            dict(period=1, m=9, s=5, team="A", type="Foul", outcome="Successful", x=60, y=60),
            dict(period=1, m=10, s=0, team="A", type="Pass", x=20, y=50, role="GK"),
            dict(period=1, m=11, s=0, team="A", type="Pass", x=60, y=60, outcome="Unsuccessful", cross=1),
        ])
        by = lambda m, t: v[(v["period"] == 1) & (v["play_seconds"] == m * 60) & (v["team"] == t)].iloc[0]
        self.assertTrue(by(5, "A")["is_touch"] and not by(5, "A")["is_intervention"])
        self.assertTrue(by(6, "B")["is_intervention"])
        self.assertTrue(by(7, "A")["is_shot"])
        self.assertFalse(by(8, "A")["is_shot"])  # own goal is not a shot
        self.assertTrue(by(9, "B")["is_intervention"])  # a foul committed counts
        self.assertFalse(v[(v["play_seconds"] == 9 * 60 + 5)].iloc[0]["is_intervention"])  # a foul suffered does not
        self.assertTrue(by(10, "A")["is_keeper"])
        self.assertTrue(by(11, "A")["is_cross"] and by(11, "A")["is_loss"])

    def test_events_without_a_clock_are_dropped(self):
        frame = _frame(_skeleton())
        frame.loc[0, "periodId"] = 14
        view = wp.prepare_events(frame, mc.PlayClock.from_events(frame))
        self.assertEqual(len(view), len(frame) - 1)


class TerritoryTests(unittest.TestCase):
    def test_counts_and_rates(self):
        rows = [
            _touch(1, x=20), _touch(2, x=40), _touch(3, x=70, y=50), _touch(4, x=92, y=50),
            dict(period=1, m=5, s=0, team="A", type="Pass", x=60, y=50, end_x=75, end_y=50),         # final-third entry
            dict(period=1, m=6, s=0, team="A", type="Pass", x=95, y=95, outcome="Unsuccessful", cross=1),
            dict(period=1, m=7, s=0, team="A", type="Miss", x=90, y=50),
            _touch(8, team="B", x=50), _touch(9, team="B", x=50),
            _tackle(10, team="A", x=20), _tackle(11, team="A", x=30), _tackle(12, team="A", x=40),
        ]
        frame, clock, v = _view(rows)
        t = wp.territory_profile(v, "A", "B", 0, 20 * 60)
        self.assertEqual(t["touches"], 7)
        self.assertAlmostEqual(t["touches_per_min"], 7 / 20, places=2)
        self.assertAlmostEqual(t["touch_share_pct"], 100 * 7 / 9, places=1)
        self.assertEqual(t["final_third_touches"], 4)           # 70, 92, 95 (cross), 90 (shot); the entry starts at 60
        self.assertEqual(t["final_third_entries"], 1)
        self.assertEqual(t["box_touches"], 2)                   # (92,50) and (90,50); the (95,95) cross is wide
        self.assertEqual((t["crosses"], t["crosses_completed"]), (1, 0))
        self.assertEqual(t["shots"], 1)
        self.assertEqual(t["interventions"], 3)
        self.assertAlmostEqual(t["block_height_m"], 30 * 1.05, places=1)   # median x of the own interventions
        self.assertAlmostEqual(t["final_third_loss_pct"], 100 * 1 / 4, places=1)

    def test_goalkeeper_interventions_do_not_move_the_block(self):
        rows = [_tackle(2, team="A", x=10, role="GK"), _tackle(3, team="A", x=40), _touch(1)]
        frame, clock, v = _view(rows)
        t = wp.territory_profile(v, "A", "B", 0, 20 * 60)
        self.assertEqual(t["interventions"], 1)
        self.assertAlmostEqual(t["block_height_m"], 40 * 1.05, places=1)

    def test_window_is_half_open_and_empty_windows_do_not_crash(self):
        frame, clock, v = _view([_touch(10), _touch(20)])
        self.assertEqual(wp.territory_profile(v, "A", "B", 0, 20 * 60)["touches"], 1)  # 20:00 belongs to the next window
        empty = wp.territory_profile(v, "A", "B", 30 * 60, 31 * 60)
        self.assertEqual(empty["touches"], 0)
        self.assertIsNone(empty["own_half_touch_pct"])


class PressureTests(unittest.TestCase):
    def test_pressed_requires_time_distance_and_period(self):
        rows = [
            _touch(10, x=50, y=50),                      # pressed: tackle 3 s later, 0 m away (rotated frame)
            _tackle(10, 3, x=50, y=50),
            _touch(20, x=50, y=50),                      # not pressed: opponent action 11+ m away
            _tackle(20, 2, x=50, y=70),                  # B's (50,70) is (50,30) in A's frame: 13.6 m
            _touch(30, x=50, y=50),                      # not pressed: the action came before the touch
            _tackle(29, 59, x=50, y=50),
            _touch(44, x=50, y=50),                      # 44:00; first half ends at 45:00, a follow-up needs 6 s
            _touch(45, x=50, y=50, period=2),            # second-half kick-off, must not see first-half events
        ]
        frame, clock, v = _view(rows)
        table = wp.pressure_table(v, "A", "B", 0, clock.total_seconds)
        by_min = {int(r.play_seconds // 60): bool(r.pressed) for r in table.itertuples()}
        self.assertTrue(by_min[10])
        self.assertFalse(by_min[20])
        self.assertFalse(by_min[30])
        self.assertEqual(table.loc[table["pressed"], "presser"].iloc[0], "p")

    def test_an_event_just_after_the_interval_does_not_press_a_first_half_touch(self):
        rows = [dict(period=1, m=45, s=0, team="A", type="Pass", x=50, y=50),
                dict(period=2, m=45, s=2, team="B", type="Tackle", x=50, y=50)]
        frame, clock, v = _view(rows)
        table = wp.pressure_table(v, "A", "B", 0, clock.total_seconds)
        self.assertFalse(table["pressed"].any())

    def test_lanes_and_thirds_use_the_touching_teams_own_frame(self):
        rows = [_touch(10, x=10, y=5), _touch(11, x=50, y=95), _touch(12, x=90, y=50)]
        frame, clock, v = _view(rows)
        table = wp.pressure_table(v, "A", "B", 0, 20 * 60)
        self.assertEqual(list(table["lane"]), ["right_wide", "left_wide", "centre"])
        self.assertEqual(list(table["third"]), ["defensive", "middle", "final"])

    def test_summary_has_counts_ci_and_a_vs_rest_test(self):
        rows = []
        for i in range(20):
            rows += [_touch(1 + i, x=50, y=10, player="wing", pid="w"), _tackle(1 + i, 2, x=50, y=90)]   # wing always pressed
        for i in range(20):
            rows.append(_touch(25 + i // 2, x=50, y=50, player="mid", pid="m"))                         # never pressed
        frame, clock, v = _view(rows)
        table = wp.pressure_table(v, "A", "B", 0, 45 * 60)
        players = wp.summarize_pressure(table, "player").set_index("group")
        self.assertEqual((int(players.loc["wing", "pressed"]), int(players.loc["wing", "touches"])), (20, 20))
        self.assertEqual(int(players.loc["mid", "pressed"]), 0)
        self.assertLess(players.loc["wing", "vs_rest_p"], 0.001)
        self.assertLess(players.loc["wing", "ci_high_pct"], 100.1)
        self.assertGreater(players.loc["wing", "ci_low_pct"], 80)
        self.assertFalse(players.loc["wing", "low_sample"] and players.loc["wing", "touches"] >= wp.LOW_SAMPLE_TOUCHES)
        lanes = wp.summarize_pressure(table, "lane")
        self.assertEqual(list(lanes["group"]), ["right_wide", "centre"])
        overall = wp.summarize_pressure(table, "all").iloc[0]
        self.assertEqual((int(overall["pressed"]), int(overall["touches"])), (20, 40))
        self.assertIsNone(overall["vs_rest_p"])  # "all" has no rest to compare with

    def test_keepers_are_excluded_from_summaries_by_default(self):
        rows = [_touch(5, role="GK", player="gk", pid="g"), _touch(6, player="p1", pid="1")]
        frame, clock, v = _view(rows)
        table = wp.pressure_table(v, "A", "B", 0, 20 * 60)
        self.assertEqual(int(wp.summarize_pressure(table, "all").iloc[0]["touches"]), 1)
        self.assertEqual(int(wp.summarize_pressure(table, "all", include_keepers=True).iloc[0]["touches"]), 2)

    def test_no_touches_gives_an_empty_table(self):
        frame, clock, v = _view([_tackle(5)])
        self.assertTrue(wp.pressure_table(v, "A", "B", 0, 10 * 60).empty)
        self.assertTrue(wp.summarize_pressure(pd.DataFrame(), "all").empty)


class PositionTests(unittest.TestCase):
    def _rows(self):
        rows = [
            dict(period=1, m=0, s=0, team="A", type="Pass", x=30, y=40, player="starter", pid="s1", starter=True),
            dict(period=1, m=10, s=0, team="A", type="Pass", x=50, y=40, player="starter", pid="s1", starter=True),
            dict(period=1, m=20, s=0, team="A", type="Pass", x=40, y=40, player="starter", pid="s1", starter=True),
            dict(period=2, m=60, s=0, team="A", type="Player off", player="starter", pid="s1"),
            dict(period=2, m=60, s=0, team="A", type="Player on", player="sub", pid="s2"),
            dict(period=2, m=70, s=0, team="A", type="Pass", x=80, y=60, player="sub", pid="s2"),
            dict(period=2, m=75, s=0, team="A", type="Pass", x=82, y=60, player="sub", pid="s2"),
            dict(period=2, m=80, s=0, team="A", type="Pass", x=84, y=60, player="sub", pid="s2"),
            dict(period=1, m=15, s=0, team="B", type="Pass", x=70, y=30, player="opp", pid="o1", starter=True),
            dict(period=1, m=25, s=0, team="B", type="Pass", x=70, y=30, player="opp", pid="o1", starter=True),
            dict(period=1, m=35, s=0, team="B", type="Pass", x=70, y=30, player="opp", pid="o1", starter=True),
        ]
        return rows

    def test_minutes_present_and_the_shared_frame(self):
        frame, clock, v = _view(self._rows())
        intervals = wp.on_pitch_intervals(v, frame, clock)
        self.assertEqual(intervals[("A", "s1")][0], 0.0)
        self.assertAlmostEqual(intervals[("A", "s1")][1], clock.play_seconds(2, 3600))
        pos_a = wp.window_positions(v, "A", 0, clock.total_seconds, intervals, "A", min_minutes=12, min_actions=3)
        by = pos_a.set_index("player")
        self.assertEqual(by.loc["starter", "x"], 40.0)                       # median of 30, 50, 40
        self.assertAlmostEqual(by.loc["starter", "minutes_present"], clock.play_seconds(2, 3600) / 60, places=1)
        self.assertEqual(by.loc["sub", "x"], 82.0)
        self.assertAlmostEqual(by.loc["sub", "minutes_present"], clock.total_seconds / 60 - clock.play_seconds(2, 3600) / 60, places=1)
        pos_b = wp.window_positions(v, "B", 0, clock.total_seconds, intervals, "A", min_minutes=12, min_actions=3)
        row = pos_b.iloc[0]
        self.assertEqual((row["x"], row["y"]), (70.0, 30.0))
        self.assertEqual((row["x_shared"], row["y_shared"]), (30.0, 70.0))      # rotated into A's frame

    def test_short_presence_is_excluded_and_low_samples_are_flagged(self):
        frame, clock, v = _view(self._rows())
        intervals = wp.on_pitch_intervals(v, frame, clock)
        window_start = clock.play_seconds(2, 69 * 60)
        pos = wp.window_positions(v, "A", window_start, window_start + 6 * 60, intervals, "A", min_minutes=12, min_actions=8, minimum_rows=1)
        # the window is 6 min (< 12): only players present throughout count; the starter is off by then
        self.assertEqual(set(pos["player"]), {"sub"})
        self.assertTrue(bool(pos["low_sample"].iloc[0]))   # a single action is flagged as low sample

    def test_dismissed_player_stops_at_the_card(self):
        rows = self._rows() + [dict(period=1, m=30, s=0, team="A", type="Card", player="starter", pid="s1")]
        frame, clock, v = _view(rows)
        frame["Red card"] = [1 if t == "Card" else np.nan for t in frame["type_name"]]
        intervals = wp.on_pitch_intervals(v, frame, clock)
        self.assertAlmostEqual(intervals[("A", "s1")][1], 30 * 60)

    def test_team_length_ignores_low_sample_players(self):
        pos = pd.DataFrame({"actions": [20, 20, 3], "x": [20.0, 70.0, 99.0]})
        self.assertAlmostEqual(wp.team_length_m(pos), 50 * 1.05)
        self.assertIsNone(wp.team_length_m(pos.iloc[:1]))
        self.assertIsNone(wp.team_length_m(pd.DataFrame()))


class DroughtTests(unittest.TestCase):
    def test_shot_drought_with_context(self):
        rows = [
            dict(period=1, m=10, s=0, team="A", type="Miss", x=90, y=50),
            dict(period=2, m=70, s=0, team="A", type="Miss", x=90, y=50),
            _touch(30, x=75, y=50), _touch(31, x=80, y=50), dict(period=1, m=32, s=0, team="A", type="Pass", x=95, y=95, cross=1, outcome="Unsuccessful"),
            dict(period=1, m=33, s=0, team="B", type="Miss", x=90, y=50),
        ]
        frame, clock, v = _view(rows)
        table = wp.droughts(v, clock, "A", "B", kind="shots", min_minutes=15)
        self.assertEqual(len(table), 2)
        first = table.iloc[0]
        self.assertEqual((first["start_clock"], first["end_clock"]), ("10:00", "70:00"))
        self.assertAlmostEqual(first["play_minutes"], 10 * 0 + (clock.play_seconds(2, 70 * 60) - 10 * 60) / 60, places=1)
        self.assertEqual(first["final_third_touches"], 3)   # (75, 50), (80, 50) and the wide cross at x=95
        self.assertEqual(first["crosses"], 1)
        self.assertEqual(first["opponent_shots"], 1)
        self.assertTrue(first["ends_with_event"])
        self.assertFalse(table.iloc[-1]["ends_with_event"])

    def test_box_touch_drought_counts_only_penalty_area_touches(self):
        rows = [_touch(5, x=92, y=50), _touch(60, x=92, y=50), _touch(30, x=75, y=50)]
        frame, clock, v = _view(rows)
        table = wp.droughts(v, clock, "A", "B", kind="box_touches", min_minutes=20)
        self.assertEqual(len(table), 2)
        self.assertEqual(int(table.iloc[0]["box_touches"]), 0)
        self.assertEqual(table.iloc[0]["kind"], "box_touches")

    def test_a_short_match_has_no_droughts(self):
        frame, clock, v = _view([dict(period=1, m=10, s=0, team="A", type="Miss", x=90, y=50)])
        self.assertTrue(wp.droughts(v, clock, "A", "B", min_minutes=200).empty)


class ProportionStatsTests(unittest.TestCase):
    def test_wilson_interval_is_sane(self):
        lo, hi = ps.wilson_interval(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        lo, hi = ps.wilson_interval(10, 10)
        self.assertAlmostEqual(hi, 1.0, places=9)
        self.assertEqual(ps.wilson_interval(0, 0), (0.0, 0.0))
        lo, hi = ps.wilson_interval(16, 45)
        self.assertTrue(lo < 16 / 45 < hi)

    def test_fisher_matches_known_values(self):
        self.assertAlmostEqual(ps.fisher_exact_two_sided(3, 1, 1, 3), 0.4857, places=3)
        self.assertAlmostEqual(ps.fisher_exact_two_sided(10, 0, 0, 10), 2 / math.comb(20, 10), places=8)
        self.assertEqual(ps.fisher_exact_two_sided(5, 5, 5, 5), 1.0)
        self.assertEqual(ps.fisher_exact_two_sided(0, 0, 0, 0), 1.0)

    def test_group_vs_rest(self):
        self.assertLess(ps.group_vs_rest(16, 45, 36, 327), 0.001)
        self.assertGreater(ps.group_vs_rest(6, 34, 46, 338), 0.5)


if __name__ == "__main__":
    unittest.main()
