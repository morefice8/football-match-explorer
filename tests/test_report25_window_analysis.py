"""REPORT-25: insight candidates and the window analysis orchestration."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.metrics import insight_candidates as ic
from src.reporting import window_analysis as wa
from test_report11_real_shaped_fixture import _processed_fixture
from test_report25_window_profile import _frame, _skeleton, _tackle, _touch


def _pressure(rows):
    """rows: (team, phase_id, group_type, group, touches, pressed, vs_rest_p)."""
    frame = pd.DataFrame(rows, columns=["team_name", "phase_id", "group_type", "group", "touches", "pressed", "vs_rest_p"])
    frame["pressed_pct"] = (100.0 * frame["pressed"] / frame["touches"]).round(1)
    frame["ci_low_pct"] = frame["pressed_pct"] - 5
    frame["ci_high_pct"] = frame["pressed_pct"] + 5
    return frame


class InsightRuleTests(unittest.TestCase):
    def _base(self):
        return [
            ("A", 0, "all", "all", 300, 45, None),
            ("A", 0, "player", "wing", 45, 16, 0.0002),
            ("A", 0, "player", "mid", 53, 0, 0.0002),
            ("A", 0, "player", "cb", 30, 1, 0.01 - 0.0001),
            ("A", 0, "player", "small", 10, 8, 0.001),         # too few touches: never a candidate
            ("A", 0, "lane", "right_wide", 82, 22, 0.001),
            ("A", 0, "lane", "left_wide", 70, 8, 0.5),
            ("A", 0, "lane", "centre", 84, 4, 0.002),
        ]

    def test_hotspot_free_players_and_lane_gap(self):
        out = ic.build_insight_candidates(pd.DataFrame(), _pressure(self._base()), pd.DataFrame())
        kinds = list(out["kind"])
        self.assertIn("pressure_hotspot_player", kinds)
        self.assertIn("pressure_free_players", kinds)
        self.assertIn("pressure_lane_gap", kinds)
        hot = out[out["kind"] == "pressure_hotspot_player"].iloc[0]
        numbers = json.loads(hot["numbers"])
        self.assertEqual((numbers["player"], numbers["pressed"], numbers["touches"]), ("wing", 16, 45))
        self.assertEqual(hot["strength"], "strong")
        free = out[out["kind"] == "pressure_free_players"].iloc[0]
        self.assertEqual(json.loads(free["numbers"])["players"], ["mid", "cb"])
        gap = json.loads(out[out["kind"] == "pressure_lane_gap"].iloc[0]["numbers"])
        self.assertEqual((gap["most_contested_lane"], gap["least_contested_lane"]), ("right_wide", "centre"))
        self.assertNotIn("small", " ".join(out["statement"]))   # 10 touches never qualify

    def test_no_candidate_when_the_difference_is_not_significant(self):
        rows = [("A", 0, "all", "all", 300, 45, None), ("A", 0, "player", "p", 40, 8, 0.4), ("A", 0, "player", "q", 40, 5, 0.6)]
        out = ic.build_insight_candidates(pd.DataFrame(), _pressure(rows), pd.DataFrame())
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), list(ic.COLUMNS))

    def test_block_height_and_length_shifts_need_size_and_sample(self):
        window = pd.DataFrame([
            ("A", 1, "0'-20'", 20.0, 30.0, 25, 45.0), ("A", 2, "21'-40'", 20.0, 16.0, 20, 44.0),
            ("A", 3, "41'-45'", 4.0, 60.0, 30, 70.0),          # too short: ignored
            ("A", 4, "46'-70'", 25.0, 31.0, 5, 30.0),          # too few interventions for the block, fine for the length
        ], columns=["team_name", "phase_id", "phase_label", "play_minutes", "block_height_m", "interventions", "team_length_m"])
        window.insert(0, "report_team", "A")
        out = ic.build_insight_candidates(window, pd.DataFrame(), pd.DataFrame())
        shifts = out[out["kind"] == "block_height_shift"]
        self.assertEqual(len(shifts), 1)
        self.assertEqual(json.loads(shifts.iloc[0]["numbers"])["delta_m"], -14.0)
        self.assertEqual(shifts.iloc[0]["strength"], "moderate")
        lengths = out[out["kind"] == "team_length_shift"]
        self.assertEqual(len(lengths), 1)
        self.assertEqual(json.loads(lengths.iloc[0]["numbers"])["delta_m"], -14.0)   # 44 -> 30 across phases 2 and 4

    def test_droughts_become_candidates_above_the_threshold(self):
        droughts = pd.DataFrame([
            ("A", "shots", "43:51", "76:07", 34.4, 4, 54, 4, 4, True), ("A", "box_touches", "48:52", "76:07", 27.2, 0, 28, 3, 3, True),
            ("B", "shots", "10:00", "25:00", 15.0, 1, 10, 0, 0, True),
        ], columns=["team_name", "kind", "start_clock", "end_clock", "play_minutes", "box_touches", "final_third_touches", "crosses", "opponent_shots", "ends_with_event"])
        out = ic.build_insight_candidates(pd.DataFrame(), pd.DataFrame(), droughts)
        self.assertEqual(set(out["kind"]), {"shot_drought", "box_touch_drought"})
        shot = out[out["kind"] == "shot_drought"].iloc[0]
        self.assertEqual(shot["strength"], "strong")                  # >= 30 minutes
        self.assertEqual(out[out["kind"] == "box_touch_drought"].iloc[0]["strength"], "moderate")
        self.assertNotIn("B", set(out["team_name"]))                   # 15 minutes is below the 20-minute bar

    def test_strong_candidates_come_first_and_ids_are_unique(self):
        window = pd.DataFrame([("A", "A", 1, "0'-20'", 20.0, 30.0, 25, 40.0), ("A", "A", 2, "21'-40'", 20.0, 22.0, 20, 40.0)],
                              columns=["report_team", "team_name", "phase_id", "phase_label", "play_minutes", "block_height_m", "interventions", "team_length_m"])
        out = ic.build_insight_candidates(window, _pressure(self._base()), pd.DataFrame())
        order = list(out["strength"].map({"strong": 0, "moderate": 1, "weak": 2}))
        self.assertEqual(order, sorted(order))
        self.assertEqual(out["id"].nunique(), len(out))


class WindowAnalysisTests(unittest.TestCase):
    def _match(self):
        rows = []
        # A has three players; the wide one is always followed by a B tackle. B's tackles are in B's own frame.
        for i in range(30):
            minute = 1 + i
            rows.append(_touch(minute, "A", x=60, y=10, player="wing", pid="w", starter=True))     # right-wide touches, always pressed
            rows.append(_tackle(minute, 2, "B", x=40, y=90))                                        # (60, 10) in A's frame
            rows.append(_touch(minute, "A", x=50, y=50, player="mid", pid="m", starter=True))      # centre touches, never pressed
            rows.append(_touch(minute, "A", x=30, y=50, player="back", pid="k", starter=True))     # deep touches, never pressed
            rows.append(_touch(minute, "B", x=50, y=50, player="b1", pid="b1", starter=True))
        rows.append(dict(period=1, m=30, s=0, team="A", type="Goal", x=95, y=50, player="wing", pid="w", starter=True))
        return _frame(_skeleton() + rows)

    def test_pipeline_on_a_match_with_a_known_pattern(self):
        frame = self._match()
        out = wa.compute_window_analysis(frame, ("A", "B"))
        self.assertEqual(len(out.phases), 2)
        self.assertEqual(list(out.phases["trigger"]), ["kickoff", "goal"])
        self.assertEqual(list(out.window_profile.columns), list(wa.WINDOW_PROFILE_COLUMNS))
        self.assertEqual(set(out.window_profile["phase_id"]), {0, 1, 2})
        self.assertEqual(len(out.window_profile), 6)                                # 3 scopes x 2 teams
        self.assertAlmostEqual(out.window_profile[(out.window_profile["phase_id"] == 0)]["play_minutes"].iloc[0], 90, places=0)
        match = out.pressure_profile[(out.pressure_profile["phase_id"] == 0) & (out.pressure_profile["team_name"] == "A")]
        player = match[(match["group_type"] == "player")].set_index("group")
        self.assertEqual((int(player.loc["wing", "pressed"]), int(player.loc["wing", "touches"])), (30, 31))   # 30 passes + the goal touch
        self.assertEqual(int(player.loc["mid", "pressed"]), 0)
        lane = match[match["group_type"] == "lane"].set_index("group")
        self.assertGreater(lane.loc["right_wide", "pressed_pct"], 90)
        self.assertEqual(lane.loc["centre", "pressed_pct"], 0)
        kinds = set(out.insights["kind"])
        self.assertIn("pressure_hotspot_player", kinds)
        self.assertEqual(out.insights[out.insights["kind"] == "pressure_hotspot_player"].iloc[0]["team_name"], "A")

    def test_positions_use_the_shared_frame_for_the_second_team(self):
        out = wa.compute_window_analysis(self._match(), ("A", "B"))
        pos = out.positions[(out.positions["phase_id"] == 0)]
        b = pos[pos["team_name"] == "B"].iloc[0]
        self.assertEqual((b["x"], b["y"]), (50.0, 50.0))                            # b1 in its own frame
        self.assertEqual((b["x_shared"], b["y_shared"]), (50.0, 50.0))              # rotation of (50, 50) is itself
        a = pos[(pos["team_name"] == "A") & (pos["player"] == "wing")].iloc[0]
        self.assertEqual((a["x_shared"], a["y_shared"]), (a["x"], a["y"]))          # the reference team is unchanged

    def test_pressure_touches_table_is_one_row_per_touch(self):
        out = wa.compute_window_analysis(self._match(), ("A", "B"))
        a = out.pressure_touches[out.pressure_touches["team_name"] == "A"]
        self.assertEqual(len(a), 30 + 30 + 30 + 1)
        self.assertEqual(list(out.pressure_touches.columns), list(wa.PRESSURE_TOUCH_COLUMNS))
        self.assertTrue(a["clock"].str.match(r"^\d+:\d\d$").all())

    def test_thin_input_degrades_to_empty_tables(self):
        empty = wa.compute_window_analysis(pd.DataFrame(), ("A", "B"))
        self.assertTrue(empty.window_profile.empty and empty.phases.empty and empty.insights.empty)
        quiet = wa.compute_window_analysis(_frame(_skeleton()), ("A", "B"))      # no touches, no shots: one phase
        self.assertEqual(len(quiet.phases), 1)
        self.assertEqual(len(quiet.window_profile), 4)
        # the only findings a match without events can have are whole-match droughts
        self.assertTrue(set(quiet.insights["kind"]) <= {"shot_drought", "box_touch_drought"})
        self.assertTrue(quiet.pressure_profile.empty)    # no touches: nothing to summarise
        self.assertTrue(wa.compute_window_analysis(self._match(), ("A",)).phases.empty)   # needs both teams

    def test_real_shaped_fixture_runs_end_to_end(self):
        frame, info = _processed_fixture()
        teams = (info["hteamName"], info["ateamName"])
        out = wa.compute_window_analysis(frame, teams)
        self.assertGreater(len(out.window_profile), 0)
        self.assertEqual(list(out.pressure_profile.columns), list(wa.PRESSURE_PROFILE_COLUMNS))
        self.assertEqual(list(out.positions.columns), list(wa.WINDOW_POSITION_COLUMNS))
        self.assertEqual(list(out.droughts.columns), list(wa.SHOT_DROUGHT_COLUMNS))


if __name__ == "__main__":
    unittest.main()
