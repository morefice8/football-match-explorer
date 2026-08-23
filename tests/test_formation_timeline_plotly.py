import unittest
from pathlib import Path

import pandas as pd

from src.visualization import formation_plotly


def formation_players(prefix, replacement=None):
    players = [f"{prefix}{index}" for index in range(1, 12)]
    if replacement is not None:
        slot, player_id = replacement
        players[slot - 1] = player_id
    return ",".join(players)


def formation_slots():
    return ",".join(str(index) for index in range(1, 12))


class FormationTimelinePlotlyTests(unittest.TestCase):

    @staticmethod
    def sample_df():
        rows = []
        sequence = 0

        def add(**kwargs):
            nonlocal sequence
            base = {
                "id": sequence + 1000,
                "eventId": sequence + 1,
                "event_sequence_index": sequence,
                "typeId": 1,
                "periodId": 1,
                "timeMin": 0,
                "timeSec": 0,
                "contestantId": "H",
                "team_name": "Home",
                "playerId": None,
                "playerName": None,
                "Mapped Jersey Number": None,
                "Team formation": None,
                "Involved": None,
                "Team player formation": None,
                "related_eventId": None,
                "Own goal": 0,
                "Red card": 0,
                "Second yellow": 0,
            }
            base.update(kwargs)
            rows.append(base)
            sequence += 1

        add(
            typeId=34,
            contestantId="H",
            team_name="Home",
            **{
                "Team formation": 4,
                "Involved": formation_players("h"),
                "Team player formation": formation_slots(),
            },
        )
        add(
            typeId=34,
            contestantId="A",
            team_name="Away",
            **{
                "Team formation": 4,
                "Involved": formation_players("a"),
                "Team player formation": formation_slots(),
            },
        )

        for team, contestant, prefix in (
            ("Home", "H", "h"),
            ("Away", "A", "a"),
        ):
            for jersey in range(1, 12):
                add(
                    typeId=1,
                    timeMin=1,
                    contestantId=contestant,
                    team_name=team,
                    playerId=f"{prefix}{jersey}",
                    playerName=f"{team} Player {jersey}",
                    **{"Mapped Jersey Number": jersey},
                )

        add(
            typeId=16,
            timeMin=10,
            timeSec=15,
            contestantId="H",
            team_name="Home",
            playerId="h9",
            playerName="Home Player 9",
            **{"Mapped Jersey Number": 9},
        )

        sub_off_event_id = 200
        add(
            eventId=sub_off_event_id,
            typeId=18,
            timeMin=20,
            contestantId="H",
            team_name="Home",
            playerId="h9",
            playerName="Home Player 9",
            **{"Mapped Jersey Number": 9},
        )
        add(
            eventId=201,
            typeId=19,
            timeMin=20,
            contestantId="H",
            team_name="Home",
            playerId="h12",
            playerName="Home Player 12",
            related_eventId=sub_off_event_id,
            **{"Mapped Jersey Number": 19},
        )

        add(
            typeId=40,
            timeMin=30,
            timeSec=5,
            contestantId="H",
            team_name="Home",
            **{
                "Team formation": 8,
                "Involved": formation_players(
                    "h",
                    replacement=(9, "h12"),
                ),
                "Team player formation": formation_slots(),
            },
        )

        add(
            typeId=17,
            timeMin=40,
            contestantId="A",
            team_name="Away",
            playerId="a3",
            playerName="Away Player 3",
            **{
                "Mapped Jersey Number": 3,
                "Red card": 0,
                "Second yellow": 0,
            },
        )

        add(
            typeId=17,
            timeMin=50,
            contestantId="A",
            team_name="Away",
            playerId="a5",
            playerName="Away Player 5",
            **{
                "Mapped Jersey Number": 5,
                "Red card": 1,
            },
        )

        return pd.DataFrame(rows)

    def build_model(self):
        return formation_plotly.build_formation_timeline_model(
            self.sample_df(),
            {
                "hteamName": "Home",
                "ateamName": "Away",
            },
        )

    def test_model_contains_required_match_moments(self):
        model = self.build_model()
        kinds = [
            event["kind"]
            for moment in model["moments"]
            for event in moment["events"]
        ]
        for expected in (
            "starting_xi",
            "goal",
            "substitution",
            "formation_change",
            "dismissal",
        ):
            self.assertIn(expected, kinds)

    def test_minute_level_moment_preserves_exact_event_time(self):
        model = self.build_model()

        goal_moment = next(
            moment
            for moment in model["moments"]
            if any(
                event["kind"] == "goal"
                for event in moment["events"]
            )
        )

        self.assertEqual(
            goal_moment["time_seconds"],
            10 * 60,
        )

        goal_event = next(
            event
            for event in goal_moment["events"]
            if event["kind"] == "goal"
        )

        self.assertEqual(
            goal_event["event_time_label"],
            "10′ 15″",
        )

    def test_ordinary_yellow_card_is_not_a_timeline_moment(self):
        model = self.build_model()
        timeline_minutes = [
            moment["time_seconds"] // 60
            for moment in model["moments"]
        ]
        self.assertNotIn(40, timeline_minutes)

    def test_substitution_replaces_player_in_same_slot(self):
        model = self.build_model()
        moment = next(
            value
            for value in model["moments"]
            if any(
                event["kind"] == "substitution"
                for event in value["events"]
            )
        )
        players = moment["home_state"]["players"]
        self.assertNotIn("h9", players)
        self.assertEqual(players["h12"], 9)
        self.assertIn("h12", moment["home_highlights"])

    def test_dismissal_removes_player_from_on_pitch_state(self):
        model = self.build_model()
        moment = next(
            value
            for value in model["moments"]
            if any(
                event["kind"] == "dismissal"
                for event in value["events"]
            )
        )
        self.assertNotIn("a5", moment["away_state"]["players"])
        self.assertEqual(
            len(moment["away_state"]["players"]),
            10,
        )

    def test_goal_updates_score_after_event(self):
        model = self.build_model()
        moment = next(
            value
            for value in model["moments"]
            if any(
                event["kind"] == "goal"
                for event in value["events"]
            )
        )
        self.assertEqual(moment["score_home"], 1)
        self.assertEqual(moment["score_away"], 0)

    def test_own_goal_is_credited_to_opposition(self):
        df = self.sample_df()
        goal_index = df.index[df["typeId"].eq(16)][0]
        df.loc[goal_index, "Own goal"] = 1

        model = formation_plotly.build_formation_timeline_model(
            df,
            {
                "hteamName": "Home",
                "ateamName": "Away",
            },
        )
        moment = next(
            value
            for value in model["moments"]
            if any(
                event["kind"] == "goal"
                for event in value["events"]
            )
        )
        self.assertEqual(moment["score_home"], 0)
        self.assertEqual(moment["score_away"], 1)

    def test_slider_marks_are_shared_match_times(self):
        model = self.build_model()
        marks = formation_plotly.build_formation_slider_marks(model)
        for second in (
            0,
            10 * 60,
            20 * 60,
            30 * 60,
            50 * 60,
        ):
            self.assertIn(second, marks)

    def test_home_and_away_figures_share_height_and_orientation(self):
        model = self.build_model()
        start = model["moments"][0]

        home = formation_plotly.plot_formation_timeline_state(
            start["home_state"],
            model["player_data"],
            is_away=False,
        )
        away = formation_plotly.plot_formation_timeline_state(
            start["away_state"],
            model["player_data"],
            is_away=True,
        )

        self.assertEqual(home.layout.height, away.layout.height)
        self.assertEqual(
            home.layout.height,
            formation_plotly.FORMATION_TIMELINE_HEIGHT,
        )
        self.assertEqual(
            list(home.data[0].x),
            list(away.data[0].x),
        )
        self.assertEqual(
            list(home.data[0].y),
            list(away.data[0].y),
        )

    def test_active_timeline_branch_contains_no_static_image_stack(self):
        source = Path("app.py").read_text(encoding="utf-8")
        start = source.find(
            "        if active_tab == 'formation_timeline':"
        )
        end = source.find(
            "        elif active_tab == 'mean_positions':",
            start,
        )
        self.assertNotEqual(start, -1)
        self.assertNotEqual(end, -1)

        branch = source[start:end]
        self.assertNotIn("plot_formation_snapshot", branch)
        self.assertNotIn("dash_html.Img", branch)
        self.assertIn('"formation-timeline-slider"', branch)
        self.assertIn('"formation-timeline-home-graph"', branch)
        self.assertIn('"formation-timeline-away-graph"', branch)


    def test_dense_substitutions_remain_selectable_but_unlabelled(self):
        model = self.build_model()
        marks = formation_plotly.build_formation_slider_marks(model)

        substitution_second = 20 * 60

        self.assertIn(
            substitution_second,
            marks,
        )
        self.assertEqual(
            marks[substitution_second]["label"],
            "",
        )

    def test_major_tactical_events_keep_slider_labels(self):
        model = self.build_model()
        marks = formation_plotly.build_formation_slider_marks(model)

        self.assertIn(
            "XI",
            marks[0]["label"],
        )
        self.assertIn(
            "G",
            marks[10 * 60]["label"],
        )
        self.assertIn(
            "FORM",
            marks[30 * 60]["label"],
        )
        self.assertIn(
            "RC",
            marks[50 * 60]["label"],
        )



    def test_same_football_minute_is_one_slider_moment(self):
        df = self.sample_df()

        extra_rows = pd.DataFrame(
            [
                {
                    "id": 9001,
                    "eventId": 901,
                    "event_sequence_index": 9001,
                    "typeId": 18,
                    "periodId": 2,
                    "timeMin": 20,
                    "timeSec": 35,
                    "contestantId": "A",
                    "team_name": "Away",
                    "playerId": "a7",
                    "playerName": "Away Player 7",
                    "Mapped Jersey Number": 7,
                    "Team formation": None,
                    "Involved": None,
                    "Team player formation": None,
                    "related_eventId": None,
                    "Own goal": 0,
                    "Red card": 0,
                    "Second yellow": 0,
                },
                {
                    "id": 9002,
                    "eventId": 902,
                    "event_sequence_index": 9002,
                    "typeId": 19,
                    "periodId": 2,
                    "timeMin": 20,
                    "timeSec": 36,
                    "contestantId": "A",
                    "team_name": "Away",
                    "playerId": "a12",
                    "playerName": "Away Player 12",
                    "Mapped Jersey Number": 18,
                    "Team formation": None,
                    "Involved": None,
                    "Team player formation": None,
                    "related_eventId": 901,
                    "Own goal": 0,
                    "Red card": 0,
                    "Second yellow": 0,
                },
            ]
        )

        df = pd.DataFrame(
            [
                *df.to_dict("records"),
                *extra_rows.to_dict("records"),
            ]
        )

        model = formation_plotly.build_formation_timeline_model(
            df,
            {
                "hteamName": "Home",
                "ateamName": "Away",
            },
        )

        minute_20 = [
            moment
            for moment in model["moments"]
            if moment["time_seconds"] == 20 * 60
        ]

        self.assertEqual(
            len(minute_20),
            1,
        )

        substitutions = [
            event
            for event in minute_20[0]["events"]
            if event["kind"] == "substitution"
        ]

        self.assertEqual(
            len(substitutions),
            2,
        )
        self.assertEqual(
            {
                event["event_time_label"]
                for event in substitutions
            },
            {
                "20′",
                "20′ 35″",
            },
        )

    def test_player_names_use_high_contrast_annotations(self):
        model = self.build_model()
        start = model["moments"][0]

        figure = formation_plotly.plot_formation_timeline_state(
            start["home_state"],
            model["player_data"],
            is_away=False,
        )

        player_labels = [
            annotation
            for annotation in figure.layout.annotations
            if annotation.bgcolor == "rgba(16,47,69,0.92)"
        ]

        self.assertEqual(
            len(player_labels),
            11,
        )
        self.assertTrue(
            all(
                annotation.font.color == "#ffffff"
                for annotation in player_labels
            )
        )



    def test_mixed_moment_uses_single_priority_label(self):
        model = self.build_model()

        formation_moment = next(
            moment
            for moment in model["moments"]
            if any(
                event["kind"] == "formation_change"
                for event in moment["events"]
            )
        )

        formation_moment["events"].append(
            {
                "kind": "substitution",
                "team": "home",
                "description": "Synthetic same-minute substitution",
                "event_time_label": "30′ 20″",
            }
        )

        marks = formation_plotly.build_formation_slider_marks(model)

        self.assertEqual(
            marks[30 * 60]["label"],
            "30′ FORM",
        )

    def test_nearby_equal_priority_labels_keep_only_later_one(self):
        model = self.build_model()

        model["moments"].append(
            {
                "time_seconds": 31 * 60,
                "time_label": "31′",
                "score_home": 1,
                "score_away": 0,
                "score": "1 – 0",
                "events": [
                    {
                        "kind": "formation_change",
                        "team": "away",
                        "description": "Synthetic later shape change",
                        "event_time_label": "31′ 05″",
                    }
                ],
                "home_state": model["moments"][-1]["home_state"],
                "away_state": model["moments"][-1]["away_state"],
                "home_highlights": [],
                "away_highlights": [],
                "home_formation_name": "3-4-2-1",
                "away_formation_name": "4-3-3",
            }
        )

        model["moments"] = sorted(
            model["moments"],
            key=lambda moment: moment["time_seconds"],
        )

        marks = formation_plotly.build_formation_slider_marks(model)

        self.assertEqual(
            marks[30 * 60]["label"],
            "",
        )
        self.assertEqual(
            marks[31 * 60]["label"],
            "31′ FORM",
        )

    def test_nearby_higher_priority_label_wins_collision(self):
        model = self.build_model()

        model["moments"].append(
            {
                "time_seconds": 31 * 60,
                "time_label": "31′",
                "score_home": 2,
                "score_away": 0,
                "score": "2 – 0",
                "events": [
                    {
                        "kind": "goal",
                        "team": "home",
                        "description": "Synthetic nearby goal",
                        "event_time_label": "31′ 10″",
                    }
                ],
                "home_state": model["moments"][-1]["home_state"],
                "away_state": model["moments"][-1]["away_state"],
                "home_highlights": [],
                "away_highlights": [],
                "home_formation_name": "3-4-2-1",
                "away_formation_name": "4-3-3",
            }
        )

        model["moments"] = sorted(
            model["moments"],
            key=lambda moment: moment["time_seconds"],
        )

        marks = formation_plotly.build_formation_slider_marks(model)

        self.assertEqual(
            marks[30 * 60]["label"],
            "",
        )
        self.assertEqual(
            marks[31 * 60]["label"],
            "31′ G",
        )



if __name__ == "__main__":
    unittest.main()
