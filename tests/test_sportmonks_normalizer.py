import unittest

import pandas as pd

from src.providers.sportmonks.normalizer import (
    SeasonNormalizer,
    coach_from_team,
    formation_from_lineups,
    metric_value,
)
from src.metrics.sportmonks import PLAYER_METRIC_GROUPS, TEAM_METRIC_GROUPS, TEAM_RADAR_GROUPS


class SportmonksNormalizerTests(unittest.TestCase):
    def test_team_coach_relation_is_normalized(self):
        team = {
            "id": 8,
            "name": "Liverpool",
            "coaches": [
                {
                    "id": 9001,
                    "team_id": 8,
                    "coach_id": 100,
                    "position_id": 221,
                    "active": False,
                    "start": "2024-07-01",
                    "end": "2025-05-31",
                },
                {
                    "id": 9002,
                    "team_id": 8,
                    "coach_id": 200,
                    "position_id": 221,
                    "active": False,
                    "start": "2025-06-01",
                    "end": "2026-06-10",
                },
                {
                    "id": 9003,
                    "team_id": 8,
                    "coach_id": 300,
                    "position_id": 221,
                    "active": True,
                    "start": "2026-06-11",
                    "end": None,
                },
            ],
        }
        profiles = {
            200: {
                "id": 200,
                "display_name": "Arne Slot",
                "image_path": "https://example.test/slot.png",
            }
        }
        coach = coach_from_team(team, profiles, as_of="2026-05-24")
        self.assertEqual(coach["coach_name"], "Arne Slot")
        self.assertEqual(coach["coach_image_path"], "https://example.test/slot.png")

        normalizer = SeasonNormalizer("2025-2026")
        normalizer.add_teams(
            {"data": [team]},
            {"league_id": 8, "league": "Premier League", "season_id": 25583},
            coach_profiles=profiles,
            coach_as_of="2026-05-24",
        )
        team_row = normalizer.frames()["teams"].iloc[0]
        self.assertEqual(team_row["coach_id"], 200)
        self.assertEqual(team_row["coach_name"], "Arne Slot")

    def test_formation_is_derived_from_starting_lineup_fields(self):
        fields = [
            "1:1",
            "2:1", "2:2", "2:3", "2:4",
            "3:1", "3:2",
            "4:1", "4:2", "4:3",
            "5:1",
        ]
        lineups = [
            {"team_id": 10, "formation_field": field}
            for field in fields
        ]
        lineups.append({"team_id": 20, "formation_field": "1:1"})
        self.assertEqual(formation_from_lineups(lineups, 10), "4-2-3-1")
        self.assertIsNone(formation_from_lineups(lineups, 20))

        normalizer = SeasonNormalizer("2025-2026")
        normalizer.add_fixture(
            {
                "data": {
                    "id": 99,
                    "participants": [
                        {"id": 10, "name": "Home", "meta": {"location": "home"}},
                        {"id": 20, "name": "Away", "meta": {"location": "away"}},
                    ],
                    "lineups": [
                        {
                            "id": index,
                            "player_id": 1000 + index,
                            "player_name": f"Player {index}",
                            "team_id": 10,
                            "type_id": 11,
                            "formation_field": field,
                        }
                        for index, field in enumerate(fields, start=1)
                    ],
                }
            },
            {"league_id": 8, "league": "Premier League", "season_id": 25583},
        )
        home_row = next(row for row in normalizer.team_matchlogs if row["team_id"] == 10)
        self.assertEqual(home_row["formation"], "4-2-3-1")

    def test_metric_value_supports_fixture_and_season_shapes(self):
        self.assertEqual(metric_value({"data": {"value": 3}}), 3)
        self.assertEqual(metric_value({"value": {"total": 12}}), 12)
        self.assertEqual(metric_value({"value": {"count": 8, "average": 2}}), 8)

    def test_fixture_builds_team_and_player_compatibility_rows(self):
        normalizer = SeasonNormalizer("2025-2026")
        context = {"league_id": 8, "league": "Premier League", "season_id": 25583}
        fixture = {
            "data": {
                "id": 100,
                "season_id": 25583,
                "starting_at": "2025-08-16 15:00:00",
                "state_id": 5,
                "state": {"short_name": "FT"},
                "participants": [
                    {"id": 10, "name": "Home", "meta": {"location": "home"}},
                    {"id": 20, "name": "Away", "meta": {"location": "away"}},
                ],
                "scores": [
                    {"participant_id": 10, "description": "CURRENT", "score": {"goals": 2}},
                    {"participant_id": 20, "description": "CURRENT", "score": {"goals": 1}},
                ],
                "statistics": [
                    {"participant_id": 10, "type_id": 42, "data": {"value": 10}},
                    {"participant_id": 20, "type_id": 42, "data": {"value": 7}},
                    {"participant_id": 10, "type_id": 5305, "data": {"value": 1.8}},
                    {"participant_id": 20, "type_id": 5305, "data": {"value": 1.1}},
                ],
                "xgfixture": [
                    {"participant_id": 10, "type_id": 5304, "data": {"value": 1.6}},
                    {"participant_id": 20, "type_id": 5304, "data": {"value": 0.9}},
                ],
                "lineups": [
                    {
                        "id": 1,
                        "player_id": 1000,
                        "player_name": "Home Keeper",
                        "team_id": 10,
                        "position_id": 24,
                        "type_id": 11,
                        "player": {"id": 1000, "name": "Home Keeper", "position_id": 24},
                        "details": [
                            {"type_id": 119, "data": {"value": 90}},
                            {"type_id": 57, "data": {"value": 3}},
                            {"type_id": 1535, "data": {"value": 1}},
                        ],
                    },
                    {
                        "id": 2,
                        "player_id": 1001,
                        "player_name": "Home Forward",
                        "team_id": 10,
                        "position_id": 27,
                        "type_id": 11,
                        "player": {"id": 1001, "name": "Home Forward", "position_id": 27},
                        "details": [
                            {"type_id": 119, "data": {"value": 90}},
                            {"type_id": 52, "data": {"value": 2}},
                            {"type_id": 86, "data": {"value": 3}},
                        ],
                        "xglineup": [
                            {"type_id": 5304, "data": {"value": 1.2}},
                        ],
                    },
                ],
                "events": [],
            }
        }
        normalizer.add_fixture(fixture, context)
        frames = normalizer.frames()

        self.assertEqual(len(frames["team_matchlogs"]), 2)
        self.assertEqual(len(frames["player_matchlogs"]), 2)
        home_team = frames["team_stats"].loc[lambda df: df["Squad"] == "Home"].iloc[0]
        self.assertEqual(home_team["MP"], 1)
        self.assertEqual(home_team["Gls"], 2)
        self.assertAlmostEqual(home_team["xG"], 1.6)
        self.assertAlmostEqual(home_team["Gls_per_90"], 2)
        self.assertAlmostEqual(home_team["xGA_per_90"], 0.9)
        self.assertAlmostEqual(home_team["Shots_Against_per_90"], 7)

        players = frames["player_stats_compat"].set_index("Player")
        self.assertAlmostEqual(players.loc["Home Keeper", "PSxG"], 1.1)
        self.assertAlmostEqual(players.loc["Home Keeper", "PSxG+/-"], 0.1)
        self.assertAlmostEqual(players.loc["Home Keeper", "xGoT_minus_GA_per_90"], 0.1)
        self.assertEqual(players.loc["Home Forward", "PSxG"], 0)
        self.assertEqual(players.loc["Home Forward", "Gls_per_90"], 2)
        self.assertAlmostEqual(players.loc["Home Forward", "xG_per_90"], 1.2)
        self.assertTrue(pd.isna(players.loc["Home Forward", "SCA90"]))

    def test_fixture_index_rows_do_not_inflate_aggregates(self):
        normalizer = SeasonNormalizer("2025-2026")
        context = {"league_id": 8, "league": "Premier League", "season_id": 25583}
        normalizer.add_fixture({"data": {
            "id": 200,
            "participants": [
                {"id": 10, "name": "Home", "meta": {"location": "home"}},
                {"id": 20, "name": "Away", "meta": {"location": "away"}},
            ],
            "scores": [
                {"participant_id": 10, "description": "CURRENT", "score": {"goals": 3}},
                {"participant_id": 20, "description": "CURRENT", "score": {"goals": 0}},
            ],
            "state": {"short_name": "FT"},
        }}, context)
        frames = normalizer.frames()
        self.assertEqual(len(frames["team_matchlogs"]), 2)
        self.assertTrue(frames["team_stats"].empty)

    def test_sportmonks_ui_registry_uses_supported_columns_and_direction(self):
        unsupported = {"SCA90", "PrgP_per_90", "Carries_F3_per_90", "Stp%", "#OPA/90", "Ast_xAG_ratio"}
        specs = [
            spec
            for groups in (TEAM_METRIC_GROUPS, PLAYER_METRIC_GROUPS)
            for group in groups.values()
            for spec in group
        ]
        self.assertFalse(unsupported.intersection(spec.column for spec in specs))
        lower_team = {spec.column for spec in TEAM_METRIC_GROUPS["defending"] if spec.ascending}
        self.assertEqual(
            lower_team,
            {"GA_per_90", "xGA_per_90", "Shots_Against_per_90", "Global_PPDA_Proxy"},
        )
        lower_goalkeeper = {spec.column for spec in PLAYER_METRIC_GROUPS["goalkeeping"] if spec.ascending}
        self.assertEqual(lower_goalkeeper, {"GA_per_90"})

        self.assertEqual(
            set(TEAM_RADAR_GROUPS),
            {"Attacking", "Possession & Territory", "Defending & Pressing", "Set Pieces"},
        )
        radar_columns = {
            column
            for group in TEAM_RADAR_GROUPS.values()
            for column in group.values()
        }
        self.assertFalse(unsupported.intersection(radar_columns))
        self.assertTrue(
            {
                "Global_PPDA_Proxy",
                "Set_Piece_xG_per_90",
                "Set_Piece_xGA_per_90",
            }.issubset(radar_columns)
        )


if __name__ == "__main__":
    unittest.main()
