from __future__ import annotations

import unittest
from pathlib import Path

from src.providers.sportmonks.config import LEAGUES, season_name_matches
from src.providers.sportmonks.importer import SportmonksImporter


class _RecordingCache:
    def __init__(self, primary_payload):
        self.primary_payload = primary_payload
        self.paths = []

    def get(self, path, loader, **kwargs):
        del kwargs
        path = Path(path)
        self.paths.append(path)

        if "season_index" not in path.parts:
            return self.primary_payload

        return loader()


class _PagedClient:
    def __init__(self, pages=None):
        self.pages = pages or {}
        self.calls = []

    def get(self, endpoint, **params):
        self.calls.append((endpoint, dict(params)))
        page = int(params.get("page", 1))
        if page not in self.pages:
            raise AssertionError(f"Unexpected Sportmonks page requested: {page}")
        return self.pages[page]


def _importer(primary_payload, *, pages=None):
    importer = SportmonksImporter.__new__(SportmonksImporter)
    importer.season = "2023-2024"
    importer.start_year = 2023
    importer.end_year = 2024
    importer.cache = _RecordingCache(primary_payload)
    importer.client = _PagedClient(pages)
    return importer


class SportmonksSeasonResolutionTests(unittest.TestCase):
    def setUp(self):
        self.league = LEAGUES["premier-league"]

    def test_primary_league_relation_remains_first_choice(self):
        target = {
            "id": 202324,
            "name": "2023-2024",
            "starting_at": "2023-08-11",
            "ending_at": "2024-05-19",
        }
        importer = _importer(
            {
                "data": {
                    "seasons": [target],
                }
            }
        )

        result = importer._resolve_season(self.league)

        self.assertEqual(result["id"], 202324)
        self.assertEqual(importer.client.calls, [])
        self.assertFalse(
            any("season_index" in path.parts for path in importer.cache.paths)
        )

    def test_fallback_is_paginated_filtered_and_cached_per_league(self):
        importer = _importer(
            {"data": {"seasons": []}},
            pages={
                1: {
                    "data": [
                        {
                            "id": 202223,
                            "name": "2022/2023",
                            "starting_at": "2022-08-05",
                            "ending_at": "2023-05-28",
                        }
                    ],
                    "pagination": {
                        "current_page": 1,
                        "has_more": True,
                        "next_page": 2,
                    },
                },
                2: {
                    "data": [
                        {
                            "id": 202324,
                            # Deliberately no years in the label:
                            # this verifies matching by start/end dates.
                            "name": "Premier League",
                            "starting_at": "2023-08-11",
                            "ending_at": "2024-05-19",
                        }
                    ],
                    "pagination": {
                        "current_page": 2,
                        "has_more": False,
                        "next_page": None,
                    },
                },
            },
        )

        result = importer._resolve_season(self.league)

        self.assertEqual(result["id"], 202324)
        self.assertEqual(
            [call[0] for call in importer.client.calls],
            ["seasons", "seasons"],
        )
        self.assertEqual(
            [call[1]["page"] for call in importer.client.calls],
            [1, 2],
        )
        for _, params in importer.client.calls:
            self.assertEqual(
                params["filters"],
                f"seasonLeagues:{self.league.league_id}",
            )
            self.assertEqual(params["per_page"], 50)

        cache_paths = [
            str(path).replace("\\", "/")
            for path in importer.cache.paths
            if "season_index" in path.parts
        ]
        self.assertTrue(
            cache_paths[0].endswith(
                "premier-league/season_index/page_001.json"
            )
        )
        self.assertTrue(
            cache_paths[1].endswith(
                "premier-league/season_index/page_002.json"
            )
        )

    def test_alternative_season_name_formats_and_league_prefixes(self):
        matching = (
            "2023-2024",
            "2023/2024",
            "2023-24",
            "Premier League 2023/2024",
            "Premier League · 2023-24",
        )
        for value in matching:
            with self.subTest(value=value):
                self.assertTrue(season_name_matches(value, 2023, 2024))

        # Preserve the old single-year behaviour.
        self.assertTrue(season_name_matches("2023", 2023, 2024))
        self.assertFalse(season_name_matches("2023-2025", 2023, 2024))

    def test_fallback_matches_league_prefixed_alternative_name(self):
        importer = _importer(
            {"data": {"seasons": []}},
            pages={
                1: {
                    "data": [
                        {
                            "id": 202324,
                            "name": "Premier League 2023/2024",
                        }
                    ],
                    "pagination": {
                        "current_page": 1,
                        "has_more": False,
                        "next_page": None,
                    },
                }
            },
        )

        result = importer._resolve_season(self.league)

        self.assertEqual(result["id"], 202324)

    def test_unavailable_season_mentions_subscription_and_available_names(self):
        importer = _importer(
            {"data": {"seasons": []}},
            pages={
                1: {
                    "data": [
                        {
                            "id": 202122,
                            "name": "2021/2022",
                            "starting_at": "2021-08-13",
                            "ending_at": "2022-05-22",
                        },
                        {
                            "id": 202223,
                            "name": "Premier League 2022/2023",
                            "starting_at": "2022-08-05",
                            "ending_at": "2023-05-28",
                        },
                    ],
                    "pagination": {
                        "current_page": 1,
                        "has_more": False,
                        "next_page": None,
                    },
                }
            },
        )

        with self.assertRaises(RuntimeError) as ctx:
            importer._resolve_season(self.league)

        message = str(ctx.exception)
        self.assertIn("Season 2023-2024", message)
        self.assertIn("Sportmonks subscription", message)
        self.assertIn("Premier League", message)
        self.assertIn("2021/2022", message)
        self.assertIn("Premier League 2022/2023", message)
        self.assertIn("Available seasons:", message)


if __name__ == "__main__":
    unittest.main()
