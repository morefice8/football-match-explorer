import csv
import os
import tempfile
import unittest
from pathlib import Path

from src.utils import path_helpers


class GetTeamLogoPathTests(unittest.TestCase):
    def setUp(self):
        self._original_cwd = os.getcwd()
        self._tmp_dir = tempfile.TemporaryDirectory()
        os.chdir(self._tmp_dir.name)
        path_helpers._get_sportmonks_team_logo.cache_clear()
        path_helpers._get_sportmonks_team_logo_by_name.cache_clear()

    def tearDown(self):
        os.chdir(self._original_cwd)
        self._tmp_dir.cleanup()
        path_helpers._get_sportmonks_team_logo.cache_clear()
        path_helpers._get_sportmonks_team_logo_by_name.cache_clear()

    def _write_teams_csv(self, rows):
        season_dir = Path("data") / "sportmonks" / "processed" / "2025-2026"
        season_dir.mkdir(parents=True, exist_ok=True)
        with (season_dir / "teams.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["league", "team_name", "image_path"])
            writer.writeheader()
            writer.writerows(rows)

    def test_direct_league_match_resolves(self):
        self._write_teams_csv([
            {"league": "Serie A", "team_name": "Inter", "image_path": "http://example.test/inter.png"},
        ])
        result = path_helpers.get_team_logo_path("Serie A", "Inter")
        self.assertEqual(result, "http://example.test/inter.png")

    def test_non_domestic_competition_falls_back_by_team_name(self):
        # Same badge data as above, but the match's own competition is a
        # European competition, not one of the five domestic leagues this
        # app has badge data for -- the team's badge still lives under its
        # real domestic league.
        self._write_teams_csv([
            {"league": "Serie A", "team_name": "Inter", "image_path": "http://example.test/inter.png"},
        ])
        result = path_helpers.get_team_logo_path("UEFA Champions League", "Inter")
        self.assertEqual(result, "http://example.test/inter.png")

    def test_team_with_no_badge_anywhere_gets_default(self):
        self._write_teams_csv([
            {"league": "Serie A", "team_name": "Inter", "image_path": "http://example.test/inter.png"},
        ])
        result = path_helpers.get_team_logo_path("UEFA Champions League", "Some Nonexistent FC")
        self.assertEqual(result, path_helpers.DEFAULT_LOGO_PATH)

    def test_no_sportmonks_data_at_all_gets_default(self):
        result = path_helpers.get_team_logo_path("UEFA Champions League", "Inter")
        self.assertEqual(result, path_helpers.DEFAULT_LOGO_PATH)


if __name__ == "__main__":
    unittest.main()
