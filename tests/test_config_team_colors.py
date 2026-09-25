import unittest

from src import config


class GetMatchTeamColorsTests(unittest.TestCase):
    def test_mapped_teams_use_their_own_colors(self):
        home_color, away_color = config.get_match_team_colors("Napoli", "Bologna")

        self.assertEqual(home_color, config.TEAM_NAME_TO_COLOR["Napoli"])
        self.assertEqual(away_color, config.TEAM_NAME_TO_COLOR["Bologna"])

    def test_unmapped_team_falls_back_to_default(self):
        home_color, away_color = config.get_match_team_colors("Frosinone", "Some Unlisted FC")

        self.assertEqual(home_color, config.TEAM_NAME_TO_COLOR["Frosinone"])
        self.assertEqual(away_color, config.DEFAULT_ACOL)

    def test_both_unmapped_falls_back_to_defaults(self):
        home_color, away_color = config.get_match_team_colors("Unlisted Home FC", "Unlisted Away FC")

        self.assertEqual(home_color, config.DEFAULT_HCOL)
        self.assertEqual(away_color, config.DEFAULT_ACOL)

    def test_color_collision_is_resolved(self):
        # Both clubs use black as their primary color.
        home_color, away_color = config.get_match_team_colors("Juventus", "Udinese")

        self.assertNotEqual(home_color.lower(), away_color.lower())
        self.assertEqual(home_color, config.TEAM_NAME_TO_COLOR["Juventus"])

    def test_collision_with_default_away_still_resolves(self):
        # An unmapped home team happens to default to the same color the
        # away team is mapped to (contrived, but keeps the guard honest).
        original = dict(config.TEAM_NAME_TO_COLOR)
        try:
            config.TEAM_NAME_TO_COLOR["Away Test FC"] = config.DEFAULT_HCOL
            home_color, away_color = config.get_match_team_colors(
                "Unlisted Home FC", "Away Test FC"
            )
            self.assertNotEqual(home_color.lower(), away_color.lower())
        finally:
            config.TEAM_NAME_TO_COLOR.clear()
            config.TEAM_NAME_TO_COLOR.update(original)


if __name__ == "__main__":
    unittest.main()
