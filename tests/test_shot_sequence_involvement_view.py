import unittest
from pathlib import Path

import pandas as pd

from src.metrics import (
    shot_sequence_involvement_metrics,
)
from src.components import (
    shot_sequence_involvement_view,
)


class ShotSequenceInvolvementViewTests(
    unittest.TestCase
):
    def _stats(self):
        return pd.DataFrame(
            {
                "Shot Sequence Involvements":
                    [5, 4, 4],
                "Shot Sequence Shots":
                    [4, 3, 1],
                "Shot Sequence Shot Assists":
                    [0, 0, 2],
                "Shot Sequence Pre-Assists":
                    [1, 1, 1],
            },
            index=[
                "Baldanzi",
                "Alisson",
                "Creator",
            ],
        )

    def _events(self):
        return pd.DataFrame(
            [
                {
                    "playerName":
                        "Baldanzi",
                    "team_name":
                        "Home",
                    "Mapped Jersey Number":
                        8,
                },
                {
                    "playerName":
                        "Alisson",
                    "team_name":
                        "Away",
                    "Mapped Jersey Number":
                        27,
                },
                {
                    "playerName":
                        "Creator",
                    "team_name":
                        "Home",
                    "Mapped Jersey Number":
                        10,
                },
            ]
        )

    def test_ranking_uses_rel08_involvements_not_weighted_score(self):
        stats = self._stats()
        stats[
            "Weighted Score"
        ] = [
            1,
            999,
            500,
        ]

        ranking = (
            shot_sequence_involvement_metrics
            .prepare_shot_sequence_ranking(
                stats,
                num_players=3,
            )
        )

        self.assertEqual(
            ranking.index.tolist(),
            [
                "Baldanzi",
                "Alisson",
                "Creator",
            ],
        )

    def test_component_keeps_three_roles_separate(self):
        ranking = (
            shot_sequence_involvement_metrics
            .prepare_shot_sequence_ranking(
                self._stats()
            )
        )

        component = (
            shot_sequence_involvement_view
            .table(
                ranking,
                self._events(),
                "Home",
                hcol="#e96a4a",
                acol="#1597c2",
            )
        )

        text = str(
            component
        )

        self.assertIn(
            "Shots",
            text,
        )
        self.assertIn(
            "Shot assists",
            text,
        )
        self.assertIn(
            "Pre-assists",
            text,
        )
        self.assertIn(
            "no weighted score",
            text,
        )

    def test_app_shot_stats_branch_no_longer_calls_legacy_stacked_plot(self):
        source = Path(
            "app.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "def render_shooting_analysis_content("
        )

        end_marker = (
            "\n# --- 3."
        )

        end = source.find(
            end_marker,
            start,
        )

        if end == -1:
            end = len(
                source
            )

        block = source[
            start:end
        ]

        self.assertIn(
            "calculate_shot_sequence_player_stats",
            block,
        )
        self.assertIn(
            "shot_sequence_involvement_view",
            block,
        )
        self.assertNotIn(
            "plot_shot_sequence_bar_plotly(",
            block,
        )


    def test_rel08_sequence_assist_alias_is_canonicalised(self):
        stats = self._stats().drop(
            columns=[
                "Shot Sequence Shot Assists"
            ]
        )

        stats[
            "Shot Sequence Assists"
        ] = [
            0,
            0,
            2,
        ]

        ranking = (
            shot_sequence_involvement_metrics
            .prepare_shot_sequence_ranking(
                stats,
                num_players=3,
            )
        )

        self.assertIn(
            "Shot Sequence Shot Assists",
            ranking.columns,
        )

        self.assertEqual(
            ranking.loc[
                "Creator",
                "Shot Sequence Shot Assists",
            ],
            2,
        )

    def test_legacy_generic_shot_assists_are_not_used_as_rel08_fallback(self):
        stats = self._stats().drop(
            columns=[
                "Shot Sequence Shot Assists"
            ]
        )

        stats[
            "Shot Assists"
        ] = [
            99,
            99,
            99,
        ]

        with self.assertRaises(
            ValueError
        ):
            (
                shot_sequence_involvement_metrics
                .prepare_shot_sequence_ranking(
                    stats,
                    num_players=3,
                )
            )

    def test_ui_copy_does_not_expose_internal_rel_codes(self):
        ranking = (
            shot_sequence_involvement_metrics
            .prepare_shot_sequence_ranking(
                self._stats()
            )
        )

        component = (
            shot_sequence_involvement_view
            .table(
                ranking,
                self._events(),
                "Home",
                hcol="#e96a4a",
                acol="#1597c2",
            )
        )

        text = str(component)

        self.assertNotIn(
            "REL-08",
            text,
        )
        self.assertNotIn(
            "PLOT-15",
            text,
        )
        self.assertIn(
            "Each shot sequence counts once per player",
            text,
        )

if __name__ == "__main__":
    unittest.main()
