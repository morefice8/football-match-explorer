import unittest
from pathlib import Path

import pandas as pd

from src.components import defensive_shape_view
from src.metrics import defensive_metrics
from src.visualization import defensive_transitions_plotly


def _row(
    minute,
    player,
    x,
    y,
    jersey,
    period=1,
    event_type="Tackle",
):
    return {
        "team_name": "Home",
        "type_name": event_type,
        "x": x,
        "y": y,
        "expandedMinute": minute,
        "periodId": period,
        "playerName": player,
        "Mapped Jersey Number":
            jersey,
        "id":
            f"{minute}-{player}-{x}-{y}",
    }


class DefensiveShapeViewTests(
    unittest.TestCase
):
    def _sample(self):
        rows = []

        for (
            player,
            jersey,
            x,
            y,
        ) in [
            ("A", 2, 35, 20),
            ("B", 4, 38, 40),
            ("C", 5, 40, 60),
            ("D", 8, 45, 80),
        ]:
            rows.append(
                _row(
                    10,
                    player,
                    x,
                    y,
                    jersey,
                    period=1,
                )
            )

        for (
            player,
            jersey,
            x,
            y,
        ) in [
            ("A", 2, 36, 22),
            ("B", 4, 39, 42),
            ("C", 5, 41, 62),
            ("D", 8, 46, 78),
        ]:
            rows.append(
                _row(
                    12,
                    player,
                    x,
                    y,
                    jersey,
                    period=1,
                )
            )

        for (
            player,
            jersey,
            x,
            y,
        ) in [
            ("E", 11, 70, 15),
            ("F", 14, 72, 35),
            ("G", 17, 74, 55),
            ("H", 20, 76, 75),
        ]:
            rows.append(
                _row(
                    55,
                    player,
                    x,
                    y,
                    jersey,
                    period=2,
                )
            )

        return pd.DataFrame(
            rows
        )

    def test_controls_expose_density_shape_and_period(self):
        control = (
            defensive_shape_view
            .controls()
        )
        text = str(
            control.to_plotly_json()
        )

        self.assertIn(
            "defensive-shape-period",
            text,
        )
        self.assertIn(
            "Full Match",
            text,
        )
        self.assertIn(
            "1H",
            text,
        )
        self.assertIn(
            "2H",
            text,
        )
        self.assertIn(
            "defensive-shape-mode",
            text,
        )
        self.assertIn(
            "Density",
            text,
        )
        self.assertIn(
            "Shape",
            text,
        )

    def test_first_half_shape_excludes_second_half_players(self):
        profile = (
            defensive_metrics
            .build_defensive_shape_profile(
                self._sample(),
                "Home",
                period="1h",
                window_minutes=5,
                min_outfield_players=4,
            )
        )

        representative = profile[
            "representative"
        ]

        self.assertIsNotNone(
            representative
        )

        players = set(
            representative[
                "player_locations"
            ][
                "player_name"
            ]
        )

        self.assertEqual(
            players,
            {
                "A",
                "B",
                "C",
                "D",
            },
        )

    def test_full_match_shape_is_one_real_window_not_whole_match_hull(self):
        profile = (
            defensive_metrics
            .build_defensive_shape_profile(
                self._sample(),
                "Home",
                period="full",
                window_minutes=5,
                min_outfield_players=4,
            )
        )

        representative = profile[
            "representative"
        ]

        self.assertIsNotNone(
            representative
        )

        players = set(
            representative[
                "player_locations"
            ][
                "player_name"
            ]
        )

        self.assertTrue(
            players
            in (
                {
                    "A",
                    "B",
                    "C",
                    "D",
                },
                {
                    "E",
                    "F",
                    "G",
                    "H",
                },
            )
        )

        self.assertEqual(
            len(players),
            4,
        )

    def test_shape_rejects_window_containing_substitution(self):
        sample = self._sample()

        substitution = {
            "team_name": "Home",
            "type_name":
                "Substitution Off",
            "typeId": 18,
            "x": 50,
            "y": 50,
            "expandedMinute": 11,
            "periodId": 1,
            "playerName": "A",
            "Mapped Jersey Number": 2,
            "id": "sub-11",
        }

        sample = pd.concat(
            [
                sample,
                pd.DataFrame(
                    [substitution]
                ),
            ],
            ignore_index=True,
        )

        profile = (
            defensive_metrics
            .build_defensive_shape_profile(
                sample,
                "Home",
                period="1h",
                window_minutes=15,
                min_outfield_players=4,
            )
        )

        representative = profile[
            "representative"
        ]

        if representative is not None:
            self.assertFalse(
                (
                    representative[
                        "window_start"
                    ]
                    <= 11
                    < representative[
                        "window_end"
                    ]
                )
            )

    def test_profile_exposes_requested_kpis(self):
        profile = (
            defensive_metrics
            .build_defensive_shape_profile(
                self._sample(),
                "Home",
                period="full",
                window_minutes=5,
                min_outfield_players=4,
            )
        )

        self.assertGreater(
            profile[
                "action_count"
            ],
            0,
        )
        self.assertIsNotNone(
            profile[
                "block_height_m"
            ]
        )
        self.assertIsNotNone(
            profile[
                "width_m"
            ]
        )
        self.assertIsNotNone(
            profile[
                "compactness_m"
            ]
        )
        self.assertGreater(
            profile[
                "snapshot_count"
            ],
            0,
        )

    def test_density_plot_uses_all_selected_actions(self):
        profile = (
            defensive_metrics
            .build_defensive_shape_profile(
                self._sample(),
                "Home",
                period="1h",
                window_minutes=5,
                min_outfield_players=4,
            )
        )

        fig = (
            defensive_transitions_plotly
            .plot_defensive_shape_profile(
                profile,
                "#e96a4a",
                mode="density",
            )
        )

        scatter = next(
            trace
            for trace
            in fig.data
            if getattr(
                trace,
                "mode",
                None,
            )
            == "markers"
        )

        self.assertEqual(
            len(scatter.x),
            profile[
                "action_count"
            ],
        )

    def test_shape_plot_uses_representative_window_players(self):
        profile = (
            defensive_metrics
            .build_defensive_shape_profile(
                self._sample(),
                "Home",
                period="1h",
                window_minutes=5,
                min_outfield_players=4,
            )
        )

        fig = (
            defensive_transitions_plotly
            .plot_defensive_shape_profile(
                profile,
                "#e96a4a",
                mode="shape",
            )
        )

        marker = next(
            trace
            for trace
            in fig.data
            if getattr(
                trace,
                "mode",
                None,
            )
            == "markers+text"
        )

        self.assertEqual(
            len(marker.x),
            profile[
                "representative"
            ][
                "player_count"
            ],
        )

    def test_app_removed_legacy_defensive_hull_tab(self):
        source = Path(
            "app.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            (
                'dbc.Tab('
                'label="Defensive Shape", '
                'tab_id="def_shape")'
            ),
            source,
        )
        self.assertNotIn(
            (
                'dbc.Tab('
                'label="Defensive Hull", '
                'tab_id="def_hull")'
            ),
            source,
        )
        self.assertNotIn(
            (
                "plot_defensive_hull_plotly("
                "df_home_agg"
            ),
            source,
        )

    def test_app_exposes_defensive_shape_callback(self):
        source = Path(
            "app.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '"defensive-shape-content"',
            source,
        )
        self.assertIn(
            '"children"',
            source,
        )
        self.assertIn(
            "render_defensive_shape_content",
            source,
        )
        self.assertIn(
            "build_defensive_shape_profile",
            source,
        )
        self.assertIn(
            "plot_defensive_shape_profile",
            source,
        )


if __name__ == "__main__":
    unittest.main()
