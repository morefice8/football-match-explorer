import unittest
from pathlib import Path

import pandas as pd

from src.metrics import player_metrics
from src.visualization import formation_plotly


class MeanPositionsProfileTests(unittest.TestCase):

    @staticmethod
    def sample_df():
        rows = []

        def add(
            player_id,
            player_name,
            *,
            team="Home",
            type_id=1,
            type_name="Pass",
            period_id=1,
            minute=0,
            second=0,
            x=50,
            y=50,
            starter=True,
            jersey=8,
            role="CM",
            **extra,
        ):
            row = {
                "id": len(rows) + 1,
                "eventId": len(rows) + 100,
                "typeId": type_id,
                "type_name": type_name,
                "periodId": period_id,
                "timeMin": minute,
                "timeSec": second,
                "team_name": team,
                "playerId": player_id,
                "playerName": player_name,
                "x": x,
                "y": y,
                "Is Starter": starter,
                "Mapped Jersey Number": jersey,
                "positional_role": role,
                "Red card": 0,
                "Second yellow": 0,
            }
            row.update(extra)
            rows.append(row)

        # Match runs to 90 minutes and has explicit period coverage.
        add(
            "h1",
            "Keeper",
            minute=1,
            x=8,
            y=50,
            jersey=1,
            role="GK",
        )
        add(
            "h1",
            "Keeper",
            period_id=2,
            minute=89,
            x=12,
            y=50,
            jersey=1,
            role="GK",
        )

        # Three full-match outfield players.
        for minute, x, y in (
            (5, 35, 30),
            (20, 40, 35),
            (40, 45, 40),
            (50, 50, 45),
            (70, 55, 50),
            (88, 60, 55),
        ):
            add(
                "h8",
                "Midfielder",
                period_id=1 if minute < 45 else 2,
                minute=minute,
                x=x,
                y=y,
                jersey=8,
                role="CM",
            )

        for minute, x, y in (
            (4, 55, 20),
            (25, 58, 25),
            (42, 62, 30),
            (52, 65, 35),
            (73, 70, 40),
            (87, 72, 45),
        ):
            add(
                "h10",
                "Forward",
                period_id=1 if minute < 45 else 2,
                minute=minute,
                x=x,
                y=y,
                jersey=10,
                role="ST",
            )

        # Player off at 60.
        add(
            "h6",
            "Holder",
            type_id=1,
            type_name="Pass",
            period_id=1,
            minute=10,
            x=42,
            y=62,
            jersey=6,
            role="DM",
        )
        add(
            "h6",
            "Holder",
            type_id=1,
            type_name="Pass",
            period_id=2,
            minute=55,
            x=46,
            y=65,
            jersey=6,
            role="DM",
        )
        add(
            "h6",
            "Holder",
            type_id=18,
            type_name="Player off",
            period_id=2,
            minute=60,
            x=None,
            y=None,
            jersey=6,
            role="DM",
        )

        # Substitute gets 30 full-match minutes, 30 second-half minutes.
        add(
            "h16",
            "Substitute",
            type_id=19,
            type_name="Player on",
            period_id=2,
            minute=60,
            x=None,
            y=None,
            starter=False,
            jersey=16,
            role="CM",
        )
        for minute, x, y in (
            (62, 52, 60),
            (75, 56, 64),
            (89, 60, 68),
        ):
            add(
                "h16",
                "Substitute",
                period_id=2,
                minute=minute,
                x=x,
                y=y,
                starter=False,
                jersey=16,
                role="CM",
            )

        # Late substitute has < 15 minutes and must be excluded.
        add(
            "h20",
            "Late Sub",
            type_id=19,
            type_name="Player on",
            period_id=2,
            minute=80,
            x=None,
            y=None,
            starter=False,
            jersey=20,
            role="RW",
        )
        add(
            "h20",
            "Late Sub",
            period_id=2,
            minute=85,
            x=78,
            y=80,
            starter=False,
            jersey=20,
            role="RW",
        )

        return pd.DataFrame(rows)

    def test_profile_uses_median_locations_and_touch_share(self):
        profile, summary = (
            player_metrics.get_mean_positions_profile(
                self.sample_df(),
                "Home",
                period="full",
                min_minutes=15,
            )
        )

        midfielder = profile[
            profile["playerName"].eq(
                "Midfielder"
            )
        ].iloc[0]

        self.assertEqual(
            midfielder["median_x"],
            47.5,
        )
        self.assertEqual(
            midfielder["median_y"],
            42.5,
        )
        self.assertGreater(
            midfielder["touch_share"],
            0,
        )
        self.assertGreater(
            midfielder["dispersion_m"],
            0,
        )
        self.assertGreater(
            summary["team_length_m"],
            0,
        )
        self.assertGreater(
            summary["team_width_m"],
            0,
        )
        self.assertGreater(
            summary["team_compactness_m"],
            0,
        )

    def test_minimum_minutes_and_slot_representative_exclude_late_substitute(self):
        profile, _ = (
            player_metrics.get_mean_positions_profile(
                self.sample_df(),
                "Home",
                period="full",
                min_minutes=15,
            )
        )

        names = profile[
            "playerName"
        ].tolist()

        self.assertNotIn(
            "Late Sub",
            names,
        )

        # Full Match: Holder played longer than the player who replaced him,
        # therefore this substitution chain contributes Holder only.
        self.assertIn(
            "Holder",
            names,
        )
        self.assertNotIn(
            "Substitute",
            names,
        )

    def test_first_half_excludes_second_half_substitute(self):
        profile, summary = (
            player_metrics.get_mean_positions_profile(
                self.sample_df(),
                "Home",
                period="1h",
                min_minutes=15,
            )
        )

        self.assertNotIn(
            "Substitute",
            profile["playerName"].tolist(),
        )
        self.assertEqual(
            summary["period"],
            "1h",
        )

    def test_second_half_minutes_include_substitute(self):
        profile, summary = (
            player_metrics.get_mean_positions_profile(
                self.sample_df(),
                "Home",
                period="2h",
                min_minutes=15,
            )
        )

        substitute = profile[
            profile["playerName"].eq(
                "Substitute"
            )
        ].iloc[0]

        self.assertGreaterEqual(
            substitute["minutes_played"],
            29,
        )
        self.assertNotIn(
            "Holder",
            profile["playerName"].tolist(),
        )
        self.assertEqual(
            summary["period"],
            "2h",
        )

    def test_goalkeeper_is_excluded_from_structural_centroid(self):
        profile, summary = (
            player_metrics.get_mean_positions_profile(
                self.sample_df(),
                "Home",
                period="full",
                min_minutes=15,
            )
        )

        outfield = profile[
            ~profile["positional_role"].eq(
                "GK"
            )
        ]

        self.assertAlmostEqual(
            summary["centroid_x"],
            outfield["median_x"].mean(),
        )

    def test_team_compactness_is_median_outfield_distance_from_team_centre(self):
        profile, summary = (
            player_metrics.get_mean_positions_profile(
                self.sample_df(),
                "Home",
                period="full",
                min_minutes=15,
            )
        )

        outfield = profile[
            ~profile["positional_role"].eq(
                "GK"
            )
        ]

        distances_m = []
        for _, row in outfield.iterrows():
            dx_m = (
                float(row["median_x"])
                - summary["centroid_x"]
            ) * 1.05
            dy_m = (
                float(row["median_y"])
                - summary["centroid_y"]
            ) * 0.68
            distances_m.append(
                (dx_m ** 2 + dy_m ** 2) ** 0.5
            )

        expected = float(
            pd.Series(distances_m).median()
        )

        self.assertAlmostEqual(
            summary["team_compactness_m"],
            expected,
        )

    def test_mean_positions_ui_uses_reader_friendly_compactness_kpi(self):
        source = Path(
            "app.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '"Team compactness"',
            source,
        )
        self.assertIn(
            '"team_compactness_m"',
            source,
        )
        self.assertIn(
            'detail="Typical distance from team centre"',
            source,
        )
        self.assertNotIn(
            '"Centroid",\n                centroid_label',
            source,
        )

    def test_plot_encodes_dispersion_and_touch_share(self):
        profile, summary = (
            player_metrics.get_mean_positions_profile(
                self.sample_df(),
                "Home",
                period="full",
                min_minutes=15,
            )
        )

        figure = (
            formation_plotly
            .plot_mean_positions_profile_plotly(
                profile,
                summary,
                is_away=False,
            )
        )

        # Player scatter + centroid.
        self.assertGreaterEqual(
            len(figure.data),
            2,
        )

        player_trace = figure.data[0]

        self.assertEqual(
            len(player_trace.x),
            len(profile),
        )
        self.assertEqual(
            len(player_trace.marker.size),
            len(profile),
        )

        ellipse_shapes = [
            shape
            for shape in figure.layout.shapes
            if shape.type == "circle"
        ]

        self.assertGreaterEqual(
            len(ellipse_shapes),
            len(profile) + 1,
        )

    def test_active_mean_positions_branch_uses_period_workspace(self):
        source = Path(
            "app.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.find(
            "        elif active_tab == 'mean_positions':"
        )
        end = source.find(
            "    except Exception as e:",
            start,
        )

        self.assertNotEqual(
            start,
            -1,
        )
        self.assertNotEqual(
            end,
            -1,
        )

        branch = source[
            start:end
        ]

        self.assertIn(
            '"mean-positions-period-selector"',
            branch,
        )
        self.assertIn(
            '"mean-positions-period-content"',
            branch,
        )
        self.assertNotIn(
            "plot_mean_positions_plotly",
            branch,
        )
        self.assertIn(
            "aggregated territorial profile",
            branch,
        )
        self.assertIn(
            "not a formation",
            branch,
        )
        self.assertIn(
            "Territorial occupation",
            branch,
        )


    def test_dense_labels_use_multiple_annotation_offsets(self):
        profile, summary = (
            player_metrics.get_mean_positions_profile(
                self.sample_df(),
                "Home",
                period="full",
                min_minutes=15,
            )
        )

        dense_profile = profile.copy()

        dense_profile.loc[
            dense_profile.index[:4],
            "median_x",
        ] = [
            50.0,
            52.0,
            54.0,
            55.0,
        ]

        dense_profile.loc[
            dense_profile.index[:4],
            "median_y",
        ] = [
            48.0,
            50.0,
            52.0,
            54.0,
        ]

        figure = (
            formation_plotly
            .plot_mean_positions_profile_plotly(
                dense_profile,
                summary,
                is_away=False,
            )
        )

        player_labels = [
            annotation
            for annotation in figure.layout.annotations
            if annotation.bgcolor == "rgba(16,47,69,0.92)"
        ]

        offsets = {
            (
                annotation.xshift,
                annotation.yshift,
            )
            for annotation in player_labels
        }

        self.assertGreater(
            len(offsets),
            1,
        )



    def test_substitution_chain_never_shows_both_players(self):
        for period in (
            "full",
            "1h",
            "2h",
        ):
            profile, _ = (
                player_metrics.get_mean_positions_profile(
                    self.sample_df(),
                    "Home",
                    period=period,
                    min_minutes=15,
                )
            )

            names = set(
                profile[
                    "playerName"
                ].tolist()
            )

            self.assertFalse(
                {
                    "Holder",
                    "Substitute",
                }.issubset(
                    names
                )
            )

    def test_compound_surname_particles_are_preserved(self):
        self.assertEqual(
            formation_plotly._mean_positions_compact_name(
                "Giovanni Di Lorenzo"
            ),
            "Di Lorenzo",
        )
        self.assertEqual(
            formation_plotly._mean_positions_compact_name(
                "Kevin De Bruyne"
            ),
            "De Bruyne",
        )
        self.assertEqual(
            formation_plotly._mean_positions_compact_name(
                "Donny van de Beek"
            ),
            "van de Beek",
        )



    def test_mean_positions_plot_preserves_compound_surnames_in_labels(self):
        profile = pd.DataFrame(
            [
                {
                    "playerName": "Giovanni Di Lorenzo",
                    "Mapped Jersey Number": 22,
                    "positional_role": "RB",
                    "median_x": 55.0,
                    "median_y": 22.0,
                    "q25_x": 50.0,
                    "q75_x": 60.0,
                    "q25_y": 18.0,
                    "q75_y": 26.0,
                    "iqr_x": 10.0,
                    "iqr_y": 8.0,
                    "dispersion_m": 6.0,
                    "touch_count": 55,
                    "touch_share": 8.0,
                    "minutes_played": 90.0,
                },
                {
                    "playerName": "Kevin De Bruyne",
                    "Mapped Jersey Number": 11,
                    "positional_role": "CM",
                    "median_x": 62.0,
                    "median_y": 50.0,
                    "q25_x": 57.0,
                    "q75_x": 67.0,
                    "q25_y": 45.0,
                    "q75_y": 55.0,
                    "iqr_x": 10.0,
                    "iqr_y": 10.0,
                    "dispersion_m": 7.0,
                    "touch_count": 65,
                    "touch_share": 9.5,
                    "minutes_played": 65.0,
                },
            ]
        )

        summary = {
            "centroid_x": 58.5,
            "centroid_y": 36.0,
            "structural_min_x": 55.0,
            "structural_max_x": 62.0,
            "structural_min_y": 22.0,
            "structural_max_y": 50.0,
        }

        figure = (
            formation_plotly
            .plot_mean_positions_profile_plotly(
                profile,
                summary,
                is_away=True,
            )
        )

        label_texts = {
            annotation.text
            for annotation in figure.layout.annotations
            if annotation.bgcolor == "rgba(16,47,69,0.92)"
        }

        self.assertIn(
            "<b>Di Lorenzo</b>",
            label_texts,
        )
        self.assertIn(
            "<b>De Bruyne</b>",
            label_texts,
        )



if __name__ == "__main__":
    unittest.main()
