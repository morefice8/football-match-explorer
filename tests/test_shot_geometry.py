import math
import unittest

import numpy as np
import pandas as pd

from src.metrics.shot_geometry import shot_distance_m
from src.metrics.shot_metrics import calculate_shot_stats
from src.visualization import shot_map_plotly


class ShotDistanceTests(unittest.TestCase):
    def test_central_shot_scales_only_the_pitch_length(self):
        # x=90 on the centre line is 10 units from the goal line = 10.5 m.
        self.assertAlmostEqual(shot_distance_m(90.0, 50.0), 10.5, places=6)

    def test_wide_shot_uses_the_pitch_width_for_the_lateral_offset(self):
        # Same depth (5.7 units = 5.985 m) but 23.1 units wide = 15.708 m.
        expected = math.hypot(5.7 * 1.05, 23.1 * 0.68)
        self.assertAlmostEqual(shot_distance_m(94.3, 73.1), expected, places=6)
        # The previous formula scaled both axes by 1.05 and returned ~25 m.
        self.assertLess(shot_distance_m(94.3, 73.1), 17.0)

    def test_pure_lateral_offset_uses_pitch_width(self):
        self.assertAlmostEqual(shot_distance_m(100.0, 0.0), 34.0, places=6)

    def test_custom_pitch_dimensions_are_respected(self):
        self.assertAlmostEqual(
            shot_distance_m(100.0, 0.0, pitch_length_m=100.0, pitch_width_m=60.0),
            30.0,
            places=6,
        )

    def test_series_and_arrays_are_supported(self):
        series = pd.Series([90.0, 100.0])
        result = shot_distance_m(series, pd.Series([50.0, 0.0]))
        self.assertIsInstance(result, pd.Series)
        np.testing.assert_allclose(result.to_numpy(), [10.5, 34.0])


class ShotDistanceIntegrationTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame([
            {"id": 1, "team_name": "Home", "type_name": "Miss", "Own goal": 0, "Blocked": pd.NA, "x": 94.3, "y": 73.1, "playerName": "Wide", "timeMin": 22},
            {"id": 2, "team_name": "Home", "type_name": "Miss", "Own goal": 0, "Blocked": pd.NA, "x": 90.0, "y": 50.0, "playerName": "Central", "timeMin": 30},
        ])

    def test_calculate_shot_stats_uses_axis_specific_scaling(self):
        shots, home, _ = calculate_shot_stats(self._frame(), "Home", "Away", None, None, None, None)
        distances = shots.set_index("id")["shot_distance_m"]
        self.assertAlmostEqual(distances.loc[2], 10.5, places=6)
        self.assertAlmostEqual(distances.loc[1], math.hypot(5.7 * 1.05, 23.1 * 0.68), places=6)
        self.assertAlmostEqual(home["avg_shot_distance"], (distances.loc[1] + distances.loc[2]) / 2, places=6)

    def test_shot_map_hover_shows_the_corrected_distance(self):
        row = pd.Series({"x": 94.3, "y": 73.1, "timeMin": 22, "playerName": "Wide"})
        text = shot_map_plotly._hover_text(row, "Home", "Miss")
        expected = math.hypot(5.7 * 1.05, 23.1 * 0.68)
        self.assertIn(f"Distance: {expected:.1f}m", text)


if __name__ == "__main__":
    unittest.main()
