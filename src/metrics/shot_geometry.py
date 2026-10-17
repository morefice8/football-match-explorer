"""Shot geometry shared by the metrics, the shot map and the report.

Opta coordinates run 0-100 on both axes, but one unit is not the same length
on the two axes: on a standard 105 x 68 m pitch it is 1.05 m along the length
(x) and 0.68 m across the width (y). Distances must therefore scale each axis
with its own pitch dimension before taking the hypotenuse. Scaling both axes
by the pitch length overstates the distance of every shot taken away from the
centre line (a shot from the edge of the area looked ~8 m further than it was).
"""

from __future__ import annotations

import numpy as np

from src.config import DEFAULT_PITCH_LENGTH_METERS, DEFAULT_PITCH_WIDTH_METERS

GOAL_X = 100.0
GOAL_Y = 50.0


def shot_distance_m(
    x,
    y,
    *,
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_METERS,
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_METERS,
):
    """Distance in metres from a shot location to the centre of the goal line.

    ``x`` and ``y`` are team-relative Opta coordinates (the attacked goal is at
    x=100). Scalars, NumPy arrays and pandas Series are all accepted.
    """
    dx_m = (GOAL_X - x) * (pitch_length_m / 100.0)
    dy_m = (GOAL_Y - y) * (pitch_width_m / 100.0)
    return np.hypot(dx_m, dy_m)
