"""Shared pitch geometry for lane / third / box questions and two-team views.

Coordinates are Opta team-relative units (0-100 on both axes, the team attacks
towards x = 100). One x unit is ``pitch_length / 100`` metres and one y unit is
``pitch_width / 100`` metres (1.05 m and 0.68 m on a 105 x 68 m pitch), so
distances must scale each axis separately (see ``shot_geometry``).

Lanes follow the app's orientation: *right* is the low-y side and *left* is the
high-y side, from the point of view of the team that is attacking. The five
lanes are the penalty-area width (``half`` lanes), the two outer strips
(``wide`` lanes) and the central corridor between the half-space lines.

Two-team views put both teams on one pitch by keeping a *reference* team's
coordinates and rotating the other team by 180 degrees
(``x' = 100 - x`` **and** ``y' = 100 - y``). A mirror in x alone would swap the
flanks, so the rotation is the only correct transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEFAULT_PITCH_LENGTH_METERS, DEFAULT_PITCH_WIDTH_METERS

X_METRES = DEFAULT_PITCH_LENGTH_METERS / 100.0
Y_METRES = DEFAULT_PITCH_WIDTH_METERS / 100.0

LANES = ("right_wide", "right_half", "centre", "left_half", "left_wide")
#: y values separating the five lanes, low y (right) to high y (left).
LANE_EDGES_Y = (21.1, 36.8, 63.2, 78.9)

THIRDS = ("defensive", "middle", "final")
THIRD_EDGES_X = (100.0 / 3.0, 200.0 / 3.0)

PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
PENALTY_AREA_X_MIN = 100.0 - PENALTY_AREA_DEPTH_M / X_METRES
PENALTY_AREA_HALF_WIDTH_Y = PENALTY_AREA_WIDTH_M / 2.0 / Y_METRES


def lane_of(y) -> str | None:
    """Lane of a team-relative y (``None`` if missing)."""
    if y is None or pd.isna(y):
        return None
    return LANES[int(np.searchsorted(LANE_EDGES_Y, float(y), side="right"))]


def third_of(x) -> str | None:
    """Third of a team-relative x (``None`` if missing)."""
    if x is None or pd.isna(x):
        return None
    return THIRDS[int(np.searchsorted(THIRD_EDGES_X, float(x), side="right"))]


def lane_series(y: pd.Series) -> pd.Series:
    values = pd.to_numeric(y, errors="coerce")
    idx = np.searchsorted(LANE_EDGES_Y, values.to_numpy(dtype=float), side="right")
    out = pd.Series(np.array(LANES, dtype=object)[idx], index=y.index, dtype=object)
    out[values.isna()] = None
    return out


def third_series(x: pd.Series) -> pd.Series:
    values = pd.to_numeric(x, errors="coerce")
    idx = np.searchsorted(THIRD_EDGES_X, values.to_numpy(dtype=float), side="right")
    out = pd.Series(np.array(THIRDS, dtype=object)[idx], index=x.index, dtype=object)
    out[values.isna()] = None
    return out


def in_penalty_area(x, y):
    """True where the point is inside the attacked penalty area (scalar or Series)."""
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    result = (xs >= PENALTY_AREA_X_MIN) & ((ys - 50.0).abs() <= PENALTY_AREA_HALF_WIDTH_Y)
    return bool(result.iloc[0]) if np.isscalar(x) and np.isscalar(y) else result


def distance_m(x1, y1, x2, y2):
    """Straight-line distance in metres between two team-relative points."""
    return np.hypot((np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)) * X_METRES,
                    (np.asarray(y1, dtype=float) - np.asarray(y2, dtype=float)) * Y_METRES)


def rotate_point(x, y):
    """Rotate team-relative coordinates by 180 degrees (opponent's point of view)."""
    return 100.0 - pd.to_numeric(x, errors="coerce"), 100.0 - pd.to_numeric(y, errors="coerce")


def add_shared_frame(frame: pd.DataFrame, reference_team: str, team_column: str = "team_name") -> pd.DataFrame:
    """Add ``x_shared``/``y_shared`` (and end points) on the reference team's frame.

    The reference team keeps its coordinates; every other team is rotated by
    180 degrees, so both teams can be drawn on one pitch facing each other.
    Lanes of the shared frame are lanes from the reference team's perspective.
    """
    result = frame.copy()
    other = result[team_column].ne(reference_team)
    for src, dst in (("x", "x_shared"), ("y", "y_shared"), ("end_x", "end_x_shared"), ("end_y", "end_y_shared")):
        if src not in result.columns:
            continue
        values = pd.to_numeric(result[src], errors="coerce")
        result[dst] = values.where(~other, 100.0 - values)
    return result
