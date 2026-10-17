"""A match clock that survives stoppage time.

Opta timestamps are ``(period, minute, second)``. The second half restarts at
45:00 even when the first half ran into stoppage time, so ``minute * 60 +
second`` is **not** monotonic across the interval: first-half stoppage events
(say 45:00-47:05) share clock values with the first minutes of the second
half. Anything that compares event times -- windows, "within N seconds"
pairing, durations -- must therefore use a continuous *play clock* and, when
pairing events, also require the same period.

``PlayClock`` builds that clock from the events themselves: each period is laid
end to end, so ``play_seconds`` is the number of seconds of play elapsed, with
no interval and no overlap. A window written in play seconds always means the
same stretch of play, and its duration is simply ``(end - start) / 60``.

Extra-time periods (3 and 4) are chained the same way. Periods outside
``PLAY_PERIODS`` (pre-match entries, shoot-outs) get no play time (``NaN``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PLAY_PERIODS = (1, 2, 3, 4)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame is None or column not in frame.columns:
        index = frame.index if frame is not None else None
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def clock_seconds(frame: pd.DataFrame) -> pd.Series:
    """Printed match clock (``minute * 60 + second``); NaN without a minute."""
    minute = _numeric(frame, "timeMin")
    second = _numeric(frame, "timeSec").fillna(0.0)
    return minute * 60.0 + second


def format_clock(seconds: float | None) -> str:
    """``mm:ss`` for a printed clock value (minutes may exceed 59)."""
    if seconds is None or pd.isna(seconds):
        return ""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


@dataclass(frozen=True)
class PlayClock:
    """Continuous play time built from the periods present in the events."""

    #: ``(period, clock_start, clock_end, play_offset)`` for each period, in order.
    periods: tuple[tuple[int, float, float, float], ...] = ()

    @classmethod
    def from_events(cls, frame: pd.DataFrame) -> "PlayClock":
        if frame is None or frame.empty:
            return cls()
        clock = clock_seconds(frame)
        period = _numeric(frame, "periodId")
        rows: list[tuple[int, float, float, float]] = []
        offset = 0.0
        for p in PLAY_PERIODS:
            mask = period.eq(p) & clock.notna()
            if not mask.any():
                continue
            start = float(clock[mask].min())
            end = float(clock[mask].max())
            rows.append((p, start, end, offset))
            offset += end - start
        return cls(tuple(rows))

    @property
    def total_seconds(self) -> float:
        if not self.periods:
            return 0.0
        _, start, end, offset = self.periods[-1]
        return offset + (end - start)

    @property
    def total_minutes(self) -> float:
        return self.total_seconds / 60.0

    def play_seconds(self, period, clock) -> float:
        """Play seconds for one ``(period, printed clock)``; NaN if unknown."""
        if period is None or clock is None or pd.isna(period) or pd.isna(clock):
            return float("nan")
        for p, start, _end, offset in self.periods:
            if int(period) == p:
                return offset + (float(clock) - start)
        return float("nan")

    def series(self, frame: pd.DataFrame) -> pd.Series:
        """Vectorised play seconds for every row of ``frame``."""
        clock = clock_seconds(frame)
        period = _numeric(frame, "periodId")
        out = pd.Series(np.nan, index=frame.index, dtype=float)
        for p, start, _end, offset in self.periods:
            mask = period.eq(p) & clock.notna()
            out[mask] = offset + (clock[mask] - start)
        return out

    def locate(self, play_seconds: float) -> tuple[int, float] | None:
        """Back-map play seconds to ``(period, printed clock)``."""
        if play_seconds is None or pd.isna(play_seconds):
            return None
        for p, start, end, offset in self.periods:
            if offset <= play_seconds <= offset + (end - start) + 1e-9:
                return p, start + (play_seconds - offset)
        return None

    def label(self, play_seconds: float) -> str:
        """Printed ``mm:ss`` clock for a play-seconds value (``""`` if unknown)."""
        located = self.locate(play_seconds)
        return "" if located is None else format_clock(located[1])


def add_play_clock(frame: pd.DataFrame, clock: PlayClock | None = None) -> pd.DataFrame:
    """Return a copy with ``clock_seconds`` and ``play_seconds`` columns."""
    clock = clock or PlayClock.from_events(frame)
    result = frame.copy()
    result["clock_seconds"] = clock_seconds(result)
    result["play_seconds"] = clock.series(result)
    return result


def same_period_followers(
    frame: pd.DataFrame,
    play_seconds: float,
    period,
    within_seconds: float,
) -> pd.DataFrame:
    """Rows of ``frame`` within ``within_seconds`` *after* ``play_seconds`` in the same period.

    ``frame`` must already carry ``play_seconds`` and ``periodId``. Requiring
    the same period matters at the interval: play seconds are continuous, so a
    touch at the end of the first half would otherwise "see" kick-off events.
    """
    t = frame["play_seconds"]
    same = _numeric(frame, "periodId").eq(period)
    return frame[same & (t >= play_seconds) & (t <= play_seconds + within_seconds)]
