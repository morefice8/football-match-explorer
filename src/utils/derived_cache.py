"""
FOUND-01 — bounded in-process cache for expensive match-derived datasets.

The cache is intentionally transparent to metric callers:
- public function signatures stay unchanged;
- cache keys prefer a real match id when present and always include a stable
  event-stream signature;
- cached objects are returned as defensive copies so callback-side mutation
  cannot poison later reads;
- a small LRU bound prevents memory growth while browsing multiple matches;
- per-key locks provide "single flight" behaviour when Dash callbacks request
  the same derived dataset concurrently.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import wraps
import hashlib
import inspect
import math
import os
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd


DERIVED_CACHE_MAX_ENTRIES = max(
    int(os.getenv("MATCH_DERIVED_CACHE_MAX_ENTRIES", "96")),
    1,
)

# Columns chosen to identify the semantic event stream while remaining stable
# across DataFrame.copy() / JSON round-trips used by Dash stores.
_SIGNATURE_COLUMNS = (
    "id",
    "eventId",
    "periodId",
    "timeMin",
    "timeSec",
    "timeStamp",
    "typeId",
    "type_name",
    "outcome",
    "team_name",
    "contestantId",
    "playerId",
    "playerName",
    "x",
    "y",
    "end_x",
    "end_y",
    "receiver",
    "receiver_player_id",
    "receiver_event_id",
    "receiver_confidence",
    "receiver_is_reliable",
    "Penalty",
    "Own goal",
    "Corner taken",
    "Free kick taken",
    "Freekick taken",
    "ThrowIn",
    "Goal kick",
    "Goal kick taken",
    "cross",
    "Length",

    # Qualifiers consumed by cached sequence/restart detectors. These must
    # participate in the cache signature: two otherwise-identical event
    # streams can have different football semantics when only one qualifier
    # changes.
    "Out of play",
    "From corner",
    "Red card",
    "Second yellow",
    "Goal mouth y co-ordinate",
    "GoalMouthY",
    "Blocked",
    "Offside",
    "Through ball",
    "Long ball",
    "Chipped",
    "Head",
    "Right footed",
    "Left footed",
    "In-swinger",
    "Out-swinger",
    "Straight",
)

_MATCH_ID_ATTRS = (
    "match_id",
    "matchId",
    "fixture_id",
    "fixtureId",
    "game_id",
    "gameId",
)

_MATCH_ID_COLUMNS = _MATCH_ID_ATTRS

_CACHE = OrderedDict()
_CACHE_LOCK = RLock()
_KEY_LOCKS = {}

_STATS = {
    "hits": 0,
    "misses": 0,
    "evictions": 0,
    "by_namespace": {},
}


def _namespace_stats(namespace: str) -> dict:
    with _CACHE_LOCK:
        return _STATS["by_namespace"].setdefault(
            namespace,
            {
                "hits": 0,
                "misses": 0,
            },
        )


def _record(namespace: str, kind: str) -> None:
    with _CACHE_LOCK:
        _STATS[kind] += 1
        ns = _STATS["by_namespace"].setdefault(
            namespace,
            {
                "hits": 0,
                "misses": 0,
            },
        )
        ns[kind] += 1


def _clone(value: Any) -> Any:
    """Return a defensive copy of supported cached result structures."""
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)

    if isinstance(value, pd.Series):
        return value.copy(deep=True)

    if isinstance(value, list):
        return [_clone(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)

    if isinstance(value, dict):
        return {
            key: _clone(item)
            for key, item in value.items()
        }

    return value


def _normalise_scalar(value: Any) -> str:
    if value is None:
        return "<NA>"

    if isinstance(value, (np.bool_, bool)):
        return "1" if bool(value) else "0"

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return "<NA>"
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.8f}".rstrip("0").rstrip(".")

    try:
        if pd.isna(value):
            return "<NA>"
    except (TypeError, ValueError):
        pass

    return str(value)


def _series_digest(series: pd.Series) -> bytes:
    """
    Hash a Series after normalising values so 1 and 1.0 round-trips match.
    """
    normalised = series.map(_normalise_scalar)
    hashed = pd.util.hash_pandas_object(
        normalised,
        index=False,
        categorize=False,
    )
    return hashed.to_numpy(dtype="uint64").tobytes()


def _explicit_match_id(df: pd.DataFrame) -> str | None:
    attrs = getattr(df, "attrs", {}) or {}

    for key in _MATCH_ID_ATTRS:
        value = attrs.get(key)
        if value not in (None, ""):
            return _normalise_scalar(value)

    for column in _MATCH_ID_COLUMNS:
        if column not in df.columns:
            continue

        values = [
            _normalise_scalar(value)
            for value in df[column].dropna().unique().tolist()
        ]

        values = sorted(set(values))

        if len(values) == 1:
            return values[0]

    return None


def dataframe_signature(df: pd.DataFrame) -> str:
    """
    Return a stable semantic signature for one processed match DataFrame.

    A real match id is included when available. The digest also includes
    event-stream identity/coordinates and the available column set, preventing
    raw and receiver-enriched DataFrames from colliding.
    """
    if df is None:
        return "none"

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "dataframe_signature expects a pandas DataFrame"
        )

    digest = hashlib.blake2b(
        digest_size=16,
    )

    match_id = _explicit_match_id(df)

    digest.update(
        f"rows:{len(df)}".encode("utf-8")
    )

    digest.update(
        (
            "columns:"
            + "|".join(
                sorted(map(str, df.columns))
            )
        ).encode("utf-8")
    )

    # Index matters because receiver enrichment joins by source index.
    digest.update(
        _series_digest(
            pd.Series(
                df.index,
                index=df.index,
            )
        )
    )

    for column in _SIGNATURE_COLUMNS:
        if column not in df.columns:
            continue

        digest.update(
            column.encode("utf-8")
        )
        digest.update(
            _series_digest(df[column])
        )

    identity = (
        f"match:{match_id}"
        if match_id is not None
        else "match:auto"
    )

    return (
        f"{identity}:"
        f"{digest.hexdigest()}"
    )


def _normalise_argument(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return (
            "dataframe",
            dataframe_signature(value),
        )

    if isinstance(value, pd.Series):
        return (
            "series",
            tuple(
                _normalise_scalar(item)
                for item in value.tolist()
            ),
        )

    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    str(key),
                    _normalise_argument(item),
                )
                for key, item in value.items()
            )
        )

    if isinstance(value, (list, tuple)):
        return tuple(
            _normalise_argument(item)
            for item in value
        )

    if isinstance(value, (set, frozenset)):
        items = [
            _normalise_argument(item)
            for item in value
        ]
        return tuple(
            sorted(
                items,
                key=repr,
            )
        )

    if isinstance(value, np.generic):
        return _normalise_argument(
            value.item()
        )

    if isinstance(value, float):
        if math.isnan(value):
            return "<NaN>"
        if math.isinf(value):
            return (
                "<Inf>"
                if value > 0
                else "<-Inf>"
            )

    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _call_signature(
    func,
    df: pd.DataFrame,
    args,
    kwargs,
) -> tuple:
    signature = inspect.signature(func)
    bound = signature.bind_partial(
        df,
        *args,
        **kwargs,
    )
    bound.apply_defaults()

    arguments = list(
        bound.arguments.items()
    )

    # The first argument is the source DataFrame and is represented by its
    # semantic match signature instead of by object identity.
    if arguments:
        arguments = arguments[1:]

    return tuple(
        (
            name,
            _normalise_argument(value),
        )
        for name, value in arguments
    )


def clear_derived_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
        _KEY_LOCKS.clear()

        _STATS["hits"] = 0
        _STATS["misses"] = 0
        _STATS["evictions"] = 0
        _STATS["by_namespace"] = {}


def derived_cache_info() -> dict:
    with _CACHE_LOCK:
        return {
            "entries": len(_CACHE),
            "max_entries":
                DERIVED_CACHE_MAX_ENTRIES,
            "hits": _STATS["hits"],
            "misses": _STATS["misses"],
            "evictions":
                _STATS["evictions"],
            "by_namespace": {
                namespace: stats.copy()
                for namespace, stats
                in _STATS[
                    "by_namespace"
                ].items()
            },
        }


def cache_derived_result(namespace: str):
    """
    Decorate a function whose first argument is the processed match DataFrame.

    The wrapped function keeps the exact same external API. ``functools.wraps``
    also exposes ``__wrapped__`` so equivalence tests can compare cached and
    uncached execution directly.
    """
    if not namespace:
        raise ValueError(
            "cache namespace must be non-empty"
        )

    def decorator(func):
        @wraps(func)
        def wrapper(
            df_processed,
            *args,
            **kwargs,
        ):
            if not isinstance(
                df_processed,
                pd.DataFrame,
            ):
                return func(
                    df_processed,
                    *args,
                    **kwargs,
                )

            frame_key = dataframe_signature(
                df_processed
            )

            params_key = _call_signature(
                func,
                df_processed,
                args,
                kwargs,
            )

            cache_key = (
                namespace,
                frame_key,
                params_key,
            )

            with _CACHE_LOCK:
                if cache_key in _CACHE:
                    cached = _CACHE.pop(
                        cache_key
                    )
                    _CACHE[
                        cache_key
                    ] = cached

                    _record(
                        namespace,
                        "hits",
                    )

                    return _clone(
                        cached
                    )

                key_lock = (
                    _KEY_LOCKS
                    .setdefault(
                        cache_key,
                        RLock(),
                    )
                )

            # Single-flight per key: if multiple Dash callbacks request the
            # same derived dataset at once, only one performs the computation.
            with key_lock:
                with _CACHE_LOCK:
                    if cache_key in _CACHE:
                        cached = _CACHE.pop(
                            cache_key
                        )
                        _CACHE[
                            cache_key
                        ] = cached

                        _record(
                            namespace,
                            "hits",
                        )

                        return _clone(
                            cached
                        )

                    _record(
                        namespace,
                        "misses",
                    )

                result = func(
                    df_processed,
                    *args,
                    **kwargs,
                )

                stored = _clone(
                    result
                )

                with _CACHE_LOCK:
                    _CACHE[
                        cache_key
                    ] = stored

                    while (
                        len(_CACHE)
                        > DERIVED_CACHE_MAX_ENTRIES
                    ):
                        _CACHE.popitem(
                            last=False
                        )
                        _STATS[
                            "evictions"
                        ] += 1

                    # Waiters hold their own reference to key_lock, so it is
                    # safe to release this bookkeeping entry after the cache
                    # value has been published.
                    _KEY_LOCKS.pop(
                        cache_key,
                        None,
                    )

                return _clone(
                    stored
                )

        wrapper.cache_namespace = namespace
        return wrapper

    return decorator
