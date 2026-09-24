"""Canonical Opta card/discipline classification shared by reporting.

Opta typeId 17 (``Card``) events use qualifiers 31/32/33 to distinguish a
plain yellow, a second yellow (which also ends the player's involvement) and
a straight red. Qualifier 171 (``Rescinded card``) marks a card the referee
later cancelled; it is surfaced as a flag rather than silently dropped,
consistent with this project's data-quality philosophy of never silently
repairing data.

This module deliberately does not touch the existing dismissal-detection
logic used by the formation timeline (``formation_plotly.py``) or the PPDA
lineup-stability boundary (``defensive_metrics.py``) — those only care about
events that split a lineup (red / second yellow), whereas this module
classifies every card, including ordinary yellows, for a discipline summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd


CARD_EVENT_TYPE = "Card"

CARD_TYPE_YELLOW = "yellow"
CARD_TYPE_SECOND_YELLOW = "second_yellow"
CARD_TYPE_RED = "red"
CARD_TYPE_UNKNOWN = "unknown"

CARD_TYPES = (
    CARD_TYPE_YELLOW,
    CARD_TYPE_SECOND_YELLOW,
    CARD_TYPE_RED,
    CARD_TYPE_UNKNOWN,
)

_YELLOW_ALIASES = ("Yellow Card", "qualifier_31")
_SECOND_YELLOW_ALIASES = ("Second yellow", "qualifier_32")
_RED_ALIASES = ("Red card", "qualifier_33")
_RESCINDED_ALIASES = ("Rescinded card", "qualifier_171")


def _truthy_value(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().casefold()
    if not text:
        return False
    return text not in {"0", "0.0", "false", "no", "n", "none", "nan", "<na>"}


def _has_truthy(row: Mapping[str, Any], aliases: Iterable[str]) -> bool:
    return any(alias in row and _truthy_value(row.get(alias)) for alias in aliases)


@dataclass(frozen=True)
class CardClassification:
    card_type: str
    resulted_in_dismissal: bool
    rescinded: bool


def classify_card_event(row: Mapping[str, Any]) -> CardClassification:
    """Classify one preprocessed Opta ``Card`` event row.

    A straight red (qualifier 33) and a second yellow (qualifier 32) both
    result in the player leaving the match; an ordinary yellow (qualifier 31)
    does not. If none of the three qualifiers are present the card type is
    ``unknown`` rather than guessed.
    """

    red = _has_truthy(row, _RED_ALIASES)
    second_yellow = _has_truthy(row, _SECOND_YELLOW_ALIASES)
    yellow = _has_truthy(row, _YELLOW_ALIASES)
    rescinded = _has_truthy(row, _RESCINDED_ALIASES)

    if red:
        card_type = CARD_TYPE_RED
    elif second_yellow:
        card_type = CARD_TYPE_SECOND_YELLOW
    elif yellow:
        card_type = CARD_TYPE_YELLOW
    else:
        card_type = CARD_TYPE_UNKNOWN

    return CardClassification(
        card_type=card_type,
        resulted_in_dismissal=card_type in (CARD_TYPE_RED, CARD_TYPE_SECOND_YELLOW),
        rescinded=rescinded,
    )


def extract_card_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one preprocessed-event row per Opta ``Card`` event.

    Rows are enriched with ``card_type``, ``card_resulted_in_dismissal`` and
    ``card_rescinded``. All original columns (player, team, minute, x/y, ...)
    are retained unchanged.
    """

    extra_columns = (
        "card_type",
        "card_resulted_in_dismissal",
        "card_rescinded",
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(getattr(frame, "columns", ())) + list(extra_columns))

    type_column = "type_name" if "type_name" in frame.columns else "type"
    if type_column not in frame.columns:
        return pd.DataFrame(columns=list(frame.columns) + list(extra_columns))

    cards = frame.loc[frame[type_column].eq(CARD_EVENT_TYPE)].copy()
    if cards.empty:
        for column in extra_columns:
            cards[column] = pd.Series(index=cards.index, dtype="object")
        return cards

    classified = [classify_card_event(row) for row in cards.to_dict("records")]
    cards["card_type"] = [item.card_type for item in classified]
    cards["card_resulted_in_dismissal"] = pd.Series(
        [item.resulted_in_dismissal for item in classified],
        index=cards.index,
        dtype="boolean",
    )
    cards["card_rescinded"] = pd.Series(
        [item.rescinded for item in classified],
        index=cards.index,
        dtype="boolean",
    )
    return cards
