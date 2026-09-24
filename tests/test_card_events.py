from __future__ import annotations

import unittest

import pandas as pd

from src.metrics.card_events import (
    CARD_TYPE_RED,
    CARD_TYPE_SECOND_YELLOW,
    CARD_TYPE_UNKNOWN,
    CARD_TYPE_YELLOW,
    classify_card_event,
    extract_card_events,
)


class CardClassificationTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = [
            {
                "id": 1,
                "team_name": "Home",
                "playerName": "A. Player",
                "type_name": "Card",
                "timeMin": 12,
                "timeSec": 5,
                "x": 0.0,
                "y": 0.0,
                "Yellow Card": 1,
            },
            {
                "id": 2,
                "team_name": "Home",
                "playerName": "B. Player",
                "type_name": "Card",
                "timeMin": 55,
                "timeSec": 0,
                "x": 0.0,
                "y": 0.0,
                "Second yellow": 1,
            },
            {
                "id": 3,
                "team_name": "Away",
                "playerName": "C. Player",
                "type_name": "Card",
                "timeMin": 78,
                "timeSec": 30,
                "x": 0.0,
                "y": 0.0,
                "Red card": 1,
            },
            {
                "id": 4,
                "team_name": "Away",
                "playerName": "D. Player",
                "type_name": "Card",
                "timeMin": 80,
                "timeSec": 0,
                "x": 0.0,
                "y": 0.0,
                "Red card": pd.NA,
            },
            {
                "id": 5,
                "team_name": "Away",
                "playerName": "E. Player",
                "type_name": "Card",
                "timeMin": 5,
                "timeSec": 0,
                "x": 0.0,
                "y": 0.0,
                "Yellow Card": 1,
                "Rescinded card": 1,
            },
            {
                "id": 6,
                "team_name": "Home",
                "playerName": "F. Player",
                "type_name": "Pass",
                "timeMin": 30,
                "timeSec": 0,
                "x": 40.0,
                "y": 50.0,
            },
        ]
        return pd.DataFrame(rows)

    def test_classifier_disambiguates_yellow_second_yellow_red_and_unknown(self):
        cards = extract_card_events(self._frame()).set_index("id")

        self.assertEqual(cards.loc[1, "card_type"], CARD_TYPE_YELLOW)
        self.assertFalse(bool(cards.loc[1, "card_resulted_in_dismissal"]))

        self.assertEqual(cards.loc[2, "card_type"], CARD_TYPE_SECOND_YELLOW)
        self.assertTrue(bool(cards.loc[2, "card_resulted_in_dismissal"]))

        self.assertEqual(cards.loc[3, "card_type"], CARD_TYPE_RED)
        self.assertTrue(bool(cards.loc[3, "card_resulted_in_dismissal"]))

        # A qualifier column that exists on the frame but is NA for this row
        # must not be treated as present.
        self.assertEqual(cards.loc[4, "card_type"], CARD_TYPE_UNKNOWN)
        self.assertFalse(bool(cards.loc[4, "card_resulted_in_dismissal"]))

    def test_rescinded_flag_is_surfaced_not_dropped(self):
        cards = extract_card_events(self._frame()).set_index("id")
        self.assertTrue(bool(cards.loc[5, "card_rescinded"]))
        self.assertEqual(cards.loc[5, "card_type"], CARD_TYPE_YELLOW)

    def test_non_card_events_are_excluded(self):
        cards = extract_card_events(self._frame())
        self.assertNotIn(6, cards["id"].tolist())
        self.assertEqual(len(cards), 5)

    def test_empty_and_missing_type_column_frames_are_handled(self):
        self.assertTrue(extract_card_events(pd.DataFrame()).empty)
        self.assertTrue(
            extract_card_events(pd.DataFrame({"id": [1]})).empty
        )

    def test_classify_card_event_treats_missing_qualifiers_as_unknown(self):
        classification = classify_card_event({"type_name": "Card"})
        self.assertEqual(classification.card_type, CARD_TYPE_UNKNOWN)
        self.assertFalse(classification.resulted_in_dismissal)
        self.assertFalse(classification.rescinded)


if __name__ == "__main__":
    unittest.main()
