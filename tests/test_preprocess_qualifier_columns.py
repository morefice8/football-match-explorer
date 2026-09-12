from __future__ import annotations

import unittest

from src.data_processing.preprocess import process_opta_events


class PreprocessQualifierColumnTests(unittest.TestCase):
    def test_substitution_slot_and_position_code_remain_distinct(self):
        opta_data = {
            "matchInfo": {
                "contestant": [
                    {"id": "home-id", "name": "Napoli"},
                    {"id": "away-id", "name": "Udinese"},
                ]
            },
            "liveData": {
                "event": [
                    {
                        "id": 2941348915,
                        "eventId": 74,
                        "typeId": 19,
                        "periodId": 1,
                        "timeMin": 9,
                        "timeSec": 45,
                        "contestantId": "home-id",
                        "playerId": "de-bruyne-id",
                        "playerName": "K. De Bruyne",
                        "outcome": 1,
                        "x": 0.0,
                        "y": 0.0,
                        "qualifier": [
                            {"qualifierId": 145, "value": "11"},
                            {"qualifierId": 292, "value": "6"},
                        ],
                    }
                ]
            },
        }

        frame, _, _, _ = process_opta_events(
            opta_data,
            event_mapping={19: "Player on"},
            qualifier_mapping={
                145: {
                    "name": "Formation slot",
                    "description": (
                        "Formation position of a player coming on"
                    ),
                }
            },
            match_info={},
        )

        self.assertTrue(frame.columns.is_unique)
        self.assertIn("Formation slot", frame.columns)
        self.assertIn("Substitution position code", frame.columns)
        self.assertEqual(frame.loc[0, "Formation slot"], "11")
        self.assertEqual(
            frame.loc[0, "Substitution position code"],
            "6",
        )


if __name__ == "__main__":
    unittest.main()
