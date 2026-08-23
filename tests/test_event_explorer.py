import unittest

import pandas as pd

from src.components.event_explorer import (
    DEFAULT_COLUMN_ORDER,
    ROW_KEY,
    event_explorer_visible_columns,
    filter_event_explorer_dataframe,
    prepare_event_explorer_dataframe,
)


class EventExplorerTests(unittest.TestCase):

    def sample_df(self):
        return pd.DataFrame(
            [
                {
                    "id": 1001,
                    "eventId": 9001,
                    "match_id": 77,
                    "timeMin": 12,
                    "timeSec": 7,
                    "periodId": 1,
                    "team_name": "Genoa",
                    "playerName": "Player A",
                    "Mapped Jersey Number": 8,
                    "positional_role": "MC",
                    "type_name": "Pass",
                    "outcome": "Successful",
                    "x": 21.25,
                    "y": 47.5,
                    "end_x": 37.5,
                    "end_y": 53.0,
                },
                {
                    "id": 1002,
                    "eventId": 9002,
                    "match_id": 77,
                    "timeMin": 54,
                    "timeSec": 42,
                    "periodId": 2,
                    "team_name": "Napoli",
                    "playerName": "Player B",
                    "Mapped Jersey Number": 10,
                    "positional_role": "AMC",
                    "type_name": "Shot",
                    "outcome": "Unsuccessful",
                    "x": 84.0,
                    "y": 50.0,
                    "end_x": 100.0,
                    "end_y": 50.0,
                },
            ]
        )

    def test_single_timestamp_replaces_minute_and_second(self):
        prepared = prepare_event_explorer_dataframe(
            self.sample_df()
        )
        self.assertEqual(
            prepared.loc[0, "event_time"],
            "12:07",
        )
        self.assertNotIn("timeMin", prepared.columns)
        self.assertNotIn("timeSec", prepared.columns)

    def test_default_columns_hide_technical_identifiers(self):
        prepared = prepare_event_explorer_dataframe(
            self.sample_df()
        )
        visible = event_explorer_visible_columns(
            prepared
        )

        self.assertEqual(
            tuple(visible),
            tuple(
                column
                for column in DEFAULT_COLUMN_ORDER
                if column in prepared.columns
            ),
        )

        for technical in (
            "id",
            "eventId",
            "match_id",
        ):
            self.assertNotIn(
                technical,
                visible,
            )

    def test_period_is_normalised(self):
        prepared = prepare_event_explorer_dataframe(
            self.sample_df()
        )
        self.assertEqual(
            prepared["event_period"].tolist(),
            ["1H", "2H"],
        )

    def test_quick_filters_combine(self):
        prepared = prepare_event_explorer_dataframe(
            self.sample_df()
        )
        filtered = filter_event_explorer_dataframe(
            prepared,
            team="Napoli",
            event="Shot",
            period="2H",
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(
            filtered.iloc[0]["playerName"],
            "Player B",
        )

    def test_source_row_is_stable(self):
        prepared = prepare_event_explorer_dataframe(
            self.sample_df()
        )
        self.assertEqual(
            prepared[ROW_KEY].tolist(),
            [0, 1],
        )

    def test_raw_dataframe_is_not_mutated(self):
        raw = self.sample_df()
        before = raw.copy(deep=True)

        prepare_event_explorer_dataframe(raw)

        pd.testing.assert_frame_equal(
            raw,
            before,
        )


    def test_generic_numeric_period_codes_fall_back_to_match_minute(self):
        raw = pd.DataFrame(
            [
                {"period": 14, "timeMin": 12, "timeSec": 0},
                {"period": 16, "timeMin": 64, "timeSec": 0},
            ]
        )

        prepared = prepare_event_explorer_dataframe(raw)

        self.assertEqual(
            prepared["event_period"].tolist(),
            ["1H", "2H"],
        )

    def test_period_id_wins_over_ambiguous_generic_period(self):
        raw = pd.DataFrame(
            [
                {
                    "period": 14,
                    "periodId": 1,
                    "timeMin": 12,
                    "timeSec": 0,
                },
                {
                    "period": 16,
                    "periodId": 2,
                    "timeMin": 64,
                    "timeSec": 0,
                },
            ]
        )

        prepared = prepare_event_explorer_dataframe(raw)

        self.assertEqual(
            prepared["event_period"].tolist(),
            ["1H", "2H"],
        )



if __name__ == "__main__":
    unittest.main()
