import unittest

import pandas as pd

from src.utils.sequence_filtering import (
    filter_sequences_exact,
    make_carousel_controller,
    step_carousel,
)


def make_sequence(
    outcome,
    flank,
    sequence_type,
):
    return pd.DataFrame([
        {
            "sequence_outcome_type":
                outcome,
            "dominant_flank":
                flank,
            "type_of_initial_loss":
                sequence_type,
        }
    ])


FILTER_GETTERS = {
    "outcomes": (
        lambda seq:
        seq.iloc[-1].get(
            "sequence_outcome_type"
        )
    ),
    "flanks": (
        lambda seq:
        seq.iloc[-1].get(
            "dominant_flank"
        )
    ),
    "types": (
        lambda seq:
        seq.iloc[0].get(
            "type_of_initial_loss"
        )
    ),
}


class SequenceFilteringTests(
    unittest.TestCase
):

    def setUp(self):
        self.sequences = [
            make_sequence(
                "Shots",
                "Left",
                "Unsuccessful Pass",
            ),
            make_sequence(
                "Lost Possessions",
                "Right",
                "Dispossessed",
            ),
        ]

    def test_no_filter_returns_all_sequences(
        self,
    ):
        result = filter_sequences_exact(
            self.sequences,
            None,
            FILTER_GETTERS,
        )

        self.assertEqual(
            len(result),
            2,
        )

    def test_matching_filter_returns_only_matches(
        self,
    ):
        result = filter_sequences_exact(
            self.sequences,
            {
                "outcomes": "Shots",
            },
            FILTER_GETTERS,
        )

        self.assertEqual(
            len(result),
            1,
        )

        self.assertEqual(
            result[0]
            .iloc[-1]
            .get(
                "sequence_outcome_type"
            ),
            "Shots",
        )

    def test_multiple_filters_are_combined(
        self,
    ):
        result = filter_sequences_exact(
            self.sequences,
            {
                "outcomes": "Shots",
                "flanks": "Left",
            },
            FILTER_GETTERS,
        )

        self.assertEqual(
            len(result),
            1,
        )

    def test_impossible_combination_returns_empty(
        self,
    ):
        result = filter_sequences_exact(
            self.sequences,
            {
                "outcomes": "Shots",
                "flanks": "Right",
            },
            FILTER_GETTERS,
        )

        self.assertEqual(
            result,
            [],
        )

    def test_unknown_filter_fails_closed(
        self,
    ):
        result = filter_sequences_exact(
            self.sequences,
            {
                "unknown_filter":
                    "anything",
            },
            FILTER_GETTERS,
        )

        self.assertEqual(
            result,
            [],
        )

    def test_empty_sequences_are_ignored(
        self,
    ):
        result = filter_sequences_exact(
            [
                pd.DataFrame(),
                *self.sequences,
            ],
            None,
            FILTER_GETTERS,
        )

        self.assertEqual(
            len(result),
            2,
        )


class CarouselStateTests(
    unittest.TestCase
):

    def test_zero_item_controller(
        self,
    ):
        self.assertEqual(
            make_carousel_controller(
                0
            ),
            {
                "active_index": 0,
                "total_items": 0,
            },
        )

    def test_controller_clamps_index(
        self,
    ):
        self.assertEqual(
            make_carousel_controller(
                3,
                active_index=99,
            ),
            {
                "active_index": 2,
                "total_items": 3,
            },
        )

    def test_zero_item_carousel_does_not_move(
        self,
    ):
        result = step_carousel(
            {
                "active_index": 0,
                "total_items": 0,
            },
            1,
        )

        self.assertEqual(
            result,
            {
                "active_index": 0,
                "total_items": 0,
            },
        )

    def test_carousel_wraps_forward_and_backward(
        self,
    ):
        forward = step_carousel(
            {
                "active_index": 2,
                "total_items": 3,
            },
            1,
        )

        backward = step_carousel(
            {
                "active_index": 0,
                "total_items": 3,
            },
            -1,
        )

        self.assertEqual(
            forward["active_index"],
            0,
        )

        self.assertEqual(
            backward["active_index"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
