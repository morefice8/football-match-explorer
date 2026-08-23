import io
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.utils.derived_cache import (
    cache_derived_result,
    clear_derived_cache,
    dataframe_signature,
    derived_cache_info,
)
from src.data_processing import pass_processing
from src.metrics import sequence_metrics


def match_events():
    return pd.DataFrame([
        {
            "matchId": 999,
            "id": 1,
            "eventId": 1,
            "periodId": 1,
            "timeMin": 1,
            "timeSec": 0,
            "typeId": 1,
            "type_name": "Pass",
            "outcome": "Successful",
            "team_name": "Home",
            "contestantId": "H",
            "playerId": 10,
            "playerName": "A",
            "shorter_name": "A",
            "Mapped Jersey Number": 10,
            "x": 20.0,
            "y": 50.0,
            "end_x": 30.0,
            "end_y": 50.0,
            "GoalMouthY": 50.0,
            "is_key_pass": False,
            "is_assist": False,
            "Corner taken": 0,
            "Free kick taken": 0,
            "Freekick taken": 0,
            "ThrowIn": 0,
            "Goal kick": 0,
            "Goal kick taken": 0,
            "cross": 0,
            "Penalty": 0,
            "Own goal": 0,
            "Length": 10.0,
        },
        {
            "matchId": 999,
            "id": 2,
            "eventId": 2,
            "periodId": 1,
            "timeMin": 1,
            "timeSec": 3,
            "typeId": 1,
            "type_name": "Pass",
            "outcome": "Successful",
            "team_name": "Home",
            "contestantId": "H",
            "playerId": 20,
            "playerName": "B",
            "shorter_name": "B",
            "Mapped Jersey Number": 20,
            "x": 34.0,
            "y": 50.0,
            "end_x": 55.0,
            "end_y": 50.0,
            "GoalMouthY": 50.0,
            "is_key_pass": True,
            "is_assist": False,
            "Corner taken": 0,
            "Free kick taken": 0,
            "Freekick taken": 0,
            "ThrowIn": 0,
            "Goal kick": 0,
            "Goal kick taken": 0,
            "cross": 0,
            "Penalty": 0,
            "Own goal": 0,
            "Length": 21.0,
        },
        {
            "matchId": 999,
            "id": 3,
            "eventId": 3,
            "periodId": 1,
            "timeMin": 1,
            "timeSec": 6,
            "typeId": 16,
            "type_name": "Goal",
            "outcome": "Successful",
            "team_name": "Home",
            "contestantId": "H",
            "playerId": 30,
            "playerName": "C",
            "shorter_name": "C",
            "Mapped Jersey Number": 30,
            "x": 88.0,
            "y": 50.0,
            "end_x": 100.0,
            "end_y": 50.0,
            "GoalMouthY": 50.0,
            "is_key_pass": False,
            "is_assist": False,
            "Corner taken": 0,
            "Free kick taken": 0,
            "Freekick taken": 0,
            "ThrowIn": 0,
            "Goal kick": 0,
            "Goal kick taken": 0,
            "cross": 0,
            "Penalty": 0,
            "Own goal": 0,
            "Length": 0.0,
        },
    ])


class DerivedCacheTests(unittest.TestCase):

    def setUp(self):
        clear_derived_cache()

    def test_same_match_copy_uses_same_signature(self):
        df = match_events()
        self.assertEqual(
            dataframe_signature(df),
            dataframe_signature(df.copy()),
        )

    def test_explicit_match_id_survives_json_round_trip(self):
        df = match_events()

        restored = pd.read_json(
            io.StringIO(
                df.to_json(orient="split")
            ),
            orient="split",
        )

        self.assertEqual(
            dataframe_signature(df),
            dataframe_signature(restored),
        )

    def test_out_of_play_qualifier_changes_dataframe_signature(self):
        base = match_events()

        in_play = base.copy()
        in_play["Out of play"] = 0

        out_of_play = base.copy()
        out_of_play["Out of play"] = 0
        out_of_play.loc[
            out_of_play.index[1],
            "Out of play",
        ] = 1

        self.assertNotEqual(
            dataframe_signature(in_play),
            dataframe_signature(out_of_play),
        )

    def test_transition_semantic_qualifiers_do_not_share_cache_entries(self):
        calls = {"count": 0}

        @cache_derived_result("test-transition-qualifier")
        def derive(df):
            calls["count"] += 1
            series = df.get(
                "Out of play",
                pd.Series(False, index=df.index),
            )
            return bool(
                series.fillna(False).astype(bool).any()
            )

        base = match_events()

        in_play = base.copy()
        in_play["Out of play"] = 0

        out_of_play = base.copy()
        out_of_play["Out of play"] = 0
        out_of_play.loc[
            out_of_play.index[1],
            "Out of play",
        ] = 1

        self.assertFalse(derive(in_play))
        self.assertTrue(derive(out_of_play))
        self.assertEqual(calls["count"], 2)

    def test_cached_results_are_defensive_copies(self):
        calls = {"count": 0}

        @cache_derived_result(
            "test-copy"
        )
        def derive(df, multiplier=1):
            calls["count"] += 1
            result = df[["id"]].copy()
            result["value"] = (
                result["id"]
                * multiplier
            )
            return result

        df = match_events()

        first = derive(
            df,
            multiplier=2,
        )

        first.loc[
            first.index[0],
            "value",
        ] = -999

        second = derive(
            df.copy(),
            multiplier=2,
        )

        self.assertEqual(
            calls["count"],
            1,
        )

        self.assertNotEqual(
            second.iloc[0]["value"],
            -999,
        )

        info = derived_cache_info()
        self.assertEqual(
            info["by_namespace"][
                "test-copy"
            ]["misses"],
            1,
        )
        self.assertEqual(
            info["by_namespace"][
                "test-copy"
            ]["hits"],
            1,
        )

    def test_different_parameters_use_different_entries(self):
        calls = {"count": 0}

        @cache_derived_result(
            "test-params"
        )
        def derive(df, multiplier=1):
            calls["count"] += 1
            return pd.DataFrame({
                "value": [
                    multiplier
                ]
            })

        df = match_events()

        derive(df, multiplier=1)
        derive(df, multiplier=2)

        self.assertEqual(
            calls["count"],
            2,
        )

    def test_get_passes_df_cached_and_uncached_are_equivalent(self):
        df = match_events()

        uncached = (
            pass_processing
            .get_passes_df
            .__wrapped__(
                df.copy()
            )
        )

        clear_derived_cache()

        cached = (
            pass_processing
            .get_passes_df(
                df.copy()
            )
        )

        assert_frame_equal(
            uncached,
            cached,
            check_dtype=True,
            check_like=False,
        )

        again = (
            pass_processing
            .get_passes_df(
                df.copy()
            )
        )

        assert_frame_equal(
            cached,
            again,
            check_dtype=True,
            check_like=False,
        )

        info = derived_cache_info()
        self.assertEqual(
            info["by_namespace"][
                "passes"
            ]["misses"],
            1,
        )
        self.assertEqual(
            info["by_namespace"][
                "passes"
            ]["hits"],
            1,
        )

    def test_infer_carries_cached_and_uncached_are_equivalent(self):
        df = match_events()

        uncached = (
            pass_processing
            .infer_carries
            .__wrapped__(
                df.copy()
            )
        )

        clear_derived_cache()

        cached = (
            pass_processing
            .infer_carries(
                df.copy()
            )
        )

        assert_frame_equal(
            uncached,
            cached,
            check_dtype=True,
            check_like=False,
        )

        again = (
            pass_processing
            .infer_carries(
                df.copy()
            )
        )

        assert_frame_equal(
            cached,
            again,
            check_dtype=True,
            check_like=False,
        )

        info = derived_cache_info()
        self.assertEqual(
            info["by_namespace"][
                "carries"
            ]["misses"],
            1,
        )
        self.assertEqual(
            info["by_namespace"][
                "carries"
            ]["hits"],
            1,
        )

    def test_shot_sequence_detector_cached_and_uncached_are_equivalent(self):
        df = match_events()

        uncached = (
            sequence_metrics
            .find_shot_sequences
            .__wrapped__(
                df.copy()
            )
        )

        clear_derived_cache()

        cached = (
            sequence_metrics
            .find_shot_sequences(
                df.copy()
            )
        )

        assert_frame_equal(
            uncached,
            cached,
            check_dtype=True,
            check_like=False,
        )

        again = (
            sequence_metrics
            .find_shot_sequences(
                df.copy()
            )
        )

        assert_frame_equal(
            cached,
            again,
            check_dtype=True,
            check_like=False,
        )

        info = derived_cache_info()
        self.assertEqual(
            info["by_namespace"][
                "shot_sequences"
            ]["misses"],
            1,
        )
        self.assertEqual(
            info["by_namespace"][
                "shot_sequences"
            ]["hits"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
