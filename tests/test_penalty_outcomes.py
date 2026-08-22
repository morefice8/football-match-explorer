import unittest

import pandas as pd

from src.data_processing.preprocess import process_opta_events
from src.metrics import buildup_metrics
from src.metrics import set_piece_metrics
from src.metrics import transition_metrics


class PenaltyOutcomeTests(unittest.TestCase):
    @staticmethod
    def _processed_realistic_penalty_pattern():
        """Minimal Opta eventing pattern matching a real penalty award."""
        opta_data = {
            'matchInfo': {
                'contestant': [
                    {'id': 'home-id', 'name': 'Home'},
                    {'id': 'away-id', 'name': 'Away'},
                ],
            },
            'liveData': {
                'event': [
                    {
                        'id': 1001,
                        'eventId': 1,
                        'typeId': 1,
                        'periodId': 1,
                        'timeMin': 0,
                        'timeSec': 9,
                        'contestantId': 'home-id',
                        'playerId': 'home-player',
                        'playerName': 'Home Player',
                        'outcome': 0,
                        'x': 45.0,
                        'y': 50.0,
                        'qualifier': [
                            {'qualifierId': 140, 'value': '55.0'},
                            {'qualifierId': 141, 'value': '50.0'},
                        ],
                    },
                    {
                        'id': 1002,
                        'eventId': 2,
                        'typeId': 3,
                        'periodId': 1,
                        'timeMin': 0,
                        'timeSec': 10,
                        'contestantId': 'away-id',
                        'playerId': 'away-attacker',
                        'playerName': 'Away Attacker',
                        'outcome': 0,
                        'x': 85.0,
                        'y': 65.0,
                        'qualifier': [],
                    },
                    {
                        'id': 1003,
                        'eventId': 3,
                        'typeId': 4,
                        'periodId': 1,
                        'timeMin': 0,
                        'timeSec': 11,
                        'contestantId': 'away-id',
                        'playerId': 'away-attacker',
                        'playerName': 'Away Attacker',
                        'outcome': 1,
                        'x': 85.5,
                        'y': 66.3,
                        'qualifier': [
                            {'qualifierId': 9},
                        ],
                    },
                    {
                        'id': 1004,
                        'eventId': 4,
                        'typeId': 4,
                        'periodId': 1,
                        'timeMin': 0,
                        'timeSec': 11,
                        'contestantId': 'home-id',
                        'playerId': 'home-keeper',
                        'playerName': 'Home Keeper',
                        'outcome': 0,
                        'x': 14.5,
                        'y': 33.7,
                        'qualifier': [
                            {'qualifierId': 9},
                        ],
                    },
                    {
                        'id': 1005,
                        'eventId': 5,
                        'typeId': 16,
                        'periodId': 1,
                        'timeMin': 2,
                        'timeSec': 34,
                        'contestantId': 'away-id',
                        'playerId': 'away-taker',
                        'playerName': 'Away Taker',
                        'outcome': 1,
                        'x': 88.0,
                        'y': 50.0,
                        'qualifier': [
                            {'qualifierId': 9},
                        ],
                    },
                ],
            },
        }

        event_mapping = {
            1: 'Pass',
            3: 'Take On',
            4: 'Foul',
            16: 'Goal',
        }
        qualifier_mapping = {
            9: {'name': 'Penalty'},
            140: {'name': 'Pass End X'},
            141: {'name': 'Pass End Y'},
        }

        df_processed, _, _, _ = process_opta_events(
            opta_data,
            event_mapping,
            qualifier_mapping,
            match_info={},
        )
        return df_processed

    @staticmethod
    def _transition_row(
        row_id,
        event_id,
        team,
        type_name,
        outcome,
        second,
        *,
        x=50.0,
        y=50.0,
        end_x=50.0,
        end_y=50.0,
        penalty=0,
    ):
        return {
            'id': row_id,
            'eventId': event_id,
            'team_name': team,
            'type_name': type_name,
            'outcome': outcome,
            'x': x,
            'y': y,
            'end_x': end_x,
            'end_y': end_y,
            'playerName': f'Player {row_id}',
            'Mapped Jersey Number': row_id,
            'timeMin': 10,
            'timeSec': second,
            'periodId': 1,
            'Penalty': penalty,
            'Own goal': 0,
            'From corner': 0,
            'Goal mouth y co-ordinate': 50.0,
        }

    def _transition_df(self, include_penalty_kick=False):
        rows = [
            self._transition_row(
                1,
                1,
                'Home',
                'Pass',
                'Unsuccessful',
                0,
                x=45.0,
                y=50.0,
                end_x=55.0,
                end_y=50.0,
            ),
            self._transition_row(
                2,
                2,
                'Away',
                'Pass',
                'Successful',
                2,
                x=45.0,
                y=50.0,
                end_x=65.0,
                end_y=50.0,
            ),
            self._transition_row(
                3,
                3,
                'Home',
                'Foul',
                'Unsuccessful',
                5,
                x=12.0,
                y=50.0,
                penalty=1,
            ),
        ]

        if include_penalty_kick:
            rows.append(
                self._transition_row(
                    4,
                    4,
                    'Away',
                    'Goal',
                    'Successful',
                    10,
                    x=88.0,
                    y=50.0,
                    penalty=1,
                )
            )

        return pd.DataFrame(rows)

    def test_offensive_transition_ends_as_penalty_won(self):
        result = (
            transition_metrics
            .find_buildup_after_possession_loss(
                self._transition_df(),
                team_that_lost_possession='Home',
                metric_to_analyze='offensive_transitions',
            )
        )

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Penalty won',
        )

    def test_defensive_transition_ends_as_penalty_conceded(self):
        result = (
            transition_metrics
            .find_buildup_after_possession_loss(
                self._transition_df(),
                team_that_lost_possession='Home',
                metric_to_analyze='defensive_transitions',
            )
        )

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Penalty conceded',
        )

    def test_penalty_kick_is_not_reclassified_as_transition_goal(self):
        result = (
            transition_metrics
            .find_buildup_after_possession_loss(
                self._transition_df(
                    include_penalty_kick=True
                ),
                team_that_lost_possession='Home',
                metric_to_analyze='offensive_transitions',
            )
        )

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Penalty won',
        )
        self.assertFalse(
            result['type_name'].eq('Goal').any()
        )

    def test_flag_qualifier_without_value_survives_preprocess(self):
        df_processed = self._processed_realistic_penalty_pattern()

        self.assertIn('Penalty', df_processed.columns)
        self.assertEqual(
            df_processed.loc[
                df_processed['eventId'] == 3,
                'Penalty',
            ].iloc[0],
            1,
        )
        self.assertEqual(
            df_processed.loc[
                df_processed['eventId'] == 1,
                'Penalty',
            ].iloc[0],
            0,
        )

    def test_real_opta_take_on_pattern_reaches_penalty_award(self):
        df_processed = self._processed_realistic_penalty_pattern()

        offensive = (
            transition_metrics
            .find_buildup_after_possession_loss(
                df_processed,
                team_that_lost_possession='Home',
                metric_to_analyze='offensive_transitions',
            )
        )
        defensive = (
            transition_metrics
            .find_buildup_after_possession_loss(
                df_processed,
                team_that_lost_possession='Home',
                metric_to_analyze='defensive_transitions',
            )
        )

        self.assertFalse(offensive.empty)
        self.assertFalse(defensive.empty)
        self.assertEqual(
            offensive['sequence_outcome_type'].iloc[-1],
            'Penalty won',
        )
        self.assertEqual(
            defensive['sequence_outcome_type'].iloc[-1],
            'Penalty conceded',
        )
        self.assertTrue(
            offensive['type_name'].eq('Foul').any()
        )
        self.assertFalse(
            offensive['type_name'].eq('Goal').any()
        )

    def test_penalty_set_piece_is_extracted_from_kick_event(self):
        df_processed = self._processed_realistic_penalty_pattern()

        sequences = (
            set_piece_metrics
            .extract_penalty_set_piece_sequences(
                df_processed,
                'Away',
            )
        )

        self.assertEqual(len(sequences), 1)
        penalty_sequence = sequences[0]
        self.assertEqual(
            penalty_sequence.iloc[0]['type_of_initial_trigger'],
            'Penalty',
        )
        self.assertEqual(
            penalty_sequence.iloc[-1]['sequence_outcome_type'],
            'Penalty Goal',
        )
        self.assertEqual(
            penalty_sequence.iloc[0]['playerName'],
            'Away Taker',
        )

        analyzed, stats = (
            set_piece_metrics
            .analyze_and_summarize_set_pieces(
                sequences,
            )
        )

        self.assertEqual(len(analyzed), 1)
        self.assertEqual(
            analyzed.iloc[0]['Action Type'],
            'Penalty',
        )
        self.assertEqual(
            analyzed.iloc[0]['Delivery'],
            'Penalty kick',
        )
        self.assertEqual(
            stats['outcomes']['Penalty Goal'],
            1,
        )

    def test_direct_penalty_extraction_classifies_saved_and_missed(self):
        rows = []
        for row_id, type_name in enumerate(
            ['Attempt Saved', 'Miss', 'Post'],
            start=1,
        ):
            rows.append({
                'id': row_id,
                'eventId': row_id,
                'team_name': 'Home',
                'type_name': type_name,
                'Penalty': 1,
                'timeMin': 30 + row_id,
                'timeSec': 0,
                'periodId': 1,
                'playerName': f'Taker {row_id}',
                'x': 88.0,
                'y': 50.0,
            })

        sequences = (
            set_piece_metrics
            .extract_penalty_set_piece_sequences(
                pd.DataFrame(rows),
                'Home',
            )
        )

        self.assertEqual(
            [
                seq.iloc[-1]['sequence_outcome_type']
                for seq in sequences
            ],
            [
                'Penalty Saved',
                'Penalty Missed',
                'Penalty Missed',
            ],
        )

    @staticmethod
    def _buildup_row(
        row_id,
        event_id,
        team,
        type_name,
        outcome,
        second,
        *,
        x=50.0,
        y=50.0,
        end_x=50.0,
        end_y=50.0,
        penalty=0,
    ):
        return {
            'id': row_id,
            'eventId': event_id,
            'team_name': team,
            'type_name': type_name,
            'outcome': outcome,
            'x': x,
            'y': y,
            'end_x': end_x,
            'end_y': end_y,
            'playerName': f'Player {row_id}',
            'Mapped Jersey Number': row_id,
            'timeMin': 20,
            'timeSec': second,
            'periodId': 1,
            'lb': 0,
            'Length': 10.0,
            'cross': 0,
            'Corner taken': 0,
            'Penalty': penalty,
            'Own goal': 0,
            'Blocked': 0,
            'Goal mouth y co-ordinate': 50.0,
            'Right footed': 1,
            'Left footed': 0,
            'In-swinger': 0,
            'Out-swinger': 0,
            'Straight': 0,
        }

    def test_buildup_ends_as_penalty_won(self):
        df = pd.DataFrame([
            self._buildup_row(
                10,
                10,
                'Away',
                'Out',
                'Unsuccessful',
                0,
                x=80.0,
            ),
            self._buildup_row(
                11,
                11,
                'Home',
                'Pass',
                'Successful',
                2,
                x=20.0,
                end_x=40.0,
            ),
            self._buildup_row(
                12,
                12,
                'Away',
                'Foul',
                'Unsuccessful',
                5,
                x=10.0,
                penalty=1,
            ),
        ])

        result = buildup_metrics.find_buildup_sequences(
            df,
            attacking_team='Home',
            defending_team='Away',
            metric_to_analyze='buildup_phase',
            triggers_buildups=['Out'],
        )

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Penalty won',
        )

    def test_penalty_set_piece_does_not_require_pass_delivery(self):
        df = pd.DataFrame([
            self._buildup_row(
                20,
                20,
                'Away',
                'Foul',
                'Unsuccessful',
                0,
                x=10.0,
                penalty=1,
            ),
            self._buildup_row(
                21,
                21,
                'Home',
                'Goal',
                'Successful',
                4,
                x=88.0,
                y=50.0,
                penalty=1,
            ),
        ])

        result = buildup_metrics.find_buildup_sequences(
            df,
            attacking_team='Home',
            defending_team='Away',
            metric_to_analyze='set_piece',
            triggers_buildups=['Foul'],
        )

        self.assertFalse(result.empty)
        self.assertEqual(
            result['sequence_outcome_type'].iloc[-1],
            'Penalty Goal',
        )
        self.assertEqual(
            result['type_of_initial_trigger'].iloc[0],
            'Penalty',
        )

        analyzed, stats = (
            set_piece_metrics
            .analyze_and_summarize_set_pieces(
                [result],
            )
        )

        self.assertEqual(len(analyzed), 1)
        self.assertEqual(
            analyzed.iloc[0]['Action Type'],
            'Penalty',
        )
        self.assertEqual(
            analyzed.iloc[0]['Delivery'],
            'Penalty kick',
        )
        self.assertEqual(
            analyzed.iloc[0]['Outcome'],
            'Penalty Goal',
        )
        self.assertEqual(
            stats['outcomes']['Penalty Goal'],
            1,
        )


if __name__ == '__main__':
    unittest.main()
