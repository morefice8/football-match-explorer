import unittest

import pandas as pd

from src.metrics.sequence_outcome_metrics import (
    aggregate_sequence_outcomes,
    summarize_sequences,
)
from src.utils.sequence_outcomes import (
    TERMINAL_OUTCOMES,
    apply_sequence_outcome_contract,
    canonicalize_legacy_outcome,
    validate_sequence_outcome_contract,
)


def buildup_event(
    sequence_id,
    event_id,
    *,
    event_type='Pass',
    outcome='Successful',
    x=20,
    y=50,
    end_x=40,
    end_y=50,
    second=0,
    legacy_outcome='Lost Possessions',
):
    return {
        'trigger_sequence_id': sequence_id,
        'eventId': event_id,
        'team_name': 'Home',
        'type_name': event_type,
        'outcome': outcome,
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
        'timeMin': 1,
        'timeSec': second,
        'trigger_zone': 'Defensive Third',
        'type_of_initial_trigger': 'Ball recovery',
        'timeMin_at_trigger': 1,
        'timeSec_at_trigger': 0,
        'buildup_pass_count': 1,
        'sequence_outcome_type': legacy_outcome,
    }


def transition_event(
    sequence_id,
    event_id,
    *,
    event_type='Pass',
    outcome='Successful',
    x=45,
    y=50,
    end_x=60,
    end_y=50,
    second=0,
    legacy_outcome='Regained Possessions',
):
    return {
        'loss_sequence_id': sequence_id,
        'eventId': event_id,
        'team_name': 'Away',
        'type_name': event_type,
        'outcome': outcome,
        'x': x,
        'y': y,
        'end_x': end_x,
        'end_y': end_y,
        'timeMin': 10,
        'timeSec': second,
        'loss_zone': 'Middle Third',
        'type_of_initial_loss': 'Unsuccessful Pass',
        'timeMin_at_loss': 10,
        'timeSec_at_loss': 0,
        'opponent_pass_count': 1,
        'sequence_outcome_type': legacy_outcome,
    }


class CanonicalOutcomeMappingTests(unittest.TestCase):
    def test_terminal_taxonomy_is_closed(self):
        self.assertEqual(
            TERMINAL_OUTCOMES,
            {
                'goal',
                'shot',
                'turnover',
                'retained',
                'foul',
                'offside',
                'out',
                'consolidated',
                'unknown',
            },
        )

    def test_goal_labels_share_one_terminal_outcome(self):
        attacking = canonicalize_legacy_outcome(
            'Goals',
            'attacking',
        )
        defending = canonicalize_legacy_outcome(
            'Goals conceded',
            'defending',
        )

        self.assertEqual(attacking['terminal_outcome'], 'goal')
        self.assertEqual(defending['terminal_outcome'], 'goal')
        self.assertEqual(attacking['viewpoint'], 'attacking')
        self.assertEqual(defending['viewpoint'], 'defending')

    def test_turnover_labels_share_one_terminal_outcome(self):
        lost = canonicalize_legacy_outcome(
            'Lost Possessions',
            'attacking',
        )
        regained = canonicalize_legacy_outcome(
            'Regained Possessions',
            'defending',
        )

        self.assertEqual(lost['terminal_outcome'], 'turnover')
        self.assertEqual(regained['terminal_outcome'], 'turnover')

    def test_penalty_award_is_foul_with_specific_reason(self):
        won = canonicalize_legacy_outcome(
            'Penalty won',
            'attacking',
        )
        conceded = canonicalize_legacy_outcome(
            'Penalty conceded',
            'defending',
        )

        self.assertEqual(won['terminal_outcome'], 'foul')
        self.assertEqual(conceded['terminal_outcome'], 'foul')
        self.assertEqual(won['termination_reason'], 'penalty_awarded')
        self.assertEqual(conceded['termination_reason'], 'penalty_awarded')

    def test_consolidation_is_canonical_and_viewpoint_independent(self):
        attacking = canonicalize_legacy_outcome(
            'Possession Consolidated',
            'attacking',
        )
        defending = canonicalize_legacy_outcome(
            'Opponent Possession Consolidated',
            'defending',
        )

        self.assertEqual(attacking['terminal_outcome'], 'consolidated')
        self.assertEqual(defending['terminal_outcome'], 'consolidated')
        self.assertEqual(
            attacking['termination_reason'],
            'time_window_elapsed',
        )


class SequenceContractTests(unittest.TestCase):
    def test_contract_is_constant_on_every_row(self):
        df = pd.DataFrame([
            buildup_event(1, 1),
            buildup_event(
                1,
                2,
                event_type='Pass',
                outcome='Unsuccessful',
                second=4,
            ),
        ])

        # Simulate stale per-row canonical metadata from a legacy caller.
        df['terminal_outcome'] = ['shot', 'turnover']
        df['termination_reason'] = ['shot', 'unsuccessful_pass']
        df['viewpoint'] = ['attacking', 'attacking']

        result = apply_sequence_outcome_contract(
            df,
            viewpoint='attacking',
            legacy_outcome='Lost Possessions',
        )

        self.assertEqual(result['terminal_outcome'].nunique(), 1)
        self.assertEqual(result['termination_reason'].nunique(), 1)
        self.assertEqual(result['viewpoint'].nunique(), 1)
        self.assertEqual(result['terminal_outcome'].iloc[0], 'turnover')
        self.assertEqual(
            result['termination_reason'].iloc[0],
            'unsuccessful_pass',
        )
        self.assertEqual(validate_sequence_outcome_contract(result), [])

    def test_validator_rejects_multiple_terminal_outcomes(self):
        df = pd.DataFrame({
            'terminal_outcome': ['shot', 'turnover'],
            'termination_reason': ['shot', 'shot'],
            'viewpoint': ['attacking', 'attacking'],
        })

        errors = validate_sequence_outcome_contract(df)
        self.assertTrue(
            any('terminal_outcome' in error for error in errors)
        )

    def test_summary_backfills_canonical_fields_from_legacy_buildup(self):
        df = pd.DataFrame([
            buildup_event(1, 1, second=1, legacy_outcome='Goals'),
            buildup_event(
                1,
                2,
                event_type='Goal',
                x=90,
                y=50,
                end_x=100,
                end_y=50,
                second=5,
                legacy_outcome='Goals',
            ),
        ])

        summary = summarize_sequences(df, 'buildup')
        row = summary.iloc[0]

        self.assertEqual(row['terminal_outcome'], 'goal')
        self.assertEqual(row['termination_reason'], 'goal')
        self.assertEqual(row['viewpoint'], 'attacking')
        self.assertEqual(row['final_outcome'], 'Goals')
        self.assertTrue(row['milestone_shot'])
        self.assertTrue(row['milestone_goal'])

    def test_summary_uses_defending_viewpoint_for_defensive_transition(self):
        df = pd.DataFrame([
            transition_event(
                1,
                1,
                event_type='Goal',
                x=88,
                second=4,
                legacy_outcome='Goals conceded',
            ),
        ])

        summary = summarize_sequences(
            df,
            'defensive_transition',
        )
        row = summary.iloc[0]

        self.assertEqual(row['terminal_outcome'], 'goal')
        self.assertEqual(row['viewpoint'], 'defending')

    def test_milestones_are_independent_from_terminal_outcome(self):
        df = pd.DataFrame([
            buildup_event(
                1,
                1,
                x=70,
                y=50,
                end_x=86,
                end_y=50,
                second=2,
                legacy_outcome='Lost Possessions',
            ),
            buildup_event(
                1,
                2,
                event_type='Pass',
                outcome='Unsuccessful',
                x=86,
                y=50,
                end_x=95,
                end_y=50,
                second=5,
                legacy_outcome='Lost Possessions',
            ),
        ])

        summary = summarize_sequences(df, 'buildup')
        row = summary.iloc[0]

        self.assertEqual(row['terminal_outcome'], 'turnover')
        self.assertTrue(row['milestone_final_third'])
        self.assertTrue(row['milestone_box'])
        self.assertFalse(row['milestone_shot'])
        self.assertFalse(row['milestone_goal'])

    def test_aggregation_exposes_canonical_and_compatibility_outcomes(self):
        summary = pd.DataFrame({
            'terminal_outcome': ['turnover', 'turnover', 'shot', 'consolidated'],
            'termination_reason': [
                'unsuccessful_pass',
                'failed_take_on',
                'shot',
                'time_window_elapsed',
            ],
            'final_outcome': [
                'Lost Possessions',
                'Lost Possessions',
                'Shots',
                'Possession Consolidated',
            ],
        })

        result = aggregate_sequence_outcomes(summary)

        self.assertEqual(
            result['terminal_outcomes']['turnover']['count'],
            2,
        )
        self.assertEqual(
            result['termination_reasons']['time_window_elapsed']['count'],
            1,
        )
        self.assertEqual(
            result['outcomes']['Lost Possessions']['count'],
            2,
        )


if __name__ == '__main__':
    unittest.main()
