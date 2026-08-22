import copy
import unittest

from app import enrich_match_info_with_raw_metadata


class MatchGoalMetadataTests(unittest.TestCase):
    def _payload(self, *, goals=None, events=None):
        live_data = {}

        if goals is not None:
            live_data['goal'] = goals

        if events is not None:
            live_data['event'] = events

        return {
            'matchInfo': {
                'contestant': [
                    {
                        'id': 'home-id',
                        'position': 'home',
                    },
                    {
                        'id': 'away-id',
                        'position': 'away',
                    },
                ],
            },
            'liveData': live_data,
        }

    def test_home_and_away_goals_use_complete_contestant_map(self):
        payload = self._payload(
            goals=[
                {
                    'optaEventId': 'goal-home',
                    'contestantId': 'home-id',
                    'scorerName': 'Home Scorer',
                    'periodId': 1,
                    'timeMin': 12,
                    'timeMinSec': '11:30',
                    'type': 'G',
                },
                {
                    'optaEventId': 'goal-away',
                    'contestantId': 'away-id',
                    'scorerName': 'Away Scorer',
                    'periodId': 2,
                    'timeMin': 67,
                    'timeMinSec': '66:10',
                    'type': 'G',
                },
            ],
        )

        result = enrich_match_info_with_raw_metadata(
            payload,
            {},
        )

        self.assertEqual(
            [goal['team_position'] for goal in result['goals']],
            ['home', 'away'],
        )

    def test_own_goal_is_attributed_to_benefiting_team(self):
        payload = self._payload(
            goals=[
                {
                    'optaEventId': 'own-goal',
                    'contestantId': 'away-id',
                    'scorerName': 'Away Defender',
                    'periodId': 1,
                    'timeMin': 26,
                    'timeMinSec': '25:06',
                    'type': 'OG',
                },
            ],
        )

        result = enrich_match_info_with_raw_metadata(
            payload,
            {},
        )

        goal = result['goals'][0]
        self.assertEqual(goal['team_position'], 'home')
        self.assertEqual(goal['scorer'], 'Away Defender')
        self.assertEqual(goal['goal_type'], 'OG')

    def test_penalty_goal_preserves_goal_type(self):
        payload = self._payload(
            goals=[
                {
                    'optaEventId': 'penalty-goal',
                    'contestantId': 'away-id',
                    'scorerName': 'Penalty Taker',
                    'periodId': 1,
                    'timeMin': 7,
                    'timeMinSec': '6:53',
                    'type': 'PG',
                },
            ],
        )

        result = enrich_match_info_with_raw_metadata(
            payload,
            {},
        )

        self.assertEqual(
            result['goals'][0]['goal_type'],
            'PG',
        )

    def test_duplicate_goal_records_are_removed(self):
        goal = {
            'optaEventId': 'same-goal',
            'contestantId': 'home-id',
            'scorerName': 'Home Scorer',
            'periodId': 1,
            'timeMin': 30,
            'timeMinSec': '29:10',
            'type': 'G',
        }

        payload = self._payload(
            goals=[
                goal,
                copy.deepcopy(goal),
            ],
        )

        result = enrich_match_info_with_raw_metadata(
            payload,
            {},
        )

        self.assertEqual(len(result['goals']), 1)

    def test_eventing_fallback_detects_penalty_goal(self):
        payload = self._payload(
            events=[
                {
                    'id': 1001,
                    'eventId': 51,
                    'typeId': 16,
                    'contestantId': 'away-id',
                    'playerId': 'player-1',
                    'playerName': 'Away Scorer',
                    'periodId': 1,
                    'timeMin': 9,
                    'timeSec': 4,
                    'qualifier': [
                        {'qualifierId': 9},
                    ],
                },
            ],
        )

        result = enrich_match_info_with_raw_metadata(
            payload,
            {},
        )

        goal = result['goals'][0]
        self.assertEqual(goal['team_position'], 'away')
        self.assertEqual(goal['goal_type'], 'PG')
        self.assertEqual(goal['timeMinSec'], '9:04')

    def test_eventing_fallback_detects_own_goal_and_flips_team(self):
        payload = self._payload(
            events=[
                {
                    'id': 1002,
                    'eventId': 60,
                    'typeId': '16',
                    'contestantId': 'away-id',
                    'playerId': 'player-2',
                    'playerName': 'Away Defender',
                    'periodId': 2,
                    'timeMin': 58,
                    'timeSec': 10,
                    'qualifier': [
                        {'qualifierId': '28'},
                    ],
                },
            ],
        )

        result = enrich_match_info_with_raw_metadata(
            payload,
            {},
        )

        goal = result['goals'][0]
        self.assertEqual(goal['team_position'], 'home')
        self.assertEqual(goal['goal_type'], 'OG')


if __name__ == '__main__':
    unittest.main()
