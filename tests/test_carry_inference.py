import unittest

import pandas as pd

from src.data_processing.pass_processing import infer_carries


class CarryInferenceTests(unittest.TestCase):

    def test_infers_carry_after_reliable_completed_pass(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'periodId': 1,
                'timeMin': 10,
                'timeSec': 0,
                'typeId': 1,
                'type_name': 'Pass',
                'outcome': 'Successful',
                'contestantId': 'A',
                'team_name': 'Team A',
                'playerId': 10,
                'playerName': 'Player A',
                'x': 50,
                'y': 50,
                'end_x': 64,
                'end_y': 50,
            },
            {
                'id': 2,
                'periodId': 1,
                'timeMin': 10,
                'timeSec': 4,
                'typeId': 3,
                'type_name': 'Take On',
                'outcome': 'Successful',
                'contestantId': 'A',
                'team_name': 'Team A',
                'playerId': 20,
                'playerName': 'Player B',
                'x': 70,
                'y': 50,
                'end_x': 74,
                'end_y': 51,
            },
        ])

        carries = infer_carries(df)

        self.assertEqual(len(carries), 1)

        carry = carries.iloc[0]

        self.assertEqual(carry['playerName'], 'Player B')
        self.assertEqual(carry['x'], 64)
        self.assertEqual(carry['end_x'], 70)
        self.assertTrue(carry['carry_is_reliable'])

    def test_does_not_infer_carry_after_unsuccessful_pass(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'periodId': 1,
                'timeMin': 10,
                'timeSec': 0,
                'typeId': 1,
                'type_name': 'Pass',
                'outcome': 'Unsuccessful',
                'contestantId': 'A',
                'playerId': 10,
                'playerName': 'Player A',
                'x': 50,
                'y': 50,
                'end_x': 64,
                'end_y': 50,
            },
            {
                'id': 2,
                'periodId': 1,
                'timeMin': 10,
                'timeSec': 3,
                'type_name': 'Ball Recovery',
                'contestantId': 'B',
                'playerId': 30,
                'playerName': 'Opponent',
                'x': 70,
                'y': 50,
                'end_x': 72,
                'end_y': 50,
            },
        ])

        carries = infer_carries(df)

        self.assertTrue(carries.empty)

    def test_infers_same_player_carry_between_actions(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'periodId': 1,
                'timeMin': 20,
                'timeSec': 0,
                'type_name': 'Ball Recovery',
                'contestantId': 'A',
                'playerId': 20,
                'playerName': 'Player B',
                'x': 55,
                'y': 40,
                'end_x': 57,
                'end_y': 40,
            },
            {
                'id': 2,
                'periodId': 1,
                'timeMin': 20,
                'timeSec': 5,
                'type_name': 'Take On',
                'contestantId': 'A',
                'playerId': 20,
                'playerName': 'Player B',
                'x': 68,
                'y': 42,
                'end_x': 72,
                'end_y': 43,
            },
        ])

        carries = infer_carries(df)

        self.assertEqual(len(carries), 1)
        self.assertEqual(carries.iloc[0]['playerId'], 20)

    def test_does_not_assign_non_pass_gap_to_different_teammate(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'periodId': 1,
                'timeMin': 30,
                'timeSec': 0,
                'type_name': 'Ball Recovery',
                'contestantId': 'A',
                'playerId': 10,
                'playerName': 'Player A',
                'x': 50,
                'y': 50,
                'end_x': 55,
                'end_y': 50,
            },
            {
                'id': 2,
                'periodId': 1,
                'timeMin': 30,
                'timeSec': 4,
                'type_name': 'Take On',
                'contestantId': 'A',
                'playerId': 20,
                'playerName': 'Player B',
                'x': 65,
                'y': 50,
                'end_x': 67,
                'end_y': 50,
            },
        ])

        carries = infer_carries(df)

        self.assertTrue(carries.empty)

    def test_ignores_tiny_movements(self):
        df = pd.DataFrame([
            {
                'id': 1,
                'periodId': 1,
                'timeMin': 40,
                'timeSec': 0,
                'type_name': 'Ball Recovery',
                'contestantId': 'A',
                'playerId': 20,
                'playerName': 'Player B',
                'x': 60,
                'y': 50,
                'end_x': 60,
                'end_y': 50,
            },
            {
                'id': 2,
                'periodId': 1,
                'timeMin': 40,
                'timeSec': 2,
                'type_name': 'Take On',
                'contestantId': 'A',
                'playerId': 20,
                'playerName': 'Player B',
                'x': 60.5,
                'y': 50,
                'end_x': 62,
                'end_y': 50,
            },
        ])

        carries = infer_carries(df)

        self.assertTrue(carries.empty)


if __name__ == '__main__':
    unittest.main()