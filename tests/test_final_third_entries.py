import unittest

import pandas as pd

from src.metrics.pass_metrics import analyze_final_third_entries


class FinalThirdEntryTests(unittest.TestCase):

    def test_counts_completed_pass_crossing_final_third_boundary(self):
        passes = pd.DataFrame([
            {
                'x': 60,
                'y': 50,
                'end_x': 70,
                'end_y': 50,
                'playerName': 'Player A',
            }
        ])

        entries, stats = analyze_final_third_entries(passes)

        self.assertEqual(len(entries), 1)
        self.assertEqual(stats['total_final_third'], 1)
        self.assertEqual(stats['pass_entries'], 1)
        self.assertEqual(stats['carry_entries'], 0)

    def test_does_not_count_pass_starting_inside_final_third(self):
        passes = pd.DataFrame([
            {
                'x': 75,
                'y': 50,
                'end_x': 90,
                'end_y': 50,
            }
        ])

        entries, stats = analyze_final_third_entries(passes)

        self.assertTrue(entries.empty)
        self.assertEqual(stats['total_final_third'], 0)

    def test_does_not_count_pass_ending_before_final_third(self):
        passes = pd.DataFrame([
            {
                'x': 40,
                'y': 50,
                'end_x': 65,
                'end_y': 50,
            }
        ])

        entries, stats = analyze_final_third_entries(passes)

        self.assertTrue(entries.empty)
        self.assertEqual(stats['total_final_third'], 0)

    def test_combines_pass_and_carry_entries(self):
        passes = pd.DataFrame([
            {
                'x': 55,
                'y': 50,
                'end_x': 70,
                'end_y': 50,
            }
        ])

        carries = pd.DataFrame([
            {
                'x': 64,
                'y': 30,
                'end_x': 72,
                'end_y': 28,
                'carry_is_reliable': True,
            }
        ])

        entries, stats = analyze_final_third_entries(
            passes,
            carries,
        )

        self.assertEqual(stats['total_final_third'], 2)
        self.assertEqual(stats['pass_entries'], 1)
        self.assertEqual(stats['carry_entries'], 1)

    def test_excludes_unreliable_carry(self):
        carries = pd.DataFrame([
            {
                'x': 64,
                'y': 50,
                'end_x': 72,
                'end_y': 50,
                'carry_is_reliable': False,
            }
        ])

        entries, stats = analyze_final_third_entries(
            pd.DataFrame(),
            carries,
        )

        self.assertTrue(entries.empty)
        self.assertEqual(stats['carry_entries'], 0)

    def test_classifies_zone14_entry(self):
        passes = pd.DataFrame([
            {
                'x': 60,
                'y': 50,
                'end_x': 75,
                'end_y': 50,
            }
        ])

        entries, stats = analyze_final_third_entries(passes)

        self.assertEqual(
            entries.iloc[0]['destination_zone'],
            'Zone 14',
        )
        self.assertEqual(stats['zone14'], 1)

    def test_classifies_right_halfspace_entry(self):
        passes = pd.DataFrame([
            {
                'x': 60,
                'y': 20,
                'end_x': 75,
                'end_y': 25,
            }
        ])

        entries, stats = analyze_final_third_entries(passes)

        self.assertEqual(
            entries.iloc[0]['destination_zone'],
            'Right Half-Space',
        )
        self.assertEqual(stats['hs_right'], 1)

    def test_counts_wide_entry_as_final_third_entry(self):
        passes = pd.DataFrame([
            {
                'x': 55,
                'y': 5,
                'end_x': 75,
                'end_y': 5,
            }
        ])

        entries, stats = analyze_final_third_entries(passes)

        self.assertEqual(stats['total_final_third'], 1)
        self.assertEqual(stats['other'], 1)


if __name__ == '__main__':
    unittest.main()