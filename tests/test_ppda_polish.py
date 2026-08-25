import unittest
from pathlib import Path

import pandas as pd

from src.visualization.defensive_transitions_plotly import (
    plot_ppda_timeline,
)


class PPDAPolishTests(unittest.TestCase):
    @staticmethod
    def _profile(overall_ppda, first_ppda, second_ppda, rates):
        def snapshot(ppda, passes, actions):
            return {
                "ppda": ppda,
                "opponent_passes": passes,
                "defensive_actions": actions,
            }

        timeline = pd.DataFrame(
            {
                "minute": [7.5, 22.5, 37.5, 52.5, 67.5, 82.5],
                "interval_label": [
                    "0'–15'",
                    "15'–30'",
                    "30'–45'",
                    "45'–60'",
                    "60'–75'",
                    "75'–90'",
                ],
                "ppda": [12.0, 10.0, 8.0, 14.0, 9.0, 11.0],
                "pressure_rate": rates,
                "opponent_passes": [10, 10, 10, 10, 10, 10],
                "defensive_actions": [1, 2, 3, 1, 2, 2],
                "low_sample": [True, False, False, True, False, False],
            }
        )

        return {
            "overall": snapshot(overall_ppda, 60, 11),
            "first_half": snapshot(first_ppda, 30, 6),
            "second_half": snapshot(second_ppda, 30, 5),
            "timeline": timeline,
        }

    def test_timeline_keeps_validated_pressure_rate_values(self):
        home = self._profile(
            10.0,
            12.0,
            8.0,
            [10.0, 20.0, 30.0, 10.0, 20.0, 20.0],
        )
        away = self._profile(
            14.0,
            15.0,
            13.0,
            [8.0, 12.0, 16.0, 10.0, 14.0, 18.0],
        )

        fig = plot_ppda_timeline(
            home,
            away,
            pd.DataFrame(),
            "Home",
            "Away",
            "#e96a4a",
            "#1597c2",
        )

        bars = [trace for trace in fig.data if trace.type == "bar"]

        self.assertEqual(
            list(bars[0].y),
            list(home["timeline"]["pressure_rate"]),
        )
        self.assertEqual(
            list(bars[1].y),
            list(away["timeline"]["pressure_rate"]),
        )

    def test_event_markers_are_hoverable_traces(self):
        home = self._profile(10, 12, 8, [10,20,30,10,20,20])
        away = self._profile(14, 15, 13, [8,12,16,10,14,18])

        events = pd.DataFrame(
            [
                {
                    "minute": 23.0,
                    "event_type": "goal",
                    "team_name": "Home",
                    "playerName": "Scorer",
                    "label": "Goal · Scorer (Home)",
                },
                {
                    "minute": 64.0,
                    "event_type": "red_card",
                    "team_name": "Away",
                    "playerName": "Dismissed",
                    "label": "Red card · Dismissed (Away)",
                },
            ]
        )

        fig = plot_ppda_timeline(
            home,
            away,
            events,
            "Home",
            "Away",
            "#e96a4a",
            "#1597c2",
        )

        event_traces = [
            trace
            for trace in fig.data
            if trace.type == "scatter" and trace.name == "Match event"
        ]

        self.assertEqual(len(event_traces), 2)
        self.assertTrue(
            all(trace.hovertemplate for trace in event_traces)
        )

    def test_higher_and_lower_pressure_are_explicit(self):
        home = self._profile(10, 12, 8, [10,20,30,10,20,20])
        away = self._profile(14, 15, 13, [8,12,16,10,14,18])

        fig = plot_ppda_timeline(
            home,
            away,
            pd.DataFrame(),
            "Home",
            "Away",
            "#e96a4a",
            "#1597c2",
        )

        text = " ".join(
            str(annotation.text)
            for annotation in fig.layout.annotations
        )

        self.assertIn("HIGHER PRESSURE", text)
        self.assertIn("LOWER PRESSURE", text)

    def test_app_summary_covers_full_match_and_halves(self):
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn('"Full match"', source)
        self.assertIn('"First half"', source)
        self.assertIn('"Second half"', source)
        self.assertIn("Higher pressure vs opponent", source)
        self.assertIn("Lower pressure vs opponent", source)
        self.assertIn("Lower PPDA = higher pressure", source)


if __name__ == "__main__":
    unittest.main()
