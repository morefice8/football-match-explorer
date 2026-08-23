import unittest
from pathlib import Path

import pandas as pd

from src.components import cross_flow_view
from src.metrics import cross_metrics
from src.visualization import cross_plots


def event(
    event_id,
    team,
    event_type,
    *,
    period=1,
    minute=10,
    second=0,
    cross=0,
    outcome="Successful",
    player="Player",
    x=70,
    y=80,
    end_x=90,
    end_y=50,
    key_pass=False,
    assist=False,
):
    return {
        "eventId": event_id,
        "id": event_id,
        "team_name": team,
        "type_name": event_type,
        "periodId": period,
        "timeMin": minute,
        "timeSec": second,
        "cross": cross,
        "outcome": outcome,
        "playerName": player,
        "x": x,
        "y": y,
        "end_x": end_x,
        "end_y": end_y,
        "is_key_pass": key_pass,
        "is_assist": assist,
        "Corner taken": 0,
        "Freekick taken": 0,
        "Right footed": 1,
        "Left footed": 0,
        "In-swinger": 0,
        "Out-swinger": 0,
        "Straight": 1,
    }


def walk(component):
    yield component

    if component is None:
        return

    children = getattr(
        component,
        "children",
        None,
    )

    if children is None:
        return

    if isinstance(
        children,
        (list, tuple),
    ):
        for child in children:
            yield from walk(
                child
            )
    else:
        yield from walk(
            children
        )


class CrossFlowViewTests(
    unittest.TestCase
):

    def test_successful_cross_is_retained(self):
        frame = pd.DataFrame([
            event(
                1,
                "Home",
                "Pass",
                cross=1,
            ),
            event(
                2,
                "Away",
                "Ball recovery",
                second=4,
            ),
        ])

        result = (
            cross_metrics
            .analyze_crosses(
                frame,
                "Home",
            )
        )

        self.assertTrue(
            bool(
                result.loc[
                    0,
                    "Retained",
                ]
            )
        )

    def test_second_ball_can_retain_failed_cross(self):
        frame = pd.DataFrame([
            event(
                1,
                "Home",
                "Pass",
                cross=1,
                outcome="Unsuccessful",
            ),
            event(
                2,
                "Home",
                "Ball recovery",
                second=3,
            ),
            event(
                3,
                "Away",
                "Pass",
                second=7,
            ),
        ])

        result = (
            cross_metrics
            .analyze_crosses(
                frame,
                "Home",
            )
        )

        self.assertTrue(
            bool(
                result.loc[
                    0,
                    "Retained",
                ]
            )
        )

    def test_opponent_control_ends_post_cross_window(self):
        frame = pd.DataFrame([
            event(
                1,
                "Home",
                "Pass",
                cross=1,
                outcome="Unsuccessful",
            ),
            event(
                2,
                "Away",
                "Ball recovery",
                second=2,
            ),
            event(
                3,
                "Home",
                "Goal",
                second=5,
            ),
        ])

        result = (
            cross_metrics
            .analyze_crosses(
                frame,
                "Home",
            )
        )

        self.assertFalse(
            bool(
                result.loc[
                    0,
                    "Retained",
                ]
            )
        )

        self.assertFalse(
            bool(
                result.loc[
                    0,
                    "Shot Generated",
                ]
            )
        )

    def test_shot_inside_window_is_generated(self):
        frame = pd.DataFrame([
            event(
                1,
                "Home",
                "Pass",
                cross=1,
                outcome="Unsuccessful",
            ),
            event(
                2,
                "Home",
                "Ball touch",
                second=2,
            ),
            event(
                3,
                "Home",
                "Attempt Saved",
                second=7,
            ),
        ])

        result = (
            cross_metrics
            .analyze_crosses(
                frame,
                "Home",
            )
        )

        self.assertTrue(
            bool(
                result.loc[
                    0,
                    "Retained",
                ]
            )
        )

        self.assertTrue(
            bool(
                result.loc[
                    0,
                    "Shot Generated",
                ]
            )
        )

    def test_shot_does_not_cross_period_boundary(self):
        frame = pd.DataFrame([
            event(
                1,
                "Home",
                "Pass",
                cross=1,
                outcome="Unsuccessful",
                period=1,
                minute=45,
                second=0,
            ),
            event(
                2,
                "Home",
                "Goal",
                period=2,
                minute=45,
                second=2,
            ),
        ])

        result = (
            cross_metrics
            .analyze_crosses(
                frame,
                "Home",
            )
        )

        self.assertFalse(
            bool(
                result.loc[
                    0,
                    "Shot Generated",
                ]
            )
        )

    def test_canonical_key_pass_marks_shot_generation(self):
        frame = pd.DataFrame([
            event(
                1,
                "Home",
                "Pass",
                cross=1,
                key_pass=True,
            ),
        ])

        result = (
            cross_metrics
            .analyze_crosses(
                frame,
                "Home",
            )
        )

        self.assertTrue(
            bool(
                result.loc[
                    0,
                    "Shot Generated",
                ]
            )
        )

    def test_flow_profile_exposes_requested_kpis(self):
        analyzed = pd.DataFrame([
            {
                "Origin Zone":
                    "Left Advanced",
                "Destination Zone":
                    "Center Deep",
                "Outcome":
                    "Completed",
                "Retained":
                    True,
                "Shot Generated":
                    True,
                "playerName":
                    "A",
            },
            {
                "Origin Zone":
                    "Left Advanced",
                "Destination Zone":
                    "Center Deep",
                "Outcome":
                    "Incomplete",
                "Retained":
                    True,
                "Shot Generated":
                    False,
                "playerName":
                    "A",
            },
            {
                "Origin Zone":
                    "Right Advanced",
                "Destination Zone":
                    "Center Deep",
                "Outcome":
                    "Incomplete",
                "Retained":
                    False,
                "Shot Generated":
                    False,
                "playerName":
                    "B",
            },
        ])

        summary, routes = (
            cross_metrics
            .build_cross_flow_profile(
                analyzed
            )
        )

        self.assertEqual(
            summary[
                "total_crosses"
            ],
            3,
        )

        self.assertAlmostEqual(
            summary[
                "retention_pct"
            ],
            200 / 3,
        )

        self.assertAlmostEqual(
            summary[
                "shot_rate_pct"
            ],
            100 / 3,
        )

        self.assertEqual(
            summary[
                "top_crosser"
            ],
            "A",
        )

        top = routes.iloc[0]

        self.assertEqual(
            int(
                top[
                    "Crosses"
                ]
            ),
            2,
        )

        self.assertAlmostEqual(
            float(
                top[
                    "Retention %"
                ]
            ),
            100.0,
        )

        self.assertAlmostEqual(
            float(
                top[
                    "Shot Rate %"
                ]
            ),
            50.0,
        )

    def test_flow_rows_are_selectable_routes(self):
        routes = pd.DataFrame([
            {
                "Origin Zone":
                    "Left Advanced",
                "Destination Zone":
                    "Center Deep",
                "Crosses":
                    4,
                "Share %":
                    40.0,
                "Completed":
                    2,
                "Completion %":
                    50.0,
                "Retained":
                    3,
                "Retention %":
                    75.0,
                "Shots":
                    2,
                "Shot Rate %":
                    50.0,
            }
        ])

        component = (
            cross_flow_view
            .flow_panel(
                {
                    "total_crosses": 10,
                    "retained_crosses": 7,
                    "retention_pct": 70.0,
                    "shot_crosses": 3,
                    "shot_rate_pct": 30.0,
                    "top_crosser": "Player A",
                    "top_crosser_count": 5,
                },
                routes,
                "#e96a4a",
            )
        )

        route_ids = [
            getattr(
                child,
                "id",
                None,
            )
            for child
            in walk(component)
            if isinstance(
                getattr(
                    child,
                    "id",
                    None,
                ),
                dict,
            )
        ]

        self.assertIn(
            {
                "type":
                    "cross-flow-route",
                "origin":
                    "Left Advanced",
                "destination":
                    "Center Deep",
            },
            route_ids,
        )

    def test_selected_flow_is_highlighted_on_heatmap(self):
        analyzed = pd.DataFrame([
            {
                "cross_id": 1,
                "playerName": "A",
                "Foot": "Right",
                "Swing": "Straight",
                "Play Type": "Open Play",
                "Outcome": "Completed",
                "Origin Zone":
                    "Left Advanced",
                "Destination Zone":
                    "Center Deep",
                "x": 70,
                "y": 80,
                "end_x": 90,
                "end_y": 50,
            },
            {
                "cross_id": 2,
                "playerName": "B",
                "Foot": "Right",
                "Swing": "Straight",
                "Play Type": "Open Play",
                "Outcome": "Incomplete",
                "Origin Zone":
                    "Right Advanced",
                "Destination Zone":
                    "Center Deep",
                "x": 70,
                "y": 20,
                "end_x": 90,
                "end_y": 50,
            },
        ])

        route = {
            "origin":
                "Left Advanced",
            "destination":
                "Center Deep",
        }

        origin = (
            cross_plots
            .plot_cross_heatmap(
                analyzed,
                "origin",
                False,
                selected_flow_route=route,
            )
        )

        destination = (
            cross_plots
            .plot_cross_heatmap(
                analyzed,
                "destination",
                False,
                selected_flow_route=route,
            )
        )

        for fig in (
            origin,
            destination,
        ):
            selected = [
                trace
                for trace in fig.data
                if getattr(
                    trace,
                    "name",
                    None,
                )
                == "selected_flow_points"
            ]

            self.assertEqual(
                len(selected),
                1,
            )

            self.assertEqual(
                len(
                    selected[0].x
                ),
                1,
            )

    def test_app_wires_flow_selection_to_both_maps(self):
        source = Path(
            "app.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "cross-flow-selection-store",
            source,
        )

        self.assertIn(
            "def select_cross_flow_route(",
            source,
        )

        self.assertIn(
            "selected_flow_route=(",
            source,
        )

        self.assertIn(
            'label="Cross flow"',
            source,
        )

    def test_cross_flow_has_no_global_reset_callback_depending_on_dynamic_team_tabs(self):
        source = Path(
            "app.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            "def clear_cross_flow_selection(",
            source,
        )


if __name__ == "__main__":
    unittest.main()
