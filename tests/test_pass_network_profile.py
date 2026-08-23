import inspect
import unittest

import pandas as pd

import app
from src.data_processing import pass_processing
from src.metrics import pass_network_metrics
from src.visualization import pass_plotly


class PassNetworkProfileTests(unittest.TestCase):
    @staticmethod
    def raw_df():
        rows = []
        for name, jersey in (("A", 2), ("B", 4), ("C", 6)):
            for minute, period in ((0, 1), (45, 1), (46, 2), (90, 2)):
                rows.append({
                    "id": f"{name}-{minute}", "eventId": f"{name}-{minute}",
                    "typeId": 1, "periodId": period, "timeMin": minute, "timeSec": 0,
                    "team_name": "Team A", "playerName": name,
                    "Mapped Jersey Number": jersey, "Is Starter": True,
                })
        # E: valid substitute; D: late sub below 15 minutes.
        for name, jersey, on in (("E", 18, 60), ("D", 20, 80)):
            rows.append({
                "id": f"{name}-on", "eventId": f"{name}-on", "typeId": 19,
                "periodId": 2, "timeMin": on, "timeSec": 0, "team_name": "Team A",
                "playerName": name, "Mapped Jersey Number": jersey, "Is Starter": False,
            })
            rows.append({
                "id": f"{name}-90", "eventId": f"{name}-90", "typeId": 1,
                "periodId": 2, "timeMin": 90, "timeSec": 0, "team_name": "Team A",
                "playerName": name, "Mapped Jersey Number": jersey, "Is Starter": False,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def pass_row(passer, receiver, minute, period, reliable=True, x=40, y=50):
        return {
            "id": f"{passer}-{receiver}-{minute}", "eventId": f"{passer}-{receiver}-{minute}",
            "periodId": period, "timeMin": minute, "timeSec": 0,
            "team_name": "Team A", "playerName": passer,
            "Mapped Jersey Number": 2, "receiver": receiver,
            "receiver_jersey_number": 4, "outcome": "Successful",
            "receiver_is_reliable": reliable, "x": x, "y": y,
            "end_x": x + 10, "end_y": y + 3,
        }

    @classmethod
    def passes_df(cls):
        rows = []
        for m in (5, 10, 15, 20): rows.append(cls.pass_row("A", "B", m, 1))
        for m in (25, 30, 35): rows.append(cls.pass_row("B", "A", m, 1))
        for m in (50, 55, 60, 65): rows.append(cls.pass_row("A", "C", m, 2))
        for m in (70, 75): rows.append(cls.pass_row("C", "A", m, 2))
        for m in (62, 66, 70, 74, 78): rows.append(cls.pass_row("E", "B", m, 2))
        for m in (81, 82, 83, 84, 85, 86): rows.append(cls.pass_row("D", "A", m, 2))
        for m in (40, 41, 42, 43, 44, 45): rows.append(cls.pass_row("A", "E", m, 1, reliable=False))
        return pd.DataFrame(rows)

    def profile(self, **kwargs):
        options = dict(period="full", min_minutes=15, min_connection=5, top_n=8)
        options.update(kwargs)
        return pass_network_metrics.build_pass_network_profile(
            self.passes_df(), self.raw_df(), "Team A", **options,
        )

    def test_unreliable_links_are_excluded(self):
        edges, _, summary = self.profile()
        pairs = set(zip(edges["player1"], edges["player2"]))
        self.assertNotIn(("A", "E"), pairs)
        self.assertGreater(summary["successful_passes"], summary["reliable_passes"])

    def test_directional_counts_are_preserved(self):
        edges, _, _ = self.profile()
        ab = edges[(edges["player1"] == "A") & (edges["player2"] == "B")].iloc[0]
        self.assertEqual(int(ab["player1_to_player2"]), 4)
        self.assertEqual(int(ab["player2_to_player1"]), 3)
        self.assertEqual(int(ab["pass_count"]), 7)

    def test_period_filter_changes_network(self):
        first, _, _ = self.profile(period="1h")
        second, _, _ = self.profile(period="2h")
        self.assertIn(("A", "B"), set(zip(first["player1"], first["player2"])))
        self.assertNotIn(("A", "C"), set(zip(first["player1"], first["player2"])))
        self.assertIn(("A", "C"), set(zip(second["player1"], second["player2"])))

    def test_late_sub_is_excluded(self):
        edges, nodes, _ = self.profile()
        self.assertNotIn("D", set(nodes["playerName"]))
        self.assertFalse((edges["player1"].eq("D") | edges["player2"].eq("D")).any())

    def test_node_involvement_is_sent_plus_received(self):
        _, nodes, _ = self.profile()
        node = nodes[nodes["playerName"] == "A"].iloc[0]
        self.assertEqual(
            int(node["pass_involvement"]),
            int(node["pass_sent"]) + int(node["pass_received"]),
        )

    def test_substitute_status_is_exposed(self):
        _, nodes, _ = self.profile()
        node = nodes[nodes["playerName"] == "E"].iloc[0]
        self.assertEqual(node["status"], "Substitute")

    def test_top_n_caps_initial_view(self):
        edges, _, summary = self.profile(top_n=2)
        self.assertLessEqual(len(edges), 2)
        self.assertEqual(summary["shown_connections"], len(edges))
        self.assertGreaterEqual(summary["qualifying_connections"], len(edges))

    def test_plot_has_directional_hover_and_sub_symbols(self):
        edges, nodes, _ = self.profile()
        fig = pass_plotly.plot_pass_network_profile_plotly(edges, nodes, "Team A")
        hover = " ".join(str(getattr(trace, "hovertemplate", "")) for trace in fig.data)
        symbols = {
            str(trace.marker.symbol) for trace in fig.data
            if getattr(trace, "marker", None) and getattr(trace.marker, "symbol", None)
        }
        self.assertIn("customdata[3]", hover)
        self.assertIn("customdata[4]", hover)
        self.assertIn("circle", symbols)
        self.assertIn("diamond", symbols)

    def test_app_exposes_period_and_threshold_controls(self):
        source = inspect.getsource(app.show_pass_network_graph_plotly)
        self.assertIn("pass_network_view.controls", source)
        callback_source = inspect.getsource(app.update_pass_network_period_content)
        self.assertIn("render_period_view", callback_source)

    def test_pass_processing_retains_period_id(self):
        self.assertIn('"periodId"', inspect.getsource(pass_processing.get_passes_df))


    def test_plot_only_shows_nodes_from_displayed_connections(self):
        edges, nodes, _ = self.profile(
            min_connection=5,
            top_n=1,
        )

        figure = (
            pass_plotly
            .plot_pass_network_profile_plotly(
                edges,
                nodes,
                "Team A",
                is_away=False,
            )
        )

        expected_players = set(
            edges["player1"].astype(str)
        ).union(
            set(
                edges["player2"].astype(str)
            )
        )

        plotted_players = set()

        for trace in figure.data:
            customdata = getattr(
                trace,
                "customdata",
                None,
            )

            if customdata is None:
                continue

            for row in customdata:
                if (
                    row is not None
                    and len(row) == 5
                ):
                    plotted_players.add(
                        str(row[0])
                    )

        self.assertEqual(
            plotted_players,
            expected_players,
        )

        self.assertFalse(
            bool(
                figure.layout.showlegend
            )
        )



if __name__ == "__main__":
    unittest.main()
