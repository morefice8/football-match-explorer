from pathlib import Path
import unittest

from dash import dcc, html

from src.components.match_graph_shell import (
    analyst_notes_panel,
    match_graph_panel,
    match_graph_shell,
    methodology_note,
)


def walk(component):
    """Yield a Dash component tree recursively."""
    if component is None:
        return

    yield component

    children = getattr(
        component,
        "children",
        None,
    )

    if children is None:
        return

    if not isinstance(
        children,
        (list, tuple),
    ):
        children = [children]

    for child in children:
        # Leaf Dash components such as dcc.Textarea do not expose a
        # ``children`` prop, but they are still components and may carry
        # important callback IDs. Recurse through every Dash component,
        # not only container-like components.
        if hasattr(child, "to_plotly_json"):
            yield from walk(child)


class MatchGraphShellTests(unittest.TestCase):

    def test_panel_renders_header_sample_graph_and_sidebar(self):
        graph = html.Div(
            "graph",
            id="demo-graph",
        )
        sidebar = html.Div(
            "kpis",
            id="demo-sidebar",
        )

        panel = match_graph_panel(
            team_name="Genoa",
            is_away=False,
            sample_size="n = 49 attempts",
            graph=graph,
            sidebar=sidebar,
        )

        nodes = list(walk(panel))

        self.assertTrue(
            any(
                getattr(node, "children", None)
                == "HOME TEAM"
                for node in nodes
            )
        )
        self.assertTrue(
            any(
                getattr(node, "children", None)
                == "Genoa"
                for node in nodes
            )
        )
        self.assertTrue(
            any(
                getattr(node, "children", None)
                == "n = 49 attempts"
                for node in nodes
            )
        )
        self.assertTrue(
            any(
                getattr(node, "id", None)
                == "demo-graph"
                for node in nodes
            )
        )
        self.assertTrue(
            any(
                getattr(node, "id", None)
                == "demo-sidebar"
                for node in nodes
            )
        )

    def test_away_panel_uses_away_eyebrow(self):
        panel = match_graph_panel(
            team_name="Napoli",
            is_away=True,
            sample_size="n = 52 attempts",
            graph=html.Div("graph"),
        )

        nodes = list(walk(panel))

        self.assertTrue(
            any(
                getattr(node, "children", None)
                == "AWAY TEAM"
                for node in nodes
            )
        )

    def test_empty_state_preserves_shell_without_graph(self):
        panel = match_graph_panel(
            team_name="Genoa",
            is_away=False,
            sample_size="n = 0 attempts",
            graph=html.Div(
                "should not render",
                id="unused-graph",
            ),
            empty_message=(
                "No progressive pass attempts "
                "for this team."
            ),
        )

        nodes = list(walk(panel))

        self.assertFalse(
            any(
                getattr(node, "id", None)
                == "unused-graph"
                for node in nodes
            )
        )
        self.assertTrue(
            any(
                getattr(node, "children", None)
                == "No data"
                for node in nodes
            )
        )

    def test_methodology_note_uses_shared_contract(self):
        note = methodology_note(
            "Open-play only."
        )

        self.assertIn(
            "match-graph-methodology",
            note.className,
        )

        nodes = list(walk(note))
        self.assertTrue(
            any(
                getattr(node, "children", None)
                == "Open-play only."
                for node in nodes
            )
        )

    def test_analyst_notes_keep_external_callback_ids(self):
        panel = analyst_notes_panel(
            textarea_id="comment-demo",
            save_button_id="save-comment-demo",
            status_id="save-status-demo",
            description="Summarise the pattern.",
            placeholder="Write your analysis...",
        )

        nodes = list(walk(panel))
        ids = {
            getattr(node, "id", None)
            for node in nodes
        }

        self.assertIn(
            "comment-demo",
            ids,
        )
        self.assertIn(
            "save-comment-demo",
            ids,
        )
        self.assertIn(
            "save-status-demo",
            ids,
        )

        textarea = next(
            node
            for node in nodes
            if getattr(node, "id", None)
            == "comment-demo"
        )
        self.assertIsInstance(
            textarea,
            dcc.Textarea,
        )

    def test_progressive_tab_no_longer_renders_legacy_comment_editor(self):
        app_source = Path("app.py").read_text(encoding="utf-8")

        progressive_start = app_source.find(
            'label="Progressive Passes"'
        )
        final_third_start = app_source.find(
            'label="Final Third Entries"',
            progressive_start,
        )

        self.assertNotEqual(progressive_start, -1)
        self.assertNotEqual(final_third_start, -1)

        static_progressive_tab = app_source[
            progressive_start:final_third_start
        ]

        self.assertNotIn(
            'id="comment-progressive-passes"',
            static_progressive_tab,
        )
        self.assertNotIn(
            'id="save-comment-progressive-passes"',
            static_progressive_tab,
        )
        self.assertNotIn(
            "Comments for Progressive Passes:",
            static_progressive_tab,
        )

    def test_shell_orders_methodology_panels_and_notes(self):
        home = html.Div(
            "home",
            id="home-panel",
        )
        away = html.Div(
            "away",
            id="away-panel",
        )

        shell = match_graph_shell(
            methodology="Method note",
            panels=[home, away],
            analyst_notes={
                "textarea_id": "comment-demo",
                "save_button_id": "save-comment-demo",
                "status_id": "save-status-demo",
                "description": "Analyse.",
                "placeholder": "Write...",
            },
            class_name="demo-analysis",
        )

        self.assertIn(
            "match-graph-shell",
            shell.className,
        )
        self.assertIn(
            "demo-analysis",
            shell.className,
        )

        nodes = list(walk(shell))
        ids = [
            getattr(node, "id", None)
            for node in nodes
            if getattr(node, "id", None)
        ]

        self.assertLess(
            ids.index("home-panel"),
            ids.index("away-panel"),
        )
        self.assertLess(
            ids.index("away-panel"),
            ids.index("comment-demo"),
        )


if __name__ == "__main__":
    unittest.main()
