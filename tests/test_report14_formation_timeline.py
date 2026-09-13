from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import unittest
from unittest.mock import patch

from PIL import Image
import plotly.graph_objects as go

from src.reporting.audit import pdf_page_count, searchable_pdf_text
from src.reporting.bundle import (
    MatchReportDataBundle,
    ReportSectionBundle,
    ReportSectionStatus,
)
from src.reporting.figure_catalog import (
    RendererRegistry,
    _formation_plot_moments,
    build_figure_plans,
    build_report_figure_catalog,
)
from src.reporting.manifest import REPORT_MANIFEST
from src.reporting.models import ReportManifest, ReportScope
from src.reporting.pdf_renderer import (
    _formation_spells_rows,
    _prepare_pdf_table_frame,
    render_match_report_pdf,
)


def _state(formation_id: int, prefix: str, *, replacement: tuple[str, str] | None = None):
    players = {f"{prefix}{index}": index for index in range(1, 12)}
    if replacement is not None:
        old_id, new_id = replacement
        slot = players.pop(old_id)
        players[new_id] = slot
    return {"formation_id": formation_id, "players": players}


def _event(kind: str, team: str, description: str):
    return {"kind": kind, "team": team, "description": description}


def _moment(
    minute: int,
    *,
    home_state,
    away_state,
    home_shape: str,
    away_shape: str,
    score: str = "0 - 0",
    events=(),
    home_highlights=(),
    away_highlights=(),
):
    return {
        "time_seconds": minute * 60,
        "time_label": f"{minute}'",
        "score": score,
        "home_state": home_state,
        "away_state": away_state,
        "home_formation_name": home_shape,
        "away_formation_name": away_shape,
        "events": list(events),
        "home_highlights": list(home_highlights),
        "away_highlights": list(away_highlights),
    }


def _formation_model(*, final_distinct: bool = True):
    start_home = _state(4, "h")
    start_away = _state(4, "a")
    sub_home = _state(4, "h", replacement=("h2", "h12"))
    tactical_away = _state(8, "a")
    final_home = (
        _state(4, "h", replacement=("h1", "h13"))
        if final_distinct
        else sub_home
    )

    moments = [
        _moment(
            0,
            home_state=start_home,
            away_state=start_away,
            home_shape="4-3-3",
            away_shape="4-3-3",
            events=[_event("starting_xi", "both", "Starting XI")],
        ),
        _moment(
            20,
            home_state=sub_home,
            away_state=start_away,
            home_shape="4-3-3",
            away_shape="4-3-3",
            events=[
                _event(
                    "substitution",
                    "home",
                    "Home FC · #12 Home Player 12 on for #2 Home Player 2",
                )
            ],
            home_highlights=["h12"],
        ),
        _moment(
            30,
            home_state=sub_home,
            away_state=start_away,
            home_shape="4-3-3",
            away_shape="4-3-3",
            score="1 - 0",
            events=[_event("goal", "home", "Goal · Home Player 9")],
        ),
        _moment(
            40,
            home_state=sub_home,
            away_state=tactical_away,
            home_shape="4-3-3",
            away_shape="4-4-2",
            score="1 - 0",
            events=[
                _event(
                    "formation_change",
                    "away",
                    "Away FC · shape change to 4-4-2",
                )
            ],
            away_highlights=["a7", "a8"],
        ),
    ]

    if final_distinct:
        moments.append(
            _moment(
                70,
                home_state=final_home,
                away_state=tactical_away,
                home_shape="4-3-3",
                away_shape="4-4-2",
                score="1 - 0",
                events=[
                    _event(
                        "substitution",
                        "home",
                        "Home FC · #13 Home Player 13 on for #1 Home Player 1",
                    )
                ],
                home_highlights=["h13"],
            )
        )
    else:
        moments.append(
            _moment(
                70,
                home_state=sub_home,
                away_state=tactical_away,
                home_shape="4-3-3",
                away_shape="4-4-2",
                score="1 - 1",
                events=[_event("goal", "away", "Goal · Away Player 9")],
            )
        )

    player_data = {
        **{f"h{i}": {"name": f"Home Player {i}", "jersey": str(i)} for i in range(1, 14)},
        **{f"a{i}": {"name": f"Away Player {i}", "jersey": str(i)} for i in range(1, 12)},
    }
    return {
        "home_team": "Home FC",
        "away_team": "Away FC",
        "player_data": player_data,
        "moments": moments,
        "match_end_seconds": 90 * 60,
    }


def _bundle(model) -> MatchReportDataBundle:
    return MatchReportDataBundle(
        manifest_id=REPORT_MANIFEST.id,
        manifest_version=REPORT_MANIFEST.schema_version,
        source_signature="report14-formation-test",
        scope=ReportScope.FULL_MATCH,
        teams=("Home FC", "Away FC"),
        match_info={
            "hteamName": "Home FC",
            "ateamName": "Away FC",
            "competitionName": "Serie A",
            "date_formatted": "13 September 2026",
        },
        sections=(
            ReportSectionBundle(
                id="formation-timeline",
                status=ReportSectionStatus.GENERATED,
                data=model,
            ),
        ),
    )


def _formation_manifest() -> ReportManifest:
    original = next(
        section
        for section in REPORT_MANIFEST.sections
        if section.id == "formation-timeline"
    )
    return ReportManifest(
        id="report14-formation-only",
        schema_version=REPORT_MANIFEST.schema_version,
        sections=(replace(original, order=1),),
    )


def _tiny_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 18), "white").save(output, format="PNG")
    return output.getvalue()


def _dummy_renderer(*args, **kwargs):
    return go.Figure()


class Report14FormationTimelineTests(unittest.TestCase):
    def test_compact_spell_table_has_only_required_scalar_fields(self):
        model = _formation_model()
        frame = _formation_spells_rows(
            model["moments"],
            match_end_seconds=model["match_end_seconds"],
        )
        prepared = _prepare_pdf_table_frame("formation-spells", frame)

        self.assertEqual(
            list(prepared.columns),
            [
                "start",
                "end",
                "duration",
                "score",
                "home formation",
                "away formation",
                "change reason",
                "subs / dismissals",
            ],
        )
        self.assertEqual(prepared.iloc[0]["start"], "0'")
        self.assertEqual(prepared.iloc[0]["end"], "20'")
        self.assertEqual(prepared.iloc[0]["duration"], "20m")
        self.assertEqual(prepared.iloc[-1]["end"], "90'")
        self.assertEqual(prepared.iloc[-1]["duration"], "20m")
        self.assertIn("Substitution", prepared.iloc[1]["change reason"])
        self.assertIn("Home Player 12", prepared.iloc[1]["subs / dismissals"])
        self.assertEqual(prepared.iloc[2]["change reason"], "Goal")
        self.assertEqual(prepared.iloc[3]["change reason"], "Tactical change")
        for forbidden in ("home_state", "away_state", "player_data", "events"):
            self.assertNotIn(forbidden, prepared.columns)

    def test_plot_selection_is_start_relevant_tactical_change_then_distinct_final(self):
        model = _formation_model(final_distinct=True)
        selected = _formation_plot_moments(model)
        self.assertEqual(list(selected), ["starting", "intermediate", "final"])
        self.assertEqual(selected["starting"]["time_seconds"], 0)
        self.assertEqual(selected["intermediate"]["time_seconds"], 40 * 60)
        self.assertEqual(selected["final"]["time_seconds"], 70 * 60)

        plans = [
            plan
            for plan in build_figure_plans(_bundle(model), _formation_manifest())
            if plan.figure_id == "formation-timeline-figure"
        ]
        self.assertEqual(len(plans), 6)
        self.assertEqual(
            {(plan.team_name, plan.variant) for plan in plans},
            {
                ("Home FC", "starting"),
                ("Away FC", "starting"),
                ("Home FC", "intermediate"),
                ("Away FC", "intermediate"),
                ("Home FC", "final"),
                ("Away FC", "final"),
            },
        )

    def test_duplicate_final_state_is_not_plotted(self):
        model = _formation_model(final_distinct=False)
        selected = _formation_plot_moments(model)
        self.assertEqual(list(selected), ["starting", "intermediate"])
        self.assertEqual(selected["intermediate"]["time_seconds"], 40 * 60)

        plans = [
            plan
            for plan in build_figure_plans(_bundle(model), _formation_manifest())
            if plan.figure_id == "formation-timeline-figure"
        ]
        self.assertEqual(len(plans), 4)
        self.assertEqual({plan.variant for plan in plans}, {"starting", "intermediate"})

    def test_formation_section_stays_within_two_pages(self):
        model = _formation_model(final_distinct=True)
        bundle = _bundle(model)
        manifest = _formation_manifest()
        png = _tiny_png()

        with patch.object(RendererRegistry, "resolve", return_value=_dummy_renderer), patch(
            "src.reporting.pdf_renderer.pio.to_image",
            return_value=png,
        ):
            catalog = build_report_figure_catalog(bundle, manifest)
            pdf = render_match_report_pdf(bundle, catalog, manifest)

        # Cover + TOC always consume two pages. REPORT-14 therefore allows at
        # most two additional pages for Formation Timeline itself.
        self.assertLessEqual(pdf_page_count(pdf), 4)
        text = searchable_pdf_text(pdf)
        self.assertIn("Formation spells", text)
        self.assertIn("Intermediate @ 40' / 1 - 0", text)
        self.assertIn("Home Player 12", text)
        self.assertNotIn("home_state", text)
        self.assertNotIn("away_state", text)
        self.assertNotIn("player_data", text)


if __name__ == "__main__":
    unittest.main()
