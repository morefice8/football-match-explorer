"""Single source of truth for Sportmonks metrics shown in the app.

The registry intentionally contains only metrics that can be reproduced from
the fixture statistics included in the audited Sportmonks subscription.  It is
kept free of Dash/Plotly imports so formulas, labels and ranking direction can
be tested without loading the web application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MetricSpec:
    column: str
    title: str
    tooltip: str
    icon: str
    ascending: bool = False
    unit: str = ""
    format_spec: str = "{:,.2f}"

    def card_kwargs(self, column_key: str) -> dict[str, Any]:
        """Return the keyword shape expected by a ranking-card component."""
        return {
            "title": self.title,
            column_key: self.column,
            "icon": self.icon,
            "ascending": self.ascending,
            "unit": self.unit,
            "format_spec": self.format_spec,
        }


TEAM_METRIC_GROUPS: dict[str, tuple[MetricSpec, ...]] = {
    "attacking": (
        MetricSpec("Gls_per_90", "Goals per 90", "Goals scored per 90 team minutes.", "fa-solid fa-futbol"),
        MetricSpec("npxG_per_90", "Non-Penalty xG per 90", "Sportmonks non-penalty expected goals per 90 team minutes. This compares open-play and set-piece threat without penalty noise.", "fa-solid fa-chart-line"),
        MetricSpec("xG_per_Shot", "xG per Shot", "Average expected-goal value of a shot. Higher values indicate better average shot quality.", "fa-solid fa-bullseye", format_spec="{:.3f}"),
        MetricSpec("Shooting_Performance_per_90", "Shooting Performance p90", "Sportmonks shooting performance (xG on target minus xG), normalized per 90. Positive values indicate that shot placement added value.", "fa-solid fa-arrow-trend-up"),
    ),
    "possession": (
        MetricSpec("Poss", "Possession", "Mean ball-possession percentage across imported fixtures.", "fa-solid fa-circle-half-stroke", unit="%", format_spec="{:,.1f}"),
        MetricSpec("Pass_Completion_Perc", "Pass Completion", "Accurate passes divided by attempted passes.", "fa-solid fa-check-double", unit="%", format_spec="{:,.1f}"),
        MetricSpec("FinalThird_per_90", "Final Third Passes p90", "Passes into the attacking third per 90 team minutes. This measures territorial volume; unlike the per-100-passes ratio, it does not reward direct teams simply for attempting fewer total passes.", "fa-solid fa-arrow-right-to-bracket"),
        MetricSpec("Chances_Created_per_90", "Chances Created p90", "Sportmonks chances created per 90 team minutes.", "fa-solid fa-wand-magic-sparkles"),
    ),
    "defending": (
        MetricSpec("GA_per_90", "Goals Against per 90", "Goals conceded per 90 team minutes. Lower is better.", "fa-solid fa-shield-halved", ascending=True),
        MetricSpec("xGA_per_90", "xG Against per 90", "Opponent Sportmonks xG per 90 team minutes. Lower is better.", "fa-solid fa-chart-area", ascending=True),
        MetricSpec("Shots_Against_per_90", "Shots Against per 90", "Opponent total shots per 90 team minutes. Lower is better.", "fa-solid fa-crosshairs", ascending=True),
        MetricSpec("Global_PPDA_Proxy", "Global PPDA Proxy", "Opponent passes divided by tackles + interceptions + fouls across the whole pitch. Lower values mean fewer opponent passes before a defensive intervention. It is not true PPDA because action locations are unavailable.", "fa-solid fa-person-falling-burst", ascending=True),
    ),
    "set_pieces": (
        MetricSpec("Set_Piece_xG_per_90", "Set-Piece xG per 90", "Sportmonks expected goals from set plays per 90 team minutes.", "fa-solid fa-chess-rook"),
        MetricSpec("Corner_xG_per_Corner", "Corner xG per Corner", "Expected goals generated from corners divided by corners taken. Higher values indicate better average corner quality.", "fa-solid fa-flag", format_spec="{:.3f}"),
        MetricSpec("Set_Piece_xGA_per_90", "Set-Piece xGA per 90", "Opponent expected goals from set plays per 90 team minutes. Lower is better.", "fa-solid fa-shield", ascending=True),
        MetricSpec("Set_Piece_xG_Difference_per_90", "Set-Piece xG Difference p90", "Set-piece xG for minus set-piece xG against, per 90. Positive values indicate a net set-piece advantage.", "fa-solid fa-scale-balanced"),
    ),
}


PLAYER_METRIC_GROUPS: dict[str, tuple[MetricSpec, ...]] = {
    "attacking": (
        MetricSpec("Gls_per_90", "Goals per 90", "Goals scored per 90 minutes played.", "fa-solid fa-futbol"),
        MetricSpec("xG_per_90", "Expected Goals per 90", "Sportmonks expected goals (xG) per 90 minutes played.", "fa-solid fa-chart-line"),
        MetricSpec("G_minus_xG_per_90", "Goals minus xG p90", "Goals minus Sportmonks xG, divided by 90s played. Positive values indicate finishing above xG.", "fa-solid fa-arrow-trend-up"),
        MetricSpec("SoT/90", "Shots on Target p90", "Shots on target per 90 minutes played.", "fa-solid fa-bullseye"),
    ),
    "creation": (
        MetricSpec("Ast_per_90", "Assists per 90", "Assists per 90 minutes played.", "fa-solid fa-hands-helping"),
        MetricSpec("Chances_Created_per_90", "Chances Created p90", "Sportmonks chances created per 90 minutes played.", "fa-solid fa-wand-magic-sparkles"),
        MetricSpec("Passes_F3_per_90", "Passes into Final Third p90", "Passes into the attacking third per 90 minutes played.", "fa-solid fa-arrow-right-to-bracket"),
        MetricSpec("Pass_Completion_Perc", "Pass Completion", "Accurate passes divided by attempted passes.", "fa-solid fa-check-double", unit="%", format_spec="{:,.1f}"),
    ),
    "defending": (
        MetricSpec("TklW_per_90", "Tackles Won p90", "Successful tackles per 90 minutes played.", "fa-solid fa-shield-halved"),
        MetricSpec("Int_per_90", "Interceptions p90", "Interceptions per 90 minutes played.", "fa-solid fa-route"),
        MetricSpec("Clr_per_90", "Clearances p90", "Clearances per 90 minutes played.", "fa-solid fa-broom"),
        MetricSpec("Aerial_Duels_perc", "Aerial Duels Won", "Percentage of aerial duels won.", "fa-solid fa-plane-up", unit="%", format_spec="{:,.1f}"),
    ),
    "goalkeeping": (
        MetricSpec("Save%", "Save Percentage", "Saves divided by shots on target faced.", "fa-solid fa-mitten", unit="%", format_spec="{:,.1f}"),
        MetricSpec("xGoT_minus_GA_per_90", "xGoT Faced minus GA p90", "Opponent xG on target allocated to the goalkeeper, minus goals conceded, per 90. Positive is better; the allocation is a minutes-based proxy when multiple goalkeepers appear.", "fa-solid fa-chart-line"),
        MetricSpec("Saves_per_90", "Saves per 90", "Goalkeeper saves per 90 minutes played.", "fa-solid fa-hands"),
        MetricSpec("GA_per_90", "Goals Against per 90", "Goals conceded by the goalkeeper per 90 minutes played. Lower is better.", "fa-solid fa-shield", ascending=True),
    ),
}


TEAM_QUADRANTS = {
    "tab-attacking": {
        "title": "Attacking Threat",
        "x_metric": "Sh_per_90",
        "y_metric": "xG_per_Shot",
        "x_label": "Shots per 90",
        "y_label": "Shot Quality (xG per Shot)",
        "quadrant_labels": {
            "top_right": "Complete Threat",
            "bottom_right": "High-Volume Shooting",
            "bottom_left": "Limited Threat",
            "top_left": "Selective & Dangerous",
        },
        "quadrant_guide": {
            "top_right": "Many shots and high average shot quality: the most complete attacking profile.",
            "bottom_right": "Many shots, but from lower-quality positions; volume compensates for shot selection.",
            "bottom_left": "Below the median for both shot volume and quality.",
            "top_left": "Fewer attempts, but chances are dangerous when they arrive.",
        },
    },
    "tab-possession": {
        "title": "Possession & Territory",
        "x_metric": "Poss",
        "y_metric": "FinalThird_per_90",
        "x_label": "Possession %",
        "y_label": "Final Third Passes per 90",
        "quadrant_labels": {
            "top_right": "Control & Territory",
            "bottom_right": "Sterile Control",
            "bottom_left": "Low Control & Territory",
            "top_left": "Direct Territory",
        },
        "quadrant_guide": {
            "top_right": "Combines above-median possession with a high volume of passes into the final third.",
            "bottom_right": "Keeps the ball, but produces a below-median volume of final-third entries.",
            "bottom_left": "Below the median for both possession and final-third passing volume.",
            "top_left": "Reaches the final third frequently despite lower possession: a more direct territorial style.",
        },
    },
    "tab-defending": {
        "title": "Defensive Suppression",
        "x_metric": "Shots_Against_per_90",
        "y_metric": "xGA_per_Shot_Against",
        "x_label": "Shots Against per 90 (lower is better)",
        "y_label": "xGA per Shot Against (lower is better)",
        "invert_x": True,
        "invert_y": True,
        "quadrant_labels": {
            "top_right": "Elite Suppression",
            "bottom_right": "Few but Dangerous",
            "bottom_left": "Exposed",
            "top_left": "Volume, Low Quality",
        },
        "quadrant_guide": {
            "top_right": "Concedes few shots and keeps their average quality low: the strongest defensive profile.",
            "bottom_right": "Limits shot volume, but the chances conceded tend to be dangerous.",
            "bottom_left": "Concedes both frequent shots and high-quality chances.",
            "top_left": "Allows more attempts, but usually from lower-quality positions.",
        },
    },
    "tab-set-pieces": {
        "title": "Set-Piece Impact",
        "x_metric": "Set_Piece_xG_per_90",
        "y_metric": "Set_Piece_xGA_per_90",
        "x_label": "Set-Piece xG per 90",
        "y_label": "Set-Piece xGA per 90 (lower is better)",
        "invert_y": True,
        "quadrant_labels": {
            "top_right": "Set-Piece Edge",
            "bottom_right": "High Impact, High Risk",
            "bottom_left": "Set-Piece Weakness",
            "top_left": "Defensive Specialist",
        },
        "quadrant_guide": {
            "top_right": "Creates above-median set-piece danger while conceding below-median set-piece xG.",
            "bottom_right": "Strong attacking output, but also vulnerable when defending set plays.",
            "bottom_left": "Limited attacking production and above-median set-piece xG conceded.",
            "top_left": "Defends set plays well, but creates below-median attacking danger from them.",
        },
    },
}


PLAYER_QUADRANTS = {
    "tab-attacking": {
        "title": "Scoring Threat",
        "x_metric": "xG_per_90",
        "y_metric": "Gls_per_90",
        "x_label": "Expected Goals p90",
        "y_label": "Goals p90",
        "quadrant_labels": ["Elite Scorer", "High Threat / Low Return", "Low Threat", "Clinical Finisher"],
    },
    "tab-possession": {
        "title": "Creation & Ball Carrying",
        "x_metric": "Successful_Dribbles_per_90",
        "y_metric": "Chances_Created_per_90",
        "x_label": "Successful Dribbles p90",
        "y_label": "Chances Created p90",
        "quadrant_labels": ["Creator & Carrier", "Primary Creator", "Low Involvement", "Primary Carrier"],
    },
    "tab-defending": {
        "title": "Defensive Activity",
        "x_metric": "TklW_per_90",
        "y_metric": "Int_per_90",
        "x_label": "Tackles Won p90",
        "y_label": "Interceptions p90",
        "quadrant_labels": ["Complete Ball Winner", "Reader", "Low Activity", "Aggressive Tackler"],
    },
    "tab-goalkeeping": {
        "title": "Goalkeeping Performance",
        "x_metric": "Save%",
        "y_metric": "xGoT_minus_GA_per_90",
        "x_label": "Save Percentage",
        "y_label": "xGoT Faced minus GA p90",
        "quadrant_labels": ["Elite Shot-Stopper", "High Save Rate", "Under-performing", "Prevents Difficult Goals"],
    },
}


TEAM_RADAR_GROUPS = {
    "Attacking": {
        "Goals p90": "Gls_per_90",
        "xG p90": "xG_per_90",
        "Conversion %": "Goal_Conversion_Perc",
        "xG / Shot": "xG_per_Shot",
    },
    "Possession & Territory": {
        "Possession %": "Poss",
        "Pass Completion %": "Pass_Completion_Perc",
        "Final Third Passes p90": "FinalThird_per_90",
        "Dribble Success %": "TakeOn_Success_Perc",
    },
    "Defending": {
        "GA p90": "GA_per_90",
        "xGA p90": "xGA_per_90",
        "Shots Against p90": "Shots_Against_per_90",
        "Defensive Actions p90": "Defensive_Actions_per_90",
    },
}


PLAYER_RADAR_GROUPS = {
    "outfield": {
        "⚔️ Attacking": {
            "Goals p90": "Gls_per_90",
            "xG p90": "xG_per_90",
            "G-xG p90": "G_minus_xG_per_90",
            "SoT p90": "SoT/90",
        },
        "⚽ Creation": {
            "Assists p90": "Ast_per_90",
            "Chances Created p90": "Chances_Created_per_90",
            "Final Third Passes p90": "Passes_F3_per_90",
            "Successful Dribbles p90": "Successful_Dribbles_per_90",
        },
        "🛡️ Defending": {
            "Tackles Won p90": "TklW_per_90",
            "Interceptions p90": "Int_per_90",
            "Clearances p90": "Clr_per_90",
            "Aerials Won %": "Aerial_Duels_perc",
        },
    },
    "goalkeeping": {
        "🧤 Goalkeeping": {
            "Save %": "Save%",
            "xGoT-GA p90": "xGoT_minus_GA_per_90",
            "Saves p90": "Saves_per_90",
            "Goals Against p90": "GA_per_90",
        }
    },
}


LOWER_IS_BETTER = {
    "GA_per_90", "xGA_per_90", "Shots_Against_per_90",
    "xGA_per_Shot_Against", "Set_Piece_xGA_per_90", "Global_PPDA_Proxy",
}


def _all_specs() -> Iterable[MetricSpec]:
    for groups in (TEAM_METRIC_GROUPS, PLAYER_METRIC_GROUPS):
        for specs in groups.values():
            yield from specs


SPORTMONKS_TOOLTIPS = {spec.column: spec.tooltip for spec in _all_specs()}
SPORTMONKS_TOOLTIPS.update({
    "Successful_Dribbles_per_90": "Successful dribbles per 90 minutes played.",
    "Dribble_Success_Perc": "Successful dribbles divided by dribble attempts.",
    "Key_Passes_per_90": "Passes that directly lead to a shot, per 90 minutes played.",
})


def card_definitions(specs: Iterable[MetricSpec], column_key: str) -> list[dict[str, Any]]:
    return [spec.card_kwargs(column_key) for spec in specs]


def uses_sportmonks(data: Any) -> bool:
    """Return whether a Series/DataFrame-like object contains Sportmonks rows."""
    source = data.get("Data_Source") if hasattr(data, "get") else None
    if hasattr(source, "eq"):
        return bool(source.eq("Sportmonks").any())
    return source == "Sportmonks"
