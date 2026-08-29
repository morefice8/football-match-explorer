"""Registry of validated Plotly renderers used by static Match Reports."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Callable


@dataclass(frozen=True)
class RendererRef:
    id: str
    module: str
    function: str

    @property
    def dotted_path(self) -> str:
        return f"{self.module}.{self.function}"


VALIDATED_RENDERERS: dict[str, RendererRef] = {
    "formation-state": RendererRef(
        "formation-state",
        "src.visualization.formation_plotly",
        "plot_formation_timeline_state",
    ),
    "mean-positions": RendererRef(
        "mean-positions",
        "src.visualization.formation_plotly",
        "plot_mean_positions_profile_plotly",
    ),
    "pass-network": RendererRef(
        "pass-network",
        "src.visualization.pass_plotly",
        "plot_pass_network_profile_plotly",
    ),
    "progressive-passes": RendererRef(
        "progressive-passes",
        "src.visualization.pass_plotly",
        "plot_progressive_pass_summary_plotly",
    ),
    "final-third-entries": RendererRef(
        "final-third-entries",
        "src.visualization.pass_plotly",
        "plot_final_third_summary_plotly",
    ),
    "pass-locations": RendererRef(
        "pass-locations",
        "src.visualization.pass_plotly",
        "plot_pass_locations_plotly",
    ),
    "cross-heatmap": RendererRef(
        "cross-heatmap",
        "src.visualization.cross_plots",
        "plot_cross_heatmap",
    ),
    "build-up-sequence": RendererRef(
        "build-up-sequence",
        "src.visualization.buildup_plotly",
        "plot_buildup_sequence_plotly",
    ),
    "defensive-shape": RendererRef(
        "defensive-shape",
        "src.visualization.defensive_transitions_plotly",
        "plot_defensive_shape_profile",
    ),
    "ppda-timeline": RendererRef(
        "ppda-timeline",
        "src.visualization.defensive_transitions_plotly",
        "plot_ppda_timeline",
    ),
    "sequence-explorer": RendererRef(
        "sequence-explorer",
        "src.visualization.sequence_explorer",
        "plot_sequence_explorer",
    ),
    "restart-map": RendererRef(
        "restart-map",
        "src.visualization.restart_map",
        "plot_restart_map",
    ),
    "player-pass-map": RendererRef(
        "player-pass-map",
        "src.visualization.player_plots",
        "plot_player_pass_map_plotly",
    ),
    "player-reception-map": RendererRef(
        "player-reception-map",
        "src.visualization.threat_reception_map",
        "plot_reception_profile",
    ),
    "player-defensive-map": RendererRef(
        "player-defensive-map",
        "src.visualization.defender_action_map",
        "plot_defensive_action_profile",
    ),
}


class RendererRegistry:
    """Resolve validated renderers without importing the interactive app."""

    def __init__(
        self,
        refs: dict[str, RendererRef] | None = None,
    ) -> None:
        self._refs = dict(refs or VALIDATED_RENDERERS)

    def ref(self, renderer_id: str) -> RendererRef:
        try:
            return self._refs[renderer_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown report renderer: {renderer_id}"
            ) from exc

    def resolve(self, renderer_id: str) -> Callable:
        ref = self.ref(renderer_id)
        module = importlib.import_module(ref.module)
        renderer = getattr(module, ref.function)
        if not callable(renderer):
            raise TypeError(
                f"Registered renderer is not callable: {ref.dotted_path}"
            )
        return renderer

    def validate(self) -> tuple[str, ...]:
        resolved = []
        for renderer_id in self._refs:
            self.resolve(renderer_id)
            resolved.append(renderer_id)
        return tuple(resolved)
