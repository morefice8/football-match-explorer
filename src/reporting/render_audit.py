"""Per-request accounting of actual PDF figure rendering."""


def artifact_key(artifact):
    return (
        artifact.section_id,
        artifact.id,
        artifact.team_name,
        artifact.variant,
    )


def prepare_render_audit(catalog, manifest):
    policies = {
        (section.id, figure.id): (
            section.required
            if figure.required is None
            else figure.required
        )
        for section in manifest.sections
        for figure in section.figures
    }
    return {
        artifact_key(artifact): {
            "section_id": artifact.section_id,
            "figure_id": artifact.id,
            "team_name": artifact.team_name,
            "variant": artifact.variant,
            "selection": getattr(artifact, "selection", None),
            "data_status": artifact.source_section_status,
            "render_status": (
                "skipped"
                if getattr(
                    artifact.status,
                    "value",
                    artifact.status,
                )
                == "skipped"
                else "error"
            ),
            "renderer_id": artifact.renderer_id,
            "required": policies.get(
                (artifact.section_id, artifact.id),
                True,
            ),
            "error_type": "ArtifactNotRendered",
            "error_message": (
                "Artifact was not included in PDF composition."
            ),
        }
        for artifact in catalog.figures
    }


def summarize_render_audit(records):
    rows = list(records)
    return {
        "figures_expected": len(rows),
        "figures_generated": sum(
            row["render_status"] == "generated"
            for row in rows
        ),
        "figures_empty": sum(
            row["render_status"] == "empty"
            for row in rows
        ),
        "figures_skipped": sum(
            row["render_status"] == "skipped"
            for row in rows
        ),
        "figures_failed": sum(
            row["render_status"] == "error"
            for row in rows
        ),
        "required_figures_failed": sum(
            row["render_status"] == "error"
            and row["required"]
            for row in rows
        ),
    }


class MatchAnalysisPackBytes(bytes):
    """Bytes API plus render summary and REPORT-22 phase timings."""

    def __new__(
        cls,
        payload,
        summary,
        timings=None,
    ):
        result = super().__new__(cls, payload)
        result.render_summary = dict(summary)
        result.timings = dict(timings or {})
        return result
