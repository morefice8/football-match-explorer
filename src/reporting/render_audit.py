"""Per-request accounting of actual PDF figure rendering."""

def artifact_key(artifact):
    return (artifact.section_id, artifact.id, artifact.team_name, artifact.variant)


def prepare_render_audit(catalog, manifest):
    policies = {
        (section.id, figure.id): (
            section.required if figure.required is None else figure.required
        )
        for section in manifest.sections for figure in section.figures
    }
    return {
        artifact_key(a): {
            "section_id": a.section_id,
            "figure_id": a.id,
            "team_name": a.team_name,
            "variant": a.variant,
            "selection": getattr(a, "selection", None),
            "data_status": a.source_section_status,
            "render_status": (
                "skipped" if getattr(a.status, "value", a.status) == "skipped" else "error"
            ),
            "renderer_id": a.renderer_id,
            "required": policies.get((a.section_id, a.id), True),
            "error_type": "ArtifactNotRendered",
            "error_message": "Artifact was not included in PDF composition.",
        }
        for a in catalog.figures
    }


def summarize_render_audit(records):
    rows = list(records)
    return {
        "figures_expected": len(rows),
        "figures_generated": sum(r["render_status"] == "generated" for r in rows),
        "figures_empty": sum(r["render_status"] == "empty" for r in rows),
        "figures_skipped": sum(r["render_status"] == "skipped" for r in rows),
        "figures_failed": sum(r["render_status"] == "error" for r in rows),
        "required_figures_failed": sum(
            r["render_status"] == "error" and r["required"] for r in rows
        ),
    }


class MatchAnalysisPackBytes(bytes):
    """Keep the bytes API while carrying the same summary sent in the ZIP."""

    def __new__(cls, payload, summary):
        result = super().__new__(cls, payload)
        result.render_summary = dict(summary)
        return result
