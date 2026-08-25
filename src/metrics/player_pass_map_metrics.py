from __future__ import annotations

import pandas as pd

CATEGORY_ORDER = (
    "Incomplete",
    "Completed",
    "Progressive",
    "Key Pass",
    "Assist",
)

def _truthy(value):
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)

def _completed(row):
    return str(row.get("outcome", "")).strip().lower() == "successful"

def visual_category(row):
    completed = _completed(row)
    if not completed:
        return "Incomplete"
    if _truthy(row.get("is_assist")):
        return "Assist"
    if _truthy(row.get("is_key_pass")):
        return "Key Pass"
    if _truthy(row.get("is_progressive")):
        return "Progressive"
    return "Completed"

def classify_player_passes(passes):
    if passes is None:
        return pd.DataFrame()
    result = passes.copy()
    if result.empty:
        result["visual_category"] = pd.Series(dtype="object")
        return result
    result["visual_category"] = result.apply(visual_category, axis=1)
    return result

def player_pass_profile(passes):
    passes = passes.copy() if passes is not None else pd.DataFrame()
    total = int(len(passes))
    if total == 0:
        return {
            "volume": 0,
            "completed": 0,
            "completion_pct": 0.0,
            "progressive_completions": 0,
            "chance_creation": 0,
            "key_passes": 0,
            "assists": 0,
        }

    completed_mask = (
        passes["outcome"].fillna("").astype(str).str.strip().str.lower().eq("successful")
        if "outcome" in passes.columns
        else pd.Series(False, index=passes.index)
    )
    progressive_mask = (
        passes.get("is_progressive", pd.Series(False, index=passes.index))
        .fillna(False).astype(bool) & completed_mask
    )
    key_mask = (
        passes.get("is_key_pass", pd.Series(False, index=passes.index))
        .fillna(False).astype(bool) & completed_mask
    )
    assist_mask = (
        passes.get("is_assist", pd.Series(False, index=passes.index))
        .fillna(False).astype(bool) & completed_mask
    )
    completed = int(completed_mask.sum())
    chance_mask = key_mask | assist_mask
    return {
        "volume": total,
        "completed": completed,
        "completion_pct": completed / total * 100.0,
        "progressive_completions": int(progressive_mask.sum()),
        "chance_creation": int(chance_mask.sum()),
        "key_passes": int(key_mask.sum()),
        "assists": int(assist_mask.sum()),
    }
