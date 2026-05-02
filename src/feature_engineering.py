from __future__ import annotations

import pandas as pd


def add_mood_labels(tracks: pd.DataFrame) -> pd.DataFrame:
    """Assign sad, neutral, or happy labels from Spotify valence values."""
    labeled_tracks = tracks.copy()
    labeled_tracks["mood_label"] = pd.cut(
        labeled_tracks["valence"],
        bins=[-float("inf"), 0.3, 0.6, float("inf")],
        labels=["sad", "neutral", "happy"],
        right=False,
    )
    return labeled_tracks


def add_emotional_intensity(listening_history: pd.DataFrame) -> pd.DataFrame:
    """Calculate duration-weighted low-valence intensity for each listening event."""
    enriched_history = listening_history.copy()
    duration_minutes = enriched_history["listening_duration_ms"] / 60_000
    enriched_history["emotional_intensity"] = ((1 - enriched_history["valence"]) * duration_minutes).astype(
        "float32"
    )
    return enriched_history


def aggregate_user_daily_history(listening_history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate listening events by user and day for trend analysis."""
    aggregated = (
        listening_history.groupby(["user_id", "listening_date"], as_index=False, observed=True)
        .agg(
            avg_valence=("valence", "mean"),
            avg_energy=("energy", "mean"),
            avg_tempo=("tempo", "mean"),
            total_duration_ms=("listening_duration_ms", "sum"),
            emotional_intensity=("emotional_intensity", "sum"),
            track_count=("track_id", "count"),
        )
        .sort_values(["user_id", "listening_date"])
    )
    aggregated["avg_valence"] = aggregated["avg_valence"].round(3)
    aggregated["avg_energy"] = aggregated["avg_energy"].round(3)
    aggregated["avg_tempo"] = aggregated["avg_tempo"].round(2)
    aggregated["total_duration_minutes"] = (aggregated["total_duration_ms"] / 60_000).round(2)
    aggregated["mood_trend"] = (
        aggregated.groupby("user_id", observed=True)["avg_valence"].diff().fillna(0).round(3)
    )
    aggregated["rolling_mean_valence_3d"] = (
        aggregated.groupby("user_id", observed=True)["avg_valence"]
        .transform(lambda series: series.rolling(window=3, min_periods=1).mean())
        .round(3)
    )
    aggregated["valence_std_3d"] = (
        aggregated.groupby("user_id", observed=True)["avg_valence"]
        .transform(lambda series: series.rolling(window=3, min_periods=2).std())
        .fillna(0)
        .round(3)
    )
    aggregated["duration_delta"] = (
        aggregated.groupby("user_id", observed=True)["total_duration_minutes"]
        .diff()
        .fillna(0)
        .round(2)
    )
    aggregated["stable_low"] = (
        (aggregated["rolling_mean_valence_3d"] < 0.4) & (aggregated["valence_std_3d"] < 0.1)
    ).astype("int8")
    aggregated = add_trend_classification(aggregated)
    aggregated = add_risk_scores(aggregated)
    return aggregated


def add_trend_classification(
    daily_history: pd.DataFrame,
    stable_threshold: float = 0.02,
) -> pd.DataFrame:
    """Classify short-term movement in daily valence."""
    results = daily_history.copy()
    results["trend_class"] = "stable"
    results.loc[results["mood_trend"] > stable_threshold, "trend_class"] = "improving"
    results.loc[results["mood_trend"] < -stable_threshold, "trend_class"] = "worsening"
    return results


def add_risk_scores(daily_history: pd.DataFrame) -> pd.DataFrame:
    """Create a normalized risk score from low valence, duration, and trend movement."""
    results = daily_history.copy()
    duration_score = _min_max_normalize(results["total_duration_minutes"])
    trend_score = _min_max_normalize(results["mood_trend"].abs())
    low_valence_score = 1 - results["avg_valence"]

    results["risk_score"] = ((low_valence_score * duration_score * 0.5) + (trend_score * 0.5)).round(4)
    results["risk_rank"] = results["risk_score"].rank(method="dense", ascending=False).astype("int32")
    return results


def _min_max_normalize(series: pd.Series) -> pd.Series:
    value_range = series.max() - series.min()
    if value_range == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / value_range
