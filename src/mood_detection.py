from __future__ import annotations

import pandas as pd


def detect_prolonged_low_mood(
    daily_history: pd.DataFrame,
    valence_threshold: float = 0.4,
    consecutive_days: int = 3,
) -> pd.DataFrame:
    """Flag users with avg_valence below threshold for a configured number of days."""
    results = daily_history.copy().sort_values(["user_id", "listening_date"])
    results["is_low_valence_day"] = results["avg_valence"] < valence_threshold
    results["low_valence_flag"] = results["is_low_valence_day"].astype("int8")

    reset_group = (~results["is_low_valence_day"]).groupby(results["user_id"], observed=True).cumsum()
    results["low_mood_streak"] = (
        results["is_low_valence_day"]
        .groupby([results["user_id"], reset_group], observed=True)
        .cumsum()
        .astype("int16")
    )
    results["consecutive_low_streak"] = results["low_mood_streak"]
    results["mood_status"] = results["low_mood_streak"].ge(consecutive_days).map(
        {True: "Prolonged Low Mood", False: "Normal"}
    )
    return results


def classify_final_user_state(daily_history: pd.DataFrame) -> pd.DataFrame:
    """Classify each user's latest emotional state as Critical, Recovering, or Stable."""
    latest_rows = (
        daily_history.sort_values(["user_id", "listening_date"]).groupby("user_id", observed=True).tail(1)
    )

    classified = latest_rows[
        [
            "user_id",
            "listening_date",
            "avg_valence",
            "mood_trend",
            "rolling_mean_valence_3d",
            "trend_class",
            "risk_score",
            "risk_rank",
            "mood_status",
            "emotional_intensity",
        ]
    ].copy()
    classified["final_state"] = classified.apply(_classify_state, axis=1)
    return classified.reset_index(drop=True)

def _classify_state(row: pd.Series) -> str:
    is_low_and_worsening = row["avg_valence"] < 0.4 and row["mood_trend"] < 0
    if is_low_and_worsening:
        return "Critical"
    if row["mood_trend"] > 0:
        return "Recovering"
    return "Stable"
