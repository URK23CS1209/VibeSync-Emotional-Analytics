from __future__ import annotations

import pandas as pd


def map_valence_to_recommendation_stage(valence: float) -> str:
    """Map current mood valence to a gradual recommendation stage."""
    if valence < 0.3:
        return "reflective"
    if valence < 0.5:
        return "neutral"
    if valence < 0.7:
        return "light positive"
    return "energetic"


def add_recommendation_stages(daily_history: pd.DataFrame) -> pd.DataFrame:
    """Attach transition stages that support gradual mood-aware recommendations."""
    staged_history = daily_history.copy()
    staged_history["recommendation_stage"] = staged_history["avg_valence"].apply(
        map_valence_to_recommendation_stage
    )
    return staged_history


def build_transition_path(current_valence: float, target_valence: float = 0.72) -> list[str]:
    """Return a stage path from the current mood range toward a brighter target range."""
    stage_order = ["reflective", "neutral", "light positive", "energetic"]
    current_stage = map_valence_to_recommendation_stage(current_valence)
    target_stage = map_valence_to_recommendation_stage(target_valence)

    start_index = stage_order.index(current_stage)
    target_index = stage_order.index(target_stage)
    if start_index <= target_index:
        return stage_order[start_index : target_index + 1]
    return stage_order[target_index : start_index + 1][::-1]


def simulate_recommendation_intervention(
    daily_history: pd.DataFrame,
    low_mood_threshold: float = 0.4,
    daily_lift: float = 0.05,
    max_lift: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate how gradual transitions could improve future valence trajectories."""
    simulated_users = []
    summaries = []

    for user_id, user_history in daily_history.sort_values(["user_id", "listening_date"]).groupby(
        "user_id", observed=True
    ):
        user_simulation, user_summary = _simulate_user_intervention(
            user_id=str(user_id),
            user_history=user_history.copy(),
            low_mood_threshold=low_mood_threshold,
            daily_lift=daily_lift,
            max_lift=max_lift,
        )
        simulated_users.append(user_simulation)
        summaries.append(user_summary)

    return pd.concat(simulated_users, ignore_index=True), pd.DataFrame(summaries)


def _simulate_user_intervention(
    user_id: str,
    user_history: pd.DataFrame,
    low_mood_threshold: float,
    daily_lift: float,
    max_lift: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    boost = 0.0
    adjusted_values = []

    for row in user_history.itertuples(index=False):
        needs_intervention = row.avg_valence < low_mood_threshold or row.mood_status == "Prolonged Low Mood"
        boost = min(max_lift, boost + daily_lift) if needs_intervention else max(0.0, boost - daily_lift / 2)
        adjusted_values.append(min(1.0, round(row.avg_valence + boost, 3)))

    user_history["intervention_valence"] = adjusted_values
    user_history["intervention_stage"] = user_history["intervention_valence"].apply(
        map_valence_to_recommendation_stage
    )
    user_history["valence_improvement"] = (user_history["intervention_valence"] - user_history["avg_valence"]).round(3)

    low_period = user_history["avg_valence"] < low_mood_threshold
    first_low_index = low_period.idxmax() if low_period.any() else None
    recovery_time = None
    if first_low_index is not None:
        after_first_low = user_history.loc[first_low_index:]
        recovered = after_first_low[after_first_low["intervention_valence"] >= low_mood_threshold]
        if not recovered.empty:
            recovery_time = int(recovered.index[0] - first_low_index)

    summary = {
        "user_id": user_id,
        "before_latest_valence": float(user_history["avg_valence"].iloc[-1]),
        "after_latest_valence": float(user_history["intervention_valence"].iloc[-1]),
        "total_valence_improvement": round(
            float(user_history["intervention_valence"].iloc[-1] - user_history["avg_valence"].iloc[-1]),
            3,
        ),
        "avg_valence_improvement": round(float(user_history["valence_improvement"].mean()), 3),
        "recovery_time_days": recovery_time,
    }
    return user_history, summary
