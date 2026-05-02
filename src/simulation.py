from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def simulate_user_listening_history(
    tracks: pd.DataFrame,
    user_count: int = 75,
    days: int = 21,
    min_tracks_per_day: int = 8,
    max_tracks_per_day: int = 18,
    low_mood_user_id: str = "user_001",
    low_mood_start_day: int = 7,
    low_mood_days: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Simulate user interactions while preserving real Spotify audio features."""
    rng = np.random.default_rng(random_seed)
    tracks = tracks.reset_index(drop=True)
    low_valence_tracks = tracks[tracks["valence"] < 0.3].reset_index(drop=True)
    if low_valence_tracks.empty:
        raise ValueError("Simulation requires real tracks with valence < 0.3.")

    user_days = _build_user_day_frame(user_count, days)
    user_days["track_count"] = rng.integers(
        min_tracks_per_day,
        max_tracks_per_day + 1,
        size=len(user_days),
    )

    expanded_days = user_days.loc[user_days.index.repeat(user_days["track_count"])].reset_index(drop=True)
    low_mood_mask = (
        (expanded_days["user_id"] == low_mood_user_id)
        & (expanded_days["day_offset"].between(low_mood_start_day, low_mood_start_day + low_mood_days - 1))
    )

    normal_count = int((~low_mood_mask).sum())
    low_mood_count = int(low_mood_mask.sum())

    normal_track_indices = rng.integers(0, len(tracks), size=normal_count)
    low_mood_track_indices = rng.integers(0, len(low_valence_tracks), size=low_mood_count)

    normal_events = expanded_days.loc[~low_mood_mask, ["user_id", "listening_date"]].reset_index(drop=True)
    low_mood_events = expanded_days.loc[low_mood_mask, ["user_id", "listening_date"]].reset_index(drop=True)

    normal_events = _attach_sampled_tracks(normal_events, tracks.iloc[normal_track_indices].reset_index(drop=True))
    low_mood_events = _attach_sampled_tracks(
        low_mood_events,
        low_valence_tracks.iloc[low_mood_track_indices].reset_index(drop=True),
    )

    events = pd.concat([normal_events, low_mood_events], ignore_index=True)
    events["listen_ratio"] = rng.uniform(0.35, 1.0, size=len(events)).astype("float32")
    events["listening_duration_ms"] = (events["duration_ms"].astype("int32") * events["listen_ratio"]).astype("int32")
    events = events.drop(columns="listen_ratio").rename(columns={"duration_ms": "track_duration_ms"})

    events["user_id"] = events["user_id"].astype("category")
    events["track_id"] = events["track_id"].astype("category")
    events["track_name"] = events["track_name"].astype("category")
    return events.sort_values(["user_id", "listening_date"]).reset_index(drop=True)


def _build_user_day_frame(user_count: int, days: int) -> pd.DataFrame:
    start_date = date.today() - timedelta(days=days - 1)
    users = [f"user_{index:03d}" for index in range(1, user_count + 1)]

    user_days = pd.MultiIndex.from_product(
        [users, range(days)],
        names=["user_id", "day_offset"],
    ).to_frame(index=False)
    user_days["listening_date"] = user_days["day_offset"].map(
        lambda day_offset: (start_date + timedelta(days=int(day_offset))).isoformat()
    )
    return user_days


def _attach_sampled_tracks(events: pd.DataFrame, sampled_tracks: pd.DataFrame) -> pd.DataFrame:
    selected_columns = ["track_id", "track_name", "valence", "energy", "tempo", "duration_ms"]
    return pd.concat([events.reset_index(drop=True), sampled_tracks[selected_columns].reset_index(drop=True)], axis=1)
