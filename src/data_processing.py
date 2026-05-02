from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


RELEVANT_FEATURES = [
    "track_id",
    "track_name",
    "valence",
    "energy",
    "tempo",
    "duration_ms",
]

NUMERIC_FEATURES = ["valence", "energy", "tempo", "duration_ms"]
FLOAT_FEATURES = ["valence", "energy", "tempo"]


def load_spotify_datasets(csv_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load multiple large Spotify CSV files using only the columns needed downstream."""
    frames = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        frame = pd.read_csv(
            csv_path,
            usecols=lambda column: column in RELEVANT_FEATURES or column.startswith("Unnamed"),
            dtype={
                "track_id": "string",
                "track_name": "string",
                "valence": "float32",
                "energy": "float32",
                "tempo": "float32",
                "duration_ms": "Int32",
            },
            low_memory=False,
        )
        frames.append(remove_unnamed_columns(frame))

    combined = pd.concat(frames, ignore_index=True, copy=False)
    return clean_spotify_tracks(combined)


def remove_unnamed_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop auto-generated index columns commonly produced during CSV export."""
    unnamed_columns = [column for column in dataframe.columns if column.startswith("Unnamed")]
    return dataframe.drop(columns=unnamed_columns, errors="ignore")


def clean_spotify_tracks(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Spotify features and normalize dtypes for scalable processing."""
    tracks = select_relevant_features(dataframe)
    tracks = tracks.dropna(subset=RELEVANT_FEATURES)

    for column in FLOAT_FEATURES:
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce").astype("float32")
    tracks["duration_ms"] = pd.to_numeric(tracks["duration_ms"], errors="coerce").astype("Int32")

    tracks = tracks.dropna(subset=NUMERIC_FEATURES)
    tracks = tracks[(tracks["valence"].between(0, 1)) & (tracks["energy"].between(0, 1))]
    tracks = tracks[tracks["duration_ms"] > 0]
    tracks = tracks.drop_duplicates(subset="track_id", keep="first")
    tracks = optimize_track_memory(tracks)
    return tracks.reset_index(drop=True)


def select_relevant_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep the canonical audio features required by the pipeline."""
    missing_columns = sorted(set(RELEVANT_FEATURES) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return dataframe[RELEVANT_FEATURES].copy()


def optimize_track_memory(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory use with compact numeric dtypes and categorical track metadata."""
    optimized = dataframe.copy()
    for column in FLOAT_FEATURES:
        optimized[column] = optimized[column].astype("float32")
    optimized["duration_ms"] = optimized["duration_ms"].astype("int32")

    # Categories are compact after deduplication and keep repeated track names inexpensive.
    optimized["track_id"] = optimized["track_id"].astype("category")
    optimized["track_name"] = optimized["track_name"].astype("category")
    return optimized


def stratified_valence_sample(
    tracks: pd.DataFrame,
    sample_size: int = 15_000,
    random_seed: int = 42,
    bins: int = 10,
) -> pd.DataFrame:
    """Sample tracks while preserving the original valence distribution."""
    if sample_size >= len(tracks):
        return tracks.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    tracks_with_bins = tracks.assign(valence_bin=pd.qcut(tracks["valence"], q=bins, duplicates="drop"))
    sampled_frames = []
    for _, group in tracks_with_bins.groupby("valence_bin", observed=True):
        group_sample_size = min(len(group), max(1, round(len(group) / len(tracks) * sample_size)))
        sampled_frames.append(group.sample(n=group_sample_size, random_state=random_seed, replace=False))

    sampled = pd.concat(sampled_frames, ignore_index=False).drop(columns="valence_bin")

    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_seed)
    elif len(sampled) < sample_size:
        remaining = tracks.drop(index=sampled.index, errors="ignore")
        top_up = remaining.sample(n=sample_size - len(sampled), random_state=random_seed)
        sampled = pd.concat([sampled, top_up], ignore_index=False)

    return optimize_track_memory(sampled.reset_index(drop=True))


def save_dataset(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    """Persist a dataframe to CSV after creating the destination directory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
