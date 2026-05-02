from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_spotify_datasets, save_dataset, stratified_valence_sample
from src.feature_engineering import (
    add_emotional_intensity,
    add_mood_labels,
    aggregate_user_daily_history,
)
from src.mood_detection import classify_final_user_state, detect_prolonged_low_mood
from src.simulation import simulate_user_listening_history
from src.transition_engine import (
    add_recommendation_stages,
    build_transition_path,
    simulate_recommendation_intervention,
)


DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DATASETS = [
    Path(r"C:\Users\sharo\Downloads\SpotifyAudioFeaturesNov2018.csv"),
    Path(r"C:\Users\sharo\Downloads\SpotifyAudioFeaturesApril2019.csv"),
]

PROCESSED_DATASET_PATH = DATA_DIR / "processed_dataset.csv"
USER_BEHAVIOR_PATH = DATA_DIR / "user_behavior.csv"
AGGREGATED_RESULTS_PATH = DATA_DIR / "aggregated_results.csv"
USER_STATES_PATH = DATA_DIR / "user_final_states.csv"
INTERVENTION_RESULTS_PATH = DATA_DIR / "intervention_results.csv"
PLOT_PATH = DATA_DIR / "mood_trend.png"
INTERVENTION_PLOT_PATH = DATA_DIR / "intervention_comparison.png"

SAMPLE_SIZE = 15_000
USER_COUNT = 75
SIMULATION_DAYS = 21


def run_pipeline() -> None:
    """Run VibeSync with real Spotify audio features and simulated user behavior."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load both real Spotify datasets using a narrow schema and compact dtypes.
    tracks = load_spotify_datasets(SOURCE_DATASETS)

    # Step 2: Preserve valence distribution while reducing runtime and memory pressure.
    tracks = stratified_valence_sample(tracks, sample_size=SAMPLE_SIZE)
    tracks = add_mood_labels(tracks)
    save_dataset(tracks, PROCESSED_DATASET_PATH)

    # Step 3: Simulate only user behavior. Audio features remain real Spotify values.
    user_behavior = simulate_user_listening_history(
        tracks,
        user_count=USER_COUNT,
        days=SIMULATION_DAYS,
    )
    user_behavior = add_emotional_intensity(user_behavior)
    save_dataset(user_behavior, USER_BEHAVIOR_PATH)

    # Step 4: Aggregate emotional trajectory signals by user and day.
    aggregated_results = aggregate_user_daily_history(user_behavior)
    aggregated_results = detect_prolonged_low_mood(aggregated_results)
    aggregated_results = add_recommendation_stages(aggregated_results)
    save_dataset(aggregated_results, AGGREGATED_RESULTS_PATH)

    # Step 5: Classify final user state and generate staged recommendation paths.
    user_states = classify_final_user_state(aggregated_results)
    user_states["transition_path"] = user_states["avg_valence"].apply(
        lambda valence: " -> ".join(build_transition_path(valence))
    )
    save_dataset(user_states, USER_STATES_PATH)

    intervention_results, intervention_summary = simulate_recommendation_intervention(aggregated_results)
    save_dataset(intervention_results, INTERVENTION_RESULTS_PATH)

    plot_mood_trends(aggregated_results, PLOT_PATH)
    plot_intervention_comparison(intervention_results, INTERVENTION_PLOT_PATH)
    train_optional_mood_classifier(aggregated_results)
    print_results(tracks, user_behavior, aggregated_results, user_states, intervention_summary)


def plot_mood_trends(aggregated_results, output_path: Path, max_users: int = 12) -> None:
    """Plot average valence over time for critical users plus a readable sample."""
    critical_users = set(
        aggregated_results.loc[
            aggregated_results["mood_status"].eq("Prolonged Low Mood"),
            "user_id",
        ].astype(str)
    )
    sampled_users = list(aggregated_results["user_id"].astype(str).drop_duplicates().head(max_users))
    plotted_users = list(dict.fromkeys([*critical_users, *sampled_users]))[:max_users]

    plot_data = aggregated_results[aggregated_results["user_id"].astype(str).isin(plotted_users)]

    plt.figure(figsize=(12, 6))
    for user_id, user_history in plot_data.groupby("user_id", observed=True):
        plt.plot(
            user_history["listening_date"],
            user_history["avg_valence"],
            marker="o",
            linewidth=2,
            label=str(user_id),
        )

    plt.axhline(0.4, color="crimson", linestyle="--", linewidth=1.5, label="low mood threshold")
    plt.title("VibeSync Mood Trend by User")
    plt.xlabel("Listening Date")
    plt.ylabel("Average Valence")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_intervention_comparison(intervention_results, output_path: Path) -> None:
    """Plot before/after valence for the highest-risk user and highlight low-mood periods."""
    top_user_id = (
        intervention_results.sort_values("risk_score", ascending=False)["user_id"].astype(str).iloc[0]
    )
    user_history = intervention_results[intervention_results["user_id"].astype(str).eq(top_user_id)]

    plt.figure(figsize=(12, 6))
    plt.plot(
        user_history["listening_date"],
        user_history["avg_valence"],
        marker="o",
        linewidth=2,
        label="before intervention",
    )
    plt.plot(
        user_history["listening_date"],
        user_history["intervention_valence"],
        marker="o",
        linewidth=2,
        label="after staged recommendations",
    )

    low_mood_periods = user_history["avg_valence"] < 0.4
    plt.scatter(
        user_history.loc[low_mood_periods, "listening_date"],
        user_history.loc[low_mood_periods, "avg_valence"],
        color="crimson",
        s=90,
        alpha=0.75,
        label="low-mood period",
        zorder=5,
    )

    transition_rows = user_history[user_history["recommendation_stage"] != user_history["intervention_stage"]]
    for _, row in transition_rows.head(4).iterrows():
        plt.annotate(
            row["intervention_stage"],
            xy=(row["listening_date"], row["intervention_valence"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.axhline(0.4, color="crimson", linestyle="--", linewidth=1.5, label="low mood threshold")
    plt.title(f"Intervention Simulation: {top_user_id}")
    plt.xlabel("Listening Date")
    plt.ylabel("Average Valence")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_optional_mood_classifier(aggregated_results) -> None:
    """Train and evaluate no-leakage ML separately from the rule-assisted hybrid system."""
    # Leakage guard: these ML features do not encode the 3-day target rule directly.
    feature_columns = [
        "avg_valence",
        "total_duration_minutes",
        "mood_trend",
        "rolling_mean_valence_3d",
        "valence_std_3d",
        "duration_delta",
        "stable_low",
        "avg_energy",
        "avg_tempo",
    ]
    modeling_data = aggregated_results.sort_values("listening_date").reset_index(drop=True)
    features = modeling_data[feature_columns]
    labels = modeling_data["mood_status"]

    if labels.nunique() < 2:
        print("\nOptional ML skipped: only one mood class is present.")
        return
    if labels.value_counts().min() < 2:
        print("\nOptional ML skipped: at least one mood class has fewer than 2 samples.")
        return

    # Time-based split avoids training on future behavior and testing on earlier days.
    train_data, validation_data, test_data = split_by_time(modeling_data)
    if test_data["mood_status"].nunique() < 2:
        print("\nOptional ML skipped: time-based test split has only one class.")
        return

    x_train = train_data[feature_columns]
    y_train = train_data["mood_status"]
    x_validation = validation_data[feature_columns]
    y_validation = validation_data["mood_status"]
    x_test = test_data[feature_columns]
    y_test = test_data["mood_status"]

    if y_train.nunique() < 2 or y_validation.nunique() < 2:
        print("\nOptional ML skipped: time-based train/validation split lacks both classes.")
        return

    baseline_model = build_random_forest(n_estimators=100, max_depth=6, min_samples_leaf=10)
    baseline_x_train, baseline_y_train = oversample_minority_class(x_train, y_train)
    baseline_model.fit(baseline_x_train, baseline_y_train)
    baseline_probabilities = get_minority_probabilities(baseline_model, x_test)
    baseline_predictions = predict_with_raw_probabilities(baseline_probabilities, threshold=0.40)

    tuned_model, selected_model_name, threshold_report = select_best_model_and_threshold(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )
    threshold = select_precision_threshold(threshold_report, minimum_recall=0.75)
    ml_predictions = predict_with_minority_threshold(tuned_model, x_test, threshold)
    hybrid_predictions = apply_hybrid_decision_system(
        test_data,
        ml_predictions,
        threshold_probabilities=get_minority_probabilities(tuned_model, x_test),
        threshold=threshold,
    )

    print("\nOptional ML: RandomForest mood-status classifier")
    print("Evaluation split: time-based train/validation/test")
    print(
        f"Train dates: {train_data['listening_date'].min()} to {train_data['listening_date'].max()} "
        f"({len(train_data):,} rows)"
    )
    print(
        f"Validation dates: {validation_data['listening_date'].min()} to {validation_data['listening_date'].max()} "
        f"({len(validation_data):,} rows)"
    )
    print(
        f"Test dates: {test_data['listening_date'].min()} to {test_data['listening_date'].max()} "
        f"({len(test_data):,} rows)"
    )
    print("\nThreshold tuning on validation split:")
    print(threshold_report.to_string(index=False, formatters={"threshold": "{:.2f}".format}))
    print(f"\nSelected model: {selected_model_name}")
    print(f"\nFinal selected threshold: {threshold:.2f}")
    print("\nBaseline ML-only (no leakage, recall-oriented 0.40 threshold)")
    print(classification_report(y_test, baseline_predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, baseline_predictions, labels=["Normal", "Prolonged Low Mood"]))
    print("\nML-only (no leakage)")
    print(classification_report(y_test, ml_predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, ml_predictions, labels=["Normal", "Prolonged Low Mood"]))
    print("\nHybrid system (rule + ML)")
    print(classification_report(y_test, hybrid_predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, hybrid_predictions, labels=["Normal", "Prolonged Low Mood"]))


def split_by_time(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split earlier days into train/validation and later days into test."""
    unique_dates = sorted(dataframe["listening_date"].unique())
    train_end = max(1, int(len(unique_dates) * 0.65))
    validation_end = max(train_end + 1, int(len(unique_dates) * 0.80))

    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[train_end:validation_end]
    test_dates = unique_dates[validation_end:]

    return (
        dataframe[dataframe["listening_date"].isin(train_dates)].copy(),
        dataframe[dataframe["listening_date"].isin(validation_dates)].copy(),
        dataframe[dataframe["listening_date"].isin(test_dates)].copy(),
    )


def select_best_model_and_threshold(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[object, str, pd.DataFrame]:
    """Compare tuned tree models and keep the highest-precision viable threshold."""
    candidates = [
        (
            "RandomForest_depth6_leaf5_est300",
            build_random_forest(n_estimators=300, max_depth=6, min_samples_leaf=5),
        ),
        (
            "RandomForest_depth8_leaf10_est400",
            build_random_forest(n_estimators=400, max_depth=8, min_samples_leaf=10),
        ),
        (
            "RandomForest_depth12_leaf20_est500",
            build_random_forest(n_estimators=500, max_depth=12, min_samples_leaf=20),
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=8,
                random_state=42,
            ),
        ),
    ]

    best_model = None
    best_name = ""
    best_report = None
    best_score = (-1.0, -1.0, -1.0)

    for model_name, model in candidates:
        model.fit(*oversample_minority_class(x_train, y_train))
        threshold_report = evaluate_thresholds(model, x_validation, y_validation)
        threshold = select_precision_threshold(threshold_report, minimum_recall=0.75)
        selected_row = threshold_report[threshold_report["threshold"].eq(threshold)].iloc[0]
        score = (
            float(selected_row["precision"]),
            float(selected_row["recall"]),
            float(selected_row["f1_score"]),
        )
        if score > best_score:
            best_model = model
            best_name = model_name
            best_report = threshold_report
            best_score = score

    return best_model, best_name, best_report


def build_random_forest(
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
) -> RandomForestClassifier:
    """Create a conservative RandomForest tuned to reduce overfitting."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=8,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )


def oversample_minority_class(features, labels):
    """Balance training data with deterministic random oversampling."""
    training_frame = features.copy()
    training_frame["mood_status"] = labels.to_numpy()
    class_counts = training_frame["mood_status"].value_counts()
    target_count = int(class_counts.max())

    balanced_frames = []
    for label, group in training_frame.groupby("mood_status"):
        balanced_frames.append(
            group.sample(n=target_count, replace=len(group) < target_count, random_state=42)
        )

    balanced = (
        pd.concat(balanced_frames, ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return balanced.drop(columns="mood_status"), balanced["mood_status"]


def evaluate_thresholds(
    model,
    features,
    labels,
    minority_label: str = "Prolonged Low Mood",
) -> pd.DataFrame:
    """Evaluate minority precision, recall, and F1 across candidate thresholds."""
    probabilities = get_minority_probabilities(model, features, minority_label)
    y_true = labels.eq(minority_label).astype(int)

    rows = []
    for threshold in [value / 100 for value in range(40, 81, 5)]:
        y_pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def select_precision_threshold(threshold_report: pd.DataFrame, minimum_recall: float) -> float:
    """Maximize precision among thresholds that keep recall above the business floor."""
    viable = threshold_report[threshold_report["recall"] >= minimum_recall]
    if viable.empty:
        best_recall_row = threshold_report.sort_values(["recall", "precision"], ascending=False).iloc[0]
        return float(best_recall_row["threshold"])

    best_row = viable.sort_values(["precision", "f1_score", "threshold"], ascending=False).iloc[0]
    return float(best_row["threshold"])


def get_minority_probabilities(model, features, minority_label: str = "Prolonged Low Mood"):
    """Return RandomForest probabilities for the minority class."""
    if minority_label not in model.classes_:
        return pd.Series(0.0, index=features.index).to_numpy()

    minority_index = list(model.classes_).index(minority_label)
    return model.predict_proba(features)[:, minority_index]


def predict_with_minority_threshold(model, features, threshold: float, minority_label: str = "Prolonged Low Mood"):
    """Convert probabilities to labels using a tuned minority-class threshold."""
    probabilities = get_minority_probabilities(model, features, minority_label)
    return predict_with_raw_probabilities(probabilities, threshold, minority_label)


def predict_with_raw_probabilities(probabilities, threshold: float, minority_label: str = "Prolonged Low Mood"):
    """Map minority probabilities to class labels at the selected threshold."""
    return pd.Series(probabilities >= threshold).map({True: minority_label, False: "Normal"}).to_numpy()


def apply_hybrid_decision_system(
    features: pd.DataFrame,
    ml_predictions,
    threshold_probabilities,
    threshold: float,
    minority_label: str = "Prolonged Low Mood",
):
    """Use a stricter low-valence streak gate for high-confidence cases, otherwise use ML."""
    strict_low_day = features["avg_valence"].lt(0.35)
    reset_group = (~strict_low_day).groupby(features["user_id"], observed=True).cumsum()
    strict_low_streak = strict_low_day.groupby([features["user_id"], reset_group], observed=True).cumsum()
    rule_positive = strict_low_streak.ge(3)
    borderline = (
        features["avg_valence"].lt(0.45)
        | features["rolling_mean_valence_3d"].lt(0.45)
        | features["stable_low"].eq(1)
    )
    ml_positive = (pd.Series(ml_predictions, index=features.index).eq(minority_label)) & borderline
    confident_ml_positive = ml_positive & (pd.Series(threshold_probabilities, index=features.index) >= threshold)
    final_positive = rule_positive | confident_ml_positive
    return final_positive.map({True: minority_label, False: "Normal"}).to_numpy()


def print_results(tracks, user_behavior, aggregated_results, user_states, intervention_summary) -> None:
    """Print a concise operational summary for the pipeline run."""
    prolonged_low_mood_rows = aggregated_results["mood_status"].eq("Prolonged Low Mood").sum()
    total_users = user_states["user_id"].nunique()
    critical_users = user_states["final_state"].eq("Critical").sum()
    high_risk_users = user_states["risk_score"].ge(user_states["risk_score"].quantile(0.8)).sum()
    average_recovery_trend = user_states.loc[user_states["final_state"].eq("Recovering"), "mood_trend"].mean()
    top_risk_users = user_states.sort_values("risk_score", ascending=False).head(5)

    print("\nVibeSync real-data pipeline completed.")
    print(f"Processed track subset rows: {len(tracks):,}")
    print(f"Simulated listening events: {len(user_behavior):,}")
    print(f"Aggregated user-day rows: {len(aggregated_results):,}")
    print(f"Prolonged low-mood detections: {prolonged_low_mood_rows:,}")
    print("\nBusiness insight summary:")
    print(f"Total users: {total_users:,}")
    print(f"Critical users: {critical_users:,}")
    print(f"High-risk users: {high_risk_users / total_users:.1%}")
    print(f"Average recovery trend: {average_recovery_trend:.3f}")
    print(f"Processed dataset: {PROCESSED_DATASET_PATH}")
    print(f"User behavior: {USER_BEHAVIOR_PATH}")
    print(f"Aggregated results: {AGGREGATED_RESULTS_PATH}")
    print(f"Final user states: {USER_STATES_PATH}")
    print(f"Intervention results: {INTERVENTION_RESULTS_PATH}")
    print(f"Mood trend plot: {PLOT_PATH}")
    print(f"Intervention comparison plot: {INTERVENTION_PLOT_PATH}")
    print("\nTop 5 highest risk users:")
    print(
        top_risk_users[
            ["user_id", "avg_valence", "mood_trend", "risk_score", "final_state", "transition_path"]
        ].to_string(index=False)
    )
    print("\nIntervention impact sample:")
    print(
        intervention_summary.sort_values("avg_valence_improvement", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print("\nUsers requiring attention:")
    print(user_states[user_states["final_state"].isin(["Critical", "Recovering"])].to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
