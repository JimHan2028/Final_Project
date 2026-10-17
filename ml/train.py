"""Train the bank-marketing model and log every run to MLflow.

The two runs expected by the assignment live in ml/configs/train.yaml:

baseline    raw dataset v1, which still carries the leaky `duration` column.
            Kept for reference only - it is not deployable.
production  cleaned dataset v2, with the decision threshold selected on a
            validation split instead of defaulting to 0.5.

Usage:
    python ml/train.py --run baseline
    python ml/train.py --run production
    python ml/train.py --run production --tracking-uri sqlite:////abs/path/mlflow.db
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import data_pipeline as dp
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="key under `runs:` in the config file, e.g. baseline / v2_default / production",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default=None)
    return parser.parse_args()


def git_metadata() -> dict[str, str]:
    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            )
        except (OSError, subprocess.CalledProcessError):
            return ""
        return result.stdout.strip()

    sha = os.environ.get("GIT_SHA") or git("rev-parse", "HEAD")
    branch = os.environ.get("GIT_BRANCH") or git("rev-parse", "--abbrev-ref", "HEAD")
    porcelain = git("status", "--porcelain")
    return {
        "git_sha": sha or "unknown",
        "git_branch": branch or "unknown",
        "git_dirty": str(bool(porcelain)).lower() if sha else "unknown",
    }


def dvc_dataset_hash() -> str:
    """The md5 DVC recorded for the dataset, read from the .dvc pointer file."""
    for candidate in (REPO_ROOT / "data.dvc", REPO_ROOT / "data" / "data.dvc"):
        if candidate.exists():
            outs = (yaml.safe_load(candidate.read_text()) or {}).get("outs") or []
            if outs:
                return str(outs[0].get("md5", ""))
    return ""


def build_estimator(
    params: dict, numeric: list[str], categorical: list[str]
) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(**params, random_state=0)),
        ]
    )


def select_threshold(spec: dict, y_true: pd.Series, probabilities: np.ndarray) -> float:
    mode = spec.get("mode", "fixed")
    if mode == "fixed":
        return float(spec.get("value", 0.5))

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if mode == "precision_target":
        target = float(spec.get("precision_target", 0.6))
        admissible = np.where(precision >= target)[0]
        if len(admissible) == 0:
            return 0.5
        best = admissible[np.argmax(recall[admissible])]
        return float(thresholds[min(best, len(thresholds) - 1)])
    if mode == "max_f1":
        f1 = (
            2
            * precision[:-1]
            * recall[:-1]
            / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        )
        return float(thresholds[int(np.argmax(f1))])
    raise ValueError(f"unknown threshold mode {mode!r}")


def recall_at_precision(
    y_true: pd.Series, probabilities: np.ndarray, target: float
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    admissible = recall[precision >= target]
    return float(admissible.max()) if len(admissible) else 0.0


def write_plots(
    directory: Path, y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> tuple[Path, Path]:
    pr_path = directory / "pr_curve.png"
    display = PrecisionRecallDisplay.from_predictions(
        y_true, probabilities, name="logreg"
    )
    display.ax_.scatter(
        recall_score(y_true, probabilities >= threshold),
        precision_score(y_true, probabilities >= threshold),
        marker="o",
        color="crimson",
        label=f"operating point (threshold={threshold:.2f})",
    )
    display.ax_.legend(loc="upper right")
    display.figure_.tight_layout()
    display.figure_.savefig(pr_path, dpi=120)
    plt.close(display.figure_)

    cm_path = directory / "confusion_matrix.png"
    ConfusionMatrixDisplay.from_predictions(
        y_true, probabilities >= threshold, display_labels=["no", "yes"]
    ).figure_.savefig(cm_path, dpi=120)
    plt.close("all")
    return pr_path, cm_path


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.run not in config["runs"]:
        raise SystemExit(
            f"run {args.run!r} is not in {args.config}: expected one of {sorted(config['runs'])}"
        )
    run_config = config["runs"][args.run]
    data_config = config["data"]

    data_path = Path(args.data_path or data_config["path"])
    tracking_uri = (
        args.tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or config["mlflow"]["tracking_uri"]
    )
    experiment = args.experiment or config["mlflow"]["experiment"]
    seed = int(data_config.get("seed", 42))
    test_size = float(data_config.get("test_size", 0.3))
    validation_size = float(data_config.get("validation_size", 0.2))
    version = run_config["data_version"]
    threshold_spec = run_config["threshold"]
    params = dict(run_config["params"])

    df = dp.load_raw(data_path)
    features = dp.build_features(df, version)
    target = dp.extract_target(df)
    numeric, categorical = dp.as_column_groups(features)
    X_train_full, X_test, y_train_full, y_test = dp.split_dataset(
        features, target, test_size=test_size, seed=seed
    )

    threshold = float(threshold_spec.get("value", 0.5))
    if threshold_spec.get("mode", "fixed") == "fixed":
        model = build_estimator(params, numeric, categorical).fit(
            X_train_full, y_train_full
        )
    else:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=validation_size,
            stratify=y_train_full,
            random_state=seed,
        )
        threshold = select_threshold(
            threshold_spec,
            y_val,
            build_estimator(params, numeric, categorical)
            .fit(X_fit, y_fit)
            .predict_proba(X_val)[:, 1],
        )
        model = build_estimator(params, numeric, categorical).fit(
            X_train_full, y_train_full
        )

    probabilities = model.predict_proba(X_test)[:, 1]
    decisions = probabilities >= threshold
    confusion = confusion_matrix(y_test, decisions)

    metrics = {
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, decisions)),
        "precision": float(precision_score(y_test, decisions)),
        "recall": float(recall_score(y_test, decisions)),
        "f1": float(f1_score(y_test, decisions)),
        "threshold": threshold,
        "predicted_positive": int(decisions.sum()),
        "missed_positive": int(confusion[1, 0]),
        # Threshold-free reference points: the best recall achievable while holding
        # precision at or above the given level, whatever the threshold.
        "recall_at_precision_0.5": recall_at_precision(y_test, probabilities, 0.5),
        "recall_at_precision_0.6": recall_at_precision(y_test, probabilities, 0.6),
        "recall_at_precision_0.7": recall_at_precision(y_test, probabilities, 0.7),
        "positive_rate_test": float(y_test.mean()),
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_config["run_name"]):
        mlflow.log_params(
            {
                **params,
                "model": run_config["model"],
                "data_version": version,
                "test_size": test_size,
                "validation_size": validation_size,
                "seed": seed,
                "n_features": len(numeric) + len(categorical),
                "threshold_mode": threshold_spec.get("mode", "fixed"),
            }
        )
        mlflow.set_tags(
            {
                **git_metadata(),
                "dataset_sha256": dp.fingerprint(data_path),
                "dataset_path": str(data_path),
                "dvc_dataset_hash": dvc_dataset_hash(),
                "deployable": str(bool(run_config.get("deployable", False))).lower(),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            (tmp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            (tmp_dir / "run_config.json").write_text(
                json.dumps(run_config, indent=2) + "\n"
            )
            pr_path, cm_path = write_plots(tmp_dir, y_test, probabilities, threshold)
            mlflow.log_artifact(str(tmp_dir / "metrics.json"))
            mlflow.log_artifact(str(tmp_dir / "run_config.json"))
            mlflow.log_artifact(str(pr_path), artifact_path="plots")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="code")
        mlflow.log_artifact(str(Path(dp.__file__).resolve()), artifact_path="code")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
