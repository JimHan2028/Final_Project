"""训练脚本 —— 银行定期存款营销预测（UCI Bank Marketing）。

用法（在项目根目录执行）：
    python ml/train.py                                  # v1 数据
    python ml/train.py --data data/bank-additional-v2.csv --dataset-version v2
    python ml/train.py --no-save                        # 只出实验结果，不覆盖 model.pkl

每次运行会：
  1. 读取数据集（DVC 跟踪的 data/，默认 v1）；
  2. 分层划分训练 / 测试集；
  3. 逐个训练候选模型，向 MLflow（本地 sqlite + mlruns/）记录
     git commit、数据集版本、超参数、指标与混淆矩阵产物；
  4. 以 ROC-AUC 择优，把最优 Pipeline 保存到 ml/registry/model.pkl
     （加 --no-save 则只记录不落盘，用于比数据版本）。
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data_pipeline import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_pipeline,
    load_dataframe,
)

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
SQLITE_DB_PATH = PROJECT_ROOT / "mlflow.db"
REGISTRY_DIR = PROJECT_ROOT / "ml" / "registry"
MODEL_PATH = REGISTRY_DIR / "model.pkl"

EXPERIMENT_NAME = "bank-marketing"
POSITIVE_LABEL = "yes"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def configure_mlflow() -> None:
    """本地自托管 MLflow：元数据存 sqlite（mlflow.db），产物存 mlruns/。"""
    mlflow.set_tracking_uri(f"sqlite:///{SQLITE_DB_PATH.as_posix()}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        MLRUNS_DIR.mkdir(exist_ok=True)
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME, artifact_location=MLRUNS_DIR.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)


def get_git_commit() -> str:
    """当前 git commit SHA；不在 git 仓库内时返回 unknown。"""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def get_file_md5(path: Path) -> str:
    """数据集指纹：把「哪份数据训出了哪个模型」钉死。"""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_confusion_matrix(cm, labels: list[str], out_path: Path) -> None:
    """把混淆矩阵画成 PNG，作为 MLflow artifact。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3.5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    threshold = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("confusion matrix (test)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def candidate_models() -> dict[str, Callable[[], Any]]:
    """两组实验：baseline（逻辑回归）与 improved（梯度提升）。"""
    return {
        "logreg_baseline": lambda: LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "hgb_improved": lambda: HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_candidate(
    name: str,
    factory: Callable[[], Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    data_md5: str,
    commit: str,
    dataset_version: str,
) -> dict[str, Any]:
    """训练一个候选模型并把全部信息写入 MLflow，返回评估结果。"""
    pipeline = build_pipeline(factory())

    with mlflow.start_run(run_name=name) as run:
        pipeline.fit(X_train, y_train)

        classes = list(pipeline.classes_)
        yes_index = classes.index(POSITIVE_LABEL)
        proba_yes = pipeline.predict_proba(X_test)[:, yes_index]
        predicted = pipeline.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "f1": float(
                f1_score(y_test, predicted, pos_label=POSITIVE_LABEL, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, proba_yes)),
        }

        classifier_params = pipeline.named_steps["classifier"].get_params()
        mlflow.log_params(
            {
                "model_name": name,
                "dataset_version": dataset_version,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                **{
                    f"clf_{key}": str(value) for key, value in classifier_params.items()
                },
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "git_commit": commit,
                "dataset_version": dataset_version,
                "data_md5": data_md5,
            }
        )

        cm = confusion_matrix(y_test, predicted, labels=classes)
        with tempfile.TemporaryDirectory() as tmp:
            cm_path = Path(tmp) / f"confusion_matrix_{name}.png"
            plot_confusion_matrix(cm, classes, cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

        print(
            f"[{name}] accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}"
        )
        return {
            "name": name,
            "pipeline": pipeline,
            "metrics": metrics,
            "run_id": run.info.run_id,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练银行营销预测模型并记录 MLflow")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="训练数据 CSV 路径（默认 data/bank-additional-full.csv，即 v1）",
    )
    parser.add_argument(
        "--dataset-version",
        default="v1",
        help="数据版本号，跟 --data 配套（v2 就写 v2），进 MLflow 的 tag/param",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="只记录实验不覆盖 ml/registry/model.pkl",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"数据文件不存在：{args.data}（请先 dvc pull 或手动放置）")

    frame = load_dataframe(args.data)
    features = frame[list(FEATURE_COLUMNS)]
    target = frame[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    data_md5 = get_file_md5(args.data)
    commit = get_git_commit()

    configure_mlflow()

    results = [
        evaluate_candidate(
            name,
            factory,
            X_train,
            X_test,
            y_train,
            y_test,
            data_md5,
            commit,
            args.dataset_version,
        )
        for name, factory in candidate_models().items()
    ]

    # 择优标准：ROC-AUC（正类占比仅 ~11%，AUC 对类别不平衡最稳健）
    best = max(results, key=lambda item: item["metrics"]["roc_auc"])

    if not args.no_save:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best["pipeline"], MODEL_PATH)

        # 回到最优 run，把最终 pkl 也作为 artifact 归档
        with mlflow.start_run(run_id=best["run_id"]):
            mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
            mlflow.set_tag("prod_candidate", "true")

    print("\n=== 训练完成 ===")
    print(f"数据：{args.data} (md5={data_md5[:12]}..., 版本 {args.dataset_version})")
    print(f"git commit：{commit}")
    for item in results:
        marker = "  <- 生产候选" if item is best else ""
        print(f"  {item['name']}: {item['metrics']}{marker}")
    if args.no_save:
        print("本次 --no-save：registry 里的 model.pkl 没动")
    else:
        print(f"最优模型已保存：{MODEL_PATH}")
    print(f"MLflow：sqlite:///{SQLITE_DB_PATH.as_posix()}（产物在 {MLRUNS_DIR}）")


if __name__ == "__main__":
    main()
