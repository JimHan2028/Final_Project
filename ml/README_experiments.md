# Experiments (MLflow)

- experiment: `bank-marketing`
- backend: local self-hosted — metadata in `sqlite:///mlflow.db`, artifacts in `mlruns/` (run `mlflow ui` to browse)
- evaluation: stratified 8:2 split (`random_state=42`), positive class `yes`
- metric we optimize: **ROC-AUC** — the positive class is only ~11.3%, plain accuracy gets dominated by the negative class.
  In business terms we score-rank customers and call the top ones first, and AUC measures exactly that ranking quality. It's also robust to class imbalance.

## The runs

Key runs: 2 models x 2 dataset versions (v1 = raw, v2 = cleaned, see [README_data.md](README_data.md)).
(The MLflow UI also keeps a few leftover reruns from early development — ignore those.)

| run | data | model | accuracy | f1(yes) | roc_auc |
| --- | --- | --- | --- | --- | --- |
| logreg_baseline | v1 | LogisticRegression(class_weight=balanced, max_iter=1000) | 0.8350 | 0.4689 | 0.8009 |
| hgb_improved | v1 | HistGradientBoostingClassifier | 0.9025 | 0.3837 | **0.8141** |
| logreg_baseline | v2 | same as above | 0.8329 | 0.4621 | 0.8021 |
| hgb_improved | v2 | same as above | 0.9021 | 0.3662 | **0.8156** |

v2 runs were trained with `--no-save` (experiment only, prod pkl untouched).

## Which run goes to prod, and why

**hgb_improved on v1** (roc_auc 0.8141) — it's the model currently in `ml/registry/model.pkl`.

Honest reasoning: the v2 hgb actually scores a hair *higher* on AUC (0.8156), but:

- the gain (+0.0015) is within noise on a 8k-row test set, and
- its f1 drops a lot (0.3837 -> 0.3662)

so swapping the prod model for that isn't worth the churn. If we later do a bigger data overhaul (v3) and the AUC gain is real, we swap then — the whole point of the registry + PROVENANCE setup is that this swap is a 5-minute job.

## What gets logged per run

| type | content |
| --- | --- |
| tag | `git_commit`, `dataset_version`, `data_md5` |
| param | `model_name`, `dataset_version`, `train_rows/test_rows`, `clf_*` (all classifier hyperparams) |
| metric | `accuracy`, `f1`, `roc_auc` |
| artifact | confusion matrix PNG (`plots/`), final `model.pkl` (`model/`) |

---

# 实验记录（MLflow）

- 实验：`bank-marketing`
- 后端：本地自托管 —— 元数据 `sqlite:///mlflow.db`，产物 `mlruns/`（`mlflow ui` 可视化）
- 评估协议：分层 8:2 划分（`random_state=42`），正类 `yes`
- 优化指标：**ROC-AUC** —— 正类仅 11.3%，accuracy 会被负类主导；
  业务上按分数排序优先触达高分客户，AUC 恰好衡量排序能力，且对类别不平衡稳健。

## run 一览

正式的 run：2 个模型 × 2 个数据版本（v1 原始 / v2 清洗版，见 [README_data.md](README_data.md)）。
（MLflow UI 里还有几个开发期重跑留下的旧 run，忽略即可。）

| run | 数据 | 模型 | accuracy | f1(yes) | roc_auc |
| --- | --- | --- | --- | --- | --- |
| logreg_baseline | v1 | LogisticRegression(class_weight=balanced, max_iter=1000) | 0.8350 | 0.4689 | 0.8009 |
| hgb_improved | v1 | HistGradientBoostingClassifier | 0.9025 | 0.3837 | **0.8141** |
| logreg_baseline | v2 | 同上 | 0.8329 | 0.4621 | 0.8021 |
| hgb_improved | v2 | 同上 | 0.9021 | 0.3662 | **0.8156** |

v2 那两组是 `--no-save` 跑的（纯实验，没动生产 pkl）。

## 哪个 run 上生产，为什么

**hgb_improved（v1 数据）**（roc_auc 0.8141），就是现在 `ml/registry/model.pkl` 里的那个。

说实话：v2 的 hgb AUC 确实还高了一丢丢（0.8156），但是

- 这点提升（+0.0015）在 8k 行测试集上就是噪声水平
- f1 掉得挺多（0.3837 -> 0.3662）

不值得为这点差异折腾生产模型。等以后 v3 做个真正的大清洗、AUC 涨得明显了再换——registry + PROVENANCE 这套东西存在的意义就是让换模型变成 5 分钟的事。

## 每个 run 记录了什么

| 类型 | 内容 |
| --- | --- |
| tag | `git_commit`、`dataset_version`、`data_md5` |
| param | `model_name`、`dataset_version`、`train_rows/test_rows`、`clf_*`（分类器全部超参） |
| metric | `accuracy`、`f1`、`roc_auc` |
| artifact | 混淆矩阵 PNG（`plots/`）、最终 `model.pkl`（`model/`） |
