# 实验记录（MLflow）

- 实验：`bank-marketing`
- 后端：本地自托管 —— 元数据 `sqlite:///mlflow.db`，产物 `mlruns/`（`mlflow ui` 可视化）
- 评估协议：分层 8:2 划分（`random_state=42`），正类 `yes`
- 优化指标：**ROC-AUC** —— 正类仅 11.3%，accuracy 会被负类主导；
  业务上按分数排序优先触达高分客户，AUC 恰好衡量排序能力，且对类别不平衡稳健。

## 两次实验

| run | 模型 | accuracy | f1(yes) | roc_auc | 结论 |
| --- | --- | --- | --- | --- | --- |
| logreg_baseline | LogisticRegression(class_weight=balanced, max_iter=1000) | 0.8350 | 0.4689 | 0.8009 | baseline |
| hgb_improved | HistGradientBoostingClassifier | 0.9025 | 0.3837 | **0.8141** | 生产候选 |

## 哪个 run 可以上生产，为什么

**hgb_improved**：ROC-AUC 最高（0.8141 vs 0.8009），accuracy 也明显更高（0.9025 vs 0.8350）。
注意其 f1 低于 baseline（正类召回偏低）—— 如果业务目标转为"最大化触达召回"，
应回退 baseline 或下调判定阈值；当前以排序质量为先，选 hgb_improved。

## 每个 run 记录了什么

| 类型 | 内容 |
| --- | --- |
| tag | `git_commit`、`dataset_version`、`data_md5` |
| param | `model_name`、`dataset_version`、`train_rows/test_rows`、`clf_*`（分类器全部超参） |
| metric | `accuracy`、`f1`、`roc_auc` |
| artifact | 混淆矩阵 PNG（`plots/`）、最终 `model.pkl`（`model/`） |
