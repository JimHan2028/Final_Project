# 当前生产模型

| 项 | 值 |
| --- | --- |
| run / 模型 | `hgb_improved`（HistGradientBoostingClassifier） |
| 路径 | `ml/registry/model.pkl`（不入 git，见 .gitignore） |
| 数据版本 | v1（`data/bank-additional-full.csv`，md5 `f6cb2c…`） |
| 训练 commit | `a967252abb802f434b7c755f068f90d60e46735f` |
| 指标 | accuracy 0.9025 / f1(yes) 0.3837 / **roc_auc 0.8141** |
| 择优依据 | ROC-AUC 最高，类别不平衡下最稳健（详见 [../README_experiments.md](../README_experiments.md)） |

**更替流程**：新实验 ROC-AUC 超过当前值 → 用新 pkl 覆盖 `ml/registry/model.pkl`
→ 更新本文件与 PROVENANCE.md → 重新部署。
