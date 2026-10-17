# Current prod model

| item | value |
| --- | --- |
| run / model | `hgb_improved` (HistGradientBoostingClassifier), **trained on data v1** |
| path | `ml/registry/model.pkl` (not in git, see .gitignore) |
| data version | v1 (`data/bank-additional-full.csv`, md5 `f6cb2c...`) |
| training commit | `a967252abb802f434b7c755f068f90d60e46735f` |
| metrics | accuracy 0.9025 / f1(yes) 0.3837 / **roc_auc 0.8141** |
| why this one | highest ROC-AUC, most robust under class imbalance (details in [../README_experiments.md](../README_experiments.md)) |

The v2-data rerun (roc_auc 0.8156) is only a hair better but its f1 is notably worse,
so we kept v1 in prod. Swap procedure when a future dataset actually wins:

new experiment roc_auc clearly beats current -> overwrite `ml/registry/model.pkl` with the new pkl
-> update this file and PROVENANCE.md -> redeploy.

---

# 当前生产模型

| 项 | 值 |
| --- | --- |
| run / 模型 | `hgb_improved`（HistGradientBoostingClassifier），**v1 数据训练** |
| 路径 | `ml/registry/model.pkl`（不入 git，见 .gitignore） |
| 数据版本 | v1（`data/bank-additional-full.csv`，md5 `f6cb2c…`） |
| 训练 commit | `a967252abb802f434b7c755f068f90d60e46735f` |
| 指标 | accuracy 0.9025 / f1(yes) 0.3837 / **roc_auc 0.8141** |
| 择优依据 | ROC-AUC 最高，类别不平衡下最稳健（详见 [../README_experiments.md](../README_experiments.md)） |

v2 数据重训那组（roc_auc 0.8156）AUC 只高了一丝但 f1 掉得多，所以生产还是留在 v1。
等以后哪个数据版本真正赢了再换，流程是：

新实验 roc_auc 明显超过当前值 → 用新 pkl 覆盖 `ml/registry/model.pkl`
→ 更新本文件与 PROVENANCE.md → 重新部署。
