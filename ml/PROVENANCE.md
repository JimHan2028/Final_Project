# 溯源（PROVENANCE）

> 回答一个问题：**哪个 commit，用哪份数据，训出了哪个模型？**

## 数据

| 项 | 值 |
| --- | --- |
| 文件 | `data/bank-additional-full.csv`（DVC 跟踪，版本 v1） |
| 来源 | UCI Bank Marketing：<https://archive.ics.uci.edu/dataset/222/bank+marketing> |
| md5 | `f6cb2c1256ffe2836b36df321f46e92c` |
| 规模 | 41,188 行 × 21 列（训练剔除 `duration` 与目标 `y`） |

## 模型

| 项 | 值 |
| --- | --- |
| 生产候选 | `hgb_improved`（HistGradientBoostingClassifier） |
| 产物 | `ml/registry/model.pkl`（预处理 + 分类器整体序列化的 Pipeline） |
| 训练 commit | `a967252abb802f434b7c755f068f90d60e46735f` |
| 训练环境 | Python 3.12.10 / scikit-learn 1.9.1 / numpy 2.5.4 / pandas 3.0.6 / scipy 1.18.1 / joblib 1.6.0 |

训练环境版本与 [requirements.txt](../requirements.txt) 的 [模型段] 完全一致，
保证服务端反序列化不会因版本漂移失败。

## 复现步骤

```bash
dvc pull                        # 远端配置后；或手动放置 data/bank-additional-full.csv
pip install -r requirements.txt
python ml/train.py              # 重建两个 run + model.pkl
mlflow ui                       # 查看实验历史
```
