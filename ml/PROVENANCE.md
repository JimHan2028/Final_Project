# PROVENANCE

> One question this file answers: **which commit, on which data, produced which model?**

## Data

| item | value |
| --- | --- |
| files | `data/bank-additional-full.csv` (v1) and `data/bank-additional-v2.csv` (v2), both DVC-tracked |
| source | UCI Bank Marketing: <https://archive.ics.uci.edu/dataset/222/bank+marketing> |
| md5 | v1 `f6cb2c1256ffe2836b36df321f46e92c` / v2 `e8a1add9000aa97898f84ceddfbe32ac` |
| shape | 41,188 x 21 (v1) / 41,016 x 21 (v2); training drops `duration` and target `y` |

## Models

| item | value |
| --- | --- |
| prod model | `hgb_improved` (HistGradientBoostingClassifier), **trained on v1** |
| artifact | `ml/registry/model.pkl` (preprocessing + classifier serialized as one Pipeline) |
| training commit | `a967252abb802f434b7c755f068f90d60e46735f` |
| training env | Python 3.12.10 / scikit-learn 1.9.1 / numpy 2.5.4 / pandas 3.0.6 / scipy 1.18.1 / joblib 1.6.0 |

The training env matches the [model section] of [requirements.txt](../requirements.txt) exactly, so the service side won't fail to deserialize because of version drift.

The v2 experiment pair (tags `dataset_version=v2`, training-time HEAD `b44417d`) did not go to prod — reasoning in [README_experiments.md](README_experiments.md).

## How to reproduce

```bash
dvc pull                        # course Azure remote (myazure), data comes back instantly
pip install -r requirements.txt
python ml/train.py              # rebuilds the v1 run pair + model.pkl
python ml/train.py --data data/bank-additional-v2.csv --dataset-version v2 --no-save   # v2 pair
mlflow ui                       # browse run history
```

To keep both artifacts side by side:

```bash
python ml/train.py --data data/bank-additional-full.csv --dataset-version v1 \
  --model-path ml/registry/model_v1.pkl
python ml/train.py --data data/bank-additional-v2.csv --dataset-version v2 \
  --model-path ml/registry/model_v2.pkl

MODEL_PATH=ml/registry/model_v2.pkl uvicorn app.main:app --port 8000
```

The `.pkl` files are git-ignored: generate them locally or track them with DVC.

---

# 溯源（PROVENANCE）

> 回答一个问题：**哪个 commit，用哪份数据，训出了哪个模型？**

## 数据

| 项 | 值 |
| --- | --- |
| 文件 | `data/bank-additional-full.csv`（v1）和 `data/bank-additional-v2.csv`（v2），都在 DVC 里 |
| 来源 | UCI Bank Marketing：<https://archive.ics.uci.edu/dataset/222/bank+marketing> |
| md5 | v1 `f6cb2c1256ffe2836b36df321f46e92c` / v2 `e8a1add9000aa97898f84ceddfbe32ac` |
| 规模 | 41,188 行 × 21 列（v1）/ 41,016 行 × 21 列（v2）；训练剔除 `duration` 与目标 `y` |

## 模型

| 项 | 值 |
| --- | --- |
| 生产模型 | `hgb_improved`（HistGradientBoostingClassifier），**v1 数据训的** |
| 产物 | `ml/registry/model.pkl`（预处理 + 分类器整体序列化的 Pipeline） |
| 训练 commit | `a967252abb802f434b7c755f068f90d60e46735f` |
| 训练环境 | Python 3.12.10 / scikit-learn 1.9.1 / numpy 2.5.4 / pandas 3.0.6 / scipy 1.18.1 / joblib 1.6.0 |

训练环境版本与 [requirements.txt](../requirements.txt) 的 [模型段] 完全一致，
保证服务端反序列化不会因版本漂移失败。

v2 那组实验（tag `dataset_version=v2`，训练时 HEAD `b44417d`）没上生产，原因见
[README_experiments.md](README_experiments.md)。

## 复现步骤

```bash
dvc pull                        # 课程 Azure 远端（myazure），数据直接拉回来
pip install -r requirements.txt
python ml/train.py              # 重建 v1 两个 run + model.pkl
python ml/train.py --data data/bank-additional-v2.csv --dataset-version v2 --no-save   # v2 那组
mlflow ui                       # 查看实验历史
```

如果要把 v1/v2 两个产物并排保留：

```bash
python ml/train.py --data data/bank-additional-full.csv --dataset-version v1 \
  --model-path ml/registry/model_v1.pkl
python ml/train.py --data data/bank-additional-v2.csv --dataset-version v2 \
  --model-path ml/registry/model_v2.pkl

MODEL_PATH=ml/registry/model_v2.pkl uvicorn app.main:app --port 8000
```

`.pkl` 文件被 git 忽略：本地生成即可，或用 DVC 管理。
