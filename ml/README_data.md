# Dataset Notes (DVC)

## Where the data comes from

UCI Machine Learning Repository — Bank Marketing Data Set (`bank-additional-full.csv`)

- Link: <https://archive.ics.uci.edu/dataset/222/bank+marketing>
- 41,188 rows x 21 columns (20 features + target `y`), the `yes` class is ~11.3%
- License: public UCI dataset (CC BY 4.0)
- Local path: `data/` — tracked by DVC, git only keeps the `.dvc` pointer files
- md5: `f6cb2c1256ffe2836b36df321f46e92c`

## Versions

| Version | File | md5 | What it is |
| --- | --- | --- | --- |
| v1 | `data/bank-additional-full.csv` | `f6cb2c12...` | raw download, nothing touched. `duration` column is dropped at training time (target leakage) |
| v2 | `data/bank-additional-v2.csv` | `e8a1add9...` | cleaned up version, see below. 41,188 -> 41,016 rows |

## What changed v1 -> v2

Three small cleanups, done by [ml/make_v2_data.py](../ml/make_v2_data.py):

1. merged education levels: `basic.4y` / `basic.9y` / `basic.6y` -> `basic` (they were too fragmented)
2. dropped rows where `campaign > 20` (calling someone 20+ times without success = spam-level, kept as outliers in v1)
3. dropped exact duplicate rows

Nothing else touched. `duration` still there in the raw file, the pipeline drops it anyway.

## Version <-> code mapping

- **v1 + commit `a967252`** -> the first `logreg_baseline` / `hgb_improved` pair (metrics in [README_experiments.md](README_experiments.md)), this is the model currently in prod
- **v2 + commit `b44417d`** (the training-time HEAD) -> the second run pair with tag `dataset_version=v2`, trained with `--no-save` so it did NOT overwrite the prod pkl

## DVC usage

- remote: course-provided Azure Blob, already configured and **actually pushed** (2 files: v1 + v2):

```bash
dvc remote list        # myazure azure://dvc (default)
dvc push               # upload, "2 files pushed"
dvc pull               # restore data on a fresh clone
```

- auth is SAS-token based: account `wut2026` lives in `.dvc/config` (tracked by git),
  the token itself sits in `.dvc/config.local` (git-ignored, as DVC expects). Token valid until 2026-12-12,
  ask the course staff for a fresh one when it expires. Do NOT commit the token or this guide's PDF anywhere.
- raw CSVs are git-ignored (`.gitignore` has `data/*.csv`), only the `.dvc` pointers get committed. That way git history stays small and you can always answer "which exact dataset version produced this model?"

---

# 数据集说明（DVC）

## 来源

UCI Machine Learning Repository —— Bank Marketing Data Set（`bank-additional-full.csv`）

- 链接：<https://archive.ics.uci.edu/dataset/222/bank+marketing>
- 41,188 行 × 21 列（20 特征 + 目标列 `y`）；正类（yes）占比 11.3%
- 许可：UCI 公开数据集（CC BY 4.0）
- 本地路径：`data/`，DVC 跟踪，git 只存 `.dvc` 指针文件
- md5：`f6cb2c1256ffe2836b36df321f46e92c`

## 版本

| 版本 | 文件 | md5 | 说明 |
| --- | --- | --- | --- |
| v1 | `data/bank-additional-full.csv` | `f6cb2c12...` | 原始下载，啥都没动。`duration` 列训练时剔除（目标泄漏） |
| v2 | `data/bank-additional-v2.csv` | `e8a1add9...` | 清洗过的版本，见下。41,188 -> 41,016 行 |

## v1 → v2 改了啥

三个小清洗，脚本在 [ml/make_v2_data.py](../ml/make_v2_data.py)：

1. 教育程度归类：`basic.4y` / `basic.9y` / `basic.6y` 合并成 `basic`（太碎了没意义）
2. 删掉 `campaign > 20` 的行（打了 20 多次电话还没成的属于骚扰级外呼，v1 里一直留着）
3. 删掉完全重复的行

其他没动。`duration` 还在原始文件里，反正管线会剔掉。

## 版本 ↔ 代码对应

- **v1 + commit `a967252`** → 第一组 `logreg_baseline` / `hgb_improved`（指标见 [README_experiments.md](README_experiments.md)），就是现在生产里跑的模型
- **v2 + commit `b44417d`**（训练时的工作区 HEAD）→ 第二组 run，tag `dataset_version=v2`，用了 `--no-save` 所以没有覆盖生产 pkl

## DVC 用法

- 远端：课程提供的 Azure Blob，已配置好而且**真的推上去了**（2 个文件：v1 + v2）：

```bash
dvc remote list        # myazure azure://dvc (default)
dvc push               # 上传，输出 "2 files pushed"
dvc pull               # 新 clone 上恢复数据
```

- 认证走 SAS token：账号 `wut2026` 在 `.dvc/config`（git 跟踪），token 本体在 `.dvc/config.local`
  （被 git 忽略，DVC 默认就这样设计）。token 有效期到 2026-12-12，过期找课程老师要新的。
  别把 token 和这份指南的 PDF 提交到任何仓库。
- 原始 CSV 被 git 忽略（`.gitignore` 里是 `data/*.csv`），只提交 `.dvc` 指针。这样 git 历史不会爆，而且随时能回答“哪个数据版本训出了哪个模型”
