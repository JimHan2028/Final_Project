# 数据集说明（DVC）

## 来源

UCI Machine Learning Repository —— Bank Marketing Data Set（`bank-additional-full.csv`）

- 链接：<https://archive.ics.uci.edu/dataset/222/bank+marketing>
- 规模：41,188 行 × 21 列（20 特征 + 目标列 `y`）；正类（yes）占比 11.3%
- 许可：UCI 公开数据集（CC BY 4.0）
- 本地路径：`data/bank-additional-full.csv`（DVC 跟踪，git 只存指针）
- md5：`f6cb2c1256ffe2836b36df321f46e92c`

## 版本

| 版本 | 内容 | md5 | 状态 |
| --- | --- | --- | --- |
| v1 | 原始数据直接入库；训练侧剔除 `duration`（目标泄漏） | `f6cb2c1256ffe2836b36df321f46e92c` | 当前 |
| v2（计划） | 类别归并（`basic.4y/9y/6y` → `basic`）、`unknown` 缺失统一处理、按月份聚合特征 | — | 未开始 |

## v1 → v2 变更（待 v2 落地后回填）

- 改动内容 / 动机：
- 对指标的影响：

## 版本 ↔ 代码对应

- **v1 + commit `a967252`** → `logreg_baseline` 与 `hgb_improved` 两个 run
  （指标见 [README_experiments.md](README_experiments.md)）。
- 原始 CSV 存放在 DVC 远端（Azure Blob，待配置），**不提交 GitHub**；
  git 仓库内只有 `.dvc` 指针文件。
