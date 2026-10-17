# Final Project — Bank Term Deposit Prediction (DevOps / MLOps)

We trained a subscription-prediction model on the UCI Bank Marketing dataset and serve it through FastAPI.
The project covers the full DevOps flow (Git / Docker / GitHub Actions CI/CD) plus the MLOps basics (DVC / MLflow).

## 1. What the app does

- **Task**: given customer profile + campaign contact records + macro indicators, predict whether the customer will subscribe to a term deposit (binary yes/no).
- **Data**: UCI Bank Marketing (41,188 rows x 21 cols); the `duration` column is target leakage, dropped in both training and inference.
- **Model**: sklearn Pipeline (column preprocessing + classifier) serialized whole into `ml/registry/model.pkl`;
  baseline is LogisticRegression, prod candidate is HistGradientBoostingClassifier (picked by ROC-AUC).
- **Service**: FastAPI — `POST /predict` single prediction, `GET /health` health check, interactive docs at `/docs`.

### API contract (frozen on day 1)

| item | agreement |
| --- | --- |
| input | `POST /predict`, 19 feature fields, see [app/schemas.py](app/schemas.py) |
| output | `{"label": "yes\|no", "score": 0~1}` |
| preprocessing | all inside the pkl Pipeline, caller does nothing |
| model path | `ml/registry/model.pkl` |

## 2. Quick start

```bash
pip install -r requirements.txt

# train (logs to MLflow, writes ml/registry/model.pkl)
python ml/train.py

# start the service
uvicorn app.main:app --reload --port 8000

# tests + lint
pytest -v
ruff check .
```

## 3. Repo layout

```
.
├── app/                  # FastAPI service (entrypoint / contract / model loader)
├── ml/
│   ├── train.py          # training script (calls MLflow)
│   ├── data_pipeline.py  # preprocessing / feature engineering (shared by train & serve)
│   ├── make_v2_data.py   # builds dataset v2 out of v1
│   ├── registry/         # current_model.md + model.pkl (pkl not in git)
│   ├── README_data.md    # dataset versions (v1 / v2)
│   ├── README_experiments.md
│   └── PROVENANCE.md
├── data/                 # DVC-tracked CSVs, git only has the .dvc pointers
├── tests/                # pytest: input validation / happy path / edge cases
├── requirements.txt      # single source of truth for deps
├── Dockerfile / .dockerignore
├── .github/workflows/    # ci.yml + cd.yml
├── DEPLOYMENT.md
└── README.md
```

## 4. CI/CD: from commit to prod

**Branch model**: `feature/*` -> `dev` (integration) -> `staging` (pre-prod) -> `main` (prod), PR-only merges.

| trigger | pipeline action |
| --- | --- |
| PR to dev / staging / main | CI: ruff check + full pytest + **docker build smoke test**; any failure blocks the merge |
| merge -> staging | CD(staging): render staging `.env` from GitHub Secrets -> build staging image |
| merge -> main | CD(prod): render prod `.env` from GitHub Secrets -> build prod image |

Typical path: `commit -> push feature/* -> PR to dev (CI gates) -> merge -> PR dev->staging
(CI gates + staging image build) -> PR staging->main (CI gates + prod image build)`.

Configs: [.github/workflows/ci.yml](.github/workflows/ci.yml) and [cd.yml](.github/workflows/cd.yml), deploy details in [DEPLOYMENT.md](DEPLOYMENT.md).

## 5. MLOps

### Data versioning (DVC)

- `data/` is tracked by DVC, git only stores the `.dvc` pointers; remote is the course Azure Blob (`myazure` -> `azure://dvc`, account `wut2026`), both dataset versions are already pushed.
- Two versions live in the repo: **v1** (raw download) and **v2** (education merged / outlier rows dropped / dupes removed, built by [ml/make_v2_data.py](ml/make_v2_data.py)).
- Which commit + which dataset produced which model: see [ml/PROVENANCE.md](ml/PROVENANCE.md) and [ml/README_data.md](ml/README_data.md).

### Experiment tracking (MLflow)

- The training script logs automatically: git commit, dataset version/md5 (tags), all hyperparams (params),
  accuracy / F1 / ROC-AUC (metrics), confusion matrix PNGs and the final model.pkl (artifacts).
- 4 runs so far: 2 models x 2 dataset versions. Locally self-hosted: metadata `sqlite:///mlflow.db`, artifacts in `mlruns/`; just run `mlflow ui`.
- Conclusions and prod-model choice: [ml/README_experiments.md](ml/README_experiments.md).

## 6. Tests

47 pytest cases in three files, one scenario each:

- **input validation** ([tests/test_schema.py](tests/test_schema.py)): missing fields / wrong types / out-of-range / blank categories / extra fields -> 422; boundary values pass;
- **happy path** ([tests/test_predict.py](tests/test_predict.py)): contract shape, valid categories, score range, inference determinism, OpenAPI;
- **edge cases** ([tests/test_edge_cases.py](tests/test_edge_cases.py)): unknown categories don't crash, unicode / extreme numerics, inference failure -> 500, fallback placeholder model.

---

# Final Project —— 银行定期存款营销预测（DevOps / MLOps）

基于 UCI Bank Marketing 数据集训练订阅预测模型，通过 FastAPI 提供推理服务；
覆盖 DevOps（Git / Docker / GitHub Actions CI/CD）与 MLOps（DVC / MLflow）全流程。

## 1. 应用概述

- **任务**：根据客户画像、营销触达记录与宏观经济指标，预测客户是否订阅银行定期存款（二分类 yes/no）。
- **数据**：UCI Bank Marketing（41,188 行 × 21 列）；`duration` 列属于目标泄漏，训练与推理均剔除。
- **模型**：sklearn Pipeline（列预处理 + 分类器）整体序列化为 `ml/registry/model.pkl`；
  baseline 为 LogisticRegression，生产候选为 HistGradientBoostingClassifier（按 ROC-AUC 择优）。
- **服务**：FastAPI —— `POST /predict` 单条预测，`GET /health` 健康检查，交互文档 `/docs`。

### 接口契约（Day1 冻结）

| 项 | 约定 |
| --- | --- |
| 输入 | `POST /predict`，19 个特征字段，见 [app/schemas.py](app/schemas.py) |
| 输出 | `{"label": "yes\|no", "score": 0~1}` |
| 预处理 | 全部封装在 pkl 的 Pipeline 内，调用方零预处理 |
| 模型路径 | `ml/registry/model.pkl` |

## 2. 快速开始

```bash
pip install -r requirements.txt

# 训练（自动记录 MLflow，产出 ml/registry/model.pkl）
python ml/train.py

# 启动服务
uvicorn app.main:app --reload --port 8000

# 测试 + lint
pytest -v
ruff check .
```

## 3. 仓库结构

```
.
├── app/                  # FastAPI 服务（入口 / 契约 / 模型加载）
├── ml/
│   ├── train.py          # 训练脚本（调用 MLflow）
│   ├── data_pipeline.py  # 预处理 / 特征工程（训练与推理共用）
│   ├── make_v2_data.py   # 从 v1 生成 v2 数据集
│   ├── registry/         # current_model.md + model.pkl（不入 git）
│   ├── README_data.md    # 数据集版本说明（v1 / v2）
│   ├── README_experiments.md
│   └── PROVENANCE.md
├── data/                 # DVC 跟踪的 CSV，git 里只有 .dvc 指针
├── tests/                # pytest：输入校验 / 正常路径 / 异常边界
├── requirements.txt      # 全项目唯一依赖事实来源
├── Dockerfile / .dockerignore
├── .github/workflows/    # ci.yml + cd.yml
├── DEPLOYMENT.md
└── README.md
```

## 4. CI/CD：代码从 commit 到 prod 的路径

**分支模型**：`feature/*` → `dev`（集成）→ `staging`（预发）→ `main`（生产），只允许 PR 合并。

| 触发 | 流水线动作 |
| --- | --- |
| 向 dev / staging / main 发 PR | CI：ruff 检查 + pytest 全量测试 + **docker build 冒烟**；任何一步失败禁止合并 |
| merge → staging | CD(staging)：用 staging 环境 GitHub Secrets 生成 staging `.env` → 构建 staging 镜像 |
| merge → main | CD(prod)：用 production 环境 GitHub Secrets 生成生产 `.env` → 构建生产镜像 |

路径示例：`commit → push feature/* → PR 到 dev（CI 把关）→ merge 进 dev →
PR dev→staging（CI 把关 + staging 镜像构建）→ PR staging→main（CI 把关 + prod 镜像构建）`。

配置见 [.github/workflows/ci.yml](.github/workflows/ci.yml) 和 [cd.yml](.github/workflows/cd.yml)，部署细节见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 5. MLOps

### 数据版本（DVC）

- `data/` 由 DVC 跟踪，git 只存 `.dvc` 指针；远端是课程提供的 Azure Blob（`myazure` → `azure://dvc`，账号 `wut2026`），两个数据版本都已 push 上去。
- 仓库里有两个版本：**v1**（原始下载）和 **v2**（教育程度合并 / 外呼异常行剔除 / 去重，由 [ml/make_v2_data.py](ml/make_v2_data.py) 生成）。
- 哪个 commit + 哪份数据训出了哪个模型：见 [ml/PROVENANCE.md](ml/PROVENANCE.md) 和 [ml/README_data.md](ml/README_data.md)。

### 实验跟踪（MLflow）

- 训练脚本自动记录：git commit、数据集版本/md5（tag）、全部超参（param）、
  accuracy / F1 / ROC-AUC（metric）、混淆矩阵 PNG 与最终 model.pkl（artifact）。
- 目前 4 个 run：2 个模型 × 2 个数据版本。本地自托管：元数据 `sqlite:///mlflow.db`，产物在 `mlruns/`；`mlflow ui` 即可查看。
- 实验结论与生产模型选择依据见 [ml/README_experiments.md](ml/README_experiments.md)。

## 6. 测试

47 个 pytest 用例，三个文件各管一类场景：

- **输入校验**（[tests/test_schema.py](tests/test_schema.py)）：缺字段 / 类型错误 / 越界 / 空白类别 / 多余字段 → 422；边界值放行；
- **正常路径**（[tests/test_predict.py](tests/test_predict.py)）：契约形状、类别合法、置信度范围、推理确定性、OpenAPI；
- **异常边界**（[tests/test_edge_cases.py](tests/test_edge_cases.py)）：未知类别兜底、Unicode/极端数值、推理失败转 500、占位模型回落。
