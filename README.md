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
│   ├── registry/         # current_model.md + model.pkl（不入 git）
│   ├── README_data.md    # 数据集版本说明
│   ├── README_experiments.md
│   └── PROVENANCE.md
├── data.dvc / data/      # DVC 指针（原始数据不入 git）
├── tests/                # pytest：输入校验 / 正常路径 / 异常边界
├── requirements.txt      # 全项目唯一依赖事实来源
├── Dockerfile / .dockerignore
├── .github/workflows/ci.yml
├── DEPLOYMENT.md
└── README.md
```

## 4. CI/CD：代码从 commit 到 prod 的路径

**分支模型**：`feature/*` → `dev`（集成）→ `staging`（预发）→ `main`（生产），只允许 PR 合并。

| 触发 | 流水线动作 |
| --- | --- |
| 向 dev / staging / main 发 PR | CI：ruff 检查 + pytest 全量测试（Docker 落地后追加镜像构建）；**任何一步失败禁止合并** |
| merge → staging | CD(staging)：用 staging 环境 GitHub Secrets 生成 staging `.env` → 构建 staging 镜像 |
| merge → main | CD(prod)：用 production 环境 GitHub Secrets 生成生产 `.env` → 构建生产镜像 |

路径示例：`commit → push feature/* → PR 到 dev（CI 复验）→ merge 进 dev →
PR dev→staging（CI 复验 + staging 构建部署）→ PR staging→main（CI 复验 + prod 构建部署）`。

配置见 [.github/workflows/ci.yml](.github/workflows/ci.yml)，部署细节见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 5. MLOps

### 数据版本（DVC）
- `data/` 由 DVC 跟踪，git 只存 `.dvc` 指针；远端存储 Azure Blob（待配置）。
- 当前版本 v1（原始数据）；v1 → v2 的变更计划见 [ml/README_data.md](ml/README_data.md)。

### 实验跟踪（MLflow）
- 训练脚本自动记录：git commit、数据集版本/md5（tag）、全部超参（param）、
  accuracy / F1 / ROC-AUC（metric）、混淆矩阵 PNG 与最终 model.pkl（artifact）。
- 本地自托管：元数据 `sqlite:///mlflow.db`，产物在 `mlruns/`；`mlflow ui` 即可查看。
- 实验结论与生产模型选择依据见 [ml/README_experiments.md](ml/README_experiments.md)。

## 6. 测试

pytest 43 个用例，覆盖三类场景：
- **输入校验**（20 例）：缺字段 / 类型错误 / 越界 / 空白类别 / 多余字段 → 422；边界值放行；
- **正常路径**：契约形状、类别合法、置信度范围、推理确定性、OpenAPI；
- **异常边界**：未知类别兜底、Unicode/极端数值、推理失败转 500、占位模型回落。
