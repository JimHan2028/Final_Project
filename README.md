# 银行定期存款认购预测系统 · MLOps / DevOps 全流程课程项目

> **一句话说明**：这是一个把「训练一个机器学习模型」升级成「一条可复现、可测试、可自动化上线的工程流水线」的课程期末项目。
> 数据用 DVC 做版本管理，实验用 MLflow 做追踪，模型用 FastAPI 包成 HTTP 服务，服务用 Docker 打成镜像，全流程用 GitHub Actions 自动跑 CI（检查 + 测试 + 冒烟）和 CD（构建镜像 → 推送到 GHCR → 再验证镜像能跑起来）。

**仓库**：<https://github.com/JimHan2028/Final_Project>
**技术栈**：Python 3.12 · scikit-learn · pandas · FastAPI · MLflow · DVC（Azure Blob）· Docker · GitHub Actions
**核心指标（生产模型）**：ROC-AUC **0.8141**、Accuracy 0.9025、F1(正类) 0.3837

---

## 目录

- [0. 怎么读这份文档](#0-怎么读这份文档)
- [1. 项目背景：要解决的问题](#1-项目背景要解决的问题)
- [2. 零基础先补的 7 个概念](#2-零基础先补的-7-个概念)
- [3. 整体架构：一条从数据到服务的流水线](#3-整体架构一条从数据到服务的流水线)
- [4. 技术栈与每个工具的具体作用](#4-技术栈与每个工具的具体作用)
- [5. 目录结构详解](#5-目录结构详解)
- [6. 数据集与数据版本管理](#6-数据集与数据版本管理)
- [7. 模型与实验追踪（含真实指标）](#7-模型与实验追踪含真实指标)
- [8. 服务层：接口契约与实现要点](#8-服务层接口契约与实现要点)
- [9. 从零开始复现（一步一步）](#9-从零开始复现一步一步)
- [10. CI/CD 流水线详解](#10-cicd-流水线详解)
- [11. 测试体系（47 个用例）](#11-测试体系47-个用例)
- [12. Docker 镜像与部署细节](#12-docker-镜像与部署细节)
- [13. 常见问题排查](#13-常见问题排查)
- [14. 项目亮点](#14-项目亮点)
- [15. 已知限制与历史遗留](#15-已知限制与历史遗留)
- [16. 相关文档索引](#16-相关文档索引)

---

## 0. 怎么读这份文档

不同目的读不同章节：

| 你的目的 | 建议读 |
|---|---|
| 我完全不懂 MLOps，想搞明白这个项目在干什么 | §1 → §2 → §3 |
| 我想把项目在自己电脑上跑起来 | §9（照着敲命令即可） |
| 我想看模型效果到底多好、为什么选这个模型 | §6 → §7 |
| 我想知道 CI/CD 是怎么自动跑的 | §10 |
| 我跑起来报错了 | §13 |

> 项目里的 `How to test.txt` 是一份**现场演示手册**（162 行，按"演示第 1~8 部分"组织，带每一步的预期输出），`DEPLOYMENT.md` 记录分支流转与环境变量注入方式。本 README 是把这些零散文档整合成一份系统说明。

---

## 1. 项目背景：要解决的问题

### 1.1 业务场景

一家银行通过**电话外呼**推销定期存款（term deposit）。每打一个电话都有成本（人力、时间），而绝大多数客户不会认购——在全量数据里，最终认购的只占 **11.27%**。

所以真正的问题不是"预测这个人买不买"，而是：

> **在只能打有限数量电话的前提下，应该先打给哪些人？**

这决定了整条技术路线：
- 模型输出不能只是一个 `yes/no`，而必须是一个**可用于排序的分数** `score`；
- 评价指标不能用 Accuracy（把所有人都判成"不认购"就已经有 88.7% 的准确率，毫无意义），而要用 **ROC-AUC**（衡量"排序质量"：随机挑一个真实认购客户和一个真实不认购客户，前者分数更高的概率）。

### 1.2 为什么不是"写个 notebook 就完事"

课程要训练的是一套**工程能力**，而一个只能在自己电脑上跑一次的 notebook 有四个致命问题：

| 问题 | 现实后果 | 本项目怎么解决 |
|---|---|---|
| 数据放在本地，换个版本就说不清训的是什么 | 模型效果无法复现 | **DVC** 把数据/模型内容存到 Azure Blob，Git 里只留 md5 指针 |
| 试了十几个参数组合，哪个最好的记录在脑子里 | 无法比较、无法回溯 | **MLflow** 每次训练自动记录 git commit、数据 md5、超参、指标、混淆矩阵图 |
| 模型只能在 notebook 里用 | 业务方用不了 | **FastAPI** 包成 HTTP 服务，带交互式文档 `/docs` |
| 手改代码、手动上传、手动重启 | 上线靠人肉，容易出错 | **GitHub Actions** 自动 lint → 测试 → 构建镜像 → 冒烟验证 → 推送镜像 |

### 1.3 这个项目实际交付了什么

可验证的交付物（都能在仓库里找到对应文件）：

- 一份可复现的训练脚本 `ml/train.py`（固定 `random_state=42`，重跑指标一致）
- 4 组核心实验记录（2 个模型 × 2 个数据版本）+ 生产模型的溯源文档 `ml/PROVENANCE.md`
- 一个在线预测服务 `app/`（2 个接口：`/health`、`/predict`）
- 47 个自动化测试（全部通过，本地实测 `47 passed in 8.09s`）
- 一条 CI 流水线（3 个 job）+ 一条 CD 流水线（1 个 job，推 GHCR）
- 一个生产级 Dockerfile（非 root 用户、依赖层缓存、健康检查）
- 两份数据版本 v1/v2（含可重跑的清洗脚本 `ml/make_v2_data.py`）

---

## 2. 零基础先补的 7 个概念

不需要提前学，看这一节就够理解后面所有内容。

| 概念 | 用大白话解释 | 在本项目里对应什么 |
|---|---|---|
| **机器学习模型** | 从历史数据里学出一条"打分规则"的东西。这里学的是"这个客户认购的可能性有多大" | `ml/registry/model.pkl` 这个文件本身 |
| **数据版本管理（DVC）** | Git 管代码，但管不了 5MB 的 CSV（会把仓库撑爆）。DVC 把大数据存到云端，Git 里只存一个"指纹"文件 | `data/*.csv.dvc`、`ml/registry/model.pkl.dvc` |
| **实验追踪（MLflow）** | 每次训练都自动记小本本：用了什么参数、拿到什么指标、对应的代码是哪个 commit | `mlflow.db` + `mlruns/` 目录，网页界面 `mlflow ui` |
| **模型注册表（Registry）** | 一个固定目录，只放"当前生产在用的那一版模型"，服务启动只认这个路径 | `ml/registry/model.pkl` + `ml/registry/current_model.md` |
| **服务化（API）** | 把模型包成一个网址，别人发 HTTP 请求就能拿到预测结果 | FastAPI 应用 `app/main.py` |
| **CI（持续集成）** | 每次提交代码，机器自动帮你查代码风格、跑测试、试构建镜像——防止把坏代码合进去 | `.github/workflows/ci.yml` |
| **CD（持续交付/部署）** | 代码合到指定分支后，机器自动构建镜像并推送到镜像仓库，随时可部署 | `.github/workflows/cd.yml` → GitHub Container Registry(GHCR) |
| **容器镜像（Docker）** | 把"代码 + 依赖 + 运行环境"打包成一个文件，在哪台机器上跑都一样 | `Dockerfile` → `bank-marketing-api:local` |

一个额外但很重要的概念：

- **数据泄漏（data leakage）**：把"结果发生时才知道的信息"当成"预测时就能拿到的信息"。本项目里的 `duration`（通话时长）就是典型——只有电话打完才知道时长，而通话越久客户越可能认购。把它当特征，模型离线指标会很漂亮，上线后立刻失效。**所以本项目在两个数据版本里都把它剔除了。**

---

## 3. 整体架构：一条从数据到服务的流水线

```
┌──────────────┐   DVC 指针(.dvc)   ┌────────────────┐
│ 原始数据 CSV │ ─────────────────> │  Azure Blob    │
│ 41,188 × 21  │ <───────────────── │  (myazure)     │
└──────┬───────┘     dvc pull       └────────────────┘
       │ dvc pull
       v
┌──────────────┐   分层切分 8:2    ┌────────────────────────────┐
│  data/v1,v2  │ ────────────────> │ ml/train.py                │
│  19 个特征   │  seed=42          │  · 2 个候选模型             │
└──────────────┘                   │  · 整体 Pipeline(预处理+分类)│
                                   └───────┬────────────────────┘
                                           │ 每次训练写记录
                    ┌──────────────────────┼──────────────────────┐
                    v                      v                      v
            ┌───────────────┐      ┌──────────────┐      ┌────────────────┐
            │ MLflow        │      │ ROC-AUC 择优 │      │ model.pkl      │
            │ sqlite+mlruns │      │ (0.8141 胜出)│      │ (registry)     │
            │ git/data md5  │      └──────────────┘      └───────┬────────┘
            └───────────────┘                                    │ joblib 加载
                                                                 v
┌──────────────────────────────────────────────────────────────────────────┐
│ FastAPI 服务  app/main.py                                                 │
│   POST /predict  → 19 个字段 → Pipeline.predict → {label, score}          │
│   GET  /health   → {status:"ok", model_loaded:true}   ← Docker HEALTHCHECK │
│   GET  /docs     → 自托管 Swagger UI（不依赖外网 CDN）                      │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ Dockerfile（python:3.12-slim, 非 root）
                               v
                    ┌─────────────────────┐
                    │ 镜像 bank-marketing │
                    │ -api:ci / :staging  │
                    │        / :latest    │
                    └──────────┬──────────┘
                               │ GitHub Actions
                    ┌──────────┴───────────┐
                    v                      v
            ┌──────────────┐      ┌──────────────────┐
            │ CI: lint     │      │ CD: build & push │
            │  test        │      │  → GHCR          │
            │  model-gate  │      │  → 拉起来验证健康 │
            │  docker+冒烟 │      └──────────────────┘
            └──────────────┘
```

五个阶段分别对应：

| 阶段 | 目的 | 关键文件 | 失败了会怎样 |
|---|---|---|---|
| ① 数据版本管理 | 谁都能拿到"当时训模型用的那份数据" | `data/*.csv.dvc`、`.dvc/config` | `dvc pull` 报错，训练脚本直接提示"数据文件不存在" |
| ② 训练与实验追踪 | 每次实验都有记录，可比较、可复现 | `ml/train.py`、`ml/data_pipeline.py` | MLflow 仍可跑（本地 sqlite），但模型质量未知 |
| ③ 模型择优与注册 | 用统一标准（ROC-AUC）选一个进生产 | `ml/registry/model.pkl`、`current_model.md` | 服务回落默认模型（不可用于业务） |
| ④ 服务化部署 | 让外部能用上模型 | `app/`、`Dockerfile` | 容器健康检查失败，CI 直接红 |
| ⑤ CI/CD 与监控 | 每次改动自动验证，防止坏代码上线 | `.github/workflows/*.yml` | 分支保护拦住合并 |

---

## 4. 技术栈与每个工具的具体作用

版本来自 `requirements.txt`（**唯一事实来源**，Dockerfile 明确禁止在其中重复声明版本）：

| 类别 | 工具 | 版本 | 在本流水线中的**具体作用** |
|---|---|---|---|
| 环境 | Python | 3.12（CI 固定 `"3.12"`，Docker 用 `python:3.12-slim`） | 三处版本对齐，避免 pkl 反序列化失败 |
| 训练 | scikit-learn | 1.9.1 | 提供 `LogisticRegression`、`HistGradientBoostingClassifier` 与 `Pipeline`/`ColumnTransformer` |
| 训练 | numpy / scipy / pandas | 2.5.4 / 1.18.1 / 3.0.6 | pandas 读分号分隔 CSV 并按固定列序构造 DataFrame；numpy/scipy 为 sklearn 依赖 |
| 序列化 | joblib | 1.6.0 | 把「预处理 + 分类器」整体存成一个 `model.pkl`，服务端只反序列化一次 |
| 实验追踪 | mlflow | 3.17.1 | 记录 params/metrics/tags/artifacts；`mlflow ui` 提供可视化对比界面 |
| 绑图 | matplotlib | 3.11.2 | 训练时把测试集混淆矩阵画成 PNG 存进 MLflow artifacts（`Agg` 后端，无界面环境可用） |
| 数据版本 | DVC（`dvc[azure]`） | 随 CI 安装 | 数据与模型的云端存储与内容指纹；Git 只提交 `.dvc` 指针 |
| 服务 | FastAPI | 0.143.2 | 提供 `/predict`、`/health`、`/docs`；自动校验请求体 |
| 服务 | pydantic | 2.14.1 | 定义 `PredictRequest` / `PredictResponse` 的字段类型、范围与校验规则 |
| 服务 | uvicorn[standard] | 0.54.0 | ASGI 服务器，容器与本地启动命令都用它 |
| 测试 | pytest | 9.1.1 | 47 个用例，含参数化边界用例 |
| 测试 | httpx | 0.28.1 | FastAPI `TestClient` 的传输层（发请求不真正开端口） |
| 代码质量 | ruff | 0.17.1 | 一条命令做 lint + 格式检查，CI 里 `ruff check .` 与 `ruff format --check .` |
| 打包 | Docker | — | 镜像化的运行环境，非 root 用户运行 |
| CI/CD | GitHub Actions | — | `ci.yml`（3 job）、`cd.yml`（1 job），产物推 GHCR |
| 运行时配置 | 环境变量 `MODEL_PATH` | — | 覆盖默认模型路径，用来把服务切到 v2 模型做对比 |

**技术选型上的两个细节，也能看出"不是随便堆工具"：**

1. `sparse_threshold=0.0`：`HistGradientBoostingClassifier` 不接受稀疏矩阵，所以在 `ColumnTransformer` 里强制稠密输出（否则训练直接报错）。
2. `OneHotEncoder(handle_unknown="ignore")`：线上可能出现训练集里没有的类别（如 `job="astronaut"`），编码成全 0 而不是抛异常——这条被 `tests/test_edge_cases.py::test_unknown_category_accepted` 钉住。

---

## 5. 目录结构详解

```
Final_Project/
├── .dvc/                          # DVC 配置与本地缓存
│   ├── config                     #   remote myazure → azure://dvc, account wut2026
│   ├── config.local               #   SAS 凭据（被 .dvc/.gitignore 忽略，绝不入库）
│   ├── cache/                     #   按 md5 分片存放的真实数据（git 忽略）
│   └── .gitignore                 #   忽略 config.local / tmp / cache
├── .github/workflows/
│   ├── ci.yml                     # CI：lint-and-test / model-gate / docker（136 行）
│   └── cd.yml                     # CD：build-and-push 到 GHCR（99 行）
├── app/                           # FastAPI 服务层
│   ├── main.py                    #   路由：/predict /health /docs + lifespan 预热
│   ├── model_loader.py            #   模型单例加载、FakeModel 回落、score 语义处理
│   ├── schemas.py                 #   Pydantic 请求/响应模型（extra="forbid"）
│   └── static/                    #   自托管 Swagger UI 资源（js/css），解决国内 CDN 打不开
├── ml/                            # 机器学习侧
│   ├── train.py                   #   训练入口：2 候选模型 + MLflow 记录 + ROC-AUC 择优
│   ├── data_pipeline.py           #   特征列契约 + 预处理 Pipeline（训练与服务共用）
│   ├── make_v2_data.py            #   v1 → v2 清洗脚本（跑一次即可）
│   ├── configs/train.yaml         #   早期实验设计稿（当前 train.py 不读取，见 §15）
│   ├── figures/                   #   4 张早期实验图（PR 曲线 / 混淆矩阵，见 §15）
│   ├── registry/
│   │   ├── model.pkl              #   生产模型（git 忽略，由 model.pkl.dvc 跟踪）
│   │   ├── model.pkl.dvc          #   md5 877d8e6c… size 259170
│   │   └── current_model.md       #   "当前生产是哪个模型"的一页结论
│   ├── REPORT.md                  #   359 行实验报告（含公式推导，指标全部实测）
│   ├── README_experiments.md      #   4 个核心 run 的对比与择优理由
│   ├── README_data.md             #   数据来源与 v1/v2 差异
│   └── PROVENANCE.md              #   溯源：哪个 commit + 哪份数据 = 哪个模型
├── data/
│   ├── bank-additional-full.csv   #   v1 原始数据（git 忽略，DVC 跟踪）
│   ├── bank-additional-full.csv.dvc   # md5 f6cb2c12… size 5834924
│   ├── bank-additional-v2.csv     #   v2 清洗后数据（git 忽略，DVC 跟踪）
│   └── bank-additional-v2.csv.dvc #   md5 e8a1add9… size 4895155
├── tests/                         # pytest：47 个用例
│   ├── conftest.py                #   client / valid_payload 夹具
│   ├── test_schema.py             #   29 用例（参数化）输入校验
│   ├── test_predict.py            #   6 用例 正常路径
│   ├── test_edge_cases.py         #   8 用例 异常与边界
│   └── test_score_semantics.py    #   4 用例 score 语义回归
├── mlruns/                        # MLflow artifacts（git 忽略，本地生成）
├── mlflow.db                      # MLflow 元数据 sqlite（git 忽略，本地生成）
├── Dockerfile                     # 生产镜像定义（42 行）
├── .dockerignore                  # 构建上下文排除规则（注意：它本身被 .gitignore 忽略，见 §13）
├── .gitignore / .dvcignore
├── .env                           # 本地环境变量（git 忽略；Dockerfile 会 COPY 它）
├── requirements.txt               # 依赖唯一事实来源（19 行）
├── DEPLOYMENT.md                  # 分支流转 + .env 注入说明
├── How to test.txt                # 现场演示手册（8 个部分，带预期输出）
└── README.md                      # 本文件
```

**几个容易误解的点，先说清楚：**

- `data/*.csv` 和 `*.pkl` **不在 Git 里**（`.gitignore` 明确忽略），Git 只保留 `.dvc` 指针。所以新克隆仓库后必须 `dvc pull` 才能拿到数据和模型。
- `mlruns/` 与 `mlflow.db` **也被 Git 忽略**。也就是说，仓库里没有实验记录文件，谁想看实验记录，必须本地跑一次 `python ml/train.py` 生成，或者直接看 `ml/REPORT.md` / `ml/README_experiments.md`（结论与指标已写进文档）。
- 本项目**没有 `dvc.yaml` / `dvc.lock`**（已确认不存在）。也就是说 DVC 只用来做"数据和模型的内容版本管理"，没有用 DVC 定义多阶段流水线（`dvc stage`/`dvc repro`）。训练流程由 `ml/train.py` 一个脚本串起来。

---

## 6. 数据集与数据版本管理

### 6.1 数据来源

- 数据集：UCI Machine Learning Repository id 222，**Bank Marketing**（`bank-additional-full.csv`，**分号分隔**，读取时必须 `sep=";"`）
- 链接：<https://archive.ics.uci.edu/dataset/222/bank+marketing>
- 许可：公开数据集（CC BY 4.0）

### 6.2 真实规模（本地实算核对）

| 项目 | v1（原始） | v2（清洗后） |
|---|---|---|
| 行数 × 列数 | **41,188 × 21** | **41,016 × 21** |
| 正类（`y = yes`，认购） | **4,640** | **4,638** |
| 正类占比 | **11.2654%** | **11.3078%** |
| md5 | `f6cb2c1256ffe2836b36df321f46e92c` | `e8a1add9000aa97898f84ceddfbe32ac` |
| 文件大小 | 5,834,924 字节 | 4,895,155 字节 |

**21 列 = 20 个原始特征 + 1 个目标列 `y`。**

训练实际使用的特征 **19 个**（20 个特征里再剔掉 `duration`）：

| 类型 | 数量 | 字段 |
|---|---|---|
| 数值特征 | 9 | `age`, `campaign`, `pdays`, `previous`, `emp_var_rate`, `cons_price_idx`, `cons_conf_idx`, `euribor3m`, `nr_employed` |
| 类别特征 | 10 | `job`(12 类), `marital`(4), `education`(8→6), `default`(3), `housing`(3), `loan`(3), `contact`(2), `month`(10), `day_of_week`(5), `poutcome`(3) |

> 注意：CSV 原始列名带点号（`emp.var.rate`、`cons.price.idx`、`cons.conf.idx`、`nr.employed`），`ml/data_pipeline.py` 的 `load_dataframe()` 会把 `.` 统一替换成 `_`，所以 API 和代码里用的是下划线版本。

### 6.3 为什么剔除 `duration`（数据泄漏）

| 群体 | `duration` 平均值 |
|---|---|
| 认购客户（`y=yes`） | **553.2 秒** |
| 未认购客户（`y=no`） | **220.8 秒** |

差 2.5 倍——光靠这一列就能把两个类分得七七八八。但 `duration` 只有在**电话打完之后**才知道，而业务决策发生在**打电话之前**。数据集作者自己在说明文档里也写了：该属性"强烈影响输出目标……若想要一个真实的预测模型就应当丢弃"。

所以在 `ml/data_pipeline.py` 里，特征列 `FEATURE_COLUMNS` 是一个**白名单**（19 列，不含 `duration`），并单独定义了 `LEAKAGE_COLUMNS = ("duration",)` 作为显式说明；同时用一个断言保证"数值列 ∪ 类别列 == 特征列"：

```python
assert set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS) == set(FEATURE_COLUMNS)
```

这条断言的作用是：以后谁想偷偷加/删一列，启动时立刻报错，而不是让泄漏悄悄回到模型里。

### 6.4 v1 → v2 到底改了什么

`ml/make_v2_data.py`（34 行，跑一次生成 v2）做三件事：

| 改动 | 动机 | 实测影响 |
|---|---|---|
| 把 `basic.4y` / `basic.9y` / `basic.6y` 合并为 `basic` | 三个近义稀疏等级，One-Hot 后每类样本太少，几乎学不到信号 | 类别数 8 → 6，行数不变 |
| 删除 `campaign > 20` 的行 | 同一活动里打 20 次以上还没成交，属于骚扰级外呼，会污染"联系次数"特征 | v1 中这类有 **157 行** |
| 删除完全重复的行 | 去重记录 | v1 中有 **12 行**完全重复 |

合计 **41,188 → 41,016（−172 行）**，正类只少了 2 个（4,640 → 4,638）。注意 157 + 12 = 169 < 172，说明两类问题行之间存在重叠（去重后合计去掉了 172 行）。

### 6.5 DVC 是怎么管这些数据的

- 远端：`myazure` → `azure://dvc`，存储账户 `wut2026`（课程提供的 Azure 容器）
- Git 里提交的是三个指针文件：`data/bank-additional-full.csv.dvc`、`data/bank-additional-v2.csv.dvc`、`ml/registry/model.pkl.dvc`
- 真实内容存在 Azure Blob（远端）与本地 `.dvc/cache/`（按 md5 前两位分目录）
- **凭据**：SAS token 放在 `.dvc/config.local`，而 `.dvc/.gitignore` 忽略该文件，所以不会进仓库。**别人克隆后需要自己配置凭据才能 `dvc pull`。**

常用命令：

```powershell
dvc remote list                                         # 看远端配置
Get-Content data\bank-additional-full.csv.dvc           # 看指针（md5 + size）
dvc pull                                                # 拉全部（数据 + 模型）
dvc pull ml/registry/model.pkl.dvc                      # 只拉模型
dvc push                                                # 推回远端
dvc status                                              # 检查本地与指针是否一致
Get-ChildItem .dvc\cache\files\md5 -Recurse -Name       # 看本地缓存分片
```

---

## 7. 模型与实验追踪（含真实指标）

### 7.1 训练流程（`ml/train.py`，313 行）

```
读取 CSV（默认 v1） 
  → 按 FEATURE_COLUMNS 取 19 列 + 目标 y
  → train_test_split(test_size=0.2, stratify=y, random_state=42)   # 分层，保证正类比例一致
  → 计算数据文件 md5、读取当前 git commit
  → 对 2 个候选模型各跑一次：
        build_pipeline(estimator).fit(...)
        mlflow.start_run → log_params / log_metrics / set_tags
        → 画混淆矩阵 PNG → log_artifact("plots")
  → 按 ROC-AUC 选最优（max(results, key=roc_auc)）
  → joblib.dump(best_pipeline, "ml/registry/model.pkl")
  → 回到最优 run，把 pkl 也归档进 artifacts/model/，并打上 prod_candidate=true
```

两个候选模型（同一个预处理 Pipeline 内，保证比较的是分类器而不是特征）：

| | `logreg_baseline` | `hgb_improved` |
|---|---|---|
| 算法 | `LogisticRegression` | `HistGradientBoostingClassifier` |
| 超参 | `max_iter=1000`, `class_weight="balanced"`, `random_state=42` | 库默认, `random_state=42` |
| 定位 | 基线 / 可解释 / 召回优先的备选 | **生产模型** |

预处理（训练与服务完全共用同一份代码 `ml/data_pipeline.py`）：

- 数值列：中位数填充（`SimpleImputer(strategy="median")`）→ 标准化（`StandardScaler`）
- 类别列：One-Hot（`handle_unknown="ignore"`）
- 两者用 `ColumnTransformer` 拼起来，再和分类器封成一个 `Pipeline` 整体序列化

**这个"整体序列化"是服务层不会出错的关键**：线上不需要重新实现任何预处理逻辑，`joblib.load` 出来就是一个能吃原始字段的完整对象。

### 7.2 真实实验结果（4 个核心 run，全部实测）

| 指标 | `logreg_baseline` v1 | **`hgb_improved` v1** | `logreg_baseline` v2 | `hgb_improved` v2 |
|---|---|---|---|---|
| 数据版本 | v1 | **v1** | v2 | v2 |
| 特征数 | 19 | 19 | 19 | 19 |
| **ROC-AUC** | 0.8009 | **0.8141** | 0.8021 | **0.8156** |
| Accuracy | 0.8350 | **0.9025** | 0.8329 | 0.9021 |
| F1（正类） | **0.4689** | 0.3837 | 0.4621 | 0.3662 |
| 是否上生产 | 否 | **是** | 否 | 否 |

拆分规模：v1 → 训练 32,950 / 测试 8,238（其中正类 928）；v2 → 训练 32,812 / 测试 8,204。

对应的 MLflow run id（`sqlite:///mlflow.db`，实验名 `bank-marketing`）：

| Run | run id |
|---|---|
| logreg_baseline (v1) | `24182ba979104956bdb3faf3b369d66a` |
| hgb_improved (v1) | `fbab6806fab4401a8643e861085c693b` |
| logreg_baseline (v2) | `48ae369350624a3aa77c0a91ec6b218c` |
| hgb_improved (v2) | `fbdf0b31d4884473a9129ca6c6295ddb` |

> **关于"4 个 run"的准确说法**：`ml/REPORT.md` 与 `ml/README_experiments.md` 里说的"4 个 run"指的是**作为结论依据的 4 组核心实验**。实际本地 tracking store（`mlflow.db`）里累计记录过 **16 次 run**，其中 3 次状态为 `FAILED`（都在早期排查 artifacts 记录问题时产生，配置与成功的 run 完全相同，重跑后指标一致：`0.8141 / 0.9025 / 0.3837`）。失败的那些不影响任何结论，但如实说明：跟踪库里是有失败痕迹的。

### 7.3 为什么选 `hgb_improved` + v1 上生产

这不是"哪个数字大就用哪个"，报告里给出的理由是完整的：

1. **同版本内比较（v1）**：HGB 的 ROC-AUC 高 **+0.0132**、Accuracy 高 6.75 个百分点（0.8350 → 0.9025），但 F1(正类) 低 **0.0852**（0.4689 → 0.3837）。F1 的差距来自逻辑回归用了 `class_weight="balanced"`（用精确率换召回率）。业务消费的是 `score` 的**排序**，不是 `label`，所以按排序质量（ROC-AUC）选——HGB 胜。
2. **跨版本比较（v1 vs v2）**：v2 只把 HGB 的 ROC-AUC 抬高了 **+0.0015**（0.8141 → 0.8156），在 8.2k 行测试集上完全在噪声范围内；而 F1 掉了 **−0.018**。v2 删掉的 172 行里只有 2 个正类，本来就没有可回收的信号。**所以生产继续用 v1**，v2 作为"记录在案、可重跑"的备选留在 DVC 里。
3. **指标口径**：Accuracy 在这里没有区分力——全部预测为"不认购"就已经有 `1 − 11.27% = 88.73%`，任何模型都在 88~90% 徘徊，所以它只作为记录项，不作决策依据。

### 7.4 阈值与 `score` 的语义（一个极容易写错的地方）

服务返回两个字段：

| 字段 | 含义 | 实现位置 |
|---|---|---|
| `label` | 模型 `predict()` 的结果（库内部 0.5 阈值判类） | `app/model_loader.py` |
| `score` | **正类概率 P(yes \| x)**，不是"预测类的置信度" | `_proba_to_scores()` |

为什么强调？因为 `predict_proba` 返回的是"每个类一个概率"，顺序由 `classes_` 决定。如果直接取 `max(row)`，一个被预测为 `no`、`p(yes)=0.03` 的客户会被报成 `score=0.97`——业务方就会优先去打**最不可能认购**的人。

所以代码里做了这件事：通过 `classes_` 定位 `yes` 的下标，直接取那一列。并且用 4 个回归测试（`tests/test_score_semantics.py`，用打桩模型构造"classes_ 顺序颠倒""classes_ 缺失"等情况）把这个行为钉死。

阈值本身是**部署参数而不是模型属性**：当前沿用库默认 0.5（等于假设"打错一个电话的成本 = 漏掉一个客户的成本"）。报告里给了成本敏感的盈亏平衡点公式 `τ* = c_FP / (c_FP + c_FN)`，并说明银行没提供成本比，所以保留 0.5、把 `score` 原样暴露，以后调运营点只需改一行，不用重训。

### 7.5 每次训练都记录了什么（溯源闭环）

| 类型 | 内容 |
|---|---|
| tag | `git_commit`、`dataset_version`、`data_md5`、`prod_candidate`（仅最优 run） |
| param | `model_name`、`dataset_version`、`train_rows`、`test_rows`、`clf_*`（分类器全部超参） |
| metric | `accuracy`、`f1`、`roc_auc` |
| artifact | `plots/confusion_matrix_<name>.png`、`model/model.pkl`（仅最优 run） |

生产模型的溯源信息（`ml/PROVENANCE.md` + `ml/registry/current_model.md`）：

| 项 | 值 |
|---|---|
| 模型 | `hgb_improved`（HistGradientBoostingClassifier），**训练于 v1** |
| 训练 commit | `a967252abb802f434b7c755f068f90d60e46735f` |
| 训练环境 | Python 3.12.10 / scikit-learn 1.9.1 / numpy 2.5.4 / pandas 3.0.6 / scipy 1.18.1 / joblib 1.6.0 |
| v2 实验的 HEAD | `b44417d`（未上生产） |
| 指标 | accuracy 0.9025 / f1(yes) 0.3837 / **roc_auc 0.8141** |

`data_md5` tag 与 `data/*.dvc` 里的 md5 是同一串值——这就是"MLflow 记录"和"DVC 数据"互相对得上的原因，不需要人去记忆"v1 到底是哪份文件"。

### 7.6 怎么看实验记录

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
# 浏览器打开 http://127.0.0.1:5000
# 1) 左侧选实验 bank-marketing
# 2) Group by → tags.dataset_version（按数据版本分组）
# 3) 列设置里勾选 roc_auc / f1 / accuracy 做对比
# 4) 点进单个 run 看 Parameters / Metrics / Artifacts（混淆矩阵图就在 plots/ 下）
```

---

## 8. 服务层：接口契约与实现要点

### 8.1 三个入口

| 方法 | 路径 | 作用 | 响应 |
|---|---|---|---|
| GET | `/health` | 健康检查：进程活着 + 模型是否真加载 | `{"status":"ok","model_loaded":true}` |
| POST | `/predict` | 单条客户预测 | `{"label":"no","score":0.0793}` |
| GET | `/docs` | 交互式 Swagger UI（本地静态资源） | HTML |

### 8.2 `/predict` 请求体（19 个必填字段，`extra="forbid"`）

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `age` | int | 17 ~ 120 | 年龄 |
| `job` | str | 非空白，≤64 | 职业（`admin.` / `blue-collar` / …，12 类） |
| `marital` | str | 非空白，≤64 | 婚姻状况（`married`/`single`/`divorced`/`unknown`） |
| `education` | str | 非空白，≤64 | 教育程度（`university.degree`/`basic.4y`/…） |
| `default` | str | 非空白，≤64 | 是否有违约记录（`no`/`unknown`/`yes`） |
| `housing` | str | 非空白，≤64 | 住房贷款（`yes`/`no`/`unknown`） |
| `loan` | str | 非空白，≤64 | 个人贷款（`yes`/`no`/`unknown`） |
| `contact` | str | 非空白，≤64 | 联系方式（`telephone`/`cellular`） |
| `month` | str | 非空白，≤64 | 最后一次联系月份（`jan`~`dec`） |
| `day_of_week` | str | 非空白，≤64 | 最后一次联系星期（`mon`~`fri`） |
| `campaign` | int | 1 ~ 100 | 本次活动中联系次数 |
| `pdays` | int | 0 ~ 999 | 距上次联系天数；**999 表示此前从未联系** |
| `previous` | int | 0 ~ 100 | 本次活动之前的联系次数 |
| `poutcome` | str | 非空白，≤64 | 上次活动结果（`nonexistent`/`failure`/`success`） |
| `emp_var_rate` | float | — | 就业变化率 |
| `cons_price_idx` | float | — | 消费者价格指数 |
| `cons_conf_idx` | float | — | 消费者信心指数 |
| `euribor3m` | float | — | 3 个月欧元同业拆借利率 |
| `nr_employed` | float | — | 雇员人数（季度） |

注意：`duration` **不在**请求体里定义——不是"可选"，而是"根本不存在"，从接口层面再次防止泄漏特征被传进来。

一条可用的请求示例（也是 `tests/conftest.py` 里的夹具数据，取自训练集真实样本）：

```json
{
  "age": 56, "job": "housemaid", "marital": "married", "education": "basic.4y",
  "default": "no", "housing": "no", "loan": "no", "contact": "telephone",
  "month": "may", "day_of_week": "mon", "campaign": 1, "pdays": 999,
  "previous": 0, "poutcome": "nonexistent", "emp_var_rate": 1.1,
  "cons_price_idx": 93.994, "cons_conf_idx": -36.4, "euribor3m": 4.857,
  "nr_employed": 5191.0
}
```

### 8.3 `model_loader.py` 的四个设计点（172 行，值得逐个看）

| 设计点 | 代码位置 | 为什么这么写 |
|---|---|---|
| **进程内单例** | `get_model()` | 服务启动时（`lifespan`）加载一次，避免第一个请求承担加载延迟 |
| **FakeModel 回落** | `load_model()` → 文件不存在时 | 真实 pkl 没到位时服务仍能启动、CI 仍能跑通测试；此时 `is_fake()` 为真 |
| **`MODEL_PATH` 环境变量** | `_resolve_model_path()` | 可以把服务临时切到 `ml/registry/model_v2.pkl` 做对比，不用改代码 |
| **分数语义统一** | `_proba_to_scores` / `_decision_to_scores` / `_sigmoid` | 屏蔽模型差异：有 `predict_proba` 就用正类概率；只有 `decision_function` 就过 sigmoid；都没有就给中性分 1.0 |

另外两个细节：
- 输入是「记录字典列表」，`_records_to_frame()` 按 `FEATURE_COLUMNS` 的**固定列序**构造 DataFrame——列序错了模型会静默给出错的结果，所以这里不许调用方自己拼 DataFrame。
- 推理异常统一转成 HTTP 500（`main.py` 里 `except Exception → HTTPException`），不让进程裸崩。

### 8.4 自托管 Swagger UI（一个真实的环境适配）

FastAPI 默认的 `/docs` 会从国外 CDN 加载 Swagger 的 js/css。在国内网络下这会加载失败，页面一片空白。所以 `app/main.py` 里做了：

- 关闭默认 `docs_url=None`，改成自己写的 `/docs` 路由；
- 用 `get_swagger_ui_html(...)` 把资源地址指向本地 `/static/swagger-ui-bundle.js` 和 `/static/swagger-ui.css`；
- 这两个文件真实存在于 `app/static/`。

代价是镜像里多带了两份静态资源，收益是**离线/内网环境也能打开文档**。

---

## 9. 从零开始复现（一步一步）

> 以下命令默认项目已在 `C:\Users\Administrator\Desktop\历史\Final_Project`（或你自己克隆的位置）。PowerShell 里激活虚拟环境用 `.\.venv\Scripts\Activate.ps1`；如果提示"禁止运行脚本"，先执行 `Set-ExecutionPolicy -Scope Process Bypass`。

### 步骤 0：装前置工具

| 工具 | 验证命令 | 说明 |
|---|---|---|
| Git | `git --version` | 拉代码 |
| Python 3.12 | `python --version` | 版本必须 3.12（pkl 反序列化与依赖解析都对版本敏感） |
| DVC（可选） | `dvc --version` | 拉数据/模型；不装也行，见 9.6 |
| Docker Desktop（可选） | `docker --version` | 构建镜像；只跑 API 不需要 |

### 步骤 1：拿代码

```powershell
git clone https://github.com/JimHan2028/Final_Project.git
cd Final_Project
```

### 步骤 2：建虚拟环境并装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Linux / macOS 对应命令：

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 步骤 3：拉数据与模型（DVC）

```powershell
pip install "dvc[azure]"      # 首次需要
dvc remote list               # 应输出 myazure  azure://dvc
dvc pull                      # 拉 data/ 两个 CSV + ml/registry/model.pkl
dvc status                    # 应输出 Data and pipelines are up to date
```

**如果 `dvc pull` 因为凭据失败**（远端是课程提供的 Azure `wut2026` 账户，SAS token 不在仓库里）：

- 你有该容器的 SAS token → `dvc remote modify --local myazure sas_token "<你的token>"`（会写进被 Git 忽略的 `.dvc/config.local`）
- 你没有 token → 走 9.6 的替代路径：自己从 UCI 下载原始数据 + 本地训一个模型

### 步骤 4：训练 + 看实验记录

```powershell
python ml/train.py
# 输出示例：两个候选模型的 accuracy / f1 / roc_auc，并标注“<- 生产候选”
# 结束后 ml/registry/model.pkl 被覆盖为最优模型

mlflow ui --backend-store-uri sqlite:///mlflow.db   # 另开一个终端
# http://127.0.0.1:5000
```

只想做 v2 对比、**不想覆盖生产模型**时（这就是 `--no-save` 的用途）：

```powershell
python ml/train.py --data data/bank-additional-v2.csv --dataset-version v2 --no-save
```

想把两个版本的模型并排保留：

```powershell
python ml/train.py --data data/bank-additional-full.csv --dataset-version v1 --model-path ml/registry/model_v1.pkl
python ml/train.py --data data/bank-additional-v2.csv   --dataset-version v2 --model-path ml/registry/model_v2.pkl
```

> 因为固定了 `random_state=42`（切分 + 两个模型的随机种子都是 42），重跑得到的指标应当与 §7.2 表格一致。

### 步骤 5：跑代码质量检查与测试

```powershell
ruff check .              # 期望：All checks passed!
ruff format --check .     # 期望：N files already formatted
pytest -v                 # 期望：47 passed
```

### 步骤 6：启动 API 并试一发

```powershell
uvicorn app.main:app --port 8000
# 另开终端：
Invoke-RestMethod http://127.0.0.1:8000/health
# 期望：status=ok  model_loaded=True

$body = @{
  age = 35; job = "admin."; marital = "married"; education = "university.degree"
  default = "no"; housing = "yes"; loan = "no"; contact = "cellular"
  month = "may"; day_of_week = "mon"; campaign = 2; pdays = 999
  previous = 0; poutcome = "nonexistent"; emp_var_rate = -1.8
  cons_price_idx = 92.893; cons_conf_idx = -46.2; euribor3m = 1.3; nr_employed = 5099.1
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body -ContentType "application/json"
# 期望：label=yes/no，score 在 0~1 之间
```

浏览器打开 <http://127.0.0.1:8000/docs>，点 `POST /predict` → Try it out → Execute，也能测。

### 步骤 7：Docker（可选，需要 Docker Desktop 在运行）

```powershell
# 前提：仓库根目录必须有 .env（Dockerfile 会 COPY 它），否则构建直接失败
#   本地随便写一行即可，例如： APP_ENV=local
docker build -t bank-marketing-api:local .
docker images bank-marketing-api
docker run -d --name bank-api -p 8001:8000 bank-marketing-api:local
Start-Sleep -Seconds 30
docker inspect --format "{{.State.Health.Status}}" bank-api    # 期望：healthy
Invoke-RestMethod http://127.0.0.1:8001/health                 # 期望：model_loaded=True
docker rm -f bank-api
```

> 这里用 8001 端口映射到容器内 8000，避免和本地已运行的 uvicorn 抢端口。

### 9.6 没有 DVC 凭据时的替代路径（完全可行）

```powershell
# 1) 从 UCI 下载 bank-additional.zip，解压出 bank-additional-full.csv
#    https://archive.ics.uci.edu/dataset/222/bank+marketing
# 2) 放到 data/bank-additional-full.csv（保持分号分隔、文件名一致）
# 3) 校验 md5 是否为 f6cb2c1256ffe2836b36df321f46e92c
Get-FileHash data\bank-additional-full.csv -Algorithm MD5
# 4) 本地训练出自己的模型（不依赖远端 pkl）
python ml/train.py
```

v2 数据不需要下载，本地可以从 v1 现算：

```powershell
python ml/make_v2_data.py     # v1 → data/bank-additional-v2.csv（注意脚本注释：跑一次就够）
```

---

## 10. CI/CD 流水线详解

### 10.1 触发条件总览（来自两个 workflow 的真实配置）

| 事件 | 分支 | 触发哪个流程 |
|---|---|---|
| `pull_request` | `dev`、`staging`、`main`、`feature` | **CI** |
| `push` | `dev`、`staging`、`main`、`feature`、`feature-app-api`、`feature-ml-rui` | **CI** |
| `push` | `staging`、`main` | **CI + CD** |
| 手动 | Actions 页面 `Run workflow` | 两个流程都支持 `workflow_dispatch` |

> 注意 `push` 的普通 `feature` 分支不在 CI 列表里（列表里是 `feature-app-api`、`feature-ml-rui` 这类具体特性分支），所以往 `feature-app-api` 推代码会跑 CI，而往其它自建特性分支推代码可能需要走 PR 才触发。

### 10.2 CI（`.github/workflows/ci.yml`，136 行，3 个 job）

**job 1：`lint-and-test`** — 代码能不能进

| 步骤 | 命令 | 目的 |
|---|---|---|
| 检出 + Python 3.12 + pip 缓存 | `actions/checkout@v4`、`setup-python@v5` | 环境准备 |
| 装依赖 | `pip install -r requirements.txt` | 与本地完全一致的依赖 |
| Lint | `ruff check .` | 静态检查（未使用变量、语法陷阱等） |
| 格式 | `ruff format --check .` | 只检查不修改，防止格式漂移 |
| 测试 | `pytest -v` | 47 个用例全绿才算过 |

**job 2：`model-gate`** — 模型是不是真的

| 步骤 | 逻辑 |
|---|---|
| 装依赖 | 同上 |
| 拉模型产物 | `if: vars.DVC_REMOTE_READY == 'true'` → `pip install "dvc[azure]"` + `dvc pull ml/registry/model.pkl.dvc` |
| 断言不是 FakeModel | 内联 Python：调用 `model_loader.load_model()`，若 `is_fake()` 为真 → 当 `DVC_REMOTE_READY=true` 时 `sys.exit(错误)`（硬门禁）；否则只打 `::warning::` 并退出 0 |

这就是"**门禁可软可硬**"的设计：仓库变量 `DVC_REMOTE_READY` 配好远端后置为 `true`，model-gate 才变成真正会拦住合并的硬检查；没配好时不会让所有人都卡在红叉上。硬门禁还会额外断言真实模型必须有 `predict` 方法。

**job 3：`docker`** — 镜像能不能跑（`needs: lint-and-test`，测试不过就不必浪费构建时间）

| 步骤 | 说明 |
|---|---|
| 拉模型（条件同上） | 构建镜像需要 `ml/registry/model.pkl` |
| 写占位 `.env` | `.env` 被 Git 忽略，checkout 拿不到，但 Dockerfile 会 `COPY .env`，所以 CI 里 `printf 'APP_ENV=ci\nLOG_LEVEL=info\n' > .env` |
| 构建 | `docker build -t bank-marketing-api:ci .` |
| 冒烟测试 | 起容器 → 最多轮询 30 次（每次 2 秒，共 60 秒）请求 `/health` → 拿到 200 才算就绪；若 `DVC_REMOTE_READY=true`，还要求响应里的 `"model_loaded":true` |
| 失败取证 | `if: failure()` → `docker logs api`，把容器日志打到 CI 日志里方便排查 |

### 10.3 CD（`.github/workflows/cd.yml`，99 行，1 个 job）

`build-and-push` 的目标：**构建镜像 → 推送到 GHCR → 把推上去的镜像再拉起来验证一次**。

| 步骤 | 细节 |
|---|---|
| 权限 | `contents: read`、`packages: write`（推 GHCR 需要） |
| GitHub Environment | `${{ github.ref_name == 'main' && 'production' \|\| 'staging' }}` → main 走 `production`，staging 走 `staging`（便于分别配置审批与密钥） |
| 渲染 `.env` | staging 用 `secrets.STAGING_ENV_CONTENT`，main 用 `secrets.PROD_ENV_CONTENT`，`printf '%s\n' "$ENV_CONTENT" > .env` |
| 拉模型 | `if: vars.DVC_REMOTE_READY == 'true'` → `dvc pull ml/registry/model.pkl.dvc` |
| 镜像名 | `echo "IMAGE_NAME=${GITHUB_REPOSITORY,,}"` → 转小写（GHCR 要求小写路径） |
| 登录 | `docker/login-action@v3`，用户 `github.actor`，密码 `secrets.GITHUB_TOKEN` |
| 打标签 | `docker/metadata-action@v5`：`type=sha,format=short`、`type=ref,event=branch`、staging 分支额外打 `staging`、main 分支额外打 `latest` |
| 构建推送 | `docker/build-push-action@v6`，`push: true` |
| 推送后验证 | 取第一个 tag `docker run -d` 起容器 → 60 秒内轮询 `/health` → 失败打印日志并 `exit 1`；`DVC_REMOTE_READY=true` 时同样要求 `model_loaded=true` |

**"推上去之后再拉起来跑一遍"这一步是这条 CD 最有价值的地方**：它验证的不是"本地能构建"，而是"**推送到镜像仓库的那个具体镜像能跑**"。镜像 tag 错了、层丢了、架构不对，这里都会红。

### 10.4 仓库需要配置的 Secrets 与 Variables

| 名称 | 类型 | 用途 |
|---|---|---|
| `STAGING_ENV_CONTENT` | Secret | staging 环境 `.env` 全文（被 `printf` 重定向成文件） |
| `PROD_ENV_CONTENT` | Secret | production 环境 `.env` 全文 |
| `DVC_REMOTE_READY` | **Variable**（不是 Secret） | 值为 `true` 时，CI/CD 才真正 `dvc pull` 模型并启用硬门禁 |
| `GITHUB_TOKEN` | 自动提供 | 登录 GHCR，无需手动配 |

### 10.5 分支策略（真实远端分支）

`git branch -a` 实际看到的远端分支：

```
remotes/origin/main
remotes/origin/dev
remotes/origin/staging
remotes/origin/feature
remotes/origin/feature-app-api
remotes/origin/feature-ml-rui
remotes/origin/feature-model-v1-v2
remotes/origin/docs/report-reproducibility-rui
```

`DEPLOYMENT.md` 约定的流转（每一步都是"PR + CI 绿 + Merge"）：

```
个人特性分支 → feature → dev → staging → main
```

对应的自动化程度：

| 环节 | 跑什么 |
|---|---|
| 个人分支上改代码 | 只有 CI（若该分支名在 CI 的 push 列表内；否则靠 PR 触发） |
| 特性分支 → 特性分支 | 只有 CI |
| 特性分支 → `dev` | 只有 CI |
| `dev` → `staging` | **CI + CD**（staging 镜像，tag 含 `staging`） |
| `staging` → `main` | **CI + CD**（生产镜像，tag 含 `latest`） |

`DEPLOYMENT.md` 里也写了这个设计的现实约束：**两个 workflow 都会 `docker build`，而构建需要 `ml/registry/model.pkl`**。CI 这边的处理是"没有真实 pkl 就回落 FakeModel，测试照样能过"（软门禁）；要变成硬门禁，就把 `DVC_REMOTE_READY` 设为 `true`，让 CI 在构建前先 `dvc pull` 模型。

---

## 11. 测试体系（47 个用例）

本地实测：`pytest -q` → **47 passed, 1 warning in 8.09s**（warning 是 starlette 关于 httpx 的弃用提示，不影响结果）。

| 文件 | 用例数 | 测什么 | 典型用例 |
|---|---|---|---|
| `test_schema.py` | **29**（参数化） | 输入校验的边界 | 全缺字段 / 缺单个字段 / `age="abc"` / `age` 是对象 / `age=35.5` / `age=15` / `age=121` / `campaign=0` / `campaign=101` / `pdays=-1` / `pdays=1000` / `previous=-5` / 空字符串类别 / 纯空白类别 / 类别为 `null` / 类别是数字 / 多余字段 → 全部 **422**；边界值（`age=17`、`age=120`、`campaign=100`、`job` 长度正好 64）→ **200**；422 响应体必须是 JSON |
| `test_predict.py` | 6 | 正常路径 | `/health` 返回 ok；`/predict` 响应结构为 `{label, score}`；label 属于已知类别；字段类型与取值范围；**同一请求两次结果一致**（确定性）；OpenAPI 里暴露了 `/predict` |
| `test_edge_cases.py` | 8 | 异常与边界 | 未知类别（如 `astronaut`）不 5xx；Unicode 与特殊字符；极端但合法的数值；`score` 与契约一致；pkl 缺失时使用 FakeModel；FakeModel 输出形状；重置单例不崩；推理抛异常时返回 **500**（用 monkeypatch 注入故障） |
| `test_score_semantics.py` | 4 | **`score` 语义回归** | `score` 必须是 yes 的概率而不是预测类置信度；`classes_` 顺序颠倒时仍取对列；`classes_` 缺失时回落并打警告；FakeModel 路径给中性分 0.5 |

两个值得单独说的设计：

1. **`client` 夹具用 `with TestClient(app)`**，会真正走 startup/shutdown 生命周期，所以 `lifespan` 里的模型加载逻辑是被覆盖到的，不是"只测了青铜路径"。
2. **`test_score_semantics.py` 用打桩模型**（`_StubModel`）故意返回"反序的 `classes_`"和"无 `classes_`"的情况。这是把一个曾经踩过的坑（§7.4 的 max(row) 陷阱）转成永久的回归测试——比在文档里写一句"注意别忘了"有用得多。

---

## 12. Docker 镜像与部署细节

`Dockerfile`（42 行）的每个决定都有理由：

| 行 | 内容 | 为什么 |
|---|---|---|
| `FROM python:3.12-slim` | 与 CI 的 `3.12` 对齐 | 版本漂移会导致 pkl 反序列化失败或依赖解析差异 |
| `ENV PYTHONDONTWRITEBYTECODE=1` 等 | 不写 `__pycache__`、日志不缓冲、不缓存 pip 包 | 镜像更干净、日志实时可见 |
| `COPY requirements.txt` 先装依赖 | 依赖层单独成层 | 只改业务代码时不会重装依赖（构建时间从分钟级降到秒级） |
| `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120` | 清华镜像 + 长超时 | 规避国内直连 `files.pythonhosted.org` 超时 |
| `COPY app ./app`、`COPY ml ./ml` | 只拷运行时需要的东西 | 服务需要 `ml/data_pipeline.py`（`FEATURE_COLUMNS`）和 `ml/registry/model.pkl` |
| `COPY .env .` | 拷入环境变量文件 | **构建前提**：根目录必须存在 `.env`，否则构建直接失败（CI 里用 `printf` 造占位） |
| `useradd --uid 10001 appuser` + `USER appuser` | 非 root 运行 | 收敛容器逃逸面 |
| `HEALTHCHECK` | 请求 `/health` 并检查 `model_loaded` | 不只判"进程活着"，还判"模型真加载了"。pkl 缺失 → 回落 FakeModel → `model_loaded=false` → 容器 **unhealthy** |
| `CMD uvicorn app.main:app --host 0.0.0.0 --port 8000` | 启动服务 | 必须绑 `0.0.0.0`，否则容器外访问不到 |

`.dockerignore` 排除了 `.git`、`.venv`、`tests`、`mlruns/`、`mlflow.db`、`data/`、`ml/figures`、`ml/configs`、各 md 文档等——镜像里只留"推理真正需要的东西"（`app/` + `ml/` 里的代码与 `model.pkl`）。这也是为什么镜像不会因为有 780MB 的实验产物而膨胀。

> ⚠️ 一个真实的坑：`.gitignore` 里有一行 `.dockerignore`，意思是**这个排除规则文件本身不会被推送到 GitHub**。所以新克隆仓库后 `docker build` 会把 `.venv/`、`mlruns/`、`data/` 等全部打进构建上下文（几十万文件），构建极慢甚至失败。见 §13 第 6 条。

---

## 13. 常见问题排查

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | `/health` 返回 `"model_loaded": false` | 真实 pkl 未就位，回落到 `FakeModel`（此时 `/predict` 一律返回 `label=no`、`score=0.5`） | `dvc pull ml/registry/model.pkl.dvc`，或 `python ml/train.py` 本地生成 |
| 2 | `python ml/train.py` 报"数据文件不存在" | `data/*.csv` 被 Git 忽略且没拉取 | `dvc pull`；或按 §9.6 从 UCI 手动放置并校验 md5 |
| 3 | `dvc pull` 报认证/403 错误 | 远端是课程 Azure 账户，SAS token 在 `.dvc/config.local`（不入库） | 有 token：`dvc remote modify --local myazure sas_token "<token>"`；无 token：走 §9.6 的替代路径 |
| 4 | 打开 `/docs` 一片空白 | 默认 Swagger 用国外 CDN | 本项目已改成自托管静态资源；如果你是改了 `docs_url` 或删了 `app/static/`，恢复 `main.py` 的自定义 `/docs` 路由 |
| 5 | `docker build` 报 `COPY .env: not found` | Dockerfile 会 `COPY .env`，而 `.env` 被 Git 忽略 | 在仓库根目录创建 `.env`（内容随意，如 `APP_ENV=local`）；CI 里由 `printf` 自动创建占位 |
| 6 | `docker build` 极慢 / 上下文几百 MB | `.dockerignore` 本身被 `.gitignore` 忽略，新克隆时不存在 | 按需重建 `.dockerignore`（内容见仓库历史或 §12 说明），或 `git update-index --no-skip-worktree` 后从本地副本恢复 |
| 7 | 容器 `unhealthy` | `HEALTHCHECK` 检查的不只是进程，还有 `model_loaded` | 确认镜像里真的带了 `ml/registry/model.pkl`（构建前先 `dvc pull`） |
| 8 | 本地起服务端口被占 | 8000 已被占用 | `uvicorn app.main:app --port 8001`；或换 Docker 映射端口 `-p 8002:8000` |
| 9 | 请求 `/predict` 返回 422 | 字段缺失/类型不对/超范围/多了字段（`extra="forbid"`） | 对照 §8.2 的字段表；尤其注意 `campaign` 最小 1、`pdays` 最大 999、类别字段不能是纯空白 |
| 10 | `pytest` 用例数不是 47 | 参数化用例被裁剪 / 测试文件缺失 | 确认 4 个测试文件都在；`pytest --collect-only -q` 末尾应显示 `47 tests collected` |
| 11 | `ruff format --check .` 报大量文件需要格式化 | 本地格式化风格与 CI 不一致 | 统一执行 `ruff format .` 再提交；注意 `ruff==0.17.1` 固定版本 |
| 12 | 想切换服务到 v2 模型对比 | 默认路径写死在 `ml/registry/model.pkl` | 训练时 `--model-path ml/registry/model_v2.pkl`，启动时 `$env:MODEL_PATH="ml/registry/model_v2.pkl"` 再起 uvicorn |
| 13 | MLflow 里看不到实验记录 | `mlflow.db` 与 `mlruns/` 都被 Git 忽略，克隆下来是空的 | 本地跑一次 `python ml/train.py` 生成；或直接读 `ml/REPORT.md` 里的指标表 |
| 14 | 改了 `ml/configs/train.yaml` 但训练行为没变 | `train.py` **不读取**这个 yaml（历史设计稿），参数写在脚本常量里 | 见 §15；要改测试集比例/种子，直接改 `train.py` 顶部的 `TEST_SIZE` / `RANDOM_STATE` |
| 15 | `/predict` 返回的 `score` 与 `label` 看起来矛盾（如 `label=no`，`score=0.9`） | `score` 是**正类概率**，不是预测类置信度 | 这是设计如此（§7.4）；若看到 `label=no` 且 `score` 很高，说明该模型认为"认购概率高"但仍在 0.5 以下判为 no，属正常 |

---

## 14. 项目亮点

从工程视角看，这个项目不只是"跑通了一条流水线"，有 8 个点值得单独拎出来：

1. **特征白名单 + 断言防泄漏**：`FEATURE_COLUMNS` 是 19 列的显式白名单，`duration` 从数据管线到 API 请求体全部不存在，再加一条 `assert` 保证数值/类别列集合与白名单一致。泄漏不是靠注释提醒，是靠代码结构挡住的。
2. **择优标准有业务解释**：选 ROC-AUC 而不是 Accuracy，理由直接对应"业务上按 score 排序打电话"，并且在报告里算了 Accuracy 的平凡基线 `1−π = 88.73%`，说明为什么它没有区分力。
3. **敢于公开"更好的结果没上线"**：v2 的 ROC-AUC 更高（0.8156 > 0.8141），但增益在噪声内、F1 反而下降，于是**不上生产**。这是一种工程判断力，比"挑最高分"更有说服力。
4. **`score` 语义有 4 个回归测试守着**：把 `predict_proba` 取 `max(row)` 这个经典陷阱，用打桩模型的 4 个用例永久钉死。
5. **CI 门禁"软硬可切"**：`DVC_REMOTE_READY` 一个仓库变量，就把 `model-gate` 从"只警告"切成"硬拦截"。远端没配好时不会让所有人的 PR 全红。
6. **CD 验证的是"推上去的那个镜像"**：不只 build，而是 `docker run` 拉起来打 `/health`。这拦截的是本地能跑、仓库里的镜像跑不了的情况。
7. **服务端不重复实现预处理**：预处理 + 分类器作为一个 Pipeline 整体序列化，`joblib.load` 一个对象就够，`app` 侧只剩"按固定列序构造 DataFrame + 取正类概率"两件事。
8. **环境适配的细节都落到了代码**：自托管 Swagger 静态资源（国内 CDN 打不开）、清华 PyPI 镜像 + 120s 超时（依赖装不上）、非 root 用户（安全）、健康检查校验 `model_loaded`（真模型 vs 占位模型可区分）。

---

## 15. 已知限制与历史遗留

如实说明仓库里存在的不一致与限制，避免后来者被误导：

| 项 | 实际情况 | 影响 |
|---|---|---|
| `ml/configs/train.yaml` | 是一份**早期实验设计稿**：3 个 run（`v1-baseline-logreg`、`v2-logreg-default-threshold`、`v2-logreg-tuned-threshold`）、`test_size: 0.3`、`validation_size: 0.2`、全部用逻辑回归、按 `precision_target: 0.6` 调阈值。**当前 `ml/train.py` 并不读取这个文件**（脚本里没有 yaml 加载逻辑），实际跑的是 2 模型 × 2 数据版本、8:2 切分 | 改这个 yaml 不会改变训练行为；要改实验配置请直接改 `train.py` 中的 `TEST_SIZE` / `RANDOM_STATE` / `candidate_models()` |
| `ml/figures/` 下 4 张图 | `cm_v2_tuned.png`（混淆矩阵，样本合计 12,357 ≈ 30% 测试集）、`pr_v1_baseline.png`（PR 曲线，AP 0.61）、`pr_v2_default.png`（AP 0.47）、`pr_v2_tuned.png`（AP 0.47，工作点阈值 0.46）。它们来自**上面那份早期 3-run 方案的那次实验**，不是当前 4-run 方案的产物 | 别把它们当作当前生产模型的评估图。当前脚本的混淆矩阵随 MLflow artifact 存在 `mlruns/<run_id>/artifacts/plots/` 下（但那部分被 Git 忽略，需要本地跑一次才有） |
| 没有 `dvc.yaml` / `dvc.lock` | 确认不存在。DVC 只做数据/模型的内容版本管理，没有定义 DVC pipeline 阶段 | 不能用 `dvc repro` 一键重跑"数据→训练→评估"；流程入口是 `python ml/train.py` |
| `app/schemas.py` 中 `score` 的字段描述 | 写的是"预测类别的置信度（0~1）"，而实际实现（`model_loader._proba_to_scores`）返回的是**正类 yes 的概率** | 文档与实现的措辞不一致，以 §7.4 的实现与 `tests/test_score_semantics.py` 为准（建议后续把描述改成"认购概率"） |
| `mlflow.db`、`mlruns/` | 被 `.gitignore` 忽略，仓库里没有实验记录文件 | 新克隆看不到历史 run；指标结论请以 `ml/REPORT.md`、`ml/README_experiments.md` 为准；想看界面需本地重跑训练 |
| `.dockerignore` | 被 `.gitignore` 忽略，不会推到远端 | 新克隆后 `docker build` 上下文会很大（见 §13 第 6 条） |
| `.env` | 只存在于本地，内容不入库；Dockerfile 强制 `COPY` 它 | 新克隆后必须先创建 `.env` 才能构建镜像 |
| 模型能力边界 | 正类 F1 约 0.38（阈值 0.5 下），说明精确率/召回率都不高 | 这是一个**排序工具**而非"自动决策器"；实际使用应按 `score` 排序，由业务方决定打多少个电话、阈值定在哪 |

---

## 16. 相关文档索引

| 文档 | 内容 |
|---|---|
| [`ml/REPORT.md`](ml/REPORT.md) | 359 行完整实验报告：数据、模型、指标公式（ROC-AUC 的 Mann–Whitney 形式、F1、成本阈值）、4 run 对比、泄漏分析、溯源与复现命令 |
| [`ml/README_experiments.md`](ml/README_experiments.md) | 4 个核心 run 的对比表 + "为什么 v2 更好却没上线"的推理 |
| [`ml/README_data.md`](ml/README_data.md) | 数据来源、md5、v1→v2 三项改动 |
| [`ml/PROVENANCE.md`](ml/PROVENANCE.md) | 中英双语溯源：哪个 commit + 哪份数据 = 哪个模型；含 `MODEL_PATH` 并排跑 v1/v2 的方法 |
| [`ml/registry/current_model.md`](ml/registry/current_model.md) | 一页纸说清"当前生产模型是什么、指标多少、为什么选它" |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | 分支流转、CI/CD 各自何时跑、`.env` 用 Secret 注入的代码片段、pkl 依赖的现实约束 |
| `How to test.txt` | 162 行现场演示手册（8 个部分：概览 → DVC → MLflow → lint → 测试 → API → Docker → CI/CD），每步带预期输出，适合答辩时照着做 |

---

## 附：30 秒验收清单

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q                                    # 47 passed
ruff check .                                 # All checks passed!
Test-Path data\bank-additional-full.csv      # True（dvc pull 过）
Test-Path ml\registry\model.pkl              # True（训练或 dvc pull 过）
Invoke-RestMethod http://127.0.0.1:8000/health   # model_loaded=True
```

以上各项都对，说明这条流水线在本机是完整可用的。
