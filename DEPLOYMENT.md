# Deployment notes (staging / prod)

## 1. Branch -> environment mapping

| branch | env | GitHub Secrets | output |
| --- | --- | --- | --- |
| staging | staging | `STAGING_*` (db url, model path, log level...) | staging image |
| main | production | `PROD_*` | prod image |

## 2. The flow

1. **PR stage**: CI runs `ruff` + `pytest` + `docker build` smoke test.
   Any failure blocks the merge (branch protection requires the check to pass).
2. **merge -> staging**: GitHub Actions renders the staging `.env` from staging Secrets,
   then builds the image (`cd.yml`, job `build_target_image`, tag `mlops-app:staging`).
3. **merge -> main**: same flow with PROD secrets -> prod image (`mlops-app:prod`).
4. Per the course requirement there's no actual release/deploy step (no release process requested) — building the image IS the deliverable.

## 3. How `.env` gets rendered

The real implementation lives in [.github/workflows/cd.yml](.github/workflows/cd.yml):

```yaml
- name: Create staging .env from staging GitHub secret
  if: github.ref == 'refs/heads/staging'
  env:
    ENV_CONTENT: ${{ secrets.STAGING_ENV_CONTENT }}
  run: printf '%s\n' "$ENV_CONTENT" > .env
```

One secret per env (`STAGING_ENV_CONTENT` / `PROD_ENV_CONTENT`) holds the whole file content — no individual keys in logs.

## 4. Status

- [x] CI (lint + tests): [.github/workflows/ci.yml](.github/workflows/ci.yml), required for merge
- [x] Dockerfile (slim python:3.12, non-root, healthcheck) + `.dockerignore` — pip uses the Tsinghua mirror inside the image build, official PyPI kept timing out
- [x] CD: [.github/workflows/cd.yml](.github/workflows/cd.yml) builds staging/prod images on merge, `.env` rendered from Secrets
- [ ] actual `docker push` to a registry: out of scope per course requirement

Note: both CI and CD run `docker build`, and the build needs `ml/registry/model.pkl` (the app loads it at startup and the healthcheck checks it). In CI this means either committing a small model or training in the pipeline — for now the local build is the verified path; CI trains/tests without the real pkl (the app falls back to a placeholder model, so tests still pass).

---

# 部署说明（staging / prod）

## 1. 分支与环境的映射

| 分支 | 环境 | GitHub Secrets | 产物 |
| --- | --- | --- | --- |
| staging | staging | `STAGING_*`（数据库地址、模型路径、日志级别等） | staging 镜像 |
| main | production | `PROD_*` | 生产镜像 |

## 2. 标准流程

1. **PR 阶段**：CI 跑 `ruff` + `pytest` + `docker build` 冒烟，
   任何一步失败禁止合并（branch protection 上要求 check 通过）。
2. **merge → staging**：GitHub Actions 用 staging 环境的 Secrets 渲染 staging `.env`，
   然后 `docker build -f Dockerfile -t bank-api:staging-${{ github.sha }} .`
   （真实实现见 `cd.yml`，job `build_target_image`，镜像 tag `mlops-app:staging`）。
3. **merge → main**：同样流程换 PROD Secrets 渲染生产 `.env`，构建生产镜像（`mlops-app:prod`）。
4. 按课程要求不做实际发布（no release process / no deployment process）——把镜像 build 出来就是交付。

## 3. `.env` 的生成方式

真实实现就在 [.github/workflows/cd.yml](.github/workflows/cd.yml) 里：

```yaml
- name: Create staging .env from staging GitHub secret
  if: github.ref == 'refs/heads/staging'
  env:
    ENV_CONTENT: ${{ secrets.STAGING_ENV_CONTENT }}
  run: printf '%s\n' "$ENV_CONTENT" > .env
```

每个环境一个 Secret（`STAGING_ENV_CONTENT` / `PROD_ENV_CONTENT`）放整个文件内容，日志里不会漏出单个 key。

## 4. 当前状态

- [x] CI（lint + tests）：`.github/workflows/ci.yml` 已启用，PR 必须通过
- [x] Dockerfile（slim python:3.12、非 root、healthcheck）+ `.dockerignore` —— 镜像内 pip 走清华源，官方源一直超时
- [x] CD：`.github/workflows/cd.yml` 在 merge 后构建 staging/prod 镜像，`.env` 由 Secrets 渲染
- [ ] 实际 `docker push` 到镜像仓库：课程不要求，没做

注：CI 和 CD 都会跑 `docker build`，而构建需要 `ml/registry/model.pkl`（服务启动要加载，healthcheck 也检查它）。
本地构建走的是已验证路径；CI 里没有真实 pkl 时应用会回落到占位模型，测试照样过。
