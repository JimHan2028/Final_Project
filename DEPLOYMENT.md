# 部署说明（staging / prod）

## 1. 分支与环境的映射

| 分支 | 环境 | GitHub Secrets | 产物 |
| --- | --- | --- | --- |
| staging | staging | `STAGING_*`（数据库地址、模型路径、日志级别等） | staging 镜像 |
| main | production | `PROD_*` | 生产镜像 |

## 2. 标准流程

1. **PR 阶段**：CI 运行 `ruff` + `pytest`（Docker 落地后追加 `docker build` 冒烟），
   任何一步失败禁止合并（branch protection 上要求 check 通过）。
2. **merge → staging**：GitHub Actions 用 staging 环境的 Secrets 渲染 staging `.env`，
   然后 `docker build -f Dockerfile -t bank-api:staging-${{ github.sha }} .`。
3. **merge → main**：同样流程改用 PROD Secrets 渲染生产 `.env`，构建生产镜像。
4. 按课程要求不做自动发布（no release process / no deployment process）。

## 3. `.env` 的生成方式（占位约定）

```yaml
# 伪代码：CI 中渲染 .env，Secrets 不落日志
- name: Create staging .env
  run: |
    cat > .env <<EOF
    APP_ENV=staging
    MODEL_PATH=${{ secrets.STAGING_MODEL_PATH }}
    LOG_LEVEL=${{ secrets.STAGING_LOG_LEVEL }}
    EOF
```

## 4. 当前状态

- [x] CI（lint + tests）：`.github/workflows/ci.yml` 已启用，PR 必须通过
- [ ] Dockerfile / `.dockerignore`：待完成（当前为空占位）
- [ ] staging / prod 构建与 `.env` 渲染 job：待 Docker 落地后按本文件第 2 节补齐
