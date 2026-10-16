# Final Project MLOps
## CI/CD Workflow
### Code flow: commit → build → test → staging → prod
1. Developers implement features on individual `feature/*` branches.
2. When feature work is ready, open Pull Request targeting protected branch `dev`.
3. Any Pull Request to protected branches (`dev`, `staging`, `main`) triggers the CI Pipeline:
    - Execute linting, formatting and static‑code analysis (ruff, black).
    - Execute full application test suite (at least three meaningful tests with multiple edge‑cases).
    - Build Docker image and verify there are no runtime build errors.
    - A pull request cannot be merged if CI pipeline fails.
4. After CI passes, merge PR into `dev`. Once features in `dev` are stable, create Pull Request from `dev` to `staging`. CI pipeline runs again.
5. After PR merged into `staging`, CD pipeline is triggered: generate staging `.env` file from GitHub secret `STAGING_ENV_CONTENT`, then build staging Docker image. No release process is performed.
6. Create Pull Request from `staging` to `main`. CI pipeline runs.
7. After PR merged into `main`, CD pipeline is triggered: generate production `.env` file from GitHub secret `PROD_ENV_CONTENT`, then build production Docker image. No deployment process is performed.

### Pipeline trigger rules
1. CI Pipeline (`/.github/workflows/ci.yml`): Triggered on Pull Request events for branches: `dev`, `staging`, `main`.
2. CD Build Pipeline (`/.github/workflows/cd.yml`): Triggered on push‑merge events for branches: `staging`, `main`.
