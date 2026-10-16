# Deployment documentation
This project uses GitHub Actions for CI and CD, no live runtime container deployment is required for this assignment.

## CI behaviour
All pull requests against protected branches dev, staging, main run CI checks: static analysis, full test suite, Docker build validation. Pull request merge is blocked if CI fails.

## CD behaviour
1. Merge commit to `staging` branch:
- Read repository secret `STAGING_ENV_CONTENT` to generate local `.env` file.
- Build Docker image tagged as `mlops-app:staging`.
- No release or live deployment.

2. Merge commit to `main` branch:
- Read repository secret `PROD_ENV_CONTENT` to generate local `.env` file.
- Build Docker image tagged as `mlops-app:prod`.
- No release or live deployment.

## Branch merge path
feature/* → dev → staging → main
