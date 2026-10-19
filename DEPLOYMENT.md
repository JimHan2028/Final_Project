# Deployment notes (staging / prod)

## 1. Branches
other feature->feature->dev->staging->main

## 2. The flows(cd only runs when push)
update their own feature:only ci
others feature -> feature:only ci
feature ->dev:only ci
dev ->staging:ci and cd
staging ->main:ci and cd

## 3. make .env in code

```cd.yml
- name: Create staging .env from staging GitHub secret
  if: github.ref == 'refs/heads/staging'
  env:
    ENV_CONTENT: ${{ secrets.STAGING_ENV_CONTENT }}
  run: printf '%s\n' "$ENV_CONTENT" > .env
```

## 4. Notice

both CI and CD run `docker build`, and the build needs `ml/registry/model.pkl` (the app loads it at startup and the healthcheck checks it). In CI this means either committing a small model or training in the pipeline — for now the local build is the verified path; CI trains/tests without the real pkl (the app falls back to a placeholder model, so tests still pass).
