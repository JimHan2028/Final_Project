# Current model

Single source of truth for what the deployed service serves. Update this file whenever
`MODEL_PATH` or `DECISION_THRESHOLD` changes.

| | |
|---|---|
| Status | production |
| MLflow run | `v2-logreg-tuned-threshold` (experiment `bank-marketing`) |
| Model | `sklearn` `Pipeline`: `StandardScaler` + `OneHotEncoder` -> `LogisticRegression(C=1.0, max_iter=5000)` |
| Data version | v2 (`duration` dropped, `pdays` -> `never_contacted`, `unknown` -> missing) |
| Decision threshold | 0.457, selected on the validation split to satisfy `precision >= 0.60` |
| Test PR-AUC | 0.468 |
| Test precision / recall at the threshold | 0.651 / 0.246 |
| Artifact | `model/` directory of the MLflow run (`mlflow.sklearn` flavour, `MLmodel` + `model.pkl`) |

The artifact is exported into the deployment environments by `dvc pull` for data and by
downloading the MLflow run artifact for the model; the service reads both through
`MODEL_PATH`, so the container never contains a baked-in model copy.

## Why this one

`v1-baseline-logreg` has the better offline score (PR-AUC 0.606) and is not shippable: it
consumes `duration`, which is only known after the call ends. `v2-logreg-default-threshold`
is the same model as this one with the library default threshold (0.5), which reaches only
21.9 % of subscribers at 0.693 precision. The production run keeps the precision guarantee
and reaches 24.6 %. Full reasoning in `ml/README_experiments.md`.

## How to roll back

1. Pick the previous run in MLflow and download its `model/` artifact.
2. Point `MODEL_PATH` at it and restore the matching `MODEL_VERSION` value in the
   environment's secrets.
3. Rebuild and redeploy staging, verify `POST /predict` against the smoke sample in
   `app/`, then promote to production.

Nothing else in the service is model-specific, so a rollback is a config change plus an
image rebuild - no code change.
