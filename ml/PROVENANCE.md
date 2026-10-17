# Provenance

The chain that answers "which exact dataset version produced this model?".

```
git commit  ->  data.dvc (hash)  ->  raw csv (sha256)  ->  MLflow run  ->  model artifact  ->  deployed .env
```

## Links

| Step | Identified by | Where it is recorded |
|---|---|---|
| Code | git commit SHA | `git_sha` tag on every MLflow run; `ml/train.py` and `ml/data_pipeline.py` are also logged as run artifacts under `code/` |
| Data (pointer) | md5 in `data.dvc` | `dvc_dataset_hash` tag on every MLflow run |
| Data (content) | `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` | `dataset_sha256` tag, plus `ml/README_data.md` |
| Experiment | MLflow run id | MLflow tracking store, experiment `bank-marketing` |
| Model | `model/` artifact of that run | MLflow artifact store |
| Serving config | `MODEL_VERSION` + `DECISION_THRESHOLD` | environment secrets, `.env` per environment, mirrored in `ml/registry/current_model.md` |

## Rules

- The raw CSV never enters git. `.dvc` pointers and `data/` are handled by DVC; the file
  itself lives in the Azure remote.
- Every training run logs the git SHA, the DVC hash and the sha256 of the raw file, so a
  model can always be traced back without relying on memory or on file names.
- `ml/registry/current_model.md` states which run is currently served. If it disagrees with
  the deployed `MODEL_VERSION`, the deployment is wrong, not the file.
