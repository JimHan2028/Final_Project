# Data

## Source

| | |
|---|---|
| Dataset | Bank Marketing (with social/economic context), UCI ML Repository id 222 |
| Download | `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip` |
| File used | `bank-additional/bank-additional-full.csv`, 5,834,924 bytes |
| sha256 | `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` |
| Shape | 41,188 rows x 21 columns (20 features + target `y`) |
| Positive class | 4,640 / 41,188 = 11.27 % |
| Origin | Real telemarketing campaigns of a Portuguese retail bank, May 2008 - Nov 2010 |
| Citation | S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014. doi:10.1016/j.dss.2014.03.001 |
| Licence | Publicly available for research (stated in the bundled `bank-additional-names.txt`) |

Format notes:

- The file is **semicolon separated**, not comma separated: `pd.read_csv(path, sep=";")`.
- The target `y` holds the strings `yes` / `no`.
- The file contains **no `NaN`s**. Missing values are written as the literal string `unknown`.
- `pdays` uses the sentinel `999` for "customer was never contacted before"; the column's median is 999.

The raw CSV is not committed to git. It is tracked with DVC and stored in the Azure remote;
git only holds the `.dvc` pointer.

## Versions

Both versions are derived from the same raw file by `ml/data_pipeline.py`, so the difference
between them is a code change rather than a second download.

### v1 - as published

`to_v1()` drops the target column and nothing else: 20 features, `duration` kept, `unknown`
kept as an ordinary category, `pdays` kept as a raw number. This is the honest starting
point: whatever a first pass over the published file would give you.

### v2 - cleaned

Three changes, each with a reason and a measured effect:

1. **Dropped `duration` (target leakage).** Call duration is only known once the call has
   finished, so it cannot be an input when scoring. The dataset authors state this in
   `bank-additional-names.txt`: *"this attribute highly affects the output target ... should
   only be included for benchmark purposes and should be discarded if the intention is to
   have a realistic predictive model."* Measured effect: mean duration is 553 s for
   subscribers vs 221 s for non-subscribers; keeping the column lifts PR-AUC from 0.468 to
   0.606 while making the model unusable in production.
2. **`pdays` -> `never_contacted`.** Feeding 999 as a quantity tells the model "contacted
   999 days ago", which is not what the value means. v2 keeps the binary signal and drops
   the raw column.
3. **`unknown` -> missing.** `default` (8,597 rows, 20.9 %), `education` (1,731),
   `housing` and `loan` (990 each), `job` (330) and `marital` (80) use `unknown` as a
   placeholder for "customer did not disclose". v1 turns that into an ordinary one-hot
   level, so the model reads non-disclosure as a category. v2 maps it to a real missing
   value, which `OneHotEncoder` encodes as its own column while leaving the disclosed
   levels untouched.

| | v1 | v2 |
|---|---|---|
| Features | 20 | 19 |
| `duration` | kept | dropped |
| `pdays` | raw 0-999 | replaced by `never_contacted` |
| `unknown` | ordinary category | missing value |
| PR-AUC, LogReg, threshold 0.5 | 0.606 | 0.468 |

## Reproduce

```bash
dvc pull                                   # fetch the raw csv from the Azure remote
python ml/train.py --run baseline          # trains on data v1
python ml/train.py --run v2_default        # trains on data v2, default threshold
python ml/train.py --run production        # trains on data v2, tuned threshold
```

Every run logs the sha256 of the raw file (`dataset_sha256`) and the DVC-tracked hash
(`dvc_dataset_hash`), which is what lets you answer "which exact dataset version produced
this model?" without guessing.

## Code to data mapping

One row per dataset version. Fill in each time `data.dvc` changes; the values come from
`dvc add` and `git rev-parse HEAD` at the commit that introduced that version.

| Data version | `data.dvc` hash | Commit hash | MLflow run name |
|---|---|---|---|
| v1 | `TODO` | `TODO` | `v1-baseline-logreg` |
| v2 | `TODO` | `TODO` | `v2-logreg-tuned-threshold` |

Reference values for the metrics quoted above were measured locally with the commands in
this file; see `ml/README_experiments.md` for the full run table.
