# Experiments

All runs are tracked in MLflow (`mlflow.db`, experiment `bank-marketing`) and produced by
`ml/train.py`. Both models are **logistic regression** (a linear model); the runs differ in
the dataset version and in how the decision threshold is chosen.

## Setup

- Split: 70 / 30 stratified on the target (`seed=42`), 12,357 test rows, 11.26 % positive.
- Threshold selection happens on a validation split carved out of the training set
  (20 % of it), never on the test set. The model is then refitted on the full training set.
- Preprocessing: `StandardScaler` on numeric columns, `OneHotEncoder(handle_unknown="ignore")`
  on categorical columns, all inside one `sklearn` `Pipeline`.

## Runs

| Run | Data | Threshold | PR-AUC | ROC-AUC | Accuracy | Precision | Recall | Missed subscribers |
|---|---|---|---|---|---|---|---|---|
| `v1-baseline-logreg` | v1 | 0.50 fixed | **0.606** | 0.938 | 0.912 | 0.675 | 0.422 | 804 |
| `v2-logreg-default-threshold` | v2 | 0.50 fixed | 0.468 | 0.804 | 0.901 | 0.693 | 0.219 | 1,087 |
| `v2-logreg-tuned-threshold` | v2 | 0.457 (validation) | 0.468 | 0.804 | 0.900 | 0.651 | 0.246 | 1,050 |

## Which run is production-worthy, and why

`v2-logreg-tuned-threshold`. `v1-baseline-logreg` scores higher on every metric and is still
the wrong model to ship: its extra PR-AUC comes entirely from `duration`, a column that
does not exist until the call is over. The `0.606` is therefore not achievable in
production - it is a benchmark number, not a performance number. Shipping it would mean
being unable to score anybody, and the gap between its offline score and what it could
deliver is exactly the failure mode this project is meant to catch.

Within the deployable feature set (v2), the improvement comes from the decision threshold
rather than from the model:

| Operating point | Threshold | Precision | Recall | Calls made | Missed subscribers |
|---|---|---|---|---|---|
| `v2-logreg-default-threshold` | 0.500 | 0.693 | 0.219 | 440 | 1,087 |
| `v2-logreg-tuned-threshold` | 0.457 | 0.651 | **0.246** | 525 | **1,050** |

Same PR curve, same PR-AUC, 12 % more subscribers reached at a comparable precision, because
the threshold was chosen from the curve instead of left at the library default. This is also
why the two runs exist as separate MLflow entries: the difference is in the decision rule,
not in the model.

## The metric we optimise, and why it matters here

**PR-AUC (average precision)** on the positive class, reported together with the operating
point `recall @ precision >= 0.60`.

Why not accuracy: predicting `no` for everybody already scores 88.7 %, so accuracy cannot
distinguish a useful model from a useless one. The positive rate is 11.27 %, and PR-AUC's
baseline is exactly that number, so the metric is calibrated to how hard the problem is.

Why recall at a fixed precision, in business terms: the bank pays for every call and earns
only on conversions. Holding precision at 0.60 means at most 40 % of the calls are spent on
customers who will not subscribe - that is the "do not harass people" side of the cost -
and within that budget we maximise how many future subscribers we reach. Missing a
subscriber is a lost deposit; calling a non-subscriber is a wasted agent and a nuisance.
The precision target is the knob that trades those two off, and it maps directly onto
`DECISION_THRESHOLD` in the deployed service (see `.env.example`).

PR-AUC is threshold-free, so it measures ranking quality; the threshold is then chosen to
express the business constraint. Optimising only PR-AUC would leave the actual decision
unspecified, and the decision is what the bank acts on.

## Threshold sensitivity

Thresholds below are all selected on the validation split, then evaluated on test:

| Precision constraint | Selected threshold | Test precision | Test recall | Test F1 | Calls |
|---|---|---|---|---|---|
| >= 0.70 | 0.600 | 0.748 | 0.170 | 0.277 | 317 |
| >= 0.60 (production setting) | 0.457 | 0.651 | 0.246 | 0.357 | 525 |
| >= 0.50 | 0.356 | 0.545 | 0.376 | 0.445 | 959 |
| default 0.500 | 0.500 | 0.693 | 0.219 | 0.333 | 440 |

The constraint is a business input, not a modelling choice: `>= 0.60` is the value the
service ships with, and it is configurable per environment.

A caution for whoever tunes this next: if the threshold is picked directly on the test set,
the best point looks like `threshold 0.41 -> precision 0.600, recall 0.301`. That number is
not reproducible in our pipeline and is not what we report, because selecting on test is
how you overfit a decision rule. The validation-selected threshold gives 0.246 instead.

## Ablations that did not help

Recorded so the next person does not re-run them:

| Change | PR-AUC | Verdict |
|---|---|---|
| v2, `class_weight="balanced"` | 0.461 | rejected, no gain over 0.468 and worse recall at the operating point |
| v2, `C` grid over {0.01, 0.1, 1, 10} | best `C=1` | rejected, the default was already optimal |
| v2 + feature engineering (cyclical month, age/campaign bins, macro interactions) | 0.466 | rejected, no gain |
| v2 + `HistGradientBoostingClassifier` | 0.493 | rejected in favour of the linear model; recorded as the ceiling this task allows |

The linear model tops out at roughly 0.47 PR-AUC on the leak-free feature set. The gain that
is actually available lies in the data version and the decision rule, so that is where the
effort went. A tree model buys about +0.03 PR-AUC, which does not change any decision the
bank makes, and it costs the interpretability that makes the coefficients reviewable.

## Reproduce

```bash
python ml/train.py --run baseline     # v1 baseline, expected PR-AUC 0.606
python ml/train.py --run v2_default   # v2 with the default threshold, PR-AUC 0.468
python ml/train.py --run production   # v2 with the tuned threshold, PR-AUC 0.468
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Runs are deterministic given `seed=42`; the numbers above reproduce exactly.
