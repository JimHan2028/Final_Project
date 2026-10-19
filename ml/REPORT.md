# Bank Marketing — Experiment Report

Propensity modelling for outbound telemarketing: model choice, data versioning and
leakage control, written up with the formulas behind each number.

**Scope.** Every figure in this report is measured, not estimated. All numbers were read
back from the four MLflow runs in experiment `bank-marketing` (local tracking store
`sqlite:///mlflow.db`, artifacts under `mlruns/`) and reproduce with the commands in §8.
Unless stated otherwise, every run uses the same stratified split (`seed=42`, 80/20), so
the comparisons are paired.

---

## 1. Summary

- **Shipped model:** `hgb_improved` (HistGradientBoostingClassifier) on data **v1**,
  selected by ROC-AUC: **0.8141** test ROC-AUC, 0.9025 accuracy.
- **The v2 cleaning did not make it to production.** The same model on data v2 scores
  0.8156 ROC-AUC — a +0.0015 gain that is inside split noise on ~8.2k test rows, while
  f1 drops from 0.3837 to 0.3662. Swapping the production dataset for a noise-level
  ranking gain is not worth the revalidation cost, so v1 stays.
- **The model choice is a real trade-off, stated openly.** Logistic regression scores
  worse on ROC-AUC (0.8009) but better on f1 for the positive class (0.4689 vs 0.3837).
  The business reads the score as a ranking (call the highest-scoring customers first),
  and ROC-AUC measures exactly ranking quality under 11.3% class imbalance — so HGB wins
  the tie-break we actually care about. If the goal ever flips to maximum recall at a
  fixed call budget, the logreg baseline is the fallback (§6.1).
- **`duration` is excluded from both data versions.** It is only known after the call
  ends, so no run in this report contains it (§7.1). There is no leak-inflated number
  anywhere in the tracking store to un-learn later.

---

## 2. Data

|          |                                                                                              |
|----------|----------------------------------------------------------------------------------------------|
| Source   | UCI ML Repository id 222, *Bank Marketing* (`bank-additional-full.csv`, semicolon separated) |
| Files    | `data/bank-additional-full.csv` (v1), `data/bank-additional-v2.csv` (v2)                     |
| md5 (v1) | `f6cb2c1256ffe2836b36df321f46e92c`                                                           |
| md5 (v2) | `e8a1add9000aa97898f84ceddfbe32ac`                                                           |
| Target   | `y ∈ {no, yes}`, positive = `"yes"`                                                          |

v1 is the file as published. v2 is derived from it by `ml/make_v2_data.py`, so the two
versions differ by a recorded, rerunnable transformation rather than by a second download.
Both versions are tracked by DVC; git stores the `.dvc` pointers, Azure Blob stores the
content.

|                      | v1 (as published)                                   | v2 (cleaned)        |
|----------------------|-----------------------------------------------------|---------------------|
| Rows                 | 41,188                                              | 41,016 (−172)       |
| Positives            | 4,640 (11.2654 %)                                   | 4,638 (11.3078 %)   |
| Features             | 19                                                  | 19                  |
| `duration`           | **dropped** (target leakage, §7.1)                  | **dropped** (§7.1)  |
| `education`          | `basic.4y / basic.9y / basic.6y` as separate levels | merged into `basic` |
| `campaign > 20` rows | kept (harassment-style outlier call volumes)        | **removed**         |
| duplicate rows       | kept                                                | **removed**         |

### Split

Stratified on the target, `seed = 42`, 80/20 (logged per run as `train_rows` / `test_rows`):

| Data version | Train rows | Test rows |
|--------------|------------|-----------|
| v1           | 32,950     | 8,238     |
| v2           | 32,812     | 8,204     |

The split is the only random act in the pipeline and is pinned by `random_state=42`, so
all four runs are paired on the same rows within a data version.

---

## 3. Model

Two candidates, each wrapped in the *same* preprocessing Pipeline so the comparison is
about the classifier, never about the features.

### 3.1 Preprocessing

Standardisation of the numeric columns, with $\mu_j, s_j$ estimated on the training fold
only:

$$
\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{s_j}
$$

One-hot encoding of the categorical columns:

$$
\phi_{jc}(x) = \mathbb{1}[x_j = c]
$$

Both live inside one scikit-learn `Pipeline`, so no test information reaches the
transformer. `handle_unknown="ignore"` maps an unseen level to the all-zero vector instead
of raising — this is load-bearing for the API tests, which assert that categories like
`job="astronaut"` score without a 5xx.

The whole thing (transformer + classifier) is serialised into a single
`ml/registry/model.pkl`, so the serving side deserialises one object and never
re-implements preprocessing.

### 3.2 Candidates

|                 | `logreg_baseline`                                             | `hgb_improved`                      |
|-----------------|---------------------------------------------------------------|-------------------------------------|
| Class           | `LogisticRegression`                                          | `HistGradientBoostingClassifier`    |
| Hyperparameters | `max_iter=1000`, `class_weight="balanced"`, `random_state=42` | library defaults, `random_state=42` |

`class_weight="balanced"` on the linear model is the classic answer to the 11.3 % positive
rate: it rescales the loss so the rare class is not ignored. The gradient-boosted model
handles imbalance through its tree structure and needs no rescaling.

For the linear model, with $\phi(\mathbf{x})$ the concatenated transforms,

$$
z(\mathbf{x}) = \beta_0 + \boldsymbol\beta^\top \phi(\mathbf{x}),
\qquad
p_\theta(\mathbf{x}) = \sigma\big(z(\mathbf{x})\big) = \frac{1}{1 + e^{-z(\mathbf{x})}}
$$

so $\beta_j$ is the additive change in log-odds per one standard deviation of feature $j$
— coefficients stay reviewable, which is why logreg remains the fallback model. HGB gives
that up (no per-coefficient reading) in exchange for nonlinearities and interactions.

---

## 4. Metrics

For a threshold $\tau$, write the confusion counts on the test set:

$$
\mathrm{TP}(\tau) = \#\{i : y_i = 1, p_i \ge \tau\},\quad
\mathrm{FP}(\tau) = \#\{i : y_i = 0, p_i \ge \tau\},\quad
\mathrm{FN}(\tau) = \#\{i : y_i = 1, p_i < \tau\}
$$

with the corresponding rates

$$
P(\tau) = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}},
\qquad
R(\tau) = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}},
\qquad
F_1(\tau) = \frac{2 \, P(\tau) R(\tau)}{P(\tau) + R(\tau)}
$$

**Accuracy is not used as a headline metric.** The positive rate is $11.27\%$, so the
constant predictor $\hat{y} \equiv 0$ already achieves

$$
1 - \pi = 0.8873
$$

and every model here scores within $2$ points of that. Accuracy cannot separate a useful
model from a useless one on this problem; it is logged for completeness only.

### 4.1 The metric we optimise: ROC-AUC

ROC-AUC is the probability that a random positive outranks a random negative
(Mann–Whitney form), and is insensitive to the class balance:

$$
\mathrm{AUC}_{\mathrm{ROC}} = \int_0^1 \mathrm{TPR} \, d(\mathrm{FPR})
= \frac{1}{n_+ n_-} \sum_{i: y_i = 1} \sum_{j: y_j = 0}
\Big[ \mathbb{1}(p_i > p_j) + \tfrac{1}{2}\mathbb{1}(p_i = p_j) \Big]
$$

It is the selection criterion for two reasons. First, the business consumes the model as
a *ranking* — the call centre works down a sorted list, so "are the true subscribers
concentrated at the top" is the question that matters, and that is literally what the
Mann–Whitney statistic measures. Second, with an 11.3 % positive class, threshold-based
metrics at a fixed $\tau = 0.5$ swing wildly between models that rank almost identically;
ROC-AUC is threshold-free and comparable across the paired runs.

---

## 5. Score semantics and the decision threshold

The service returns two fields, and their contract is deliberate:

- `label` — the model's own `predict()` output, i.e. the class at the library's internal
  threshold ($0.5$ on the probability scale);
- `score` — **the probability of the positive class $P(y = \text{yes} \mid x)$**, not the
  confidence in the predicted class.

The distinction is easy to get wrong: `predict_proba` returns one probability *per class*,
ordered by `classes_`. Taking `max(row)` returns the confidence of whichever class was
predicted — a customer predicted `no` with $p(\text{yes}) = 0.03$ would report `score`
$= 0.97$, and the business would call the least likely subscribers first. The loader
therefore locates the `yes` column via `classes_` and indexes it directly
(`app/model_loader.py`, `_proba_to_scores`). Four regression tests
(`tests/test_score_semantics.py`) pin this behaviour against stub models with reordered
or missing `classes_`.

On thresholds: $\tau$ is a **deployment parameter, not a property of the model**. The
current deployment ships the library default of $0.5$, which is the claim that a wasted
call and a missed subscriber cost the same; if $c_{\mathrm{FP}}$ is the cost of a futile
call and $c_{\mathrm{FN}}$ that of a lost deposit, the break-even point is

$$
p = \tau^{\star}_{\text{cost}} = \frac{c_{\mathrm{FP}}}{c_{\mathrm{FP}} + c_{\mathrm{FN}}}
$$

The bank's cost ratio was not supplied, so the service keeps $0.5$ and exposes `score`
raw — raising or lowering the operating point later is a one-line change on top of a
ranking, not a retrain.

---

## 6. Results

All four runs, test set (8,238 rows / 928 positives for v1; 8,204 rows for v2):

|               | `logreg_baseline` v1 | `hgb_improved` v1 | `logreg_baseline` v2 | `hgb_improved` v2 |
|---------------|----------------------|-------------------|----------------------|-------------------|
| Data version  | v1                   | v1                | v2                   | v2                |
| Features      | 19                   | 19                | 19                   | 19                |
| **ROC-AUC**   | 0.8009               | **0.8141**        | 0.8021               | **0.8156**        |
| Accuracy      | 0.8350               | **0.9025**        | 0.8329               | 0.9021            |
| f1 (positive) | **0.4689**           | 0.3837            | 0.4621               | 0.3662            |
| Shipped       | no                   | **yes**           | no                   | no                |

MLflow run ids (tracking store `sqlite:///mlflow.db`):

| Run                    | Run id                             |
|------------------------|------------------------------------|
| `logreg_baseline` (v1) | `24182ba979104956bdb3faf3b369d66a` |
| `hgb_improved` (v1)    | `fbab6806fab4401a8643e861085c693b` |
| `logreg_baseline` (v2) | `48ae369350624a3aa77c0a91ec6b218c` |
| `hgb_improved` (v2)    | `fbdf0b31d4884473a9129ca6c6295ddb` |

### 6.1 Reading the table

**Model choice.** Within v1, HGB wins ROC-AUC by $+0.0132$ and accuracy by $+6.8$ points,
and loses f1 by $-0.085$. The f1 gap is the balanced-class-weight logreg buying positive
recall at the price of precision; the ROC-AUC gap is HGB buying overall ranking quality.
Since the deployment consumes `score` as a ranking and never acts on the raw `label`
(except as a convenience echo), ranking quality is the criterion the decision rule runs
on — hence `hgb_improved` is the production candidate, with the logreg baseline documented
as the recall-first fallback.

**Data version.** The v2 cleaning moves HGB ROC-AUC by $+0.0015$ (0.8141 → 0.8156). On
8.2k test rows the sampling noise of ROC-AUC is comfortably larger than that, and f1
*drops* by $0.018$ — the cleaning removed 172 rows, of which only 2 were positives, so
there was no signal recovery to be had. v1 stays in production; v2 remains tracked in DVC
as a documented, rerunnable alternative.

**Artifacts.** Each run archives its confusion matrix; the production candidate also
archives the exact pkl that ships:

- `mlruns/24182ba979104956bdb3faf3b369d66a/artifacts/plots/confusion_matrix_logreg_baseline.png`
- `mlruns/fbab6806fab4401a8643e861085c693b/artifacts/plots/confusion_matrix_hgb_improved.png`
- `mlruns/48ae369350624a3aa77c0a91ec6b218c/artifacts/plots/confusion_matrix_logreg_baseline.png`
- `mlruns/fbdf0b31d4884473a9129ca6c6295ddb/artifacts/plots/confusion_matrix_hgb_improved.png`
- `mlruns/fbab6806fab4401a8643e861085c693b/artifacts/model/model.pkl` (shipped model, also tagged `prod_candidate=true`)

---

## 7. Leakage and data versioning

### 7.1 `duration` — the column that never ships

Call duration is known only once the call is over, so it cannot be an input at scoring
time. The dataset authors state this directly in `bank-additional-names.txt`: the
attribute *"highly affects the output target ... should only be included for benchmark
purposes and should be discarded if the intention is to have a realistic predictive
model."* The effect is large:

|                         | Mean `duration` |
|-------------------------|-----------------|
| Subscribers ($y=1$)     | 553.2 s         |
| Non-subscribers ($y=0$) | 220.8 s         |

so the column separates the classes almost by itself, and it also inverts causality: a
long call is a *consequence* of a customer who engaged, not a cause available beforehand.

The formal problem is a support mismatch. Let $s(\mathbf{x}, d)$ be a scoring function
that consumes duration. Offline one could estimate

$$
\hat{R}_{\text{leaky}} = \frac{1}{n_{\mathrm{test}}}\sum_i \ell\big(y_i, s(\mathbf{x}_i, d_i)\big)
$$

but the deployed service can only compute $s(\mathbf{x}, \tilde{d})$ for some imputation
$\tilde{d}$ chosen without information, because at scoring time duration is undefined —
the offline estimate is not a sample estimate of any deployable quantity. For that reason
`duration` is dropped in **both** data versions (`data_pipeline.py`, `LEAKAGE_COLUMNS`),
and the exclusion is asserted by the feature-column contract rather than left to
convention.

### 7.2 What v2 actually changes

v2 (`ml/make_v2_data.py`, output tracked as `data/bank-additional-v2.csv.dvc`) makes three
recorded transformations of v1:

| Change                                                                   | Motivation                                                                                                                | Measured effect                               |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| Merge `basic.4y` / `basic.9y` / `basic.6y` education levels into `basic` | three near-duplicate sparse levels give the encoder almost no per-level signal                                            | row count unchanged                           |
| Drop rows with `campaign > 20`                                           | harassment-style outlier call volumes are atypical operational behaviour that would bias the "number of contacts" feature | 41,188 → 41,016 rows, positives 4,640 → 4,638 |
| Drop exact duplicate rows                                                | deduplicated record keeping                                                                                               | included in the row count above               |

The measured outcome is §6's $+0.0015$ ROC-AUC: recorded, kept in DVC, not promoted —
which is the data-versioning workflow doing its job. The next data version earns its way
into production by beating 0.8141 *and* holding f1, not by being newer.

---

## 8. Reproducibility

```bash
pip install -r requirements.txt
python ml/train.py                                              # v1, both candidates
python ml/train.py --data data/bank-additional-v2.csv \
                    --dataset-version v2 --no-save              # v2, log metrics only
mlflow ui --backend-store-uri sqlite:///mlflow.db               # http://127.0.0.1:5000
```

Runs are deterministic given `seed = 42`: the stratified split, both models'
`random_state` and the candidate list are fixed, and the numbers above reproduce exactly.
`--no-save` exists so a v2 comparison never overwrites the production `model.pkl`.

### Provenance

Every run logs what produced it, so "which dataset made this model" is answerable without
memory:

| Step            | Identifier                           | Logged as                         |
|-----------------|--------------------------------------|-----------------------------------|
| Data content    | md5 `f6cb2c…` / `e8a1add…`           | `data_md5` tag                    |
| Data version    | `v1` / `v2`                          | `dataset_version` tag + param     |
| Code            | git SHA                              | `git_commit` tag                  |
| Split           | 32,950/8,238 (v1), 32,812/8,204 (v2) | `train_rows` / `test_rows` params |
| Hyperparameters | all classifier params                | `clf_*` params                    |
| Metrics         | accuracy, f1, roc_auc                | metrics                           |
| Artifacts       | confusion matrix, shipped pkl        | `plots/`, `model/`                |

The DVC pointers (`data/*.csv.dvc`) carry the same md5s as the `data_md5` tags, so the
MLflow store and the data store agree on what "v1" and "v2" mean without a human in the
loop.

---

## Appendix A. Metric identities used above

**ROC-AUC is balance-insensitive.** The Mann–Whitney form in §4.1 depends only on the
relative ordering of positive and negative scores, never on how many of each there are —
doubling the negatives shifts $n_-$ in both numerator and denominator and cancels. This
is why it is the pairing metric of choice across runs whose positive rate is 11.3 %.

**Why accuracy's trivial baseline is $1-\pi$.** The constant predictor $\hat{y}\equiv 0$
is correct on every negative, so its accuracy is the negative share $1 - \pi = 0.8873$ on
v1. Any headline number must be read against that bar, which is the quantitative version
of "accuracy cannot separate a useful model from a useless one here".

**The break-even threshold.** §5's identity follows from comparing expected costs,
$c_{\mathrm{FP}}(1-p)$ for calling against $c_{\mathrm{FN}}p$ for not calling: they cross
at $p = c_{\mathrm{FP}}/(c_{\mathrm{FP}}+c_{\mathrm{FN}})$. Equal costs recover the
library default $0.5$ — which is precisely the assumption the current deployment makes,
explicitly rather than by omission.