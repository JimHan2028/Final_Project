# Current prod model v1

| item            | value                                                                          |
|-----------------|--------------------------------------------------------------------------------|
| run / model     | `hgb_improved` (HistGradientBoostingClassifier), trained on data v1            |
| path            | `ml/registry/model.pkl`                                                        |
| data version    | v1 (`data/bank-additional-full.csv`, md5 `f6cb2c...`)                          |
| training commit | `a967252abb802f434b7c755f068f90d60e46735f`                                     |
| metrics         | accuracy 0.9025 / f1(yes) 0.3837 / **roc_auc 0.8141**                          |
| why this one    | highest ROC-AUC, most robust under class imbalance (see README_experiments.md) |