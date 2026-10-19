## Data

| item | value |
| --- | --- |
| files | `data/bank-additional-full.csv` (v1) and `data/bank-additional-v2.csv` (v2), both DVC-tracked |
| source | UCI Bank Marketing: <https://archive.ics.uci.edu/dataset/222/bank+marketing> |
| md5 | v1 `f6cb2c1256ffe2836b36df321f46e92c` / v2 `e8a1add9000aa97898f84ceddfbe32ac` |
| shape | 41,188 x 21 (v1) / 41,016 x 21 (v2); training drops `duration` and target `y` |

## Models

| item            | value                                                                                          |
|-----------------|------------------------------------------------------------------------------------------------|
| prod model      | `hgb_improved` (HistGradientBoostingClassifier), **trained on v1**                             |
| artifact        | `ml/registry/model.pkl` (preprocessing + classifier serialized as one Pipeline)                |
| training commit | `a967252abb802f434b7c755f068f90d60e46735f`                                                     |
| training env    | Python 3.12.10 / scikit-learn 1.9.1 / numpy 2.5.4 / pandas 3.0.6 / scipy 1.18.1 / joblib 1.6.0 |

The v2 experiment pair  did not go to prod — reasoning in [README_experiments.md](README_experiments.md).

## How to reproduce

```bash
dvc pull                        # course Azure remote (myazure), data comes back instantly
pip install -r requirements.txt
python ml/train.py              # rebuilds the v1 run pair + model.pkl
python ml/train.py --data data/bank-additional-v2.csv --dataset-version v2 --no-save   # v2 pair
```