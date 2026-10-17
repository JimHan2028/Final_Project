"""Dataset construction for the bank-marketing project.

Both versions come from the same published file, so v1 and v2 differ only in how
the raw columns are interpreted (the reasoning is in ml/README_data.md):

v1  As published - 20 features, `duration` kept, `unknown` kept as an ordinary
    category, `pdays` kept as a raw count.
v2  Cleaned - `duration` removed (target leakage), the `pdays` sentinel 999 turned
    into a binary `never_contacted` flag, and `unknown` mapped to a genuine
    missing value so it is encoded as its own column instead of being folded
    into the reference level.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"

# Not observable before the call is dialled, and the dataset authors say it
# should be dropped for a realistic model.
LEAKY_COLUMNS = ("duration",)

PLACEHOLDER = "unknown"
PLACEHOLDER_COLUMNS = ("job", "marital", "education", "default", "housing", "loan")

NEVER_CONTACTED_SENTINEL = 999
NEVER_CONTACTED_COLUMN = "never_contacted"

DatasetVersion = Literal["v1", "v2"]
DATASET_VERSIONS: tuple[DatasetVersion, ...] = ("v1", "v2")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the published CSV. It is semicolon separated, not comma separated."""
    return pd.read_csv(Path(path), sep=";")


def extract_target(df: pd.DataFrame) -> pd.Series:
    return (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)


def to_v1(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[TARGET_COLUMN])


def to_v2(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[TARGET_COLUMN, *LEAKY_COLUMNS]).copy()
    features[NEVER_CONTACTED_COLUMN] = (
        features["pdays"] == NEVER_CONTACTED_SENTINEL
    ).astype(int)
    features = features.drop(columns=["pdays"])
    for column in PLACEHOLDER_COLUMNS:
        features[column] = features[column].astype(object).replace(PLACEHOLDER, np.nan)
    return features


def build_features(df: pd.DataFrame, version: DatasetVersion) -> pd.DataFrame:
    if version == "v1":
        return to_v1(df)
    if version == "v2":
        return to_v2(df)
    raise ValueError(
        f"unknown dataset version {version!r}, expected one of {DATASET_VERSIONS}"
    )


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features, target, test_size=test_size, stratify=target, random_state=seed
    )


def load_dataset(
    version: DatasetVersion,
    path: str | Path,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test for the requested dataset version."""
    df = load_raw(path)
    return split_dataset(
        build_features(df, version), extract_target(df), test_size, seed
    )


def as_column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical for the ColumnTransformer."""
    numeric = features.select_dtypes("number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return numeric, categorical


def fingerprint(path: str | Path) -> str:
    """sha256 of the raw file, logged with every run to identify the data version."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
