"""
Data loading + preprocessing for the UCI Heart Disease dataset.

Your ZIP contains 'processed.cleveland.data' etc.
These files are comma-separated with 14 columns, where the last column 'num'
is 0..4 (disease severity). We convert it to binary:
    target = 1 if num > 0 else 0

Missing values are represented as '?' in the raw files, so we treat '?' as NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# Standard 14-column schema for processed.*.data files
COLUMNS = [
    "age",       # numeric
    "sex",       # categorical (0/1 but treat as category is fine)
    "cp",        # chest pain type (categorical 1-4)
    "trestbps",  # numeric
    "chol",      # numeric
    "fbs",       # categorical (0/1)
    "restecg",   # categorical (0-2)
    "thalach",   # numeric
    "exang",     # categorical (0/1)
    "oldpeak",   # numeric
    "slope",     # categorical (1-3)
    "ca",        # categorical-ish but numeric count (0-3); has missing '?'
    "thal",      # categorical (3,6,7); can have missing '?'
    "num",       # target raw (0..4)
]


@dataclass(frozen=True)
class DatasetPaths:
    """Centralize dataset paths to avoid hardcoding all over the repo."""
    zip_path: Path
    processed_dir: Path


def extract_zip_if_needed(paths: DatasetPaths) -> None:
    """
    Extracts the ZIP into processed_dir only if files are not already present.

    This keeps your pipeline idempotent: re-running won't re-extract unnecessarily.
    """
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    # If one known file exists, assume extracted already
    expected = paths.processed_dir / "processed.cleveland.data"
    if expected.exists():
        return

    with zipfile.ZipFile(paths.zip_path, "r") as zf:
        zf.extractall(paths.processed_dir)


def load_processed_dataset(paths: DatasetPaths, which: str = "cleveland") -> pd.DataFrame:
    """
    Loads one processed dataset file into a DataFrame.

    Parameters
    ----------
    which:
        One of: 'cleveland', 'hungarian', 'switzerland', 'va', or 'all'.
        'all' concatenates the four processed datasets.

    Returns
    -------
    DataFrame with columns COLUMNS plus a binary 'target' column, and without 'num'.
    """
    extract_zip_if_needed(paths)

    mapping = {
        "cleveland": ["processed.cleveland.data"],
        "hungarian": ["processed.hungarian.data"],
        "switzerland": ["processed.switzerland.data"],
        "va": ["processed.va.data"],
        "all": [
            "processed.cleveland.data",
            "processed.hungarian.data",
            "processed.switzerland.data",
            "processed.va.data",
        ],
    }
    if which not in mapping:
        raise ValueError(f"Invalid which='{which}'. Use one of {list(mapping.keys())}.")

    frames = []
    for fname in mapping[which]:
        fpath = paths.processed_dir / fname
        df = pd.read_csv(
            fpath,
            header=None,
            names=COLUMNS,
            na_values="?",           # '?' indicates missing
            engine="python",
        )
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Convert raw severity label (0..4) into binary risk label (0/1)
    data["target"] = (data["num"].astype(float) > 0).astype(int)

    # Drop original multi-class label
    data = data.drop(columns=["num"])

    return data


def build_preprocess_pipeline(df: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    """
    Builds a sklearn Pipeline that:
      - imputes missing values
      - scales numeric features
      - one-hot encodes categorical features

    Returns:
      pipeline, numeric_cols, categorical_cols
    """
    # Define which columns are numeric vs categorical.
    # Even though many are coded as numbers, several are actually categories.
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ["target"]]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    full_pipeline = Pipeline(steps=[("preprocess", preprocessor)])
    return full_pipeline, numeric_cols, categorical_cols


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits df into train/val/test using stratification on target.

    val_size is relative to the remaining train portion.
    Example: test_size=0.2 -> 80% remaining
             val_size=0.2 -> val becomes 16% overall (0.2 of 0.8)
    """
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["target"]
    )

    train, val = train_test_split(
        train_val, test_size=val_size, random_state=random_state, stratify=train_val["target"]
    )
    return train, val, test
