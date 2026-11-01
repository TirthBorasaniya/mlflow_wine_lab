"""
Loads the red and white wine CSV files, cleans and deduplicates them, and builds stratified splits.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from wine_lab import config

RAW_FEATURE_COLUMNS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
    "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
]
FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + ["is_red"]
SPLIT_NAMES = ("train", "validation", "test")


def normalize_column_name(name: str) -> str:
    """Strip quotes and whitespace and replace spaces with underscores."""
    return name.strip().strip('"').strip().replace(" ", "_")


def read_wine_csv(csv_path) -> pd.DataFrame:
    """
    Read one wine CSV, detecting whether it is comma- or semicolon-separated.

    The course copy of the red file uses commas while the white file uses semicolons;
    detecting the separator means either format of either file loads correctly.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to a wine quality CSV with a header row.

    Returns
    -------
    df : pandas.DataFrame
        Rows of the file with normalized column names.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [normalize_column_name(c) for c in df.columns]
    expected_set = set(RAW_FEATURE_COLUMNS + [config.TARGET_COLUMN])
    missing_list = sorted(expected_set - set(df.columns))
    if missing_list:
        raise ValueError(f"{csv_path}: missing columns {missing_list}; found {list(df.columns)}")
    return df[RAW_FEATURE_COLUMNS + [config.TARGET_COLUMN]]


def load_wine_data(red_path=config.RED_PATH, white_path=config.WHITE_PATH) -> pd.DataFrame:
    """
    Load both files, add the is_red indicator, and combine them.

    Parameters
    ----------
    red_path : str or pathlib.Path
        Red wine CSV.
    white_path : str or pathlib.Path
        White wine CSV.

    Returns
    -------
    df : pandas.DataFrame
        All rows with FEATURE_COLUMNS and the original quality score (0 to 10).
    """
    red_df = read_wine_csv(red_path).assign(is_red=1)
    white_df = read_wine_csv(white_path).assign(is_red=0)
    df = pd.concat([red_df, white_df], ignore_index=True)
    if df.isna().any().any():
        raise ValueError("wine data contains missing values")
    return df[FEATURE_COLUMNS + [config.TARGET_COLUMN]]


def prepare_dataset(df: pd.DataFrame, threshold: int = config.HIGH_QUALITY_THRESHOLD):
    """
    Remove duplicate rows and convert the quality score into a binary label.

    Duplicates are removed before splitting; otherwise identical wines can land in both
    training and test data, which inflates every reported metric.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of load_wine_data.
    threshold : int
        Scores at or above this value are labeled high quality (1).

    Returns
    -------
    clean_df : pandas.DataFrame
        Deduplicated rows with a binary quality column.
    report_dict : dict
        rows_raw, duplicates_removed, rows_clean, and positive_rate.
    """
    clean_df = df.drop_duplicates().reset_index(drop=True)
    clean_df[config.TARGET_COLUMN] = (clean_df[config.TARGET_COLUMN] >= threshold).astype(int)
    report_dict = {
        "rows_raw": len(df),
        "duplicates_removed": len(df) - len(clean_df),
        "rows_clean": len(clean_df),
        "positive_rate": float(clean_df[config.TARGET_COLUMN].mean()),
    }
    return clean_df, report_dict


def split_dataset(df: pd.DataFrame, train_fraction: float = config.TRAIN_FRACTION,
                  random_state: int = config.RANDOM_STATE) -> dict:
    """
    Split into train, validation, and test sets, stratified by the label.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of prepare_dataset.
    train_fraction : float
        Share of rows used for training; the rest is split evenly into validation and test.
    random_state : int
        Seed for reproducible splits.

    Returns
    -------
    split_dict : dict of str to tuple
        Maps "train", "validation", and "test" to (X, y), where X has FEATURE_COLUMNS.
    """
    X = df[FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=train_fraction, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=random_state
    )
    return {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}
