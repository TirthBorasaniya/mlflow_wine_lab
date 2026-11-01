"""
Tests for loading, cleaning, deduplicating, and splitting the wine data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wine_lab import config
from wine_lab.data import (
    FEATURE_COLUMNS,
    load_wine_data,
    normalize_column_name,
    prepare_dataset,
    read_wine_csv,
    split_dataset,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
RED_SAMPLE = FIXTURE_DIR / "red_sample.csv"
WHITE_SAMPLE = FIXTURE_DIR / "white_sample.csv"


def make_synthetic(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Build a labeled frame with the real column layout for split tests."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n_rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df["is_red"] = rng.integers(0, 2, n_rows)
    df[config.TARGET_COLUMN] = (rng.random(n_rows) < 0.2).astype(int)
    return df


@pytest.mark.parametrize(
    "raw, expected",
    [('"fixed acidity"', "fixed_acidity"), (" pH ", "pH"), ("alcohol", "alcohol")],
)
def test_normalize_column_name(raw, expected):
    assert normalize_column_name(raw) == expected


@pytest.mark.parametrize("path", [RED_SAMPLE, WHITE_SAMPLE])
def test_both_separators_are_detected(path):
    df = read_wine_csv(path)
    assert df.shape == (10, 12)
    assert df["alcohol"].dtype.kind == "f"


def test_missing_column_is_reported(tmp_path):
    bad_path = tmp_path / "bad.csv"
    bad_path.write_text("fixed acidity,alcohol,quality\n7.4,9.4,5\n7.8,9.8,5\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_wine_csv(bad_path)


def test_load_adds_color_indicator():
    df = load_wine_data(RED_SAMPLE, WHITE_SAMPLE)
    assert list(df.columns) == FEATURE_COLUMNS + [config.TARGET_COLUMN]
    assert df["is_red"].tolist() == [1] * 10 + [0] * 10


def test_prepare_removes_duplicates_and_labels():
    clean_df, report_dict = prepare_dataset(load_wine_data(RED_SAMPLE, WHITE_SAMPLE))
    # each fixture contains one exact duplicate row
    assert report_dict["duplicates_removed"] == 2
    assert report_dict["rows_clean"] == len(clean_df) == 18
    assert set(clean_df[config.TARGET_COLUMN]) == {0, 1}
    assert report_dict["positive_rate"] == pytest.approx(clean_df[config.TARGET_COLUMN].mean())


def test_prepare_threshold():
    df = load_wine_data(RED_SAMPLE, WHITE_SAMPLE)
    strict_df, _ = prepare_dataset(df, threshold=8)
    assert strict_df[config.TARGET_COLUMN].sum() == 4


def test_split_sizes_are_disjoint_and_stratified():
    df = make_synthetic()
    split_dict = split_dataset(df)
    size_list = [len(split_dict[name][0]) for name in ("train", "validation", "test")]
    assert size_list == [360, 120, 120]

    index_set_list = [set(split_dict[name][0].index) for name in split_dict]
    assert not (index_set_list[0] & index_set_list[1])
    assert not (index_set_list[0] & index_set_list[2])
    assert not (index_set_list[1] & index_set_list[2])

    overall_rate = df[config.TARGET_COLUMN].mean()
    for _, y in split_dict.values():
        assert y.mean() == pytest.approx(overall_rate, abs=0.03)


def test_split_is_reproducible():
    df = make_synthetic()
    first = split_dataset(df)["test"][0].index.tolist()
    second = split_dataset(df)["test"][0].index.tolist()
    assert first == second
