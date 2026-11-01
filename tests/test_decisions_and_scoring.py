"""
Tests for the promotion rule and for batch scoring.
"""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from wine_lab.data import FEATURE_COLUMNS
from wine_lab.decisions import decide_promotion
from wine_lab.scoring import score_batch

MIN_GAIN = 0.002


@pytest.mark.parametrize(
    "challenger, champion, expected",
    [
        (0.85, None, True),
        (0.86, 0.85, True),
        (0.852, 0.85, True),
        (0.851, 0.85, False),
        (0.84, 0.85, False),
        (math.nan, None, False),
    ],
)
def test_decide_promotion(challenger, champion, expected):
    o_promote, reason = decide_promotion(challenger, champion, MIN_GAIN)
    assert o_promote is expected
    assert reason


def make_model_and_rows(n_rows=40):
    """Fit a small classifier on random data with the real feature columns."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(n_rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (X["alcohol"] > 0).astype(int)
    return LogisticRegression().fit(X, y), X


def test_score_batch_adds_columns_and_keeps_extras():
    model, X = make_model_and_rows()
    input_df = X.assign(wine_id=range(len(X)))
    scored_df = score_batch(model, input_df)
    assert len(scored_df) == len(X)
    assert scored_df["high_quality_probability"].between(0, 1).all()
    assert set(scored_df["predicted_high_quality"]) <= {0, 1}
    assert scored_df["wine_id"].tolist() == list(range(len(X)))
    assert "high_quality_probability" not in input_df


def test_score_batch_rejects_bad_input():
    model, X = make_model_and_rows()
    with pytest.raises(ValueError, match="missing columns"):
        score_batch(model, X.drop(columns=["alcohol"]))
    with pytest.raises(ValueError, match="no rows"):
        score_batch(model, X.iloc[0:0])
