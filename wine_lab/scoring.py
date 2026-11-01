"""
Batch scoring of wine measurements with a fitted classifier.
"""

import pandas as pd

from wine_lab.data import FEATURE_COLUMNS
from wine_lab.evaluation import DECISION_THRESHOLD


def score_batch(model, input_df: pd.DataFrame, threshold: float = DECISION_THRESHOLD):
    """
    Add predicted probabilities and labels to a batch of wines.

    Parameters
    ----------
    model : object
        Fitted classifier with predict_proba.
    input_df : pandas.DataFrame
        Rows containing at least FEATURE_COLUMNS; extra columns are kept unchanged.
    threshold : float
        Probability at or above which a wine is labeled high quality.

    Returns
    -------
    scored_df : pandas.DataFrame
        Copy of input_df with high_quality_probability and predicted_high_quality columns.
    """
    missing_list = [c for c in FEATURE_COLUMNS if c not in input_df.columns]
    if missing_list:
        raise ValueError(f"input is missing columns {missing_list}")
    if input_df.empty:
        raise ValueError("input contains no rows")

    probability = model.predict_proba(input_df[FEATURE_COLUMNS])[:, 1]
    scored_df = input_df.copy()
    scored_df["high_quality_probability"] = probability
    scored_df["predicted_high_quality"] = (probability >= threshold).astype(int)
    return scored_df
