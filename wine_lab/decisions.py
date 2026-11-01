"""
Promotion rule deciding whether a newly registered model replaces the current champion.
"""

import math


def decide_promotion(challenger_auc: float, champion_auc, min_improvement: float):
    """
    Decide whether the challenger should become the champion.

    Both models must have been scored on the same held-out test set.

    Parameters
    ----------
    challenger_auc : float
        Test AUC of the newly registered model.
    champion_auc : float or None
        Test AUC of the current champion, or None if there is no champion yet.
    min_improvement : float
        Required AUC gain, so retraining noise does not replace a model that is just as good.

    Returns
    -------
    o_promote : bool
        True if the challenger should be promoted.
    reason : str
        Short explanation recorded on the model version.
    """
    if not math.isfinite(challenger_auc):
        return False, "challenger AUC is not a finite number"
    if champion_auc is None:
        return True, "no current champion"
    gain = challenger_auc - champion_auc
    if gain >= min_improvement:
        return True, f"test AUC improved by {gain:.4f} (required {min_improvement})"
    return False, f"test AUC changed by {gain:+.4f}, below the required {min_improvement}"
