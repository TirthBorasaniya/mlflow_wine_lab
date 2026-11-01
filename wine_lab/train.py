"""
Trains a random forest baseline and an XGBoost hyperparameter search, tracking every run in MLflow.

Usage:
    python -m wine_lab.train --max-evals 30
"""

import argparse
import tempfile
import uuid
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from wine_lab import config
from wine_lab.data import FEATURE_COLUMNS, load_wine_data, prepare_dataset, split_dataset
from wine_lab.evaluation import (
    compute_metrics,
    save_confusion_matrix,
    save_feature_importance,
    save_roc_curve,
)
from wine_lab.tracking import log_split_inputs, setup_tracking

BASELINE_PARAM_DICT = {"n_estimators": 200, "min_samples_leaf": 2}
XGB_MAX_ROUNDS = 1000
XGB_EARLY_STOPPING_ROUNDS = 50
SEARCH_SPACE = {
    "max_depth": scope.int(hp.quniform("max_depth", 3, 12, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
}


# ---------------------------------------------------------------------------
# Logging one candidate
# ---------------------------------------------------------------------------

def log_candidate(model, split_dict: dict, family: str, pipeline_id: str) -> float:
    """
    Log validation metrics, diagnostic plots, and the model to the active MLflow run.

    Every candidate is scored on the same validation set and logged under the same
    artifact path, so candidates of any model family can be compared and registered alike.

    Parameters
    ----------
    model : object
        Fitted classifier with predict_proba and feature_importances_.
    split_dict : dict
        Output of split_dataset.
    family : str
        Model family tag, such as "random_forest" or "xgboost".
    pipeline_id : str
        Identifier of this training execution, used to find its candidates later.

    Returns
    -------
    val_auc : float
        Validation AUC of the candidate.
    """
    X_val, y_val = split_dict["validation"]
    val_prob = model.predict_proba(X_val)[:, 1]
    metric_dict = compute_metrics(y_val, val_prob)
    mlflow.log_metrics({f"val_{name}": value for name, value in metric_dict.items()})
    mlflow.set_tags({"candidate": "true", "model_family": family, "pipeline_id": pipeline_id})

    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_dir = Path(tmp_dir)
        save_roc_curve(y_val, val_prob, plot_dir / "roc_curve.png")
        save_confusion_matrix(y_val, val_prob, plot_dir / "confusion_matrix.png")
        save_feature_importance(model.feature_importances_, FEATURE_COLUMNS,
                                plot_dir / "feature_importance.png")
        mlflow.log_artifacts(str(plot_dir), artifact_path="plots")

    signature = infer_signature(X_val, model.predict_proba(X_val))
    mlflow.sklearn.log_model(
        model, artifact_path=config.MODEL_ARTIFACT_PATH, signature=signature,
        input_example=X_val.head(3), pyfunc_predict_fn="predict_proba",
    )
    return metric_dict["auc"]


# ---------------------------------------------------------------------------
# Baseline and hyperparameter search
# ---------------------------------------------------------------------------

def run_baseline(split_dict: dict, pipeline_id: str) -> float:
    """Train and log the random forest baseline; return its validation AUC."""
    X_train, y_train = split_dict["train"]
    with mlflow.start_run(run_name="baseline_random_forest"):
        model = RandomForestClassifier(
            **BASELINE_PARAM_DICT, random_state=config.RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_train, y_train)
        mlflow.log_params(BASELINE_PARAM_DICT)
        val_auc = log_candidate(model, split_dict, "random_forest", pipeline_id)
    print(f"baseline random forest: validation AUC {val_auc:.4f}")
    return val_auc


def make_objective(split_dict: dict, pipeline_id: str):
    """
    Build the Hyperopt objective that trains and logs one XGBoost model per call.

    Parameters
    ----------
    split_dict : dict
        Output of split_dataset.
    pipeline_id : str
        Identifier of this training execution.

    Returns
    -------
    objective : callable
        Takes a parameter dict and returns a Hyperopt result dict with loss = -AUC.
    """
    X_train, y_train = split_dict["train"]
    X_val, y_val = split_dict["validation"]

    def objective(param_dict):
        with mlflow.start_run(nested=True, run_name="xgboost_trial"):
            model = XGBClassifier(
                **param_dict, n_estimators=XGB_MAX_ROUNDS, objective="binary:logistic",
                eval_metric="auc", early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                random_state=config.RANDOM_STATE, n_jobs=-1,
            )
            # the validation set stops training early; the test set is never used here
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            mlflow.log_params({**param_dict, "best_iteration": model.best_iteration})
            val_auc = log_candidate(model, split_dict, "xgboost", pipeline_id)
        return {"loss": -val_auc, "status": STATUS_OK}

    return objective


def run_search(split_dict: dict, pipeline_id: str, max_evals: int) -> float:
    """
    Run the XGBoost search with each trial as a nested run under one parent run.

    Returns
    -------
    best_auc : float
        Best validation AUC found.
    """
    with mlflow.start_run(run_name="xgboost_search"):
        trials = Trials()
        best_param_dict = fmin(
            fn=make_objective(split_dict, pipeline_id), space=SEARCH_SPACE, algo=tpe.suggest,
            max_evals=max_evals, trials=trials,
            rstate=np.random.default_rng(config.RANDOM_STATE), show_progressbar=False,
        )
        best_auc = -min(trials.losses())
        mlflow.log_params({f"best_{name}": value for name, value in best_param_dict.items()})
        mlflow.log_params({"max_evals": max_evals})
        mlflow.log_metric("best_val_auc", best_auc)
        mlflow.set_tags({"pipeline_id": pipeline_id, "step": "search"})
    print(f"xgboost search ({max_evals} trials): best validation AUC {best_auc:.4f}")
    return best_auc


def run_training(max_evals: int = config.DEFAULT_MAX_EVALS) -> str:
    """
    Load data, log the dataset, and train every candidate.

    Parameters
    ----------
    max_evals : int
        Number of XGBoost trials.

    Returns
    -------
    pipeline_id : str
        Identifier tagged on every candidate run of this execution.
    """
    setup_tracking()
    pipeline_id = uuid.uuid4().hex[:12]
    clean_df, report_dict = prepare_dataset(load_wine_data())
    split_dict = split_dataset(clean_df)
    print(f"data: {report_dict['rows_raw']} rows, {report_dict['duplicates_removed']} "
          f"duplicates removed, {report_dict['rows_clean']} used")

    with mlflow.start_run(run_name="data_preparation"):
        mlflow.log_metrics(report_dict)
        mlflow.log_params({"high_quality_threshold": config.HIGH_QUALITY_THRESHOLD,
                           "train_fraction": config.TRAIN_FRACTION,
                           "random_state": config.RANDOM_STATE})
        mlflow.set_tags({"pipeline_id": pipeline_id, "step": "data"})
        log_split_inputs(split_dict, source=str(config.DATA_DIR))

    # unlabeled test features become the input for the batch inference step
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    split_dict["test"][0].to_csv(config.BATCH_INPUT_PATH, index=False)

    run_baseline(split_dict, pipeline_id)
    run_search(split_dict, pipeline_id, max_evals)
    return pipeline_id


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train and track wine quality models")
    parser.add_argument("--max-evals", type=int, default=config.DEFAULT_MAX_EVALS)
    args = parser.parse_args(argv)
    pipeline_id = run_training(args.max_evals)
    print(f"training complete, pipeline id {pipeline_id}")


if __name__ == "__main__":
    main()
