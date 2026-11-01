"""
Registers the best candidate as a challenger and promotes it to champion only if it beats
the current champion on the held-out test set.

Usage:
    python -m wine_lab.promote [--pipeline-id ID]
"""

import argparse

import mlflow
import mlflow.sklearn

from wine_lab import config
from wine_lab.data import load_wine_data, prepare_dataset, split_dataset
from wine_lab.decisions import decide_promotion
from wine_lab.evaluation import compute_metrics
from wine_lab.tracking import get_version_by_alias, setup_tracking


def find_best_candidate(pipeline_id=None):
    """
    Find the candidate run with the highest validation AUC.

    Only runs tagged as candidates of one pipeline execution are considered, so parent
    runs, data runs, and runs from earlier executions cannot be selected by mistake.

    Parameters
    ----------
    pipeline_id : str or None
        Execution to search; defaults to the most recent one.

    Returns
    -------
    best_run : pandas.Series
        Row from mlflow.search_runs for the best candidate.
    """
    candidate_filter = "tags.candidate = 'true'"
    if pipeline_id is None:
        latest_df = mlflow.search_runs(filter_string=candidate_filter,
                                       order_by=["attributes.start_time DESC"], max_results=1)
        if latest_df.empty:
            raise RuntimeError("no candidate runs found; run python -m wine_lab.train first")
        pipeline_id = latest_df.iloc[0]["tags.pipeline_id"]

    run_df = mlflow.search_runs(
        filter_string=f"{candidate_filter} and tags.pipeline_id = '{pipeline_id}'",
        order_by=["metrics.val_auc DESC"], max_results=1,
    )
    if run_df.empty:
        raise RuntimeError(f"no candidate runs found for pipeline {pipeline_id}")
    return run_df.iloc[0]


def test_auc_of_version(model_version, X_test, y_test) -> float:
    """Load a registered model version and return its AUC on the test set."""
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{model_version.version}")
    return compute_metrics(y_test, model.predict_proba(X_test)[:, 1])["auc"]


def run_promotion(pipeline_id=None) -> dict:
    """
    Register the best candidate, compare it with the champion, and move aliases accordingly.

    Parameters
    ----------
    pipeline_id : str or None
        Training execution whose candidates are considered; defaults to the latest.

    Returns
    -------
    result_dict : dict
        version, challenger_test_auc, champion_version, champion_test_auc, promoted, reason.
    """
    client = setup_tracking()
    best_run = find_best_candidate(pipeline_id)
    model_uri = f"runs:/{best_run['run_id']}/{config.MODEL_ARTIFACT_PATH}"
    new_version = mlflow.register_model(model_uri, config.MODEL_NAME)
    client.set_registered_model_alias(config.MODEL_NAME, config.CHALLENGER_ALIAS,
                                      new_version.version)

    # the same seed reproduces the exact split used during training
    X_test, y_test = split_dataset(prepare_dataset(load_wine_data())[0])["test"]
    challenger_auc = test_auc_of_version(new_version, X_test, y_test)
    champion_version = get_version_by_alias(client, config.CHAMPION_ALIAS)
    champion_auc = (test_auc_of_version(champion_version, X_test, y_test)
                    if champion_version else None)
    o_promote, reason = decide_promotion(challenger_auc, champion_auc,
                                         config.MIN_AUC_IMPROVEMENT)

    tag_dict = {
        "source_run_id": best_run["run_id"],
        "model_family": best_run["tags.model_family"],
        "val_auc": f"{best_run['metrics.val_auc']:.4f}",
        "test_auc": f"{challenger_auc:.4f}",
        "promotion_decision": "promoted" if o_promote else "rejected",
        "promotion_reason": reason,
        "compared_to_version": champion_version.version if champion_version else "none",
    }
    for key, value in tag_dict.items():
        client.set_model_version_tag(config.MODEL_NAME, new_version.version, key, value)
    if o_promote:
        client.set_registered_model_alias(config.MODEL_NAME, config.CHAMPION_ALIAS,
                                          new_version.version)

    with mlflow.start_run(run_name=f"promotion_v{new_version.version}"):
        mlflow.log_metric("challenger_test_auc", challenger_auc)
        if champion_auc is not None:
            mlflow.log_metric("champion_test_auc", champion_auc)
        mlflow.set_tags({"step": "promotion", "model_version": new_version.version,
                         "decision": tag_dict["promotion_decision"], "reason": reason})

    result_dict = {
        "version": new_version.version,
        "challenger_test_auc": challenger_auc,
        "champion_version": champion_version.version if champion_version else None,
        "champion_test_auc": champion_auc,
        "promoted": o_promote,
        "reason": reason,
    }
    print(f"version {new_version.version} ({best_run['tags.model_family']}): test AUC "
          f"{challenger_auc:.4f}; {tag_dict['promotion_decision']} ({reason})")
    return result_dict


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Register and promote the best candidate")
    parser.add_argument("--pipeline-id", default=None)
    args = parser.parse_args(argv)
    run_promotion(args.pipeline_id)


if __name__ == "__main__":
    main()
