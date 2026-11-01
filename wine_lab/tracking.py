"""
MLflow setup shared by every step: tracking backend, experiment, and dataset logging.
"""

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from wine_lab import config


def setup_tracking() -> MlflowClient:
    """
    Point MLflow at the SQLite backend and select the lab experiment, creating it if needed.

    Returns
    -------
    client : mlflow.tracking.MlflowClient
        Client bound to the configured tracking URI.
    """
    mlflow.set_tracking_uri(config.TRACKING_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.EXPERIMENT_NAME, artifact_location=config.ARTIFACT_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    return MlflowClient()


def log_split_inputs(split_dict: dict, source: str) -> None:
    """Log each data split as an MLflow dataset input of the active run."""
    for split_name, (X, y) in split_dict.items():
        dataset = mlflow.data.from_pandas(
            X.assign(**{config.TARGET_COLUMN: y}), source=source,
            name=f"wine-{split_name}", targets=config.TARGET_COLUMN,
        )
        mlflow.log_input(dataset, context=split_name)


def get_version_by_alias(client: MlflowClient, alias: str):
    """Return the registered model version holding alias, or None if the alias is unset."""
    try:
        return client.get_model_version_by_alias(config.MODEL_NAME, alias)
    except MlflowException:
        return None
