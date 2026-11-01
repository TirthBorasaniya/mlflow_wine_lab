"""
Scores a CSV of wines with the current champion model and logs the batch as an MLflow run.

Usage:
    python -m wine_lab.batch_predict [--input PATH] [--output PATH]
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from wine_lab import config
from wine_lab.scoring import score_batch
from wine_lab.tracking import get_version_by_alias, setup_tracking


def run_batch(input_path=config.BATCH_INPUT_PATH, output_path=config.PREDICTIONS_PATH):
    """
    Score input_path with the champion and write the predictions to output_path.

    Parameters
    ----------
    input_path : str or pathlib.Path
        CSV with the wine feature columns.
    output_path : str or pathlib.Path
        CSV to write, with probability and label columns added.

    Returns
    -------
    scored_df : pandas.DataFrame
        Scored rows.
    """
    client = setup_tracking()
    champion = get_version_by_alias(client, config.CHAMPION_ALIAS)
    if champion is None:
        raise RuntimeError("no champion model; run python -m wine_lab.promote first")

    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}")
    scored_df = score_batch(model, pd.read_csv(input_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False)

    with mlflow.start_run(run_name="batch_inference"):
        mlflow.log_params({"model_version": champion.version, "input_file": str(input_path)})
        mlflow.log_metrics({
            "rows_scored": len(scored_df),
            "mean_probability": float(scored_df["high_quality_probability"].mean()),
            "predicted_high_quality_rate": float(scored_df["predicted_high_quality"].mean()),
        })
        mlflow.set_tag("step", "batch_inference")
        mlflow.log_artifact(str(output_path))

    print(f"scored {len(scored_df)} wines with version {champion.version}; wrote {output_path}")
    return scored_df


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Batch scoring with the champion model")
    parser.add_argument("--input", default=str(config.BATCH_INPUT_PATH))
    parser.add_argument("--output", default=str(config.PREDICTIONS_PATH))
    args = parser.parse_args(argv)
    run_batch(args.input, args.output)


if __name__ == "__main__":
    main()
