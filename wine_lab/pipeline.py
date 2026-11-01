"""
Runs the full workflow: train and track candidates, register and promote, then batch score.

Usage:
    python -m wine_lab.pipeline --max-evals 30
"""

import argparse

from wine_lab import config
from wine_lab.batch_predict import run_batch
from wine_lab.promote import run_promotion
from wine_lab.train import run_training


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train, promote, and batch score")
    parser.add_argument("--max-evals", type=int, default=config.DEFAULT_MAX_EVALS)
    args = parser.parse_args(argv)

    pipeline_id = run_training(args.max_evals)
    run_promotion(pipeline_id)
    run_batch()
    print("pipeline complete; open the UI with: "
          "mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
