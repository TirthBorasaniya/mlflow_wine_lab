"""
Central configuration for the wine quality MLflow lab: paths, names, and tuning settings.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RED_PATH = DATA_DIR / "winequality-red.csv"
WHITE_PATH = DATA_DIR / "winequality-white.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"
BATCH_INPUT_PATH = OUTPUT_DIR / "batch_input.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

# as_posix keeps the URI valid on Windows, where backslashes would break it
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
)
ARTIFACT_URI = (REPO_ROOT / "mlartifacts").as_uri()
EXPERIMENT_NAME = "wine-quality"
MODEL_NAME = "wine_quality"
MODEL_ARTIFACT_PATH = "model"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

TARGET_COLUMN = "quality"
HIGH_QUALITY_THRESHOLD = 7
RANDOM_STATE = 123
TRAIN_FRACTION = 0.6
DEFAULT_MAX_EVALS = 30
MIN_AUC_IMPROVEMENT = 0.002
SERVE_PORT = 5001
