"""
Sends sample wines to the served champion model and prints the predicted probabilities.

Usage, with python -m wine_lab.serve running in another terminal:
    python -m wine_lab.serve_client [--rows 5]
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

import pandas as pd

from wine_lab import config
from wine_lab.data import FEATURE_COLUMNS

REQUEST_TIMEOUT_SECONDS = 30


def build_payload(input_df: pd.DataFrame) -> bytes:
    """Encode rows in MLflow's dataframe_split request format."""
    split_dict = input_df[FEATURE_COLUMNS].to_dict(orient="split")
    split_dict.pop("index", None)
    return json.dumps({"dataframe_split": split_dict}).encode("utf-8")


def main(argv=None):
    """Request predictions for the first rows of the batch input file."""
    parser = argparse.ArgumentParser(description="Query the served model")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--port", type=int, default=config.SERVE_PORT)
    args = parser.parse_args(argv)

    input_df = pd.read_csv(config.BATCH_INPUT_PATH).head(args.rows)
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/invocations", data=build_payload(input_df),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            prediction_list = json.loads(response.read())["predictions"]
    except urllib.error.URLError as err:
        print(f"could not reach the model server: {err}. Is python -m wine_lab.serve running?")
        sys.exit(1)

    for i, probability_pair in enumerate(prediction_list):
        print(f"wine {i + 1}: P(high quality) = {probability_pair[1]:.3f}")


if __name__ == "__main__":
    main()
