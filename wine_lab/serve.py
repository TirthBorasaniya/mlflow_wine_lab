"""
Serves the champion model as a REST API with MLflow, on any operating system.

Usage:
    python -m wine_lab.serve [--port 5001]
"""

import argparse
import os
import subprocess
import sys

from wine_lab import config


def main(argv=None):
    """Start mlflow models serve for the champion alias with the lab's tracking backend."""
    parser = argparse.ArgumentParser(description="Serve the champion model")
    parser.add_argument("--port", type=int, default=config.SERVE_PORT)
    args = parser.parse_args(argv)

    # the serve command resolves models:/ URIs through this environment variable
    env = {**os.environ, "MLFLOW_TRACKING_URI": config.TRACKING_URI}
    command = [
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}",
        "--env-manager", "local", "--host", "127.0.0.1", "--port", str(args.port),
    ]
    print(f"serving models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS} "
          f"on http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    sys.exit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
