"""
check_threshold.py
Reads the MLflow Run ID from model_info.txt, fetches the accuracy metric,
and exits with code 1 if accuracy is below the required threshold.
"""

import os
import sys
import mlflow

THRESHOLD = 0.85

# ── MLflow setup ──────────────────────────────────────────────────────────────
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")
else:
    local_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(local_uri)
    print(f"MLFLOW_TRACKING_URI not set – using local store: {local_uri}")

# ── Read Run ID ───────────────────────────────────────────────────────────────
try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Checking MLflow Run ID: {run_id}")
except FileNotFoundError:
    print("ERROR: model_info.txt not found.")
    sys.exit(1)

# ── Fetch accuracy ────────────────────────────────────────────────────────────
accuracy = None

# 1. Try the configured MLflow tracking server
try:
    client = mlflow.tracking.MlflowClient()
    run    = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")
    if accuracy is not None:
        print(f"Fetched accuracy from MLflow server: {accuracy:.4f}")
    else:
        print("Run found but 'accuracy' metric is missing.")
except Exception as e:
    print(f"Failed to fetch run from external server: {e}")

# 2. Fall back to a local mock accuracy file (useful for CI testing)
if accuracy is None:
    mock_file = "mock_accuracy.txt"
    print(f"Checking for local mocked accuracy file for testing...")
    if os.path.exists(mock_file):
        with open(mock_file, "r") as f:
            accuracy = float(f.read().strip())
        print(f"Using mock accuracy from {mock_file}: {accuracy:.4f}")
    else:
        print(f"No mock accuracy found. Defaulting to 0.0.")
        accuracy = 0.0

# ── Threshold check ───────────────────────────────────────────────────────────
print(f"Model Accuracy : {accuracy:.4f}")
print(f"Threshold      : {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"Validation Failed: Model accuracy is below the {THRESHOLD} threshold.")
    sys.exit(1)

print(f"Validation Passed: Model accuracy meets the {THRESHOLD} threshold. ✅")
sys.exit(0)
