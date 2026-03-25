"""
train.py
Trains a simple classifier, logs metrics to MLflow,
and writes the Run ID to model_info.txt.
"""

import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── MLflow setup ──────────────────────────────────────────────────────────────
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")
else:
    # Fall back to a local SQLite-backed store so the run is always queryable
    local_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(local_uri)
    print(f"MLFLOW_TRACKING_URI not set – using local store: {local_uri}")

mlflow.set_experiment("model-validation-pipeline")

# ── Train ─────────────────────────────────────────────────────────────────────
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    n_estimators = 100
    max_depth    = 5

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth",    max_depth)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

    run_id = run.info.run_id
    print(f"MLflow Run ID : {run_id}")
    print(f"Model Accuracy: {acc:.4f}")

# ── Export ────────────────────────────────────────────────────────────────────
with open("model_info.txt", "w") as f:
    f.write(run_id)

print(f"Run ID written to model_info.txt")
