import os
import sys
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Attempt to pull data (simulated DVC behaviour if this was a real file)
    print("Loading data...")
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # We simulate pass/fail for GitHub Actions by looking at an env var
    # Or just default to a good model
    force_fail = os.environ.get("FORCE_FAIL", "false").lower() == "true"
    
    if force_fail:
        print("Training a weak model to simulate failure (< 0.85 accuracy).")
        # Depth 1 with 1 estimator on Iris will be very bad
        clf = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=42)
    else:
        print("Training a strong model to simulate success (> 0.85 accuracy).")
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        with mlflow.start_run() as run:
            mlflow.log_param("n_estimators", clf.n_estimators)
            mlflow.log_param("max_depth", clf.max_depth)
            mlflow.log_metric("accuracy", accuracy)
            
            run_id = run.info.run_id
            print(f"Training completed. Accuracy: {accuracy:.4f}")
            print(f"MLflow Run ID: {run_id}")
            
            with open("model_info.txt", "w") as f:
                f.write(run_id)
                
    except Exception as e:
        print(f"Failed to log to MLflow: {e}")
        # In a generic environment we might not have the tracking server running.
        # Still generate model_info.txt for the pipeline mock testing
        run_id = "mock-run-id-12345"
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        # Also mock the accuracy for the check limit script
        with open("mock_accuracy.txt", "w") as f:
            f.write(str(accuracy))
        print(f"Wrote mock run_id {run_id} to model_info.txt")

if __name__ == "__main__":
    main()
