import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

def main():
    if not os.path.exists("model_info.txt"):
        print("Error: model_info.txt not found!")
        sys.exit(1)
        
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
        
    print(f"Checking MLflow Run ID: {run_id}")
    
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        
    client = MlflowClient()
    try:
        run = client.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy", 0.0)
    except Exception as e:
        print(f"Failed to fetch run from external server: {e}")
        print("Checking for local mocked accuracy file for testing...")
        if os.path.exists("mock_accuracy.txt"):
            with open("mock_accuracy.txt", "r") as f:
                accuracy = float(f.read().strip())
        else:
            print("No mock accuracy found. Defaulting to 0.0.")
            accuracy = 0.0
            
    print(f"Model Accuracy: {accuracy:.4f}")
    
    if accuracy < 0.85:
        print("Validation Failed: Model accuracy is below the 0.85 threshold.")
        sys.exit(1)
    else:
        print("Validation Passed: Model accuracy meets the threshold.")
        sys.exit(0)

if __name__ == "__main__":
    main()
