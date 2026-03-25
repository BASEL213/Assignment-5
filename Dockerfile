FROM python:3.10-slim

# Accept the Run ID as a build argument
ARG RUN_ID
ENV MLFLOW_RUN_ID=$RUN_ID

# Normally we would use the MLflow API to download the model artifact
# Instead, we will simulate the download here
RUN echo "Downloading model for Run ID: $MLFLOW_RUN_ID..."

# Define the command to run the model
CMD ["python", "-c", "print('Model is running for Run ID:', __import__('os').environ.get('MLFLOW_RUN_ID'))"]
