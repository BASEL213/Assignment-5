# Dockerfile
# Base image
FROM python:3.10-slim

# Accept the MLflow Run ID as a build argument
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir mlflow scikit-learn

# Copy application files
COPY train.py .
COPY check_threshold.py .

# Simulate downloading the model from MLflow
# (Replace the echo with a real mlflow artifacts download command
#  once you have a reachable tracking server)
RUN echo "Downloading model for Run ID: ${RUN_ID}" && \
    mkdir -p /app/model && \
    echo "Model artifact placeholder – Run ID: ${RUN_ID}" > /app/model/info.txt

# Default command – swap for your actual serving entrypoint
CMD ["python", "-c", "import os; print('Serving model for Run ID:', os.getenv('RUN_ID','unknown'))"]
