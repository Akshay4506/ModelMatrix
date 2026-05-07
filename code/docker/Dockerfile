# =============================================================================
# SAP RPT-1 Benchmarking - Multi-stage Dockerfile
# =============================================================================
# Builds two targets:
#   - sap-rpt1: Python 3.11 with SAP RPT-1 OSS + all dependencies
#   - baselines: Python 3.11 with XGBoost, CatBoost, LightGBM
#
# Usage:
#   docker-compose build
#   docker-compose run sap-rpt1
#   docker-compose run baselines
# =============================================================================

# ---------- Base stage (shared by all targets) ----------
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt /app/requirements.txt

# ---------- SAP RPT-1 target ----------
FROM base AS sap-rpt1

# Install core scientific stack first (heavy packages)
RUN pip install --default-timeout=1000 --retries 5 --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    scipy==1.14.1 \
    matplotlib==3.9.2 \
    seaborn==0.13.2

# Install Hugging Face and PyTorch stack
RUN pip install --default-timeout=1000 --retries 5 --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.0+cpu \
    transformers==4.52.4 \
    accelerate==1.6.0 \
    huggingface-hub==0.30.2 \
    datasets==3.5.0 \
    pyarrow==20.0.0 \
    torcheval==0.0.7

# Install SAP RPT-1 and remaining requirements
RUN pip install --default-timeout=1000 --retries 5 --no-cache-dir -r requirements.txt

# Copy project code
COPY . /app

# Set Python path
ENV PYTHONPATH=/app/code

WORKDIR /app/code

# Set entrypoint so you can run via arguments natively
ENTRYPOINT ["python"]
CMD ["-m", "runners.run_experiment", "--dataset", "adult", "--model", "sap-rpt1-hf"]

# ---------- Baselines target ----------
FROM base AS baselines

# Install core scientific stack (heavy packages)
RUN pip install --default-timeout=1000 --retries 5 --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    scipy==1.14.1

# Install visualization and utilities
RUN pip install --default-timeout=1000 --retries 5 --no-cache-dir \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    pyyaml==6.0.2 \
    tqdm==4.67.1 \
    joblib==1.4.2 \
    python-dotenv==1.0.1

# Install ML frameworks and OpenML
RUN pip install --default-timeout=1000 --retries 5 --no-cache-dir \
    openml==0.14.2 \
    xgboost \
    catboost \
    lightgbm

# Copy project code
COPY . /app

# Set Python path
ENV PYTHONPATH=/app/code

WORKDIR /app/code

# Set entrypoint so you can run via arguments natively
ENTRYPOINT ["python"]
CMD ["-m", "runners.run_batch", "--datasets", "config/datasets.yaml", "--models", "config/models.yaml"]
