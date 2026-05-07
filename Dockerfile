FROM python:3.11-slim

# Create user to run the app
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install system dependencies (e.g. for lightgbm/xgboost)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy the entire project
COPY --chown=user . $HOME/app/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r webapp/requirements.txt

# Install SAP-RPT-1 OSS directly from GitHub (needed for the real model)
RUN pip install --no-cache-dir git+https://github.com/SAP-samples/sap-rpt-1-oss.git

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Run the FastAPI app
CMD ["python", "-m", "uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "7860"]
