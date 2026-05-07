---
title: SAP RPT1 Benchmarker
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# SAP RPT-1 Benchmarking
## 🚀 Setup

### Option 1: Docker (Recommended for Reproducibility)

```bash
# Clone the repo
git clone <repo-url>
cd "MINI proj SAP"

# Copy .env.example to .env and paste your HuggingFace token
cp .env.example .env

# Build containers
docker-compose build

# Run SAP RPT-1 experiment
docker-compose run sap-rpt1 -m runners.run_experiment --dataset analcatdata_authorship --model sap-rpt1-hf

# Run baselines batch
docker-compose run baselines -m runners.run_batch --datasets config/datasets.yaml --models config/models.yaml
```

### Option 2: Local Install (Python >= 3.11 required)

```bash
# Clone the repo
git clone <repo-url>
cd "MINI proj SAP"

# Install everything in one command
pip install -e ".[models,baselines]"

# Download datasets (19 datasets from OpenML)
cd code
python -m datasets.download_tabarena
cd ..
```

## 🔑 Hugging Face Token Setup (Required for SAP RPT-1 OSS)

The SAP RPT-1 OSS model weights are **gated** on Hugging Face:

1. Create account at [huggingface.co/join](https://huggingface.co/join)
2. Accept the license at [huggingface.co/SAP/sap-rpt-1-oss](https://huggingface.co/SAP/sap-rpt-1-oss)
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set the token:

**Windows (PowerShell):**
```powershell
$env:HUGGING_FACE_HUB_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Linux/Mac:**
```bash
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Or using .env file** (recommended):
```bash
cp .env.example .env
# Edit .env and paste your token
```

## 🧪 Quick Test

```bash
cd code
python ../scripts/test_sap_rpt1.py
```

This verifies HF token authentication, model download, and prediction accuracy.

## 📊 Run Experiments

### Single Experiment
```bash
cd code

# SAP RPT-1 OSS
python -m runners.run_experiment --dataset analcatdata_authorship --model sap-rpt1-hf

# XGBoost baseline
python -m runners.run_experiment --dataset analcatdata_authorship --model xgboost
```

### Baseline Models Only (XGBoost, CatBoost, LightGBM)
```bash
cd code

# Run on ALL datasets
python -m runners.run_baselines

# Run on specific datasets
python -m runners.run_baselines --dataset analcatdata_authorship diabetes
```

### Full Batch (All Models × All Datasets)
```bash
cd code
python -m runners.run_batch --datasets config/datasets.yaml --models config/models.yaml
```

### Available Models

| Model Name | Type | Description |
|---|---|---|
| `sap-rpt1-hf` | Pretrained (OSS) | SAP RPT-1 OSS via HuggingFace |
| `xgboost` | Baseline | XGBoost |
| `catboost` | Baseline | CatBoost |
| `lightgbm` | Baseline | LightGBM |

## 📈 View Results

Results are saved to `results/raw/[dataset]_[model].json`

Example output:
```json
{
  "dataset": "analcatdata_authorship",
  "model": "sap-rpt1-hf",
  "task_type": "classification",
  "n_samples": 841,
  "n_features": 70,
  "mean_metrics": {
    "accuracy": 1.0,
    "roc_auc": 1.0,
    "f1_macro": 1.0
  }
}
```

## 📊 Aggregate Results
```bash
cd code
python -m analysis.aggregate_results
```

## 🌐 Web Interface (Advanced Version)

We've completely overhauled the interactive web application to provide a production-grade, scientific benchmarking experience directly in your browser.

**Tech Stack & Architecture:**
- **Frontend**: Pure HTML/CSS/Vanilla JS. Built with a custom "Midnight Precision" design system featuring glassmorphism, dynamic data-aware input generation, and theme-aware custom scrollbars.
- **Backend**: Python with FastAPI and Scikit-Learn/Scipy.
- **Visualizations**: Chart.js for rendering dynamic metric comparisons.

**Key Features Built:**
- **Midnight Precision Aesthetics**: A premium, ultra-modern UI featuring animated liquid gradients, responsive design, and seamless user interaction flows.
- **Advanced Ensemble Engine**: Automatically builds and benchmarks Meta-Models on the fly:
  - *Voting Ensembles*: Soft-voting probabilities across top models.
  - *Stacking Ensembles*: Sklearn-native meta-learning (LogisticRegression/Ridge) layered on top of base models.
- **Statistical Rigor & Ranking**: Moves beyond simple average scores to actual scientific analysis:
  - *Cross-Fold Ranking*: Olympic-style "min" ranking across all CV folds.
  - *Friedman Significance Testing*: Computes P-Values to formally test if the champion model's lead is statistically significant.
  - *Stability Badges*: Automatically tags models as 'Dominant', 'Competitive', or 'Volatile' based on their consistency in winning folds.
- **Interactive Live Playground**: Once the benchmark finishes, a live interface is generated. 
  - *Stateful Pipeline*: The backend caches the exact `LabelEncoder` states from the training phase, ensuring the live playground data is mathematically aligned with the original dataset.
  - *Data-Aware UI*: Input fields dynamically adapt to numeric or categorical columns based on backend typing.

**How to start the Web App:**
```bash
cd webapp
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```
Then open your browser and navigate to `http://localhost:8000`.

## 🏗️ Project Structure

```text
MINI proj SAP/
├── code/
│   ├── docker/              # Docker environments
│   ├── models/              # Model wrappers (sklearn-compatible)
│   │   ├── sap_rpt1_hf_wrapper.py  # SAP RPT-1 OSS via HuggingFace
│   │   ├── base_wrapper.py         # Abstract base class
│   │   └── ...
│   ├── evaluation/          # Metrics, cross-validation, compute tracking
│   ├── runners/             # Experiment execution
│   │   ├── run_experiment.py    # Single experiment
│   │   ├── run_batch.py         # Batch experiments
│   │   └── run_baselines.py     # Baseline models only
│   ├── analysis/            # Results aggregation
│   └── config/              # YAML configurations
├── webapp/                  # Interactive Web Application
│   ├── main.py              # FastAPI Backend Server
│   ├── benchmark.py         # Advanced Benchmarking Engine
│   ├── ensemble.py          # Meta-Model Generators
│   ├── requirements.txt     # Web-specific dependencies
│   └── static/              # Frontend Assets
│       ├── landing.html     # Animated Landing Page
│       ├── uploader.html    # Drag & Drop Interface
│       ├── arena.html       # Results & Statistical Rigor UI
│       ├── app.js           # Client-side Logic
│       └── style.css        # Midnight Precision Styles
├── results/                 # Experiment outputs
├── scripts/
│   └── test_sap_rpt1.py     # Quick-start validation test
├── requirements.txt         # Pinned dependencies
├── setup.py                 # Package configuration
├── docker-compose.yml       # Docker orchestration
└── .env.example             # HF token template
```

## 🔄 Reproducibility

This repo follows NeurIPS/ICML reproducibility standards:

- **Pinned dependencies**: All packages have exact versions in `requirements.txt`
- **Fixed random seeds**: `random_state=42` across all experiments
- **Docker containers**: Isolated environments for incompatible dependencies
- **Gated model weights**: SAP RPT-1 OSS uses a fixed checkpoint (`v1.1.2`)
- **5-fold cross-validation**: Stratified splits ensure identical data partitions


## 🆘 Troubleshooting

**Python version error:**
SAP RPT-1 OSS requires Python >= 3.11. Check with `python --version`.

**Missing TabPFN Error (ModuleNotFoundError):**
If you encounter an error stating that `tabpfn` is missing when running the benchmark, install it manually:
```bash
pip install tabpfn
```

**HF Token not working:**
```bash
huggingface-cli whoami
huggingface-cli login
```

**Docker build fails:**
```bash
docker-compose build --no-cache
```

**Out of memory:**
Edit `code/config/experiments.yaml` and reduce:
```yaml
sap_rpt1_hf:
  max_context_size: 2048  # Lower from 4096
  bagging: 1              # Lower from 4
```