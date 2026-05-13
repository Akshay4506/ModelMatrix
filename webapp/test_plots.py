import sys
from pathlib import Path
import os

# Add webapp to path to import plots_generator
_here = Path(__file__).parent
sys.path.append(str(_here))

import plots_generator

# Mock Results Data
mock_results = {
    "task": "regression",
    "results": {
        "XGBoost": {
            "mean": {"r2": 0.9967, "rmse": 1.3718, "fit_time": 0.5},
            "folds": [{"r2": 0.996 + 0.001*i, "rmse": 1.3 + 0.1*i} for i in range(5)]
        },
        "LightGBM": {
            "mean": {"r2": 0.9774, "rmse": 3.5754, "fit_time": 0.3},
            "folds": [{"r2": 0.97 + 0.005*i, "rmse": 3.5 + 0.2*i} for i in range(5)]
        },
        "CatBoost": {
            "mean": {"r2": 0.9840, "rmse": 2.9763, "fit_time": 2.1},
            "folds": [{"r2": 0.98 + 0.004*i, "rmse": 2.9 + 0.15*i} for i in range(5)]
        },
        "SAP RPT-1 OSS": {
            "mean": {"r2": 0.9708, "rmse": 4.1184, "fit_time": 15.0},
            "folds": [{"r2": 0.96 + 0.008*i, "rmse": 4.1 + 0.3*i} for i in range(5)]
        },
        "Voting Ensemble": {
            "mean": {"r2": 0.9913, "rmse": 2.1924, "fit_time": 0.8},
            "folds": [{"r2": 0.99 + 0.002*i, "rmse": 2.1 + 0.1*i} for i in range(5)]
        },
        "Stacking Ensemble": {
            "mean": {"r2": 0.9968, "rmse": 1.3639, "fit_time": 1.2},
            "folds": [{"r2": 0.9965 + 0.0001*i, "rmse": 1.36 + 0.01*i} for i in range(5)]
        }
    },
    "stats": {
        "ranking": [
            {"model": "XGBoost", "avg_rank": 1.40},
            {"model": "Stacking Ensemble", "avg_rank": 1.60},
            {"model": "Voting Ensemble", "avg_rank": 3.00},
            {"model": "CatBoost", "avg_rank": 4.20},
            {"model": "LightGBM", "avg_rank": 4.80},
            {"model": "SAP RPT-1 OSS", "avg_rank": 6.00}
        ]
    }
}

def test_generation():
    print("Generating Mock PDF Report...")
    pdf_buffer = plots_generator.generate_plots_pdf(mock_results)
    
    output_path = _here / "mock_report.pdf"
    with open(output_path, "wb") as f:
        f.write(pdf_buffer.getbuffer())
    
    print(f"Success! Report saved to: {output_path}")

if __name__ == "__main__":
    test_generation()
