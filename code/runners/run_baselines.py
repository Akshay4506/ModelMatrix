"""
Baseline Models Batch Runner
==============================

Run all baseline models (XGBoost, CatBoost, LightGBM) on all or specific datasets.

Usage:
    # Run on ALL datasets
    py -3.12 -m runners.run_baselines

    # Run on specific datasets
    py -3.12 -m runners.run_baselines --dataset analcatdata_authorship diabetes

Author: UW MSIM Team
Date: April 2026
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runners.run_batch import main as run_batch_main


BASELINE_MODELS = ['xgboost', 'catboost', 'lightgbm']


def main():
    """Run all baseline models on all or specific datasets."""
    parser = argparse.ArgumentParser(description='Run baseline models')
    parser.add_argument('--dataset', nargs='*', default=None,
                        help='Specific dataset(s) to run (e.g., --dataset analcatdata_authorship diabetes)')

    args = parser.parse_args()

    # Build sys.argv for run_batch
    batch_args = ['run_baselines', '--model-filter', *BASELINE_MODELS]

    if args.dataset:
        batch_args.extend(['--dataset-filter', *args.dataset])

    sys.argv = batch_args
    run_batch_main()


if __name__ == '__main__':
    main()
