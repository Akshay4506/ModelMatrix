"""
Batch Experiment Runner
========================

Run multiple models on multiple datasets.

Usage:
    python -m runners.run_batch \
        --datasets config/datasets.yaml \
        --models config/models.yaml

Author: UW MSIM Team
Date: April 2026
"""

import argparse
import yaml
import logging
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runners.run_experiment import run_single_experiment, get_model

logger = logging.getLogger(__name__)


def get_dataset_list(datasets_config: dict, dataset_dir: str = None) -> List[str]:
    """
    Get list of available dataset names from the download directory.

    Parameters
    ----------
    datasets_config : dict
        Datasets YAML configuration
    dataset_dir : str
        Directory containing downloaded datasets

    Returns
    -------
    datasets : list of str
        List of dataset names
    """
    datasets = []

    if dataset_dir is None:
        dataset_dir = str(Path(__file__).parent.parent.parent / 'datasets')

    if os.path.isdir(dataset_dir):
        # Find all *_X.csv files and extract dataset names
        for f in sorted(os.listdir(dataset_dir)):
            if f.endswith('_X.csv'):
                name = f[:-6]  # Remove '_X.csv'
                # Verify y file also exists
                y_file = os.path.join(dataset_dir, f"{name}_y.csv")
                if os.path.exists(y_file):
                    datasets.append(name)

        logger.info(f"Found {len(datasets)} datasets in {dataset_dir}")
    else:
        logger.warning(f"Dataset directory not found: {dataset_dir}")

    return datasets


def get_model_list(models_config: dict) -> List[str]:
    """
    Get list of enabled model names from configuration.

    Parameters
    ----------
    models_config : dict
        Models YAML configuration

    Returns
    -------
    models : list of str
        List of enabled model names
    """
    models = []

    for model_entry in models_config.get('models', []):
        if model_entry.get('enabled', True):
            models.append(model_entry['name'])

    return models


def run_batch_experiments(
    datasets: List[str],
    models: List[str],
    experiment_config: dict,
    output_dir: str = '../results/raw',
    skip_existing: bool = True
) -> dict:
    """
    Run experiments for all dataset × model combinations.

    Parameters
    ----------
    datasets : list of str
        Dataset names
    models : list of str
        Model names
    experiment_config : dict
        Experiment configuration (n_folds, random_state, etc.)
    output_dir : str
        Where to save results
    skip_existing : bool
        If True, skip experiments that already have result files

    Returns
    -------
    summary : dict
        Batch run summary with successes and failures
    """
    total_experiments = len(datasets) * len(models)
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH RUN: {len(datasets)} datasets × {len(models)} models = {total_experiments} experiments")
    logger.info(f"{'='*60}\n")

    summary = {
        'total': total_experiments,
        'completed': 0,
        'skipped': 0,
        'failed': 0,
        'results': [],
        'errors': []
    }

    batch_start_time = time.time()

    for i, dataset_name in enumerate(datasets):
        for j, model_name in enumerate(models):
            experiment_num = i * len(models) + j + 1
            output_file = os.path.join(output_dir, f"{dataset_name}_{model_name}.json")

            # Skip existing results
            if skip_existing and os.path.exists(output_file):
                logger.info(
                    f"[{experiment_num}/{total_experiments}] "
                    f"SKIP {model_name} on {dataset_name} (result exists)"
                )
                summary['skipped'] += 1
                continue

            logger.info(
                f"\n[{experiment_num}/{total_experiments}] "
                f"Running {model_name} on {dataset_name}..."
            )

            try:
                result = run_single_experiment(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    config=experiment_config,
                    output_dir=output_dir
                )
                summary['completed'] += 1
                summary['results'].append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"FAILED: {model_name} on {dataset_name}: {e}")
                summary['failed'] += 1
                summary['errors'].append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'error': str(e)
                })

    batch_elapsed = time.time() - batch_start_time

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH RUN COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Total experiments: {summary['total']}")
    logger.info(f"  Completed: {summary['completed']}")
    logger.info(f"  Skipped: {summary['skipped']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Total time: {batch_elapsed / 3600:.2f} hours")
    logger.info(f"{'='*60}\n")

    # Save batch summary
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, '_batch_summary.json')
    summary['elapsed_hours'] = batch_elapsed / 3600
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Batch summary saved to {summary_file}")

    return summary


def main():
    """Entry point for batch runner."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run batch benchmarking experiments')
    parser.add_argument('--datasets', default='config/datasets.yaml',
                        help='Datasets config file')
    parser.add_argument('--models', default='config/models.yaml',
                        help='Models config file')
    parser.add_argument('--config', default='config/experiments.yaml',
                        help='Experiment config file')
    parser.add_argument('--output-dir', default='../results/raw',
                        help='Output directory')
    parser.add_argument('--dataset-dir', default=None,
                        help='Directory containing downloaded datasets')
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-run experiments even if results exist')
    parser.add_argument('--model-filter', nargs='*', default=None,
                        help='Only run specific models (e.g., --model-filter sap-rpt1-hf xgboost)')
    parser.add_argument('--dataset-filter', nargs='*', default=None,
                        help='Only run specific datasets')

    args = parser.parse_args()

    # Load configs
    if os.path.exists(args.datasets):
        with open(args.datasets) as f:
            datasets_config = yaml.safe_load(f)
    else:
        datasets_config = {}

    if os.path.exists(args.models):
        with open(args.models) as f:
            models_config = yaml.safe_load(f)
    else:
        models_config = {}

    if os.path.exists(args.config):
        with open(args.config) as f:
            experiment_config = yaml.safe_load(f)
    else:
        experiment_config = {
            'n_folds': 10,
            'random_state': 42,
            'cost_per_hour': 0.90,
            'gpu_type': 'H200'
        }

    # Get dataset and model lists
    dataset_list = args.dataset_filter or get_dataset_list(datasets_config, args.dataset_dir)
    model_list = args.model_filter or get_model_list(models_config)

    if not dataset_list:
        print("[ERROR] No datasets found in the datasets directory.")
        sys.exit(1)

    if not model_list:
        print("[ERROR] No models enabled in config. Check config/models.yaml")
        sys.exit(1)

    print(f"\n[INFO] Datasets ({len(dataset_list)}): {dataset_list[:5]}{'...' if len(dataset_list) > 5 else ''}")
    print(f"[INFO] Models ({len(model_list)}): {model_list}")

    # Add dataset_dir to config for run_experiment to use
    experiment_config['dataset_dir'] = args.dataset_dir if args.dataset_dir else str(Path(__file__).parent.parent.parent / 'datasets')

    # Run batch
    summary = run_batch_experiments(
        datasets=dataset_list,
        models=model_list,
        experiment_config=experiment_config,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip
    )

    print(f"\n[SUCCESS] Batch complete! {summary['completed']} succeeded, {summary['failed']} failed")


if __name__ == "__main__":
    main()
