#!/bin/bash
set -e

echo "Running all experiments..."

cd code

python -m runners.run_experiment --dataset analcatdata_authorship.csv --model random-forest
python -m runners.run_experiment --dataset analcatdata_authorship.csv --model xgboost
python -m runners.run_experiment --dataset analcatdata_authorship.csv --model catboost

echo "Done ✅"