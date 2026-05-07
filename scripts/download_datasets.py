"""
Dataset Downloader
==================
Downloads real datasets into datasets/ as <name>_X.csv and <name>_y.csv pairs.

Sources:
  - sklearn built-ins  (iris, breast_cancer, diabetes, wine, digits)
  - OpenML             (titanic, adult, credit-g)

Run from repo root:
    python scripts/download_datasets.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(name, X, y):
    x_path = OUT_DIR / f"{name}_X.csv"
    y_path = OUT_DIR / f"{name}_y.csv"
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name="target")
    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)
    print(f"  [OK] {name:30s}  {X.shape[0]:>5} rows x {X.shape[1]:>3} cols  ->  datasets/")


def load_sklearn_datasets():
    from sklearn import datasets

    print("\n[1/2] Downloading sklearn built-in datasets...")

    # Iris — 3-class classification
    d = datasets.load_iris(as_frame=True)
    save("iris", d.data, d.target)

    # Breast Cancer — binary classification
    d = datasets.load_breast_cancer(as_frame=True)
    save("breast_cancer", d.data, d.target)

    # Diabetes — regression
    d = datasets.load_diabetes(as_frame=True)
    save("diabetes", d.data, d.target)

    # Wine — 3-class classification
    d = datasets.load_wine(as_frame=True)
    save("wine", d.data, d.target)

    # Digits — 10-class classification (flatten 8x8 images)
    d = datasets.load_digits(as_frame=True)
    save("digits", d.data, d.target)


def load_openml_datasets():
    print("\n[2/2] Downloading OpenML datasets...")
    try:
        from sklearn.datasets import fetch_openml

        # Titanic — binary classification
        try:
            d = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
            X = d.data.select_dtypes(include=[np.number]).fillna(0)
            y = (d.target.astype(str).str.strip() == "1").astype(int)
            save("titanic", X, y)
        except Exception as e:
            print(f"  [SKIP] titanic: {e}")

        # Credit-G — binary classification
        try:
            d = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
            X = d.data.copy()
            # encode categoricals
            for col in X.select_dtypes(include="category").columns:
                X[col] = X[col].cat.codes
            for col in X.select_dtypes(include="object").columns:
                X[col] = X[col].astype("category").cat.codes
            y = (d.target.astype(str).str.strip() == "good").astype(int)
            save("credit_g", X, y)
        except Exception as e:
            print(f"  [SKIP] credit-g: {e}")

        # California Housing — regression
        try:
            d = fetch_openml("house_prices", version=1, as_frame=True, parser="auto")
            X = d.data.select_dtypes(include=[np.number]).fillna(0)
            y = d.target.astype(float)
            save("house_prices", X, y)
        except Exception as e:
            print(f"  [SKIP] house_prices: {e}")

    except ImportError:
        print("  [SKIP] OpenML requires scikit-learn>=0.22 and internet access")


def print_summary():
    files = sorted(OUT_DIR.glob("*_X.csv"))
    print(f"\n{'='*55}")
    print(f"  {len(files)} dataset(s) ready in datasets/")
    print(f"{'='*55}")
    for f in files:
        name = f.stem.replace("_X", "")
        rows = sum(1 for _ in open(f)) - 1
        cols = len(open(f).readline().split(","))
        y_file = OUT_DIR / f"{name}_y.csv"
        # count unique targets
        try:
            uniq = pd.read_csv(y_file).iloc[:, 0].nunique()
            task = "classification" if uniq < 20 else "regression"
        except Exception:
            task = "?"
        print(f"  {name:30s}  {rows:>5} rows  {cols:>3} feat  [{task}]")

    print(f"\nRun an experiment with:")
    print(f"  cd code")
    for f in files[:3]:
        name = f.stem.replace("_X", "")
        print(f"  python -m runners.run_experiment --dataset {name} --model xgboost")


if __name__ == "__main__":
    print("="*55)
    print("  SAP RPT-1 Benchmarking — Dataset Downloader")
    print("="*55)

    load_sklearn_datasets()
    load_openml_datasets()
    print_summary()
