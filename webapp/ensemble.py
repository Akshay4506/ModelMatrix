"""
ensemble.py — Ensemble builder for the SAP RPT-1 Benchmarking Web App.

Given individual CV results, this module:
  1. Selects the top-N performing models
  2. Runs a Soft Voting ensemble (works with ALL model types)
  3. Runs a Stacking ensemble (sklearn-native models only)
  4. Returns CV results in the same schema as individual models
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             r2_score, mean_absolute_error, mean_squared_error)
from sklearn.linear_model import LogisticRegression, Ridge

warnings.filterwarnings("ignore")

N_FOLDS = int(os.getenv("N_FOLDS",      "5"))
RAND    = int(os.getenv("RANDOM_STATE", "42"))

# Sklearn-native builders safe to use in StackingClassifier/Regressor
SKLEARN_SAFE = {"XGBoost", "LightGBM", "CatBoost"}


# ── Model selection ────────────────────────────────────────────────────────────

def select_top_models(results: dict, builders: dict, task: str, n: int = 3):
    """
    Return top-N (name, builder) pairs by primary metric, skipping errored models.
    Only includes models that have >0.5 ROC-AUC or >0.0 R².
    """
    primary   = "roc_auc" if task == "classification" else "r2"
    threshold = 0.50 if task == "classification" else 0.0

    ranked = []
    for name in builders:
        if name not in results or "error" in results[name]:
            continue
        score = results[name]["mean"].get(primary, 0) or 0
        if score >= threshold:
            ranked.append((name, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    top = ranked[:n]
    return [(name, builders[name]) for name, _ in top]


# ── Voting ensemble (manual soft voting) ──────────────────────────────────────

def run_voting_ensemble(top_pairs: list, X: pd.DataFrame, y: pd.Series,
                        task: str, prep_fn) -> dict:
    """
    Manual soft-voting ensemble. Works with ANY model (sklearn or custom).
    Each fold trains all top models and averages probabilities / predictions.
    """
    if len(top_pairs) < 2:
        raise ValueError("Need at least 2 models to form an ensemble.")

    if task == "classification":
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=RAND).split(X, y))
    else:
        splits = list(KFold(N_FOLDS, shuffle=True, random_state=RAND).split(X))

    n_classes = int(y.nunique()) if task == "classification" else None
    fold_results = []

    for tr_idx, val_idx in splits:
        Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]
        Xtr_p, encoders = prep_fn(Xtr)
        Xval_p, _       = prep_fn(Xval, encoders=encoders)

        t0 = time.perf_counter()

        if task == "classification":
            n_cls = n_classes or int(np.unique(ytr).size)
            all_probas = []
            for _, builder in top_pairs:
                try:
                    model = builder(task)
                    model.fit(Xtr_p, ytr)
                    try:
                        proba = model.predict_proba(Xval_p)
                        # Normalise rows
                        row_sum = proba.sum(axis=1, keepdims=True) + 1e-9
                        all_probas.append(proba / row_sum)
                    except Exception:
                        # Fallback: one-hot from predict
                        pred = model.predict(Xval_p).astype(int)
                        oh = np.zeros((len(pred), n_cls))
                        for i, p in enumerate(pred):
                            if 0 <= p < n_cls:
                                oh[i, p] = 1.0
                        all_probas.append(oh)
                except Exception:
                    continue  # skip failing models within the fold

            fit_t = time.perf_counter() - t0
            if not all_probas:
                continue

            avg_proba = np.mean(all_probas, axis=0)
            y_pred    = np.argmax(avg_proba, axis=1)

            acc = accuracy_score(yval, y_pred)
            f1  = f1_score(yval, y_pred, average="macro", zero_division=0)
            try:
                auc = (roc_auc_score(yval, avg_proba[:, 1])
                       if avg_proba.shape[1] == 2
                       else roc_auc_score(yval, avg_proba,
                                          multi_class="ovr", average="macro"))
            except Exception:
                auc = float("nan")

            fold_results.append({"accuracy": acc, "f1_macro": f1,
                                  "roc_auc": auc, "fit_time": fit_t})

        else:  # regression
            all_preds = []
            for _, builder in top_pairs:
                try:
                    model = builder(task)
                    model.fit(Xtr_p, ytr)
                    all_preds.append(model.predict(Xval_p))
                except Exception:
                    continue

            fit_t = time.perf_counter() - t0
            if not all_preds:
                continue

            avg_pred = np.mean(all_preds, axis=0)
            fold_results.append({
                "r2":       r2_score(yval, avg_pred),
                "mae":      mean_absolute_error(yval, avg_pred),
                "rmse":     float(np.sqrt(mean_squared_error(yval, avg_pred))),
                "fit_time": fit_t,
            })

    if not fold_results:
        raise ValueError("All folds failed for voting ensemble.")

    df = pd.DataFrame(fold_results)
    return {"mean": df.mean().to_dict(), "std": df.std().to_dict(),
            "folds": df.to_dict("records")}


# ── Stacking ensemble (sklearn-safe models only) ───────────────────────────────

def run_stacking_ensemble(sklearn_pairs: list, X: pd.DataFrame, y: pd.Series,
                          task: str, prep_fn) -> dict:
    """
    Stacking ensemble using sklearn StackingClassifier / StackingRegressor.
    Only XGBoost, LightGBM, CatBoost (sklearn-native) are used as base learners.
    Meta-learner: LogisticRegression (clf) or Ridge (reg).
    """
    from sklearn.ensemble import StackingClassifier, StackingRegressor

    if len(sklearn_pairs) < 2:
        raise ValueError("Need at least 2 sklearn-compatible models for stacking.")

    if task == "classification":
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=RAND).split(X, y))
        meta   = LogisticRegression(max_iter=1000, random_state=RAND, C=1.0)
    else:
        splits = list(KFold(N_FOLDS, shuffle=True, random_state=RAND).split(X))
        meta   = Ridge(random_state=RAND)

    fold_results = []

    for tr_idx, val_idx in splits:
        Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]
        Xtr_p, encoders = prep_fn(Xtr)
        Xval_p, _       = prep_fn(Xval, encoders=encoders)

        estimators = [(name, builder(task)) for name, builder in sklearn_pairs]

        if task == "classification":
            stacker = StackingClassifier(
                estimators=estimators,
                final_estimator=meta,
                cv=3,
                passthrough=False,
                n_jobs=1,
            )
        else:
            stacker = StackingRegressor(
                estimators=estimators,
                final_estimator=meta,
                cv=3,
                passthrough=False,
                n_jobs=1,
            )

        t0 = time.perf_counter()
        stacker.fit(Xtr_p, ytr)
        fit_t = time.perf_counter() - t0

        if task == "classification":
            y_pred = stacker.predict(Xval_p)
            acc    = accuracy_score(yval, y_pred)
            f1     = f1_score(yval, y_pred, average="macro", zero_division=0)
            try:
                proba = stacker.predict_proba(Xval_p)
                auc   = (roc_auc_score(yval, proba[:, 1])
                         if proba.shape[1] == 2
                         else roc_auc_score(yval, proba,
                                            multi_class="ovr", average="macro"))
            except Exception:
                auc = float("nan")
            fold_results.append({"accuracy": acc, "f1_macro": f1,
                                  "roc_auc": auc, "fit_time": fit_t})
        else:
            y_pred = stacker.predict(Xval_p)
            fold_results.append({
                "r2":       r2_score(yval, y_pred),
                "mae":      mean_absolute_error(yval, y_pred),
                "rmse":     float(np.sqrt(mean_squared_error(yval, y_pred))),
                "fit_time": fit_t,
            })

    if not fold_results:
        raise ValueError("All folds failed for stacking ensemble.")

    df = pd.DataFrame(fold_results)
    return {"mean": df.mean().to_dict(), "std": df.std().to_dict(),
            "folds": df.to_dict("records")}
