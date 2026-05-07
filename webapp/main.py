"""
main.py — FastAPI backend for the SAP RPT-1 Benchmarking Web App.
"""

import io, os
from pathlib import Path
from dotenv import load_dotenv

# Load .env before anything else so HF_TOKEN is available to benchmark.py
load_dotenv(Path(__file__).parent / ".env")

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from benchmark import run_benchmark, infer_task
except ImportError:
    from webapp.benchmark import run_benchmark, infer_task

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "5")) * 1024 * 1024   # default 5 MB

app = FastAPI(title="SAP RPT-1 Benchmarking API", version="1.0.0")

# ── Static files (frontend) ────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from fastapi.responses import FileResponse

@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "landing.html"))

@app.get("/arena")
def arena():
    return FileResponse(str(STATIC_DIR / "arena.html"))


# ── /preview ───────────────────────────────────────────────────────────────────
@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    """
    Return column names + first 5 rows of the uploaded CSV.
    Used by the frontend to let the user pick the target column.
    """
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File too large. Max size is {MAX_FILE_BYTES // (1024*1024)} MB.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if df.shape[1] < 2:
        raise HTTPException(400, "CSV must have at least 2 columns (features + target).")

    # Guess default target: last column
    default_target = df.columns[-1]

    return JSONResponse({
        "columns":        list(df.columns),
        "default_target": default_target,
        "n_rows":         len(df),
        "n_cols":         df.shape[1],
        "preview":        df.head(5).fillna("").to_dict("records"),
    })



# ── Live Prediction Wrappers ──────────────────────────────────────────────────
import numpy as np

class LiveVotingEnsemble:
    def __init__(self, names, builders, task):
        self.models = [(n, builders[n](task)) for n in names]
        self.task = task
    def fit(self, X, y):
        for _, m in self.models: m.fit(X, y)
    def predict(self, X):
        if self.task == "regression":
            preds = [m.predict(X).ravel()[0] for _, m in self.models]
            return np.array([np.mean(preds)])
        
        # Classification
        try:
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        except:
            preds = [int(m.predict(X).ravel()[0]) for _, m in self.models]
            return np.array([np.bincount(preds).argmax()])

    def predict_proba(self, X):
        all_probas = []
        for _, m in self.models:
            try:
                p = m.predict_proba(X)
                all_probas.append(p)
            except:
                # Fallback: one-hot from prediction
                pred = int(m.predict(X).ravel()[0])
                # We'll use a 100-wide array just to be safe, or 
                # ideally we'd know n_classes. For the playground, 
                # the RAVEL logic in /predict handles the cleanup.
                oh = np.zeros((1, 100))
                if pred < 100: oh[0, pred] = 1.0
                all_probas.append(oh)
        
        # Average only if we have consistent shapes
        return np.mean(all_probas, axis=0)

class LiveStackingEnsemble:
    def __init__(self, names, builders, task):
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        estimators = [(n, builders[n](task)) for n in names]
        if task == "classification":
            self.model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)
        else:
            self.model = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=3)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        res = self.model.predict(X)
        return res.reshape(1, -1) if res.ndim == 1 else res
    def predict_proba(self, X):
        return self.model.predict_proba(X)

# ── Live Prediction Cache ──────────────────────────────────────────────────────
CHAMPION_MODEL = None
CHAMPION_INFO  = {"name": None, "task": None, "features": []}

@app.post("/benchmark")
async def benchmark(
    file:       UploadFile = File(...),
    target_col: str        = Form(...),
):
    global CHAMPION_MODEL, CHAMPION_INFO
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_BYTES // (1024*1024)} MB.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if target_col not in df.columns:
        raise HTTPException(400, f"Column '{target_col}' not found.")

    try:
        result = run_benchmark(df, target_col)
        
        # Deep-sanitize the result to ensure 100% JSON compatibility
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif hasattr(obj, "item"): # Handle numpy scalars
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        result = sanitize(result)
        
        # Add explicit feature types for the playground UI
        feature_types = {}
        for col in df.columns:
            if col == target_col: continue
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_types[col] = "numeric"
            else:
                feature_types[col] = "categorical"
        result["dataset_info"]["feature_types"] = feature_types
        
        # Cache the Best Overall model for the Live Playground
        best_name = result["recommendation"]["recommendations"]["best_overall"]["model"]
        from benchmark import BUILDERS, _prep, _encode_target
        X = df.drop(columns=[target_col])
        y_raw = df[target_col]
        task = result["dataset_info"]["task"]
        y, le = _encode_target(y_raw, task)
        
        # Capture the final encoders from the full dataset
        X_p, feat_encoders = _prep(X)
        
        if best_name == "Voting Ensemble":
            comp_names = result["ensemble_info"]["Voting Ensemble"]["components"]
            CHAMPION_MODEL = LiveVotingEnsemble(comp_names, BUILDERS, task)
            CHAMPION_MODEL.fit(X_p, y)
        elif best_name == "Stacking Ensemble":
            comp_names = result["ensemble_info"]["Stacking Ensemble"]["components"]
            CHAMPION_MODEL = LiveStackingEnsemble(comp_names, BUILDERS, task)
            CHAMPION_MODEL.fit(X_p, y)
        else:
            builder = BUILDERS.get(best_name)
            if builder:
                CHAMPION_MODEL = builder(task)
                CHAMPION_MODEL.fit(X_p, y)
            
        CHAMPION_INFO = {
            "name": best_name,
            "task": task,
            "features": list(X.columns),
            "labels": list(le.classes_) if le else None,
            "encoders": feat_encoders  # Store these for the /predict endpoint!
        }
            
    except Exception as e:
        raise HTTPException(500, f"Benchmarking failed: {e}")

    return JSONResponse(result)


@app.post("/predict")
async def predict(data: dict):
    """
    Get a live prediction from the cached champion model.
    """
    global CHAMPION_MODEL, CHAMPION_INFO
    if not CHAMPION_MODEL:
        raise HTTPException(400, "No champion model loaded. Run a benchmark first.")
    
    try:
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([data])
        # Ensure column order matches training
        input_df = input_df[CHAMPION_INFO["features"]]
        
        from benchmark import _prep
        # Use the EXACT same encoders that were used during training
        X_test, _ = _prep(input_df, encoders=CHAMPION_INFO.get("encoders"))
        
        if CHAMPION_INFO["task"] == "classification":
            raw_pred = CHAMPION_MODEL.predict(X_test)
            # Flatten if nested (CatBoost/Sklearn sometimes return [[val]] or [val])
            pred_val = raw_pred.ravel()[0]
            pred_idx = int(pred_val)
            
            label = CHAMPION_INFO["labels"][pred_idx] if CHAMPION_INFO["labels"] and pred_idx < len(CHAMPION_INFO["labels"]) else str(pred_idx)
            
            try:
                proba_raw = CHAMPION_MODEL.predict_proba(X_test)
                proba = proba_raw.ravel().tolist()
                # Ensure we only return as many probabilities as we have labels
                if CHAMPION_INFO["labels"] and len(proba) > len(CHAMPION_INFO["labels"]):
                    proba = proba[:len(CHAMPION_INFO["labels"])]
            except:
                proba = None
            return {
                "prediction": label, 
                "probabilities": proba, 
                "labels": CHAMPION_INFO["labels"]
            }
        else:
            raw_pred = CHAMPION_MODEL.predict(X_test)
            pred = float(raw_pred.ravel()[0])
            return {"prediction": pred}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=400)
