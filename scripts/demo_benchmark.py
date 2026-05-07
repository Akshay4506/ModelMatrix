"""
SAP RPT-1 Benchmarking Demo
============================
Self-contained demo: runs XGBoost, LightGBM, CatBoost, and SAP RPT-1 (simulated)
on classic sklearn datasets (Iris, Breast Cancer, Diabetes regression) using
5-fold cross-validation. Saves JSON results and a beautiful HTML report.

Run from repo root:
    python scripts/demo_benchmark.py
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def timer():
    return time.perf_counter()


def load_datasets():
    datasets = {}

    # 1. Iris  (multi-class classification)
    d = load_iris(as_frame=True)
    datasets["iris"] = {
        "X": d.data,
        "y": d.target,
        "task": "classification",
        "desc": "Iris flower species (3 classes, 150 rows, 4 features)"
    }

    # 2. Breast Cancer (binary classification)
    d = load_breast_cancer(as_frame=True)
    datasets["breast_cancer"] = {
        "X": d.data,
        "y": d.target,
        "task": "classification",
        "desc": "Wisconsin Breast Cancer (binary, 569 rows, 30 features)"
    }

    # 3. Diabetes (regression)
    d = load_diabetes(as_frame=True)
    datasets["diabetes"] = {
        "X": d.data,
        "y": d.target,
        "task": "regression",
        "desc": "Diabetes progression (regression, 442 rows, 10 features)"
    }

    return datasets


# ─────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────

def build_xgboost(task):
    import xgboost as xgb
    if task == "classification":
        return xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=RANDOM_STATE, use_label_encoder=False,
                                  eval_metric="logloss", verbosity=0)
    return xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                             random_state=RANDOM_STATE, verbosity=0)


def build_lightgbm(task):
    import lightgbm as lgb
    if task == "classification":
        return lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1,
                                   random_state=RANDOM_STATE, verbose=-1)
    return lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1,
                              random_state=RANDOM_STATE, verbose=-1)


def build_catboost(task):
    from catboost import CatBoostClassifier, CatBoostRegressor
    if task == "classification":
        return CatBoostClassifier(iterations=100, learning_rate=0.1,
                                   random_state=RANDOM_STATE, verbose=False)
    return CatBoostRegressor(iterations=100, learning_rate=0.1,
                              random_state=RANDOM_STATE, verbose=False)


class SAPSimulator:
    """
    SAP RPT-1 Simulator.
    Mimics SAP RPT-1's in-context learning behaviour using a fast
    k-NN retrieval backbone (conceptually similar to how RPT-1 retrieves
    nearest context rows and predicts via its pretrained head).
    
    NOTE: This is a *demonstration substitute* for the real SAP RPT-1 OSS
    model which requires a gated HuggingFace token + pip install of the
    SAP-samples package. The real wrapper is in code/models/sap_rpt1_hf_wrapper.py.
    """
    def __init__(self, task, k=15):
        self.task = task
        self.k = k
        if task == "classification":
            self.model = KNeighborsClassifier(n_neighbors=k)
        else:
            self.model = KNeighborsRegressor(n_neighbors=k)
        self.le = LabelEncoder() if task == "classification" else None

    def fit(self, X, y):
        if self.task == "classification":
            y_enc = self.le.fit_transform(y)
            self.model.fit(X, y_enc)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        if self.task == "classification":
            return self.le.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        return self.model.predict_proba(X)


MODELS = {
    "XGBoost":   build_xgboost,
    "LightGBM":  build_lightgbm,
    "CatBoost":  build_catboost,
    "SAP-RPT1 (sim)": lambda task: SAPSimulator(task),
}


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def eval_fold_classification(model, X_train, y_train, X_val, y_val):
    t0 = timer()
    model.fit(X_train, y_train)
    fit_time = timer() - t0

    t0 = timer()
    y_pred = model.predict(X_val)
    pred_time = timer() - t0

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)

    try:
        proba = model.predict_proba(X_val)
        n_cls = len(np.unique(y_val))
        if n_cls == 2:
            auc = roc_auc_score(y_val, proba[:, 1])
        else:
            auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "f1_macro": f1, "roc_auc": auc,
            "fit_time": fit_time, "pred_time": pred_time}


def eval_fold_regression(model, X_train, y_train, X_val, y_val):
    t0 = timer()
    model.fit(X_train, y_train)
    fit_time = timer() - t0

    t0 = timer()
    y_pred = model.predict(X_val)
    pred_time = timer() - t0

    r2  = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    return {"r2": r2, "mae": mae, "fit_time": fit_time, "pred_time": pred_time}


def run_cv(model_fn, dataset_name, ds):
    X, y, task = ds["X"], ds["y"], ds["task"]

    if task == "classification":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X, y))
    else:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X))

    fold_results = []
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn(task)

        if task == "classification":
            fold_results.append(eval_fold_classification(model, X_tr, y_tr, X_val, y_val))
        else:
            fold_results.append(eval_fold_regression(model, X_tr, y_tr, X_val, y_val))

    df = pd.DataFrame(fold_results)
    return {"mean": df.mean().to_dict(), "std": df.std().to_dict(), "folds": fold_results}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  SAP RPT-1 Benchmarking Demo")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    datasets = load_datasets()
    all_results = {}

    for ds_name, ds in datasets.items():
        print(f"\n[DATASET] {ds_name}  ({ds['desc']})")
        all_results[ds_name] = {"task": ds["task"], "models": {}}

        for model_name, model_fn in MODELS.items():
            try:
                print(f"   >> Running {model_name}...", end=" ", flush=True)
                t_total = timer()
                cv_res = run_cv(model_fn, ds_name, ds)
                elapsed = timer() - t_total

                all_results[ds_name]["models"][model_name] = cv_res
                task = ds["task"]
                if task == "classification":
                    primary = cv_res["mean"].get("roc_auc", cv_res["mean"]["accuracy"])
                    print(f"ROC-AUC={primary:.4f}  ({elapsed:.1f}s)")
                else:
                    primary = cv_res["mean"]["r2"]
                    print(f"R²={primary:.4f}  ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                all_results[ds_name]["models"][model_name] = {"error": str(e)}

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"demo_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] JSON saved -> {json_path}")

    # Generate HTML dashboard
    html_path = Path(__file__).parent.parent / "results" / f"demo_dashboard_{ts}.html"
    generate_html(all_results, html_path, ts)
    print(f"[OK] HTML dashboard -> {html_path}")
    print("\nOpen the HTML file in your browser to see the results!\n")

    return all_results, html_path


# ─────────────────────────────────────────────
# HTML Report Generator
# ─────────────────────────────────────────────

def color_for_metric(val, task):
    """Return a CSS color class based on metric value."""
    if task == "classification":  # ROC-AUC or Accuracy
        if val >= 0.97: return "excellent"
        if val >= 0.92: return "good"
        if val >= 0.85: return "fair"
        return "poor"
    else:  # R²
        if val >= 0.55: return "excellent"
        if val >= 0.40: return "good"
        if val >= 0.20: return "fair"
        return "poor"


def generate_html(results, out_path, ts):
    MODEL_COLORS = {
        "XGBoost":        "#f59e0b",
        "LightGBM":       "#10b981",
        "CatBoost":       "#6366f1",
        "SAP-RPT1 (sim)": "#ec4899",
    }

    # Build chart data JSON
    chart_datasets = {}
    for ds_name, ds_data in results.items():
        task = ds_data["task"]
        metric = "roc_auc" if task == "classification" else "r2"
        fallback = "accuracy"
        chart_datasets[ds_name] = {
            "task": task,
            "models": {},
        }
        for m_name, m_data in ds_data["models"].items():
            if "error" in m_data:
                continue
            val = m_data["mean"].get(metric, m_data["mean"].get(fallback, 0))
            std = m_data["std"].get(metric, m_data["std"].get(fallback, 0))
            chart_datasets[ds_name]["models"][m_name] = {"val": round(val, 4), "std": round(std, 4)}

    chart_json = json.dumps(chart_datasets)
    colors_json = json.dumps(MODEL_COLORS)

    # Table rows
    table_rows = ""
    for ds_name, ds_data in results.items():
        task = ds_data["task"]
        metric_key = "roc_auc" if task == "classification" else "r2"
        for m_name, m_data in ds_data["models"].items():
            if "error" in m_data:
                table_rows += f"""<tr><td>{ds_name}</td><td>{m_name}</td>
                    <td>{task}</td><td colspan="4" style="color:#ef4444">ERROR: {m_data['error'][:60]}</td></tr>"""
                continue
            acc  = m_data["mean"].get("accuracy", "-")
            f1   = m_data["mean"].get("f1_macro", "-")
            auc  = m_data["mean"].get("roc_auc", "-")
            r2   = m_data["mean"].get("r2", "-")
            mae  = m_data["mean"].get("mae", "-")
            ft   = m_data["mean"].get("fit_time", 0)
            prim = m_data["mean"].get(metric_key, m_data["mean"].get("accuracy", 0))
            cls  = color_for_metric(prim, task)

            def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "-"
            color = MODEL_COLORS.get(m_name, "#888")
            dot = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:6px"></span>'
            table_rows += f"""<tr>
                <td><strong>{ds_name}</strong></td>
                <td>{dot}{m_name}</td>
                <td><span class="badge {'badge-clf' if task=='classification' else 'badge-reg'}">{task}</span></td>
                <td class="metric {cls}">{fmt(acc) if task=='classification' else '-'}</td>
                <td class="metric {cls}">{fmt(f1) if task=='classification' else '-'}</td>
                <td class="metric {cls}">{fmt(auc) if task=='classification' else '-'}</td>
                <td class="metric {cls}">{'-' if task=='classification' else fmt(r2)}</td>
                <td class="metric">{fmt(mae) if task=='regression' else '-'}</td>
                <td class="metric">{ft:.3f}s</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SAP RPT-1 Benchmarking Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:#0a0f1e;color:#e2e8f0;min-height:100vh}}

/* Hero */
.hero{{background:linear-gradient(135deg,#1a1f3a 0%,#0d1226 50%,#1a0a2e 100%);padding:60px 40px 40px;text-align:center;border-bottom:1px solid #1e2a4a;position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at center,rgba(99,102,241,.12) 0%,transparent 60%);pointer-events:none}}
.hero h1{{font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#ec4899,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:12px}}
.hero p{{color:#94a3b8;font-size:1.1rem;max-width:700px;margin:0 auto 20px}}
.badge-info{{display:inline-block;background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.4);color:#818cf8;padding:4px 14px;border-radius:999px;font-size:.8rem;margin:4px}}

/* Layout */
.container{{max-width:1400px;margin:0 auto;padding:40px 24px}}
.section-title{{font-size:1.4rem;font-weight:700;color:#f1f5f9;margin-bottom:24px;display:flex;align-items:center;gap:10px}}
.section-title::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(99,102,241,.4),transparent)}}

/* Cards */
.grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:40px}}
@media(max-width:900px){{.grid-3{{grid-template-columns:1fr}}}}
.card{{background:linear-gradient(145deg,#111827,#0f172a);border:1px solid #1e2a4a;border-radius:16px;padding:24px;position:relative;overflow:hidden;transition:transform .2s,border-color .2s}}
.card:hover{{transform:translateY(-3px);border-color:#374151}}
.card::after{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#6366f1,#ec4899)}}
.card h3{{font-size:.85rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}}
.card .value{{font-size:2.2rem;font-weight:800;color:#f1f5f9}}
.card .sub{{font-size:.85rem;color:#64748b;margin-top:4px}}

/* Charts */
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px;margin-bottom:40px}}
.chart-card{{background:linear-gradient(145deg,#111827,#0f172a);border:1px solid #1e2a4a;border-radius:16px;padding:24px}}
.chart-card h4{{font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:4px}}
.chart-card .sub{{font-size:.8rem;color:#64748b;margin-bottom:16px}}
canvas{{max-height:280px}}

/* Table */
.table-card{{background:linear-gradient(145deg,#111827,#0f172a);border:1px solid #1e2a4a;border-radius:16px;overflow:hidden;margin-bottom:40px}}
.table-header{{padding:20px 24px;border-bottom:1px solid #1e2a4a;display:flex;justify-content:space-between;align-items:center}}
.table-header h3{{font-size:1rem;font-weight:600;color:#e2e8f0}}
table{{width:100%;border-collapse:collapse}}
th{{padding:12px 16px;text-align:left;font-size:.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #1e2a4a;white-space:nowrap}}
td{{padding:12px 16px;font-size:.875rem;border-bottom:1px solid #0f172a;vertical-align:middle}}
tr:hover td{{background:rgba(255,255,255,.02)}}
.metric{{font-family:'Courier New',monospace;font-weight:600}}
.excellent{{color:#10b981}}
.good{{color:#6366f1}}
.fair{{color:#f59e0b}}
.poor{{color:#ef4444}}
.badge{{padding:3px 10px;border-radius:999px;font-size:.72rem;font-weight:600}}
.badge-clf{{background:rgba(99,102,241,.2);color:#818cf8;border:1px solid rgba(99,102,241,.3)}}
.badge-reg{{background:rgba(16,185,129,.2);color:#34d399;border:1px solid rgba(16,185,129,.3)}}

/* Legend */
.legend{{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:32px}}
.legend-item{{display:flex;align-items:center;gap:8px;font-size:.85rem;color:#94a3b8}}
.legend-dot{{width:12px;height:12px;border-radius:3px;flex-shrink:0}}

/* Note */
.note{{background:rgba(236,72,153,.08);border:1px solid rgba(236,72,153,.25);border-radius:12px;padding:16px 20px;margin-bottom:32px;font-size:.875rem;color:#f0abfc;line-height:1.6}}
.note strong{{color:#ec4899}}

/* Footer */
.footer{{text-align:center;padding:24px;color:#374151;font-size:.8rem;border-top:1px solid #1e2a4a}}
</style>
</head>
<body>

<div class="hero">
  <h1>🔬 SAP RPT-1 Benchmarking</h1>
  <p>Comparative evaluation of tabular machine learning models across classification and regression datasets</p>
  <span class="badge-info">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
  <span class="badge-info">{N_FOLDS}-Fold Cross-Validation</span>
  <span class="badge-info">Seed: {RANDOM_STATE}</span>
</div>

<div class="container">

  <div class="note">
    <strong>ℹ️ About SAP RPT-1 (sim):</strong> The real <em>SAP RPT-1 OSS</em> model is a 
    Retrieval-Pretrained Transformer for tabular data available at 
    <code>huggingface.co/SAP/sap-rpt-1-oss</code> — it requires a gated HuggingFace token and 
    <code>pip install git+https://github.com/SAP-samples/sap-rpt-1-oss.git</code>. 
    In this demo, <strong>SAP-RPT1 (sim)</strong> is a conceptually faithful substitute 
    (k-NN in-context retrieval, k=15) to demonstrate the pipeline without authentication. 
    See <code>code/models/sap_rpt1_hf_wrapper.py</code> for the real wrapper.
  </div>

  <!-- KPI cards -->
  <h2 class="section-title">📈 Summary Statistics</h2>
  <div class="grid-3" id="kpi-cards"></div>

  <!-- Legend -->
  <div class="legend" id="legend"></div>

  <!-- Charts -->
  <h2 class="section-title">📊 Model Comparison Charts</h2>
  <div class="chart-grid" id="charts"></div>

  <!-- Table -->
  <h2 class="section-title">📋 Full Results Table</h2>
  <div class="table-card">
    <div class="table-header">
      <h3>All Metrics (mean across {N_FOLDS} folds)</h3>
      <span style="color:#64748b;font-size:.8rem">↑ higher is better (except MAE)</span>
    </div>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>Dataset</th><th>Model</th><th>Task</th>
        <th>Accuracy</th><th>F1-Macro</th><th>ROC-AUC</th>
        <th>R²</th><th>MAE</th><th>Fit Time</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    </div>
  </div>

</div>

<div class="footer">SAP RPT-1 Benchmarking · Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

<script>
const DATA = {chart_json};
const COLORS = {colors_json};

const modelNames = Object.keys(COLORS);

// Legend
const legendEl = document.getElementById('legend');
modelNames.forEach(m => {{
  legendEl.innerHTML += `<div class="legend-item">
    <div class="legend-dot" style="background:${{COLORS[m]}}"></div>
    <span>${{m}}</span>
  </div>`;
}});

// KPI cards
const kpiEl = document.getElementById('kpi-cards');
const dsNames = Object.keys(DATA);
dsNames.forEach(ds => {{
  const task = DATA[ds].task;
  const metric = task === 'classification' ? 'roc_auc' : 'r2';
  const label  = task === 'classification' ? 'Best ROC-AUC' : 'Best R²';
  const models = DATA[ds].models;
  let best = {{val:0, name:''}};
  Object.entries(models).forEach(([m, v]) => {{ if(v.val > best.val) best = {{val:v.val, name:m}}; }});
  const color = COLORS[best.name] || '#6366f1';
  kpiEl.innerHTML += `<div class="card">
    <h3>${{ds}}</h3>
    <div class="value" style="color:${{color}}">${{best.val.toFixed(4)}}</div>
    <div class="sub">${{label}} · ${{best.name}} · ${{task}}</div>
  </div>`;
}});

// Charts — one per dataset
const chartsEl = document.getElementById('charts');
dsNames.forEach(ds => {{
  const task = DATA[ds].task;
  const metric = task === 'classification' ? 'roc_auc' : 'r2';
  const metricLabel = task === 'classification' ? 'ROC-AUC' : 'R²';
  const models = DATA[ds].models;
  const labels = Object.keys(models);
  const vals = labels.map(m => models[m].val);
  const errs = labels.map(m => models[m].std);
  const bgColors = labels.map(m => COLORS[m] || '#888');

  const div = document.createElement('div');
  div.className = 'chart-card';
  div.innerHTML = `<h4>${{ds}}</h4><div class="sub">${{task}} · ${{metricLabel}} (mean ± std over {N_FOLDS} folds)</div><canvas id="chart-${{ds}}"></canvas>`;
  chartsEl.appendChild(div);

  new Chart(document.getElementById(`chart-${{ds}}`), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: metricLabel,
        data: vals,
        backgroundColor: bgColors.map(c => c + 'cc'),
        borderColor: bgColors,
        borderWidth: 2,
        borderRadius: 8,
        errorBars: {{}}
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => `${{metricLabel}}: ${{ctx.parsed.y.toFixed(4)}} ± ${{errs[ctx.dataIndex].toFixed(4)}}`
          }}
        }}
      }},
      scales: {{
        y: {{
          beginAtZero: false,
          min: Math.max(0, Math.min(...vals) - 0.1),
          max: Math.min(1.0, Math.max(...vals) + 0.05),
          grid: {{ color: '#1e2a4a' }},
          ticks: {{ color: '#64748b', font: {{ size: 11 }} }}
        }},
        x: {{
          grid: {{ display: false }},
          ticks: {{ color: '#94a3b8', font: {{ size: 12 }} }}
        }}
      }}
    }}
  }});
}});
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
