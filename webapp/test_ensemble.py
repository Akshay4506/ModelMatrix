import sys
sys.path.insert(0, "webapp")
import pandas as pd
from sklearn.datasets import load_breast_cancer
from benchmark import run_benchmark

d = load_breast_cancer(as_frame=True)
df = d.data.copy()
df["target"] = d.target

print("Running benchmark with ensembles...")
result = run_benchmark(df, "target")

print("Task:", result["task"])
print()

for name, r in result["results"].items():
    if "error" in r:
        msg = r["error"][:60]
        print(f"  {name:22s}  ERROR: {msg}")
    else:
        auc = r["mean"].get("roc_auc", 0)
        print(f"  {name:22s}  ROC-AUC={auc:.4f}")

print()
print("Ensemble info:")
for name, info in result["ensemble_info"].items():
    print(f"  {name}: type={info['type']}, components={info['components']}")

print()
best = result["recommendation"]["recommendations"]["best_overall"]
print("Best Overall:", best["model"], "| score:", round(best["score"], 4))
