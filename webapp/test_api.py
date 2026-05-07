import requests, json, time

print("Running benchmark on breast_cancer (569 rows, 30 features)...")
t0 = time.time()

with open("webapp/test_upload.csv", "rb") as f:
    r = requests.post(
        "http://localhost:8000/benchmark",
        files={"file": ("test.csv", f, "text/csv")},
        data={"target_col": "target"},
        timeout=300
    )

elapsed = time.time() - t0

if r.status_code == 200:
    d = r.json()
    task = d["task"]
    pk   = "roc_auc" if task == "classification" else "r2"
    print(f"Task: {task}  |  Time: {elapsed:.1f}s\n")

    for model, res in d["results"].items():
        if "error" in res:
            err = res["error"]
            print(f"  {model:15s}  ERROR: {err}")
        else:
            score = res["mean"].get(pk, res["mean"].get("accuracy", 0))
            ft    = res["mean"]["fit_time"]
            print(f"  {model:15s}  {pk}={score:.4f}  fit_time={ft:.3f}s")

    print()
    rec = d["recommendation"]["recommendations"]
    print("RECOMMENDATION:")
    print(f"  Best Overall:     {rec['best_overall']['model']}")
    print(f"  Best Accuracy:    {rec['best_accuracy']['model']}")
    print(f"  Fastest:          {rec['best_speed']['model']}")
    print(f"  Most Consistent:  {rec['best_consistency']['model']}")
    print(f"  Production:       {rec['production']['model']}")
else:
    print("ERROR", r.status_code, r.text[:500])
