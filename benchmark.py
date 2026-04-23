import time
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)

DATA_PATH = "creditcard.csv"

# ── Load data ────────────────────────────────────────────────────────────────
t0 = time.time()
df = pd.read_csv(DATA_PATH)
load_time = time.time() - t0
print(f"[1/4] Data loaded: {len(df):,} rows in {load_time:.2f}s")

X = df.drop("Class", axis=1).values
y = df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── Train LightGBM ────────────────────────────────────────────────────────────
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
    "verbosity": -1,
    "n_jobs": -1,
}

train_ds = lgb.Dataset(X_train, label=y_train)
valid_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

print("[2/4] Training …")
t1 = time.time()
model = lgb.train(
    params,
    train_ds,
    num_boost_round=300,
    valid_sets=[valid_ds],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
)
train_time = time.time() - t1
print(f"      Training finished in {train_time:.2f}s  (best iter: {model.best_iteration})")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("[3/4] Evaluating …")
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    "load_time_s":     round(load_time, 3),
    "train_time_s":    round(train_time, 3),
    "best_iteration":  model.best_iteration,
    "auc_roc":         round(roc_auc_score(y_test, y_prob), 6),
    "accuracy":        round(accuracy_score(y_test, y_pred), 6),
    "f1_score":        round(f1_score(y_test, y_pred), 6),
    "precision":       round(precision_score(y_test, y_pred), 6),
    "recall":          round(recall_score(y_test, y_pred), 6),
}

# ── Inference latency ─────────────────────────────────────────────────────────
print("[4/4] Measuring inference latency …")
single_row = X_test[:1]
t2 = time.time()
for _ in range(100):
    model.predict(single_row)
latency_1row_ms = (time.time() - t2) / 100 * 1000

batch_1000 = X_test[:1000]
t3 = time.time()
model.predict(batch_1000)
throughput_1000_ms = (time.time() - t3) * 1000

metrics["inference_latency_1row_ms"]    = round(latency_1row_ms, 4)
metrics["inference_throughput_1000rows_ms"] = round(throughput_1000_ms, 4)

# ── Output ────────────────────────────────────────────────────────────────────
print("\n========== BENCHMARK RESULTS ==========")
for k, v in metrics.items():
    print(f"  {k:<38} {v}")
print("=======================================\n")

with open("benchmark_result.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved → benchmark_result.json")
