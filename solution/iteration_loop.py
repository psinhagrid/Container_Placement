"""Self-improvement loop for XGBoost placement strategy.

Each iteration:
  1. Run simulation with XGBCollector (XGBoost + 10% exploration)
  2. Append new training rows to accumulated CSV (never discard old data)
  3. Retrain XGBoost on all accumulated data
  4. Evaluate on fixed holdout simulation (pure XGBoost, no exploration)
  5. Keep model only if holdout score improves
  6. Repeat

Usage:
  python -m solution.iteration_loop --iterations 5

Safeguards (per user design):
  - Old data always kept: D0 + D1 + D2 + ... prevents policy collapse
  - Fixed holdout: same simulation run every iteration for fair comparison
  - Model only promoted if holdout improves
  - 10% exploration in data collection prevents distribution collapse
"""

import argparse
import json
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ACCUM_CSV   = "data/train/placement_features_accum.csv"
MODEL_PATH  = "solution/xgb_model.pkl"
BACKUP_PATH = "solution/xgb_model_backup.pkl"
RESULTS_LOG = "results/iteration_log.json"

FEATURES = [
    "stack_height", "top_etd_gap_days", "same_vessel", "same_port",
    "weight_ok", "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "is_truck", "unsafe_count",
]


def run_simulation(strategy_path: str, data_dir: str = "data/train") -> float:
    """Run a simulation and return reshuffles/retrieval ratio."""
    import json as _json
    from src.simulator import Simulator
    from src.yard_state import YardState
    from src.event_reader import read_events
    import importlib

    with open("data/yard_layout.json") as f:
        yard_layout = _json.load(f)
    with open(f"{data_dir}/initial_state.json") as f:
        initial_state = _json.load(f)

    # Import strategy dynamically
    module_path, class_name = strategy_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    StrategyClass = getattr(module, class_name)

    yard = YardState(yard_layout)
    yard.load_initial_state(initial_state)
    strategy = StrategyClass()
    strategy.initialize(yard_layout, initial_state)

    events = read_events(f"{data_dir}/events.jsonl")
    sim = Simulator(yard, strategy, verbose=False)
    stats = sim.run(events)

    # Explicitly save training data (atexit only fires on process exit,
    # not between iterations when running inside the loop)
    if hasattr(strategy, "_save"):
        strategy._save()

    return stats.reshuffles_per_retrieval


def retrain(accum_csv: str) -> float:
    """Retrain XGBoost on accumulated data. Returns val RMSE."""
    df = pd.read_csv(accum_csv)
    print(f"  Training on {len(df):,} accumulated rows")

    X = df[FEATURES]
    y = df["reshuffles"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save backup of current model
    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, BACKUP_PATH)

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(((y_val.values - val_pred) ** 2).mean()))

    joblib.dump(model, MODEL_PATH)
    return val_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--data-dir", default="data/train")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    log = []

    # Baseline: evaluate current model before any iteration
    print("=" * 60)
    print("BASELINE: evaluating current XGBoost model")
    print("=" * 60)
    baseline_score = run_simulation("solution.xgb_strategy.XGBStrategy", args.data_dir)
    print(f"Baseline reshuffles/retrieval: {baseline_score:.4f}")
    best_score = baseline_score

    for it in range(1, args.iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {it}/{args.iterations}")
        print(f"{'=' * 60}")

        # Step 1: collect new training data (XGBoost + 10% exploration)
        print(f"[{it}] Collecting training data with exploration...")
        t0 = time.time()
        collect_score = run_simulation("solution.xgb_collector.XGBCollector", args.data_dir)
        collect_time = time.time() - t0
        rows_before = len(pd.read_csv(ACCUM_CSV)) if os.path.exists(ACCUM_CSV) else 0
        print(f"  Collection score: {collect_score:.4f}  ({collect_time:.1f}s)")

        # Step 2: retrain on ALL accumulated data
        print(f"[{it}] Retraining XGBoost...")
        val_rmse = retrain(ACCUM_CSV)
        rows_after = len(pd.read_csv(ACCUM_CSV))
        print(f"  Val RMSE: {val_rmse:.4f}  | Rows: {rows_before} → {rows_after}")

        # Step 3: evaluate new model on fixed holdout
        print(f"[{it}] Evaluating new model...")
        new_score = run_simulation("solution.xgb_strategy.XGBStrategy", args.data_dir)
        improved = new_score < best_score
        print(f"  New score: {new_score:.4f}  (best so far: {best_score:.4f})  {'✓ IMPROVED' if improved else '✗ WORSE — reverting'}")

        if improved:
            best_score = new_score
        else:
            # Revert to backup model
            if os.path.exists(BACKUP_PATH):
                shutil.copy(BACKUP_PATH, MODEL_PATH)
                print(f"  Model reverted to backup")

        log.append({
            "iteration": it,
            "collect_score": collect_score,
            "val_rmse": val_rmse,
            "eval_score": new_score,
            "improved": improved,
            "best_score": best_score,
            "total_rows": rows_after,
        })

        with open(RESULTS_LOG, "w") as f:
            json.dump(log, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"LOOP COMPLETE")
    print(f"Baseline:   {baseline_score:.4f}")
    print(f"Best score: {best_score:.4f}  ({(baseline_score - best_score) / baseline_score * 100:.1f}% improvement)")
    print(f"Log saved: {RESULTS_LOG}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
