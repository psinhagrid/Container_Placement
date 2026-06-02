"""Parallel training data collection — like Random Forest's bootstrap sampling.

Runs N simulation instances simultaneously, each with a different random seed.
Each worker explores a different subset of yard states → diverse training data.
Combines all worker outputs and retrains XGBoost after each round.

Why this helps:
  - N workers in parallel = N× faster data collection vs sequential loop
  - Different seeds → different random exploration choices → different yard states
  - Model trains on N× more diverse scenarios per time unit
  - Reduces overfitting to any single simulation trajectory

Usage:
  python -m solution.parallel_collector --workers 3 --rounds 5

Example (3 workers, 5 rounds):
  Round 1: 3 sims × 6K rows = 18K new rows  (total: 18K)
  Round 2: 3 sims × 6K rows = 18K new rows  (total: 36K)
  Round 5: total: 90K rows — same time as 5 sequential iterations
"""

import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ACCUM_CSV   = "data/train/placement_features_accum.csv"
MODEL_PATH  = "solution/xgb_model.pkl"
BACKUP_PATH = "solution/xgb_model_backup.pkl"
RESULTS_LOG = "results/parallel_log.json"
TEMP_DIR    = "data/train/parallel_tmp"

FEATURES = [
    "stack_height", "top_etd_gap_days",
    "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "unsafe_count",
    "intra_vessel_rank",
    "unsafe_rank_count", "rank_gap_to_top",
    "unsafe_x_height", "min_height_pct",
    "hours_until_load", "same_group_in_stack", "initial_below_count",
]


# ── Worker function (must be top-level for multiprocessing) ────────────────────

def run_worker(worker_id: int, seed: int, output_path: str,
               shuffle_state: bool = True) -> Tuple[float, int]:
    """Run one collection simulation with a unique random seed and shuffled initial state.

    Each worker sees:
      1. A different shuffled initial yard state (different starting configuration)
      2. Different exploration choices (different random seed)

    This gives genuinely diverse training scenarios, not just different exploration paths
    through the same yard. The model learns universal placement principles.
    """
    import random
    random.seed(seed)   # unique seed → unique exploration choices

    import json as _json
    from src.simulator import Simulator
    from src.yard_state import YardState
    from src.event_reader import read_events
    from solution.xgb_collector import XGBCollector

    with open("data/yard_layout.json") as f:
        yard_layout = _json.load(f)
    with open("data/train/initial_state.json") as f:
        original_state = _json.load(f)

    # KEY CHANGE: each worker starts from a different shuffled initial state
    if shuffle_state:
        from solution.initial_state_shuffler import shuffle_initial_state
        initial_state = shuffle_initial_state(original_state, seed=seed)
        print(f"  [Worker {worker_id}] Using shuffled initial state (seed={seed})")
    else:
        initial_state = original_state

    yard = YardState(yard_layout)
    yard.load_initial_state(initial_state)
    strategy = XGBCollector()
    strategy.initialize(yard_layout, initial_state)

    from src.event_reader import read_events
    events = read_events("data/train/events.jsonl")
    sim = Simulator(yard, strategy, verbose=False)
    stats = sim.run(events)

    # Save to worker-specific path (no conflicts with other workers)
    strategy.save_to_path(output_path)

    return stats.reshuffles_per_retrieval, len(strategy._training_rows)


# ── Training ───────────────────────────────────────────────────────────────────

def retrain(accum_csv: str) -> float:
    df = pd.read_csv(accum_csv)
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()
    print(f"  Training on {len(df):,} rows")

    X = df[FEATURES]
    y = df["reshuffles"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, BACKUP_PATH)

    model = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, eval_metric="rmse",
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_rmse = float(np.sqrt(((y_val.values - model.predict(X_val)) ** 2).mean()))
    joblib.dump(model, MODEL_PATH)
    return val_rmse


def run_simulation(strategy_path: str) -> float:
    import json as _json
    from src.simulator import Simulator
    from src.yard_state import YardState
    from src.event_reader import read_events
    import importlib

    with open("data/yard_layout.json") as f:
        yard_layout = _json.load(f)
    with open("data/train/initial_state.json") as f:
        initial_state = _json.load(f)

    module_path, class_name = strategy_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    StrategyClass = getattr(module, class_name)

    yard = YardState(yard_layout)
    yard.load_initial_state(initial_state)
    strategy = StrategyClass()
    strategy.initialize(yard_layout, initial_state)

    from src.event_reader import read_events
    events = read_events("data/train/events.jsonl")
    sim = Simulator(yard, strategy, verbose=False)
    stats = sim.run(events)

    if hasattr(strategy, "_save"):
        strategy._save()

    return stats.reshuffles_per_retrieval


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",    type=int, default=3,
                        help="Parallel simulation workers per round")
    parser.add_argument("--rounds",     type=int, default=5,
                        help="Number of collection+retrain rounds")
    parser.add_argument("--data-dir",   default="data/train")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable initial state shuffling (use original for all workers)")
    args = parser.parse_args()

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log = []

    print(f"Parallel collector: {args.workers} workers × {args.rounds} rounds")
    print(f"Each round generates ~{args.workers * 6000:,} new training rows\n")

    # Baseline
    baseline = run_simulation("solution.xgb_strategy.XGBStrategy")
    print(f"Baseline: {baseline:.4f}\n")
    best_score = baseline

    for rnd in range(1, args.rounds + 1):
        print(f"{'='*60}")
        print(f"ROUND {rnd}/{args.rounds}  ({args.workers} parallel workers)")
        print(f"{'='*60}")

        worker_files = [
            os.path.join(TEMP_DIR, f"worker_{rnd}_{i}.csv")
            for i in range(args.workers)
        ]
        # Different seeds each round, different seeds each worker
        seeds = [rnd * 1000 + i for i in range(args.workers)]

        t0 = time.time()
        scores, row_counts = [], []

        shuffle = not args.no_shuffle
        if shuffle:
            print(f"  Using shuffled initial states (different yard per worker)")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_worker, i, seeds[i], worker_files[i], shuffle): i
                for i in range(args.workers)
            }
            for future in as_completed(futures):
                wid = futures[future]
                score, rows = future.result()
                scores.append(score)
                row_counts.append(rows)
                print(f"  Worker {wid}: score={score:.4f}, rows={rows}")

        elapsed = time.time() - t0
        avg_score = sum(scores) / len(scores)
        total_new = sum(row_counts)
        print(f"  Avg collection score: {avg_score:.4f}  ({elapsed:.1f}s)")

        # Combine worker files → append to accumulated CSV
        dfs = []
        for wf in worker_files:
            if os.path.exists(wf):
                dfs.append(pd.read_csv(wf))
                os.remove(wf)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            file_exists = os.path.exists(ACCUM_CSV)
            combined.to_csv(ACCUM_CSV, mode="a", header=not file_exists, index=False)
            total_rows = len(pd.read_csv(ACCUM_CSV))
            print(f"  +{total_new} rows appended → {total_rows:,} total")

        # Retrain
        print(f"  Retraining XGBoost...")
        val_rmse = retrain(ACCUM_CSV)
        total_rows = len(pd.read_csv(ACCUM_CSV))
        print(f"  Val RMSE: {val_rmse:.4f}")

        # Evaluate
        print(f"  Evaluating...")
        new_score = run_simulation("solution.xgb_strategy.XGBStrategy")
        improved = new_score < best_score
        print(f"  Score: {new_score:.4f} (best: {best_score:.4f})"
              f"  {'✓ IMPROVED' if improved else '✗ reverting'}")

        if improved:
            best_score = new_score
        else:
            if os.path.exists(BACKUP_PATH):
                shutil.copy(BACKUP_PATH, MODEL_PATH)

        log.append({
            "round": rnd, "workers": args.workers,
            "avg_collection_score": avg_score,
            "val_rmse": val_rmse, "eval_score": new_score,
            "improved": improved, "best_score": best_score,
            "total_rows": total_rows,
        })
        with open(RESULTS_LOG, "w") as f:
            json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE — Baseline: {baseline:.4f}  Best: {best_score:.4f}")
    print(f"Improvement: {(baseline - best_score) / baseline * 100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
