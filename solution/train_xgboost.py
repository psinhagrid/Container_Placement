"""Train XGBoost model on placement features collected from the heuristic simulation.

Run:
    python -m solution.train_xgboost

Reads:  data/train/placement_features.csv
Writes: solution/xgb_model.pkl
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

FEATURES = [
    "stack_height", "top_etd_gap_days",
    "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "unsafe_count",
    "intra_vessel_rank",
    "unsafe_rank_count", "rank_gap_to_top",
    "unsafe_x_height", "min_height_pct",
    "hours_until_load", "same_group_in_stack", "initial_below_count",
]
TARGET = "reshuffles"

DATA_PATH  = "data/train/placement_features.csv"
MODEL_PATH = "solution/xgb_model.pkl"


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} training rows from {DATA_PATH}")
    print(f"\nReshuffle distribution:\n{df[TARGET].value_counts().sort_index()}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train):,}  Val: {len(X_val):,}")

    # ── Train ──────────────────────────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=1000,       # high ceiling — early stopping will cut it
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
        early_stopping_rounds=30,  # stop if val RMSE doesn't improve for 30 rounds
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"Best iteration: {model.best_iteration} | Best val RMSE: {model.best_score:.4f}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_pred) ** 0.5
    mae  = np.abs(y_val.values - val_pred).mean()

    print(f"\nValidation RMSE : {rmse:.4f}")
    print(f"Validation MAE  : {mae:.4f}")
    print(f"Baseline MAE (predict mean): {np.abs(y_val - y_val.mean()).mean():.4f}")

    # ── Feature importance ─────────────────────────────────────────────────────
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"\nFeature importances:")
    for feat, score in imp.items():
        bar = "█" * int(score * 40)
        print(f"  {feat:<22} {score:.4f}  {bar}")

    # ── Save ───────────────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
