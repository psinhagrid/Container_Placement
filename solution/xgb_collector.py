"""XGBoost strategy with exploration + training data collection.

Used in the self-improvement loop. Same as XGBStrategy but:
  - EXPLORE_RATE (10%) of placements pick a RANDOM candidate from min-height stacks
    instead of XGBoost's top choice. This prevents the feedback loop from collapsing
    into a narrow distribution and ensures training data has variance.
  - Collects (features, reshuffles) pairs for retraining.
  - Appends to accumulated CSV (never overwrites previous iteration data).
"""

import atexit
import csv
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Set

import joblib
import numpy as np
import pandas as pd

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK   = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}
MODEL_PATH    = "solution/xgb_model.pkl"
ONE_HOUR      = 3_600.0
EXPLORE_RATE  = 0.10          # 10% random exploration
ACCUM_CSV     = "data/train/placement_features_accum.csv"

FEATURES = [
    "stack_height", "top_etd_gap_days", "same_vessel", "same_port",
    "weight_ok", "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "is_truck", "unsafe_count",
]
FEATURE_COLS = FEATURES + ["reshuffles"]


class XGBCollector(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache:         Dict[str, float] = {}
        self._container_etd:     Dict[str, float] = {}
        self._stack_containers:  Dict[tuple, Set[str]] = {}

        for c in initial_state.get("containers", []):
            cid = c["container_id"]
            p   = c["position"]
            key = (p["block"], p["bay"], p["row"])
            etd = self._etd(c.get("departure_time", ""))
            self._container_etd[cid] = etd
            self._stack_containers.setdefault(key, set()).add(cid)

        # Load current model
        self._model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        if self._model:
            print(f"[XGBCollector] Model loaded, explore_rate={EXPLORE_RATE:.0%}")
        else:
            print("[XGBCollector] No model found — pure exploration")

        # Data collection
        self._placement_features: Dict[str, dict] = {}
        self._training_rows: List[dict] = []
        atexit.register(self._save)

    def _etd(self, s: str) -> float:
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except Exception:
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    def _unsafe_count(self, block: str, bay: int, row: int, inc_etd: float) -> int:
        key = (block, bay, row)
        return sum(
            1 for cid in self._stack_containers.get(key, set())
            if self._container_etd.get(cid, float("inf")) < inc_etd - ONE_HOUR
        )

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd      = self._etd(event.departure_time)
        inc_rank     = WEIGHT_RANK.get(event.weight_class, 2)
        placement_ts = self._etd(event.timestamp)
        is_truck     = int(event.vessel_id in TRUCK_VESSELS)
        days_until   = max(0.0, (inc_etd - placement_ts) / 86_400) if inc_etd != float("inf") else 0.0

        # Cache block occupancies
        block_occ: Dict[str, float] = {}
        for bn, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(bn)
            block_occ[bn] = occ / cap if cap > 0 else 0.0

        # Find global minimum height
        min_h = float("inf")
        for bn, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(bn, bay, row)
                    if h < bi.tiers:
                        min_h = min(min_h, h)
                        if min_h == 0:
                            break
                if min_h == 0: break
            if min_h == 0: break

        if min_h == float("inf"):
            return self._fallback(yard_state)

        # Collect candidates at min height
        candidates = [
            (bn, bay, row)
            for bn, bi in yard_state.blocks.items()
            for bay in range(1, bi.bays + 1)
            for row in range(1, bi.rows + 1)
            if yard_state.get_stack_height(bn, bay, row) == min_h
            and min_h < yard_state.blocks[bn].tiers
        ]

        if not candidates:
            return self._fallback(yard_state)

        # Build feature rows for all candidates
        rows, positions = [], []
        for bn, bay, row in candidates:
            h  = min_h
            uc = self._unsafe_count(bn, bay, row, inc_etd)

            top_etd_gap, same_vessel, same_port, weight_ok, top_rank = 0.0, 0, 0, 1, 0
            if h > 0:
                top_cid  = yard_state.get_container_at(bn, bay, row, h)
                top_info = yard_state.get_container_info(top_cid) if top_cid else None
                if top_info:
                    te = self._etd(top_info.departure_time)
                    top_rank    = WEIGHT_RANK.get(top_info.weight_class, 2)
                    top_etd_gap = round((te - inc_etd) / 86_400, 4) if te != float("inf") else 0.0
                    same_vessel = int(top_info.vessel_id == event.vessel_id)
                    same_port   = int(top_info.port_of_discharge == event.port_of_discharge)
                    if event.vessel_id not in TRUCK_VESSELS:
                        weight_ok = int(inc_rank >= top_rank)

            rows.append({
                "stack_height":     h,
                "top_etd_gap_days": top_etd_gap,
                "same_vessel":      same_vessel,
                "same_port":        same_port,
                "weight_ok":        weight_ok,
                "weight_rank_inc":  inc_rank,
                "weight_rank_top":  top_rank,
                "block_occ":        round(block_occ[bn], 4),
                "days_until_dep":   round(days_until, 4),
                "is_truck":         is_truck,
                "unsafe_count":     uc,
            })
            positions.append(Position(bn, bay, row, h + 1))

        # Choose: XGBoost best OR random exploration
        if self._model is not None and random.random() > EXPLORE_RATE:
            X    = pd.DataFrame(rows, columns=FEATURES)
            pred = self._model.predict(X)
            idx  = int(np.argmin(pred))
        else:
            idx = random.randrange(len(candidates))   # exploration

        chosen_pos = positions[idx]
        chosen_feat = rows[idx]

        # Store features for this placement
        self._placement_features[event.container_id] = chosen_feat

        # Update stack tracking
        key = (chosen_pos.block, chosen_pos.bay, chosen_pos.row)
        self._container_etd[event.container_id] = inc_etd
        self._stack_containers.setdefault(key, set()).add(event.container_id)

        return chosen_pos

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        feats = self._placement_features.pop(container_id, None)
        if feats is not None:
            row = dict(feats)
            row["reshuffles"] = reshuffles
            self._training_rows.append(row)

        key = (position.block, position.bay, position.row)
        self._stack_containers.get(key, set()).discard(container_id)
        self._container_etd.pop(container_id, None)

    def _fallback(self, yard_state: YardState) -> Position:
        for bn, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(bn, bay, row)
                    if h < bi.tiers:
                        return Position(bn, bay, row, h + 1)
        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    def _save(self) -> None:
        if not self._training_rows:
            return
        file_exists = os.path.exists(ACCUM_CSV)
        os.makedirs(os.path.dirname(ACCUM_CSV), exist_ok=True)
        with open(ACCUM_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._training_rows)
        print(f"[XGBCollector] +{len(self._training_rows)} rows → {ACCUM_CSV}")
