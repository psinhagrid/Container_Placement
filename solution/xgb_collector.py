"""XGBoost strategy with exploration + training data collection.

Identical feature set to greedy_collector.py (19 features + reshuffles).
Uses XGBoost scoring with 10% random exploration to prevent distribution collapse.
Appends new rows to accumulated CSV for self-improvement loop.
"""

import csv
import json
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
WEIGHT_OFFSET = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}
TRUCK_VESSELS = {"VSL019", "VSL020"}
MODEL_PATH    = "solution/xgb_model.pkl"
SCHEDULE_PATH = "data/vessel_schedule.json"
ACCUM_CSV     = "data/train/placement_features_accum.csv"
ONE_HOUR      = 3_600.0
EXPLORE_RATE  = 0.10

FEATURE_COLS = [
    "stack_height", "top_etd_gap_days", "same_vessel", "same_port",
    "weight_ok", "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "is_truck", "unsafe_count",
    "port_order", "weight_offset", "intra_vessel_rank",
    "unsafe_rank_count", "free_slots", "rank_gap_to_top",
    "unsafe_x_height", "min_height_pct",
    "reshuffles",
]

FEATURES = FEATURE_COLS[:-1]  # all except reshuffles


class XGBCollector(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}
        self._placement_features: Dict[str, dict] = {}
        self._training_rows: List[dict] = []

        # Vessel schedule for port order
        self._vessel_ports: Dict[str, List[str]] = {}
        if os.path.exists(SCHEDULE_PATH):
            with open(SCHEDULE_PATH) as f:
                sched = json.load(f)
            for v in sched.get("vessels", []):
                self._vessel_ports[v["vessel_id"]] = v.get("ports", [])

        # Stack tracking
        self._container_etd:         Dict[str, float] = {}
        self._container_intra_rank:  Dict[str, int]   = {}
        self._stack_containers:      Dict[tuple, Set[str]] = {}

        for c in initial_state.get("containers", []):
            cid = c["container_id"]
            p   = c["position"]
            key = (p["block"], p["bay"], p["row"])
            etd = self._etd(c.get("departure_time", ""))
            ir  = self._intra_rank(c.get("vessel_id", ""),
                                   c.get("port_of_discharge", ""),
                                   c.get("weight_class", "MEDIUM"))
            self._container_etd[cid]        = etd
            self._container_intra_rank[cid] = ir
            self._stack_containers.setdefault(key, set()).add(cid)

        self._model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        if self._model:
            print(f"[XGBCollector] Model loaded, explore_rate={EXPLORE_RATE:.0%}")
        else:
            print("[XGBCollector] No model — pure exploration")

    def _etd(self, s: str) -> float:
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except Exception:
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    def _intra_rank(self, vessel_id: str, port: str, weight: str) -> int:
        if vessel_id in TRUCK_VESSELS:
            return 0
        ports    = self._vessel_ports.get(vessel_id, [])
        port_idx = ports.index(port) if port in ports else len(ports)
        return port_idx * 3 + WEIGHT_OFFSET.get(weight, 1)

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd      = self._etd(event.departure_time)
        inc_rank     = WEIGHT_RANK.get(event.weight_class, 2)
        placement_ts = self._etd(event.timestamp)
        is_truck     = int(event.vessel_id in TRUCK_VESSELS)
        inc_port_ord = 0 if is_truck else (
            self._vessel_ports.get(event.vessel_id, []).index(event.port_of_discharge)
            if event.port_of_discharge in self._vessel_ports.get(event.vessel_id, [])
            else len(self._vessel_ports.get(event.vessel_id, []))
        )
        inc_w_off = WEIGHT_OFFSET.get(event.weight_class, 1)
        inc_ir    = inc_port_ord * 3 + inc_w_off
        days_until = max(0.0, (inc_etd - placement_ts) / 86_400) if inc_etd != float("inf") else 0.0

        # Cache block occupancies
        block_occ: Dict[str, float] = {}
        for bn, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(bn)
            block_occ[bn] = occ / cap if cap > 0 else 0.0

        # Find min height + yard context
        min_h = float("inf")
        min_h_count = 0
        total_open  = 0
        for bn, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(bn, bay, row)
                    if h < bi.tiers:
                        total_open += 1
                        if h < min_h:
                            min_h = h
        if min_h < float("inf"):
            for bn, bi in yard_state.blocks.items():
                for bay in range(1, bi.bays + 1):
                    for row in range(1, bi.rows + 1):
                        if yard_state.get_stack_height(bn, bay, row) == min_h:
                            min_h_count += 1

        if min_h == float("inf"):
            return self._fallback(yard_state)

        min_height_pct = round(min_h_count / max(total_open, 1), 4)

        # Collect all min-height candidates
        candidates = []
        for bn, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(bn, bay, row)
                    if h == min_h and h < bi.tiers:
                        candidates.append((bn, bay, row))

        if not candidates:
            return self._fallback(yard_state)

        # Build feature rows
        rows, positions = [], []
        for bn, bay, row in candidates:
            h   = min_h
            key = (bn, bay, row)

            top_etd_gap = 0.0; same_vessel = 0; same_port = 0
            weight_ok   = 1;   top_rank = 0;    top_ir = 0

            if h > 0:
                top_cid  = yard_state.get_container_at(bn, bay, row, h)
                top_info = yard_state.get_container_info(top_cid) if top_cid else None
                if top_info:
                    te = self._etd(top_info.departure_time)
                    top_rank    = WEIGHT_RANK.get(top_info.weight_class, 2)
                    top_ir      = self._container_intra_rank.get(top_cid, 0)
                    top_etd_gap = round((te - inc_etd) / 86_400, 4) if te != float("inf") else 0.0
                    same_vessel = int(top_info.vessel_id == event.vessel_id)
                    same_port   = int(top_info.port_of_discharge == event.port_of_discharge)
                    if not is_truck:
                        weight_ok = int(inc_rank >= top_rank)

            unsafe_count      = sum(1 for cid in self._stack_containers.get(key, set())
                                    if self._container_etd.get(cid, float("inf")) < inc_etd - ONE_HOUR)
            unsafe_rank_count = sum(1 for cid in self._stack_containers.get(key, set())
                                    if self._container_intra_rank.get(cid, 0) < inc_ir
                                    and self._container_etd.get(cid, float("inf")) < inc_etd + ONE_HOUR)

            rows.append({
                "stack_height":       h,
                "top_etd_gap_days":   top_etd_gap,
                "same_vessel":        same_vessel,
                "same_port":          same_port,
                "weight_ok":          weight_ok,
                "weight_rank_inc":    inc_rank,
                "weight_rank_top":    top_rank,
                "block_occ":          round(block_occ[bn], 4),
                "days_until_dep":     round(days_until, 4),
                "is_truck":           is_truck,
                "unsafe_count":       unsafe_count,
                "port_order":         inc_port_ord,
                "weight_offset":      inc_w_off,
                "intra_vessel_rank":  inc_ir,
                "unsafe_rank_count":  unsafe_rank_count,
                "free_slots":         5 - h,
                "rank_gap_to_top":    float(inc_ir - top_ir),
                "unsafe_x_height":    unsafe_count * h,
                "min_height_pct":     min_height_pct,
            })
            positions.append(Position(bn, bay, row, h + 1))

        # XGBoost score or explore
        if self._model is not None and random.random() > EXPLORE_RATE:
            X    = pd.DataFrame(rows, columns=FEATURES)
            pred = self._model.predict(X)
            idx  = int(np.argmin(pred))
        else:
            idx = random.randrange(len(candidates))

        chosen_pos  = positions[idx]
        chosen_feat = rows[idx]
        self._placement_features[event.container_id] = chosen_feat

        # Update tracking
        key = (chosen_pos.block, chosen_pos.bay, chosen_pos.row)
        self._container_etd[event.container_id]        = inc_etd
        self._container_intra_rank[event.container_id] = inc_ir
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
        self._container_intra_rank.pop(container_id, None)

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
