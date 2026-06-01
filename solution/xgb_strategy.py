"""XGBoost placement strategy.

Workflow per placement decision:
  1. Scan all stacks → find the global minimum height among valid stacks
  2. Collect all candidates at min_height (and min_height+1 if < 10 candidates)
  3. Extract features for each candidate
  4. XGBoost batch-predicts reshuffles for all candidates
  5. Return position with lowest predicted reshuffles

Falls back to greedy+tiebreaker (v6) if model not loaded or no candidates found.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}
MODEL_PATH = "solution/xgb_model.pkl"

FEATURES = [
    "stack_height", "top_etd_gap_days", "same_vessel", "same_port",
    "weight_ok", "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "is_truck",
]


class XGBStrategy(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}

        # Load pre-trained model
        if os.path.exists(MODEL_PATH):
            self._model = joblib.load(MODEL_PATH)
            print(f"[XGBStrategy] Model loaded from {MODEL_PATH}")
        else:
            self._model = None
            print(f"[XGBStrategy] WARNING: model not found at {MODEL_PATH} — using v6 fallback")

    # ── ETD helper ─────────────────────────────────────────────────────────────

    def _etd(self, s: str) -> float:
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except Exception:
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    # ── Feature extraction for one candidate stack ─────────────────────────────

    def _stack_features(
        self,
        yard_state: YardState,
        block: str,
        bay: int,
        row: int,
        h: int,
        inc_etd: float,
        inc_rank: int,
        block_occ: float,
        days_until_dep: float,
        is_truck: int,
        event: Event,
    ) -> dict:
        top_etd_gap = 0.0
        same_vessel  = 0
        same_port    = 0
        weight_ok    = 1
        top_rank     = 0

        if h > 0:
            top_cid = yard_state.get_container_at(block, bay, row, h)
            top_info = yard_state.get_container_info(top_cid) if top_cid else None
            if top_info:
                top_etd = self._etd(top_info.departure_time)
                top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                top_etd_gap = (
                    (top_etd - inc_etd) / 86_400
                    if top_etd != float("inf") and inc_etd != float("inf")
                    else 0.0
                )
                same_vessel = int(top_info.vessel_id == event.vessel_id)
                same_port   = int(top_info.port_of_discharge == event.port_of_discharge)
                if event.vessel_id not in TRUCK_VESSELS:
                    weight_ok = int(inc_rank >= top_rank)

        return {
            "stack_height":     h,
            "top_etd_gap_days": round(top_etd_gap, 4),
            "same_vessel":      same_vessel,
            "same_port":        same_port,
            "weight_ok":        weight_ok,
            "weight_rank_inc":  inc_rank,
            "weight_rank_top":  top_rank,
            "block_occ":        round(block_occ, 4),
            "days_until_dep":   round(days_until_dep, 4),
            "is_truck":         is_truck,
        }

    # ── Main placement ─────────────────────────────────────────────────────────

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd = self._etd(event.departure_time)
        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)
        placement_ts = self._etd(event.timestamp)
        is_truck = int(event.vessel_id in TRUCK_VESSELS)
        days_until_dep = (
            max(0.0, (inc_etd - placement_ts) / 86_400)
            if inc_etd != float("inf") and placement_ts != float("inf")
            else 0.0
        )

        # Pre-compute block occupancies once
        block_occ_map: Dict[str, float] = {}
        for bn, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(bn)
            block_occ_map[bn] = occ / cap if cap > 0 else 0.0

        # ── Collect all candidates ─────────────────────────────────────────────
        # Candidate: (h, block, bay, row)
        candidates: List[Tuple[int, str, int, int]] = []
        min_h = float("inf")

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h >= bi.tiers:
                        continue
                    if h < min_h:
                        min_h = h
                    candidates.append((h, block_name, bay, row))

        if not candidates or min_h == float("inf"):
            return self._fallback(yard_state)

        # Only score candidates at exact min height — never sacrifice height balance
        # (allowing min_h+1 lets XGBoost pick taller stacks, which always hurts)
        filtered = [(h, bn, bay, row) for h, bn, bay, row in candidates if h == min_h]

        # ── XGBoost scoring ────────────────────────────────────────────────────
        if self._model is not None and filtered:
            rows = []
            positions = []
            for h, bn, bay, row in filtered:
                feat = self._stack_features(
                    yard_state, bn, bay, row, h,
                    inc_etd, inc_rank,
                    block_occ_map[bn],
                    days_until_dep, is_truck, event,
                )
                rows.append(feat)
                positions.append(Position(bn, bay, row, h + 1))

            X = pd.DataFrame(rows, columns=FEATURES)
            preds = self._model.predict(X)
            best_idx = int(np.argmin(preds))
            return positions[best_idx]

        # ── v6 fallback (no model) ─────────────────────────────────────────────
        return self._v6_fallback(yard_state, event, inc_etd, inc_rank)

    def _v6_fallback(
        self, yard_state: YardState, event: Event,
        inc_etd: float, inc_rank: int,
    ) -> Position:
        """Greedy + domain tiebreakers (v6 logic)."""
        apply_weight = event.vessel_id not in TRUCK_VESSELS
        best_pos: Optional[Position] = None
        best_h = float("inf")
        best_etd_ok = False
        best_vessel = False
        best_port = False
        best_weight = False

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h >= bi.tiers:
                        continue
                    etd_ok = True
                    vessel_match = False
                    port_match = False
                    weight_ok = True

                    if h > 0:
                        top_cid = yard_state.get_container_at(block_name, bay, row, h)
                        top_info = yard_state.get_container_info(top_cid) if top_cid else None
                        if top_info:
                            top_etd = self._etd(top_info.departure_time)
                            top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                            etd_ok = top_etd >= inc_etd - 3_600
                            vessel_match = top_info.vessel_id == event.vessel_id
                            port_match = top_info.port_of_discharge == event.port_of_discharge
                            if apply_weight:
                                weight_ok = inc_rank >= top_rank

                    better = False
                    if h < best_h:
                        better = True
                    elif h == best_h:
                        if etd_ok > best_etd_ok:
                            better = True
                        elif etd_ok == best_etd_ok:
                            if vessel_match > best_vessel:
                                better = True
                            elif vessel_match == best_vessel:
                                if port_match > best_port:
                                    better = True
                                elif port_match == best_port:
                                    better = weight_ok > best_weight

                    if better:
                        best_h = h
                        best_etd_ok = etd_ok
                        best_vessel = vessel_match
                        best_port = port_match
                        best_weight = weight_ok
                        best_pos = Position(block_name, bay, row, h + 1)

        return best_pos if best_pos is not None else self._fallback(yard_state)

    def _fallback(self, yard_state: YardState) -> Position:
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        return Position(block_name, bay, row, h + 1)
        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        pass
