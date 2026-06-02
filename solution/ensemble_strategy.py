"""Ensemble Strategy — averages predictions from two model checkpoints.

Uses the two best model snapshots saved by the parallel collector.
Models trained on different data snapshots learned complementary patterns.
Ensemble typically beats either model alone on borderline decisions.

Requires:
  solution/xgb_model.pkl          — current best model
  solution/xgb_model_ensemble.pkl — previous best model (saved by parallel_collector)

If xgb_model_ensemble.pkl doesn't exist, falls back to single model.
"""

import os
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd

from src.models import Event, Position
from src.yard_state import YardState
from solution.features import FEATURES
from solution.xgb_strategy import XGBStrategy, WEIGHT_RANK, TRUCK_VESSELS

PRIMARY_MODEL   = "solution/xgb_model.pkl"
SECONDARY_MODEL = "solution/xgb_model_ensemble.pkl"


class EnsembleStrategy(XGBStrategy):
    """Average predictions from two XGBoost checkpoints.

    Inherits ALL logic from XGBStrategy (features, tracking, fallback).
    Only overrides the scoring step to average two model predictions.
    """

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        super().initialize(yard_layout, initial_state)

        self._secondary_model = None
        if os.path.exists(SECONDARY_MODEL):
            self._secondary_model = joblib.load(SECONDARY_MODEL)
            print(f"[Ensemble] Primary + secondary model loaded — averaging predictions")
        else:
            print(f"[Ensemble] Secondary model not found — using primary only")

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        # If no secondary model, fall back to parent behavior
        if self._secondary_model is None or self._model is None:
            return super().place_container(yard_state, event)

        inc_etd      = self._etd(event.departure_time)
        inc_rank     = WEIGHT_RANK.get(event.weight_class, 2)
        placement_ts = self._etd(event.timestamp)
        is_truck     = int(event.vessel_id in TRUCK_VESSELS)
        days_until   = (
            max(0.0, (inc_etd - placement_ts) / 86_400)
            if inc_etd != float("inf") and placement_ts != float("inf")
            else 0.0
        )

        # Pre-compute block occupancies once
        block_occ_map: Dict[str, float] = {}
        for bn, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(bn)
            block_occ_map[bn] = occ / cap if cap > 0 else 0.0

        # Collect all candidates
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

        # Only score candidates at exact min height
        filtered = [(h, bn, bay, row) for h, bn, bay, row in candidates if h == min_h]

        min_h_count = len(filtered)
        total_open  = len(candidates)
        mh_pct      = round(min_h_count / max(total_open, 1), 4)

        rows, positions = [], []
        for h, bn, bay, row in filtered:
            uc   = self._unsafe_count(bn, bay, row, inc_etd)
            feat = self._stack_features(
                yard_state, bn, bay, row, h,
                inc_etd, inc_rank,
                block_occ_map[bn],
                days_until, is_truck, event,
                unsafe_cnt=uc, min_height_pct=mh_pct,
            )
            rows.append(feat)
            positions.append(Position(bn, bay, row, h + 1))

        X = pd.DataFrame(rows, columns=FEATURES)

        # Average predictions from both models
        pred_primary   = self._model.predict(X)
        pred_secondary = self._secondary_model.predict(X)
        preds = 0.5 * pred_primary + 0.5 * pred_secondary

        best_idx = int(np.argmin(preds))
        chosen   = positions[best_idx]

        # Update stack tracking
        cid    = event.container_id
        inc_ir = self._intra_rank(event.vessel_id, event.port_of_discharge, event.weight_class)
        key    = (chosen.block, chosen.bay, chosen.row)
        self._container_etd[cid]        = inc_etd
        self._container_intra_rank[cid] = inc_ir
        self._stack_containers.setdefault(key, set()).add(cid)

        return chosen
