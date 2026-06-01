"""Greedy strategy with XGBoost training data collection.

Now includes unsafe_count feature: number of containers in the chosen stack
with ETD < incoming ETD. This is the DIRECT count of reshuffles we're adding
to existing containers by placing here. Far more informative than just top_etd_gap_days.
"""

import atexit
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Set

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK   = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}
TRAINING_DATA_PATH = "data/train/placement_features.csv"
ONE_HOUR = 3_600.0

FEATURE_COLS = [
    "stack_height", "top_etd_gap_days", "same_vessel", "same_port",
    "weight_ok", "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "is_truck",
    "unsafe_count",   # NEW: containers in stack with ETD < ours (direct reshuffle count)
    "reshuffles",
]


class GreedyCollector(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}
        self._placement_features: Dict[str, dict] = {}
        self._training_rows: List[dict] = []

        # Precompute stack tracking from initial state
        self._container_etd: Dict[str, float] = {}
        self._stack_containers: Dict[tuple, Set[str]] = {}

        for c in initial_state.get("containers", []):
            cid = c["container_id"]
            p   = c["position"]
            key = (p["block"], p["bay"], p["row"])
            etd = self._etd(c.get("departure_time", ""))

            self._container_etd[cid] = etd
            if key not in self._stack_containers:
                self._stack_containers[key] = set()
            self._stack_containers[key].add(cid)

        atexit.register(self._save_training_data)

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
        """Count containers in this stack with ETD < inc_etd (will cause reshuffles)."""
        key = (block, bay, row)
        return sum(
            1 for cid in self._stack_containers.get(key, set())
            if self._container_etd.get(cid, float("inf")) < inc_etd - ONE_HOUR
        )

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd      = self._etd(event.departure_time)
        inc_rank     = WEIGHT_RANK.get(event.weight_class, 2)
        placement_ts = self._etd(event.timestamp)

        best_pos:    Optional[Position] = None
        best_height  = float("inf")
        chosen_h     = 0
        chosen_top_etd: float = float("inf")
        chosen_top_rank = 0
        chosen_vessel   = False
        chosen_port     = False
        chosen_weight   = True
        chosen_block: Optional[str] = None

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers and h < best_height:
                        best_height   = h
                        best_pos      = Position(block_name, bay, row, h + 1)
                        chosen_h      = h
                        chosen_block  = block_name
                        chosen_top_etd  = float("inf")
                        chosen_top_rank = 0
                        chosen_vessel   = False
                        chosen_port     = False
                        chosen_weight   = True

                        if h > 0:
                            top_cid  = yard_state.get_container_at(block_name, bay, row, h)
                            top_info = yard_state.get_container_info(top_cid) if top_cid else None
                            if top_info:
                                chosen_top_etd  = self._etd(top_info.departure_time)
                                chosen_top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                                chosen_vessel   = top_info.vessel_id == event.vessel_id
                                chosen_port     = top_info.port_of_discharge == event.port_of_discharge
                                if event.vessel_id not in TRUCK_VESSELS:
                                    chosen_weight = inc_rank >= chosen_top_rank

                        if best_height == 0:
                            break
                if best_height == 0:
                    break
            if best_height == 0:
                break

        if best_pos is not None and chosen_block is not None:
            occ, cap = yard_state.get_block_occupancy(chosen_block)
            block_occ = occ / cap if cap > 0 else 0.0

            top_etd_gap = (
                (chosen_top_etd - inc_etd) / 86_400
                if chosen_top_etd != float("inf") and inc_etd != float("inf")
                else 0.0
            )
            days_until_dep = (
                max(0.0, (inc_etd - placement_ts) / 86_400)
                if inc_etd != float("inf") and placement_ts != float("inf")
                else 0.0
            )

            # NEW: count containers in chosen stack that have ETD < ours
            uc = self._unsafe_count(chosen_block, best_pos.bay, best_pos.row, inc_etd)

            self._placement_features[event.container_id] = {
                "stack_height":     chosen_h,
                "top_etd_gap_days": round(top_etd_gap, 4),
                "same_vessel":      int(chosen_vessel),
                "same_port":        int(chosen_port),
                "weight_ok":        int(chosen_weight),
                "weight_rank_inc":  inc_rank,
                "weight_rank_top":  chosen_top_rank,
                "block_occ":        round(block_occ, 4),
                "days_until_dep":   round(days_until_dep, 4),
                "is_truck":         int(event.vessel_id in TRUCK_VESSELS),
                "unsafe_count":     uc,
            }

            # Update stack tracking
            cid = event.container_id
            key = (chosen_block, best_pos.bay, best_pos.row)
            self._container_etd[cid] = inc_etd
            if key not in self._stack_containers:
                self._stack_containers[key] = set()
            self._stack_containers[key].add(cid)

            return best_pos

        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        feats = self._placement_features.pop(container_id, None)
        if feats is not None:
            row = dict(feats)
            row["reshuffles"] = reshuffles
            self._training_rows.append(row)

        # Update stack tracking
        key = (position.block, position.bay, position.row)
        if key in self._stack_containers:
            self._stack_containers[key].discard(container_id)
        self._container_etd.pop(container_id, None)

    def _save_training_data(self) -> None:
        if not self._training_rows:
            return
        os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
        with open(TRAINING_DATA_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
            writer.writeheader()
            writer.writerows(self._training_rows)
        print(f"\n[XGBoost data] {len(self._training_rows)} rows → {TRAINING_DATA_PATH}")
