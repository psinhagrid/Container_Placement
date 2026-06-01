"""Greedy strategy with XGBoost training data collection.

Placement: identical to baseline_greedy (always pick shortest stack).
Collection: records features at placement time, labels at retrieval time.

Why use greedy instead of v6 for data collection:
  Greedy makes random ETD/vessel/weight decisions → more violations in features
  → XGBoost sees stronger signal: "when same_vessel=False AND etd_gap<0 → high reshuffles"
  v6 already avoids those mistakes → weak signal → XGBoost only learns height matters
"""

import atexit
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}
TRAINING_DATA_PATH = "data/train/placement_features.csv"

FEATURE_COLS = [
    "stack_height", "top_etd_gap_days", "same_vessel", "same_port",
    "weight_ok", "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "is_truck", "reshuffles",
]


class GreedyCollector(PlacementStrategy):
    """Greedy placement + training data collection for XGBoost."""

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}
        self._placement_features: Dict[str, dict] = {}
        self._training_rows: List[dict] = []
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

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd = self._etd(event.departure_time)
        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)
        placement_ts = self._etd(event.timestamp)

        best_pos: Optional[Position] = None
        best_height = float("inf")

        # Feature vars for the chosen position
        chosen_h = 0
        chosen_top_etd: float = float("inf")
        chosen_top_rank = 0
        chosen_vessel = False
        chosen_port = False
        chosen_weight = True
        chosen_block: Optional[str] = None

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers and h < best_height:
                        best_height = h
                        best_pos = Position(block_name, bay, row, h + 1)
                        chosen_h = h
                        chosen_block = block_name
                        # Reset top-container info
                        chosen_top_etd = float("inf")
                        chosen_top_rank = 0
                        chosen_vessel = False
                        chosen_port = False
                        chosen_weight = True

                        if h > 0:
                            top_cid = yard_state.get_container_at(block_name, bay, row, h)
                            top_info = yard_state.get_container_info(top_cid) if top_cid else None
                            if top_info:
                                chosen_top_etd = self._etd(top_info.departure_time)
                                chosen_top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                                chosen_vessel = top_info.vessel_id == event.vessel_id
                                chosen_port = top_info.port_of_discharge == event.port_of_discharge
                                if event.vessel_id not in TRUCK_VESSELS:
                                    chosen_weight = inc_rank >= chosen_top_rank

                        if best_height == 0:
                            break  # greedy short-circuit
                if best_height == 0:
                    break
            if best_height == 0:
                break

        # Collect features for this placement
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
            }
            return best_pos

        if best_pos is None:
            first_block = list(yard_state.blocks.keys())[0]
            return Position(first_block, 1, 1, 999)

        return best_pos

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        feats = self._placement_features.pop(container_id, None)
        if feats is None:
            return
        row = dict(feats)
        row["reshuffles"] = reshuffles
        self._training_rows.append(row)

    def _save_training_data(self) -> None:
        if not self._training_rows:
            return
        os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
        with open(TRAINING_DATA_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
            writer.writeheader()
            writer.writerows(self._training_rows)
        print(f"\n[XGBoost data] {len(self._training_rows)} rows → {TRAINING_DATA_PATH}")
