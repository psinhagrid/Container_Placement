"""Greedy + domain tiebreakers — v6, with XGBoost training data collection.

Placement logic (unchanged from v6):
  Primary  : shortest stack height  (greedy core — never compromised)
  Tie 1    : top ETD >= incoming ETD
  Tie 2    : same vessel
  Tie 3    : same port
  Tie 4    : weight ordering correct (ship only)

Data collection:
  At place_container()         → snapshot features of the chosen stack
  At on_container_retrieved()  → attach actual reshuffles to those features
  At process exit (atexit)     → write all rows to CSV for XGBoost training

Features recorded at placement time (labels added at retrieval time):
  stack_height       int    0-4   height before placing
  top_etd_gap_days   float         (top_etd - inc_etd) / 86400; negative = ETD violation
  same_vessel        0/1          top container is same vessel
  same_port          0/1          top container is same port
  weight_ok          0/1          incoming weight >= top weight (or empty stack)
  weight_rank_inc    1/2/3        LIGHT / MEDIUM / HEAVY
  weight_rank_top    0/1/2/3      0 if empty stack
  block_occ          float  0-1   fraction of block occupied at placement time
  days_until_dep     float        (inc_etd - event_timestamp) / 86400
  is_truck           0/1          VSL019 / VSL020 vessel

Target:
  reshuffles         int    0-4   containers moved to retrieve this one
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


class HeuristicStrategy(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}
        self._placement_features: Dict[str, dict] = {}  # container_id → features
        self._training_rows: List[dict] = []
        atexit.register(self._save_training_data)

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

    # ── Placement ──────────────────────────────────────────────────────────────

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd = self._etd(event.departure_time)
        apply_weight = event.vessel_id not in TRUCK_VESSELS
        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)

        best_pos: Optional[Position] = None
        best_h = float("inf")
        best_etd_ok = False
        best_vessel = False
        best_port = False
        best_weight = False

        # Extra tracking for feature collection
        best_top_etd: float = float("inf")
        best_top_rank: int = 0
        best_block: Optional[str] = None

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
                    top_etd: float = float("inf")
                    top_rank: int = 0

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

                    # Primary: shorter height. Equal height: tiebreakers.
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
                        best_top_etd = top_etd
                        best_top_rank = top_rank
                        best_block = block_name
                        best_pos = Position(block_name, bay, row, h + 1)

        # ── Collect features for this placement ────────────────────────────────
        if best_pos is not None and best_block is not None:
            placement_ts = self._etd(event.timestamp)
            occ, cap = yard_state.get_block_occupancy(best_block)
            block_occ = occ / cap if cap > 0 else 0.0

            top_etd_gap = (
                (best_top_etd - inc_etd) / 86_400
                if best_top_etd != float("inf") and inc_etd != float("inf")
                else 0.0
            )
            days_until_dep = (
                max(0.0, (inc_etd - placement_ts) / 86_400)
                if inc_etd != float("inf") and placement_ts != float("inf")
                else 0.0
            )

            self._placement_features[event.container_id] = {
                "stack_height":    best_h if best_h != float("inf") else 0,
                "top_etd_gap_days": round(top_etd_gap, 4),
                "same_vessel":     int(best_vessel),
                "same_port":       int(best_port),
                "weight_ok":       int(best_weight),
                "weight_rank_inc": inc_rank,
                "weight_rank_top": best_top_rank,
                "block_occ":       round(block_occ, 4),
                "days_until_dep":  round(days_until_dep, 4),
                "is_truck":        int(event.vessel_id in TRUCK_VESSELS),
            }
            return best_pos

        # Fallback — any open slot (rare; don't collect features for these)
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        return Position(block_name, bay, row, h + 1)

        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    # ── Retrieval callback ─────────────────────────────────────────────────────

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        """Attach actual reshuffle count to the placement features."""
        feats = self._placement_features.pop(container_id, None)
        if feats is None:
            return  # initial-state container — we didn't place it

        row = dict(feats)
        row["reshuffles"] = reshuffles
        self._training_rows.append(row)

    # ── Persist ────────────────────────────────────────────────────────────────

    def _save_training_data(self) -> None:
        if not self._training_rows:
            return
        os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
        with open(TRAINING_DATA_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
            writer.writeheader()
            writer.writerows(self._training_rows)
        print(f"\n[XGBoost data] {len(self._training_rows)} rows → {TRAINING_DATA_PATH}")
