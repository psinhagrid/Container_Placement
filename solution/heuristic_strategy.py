"""Expert heuristic placement strategy — v3.

Only the TOP container is checked (not all tiers).
Bug in v1/v2: scanning all tiers caused almost every non-empty stack to be
rejected (initial-state containers had mixed ETDs), so all containers piled
into the first empty slot in B01 — worse than greedy.

Two hard rules (skip stack entirely if violated):
  1. top_etd < inc_etd  → top departs before us → we would block its retrieval
  2. top_weight > inc_weight (ship only) → violates HEAVY-first loading order

Among valid stacks, soft score:
  - Same vessel on top   +200   same vessel = same ETD = perfect neighbour
  - Same port on top      +80   LOAD is port-by-port
  - Correct weight order  +50   incoming weight >= top weight
  - ETD proximity        variable  tighter gap = better ETD cluster
  - Height penalty        -15/tier  keep stacks short
  - Block occupancy       -20*ratio  spread containers across all blocks
"""

from datetime import datetime
from typing import Dict, Optional

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}


class HeuristicStrategy(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}

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
        apply_weight = event.vessel_id not in TRUCK_VESSELS
        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)

        # Cache block occupancy once (O(blocks), not O(total stacks))
        block_occ: Dict[str, float] = {}
        for bn, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(bn)
            block_occ[bn] = occ / cap if cap > 0 else 0.0

        best_pos: Optional[Position] = None
        best_score = float("-inf")

        for block_name, bi in yard_state.blocks.items():
            occ_ratio = block_occ[block_name]

            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h >= bi.tiers:
                        continue

                    # Spread containers across blocks (tiebreaker for empty stacks)
                    score = -occ_ratio * 20

                    if h > 0:
                        top_cid = yard_state.get_container_at(block_name, bay, row, h)
                        top_info = yard_state.get_container_info(top_cid) if top_cid else None

                        if top_info:
                            top_etd = self._etd(top_info.departure_time)
                            top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)

                            # Hard rule 1: top departs before us → skip
                            if top_etd < inc_etd - 3_600:
                                continue

                            # Hard rule 2: lighter on heavier → skip (ship only)
                            if apply_weight and inc_rank < top_rank:
                                continue

                            # ETD proximity reward: tighter cluster = safer
                            etd_gap_days = (top_etd - inc_etd) / 86_400
                            score += max(-100.0, -etd_gap_days * 10)

                            # Grouping bonuses
                            if top_info.vessel_id == event.vessel_id:
                                score += 200
                            if top_info.port_of_discharge == event.port_of_discharge:
                                score += 80
                            if apply_weight and inc_rank >= top_rank:
                                score += 50

                        score -= h * 15  # height penalty

                    if score > best_score:
                        best_score = score
                        best_pos = Position(block_name, bay, row, h + 1)

        if best_pos is not None:
            return best_pos

        # Fallback: any open slot
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        return Position(block_name, bay, row, h + 1)

        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        pass  # Phase 2: collect (features, reshuffles) for XGBoost
