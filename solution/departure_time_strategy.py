"""Departure-Time Aware Strategy — Phase 3, v3.

Lessons from all previous attempts:
  - ETD as a hard rule always forces taller stacks → worse than greedy
  - Block assignment restricts options → worse than greedy
  - Weight ordering as a HARD rule + vessel grouping is the clean path forward

Logic:
  Pass 1: Weight ordering respected (inc_rank >= top_rank) — HARD RULE
           → shortest valid stack, same vessel preferred, same port second
  Pass 2: Weight relaxed — greedy fallback (any open slot, shortest first)

Why weight ordering matters:
  LOAD retrieves HEAVY first, then MEDIUM, then LIGHT.
  If HEAVY is on top, retrieval goes top→bottom in correct order → 0 reshuffles.
  Within same-vessel stacks: HEAVY on top = 0 reshuffles for LOAD.

No ETD filtering (always hurt). No block assignment (always hurt).
Height is always primary — we never pick a taller stack for any reason.
"""

from datetime import datetime
from typing import Dict, Optional

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}


class DepartureTimeStrategy(PlacementStrategy):

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

    def _scan(
        self,
        yard_state: YardState,
        event: Event,
        inc_rank: int,
        apply_weight: bool,
        require_weight: bool,
    ) -> Optional[Position]:
        """Shortest valid stack; vessel/port as tiebreakers."""
        best_pos: Optional[Position] = None
        best_h = float("inf")
        best_vessel = False
        best_port = False

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h >= bi.tiers:
                        continue

                    vessel_match = False
                    port_match = False

                    if h > 0:
                        top_cid = yard_state.get_container_at(block_name, bay, row, h)
                        top_info = yard_state.get_container_info(top_cid) if top_cid else None
                        if top_info:
                            if require_weight and apply_weight:
                                top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                                if inc_rank < top_rank:
                                    continue  # lighter on heavier → LOAD order broken

                            vessel_match = top_info.vessel_id == event.vessel_id
                            port_match = top_info.port_of_discharge == event.port_of_discharge

                    better = False
                    if h < best_h:
                        better = True
                    elif h == best_h:
                        if vessel_match > best_vessel:
                            better = True
                        elif vessel_match == best_vessel and port_match > best_port:
                            better = True

                    if better:
                        best_h = h
                        best_vessel = vessel_match
                        best_port = port_match
                        best_pos = Position(block_name, bay, row, h + 1)

        return best_pos

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        apply_weight = event.vessel_id not in TRUCK_VESSELS
        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)

        # Pass 1: weight ordering enforced — HEAVY must be on top
        pos = self._scan(yard_state, event, inc_rank, apply_weight, require_weight=True)
        if pos:
            return pos

        # Pass 2: fallback — any open slot (pure greedy)
        pos = self._scan(yard_state, event, inc_rank, apply_weight, require_weight=False)
        if pos:
            return pos

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
