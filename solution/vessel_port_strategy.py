"""Vessel + Port Grouping Strategy — the correct departure-time-aware approach.

Why previous approaches failed:
  - Scoring individual stacks (XGBoost, heuristics) → can't fix grouping
  - Block assignment + hard rules → too restrictive, worse than greedy
  - Pre-reading events → not valid (uses future information)

What actually causes low reshuffles:
  LOAD events (75% of retrievals) retrieve vessel containers in order:
    port by port → within each port: HEAVY first, MEDIUM, LIGHT last.

  If all containers for (vessel_id, port_of_discharge) are in dedicated
  stacks with HEAVY on top, the retrieval goes top→bottom naturally → 0 reshuffles.

Core rules:
  1. NEVER mix containers from different (vessel, port) groups in the same stack.
  2. Within a group stack: enforce weight ordering (HEAVY on top, hard rule).
  3. Among valid stacks: prefer existing group stacks, then empty stacks.
  4. Truck vessels (VSL019/020): use greedy (TRUCK_DLVR has no weight ordering).
  5. Fallback: greedy if no valid stack found (prevents deadlock).
"""

from datetime import datetime
from typing import Dict, Optional

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}


class VesselPortStrategy(PlacementStrategy):

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
        # Truck vessels: no LOAD ordering → just use greedy
        if event.vessel_id in TRUCK_VESSELS:
            return self._greedy(yard_state)

        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)
        group_vessel = event.vessel_id
        group_port = event.port_of_discharge

        best_pos: Optional[Position] = None
        best_h = float("inf")
        best_same_group = False

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h >= bi.tiers:
                        continue

                    same_group = False

                    if h > 0:
                        top_cid = yard_state.get_container_at(block_name, bay, row, h)
                        top_info = (
                            yard_state.get_container_info(top_cid) if top_cid else None
                        )
                        if top_info:
                            top_vessel = top_info.vessel_id
                            top_port = top_info.port_of_discharge
                            top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)

                            # Rule 1: never mix groups
                            if top_vessel != group_vessel or top_port != group_port:
                                continue  # different group — hard skip

                            # Rule 2: weight ordering within group
                            if inc_rank < top_rank:
                                continue  # lighter on heavier — hard skip

                            same_group = True

                    # Prefer: same group (keeps retrieval clean) then empty
                    # Height always primary
                    better = False
                    if h < best_h:
                        better = True
                    elif h == best_h:
                        if same_group and not best_same_group:
                            better = True

                    if better:
                        best_h = h
                        best_same_group = same_group
                        best_pos = Position(block_name, bay, row, h + 1)

        if best_pos is not None:
            return best_pos

        # Fallback: greedy (ignores group purity — last resort when yard is constrained)
        return self._greedy(yard_state)

    def _greedy(self, yard_state: YardState) -> Position:
        """Shortest stack across all blocks, no group constraints."""
        best_pos: Optional[Position] = None
        best_h = float("inf")

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers and h < best_h:
                        best_h = h
                        best_pos = Position(block_name, bay, row, h + 1)
                        if best_h == 0:
                            return best_pos

        return best_pos or Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    def on_container_retrieved(
        self, container_id: str, position: Position, reshuffles: int
    ) -> None:
        pass
