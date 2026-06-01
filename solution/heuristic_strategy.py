"""Greedy + domain tiebreakers — v6.

Lesson from v1-v5: every time we overrode greedy's height-first logic with
domain rules (ETD filtering, block assignment), the score got worse.
Greedy (0.79) beats all our attempts because height balance is the primary
driver of reshuffles.

This version keeps greedy's core (always pick the globally shortest stack)
and adds domain knowledge ONLY as tiebreakers for equal-height stacks:

  Primary  : shortest stack height  (greedy — never compromised)
  Tie 1    : top ETD >= incoming ETD  (no ETD violation on top)
  Tie 2    : same vessel on top
  Tie 3    : same port on top
  Tie 4    : weight ordering correct (incoming weight >= top weight, ship only)

No hard skip rules. No block restrictions. Domain knowledge nudges decisions
when height is equal — never forces a taller stack.
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

                    etd_ok = True       # empty stack: no ETD issue
                    vessel_match = False
                    port_match = False
                    weight_ok = True    # empty stack: no weight issue

                    if h > 0:
                        top_cid = yard_state.get_container_at(block_name, bay, row, h)
                        top_info = yard_state.get_container_info(top_cid) if top_cid else None

                        if top_info:
                            top_etd = self._etd(top_info.departure_time)
                            etd_ok = top_etd >= inc_etd - 3_600  # top departs after us
                            vessel_match = top_info.vessel_id == event.vessel_id
                            port_match = top_info.port_of_discharge == event.port_of_discharge
                            if apply_weight:
                                top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                                weight_ok = inc_rank >= top_rank

                    # ── Comparison ──────────────────────────────────────────
                    # Primary: shorter height always wins.
                    # Equal height: tiebreakers in priority order.
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

        if best_pos is not None:
            return best_pos

        # Yard full fallback
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
