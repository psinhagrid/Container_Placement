"""ETD-Smart Placement Strategy — Phase 5.

Core insight:
  For 0 reshuffles when container X is retrieved, X must be on TOP of its
  stack at retrieval time. This is guaranteed if every container placed ABOVE
  X has ETD ≤ X.etd — they depart before X, so they are retrieved before X,
  meaning they are already gone when X is retrieved.

  Perfect ETD ordering (decreasing from bottom to top) → 0 reshuffles for all.

Why previous ETD attempts failed:
  When no ETD-compatible stack was found, we fell to GREEDY fallback (any
  shortest stack). Greedy ignores violation size:
    - A 10-day ETD violation → guaranteed reshuffle
    - A 1-hour ETD violation → nearly harmless (both containers retrieved ~same time)
  Picking a stack with a 1-day violation is FAR better than picking a random
  short stack with a 10-day violation.

The fix — Minimum-Violation Fallback:
  Pass 1: ETD-compatible stack (top_etd ≥ inc_etd) + weight-compatible
          → shortest height, vessel/port tiebreaker
  Pass 2: Empty stack (always ETD-safe, no constraint)
          → prefer less-occupied blocks
  Pass 3: Minimum-violation fallback
          → score = violation_days + height; pick argmin
          (small violation + short stack = least future damage)
  Pass 4: Pure greedy (absolute last resort — rare)

Height is NOT penalized in Pass 1: tall stacks with perfect ETD ordering
beat short stacks with random ordering. A height-5 perfectly-ordered stack
generates 0 reshuffles. A height-2 randomly-ordered stack generates 1+.
"""

from datetime import datetime
from typing import Dict, Optional

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK   = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}
ONE_HOUR      = 3_600.0
ONE_DAY       = 86_400.0


class ETDSmartStrategy(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}

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

    # ── Pass 1: ETD-compatible + weight-compatible ─────────────────────────────

    def _pass1(
        self,
        yard_state: YardState,
        event: Event,
        inc_etd: float,
        inc_rank: int,
        apply_weight: bool,
    ) -> Optional[Position]:
        """Shortest non-empty stack where top departs AFTER us and weight is OK.
        Vessel/port match is a tiebreaker for equal height — never sacrifices ETD.
        Height is NOT penalised here: tall ETD-ordered stacks are fine.
        """
        best_pos:    Optional[Position] = None
        best_h       = float("inf")
        best_vessel  = False
        best_port    = False

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h == 0 or h >= bi.tiers:
                        continue

                    top_cid  = yard_state.get_container_at(block_name, bay, row, h)
                    top_info = (yard_state.get_container_info(top_cid)
                                if top_cid else None)
                    if not top_info:
                        continue

                    top_etd  = self._etd(top_info.departure_time)
                    top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)

                    # Hard ETD rule: top must depart after (or same time as) us.
                    # With 1-hour grace for same-vessel rounding.
                    if top_etd < inc_etd - ONE_HOUR:
                        continue

                    # Hard weight rule (ship vessels only): HEAVY must be on top.
                    if apply_weight and inc_rank < top_rank:
                        continue

                    vessel_match = top_info.vessel_id == event.vessel_id
                    port_match   = top_info.port_of_discharge == event.port_of_discharge

                    better = False
                    if h < best_h:
                        better = True
                    elif h == best_h:
                        if vessel_match > best_vessel:
                            better = True
                        elif vessel_match == best_vessel and port_match > best_port:
                            better = True

                    if better:
                        best_h      = h
                        best_vessel = vessel_match
                        best_port   = port_match
                        best_pos    = Position(block_name, bay, row, h + 1)

        return best_pos

    # ── Pass 2: empty stack ────────────────────────────────────────────────────

    def _pass2(self, yard_state: YardState) -> Optional[Position]:
        """Any empty (height-0) stack — always ETD-safe.
        Prefers blocks with lower occupancy to spread containers evenly.
        """
        best_pos: Optional[Position] = None
        best_occ  = float("inf")

        for block_name, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(block_name)
            occ_ratio = occ / cap if cap > 0 else 0.0

            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    if yard_state.get_stack_height(block_name, bay, row) == 0:
                        if occ_ratio < best_occ:
                            best_occ = occ_ratio
                            best_pos = Position(block_name, bay, row, 1)

        return best_pos

    # ── Pass 3: minimum-violation fallback ─────────────────────────────────────

    def _pass3(
        self,
        yard_state: YardState,
        inc_etd: float,
    ) -> Optional[Position]:
        """Forced ETD violation — choose the stack that does LEAST damage.

        Score = violation_days + height
          violation_days: how many days our ETD exceeds the top container's ETD
          height:         current stack height (more containers = more future risk)

        A 1-day violation on a height-1 stack (score ≈ 2) is far better than
        a 10-day violation on a height-3 stack (score ≈ 13).
        """
        best_pos:   Optional[Position] = None
        best_score  = float("inf")

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h == 0 or h >= bi.tiers:
                        continue

                    top_cid  = yard_state.get_container_at(block_name, bay, row, h)
                    top_info = (yard_state.get_container_info(top_cid)
                                if top_cid else None)
                    if not top_info:
                        continue

                    top_etd = self._etd(top_info.departure_time)

                    # Only look at ETD-incompatible stacks (pass 1 already tried compatible)
                    if top_etd >= inc_etd - ONE_HOUR:
                        continue

                    violation_days = (inc_etd - top_etd) / ONE_DAY
                    score = violation_days + h    # minimise: small gap + short stack

                    if score < best_score:
                        best_score = score
                        best_pos   = Position(block_name, bay, row, h + 1)

        return best_pos

    # ── Main placement ─────────────────────────────────────────────────────────

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd      = self._etd(event.departure_time)
        apply_weight = event.vessel_id not in TRUCK_VESSELS
        inc_rank     = WEIGHT_RANK.get(event.weight_class, 2)

        # Pass 1: ETD + weight compatible — ideal placement
        pos = self._pass1(yard_state, event, inc_etd, inc_rank, apply_weight)
        if pos:
            return pos

        # Pass 2: empty stack — safe start for a new ETD-ordered chain
        pos = self._pass2(yard_state)
        if pos:
            return pos

        # Pass 3: forced violation — minimise the damage
        pos = self._pass3(yard_state, inc_etd)
        if pos:
            return pos

        # Pass 4: pure greedy — absolute last resort (yard nearly full)
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        return Position(block_name, bay, row, h + 1)

        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    def on_container_retrieved(
        self, container_id: str, position: Position, reshuffles: int
    ) -> None:
        pass
