"""Precomputed Min-ETD Strategy — uses initial state knowledge properly.

Key insight finally used:
  initialize() gives us ALL 4,800 initial containers with their positions and ETDs
  BEFORE event #1 fires. We precompute the minimum ETD of every stack.

  stack_min_etd[(block, bay, row)] = earliest departure of any container in that stack

  For incoming container X (ETD = T):
    SAFE stack   → stack_min_etd >= T  (all containers depart AFTER X → gone when X retrieved)
    UNSAFE stack → stack_min_etd <  T  (at least one container departs BEFORE X → blocks it)

  We update this map live:
    - When WE place a container: update min_etd for that stack
    - When on_container_retrieved() fires: remove container, recompute min_etd
      (a container left → stack may now be cleaner → min_etd goes up)

Why this works on TEST data:
  Test initial state (day 20) contains only containers waiting for Jan21–Feb9 vessels.
  stack_min_etd ≥ Jan 21 for almost every stack.
  New containers also arrive for Jan21–Feb9 vessels.
  → Almost every stack is SAFE → ETD ordering maintained → 0 reshuffles.

Placement priority:
  Pass 1: SAFE non-empty stacks (stack_min_etd >= inc_etd) + weight OK
           → shortest height, vessel/port as tiebreaker
  Pass 2: Empty stacks (always safe)
           → prefer less-occupied blocks
  Pass 3: Least-unsafe fallback
           → argmin(unsafe_count + height * 0.1)
           unsafe_count = containers in stack with ETD < ours
  Pass 4: Pure greedy last resort
"""

from datetime import datetime
from typing import Dict, Optional, Set

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK   = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}
ONE_HOUR      = 3_600.0


class MinETDStrategy(PlacementStrategy):

    # ── Initialize: precompute from initial state ──────────────────────────────

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}

        # container_id → ETD timestamp (all containers we know about)
        self._container_etd: Dict[str, float] = {}

        # (block, bay, row) → set of container_ids currently in that stack
        self._stack_containers: Dict[tuple, Set[str]] = {}

        # (block, bay, row) → minimum ETD of any container in the stack
        self._stack_min_etd: Dict[tuple, float] = {}

        # Parse all containers from initial state
        for c in initial_state.get("containers", []):
            cid = c["container_id"]
            pos = c["position"]
            key = (pos["block"], pos["bay"], pos["row"])
            etd = self._etd(c.get("departure_time", ""))

            self._container_etd[cid] = etd

            if key not in self._stack_containers:
                self._stack_containers[key] = set()
            self._stack_containers[key].add(cid)

            # Track minimum ETD per stack
            if key not in self._stack_min_etd:
                self._stack_min_etd[key] = etd
            else:
                self._stack_min_etd[key] = min(self._stack_min_etd[key], etd)

        print(
            f"[MinETD] Precomputed {len(self._stack_min_etd)} stacks "
            f"from {len(self._container_etd)} initial containers"
        )

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

    def _stack_unsafe_count(self, key: tuple, inc_etd: float) -> int:
        """Count containers in this stack with ETD < inc_etd (will cause reshuffles)."""
        return sum(
            1 for cid in self._stack_containers.get(key, set())
            if self._container_etd.get(cid, float("inf")) < inc_etd - ONE_HOUR
        )

    # ── Pass 1: safe non-empty stacks ─────────────────────────────────────────

    def _pass1(
        self,
        yard_state: YardState,
        event: Event,
        inc_etd: float,
        inc_rank: int,
        apply_weight: bool,
    ) -> Optional[Position]:
        """Shortest non-empty stack where ALL containers have ETD >= ours.
        We add 0 reshuffles to any existing container in these stacks.
        """
        best_pos:   Optional[Position] = None
        best_h      = float("inf")
        best_vessel = False
        best_port   = False

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h == 0 or h >= bi.tiers:
                        continue

                    key = (block_name, bay, row)

                    # Fast O(1) safety check using precomputed min_etd
                    min_etd = self._stack_min_etd.get(key, float("inf"))
                    if min_etd < inc_etd - ONE_HOUR:
                        continue  # at least one container would cause a reshuffle

                    # Weight check on top container only (single yard_state call)
                    top_cid  = yard_state.get_container_at(block_name, bay, row, h)
                    top_info = yard_state.get_container_info(top_cid) if top_cid else None

                    if top_info and apply_weight:
                        top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                        if inc_rank < top_rank:
                            continue

                    vessel_match = bool(top_info and top_info.vessel_id == event.vessel_id)
                    port_match   = bool(top_info and top_info.port_of_discharge == event.port_of_discharge)

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

    # ── Pass 2: empty stacks ──────────────────────────────────────────────────

    def _pass2(self, yard_state: YardState) -> Optional[Position]:
        """Any empty stack — always safe. Prefer less-occupied blocks."""
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

    # ── Pass 3: least-unsafe fallback ─────────────────────────────────────────

    def _pass3(
        self, yard_state: YardState, inc_etd: float,
    ) -> Optional[Position]:
        """Forced to violate — pick the stack that causes LEAST damage.
        Score = unsafe_count + height * 0.1
        (fewer containers with early ETD below us + shorter stack = less damage)
        """
        best_pos:   Optional[Position] = None
        best_score  = float("inf")

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h == 0 or h >= bi.tiers:
                        continue

                    key = (block_name, bay, row)
                    min_etd = self._stack_min_etd.get(key, float("inf"))
                    if min_etd >= inc_etd - ONE_HOUR:
                        continue  # safe stack — already handled in pass 1

                    unsafe_count = self._stack_unsafe_count(key, inc_etd)
                    score = unsafe_count + h * 0.1

                    if score < best_score:
                        best_score = score
                        best_pos   = Position(block_name, bay, row, h + 1)

        return best_pos

    # ── Main placement ─────────────────────────────────────────────────────────

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd      = self._etd(event.departure_time)
        apply_weight = event.vessel_id not in TRUCK_VESSELS
        inc_rank     = WEIGHT_RANK.get(event.weight_class, 2)

        pos = (
            self._pass1(yard_state, event, inc_etd, inc_rank, apply_weight)
            or self._pass2(yard_state)
            or self._pass3(yard_state, inc_etd)
        )

        if pos is None:
            # Pass 4: pure greedy
            for bn, bi in yard_state.blocks.items():
                for bay in range(1, bi.bays + 1):
                    for row in range(1, bi.rows + 1):
                        h = yard_state.get_stack_height(bn, bay, row)
                        if h < bi.tiers:
                            pos = Position(bn, bay, row, h + 1)
                            break
                    if pos: break
                if pos: break

        if pos is None:
            pos = Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

        # ── Update our tracking for this placement ─────────────────────────────
        key = (pos.block, pos.bay, pos.row)
        cid = event.container_id

        self._container_etd[cid] = inc_etd

        if key not in self._stack_containers:
            self._stack_containers[key] = set()
        self._stack_containers[key].add(cid)

        # Update min_etd: can only decrease or stay same when adding
        if key not in self._stack_min_etd or inc_etd < self._stack_min_etd[key]:
            self._stack_min_etd[key] = inc_etd

        return pos

    # ── Retrieval callback: update tracking ────────────────────────────────────

    def on_container_retrieved(
        self, container_id: str, position: Position, reshuffles: int
    ) -> None:
        """Container removed from yard — recompute min_etd for its stack.
        As early-ETD containers leave, their stacks become cleaner (min_etd increases).
        """
        key = (position.block, position.bay, position.row)

        # Remove from stack tracking
        stack = self._stack_containers.get(key)
        if stack:
            stack.discard(container_id)
            # Recompute min_etd from remaining containers
            if not stack:
                self._stack_min_etd[key] = float("inf")  # empty now
            else:
                self._stack_min_etd[key] = min(
                    self._container_etd.get(cid, float("inf"))
                    for cid in stack
                )

        # Remove from container ETD lookup
        self._container_etd.pop(container_id, None)
