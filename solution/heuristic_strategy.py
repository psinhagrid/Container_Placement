"""Expert heuristic placement strategy.

Priority order (strict → soft):
  1. ETD ordering   [strict]  — never let a late-ETD container block an early-ETD one
  2. Weight ordering [strict] — HEAVY must sit above LIGHT/MEDIUM (LOAD retrieves HEAVY first)
  3. Vessel grouping [soft]   — same vessel = same ETD = safe neighbours
  4. Port grouping   [soft]   — LOAD goes port-by-port; same port in one stack = no reshuffles
  5. Height balance  [soft]   — prefer shorter stacks to spread risk
"""

from datetime import datetime
from typing import Dict, Optional, Tuple

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

# HEAVY loaded first during LOAD → HEAVY retrieved from yard first → must sit on TOP
WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}

# Truck-only vessels: retrieved by individual truck, no HEAVY-first loading order
TRUCK_VESSELS = {"VSL019", "VSL020"}

# ── Scoring weights ────────────────────────────────────────────────────────────

# STRICT rules — large enough to beat any combination of soft bonuses (~400 max)
ETD_VIOLATION  = -1000  # existing container departs before us → we'll block it
WEIGHT_BAD     =  -500  # lighter incoming on top of heavier top → violates HEAVY-first order

# SOFT preferences
ETD_CLOSE      =   +30  # existing departs within 24 h of us (safe neighbour)
SAME_VESSEL_TOP  = +80  # top of stack is same vessel
SAME_VESSEL_MID  = +15  # other stack member is same vessel
SAME_PORT_TOP    = +50  # top of stack is same port
SAME_PORT_MID    = +10  # other stack member is same port
WEIGHT_GOOD      = +60  # incoming weight ≥ top (correct HEAVY-on-top ordering)

HEIGHT_PENALTY    = -25  # per tier — keep stacks short to limit blast radius
BLOCK_FULL        = -150 # block occupancy > 85 %

# ──────────────────────────────────────────────────────────────────────────────


class HeuristicStrategy(PlacementStrategy):
    """Score-based heuristic encoding terminal domain knowledge."""

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _etd(self, s: str) -> float:
        """Parse ISO-8601 departure string → Unix timestamp (cached)."""
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except Exception:
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    # ── Stack scorer ───────────────────────────────────────────────────────────

    def _score(
        self,
        yard_state: YardState,
        block: str,
        bay: int,
        row: int,
        event: Event,
        inc_etd: float,
        apply_weight: bool,      # False for truck-only vessels
        block_penalty: float,    # pre-computed per block
    ) -> float:
        bi = yard_state.blocks[block]
        height = yard_state.get_stack_height(block, bay, row)

        if height >= bi.tiers:
            return float("-inf")  # stack full

        score = HEIGHT_PENALTY * height + block_penalty

        if height == 0:
            return score  # empty stack — no composition to analyse

        top_cid = yard_state.get_container_at(block, bay, row, height)
        top_info = yard_state.get_container_info(top_cid) if top_cid else None

        for tier in range(1, height + 1):
            cid = yard_state.get_container_at(block, bay, row, tier)
            if not cid:
                continue
            info = yard_state.get_container_info(cid)
            if not info:
                continue

            c_etd = self._etd(info.departure_time)
            is_top = (tier == height)

            # ── Rule 1: ETD ordering (STRICT) ─────────────────────────────
            # Container below us departs before us → we will block its retrieval
            if c_etd < inc_etd - 3_600:
                score += ETD_VIOLATION          # -1000 per violation
            elif abs(c_etd - inc_etd) <= 86_400:
                score += ETD_CLOSE              # +30 for safe neighbours

            # ── Preference: vessel / port grouping ─────────────────────────
            if info.vessel_id == event.vessel_id:
                score += SAME_VESSEL_TOP if is_top else SAME_VESSEL_MID

            if info.port_of_discharge == event.port_of_discharge:
                score += SAME_PORT_TOP if is_top else SAME_PORT_MID

        # ── Rule 2: weight ordering (STRICT for ship vessels) ─────────────
        # HEAVY retrieved first during LOAD → must sit on top
        if apply_weight and top_info:
            inc_rank = WEIGHT_RANK.get(event.weight_class, 2)
            top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
            if inc_rank >= top_rank:
                score += WEIGHT_GOOD            # +60 correct order
            else:
                score += WEIGHT_BAD             # -500 strict violation

        return score

    # ── Public API ─────────────────────────────────────────────────────────────

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd = self._etd(event.departure_time)
        apply_weight = event.vessel_id not in TRUCK_VESSELS

        # Cache block occupancy once per call (O(blocks) not O(total stacks))
        block_penalty: Dict[str, float] = {}
        for bn, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(bn)
            block_penalty[bn] = BLOCK_FULL if (cap > 0 and occ / cap > 0.85) else 0.0

        best_pos: Optional[Position] = None
        best_score = float("-inf")

        for block_name, bi in yard_state.blocks.items():
            bp = block_penalty[block_name]
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    s = self._score(
                        yard_state, block_name, bay, row,
                        event, inc_etd, apply_weight, bp,
                    )
                    if s > best_score:
                        best_score = s
                        h = yard_state.get_stack_height(block_name, bay, row)
                        best_pos = Position(block_name, bay, row, h + 1)

        if best_pos is not None:
            return best_pos

        # Fallback: yard nearly full — any open slot
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        return Position(block_name, bay, row, h + 1)

        first_block = list(yard_state.blocks.keys())[0]
        return Position(first_block, 1, 1, 999)

    def on_container_retrieved(
        self, container_id: str, position: Position, reshuffles: int
    ) -> None:
        pass  # Phase 2: collect (features, reshuffles) for XGBoost here
