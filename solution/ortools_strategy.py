"""OR-Tools CP-SAT Placement Strategy — Phase 4.

Two-phase approach:

Phase A — Pre-assignment (initialize, runs once):
  Read vessel_schedule.json.
  CP-SAT assigns each vessel to a dedicated block ensuring:
  - Vessels with overlapping LOAD windows → different blocks
    (when vessel A loads, vessel B's containers are in a separate block)
  - Balanced block utilization (spread vessels evenly)

Phase B — Placement (place_container, per container):
  Route container to its vessel's assigned block.
  Within block: vessel+port grouping + strict weight ordering (HEAVY on top).
  Empty-stack-first policy prevents contaminating initial-state stacks.

Priority within block:
  1. Existing same-(vessel,port) stack where weight ordering is OK
  2. Empty stack in assigned block (start new group stack)
  3. Existing same-(vessel,port) stack ignoring weight order (purity > weight)
  4. Spill: empty stack in any other block
  5. True greedy last resort

Why this beats our previous attempts:
  - Block assignment ensures LOAD events retrieve from one block only → no cross-block reshuffles
  - Vessel+port grouping ensures containers loaded together are in same stacks
  - Weight ordering ensures HEAVY retrieved first → 0 reshuffles within stack
  - Empty-first prevents our containers from being placed above initial-state containers
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}
SHIP_BLOCKS   = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08"]
TRUCK_BLOCKS  = ["B09", "B10"]
ALL_BLOCKS    = SHIP_BLOCKS + TRUCK_BLOCKS
SCHEDULE_PATH = "data/vessel_schedule.json"


class ORToolsStrategy(PlacementStrategy):

    # ── Initialization ─────────────────────────────────────────────────────────

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}
        self._vessel_block: Dict[str, str] = {}   # vessel_id → block_name

        if os.path.exists(SCHEDULE_PATH):
            with open(SCHEDULE_PATH) as f:
                schedule = json.load(f)
            self._solve_block_assignment(schedule)
        else:
            print("[ORTools] vessel_schedule.json not found — using simple assignment")

    def _parse_ts(self, s: str) -> float:
        try:
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            return 0.0

    def _windows_overlap(self, a: List[Tuple], b: List[Tuple]) -> bool:
        """Return True if any window in list a overlaps any window in list b."""
        for (sa, ea) in a:
            for (sb, eb) in b:
                if sa < eb and sb < ea:
                    return True
        return False

    def _solve_block_assignment(self, schedule: dict) -> None:
        """CP-SAT: assign each vessel to a block, respecting load-window conflicts."""
        ship_vessels = [v for v in schedule["vessels"]
                        if v["vessel_id"] not in TRUCK_VESSELS]

        # Build load windows per vessel (union across all rotations)
        vessel_windows: Dict[str, List[Tuple[float, float]]] = {}
        for v in ship_vessels:
            vid = v["vessel_id"]
            windows = []
            for rot in v.get("rotations", []):
                ls = self._parse_ts(rot.get("load_start", ""))
                le = self._parse_ts(rot.get("load_end", ""))
                if ls > 0 and le > 0:
                    windows.append((ls, le))
            vessel_windows[vid] = windows

        vessel_ids = [v["vessel_id"] for v in ship_vessels]

        # Build conflict graph: which pairs of vessels overlap?
        conflicts: Dict[str, Set[str]] = {vid: set() for vid in vessel_ids}
        for i, vi in enumerate(vessel_ids):
            for vj in vessel_ids[i + 1:]:
                if self._windows_overlap(vessel_windows[vi], vessel_windows[vj]):
                    conflicts[vi].add(vj)
                    conflicts[vj].add(vi)

        # Try CP-SAT first, fall back to greedy graph coloring
        try:
            from ortools.sat.python import cp_model
            self._cpsat_assign(vessel_ids, conflicts, cp_model)
            print(f"[ORTools] CP-SAT block assignment complete")
        except Exception as e:
            print(f"[ORTools] CP-SAT unavailable ({e}), using greedy graph coloring")
            self._greedy_color(vessel_ids, conflicts)

        # Assign truck vessels to truck blocks
        for i, vid in enumerate(TRUCK_VESSELS):
            self._vessel_block[vid] = TRUCK_BLOCKS[i % len(TRUCK_BLOCKS)]

        print("[ORTools] Vessel→Block:", self._vessel_block)

    def _cpsat_assign(self, vessel_ids, conflicts, cp_model) -> None:
        """CP-SAT optimal block assignment."""
        model = cp_model.CpModel()
        n_blocks = len(SHIP_BLOCKS)

        block_vars = {vid: model.NewIntVar(0, n_blocks - 1, f"b_{vid}")
                      for vid in vessel_ids}

        # Hard constraint: conflicting vessels must have different blocks
        for vi in vessel_ids:
            for vj in conflicts.get(vi, set()):
                if vi < vj:  # avoid duplicate constraints
                    model.Add(block_vars[vi] != block_vars[vj])

        # Soft objective: balance block loads
        # Count vessels per block and minimize the max
        block_counts = []
        for b in range(n_blocks):
            count = model.NewIntVar(0, len(vessel_ids), f"cnt_{b}")
            model.Add(count == sum(
                model.NewBoolVar(f"in_{vid}_{b}") for vid in vessel_ids
            ))
            block_counts.append(count)

        max_count = model.NewIntVar(0, len(vessel_ids), "max_count")
        model.AddMaxEquality(max_count, block_counts)
        model.Minimize(max_count)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for vid in vessel_ids:
                self._vessel_block[vid] = SHIP_BLOCKS[solver.Value(block_vars[vid])]
        else:
            self._greedy_color(vessel_ids, conflicts)

    def _greedy_color(self, vessel_ids, conflicts) -> None:
        """Greedy graph coloring fallback — sorts by conflict count."""
        ordered = sorted(vessel_ids, key=lambda v: len(conflicts.get(v, set())),
                         reverse=True)
        for vid in ordered:
            used = {self._vessel_block[c] for c in conflicts.get(vid, set())
                    if c in self._vessel_block}
            for block in SHIP_BLOCKS:
                if block not in used:
                    self._vessel_block[vid] = block
                    break
            else:
                # All blocks used by conflicts — use least-loaded block
                self._vessel_block[vid] = SHIP_BLOCKS[
                    len(self._vessel_block) % len(SHIP_BLOCKS)
                ]

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
        if event.vessel_id in TRUCK_VESSELS:
            pos = self._find_empty_in(yard_state, TRUCK_BLOCKS[0])
            return pos or self._greedy(yard_state)

        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)
        vessel = event.vessel_id
        port = event.port_of_discharge
        target = self._vessel_block.get(vessel, SHIP_BLOCKS[0])

        # 1. Existing same-(vessel,port) stack in target block — weight OK
        pos = self._group_stack(yard_state, target, vessel, port, inc_rank, strict=True)
        if pos: return pos

        # 2. Empty stack in target block (start new group stack)
        pos = self._find_empty_in(yard_state, target)
        if pos: return pos

        # 3. Same-(vessel,port) stack in target block — weight relaxed (purity > weight)
        pos = self._group_stack(yard_state, target, vessel, port, inc_rank, strict=False)
        if pos: return pos

        # 4. Empty stack in any other block (spill)
        for block in ALL_BLOCKS:
            if block == target: continue
            pos = self._find_empty_in(yard_state, block)
            if pos: return pos

        # 5. Group stack in any block (weight OK)
        for block in ALL_BLOCKS:
            if block == target: continue
            pos = self._group_stack(yard_state, block, vessel, port, inc_rank, strict=True)
            if pos: return pos

        # 6. True greedy last resort
        return self._greedy(yard_state)

    # ── Stack finders ──────────────────────────────────────────────────────────

    def _group_stack(
        self, yard_state: YardState, block: str,
        vessel: str, port: str, inc_rank: int, strict: bool,
    ) -> Optional[Position]:
        if block not in yard_state.blocks:
            return None
        bi = yard_state.blocks[block]
        best_pos: Optional[Position] = None
        best_h = float("inf")

        for bay in range(1, bi.bays + 1):
            for row in range(1, bi.rows + 1):
                h = yard_state.get_stack_height(block, bay, row)
                if h == 0 or h >= bi.tiers:
                    continue
                top_cid = yard_state.get_container_at(block, bay, row, h)
                top_info = yard_state.get_container_info(top_cid) if top_cid else None
                if not top_info:
                    continue
                if top_info.vessel_id != vessel or top_info.port_of_discharge != port:
                    continue
                if strict:
                    top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                    if inc_rank < top_rank:
                        continue
                if h < best_h:
                    best_h = h
                    best_pos = Position(block, bay, row, h + 1)

        return best_pos

    def _find_empty_in(self, yard_state: YardState, block: str) -> Optional[Position]:
        if block not in yard_state.blocks:
            return None
        bi = yard_state.blocks[block]
        for bay in range(1, bi.bays + 1):
            for row in range(1, bi.rows + 1):
                if yard_state.get_stack_height(block, bay, row) == 0:
                    return Position(block, bay, row, 1)
        return None

    def _greedy(self, yard_state: YardState) -> Position:
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
        return best_pos or Position(SHIP_BLOCKS[0], 1, 1, 999)

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        pass
