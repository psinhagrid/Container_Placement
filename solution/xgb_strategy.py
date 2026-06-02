"""XGBoost placement strategy.

Workflow per placement decision:
  1. Scan all stacks → find the global minimum height among valid stacks
  2. Collect all candidates at min_height (and min_height+1 if < 10 candidates)
  3. Extract features for each candidate
  4. XGBoost batch-predicts reshuffles for all candidates
  5. Return position with lowest predicted reshuffles

Falls back to greedy+tiebreaker (v6) if model not loaded or no candidates found.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState
from solution.features import FEATURES

WEIGHT_RANK   = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
WEIGHT_OFFSET = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}
TRUCK_VESSELS = {"VSL019", "VSL020"}
MODEL_PATH    = "solution/xgb_model.pkl"
SCHEDULE_PATH = "data/vessel_schedule.json"
ONE_HOUR      = 3_600.0


class XGBStrategy(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}

        # Load vessel schedule for port order features
        self._vessel_ports: Dict[str, List[str]] = {}
        sched: dict = {}
        if os.path.exists(SCHEDULE_PATH):
            with open(SCHEDULE_PATH) as f:
                sched = json.load(f)
            for v in sched.get("vessels", []):
                self._vessel_ports[v["vessel_id"]] = v.get("ports", [])

        # Build vessel rotation load windows for hours_until_load feature
        self._vessel_rotations: Dict[str, List[dict]] = {}
        for v in sched.get("vessels", []):
            vid = v["vessel_id"]
            rots = []
            for rot in v.get("rotations", []):
                ls = self._parse_ts(rot.get("load_start", ""))
                etd_ts = self._parse_ts(rot.get("etd", ""))
                if ls > 0:
                    rots.append({"etd_ts": etd_ts, "load_start_ts": ls})
            self._vessel_rotations[vid] = rots

        # Precompute stack tracking
        self._container_etd:         Dict[str, float] = {}
        self._container_intra_rank:  Dict[str, int]   = {}
        self._stack_containers:      Dict[tuple, Set[str]] = {}

        for c in initial_state.get("containers", []):
            cid = c["container_id"]
            p   = c["position"]
            key = (p["block"], p["bay"], p["row"])
            etd = self._etd(c.get("departure_time", ""))
            ir  = self._intra_rank(c.get("vessel_id", ""),
                                   c.get("port_of_discharge", ""),
                                   c.get("weight_class", "MEDIUM"))
            self._container_etd[cid]        = etd
            self._container_intra_rank[cid] = ir
            self._stack_containers.setdefault(key, set()).add(cid)

        # Track initial container IDs for initial_below_count feature
        self._initial_cids: Set[str] = {
            c["container_id"] for c in initial_state.get("containers", [])
        }
        self._current_time: float = 0.0

        # Load pre-trained model
        if os.path.exists(MODEL_PATH):
            self._model = joblib.load(MODEL_PATH)
            print(f"[XGBStrategy] Model loaded from {MODEL_PATH}")
        else:
            self._model = None
            print(f"[XGBStrategy] WARNING: model not found at {MODEL_PATH} — using v6 fallback")

    def _parse_ts(self, s: str) -> float:
        try:
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            return 0.0

    def on_event(self, event: Event) -> None:
        ts = self._parse_ts(event.timestamp)
        if ts > 0:
            self._current_time = ts

    def _intra_rank(self, vessel_id: str, port: str, weight: str) -> int:
        if vessel_id in TRUCK_VESSELS:
            return 0
        ports    = self._vessel_ports.get(vessel_id, [])
        port_idx = ports.index(port) if port in ports else len(ports)
        return port_idx * 3 + WEIGHT_OFFSET.get(weight, 1)

    def _unsafe_count(self, block: str, bay: int, row: int, inc_etd: float) -> int:
        key = (block, bay, row)
        return sum(
            1 for cid in self._stack_containers.get(key, set())
            if self._container_etd.get(cid, float("inf")) < inc_etd - ONE_HOUR
        )

    def _unsafe_rank_count(self, block: str, bay: int, row: int,
                            inc_ir: int, inc_etd: float) -> int:
        key = (block, bay, row)
        return sum(
            1 for cid in self._stack_containers.get(key, set())
            if self._container_intra_rank.get(cid, 0) < inc_ir
            and self._container_etd.get(cid, float("inf")) < inc_etd + ONE_HOUR
        )

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        key = (position.block, position.bay, position.row)
        self._stack_containers.get(key, set()).discard(container_id)
        self._container_etd.pop(container_id, None)
        self._container_intra_rank.pop(container_id, None)

    def _hours_until_load(self, vessel_id: str, etd_ts: float) -> float:
        """Hours until this container's vessel load window. Capped at 240h."""
        if vessel_id in TRUCK_VESSELS or self._current_time == 0:
            return 240.0
        best = 240.0
        for rot in self._vessel_rotations.get(vessel_id, []):
            if abs(rot["etd_ts"] - etd_ts) < 3600:  # match by ETD proximity
                ls = rot["load_start_ts"]
                hours = max(0.0, (ls - self._current_time) / 3600)
                best = min(best, hours)
        return round(best, 2)

    def _same_group_in_stack(self, block: str, bay: int, row: int,
                              inc_etd: float, inc_ir: int) -> int:
        """Containers in stack with same (vessel, port, weight) as incoming."""
        key = (block, bay, row)
        return sum(
            1 for cid in self._stack_containers.get(key, set())
            if abs(self._container_etd.get(cid, float("inf")) - inc_etd) < ONE_HOUR
            and self._container_intra_rank.get(cid, -1) == inc_ir
        )

    def _initial_below_count(self, block: str, bay: int, row: int,
                               inc_etd: float) -> int:
        """Initial-state containers in stack with ETD < ours (contribute to bottleneck)."""
        key = (block, bay, row)
        return sum(
            1 for cid in self._stack_containers.get(key, set())
            if cid in self._initial_cids
            and self._container_etd.get(cid, float("inf")) < inc_etd - ONE_HOUR
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

    # ── Feature extraction for one candidate stack ─────────────────────────────

    def _stack_features(
        self,
        yard_state: YardState,
        block: str,
        bay: int,
        row: int,
        h: int,
        inc_etd: float,
        inc_rank: int,
        block_occ: float,
        days_until_dep: float,
        is_truck: int,
        event: Event,
        unsafe_cnt: int = 0,
        min_height_pct: float = 0.0,
    ) -> dict:
        top_etd_gap = 0.0
        same_vessel  = 0
        same_port    = 0
        weight_ok    = 1
        top_rank     = 0

        if h > 0:
            top_cid = yard_state.get_container_at(block, bay, row, h)
            top_info = yard_state.get_container_info(top_cid) if top_cid else None
            if top_info:
                top_etd = self._etd(top_info.departure_time)
                top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                top_etd_gap = (
                    (top_etd - inc_etd) / 86_400
                    if top_etd != float("inf") and inc_etd != float("inf")
                    else 0.0
                )
                same_vessel = int(top_info.vessel_id == event.vessel_id)
                same_port   = int(top_info.port_of_discharge == event.port_of_discharge)
                if event.vessel_id not in TRUCK_VESSELS:
                    weight_ok = int(inc_rank >= top_rank)

        # New features
        inc_port_ord = 0 if event.vessel_id in TRUCK_VESSELS else (
            self._vessel_ports.get(event.vessel_id, []).index(event.port_of_discharge)
            if event.port_of_discharge in self._vessel_ports.get(event.vessel_id, [])
            else len(self._vessel_ports.get(event.vessel_id, []))
        )
        inc_w_off = WEIGHT_OFFSET.get(event.weight_class, 1)
        inc_ir    = inc_port_ord * 3 + inc_w_off

        top_ir = self._container_intra_rank.get(
            yard_state.get_container_at(block, bay, row, h) or "", 0
        ) if h > 0 else 0

        unsafe_rank_cnt = self._unsafe_rank_count(block, bay, row, inc_ir,
                                                   self._etd(event.departure_time))

        return {
            "stack_height":       h,
            "top_etd_gap_days":   round(top_etd_gap, 4),
            "same_vessel":        same_vessel,
            "same_port":          same_port,
            "weight_ok":          weight_ok,
            "weight_rank_inc":    inc_rank,
            "weight_rank_top":    top_rank,
            "block_occ":          round(block_occ, 4),
            "days_until_dep":     round(days_until_dep, 4),
            "unsafe_count":       unsafe_cnt,
            "intra_vessel_rank":  inc_ir,
            "unsafe_rank_count":  unsafe_rank_cnt,
            "rank_gap_to_top":    float(inc_ir - top_ir),
            "unsafe_x_height":    unsafe_cnt * h,
            "min_height_pct":     min_height_pct,
            "hours_until_load":   self._hours_until_load(event.vessel_id, inc_etd),
            "same_group_in_stack": self._same_group_in_stack(block, bay, row, inc_etd, inc_ir),
            "initial_below_count": self._initial_below_count(block, bay, row, inc_etd),
        }

    # ── Main placement ─────────────────────────────────────────────────────────

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd = self._etd(event.departure_time)
        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)
        placement_ts = self._etd(event.timestamp)
        is_truck = int(event.vessel_id in TRUCK_VESSELS)
        days_until_dep = (
            max(0.0, (inc_etd - placement_ts) / 86_400)
            if inc_etd != float("inf") and placement_ts != float("inf")
            else 0.0
        )

        # Pre-compute block occupancies once
        block_occ_map: Dict[str, float] = {}
        for bn, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(bn)
            block_occ_map[bn] = occ / cap if cap > 0 else 0.0

        # ── Collect all candidates ─────────────────────────────────────────────
        # Candidate: (h, block, bay, row)
        candidates: List[Tuple[int, str, int, int]] = []
        min_h = float("inf")

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h >= bi.tiers:
                        continue
                    if h < min_h:
                        min_h = h
                    candidates.append((h, block_name, bay, row))

        if not candidates or min_h == float("inf"):
            return self._fallback(yard_state)

        # Only score candidates at exact min height — never sacrifice height balance
        filtered = [(h, bn, bay, row) for h, bn, bay, row in candidates if h == min_h]

        # Compute min_height_pct for yard context feature (same as in collectors)
        min_h_count  = sum(1 for h, bn, bay, row in candidates if h == min_h)
        total_open   = len(candidates)
        mh_pct       = round(min_h_count / max(total_open, 1), 4)

        # ── XGBoost scoring ────────────────────────────────────────────────────
        if self._model is not None and filtered:
            rows = []
            positions = []
            for h, bn, bay, row in filtered:
                uc = self._unsafe_count(bn, bay, row, inc_etd)
                feat = self._stack_features(
                    yard_state, bn, bay, row, h,
                    inc_etd, inc_rank,
                    block_occ_map[bn],
                    days_until_dep, is_truck, event,
                    unsafe_cnt=uc,
                    min_height_pct=mh_pct,
                )
                rows.append(feat)
                positions.append(Position(bn, bay, row, h + 1))

            X = pd.DataFrame(rows, columns=FEATURES)
            preds = self._model.predict(X)
            best_idx = int(np.argmin(preds))
            chosen = positions[best_idx]

            # Update stack tracking
            inc_ir = self._intra_rank(event.vessel_id,
                                      event.port_of_discharge,
                                      event.weight_class)
            self._container_etd[event.container_id] = inc_etd
            self._container_intra_rank[event.container_id] = inc_ir
            self._stack_containers.setdefault(
                (chosen.block, chosen.bay, chosen.row), set()
            ).add(event.container_id)

            return chosen

        # ── v6 fallback (no model) ─────────────────────────────────────────────
        return self._v6_fallback(yard_state, event, inc_etd, inc_rank)

    def _v6_fallback(
        self, yard_state: YardState, event: Event,
        inc_etd: float, inc_rank: int,
    ) -> Position:
        """Greedy + domain tiebreakers (v6 logic)."""
        apply_weight = event.vessel_id not in TRUCK_VESSELS
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
                    etd_ok = True
                    vessel_match = False
                    port_match = False
                    weight_ok = True

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

        return best_pos if best_pos is not None else self._fallback(yard_state)

    def _fallback(self, yard_state: YardState) -> Position:
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        return Position(block_name, bay, row, h + 1)
        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

