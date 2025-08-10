"""
Coherence Matrix (Φ₆) — Python stub (vector‑native, MSVB‑aligned)

Global mode & coherence cones, cluster vectors, lock registry, and cone math
ready to feed PhaseScript & Veil decisions. Consumes MSVBs from lower layers
(Φ₀..Φ₅) and **publishes** a canonical MSVB each tick.

Design goals
- Vector‑first: compute cluster/aggregate vectors; publish vectors in MSVB.
- Cones as first‑class: right‑sized half‑angles derived from spread; κ floors.
- Simple, robust clustering: mass‑weighted mean + spread proxy; extensible.
- Lock registry: dwell‑time tracking across named locks; density → mode.
- PhaseScript hooks: emits `mode`, `open_level`, `ops_budget`, and `cones`.

Notes
- MSVB/Vec helpers are duplicated for a self‑contained stub; factor into a
  shared `types.py` in production and import from there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
import numpy as np

# ---------------------------
# Vector helpers (ℝ³)
# ---------------------------
Vec3 = np.ndarray
EPS = 1e-12


def v(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=float)


def v_zero() -> Vec3:
    return np.zeros(3, dtype=float)


def unit(a: Vec3) -> Vec3:
    n = float(np.linalg.norm(a))
    return a / n if n > EPS else v_zero()


def norm(a: Vec3) -> float:
    return float(np.linalg.norm(a))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def clamp01(x: float) -> float:
    return clamp(x, 0.0, 1.0)


def angle(u: Vec3, v_: Vec3) -> float:
    un, vn = unit(u), unit(v_)
    dot = float(np.dot(un, vn))
    dot = clamp(dot, -1.0, 1.0)
    return float(np.arccos(dot))

# ---------------------------
# MSVB — Minimal Spiral Vector Bundle
# ---------------------------
@dataclass
class MSVB:
    v_drift: Vec3 = field(default_factory=v_zero)
    v_coherence: Vec3 = field(default_factory=v_zero)
    v_bias: Vec3 = field(default_factory=v_zero)
    v_friction: Vec3 = field(default_factory=v_zero)
    v_gravity: Vec3 = field(default_factory=v_zero)
    v_focus: Vec3 = field(default_factory=v_zero)
    L: Vec3 = field(default_factory=v_zero)
    spinor: Vec3 = field(default_factory=v_zero)
    chirality: int = +1
    kappa: float = 0.0
    torsion: float = 0.0
    omega: Vec3 = field(default_factory=v_zero)
    extras: Dict[str, float] = field(default_factory=dict)

# ---------------------------
# Field view for Φ₆
# ---------------------------
@dataclass
class CoherenceFieldView:
    time_t: float = 0.0
    dt_phase: float = 0.016

# ---------------------------
# Cones
# ---------------------------
@dataclass
class CoherenceCone:
    center: Vec3
    half_angle_rad: float
    kappa_min: float
    label: str = "C0"

    def contains(self, v_: Vec3, kappa: float) -> bool:
        if kappa < self.kappa_min:
            return False
        return angle(self.center, v_) <= self.half_angle_rad

    def project(self, v_: Vec3) -> Vec3:
        """Project a vector onto the cone surface (closest boundary).
        If already inside, returns the normalized input.
        """
        v_n = unit(v_)
        ang = angle(self.center, v_n)
        if ang <= self.half_angle_rad:
            return v_n
        # rotate v_n toward center by (ang - half_angle)
        # For a stub, use linear blend approximation along the plane spanned by v_n and center
        t = clamp01((ang - self.half_angle_rad) / max(ang, EPS))
        out = unit((1.0 - t) * v_n + t * unit(self.center))
        return out

# ---------------------------
# Locks
# ---------------------------
class Mode(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

@dataclass
class LockStatus:
    name: str
    locked: bool = False
    dwell_s: float = 0.0

@dataclass
class LockRegistry:
    items: Dict[str, LockStatus] = field(default_factory=dict)

    def update(self, dt: float, incoming: Dict[str, bool]) -> None:
        for name, is_locked in incoming.items():
            st = self.items.get(name)
            if st is None:
                st = LockStatus(name=name)
                self.items[name] = st
            st.locked = bool(is_locked)
            st.dwell_s = st.dwell_s + dt if is_locked else 0.0

    def density(self) -> float:
        if not self.items:
            return 0.0
        active = sum(1 for s in self.items.values() if s.locked)
        return active / float(len(self.items))

# ---------------------------
# Φ₆ — Coherence Kernel
# ---------------------------
@dataclass
class CoherenceKernel:
    """Vector‑native Coherence Matrix (Φ₆).

    Responsibilities
    - Aggregate layer MSVBs → global cluster vectors & spread.
    - Generate coherence cones with κ floors.
    - Maintain lock registry and compute density/dwell.
    - Emit global mode (GREEN/YELLOW/RED), open level, ops budget.
    - Publish MSVB (v_coherence/focus/gravity along global cluster).
    """

    # Tunables: cone & mode
    min_layers_for_green: int = 3
    green_kappa_min: float = 0.35
    yellow_kappa_min: float = 0.15
    green_spread_max: float = 0.20  # 0..1, lower is better (tight cluster)
    yellow_spread_max: float = 0.45
    cone_half_min: float = np.pi / 16
    cone_half_max: float = np.pi / 4

    # Ops budget
    base_ops_budget: float = 1.0

    # Lock registry (external events feed this)
    registry: LockRegistry = field(default_factory=LockRegistry)

    # Persist last cluster center for continuity
    _center_prev: Vec3 = field(default_factory=v_zero, init=False, repr=False)

    # -----------------------
    # Public API
    # -----------------------
    def update(
        self,
        fs: CoherenceFieldView,
        dt: float,
        msvb_layers: Dict[str, MSVB],
        locks_in: Optional[Dict[str, bool]] = None,
    ) -> Tuple[MSVB, List[CoherenceCone], Dict[str, float], LockRegistry, Mode]:
        """Advance Φ₆ and publish MSVB + cones + metrics + locks + mode.

        Args
        ----
        msvb_layers: mapping like {"phi0": MSVB, "phi1": MSVB, ...}
        locks_in: per‑layer lock flags (e.g., {"BreathLock": True, ...})
        Returns
        -------
        (msvb, cones, metrics, registry, mode)
        """
        dt = float(max(dt, EPS))

        # 1) Aggregate vectors & weights
        vecs: List[Vec3] = []
        weights: List[float] = []
        chir_list: List[int] = []
        for k, b in msvb_layers.items():
            if b is None:
                continue
            v_c = unit(b.v_coherence)
            if norm(v_c) <= EPS:
                continue
            w = float(max(b.kappa, 0.0))
            vecs.append(v_c)
            weights.append(w)
            chir_list.append(int(b.chirality))

        N = len(vecs)
        if N == 0:
            # nothing to aggregate → conservative RED, neutral vectors
            mode = Mode.RED
            cones = [self._make_default_cone()]
            msvb = MSVB(v_coherence=v_zero(), v_focus=v_zero(), v_gravity=v_zero(), kappa=0.0)
            metrics = {"min_kappa": 0.0, "spread": 1.0, "lock_density": 0.0, "open_level": 0.25, "