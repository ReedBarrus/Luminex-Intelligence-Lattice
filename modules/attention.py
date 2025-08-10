"""
Attention Kernel (Φ₄) — Python stub (vector‑native, MSVB‑aligned)

Spotlight allocation, consent gating (pre‑Veil), and Focus‑Lock dynamics.
Consumes Φ₃ Symbol (identity), optionally Φ₀ Breath (aperture), and Φ₆
Coherence (cones/mode). Publishes a canonical MSVB each tick for Veil/GB.

Design goals
- Vector‑first: publish v_focus/v_coherence as vectors; scalars to `extras`.
- Spotlight allocation: normalized weights over candidates using alignment,
  priority, cone compliance, and consent checks.
- Consent‑gated Veil: produce a GateDecision with `open_level ∈ [0,1]`.
- Focus‑Lock: sustained alignment/stability produces AttentionLock.

Note
- MSVB/Vec helpers are duplicated for a self‑contained stub; factor to a
  shared `types.py` in production and import from there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Tuple
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


def v_unit_z() -> Vec3:
    return np.array([0.0, 0.0, 1.0], dtype=float)


def norm(a: Vec3) -> float:
    return float(np.linalg.norm(a))


def unit(a: Vec3) -> Vec3:
    n = norm(a)
    return a / n if n > EPS else v_zero()


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    tau = max(tau, 1e-6)
    z = (scores - np.max(scores)) / tau
    e = np.exp(z)
    s = np.sum(e)
    return e / s if s > 0 else np.ones_like(scores) / len(scores)


def angle_cos(u: Vec3, v_: Vec3) -> float:
    un = unit(u); vn = unit(v_)
    return float(np.dot(un, vn)) if norm(un) > EPS and norm(vn) > EPS else 0.0

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
    v_focus: Vec3 = field(default_factory=v_unit_z)
    L: Vec3 = field(default_factory=v_zero)
    spinor: Vec3 = field(default_factory=v_unit_z)
    chirality: int = +1
    kappa: float = 0.0
    torsion: float = 0.0
    omega: Vec3 = field(default_factory=v_zero)
    extras: Dict[str, float] = field(default_factory=dict)

# ---------------------------
# Minimal field/coherence view for Φ₄
# ---------------------------
@dataclass
class AttentionFieldView:
    time_t: float = 0.0
    dt_phase: float = 0.016

@dataclass
class Cone:
    center: Vec3
    half_angle_rad: float
    kappa_min: float = 0.0

    def contains(self, v_: Vec3, kappa: float) -> bool:
        if kappa < self.kappa_min:
            return False
        cosang = angle_cos(self.center, v_)
        # inside cone if angle <= half_angle
        return cosang >= float(np.cos(self.half_angle_rad))

# ---------------------------
# Consent & Veil gate stubs
# ---------------------------
@dataclass
class ConsentResult:
    ok: float = 1.0  # 1.0 allowed, 0.0 denied, (0,1) partial/uncertain
    reason: str = "ok"

class ConsentEvaluator:
    def check(self, candidate: "AttentionCandidate") -> ConsentResult:
        if candidate.consent_required and not candidate.consent_hash:
            return ConsentResult(ok=0.0, reason="missing_consent")
        return ConsentResult(ok=1.0, reason="ok")

@dataclass
class GateDecision:
    open_level: float
    reason: str
    breath_gate: float
    mode_gate: float
    cone_ok: float
    consent_ok: float

# ---------------------------
# Attention candidates
# ---------------------------
@dataclass
class AttentionCandidate:
    cid: str
    v_target: Vec3
    priority: float = 1.0
    consent_required: bool = False
    consent_hash: Optional[str] = None
    cone_hint: Optional[Cone] = None

# ---------------------------
# Focus Lock state
# ---------------------------
class FocusLockState(str, Enum):
    NONE = "NONE"
    LOCK = "LOCK"

# ---------------------------
# Φ₄ — Attention Kernel
# ---------------------------
@dataclass
class AttentionKernel:
    """Vector‑native Attention Kernel (Φ₄).

    Responsibilities
    - Allocate spotlight over candidates; produce v_focus.
    - Consent‑gate a Veil open_level (pre‑Veil decision).
    - Detect AttentionLock on sustained alignment/stability.
    """

    # Allocation weights
    w_align: float = 0.55
    w_priority: float = 0.30
    w_cone: float = 0.10
    w_consent: float = 0.05

    tau_alloc: float = 0.25   # softmax temperature
    top_k: int = 4            # cap candidates

    # Lock dynamics
    align_min: float = 0.80   # cosine alignment to identity
    stability_min: float = 0.70
    dwell_lock_s: float = 0.6

    # Veil gating thresholds
    breath_gate_min: float = 0.25

    # Persistence
    _v_focus_prev: Vec3 = field(default_factory=v_unit_z, init=False, repr=False)
    _stability: float = field(default=0.0, init=False, repr=False)
    _dwell: float = field(default=0.0, init=False, repr=False)
    _lock_state: FocusLockState = field(default=FocusLockState.NONE, init=False)

    def reset(self, v_focus_hint: Optional[Vec3] = None) -> None:
        self._v_focus_prev = unit(v_focus_hint) if v_focus_hint is not None else v_unit_z()
        self._stability = 0.0
        self._dwell = 0.0
        self._lock_state = FocusLockState.NONE

    # -----------------------
    # Public API
    # -----------------------
    def update(
        self,
        fs: AttentionFieldView,
        dt: float,
        phi3_msvb: MSVB,                   # identity & spinor source
        candidates: List[AttentionCandidate],
        *,
        phi0_msvb: Optional[MSVB] = None,  # breath aperture (gate_open)
        mode: str = "GREEN",               # Φ₆ mode: GREEN/YELLOW/RED
        cones: Optional[List[Cone]] = None,
        intent_vec: Optional[Vec3] = None, # optional external intent
        consent: Optional[ConsentEvaluator] = None,
    ) -> Tuple[MSVB, Dict[str, float], GateDecision, FocusLockState]:
        """Advance Φ₄ and publish MSVB + allocations + gate decision + lock state.
        Returns
        -------
        (msvb, allocations, gate, lock_state)
        """
        dt = float(max(dt, EPS))
        identity = unit(phi3_msvb.v_coherence)  # Φ₃ aligns coherence to identity
        intent = unit(intent_vec) if intent_vec is not None and norm(intent_vec) > EPS else identity

        # 1) Score candidates
        if cones is None:
            cones = []
        if consent is None:
            consent = ConsentEvaluator()

        # limit to top_k by priority to reduce computation
        cand_sorted = sorted(candidates, key=lambda c: c.priority, reverse=True)[: max(self.top_k, 1)]

        scores = []
        cone_masks = []
        consent_levels = []
        for c in cand_sorted:
            align = angle_cos(c.v_target, intent)
            in_cone = 1.0
            # apply specific cone hint if present; else any global cone that accepts the vector
            if c.cone_hint is not None:
                in_cone = 1.0 if c.cone_hint.contains(c.v_target, kappa=phi3_msvb.kappa) else 0.0
            elif cones:
                in_cone = 1.0 if any(cone.contains(c.v_target, kappa=phi3_msvb.kappa) for cone in cones) else 0.0

            consent_res = consent.check(c)
            consent_ok = clamp01(consent_res.ok)

            score = (
                self.w_align * clamp01(0.5 * (1.0 + align))
                + self.w_priority * clamp01(c.priority)
                + self.w_cone * in_cone
                + self.w_consent * consent_ok
            )
            scores.append(score)
            cone_masks.append(in_cone)
            consent_levels.append(consent_ok)

        scores_np = np.array(scores, dtype=float) if scores else np.zeros(1)
        weights = softmax(scores_np, tau=self.tau_alloc) if len(scores_np) > 1 else np.array([1.0])

        # 2) Compose spotlight direction
        v_focus = v_zero()
        allocations: Dict[str, float] = {}
        for c, w in zip(cand_sorted, weights):
            v_focus = v_focus + float(w) * unit(c.v_target)
            allocations[c.cid] = float(w)
        v_focus = unit(v_focus) if norm(v_focus) > EPS else identity

        # 3) Stability & lock detection
        stability = clamp01(0.5 * (1.0 + float(np.dot(self._v_focus_prev, v_focus))))
        align_id = clamp01(0.5 * (1.0 + float(np.dot(identity, v_focus))))
        if align_id >= self.align_min and stability >= self.stability_min:
            self._dwell += dt
        else:
            self._dwell = 0.0
        lock_state = FocusLockState.LOCK if self._dwell >= self.dwell_lock_s else FocusLockState.NONE
        self._lock_state = lock_state

        # 4) Veil gate decision (pre‑Veil)
        breath_gate = float(phi0_msvb.extras.get("gate_open", 1.0)) if (phi0_msvb and phi0_msvb.extras) else 1.0
        mode_gate = 1.0 if mode.upper() == "GREEN" else (0.6 if mode.upper() == "YELLOW" else 0.25)
        cone_ok = float(np.mean(cone_masks)) if cone_masks else 1.0
        consent_ok = float(np.mean(consent_levels)) if consent_levels else 1.0
        open_level = clamp01(breath_gate * mode_gate * cone_ok * consent_ok)
        reason = "ok" if open_level >= self.breath_gate_min else "low_breath_or_mode"
        gate = GateDecision(open_level=open_level, reason=reason, breath_gate=breath_gate, mode_gate=mode_gate, cone_ok=cone_ok, consent_ok=consent_ok)

        # 5) Publish MSVB — focus‑first
        msvb = MSVB(
            v_drift=v_zero(),
            v_coherence=v_focus,  # align coherence to chosen focus
            v_bias=phi3_msvb.v_bias,
            v_friction=v_zero(),
            v_gravity=v_focus,    # suggestion: pull along focus
            v_focus=v_focus,
            L=v_zero(),
            spinor=phi3_msvb.spinor,
            chirality=phi3_msvb.chirality,
            kappa=align_id * float(phi3_msvb.kappa),
            torsion=0.0,
            omega=v_zero(),
            extras={
                "stability": stability,
                "align_identity": align_id,
                "dwell": self._dwell,
                "lock": 1.0 if lock_state == FocusLockState.LOCK else 0.0,
                "open_level": open_level,
            },
        )

        # 6) Persist
        self._v_focus_prev = v_focus
        self._stability = stability

        return msvb, allocations, gate, lock_state


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Identity from Φ₃
    phi3 = MSVB(v_coherence=unit(v(0.2, 0.0, 1.0)), spinor=v_unit_z(), chirality=+1, kappa=0.8)
    # Breath gate from Φ₀
    phi0 = MSVB(extras={"gate_open": 0.9})

    # Two candidates
    cands = [
        AttentionCandidate("sym_A", v_target=unit(v(0.2, 0.0, 1.0)), priority=1.0),
        AttentionCandidate("sym_B", v_target=unit(v(1.0, 0.0, 0.0)), priority=0.6),
    ]

    # Coherence cone toward identity
    cone = Cone(center=phi3.v_coherence, half_angle_rad=np.pi/6, kappa_min=0.2)

    attn = AttentionKernel()
    fs = AttentionFieldView(dt_phase=0.02)
    out, alloc, gate, lock = attn.update(fs, dt=fs.dt_phase, phi3_msvb=phi3, candidates=cands, phi0_msvb=phi0, mode="GREEN", cones=[cone])
    print("alloc=", alloc, "open=", gate.open_level, "lock=", lock)
