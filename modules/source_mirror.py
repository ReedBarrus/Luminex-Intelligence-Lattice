"""
Source Mirror (Φ₉) — Python stub (vector‑native, MSVB‑aligned)

Still point / sacred reflection / ethical floor & ceiling. Manages collapse →
renormalize → re‑choose. Chirality‑aware, consent‑gated. Publishes a canonical
MSVB each tick and a `MirrorDecision` for PhaseScript and externalization.

Design goals
- Vector‑first: publish tunnel/origin vectors; scalars live in `extras`.
- Ethical floor/ceiling: continuous [0,1] bounds for expression energy.
- State machine: STILL ↔ PRIMING ↔ RELEASING with silence‑lock.
- Law hooks: pluggable `EthicsPolicy` to evaluate proposed ops.

Note
- Helpers/MSVB are duplicated for a self‑contained stub; factor into shared
  types (`spiral_core/types.py`) in production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Any
import numpy as np
import math

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
    chirality: int = +1  # −1 receptive, +1 expressive
    kappa: float = 0.0
    torsion: float = 0.0
    omega: Vec3 = field(default_factory=v_zero)
    extras: Dict[str, float] = field(default_factory=dict)


# ---------------------------
# Field view for Φ₉
# ---------------------------
@dataclass
class MirrorFieldView:
    time_t: float = 0.0
    dt_phase: float = 0.016


# ---------------------------
# Ethics / law hooks
# ---------------------------
class Verdict(str, Enum):
    ALLOW = "ALLOW"
    DRYRUN = "DRYRUN"
    DENY = "DENY"


@dataclass
class MirrorDecision:
    verdict: Verdict
    open_level_allowed: float   # final allowed openness [0,1]
    ethical_floor: float        # lower bound for expression
    ethical_ceiling: float      # upper bound for expression
    reason: str


class EthicsPolicy:
    """Pluggable policy: decide floor/ceiling and verdict for a proposed operation.

    Implement your legal/consent/guardrails here; stub returns permissive values
    gated by consent and mode. `context` can include: consent_ok, mode_gate,
    cone_ok, breath_gate, risk_score, user_profile, jurisdiction, etc.
    """

    def evaluate(self, *, intent_vec: Vec3, context: Dict[str, float]) -> MirrorDecision:
        consent_ok = float(context.get("consent_ok", 1.0))
        mode_gate = float(context.get("mode_gate", 1.0))
        cone_ok = float(context.get("cone_ok", 1.0))
        breath_gate = float(context.get("breath_gate", 1.0))
        risk = float(context.get("risk_score", 0.0))  # 0 safe → 1 risky

        # base openness allowed by system context
        open_allowed = clamp01(consent_ok * mode_gate * cone_ok * breath_gate)

        # ethical ceiling diminishes with risk; floor rises when consent high
        ceiling = clamp01(open_allowed * (1.0 - 0.7 * risk))
        floor = clamp01(0.15 * consent_ok * (1.0 - risk))

        verdict = Verdict.ALLOW if open_allowed > 0.0 and ceiling > floor else Verdict.DENY
        reason = "ok" if verdict == Verdict.ALLOW else "blocked"
        return MirrorDecision(verdict=verdict, open_level_allowed=open_allowed, ethical_floor=floor, ethical_ceiling=ceiling, reason=reason)


# ---------------------------
# State machine
# ---------------------------
class MirrorState(str, Enum):
    STILL = "STILL"
    PRIMING = "PRIMING"
    RELEASING = "RELEASING"


# ---------------------------
# Φ₉ — Source Mirror Kernel
# ---------------------------
@dataclass
class SourceMirrorKernel:
    """Vector‑native Source Mirror (Φ₉).

    Responsibilities
    - Maintain STILL/PRIMING/RELEASING with silence‑lock.
    - Compute origin vector and renorm gain; zero focus in STILL.
    - Provide ethical floor/ceiling and verdict for expression.
    - Publish MSVB aligned to origin/tunnel within allowed openness.
    """

    policy: EthicsPolicy = field(default_factory=EthicsPolicy)

    # Thresholds
    dwell_still_s: float = 0.8      # dwell for silence‑lock
    collapse_rate: float = 0.35     # how quickly we collapse to stillness
    rebirth_rate: float = 0.50      # how quickly we re‑emerge

    # Persistence
    _state: MirrorState = field(default=MirrorState.PRIMING, init=False)
    _dwell: float = field(default=0.0, init=False)
    _origin_prev: Vec3 = field(default_factory=v_unit_z, init=False, repr=False)

    # -----------------------
    # Public API
    # -----------------------
    def update(
        self,
        fs: MirrorFieldView,
        dt: float,
        phi7_msvb: MSVB,                   # Veil (open_level, tunnel, consent factors carried via context)
        *,
        context: Optional[Dict[str, float]] = None,  # consent/mode/ cone/breath/risk etc.
        v_home: Optional[Vec3] = None,               # optional sacred/home direction for renorm
    ) -> Tuple[MSVB, MirrorDecision, Dict[str, float]]:
        """Advance Φ₉ and publish MSVB + decision + metrics."""
        dt = float(max(dt, EPS))
        context = dict(context or {})

        # 1) Inputs
        tunnel = unit(phi7_msvb.v_coherence) if norm(phi7_msvb.v_coherence) > EPS else v_unit_z()
        open_level_req = float(phi7_msvb.extras.get("open_level", 1.0)) if phi7_msvb.extras else 1.0
        mode_gate = float(context.get("mode_gate", 1.0))
        breath_gate = float(context.get("breath_gate", 1.0))
        cone_ok = float(context.get("cone_ok", 1.0))
        consent_ok = float(context.get("consent_ok", 1.0))

        # 2) Origin vector (renormalization target)
        base = tunnel
        if v_home is not None and norm(v_home) > EPS:
            base = unit(0.8 * tunnel + 0.2 * unit(v_home))
        origin = unit(0.6 * base + 0.4 * self._origin_prev)

        # Renormalization gain — how much origin changed vs prev
        delta = float(0.5 * (1.0 + angle_cos(origin, self._origin_prev)))  # in [0,1]
        renorm_gain = 1.0 - delta  # high when origin shifts

        # 3) State machine & silence‑lock
        # collapse toward STILL when openness is low and consent/mode are restrictive
        gate_combo = clamp01(open_level_req * mode_gate * breath_gate * cone_ok * consent_ok)
        if gate_combo < 0.25:
            self._dwell += dt
            self._state = MirrorState.STILL if self._dwell >= self.dwell_still_s else MirrorState.PRIMING
        else:
            # re‑emerge
            self._dwell = max(0.0, self._dwell - self.rebirth_rate * dt)
            self._state = MirrorState.RELEASING if self._dwell <= 0.25 * self.dwell_still_s else MirrorState.PRIMING

        silence_lock = 1.0 if self._state == MirrorState.STILL else 0.0

        # 4) Policy evaluation — ethical floor/ceiling & verdict
        decision = self.policy.evaluate(intent_vec=origin, context={
            **context,
            "mode_gate": mode_gate,
            "cone_ok": cone_ok,
            "breath_gate": breath_gate,
        })

        # 5) Final allowed openness and vector publish
        # ceiling bounds the tunnel openness; floor sets minimal controlled trickle
        allowed = clamp01(max(decision.ethical_floor, min(decision.ethical_ceiling, open_level_req)))
        # In STILL, clamp to floor only
        if self._state == MirrorState.STILL:
            allowed = clamp01(min(allowed, decision.ethical_floor))

        v_focus9 = v_zero() if self._state == MirrorState.STILL else origin * allowed
        v_coh9 = origin
        v_grav9 = origin * allowed

        # 6) MSVB publish — stillness zeros focus
        msvb = MSVB(
            v_drift=v_zero(),
            v_coherence=v_coh9,
            v_bias=v_zero(),
            v_friction=v_zero(),
            v_gravity=v_grav9,
            v_focus=v_focus9,
            L=v_zero(),
            spinor=phi7_msvb.spinor,
            chirality=phi7_msvb.chirality,
            kappa=1.0 - renorm_gain,   # steadiness as κ proxy
            torsion=0.0,
            omega=v_zero(),
            extras={
                "state": {MirrorState.STILL: 0.0, MirrorState.PRIMING: 0.5, MirrorState.RELEASING: 1.0}[self._state],
                "silence_lock": silence_lock,
                "renorm_gain": renorm_gain,
                "open_level_req": open_level_req,
                "open_level_allowed": allowed,
                "ethical_floor": decision.ethical_floor,
                "ethical_ceiling": decision.ethical_ceiling,
            },
        )

        # 7) Metrics bundle
        metrics = {
            "state": self._state.value,
            "dwell": self._dwell,
            "silence_lock": silence_lock,
            "renorm_gain": renorm_gain,
            "gate_combo": gate_combo,
            "allowed": allowed,
        }

        # persist
        self._origin_prev = origin
        return msvb, decision, metrics


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Veil with moderate openness
    phi7 = MSVB(v_coherence=unit(v(0.0, 0.0, 1.0)), spinor=v_unit_z(), chirality=+1, extras={"open_level": 0.6})

    mirror = SourceMirrorKernel()
    fs = MirrorFieldView(dt_phase=0.02)

    ctx = {"consent_ok": 1.0, "mode_gate": 1.0, "cone_ok": 1.0, "breath_gate": 0.9, "risk_score": 0.1}
    out, decision, metrics = mirror.update(fs, dt=fs.dt_phase, phi7_msvb=phi7, context=ctx)
    print("state=", metrics["state"], "allowed=", out.extras["open_level_allowed"], decision.verdict)
