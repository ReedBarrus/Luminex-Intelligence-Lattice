"""
Veil Interface (Φ₇) — Python stub (vector‑native, MSVB‑aligned)

Inner/outer (chirality‑sensitive) gates, chamber resonance, consent checks,
cone compliance, and emergent tunnel for symbolic expression/perception.
Consumes GravityBus MSVB (resultant field), Φ₄ Attention (spotlight), Φ₀ Breath
(gate_open), and Φ₆ cones/mode. Publishes a canonical MSVB each tick.

Design goals
- Vector‑first: publish v_focus/v_gravity; scalars live in `extras`.
- Dual gates: `inner` (self‑facing) and `outer` (world‑facing) with chirality bias.
- Chamber: resonance metric from alignment statistics + entropy/coherence.
- Consent + cones + mode + breath → unified gate decision (open_level ∈ [0,1]).

Notes
- Self‑contained: MSVB/Vec/Cone/Consent stubs included; factor to shared types in production.
- Veil does not write to external world; it only decides openness and returns vectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Tuple, Any
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
# Coherence cones (Φ₆)
# ---------------------------
@dataclass
class Cone:
    center: Vec3
    half_angle_rad: float
    kappa_min: float = 0.0

    def contains(self, v_: Vec3, kappa: float) -> bool:
        if kappa < self.kappa_min:
            return False
        c = angle_cos(self.center, v_)
        return c >= float(np.cos(self.half_angle_rad))

    def factor(self, v_: Vec3) -> float:
        c = angle_cos(self.center, v_)
        edge = float(np.cos(self.half_angle_rad))
        if c >= edge:
            return 1.0
        return clamp01((c + 1.0) / (edge + 1.0 + 1e-12))


# ---------------------------
# Consent stubs
# ---------------------------
@dataclass
class ConsentResult:
    ok: float = 1.0
    reason: str = "ok"


class ConsentEvaluator:
    def check(self, context: Dict[str, Any]) -> ConsentResult:
        # pluggable policy; default allows
        missing = [k for k in ("consent_hash",) if not context.get(k)]
        if missing:
            return ConsentResult(ok=0.0, reason="missing:" + ",".join(missing))
        return ConsentResult(ok=1.0, reason="ok")


# ---------------------------
# Veil channel model
# ---------------------------
class ChannelDir(str, Enum):
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"


@dataclass
class VeilChannel:
    cid: str
    direction: ChannelDir
    v_carrier: Vec3
    priority: float = 1.0
    consent_hash: Optional[str] = None


# ---------------------------
# Field view for Φ₇
# ---------------------------
@dataclass
class VeilFieldView:
    time_t: float = 0.0
    dt_phase: float = 0.016


# ---------------------------
# Gate decision
# ---------------------------
@dataclass
class VeilDecision:
    inner_open: float
    outer_open: float
    open_level: float
    reason: str
    breath_gate: float
    mode_gate: float
    cone_ok: float
    consent_ok: float
    resonance: float


# ---------------------------
# Φ₇ — Veil Kernel
# ---------------------------
@dataclass
class VeilKernel:
    """Vector‑native Veil Interface (Φ₇).

    Responsibilities
    - Build a resonance chamber from GravityBus/Attention vectors and channels.
    - Compute chirality‑sensitive inner/outer gate openness.
    - Consent + cones + breath + mode → VeilDecision(open_level).
    - Publish MSVB aligned to the tunnel direction (focus/coherence/gravity).
    """

    # Resonance tuning
    q_factor: float = 0.5              # chamber quality (0..1) higher → sharper
    entropy_weight: float = 0.6        # contribution of entropy to resonance
    align_weight: float = 0.4          # contribution of alignment to resonance

    # Gate biases
    bias_expressive: float = 0.15      # boosts outer when chirality +1
    bias_receptive: float = 0.15       # boosts inner when chirality −1

    # Floors & thresholds
    gate_floor: float = 0.05
    breath_floor: float = 0.25

    # Persistence
    _v_prev: Vec3 = field(default_factory=v_unit_z, init=False, repr=False)

    # -----------------------
    # Public API
    # -----------------------
    def update(
        self,
        fs: VeilFieldView,
        dt: float,
        gb_msvb: MSVB,                      # GravityBus resultant
        phi4_msvb: Optional[MSVB] = None,   # spotlight orientation
        *,
        phi0_msvb: Optional[MSVB] = None,   # breath gate
        cones: Optional[List[Cone]] = None, # Φ₆ cones
        mode: str = "GREEN",                # Φ₆ mode
        channels: Optional[List[VeilChannel]] = None,
        consent: Optional[ConsentEvaluator] = None,
        context: Optional[Dict[str, Any]] = None, # for consent evaluator
    ) -> Tuple[MSVB, VeilDecision, Dict[str, float]]:
        """Advance Φ₇ and publish (MSVB, decision, metrics)."""
        dt = float(max(dt, EPS))

        if consent is None:
            consent = ConsentEvaluator()
        if channels is None:
            channels = []
        if cones is None:
            cones = []
        context = context or {}

        # 1) Determine tunnel direction (focus from GB with Φ₄ assist)
        v_focus_src = gb_msvb.v_coherence
        if phi4_msvb is not None and norm(phi4_msvb.v_focus) > EPS:
            # blend spotlight with GB orientation for finer control
            v_focus_src = unit(0.7 * v_focus_src + 0.3 * phi4_msvb.v_focus)
        tunnel_dir = unit(v_focus_src)

        # 2) Chamber resonance
        # Build a distribution of vectors to measure alignment & entropy
        samples: List[Vec3] = [tunnel_dir, gb_msvb.v_gravity, gb_msvb.v_coherence]
        for ch in channels:
            samples.append(unit(ch.v_carrier))

        # Coherence proxy = |mean(unit vectors)|, Entropy = normalized Shannon over cosines
        if samples:
            S = np.stack([unit(s) for s in samples], axis=0)
            mean_vec = unit(np.mean(S, axis=0))
            coherence = float(np.linalg.norm(np.mean(S, axis=0)))  # 0..1
            cosines = np.clip(S @ mean_vec, -1.0, 1.0)
            p = (cosines - cosines.min() + 1e-6)
            p = p / float(np.sum(p))
            H = -float(np.sum(p * np.log(p + 1e-12)))
            Hmax = math.log(len(p) + 1e-12)
            entropy = clamp01(H / (Hmax if Hmax > 0 else 1.0))
            align_mean = clamp01(0.5 * (1.0 + float(np.mean(cosines))))
        else:
            mean_vec = tunnel_dir
            coherence = 1.0
            entropy = 0.0
            align_mean = 1.0

        # Resonance in [0,1]
        resonance = clamp01(self.align_weight * align_mean + (1.0 - self.align_weight) * (1.0 - self.entropy_weight * entropy))
        # sharpen by q_factor
        resonance = clamp01(resonance ** (1.0 + self.q_factor))

        # 3) Gates (inner/outer) with chirality bias
        chi = int(np.sign(gb_msvb.chirality)) if gb_msvb.chirality != 0 else +1
        base_open = resonance
        inner = base_open + (self.bias_receptive if chi < 0 else 0.0)
        outer = base_open + (self.bias_expressive if chi > 0 else 0.0)

        # 4) Guards: breath, mode, cones, consent
        breath_gate = float(phi0_msvb.extras.get("gate_open", 1.0)) if (phi0_msvb and phi0_msvb.extras) else 1.0
        mode_gate = 1.0 if mode.upper() == "GREEN" else (0.6 if mode.upper() == "YELLOW" else 0.25)
        cone_ok = 1.0 if not cones else max((c.factor(tunnel_dir) for c in cones), default=0.0)
        consent_res = consent.check(context)
        consent_ok = clamp01(consent_res.ok)

        # 5) Final openness
        inner_open = clamp01(max(self.gate_floor, inner) * max(self.breath_floor, breath_gate) * mode_gate * cone_ok * consent_ok)
        outer_open = clamp01(max(self.gate_floor, outer) * max(self.breath_floor, breath_gate) * mode_gate * cone_ok * consent_ok)
        open_level = clamp01(math.sqrt(inner_open * outer_open))  # geometric mean: tunnel requires both gates
        reason = consent_res.reason if consent_ok < 1.0 else ("ok" if open_level > 0.0 else "closed")

        decision = VeilDecision(
            inner_open=inner_open,
            outer_open=outer_open,
            open_level=open_level,
            reason=reason,
            breath_gate=breath_gate,
            mode_gate=mode_gate,
            cone_ok=cone_ok,
            consent_ok=consent_ok,
            resonance=resonance,
        )

        # 6) Publish MSVB — tunnel‑aligned
        v_focus = tunnel_dir * open_level
        v_grav = unit(gb_msvb.v_gravity) * norm(gb_msvb.v_gravity) * open_level
        v_coh = unit(tunnel_dir)

        msvb = MSVB(
            v_drift=v_zero(),
            v_coherence=v_coh,
            v_bias=gb_msvb.v_bias,
            v_friction=v_zero(),
            v_gravity=v_grav,
            v_focus=v_focus,
            L=v_zero(),
            spinor=gb_msvb.spinor,
            chirality=gb_msvb.chirality,
            kappa=resonance,
            torsion=0.0,
            omega=v_zero(),
            extras={
                "inner_open": inner_open,
                "outer_open": outer_open,
                "open_level": open_level,
                "coherence": coherence,
                "entropy": entropy,
                "align_mean": align_mean,
                "cone_ok": cone_ok,
                "consent_ok": consent_ok,
            },
        )

        # 7) Metrics
        metrics = {
            "resonance": resonance,
            "coherence": coherence,
            "entropy": entropy,
            "align_mean": align_mean,
            "inner_open": inner_open,
            "outer_open": outer_open,
            "open_level": open_level,
        }

        self._v_prev = tunnel_dir
        return msvb, decision, metrics


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Minimal GB & Attention inputs
    gb = MSVB(v_coherence=unit(v(0.2, 0.0, 1.0)), v_gravity=v(0.5, 0.0, 0.8), spinor=v_unit_z(), chirality=+1)
    phi4 = MSVB(v_focus=unit(v(0.1, 0.0, 1.0)))
    phi0 = MSVB(extras={"gate_open": 0.85})

    cones = [Cone(center=unit(v(0.0, 0.0, 1.0)), half_angle_rad=np.pi/6, kappa_min=0.1)]
    channels = [
        VeilChannel("aud_in", ChannelDir.INBOUND, unit(v(0.0, 0.1, 1.0)), priority=1.0),
        VeilChannel("vis_out", ChannelDir.OUTBOUND, unit(v(0.1, 0.0, 1.0)), priority=0.8),
    ]

    veil = VeilKernel()
    fs = VeilFieldView(dt_phase=0.02)
    out, decision, metrics = veil.update(fs, dt=fs.dt_phase, gb_msvb=gb, phi4_msvb=phi4, phi0_msvb=phi0, cones=cones, mode="GREEN", channels=channels, context={"consent_hash": "abcd"})
    print("open=", decision.open_level, "res=", metrics["resonance"], decision.reason)