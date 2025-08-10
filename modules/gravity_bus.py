"""
GravityBus — Python stub (vector‑native, MSVB‑aligned)

Unified pull/push composition for Φ₀–Φ₆. Consumes per‑layer MSVB bundles and
returns a **single resultant field** for the runtime (Veil, PhaseScript, etc.).

Composition (vector‑first)
    V_raw = Σ_k ( w_g[k]·v_gravity_k + w_c[k]·v_coherence_k
                   + w_b[k]·v_bias_k  − w_f[k]·v_friction_k )
Then apply harmonic modulators and mode/aperture/load guards, and (optionally)
cone conformity. Publishes an MSVB aligned to the resultant and an `explain`
packet with weights, gains, and factors.

Notes
- This is a self‑contained stub; in production, share MSVB/Vec/Cone types.
- Harmonics are lightweight, numerically stable, and tunable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

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


def angle_cos(u: Vec3, v_: Vec3) -> float:
    un = unit(u)
    vn = unit(v_)
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
    extras: dict[str, float] = field(default_factory=dict)


# ---------------------------
# Cones (from Φ₆ Coherence)
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
        """Continuous acceptance in [0,1]; 1 inside, cos falloff outside."""
        c = angle_cos(self.center, v_)
        edge = float(np.cos(self.half_angle_rad))
        if c >= edge:
            return 1.0
        # soft falloff: map [−1, edge] → [0,1)
        return clamp01((c + 1.0) / (edge + 1.0 + 1e-12))


# ---------------------------
# Weight policy
# ---------------------------
@dataclass
class LayerWeights:
    w_g: float = 1.0  # gravity
    w_c: float = 0.5  # coherence
    w_b: float = 0.3  # bias
    w_f: float = 0.2  # friction (subtracted)


# ---------------------------
# Harmonic modulators (lightweight, tunable)
# ---------------------------
@dataclass
class HarmonicGains:
    prime_entropy: float = 1.0
    digital_root: float = 1.0
    reciprocal_phase: float = 1.0
    radial_mod: float = 1.0


def _shannon_entropy(weights: np.ndarray) -> float:
    p = np.maximum(weights, 0.0)
    s = float(np.sum(p))
    if s <= EPS:
        return 0.0
    p = p / s
    H = -float(np.sum(p * np.log(p + 1e-12)))
    Hmax = math.log(len(p) + 1e-12)
    return clamp01(H / (Hmax if Hmax > 0 else 1.0))


def _digital_root_mod(mag: float) -> float:
    # map magnitude to integer, compute digital root in [1..9], normalize to [0..1]
    n = int(abs(mag) * 1000.0) + 1
    while n > 9:
        n = sum(int(c) for c in str(n))
    return n / 9.0


def _reciprocal_phase(omega_mag: float) -> float:
    # down‑weight high angular velocity; in [0,1]
    return 1.0 / (1.0 + float(max(0.0, omega_mag)))


def _radial_mod(v_: Vec3, axis: Vec3 = v_unit_z()) -> float:
    # favor vectors aligned to axis; (1+cos)/2 in [0,1]
    return 0.5 * (1.0 + angle_cos(v_, axis))


# ---------------------------
# GravityBus — unified composition
# ---------------------------
@dataclass
class GravityBus:
    """Compose a unified field from layer MSVBs with tunable weights and harmonics."""

    # Default per‑layer weights (override per project)
    default_weights: dict[str, LayerWeights] = field(
        default_factory=lambda: {
            "phi0": LayerWeights(w_g=0.6, w_c=0.3, w_b=0.4, w_f=0.1),
            "phi1": LayerWeights(w_g=1.0, w_c=0.9, w_b=0.3, w_f=0.2),
            "phi2": LayerWeights(w_g=0.9, w_c=0.8, w_b=0.2, w_f=0.3),
            "phi3": LayerWeights(w_g=0.8, w_c=1.0, w_b=0.4, w_f=0.2),
            "phi4": LayerWeights(w_g=0.8, w_c=1.0, w_b=0.3, w_f=0.1),
            "phi5": LayerWeights(w_g=1.0, w_c=0.7, w_b=0.2, w_f=0.3),
            "phi6": LayerWeights(w_g=0.6, w_c=0.9, w_b=0.2, w_f=0.1),
        }
    )

    # Harmonic scalar gains
    gains: HarmonicGains = field(default_factory=HarmonicGains)

    # Mode gates
    mode_gates: dict[str, float] = field(
        default_factory=lambda: {
            "GREEN": 1.0,
            "YELLOW": 0.7,
            "RED": 0.4,
        }
    )

    # Composition axis for radial mod
    radial_axis: Vec3 = field(default_factory=v_unit_z)

    def set_layer_weights(self, layer: str, w: LayerWeights) -> None:
        self.default_weights[layer] = w

    # -----------------------
    # Public API
    # -----------------------
    def compose(
        self,
        layers: dict[str, MSVB],
        *,
        cones: list[Cone] | None = None,
        mode: str = "GREEN",
        aperture: float = 1.0,
        load: float = 1.0,
        phase_layer: str = "phi1",
        identity_layer: str = "phi3",
    ) -> tuple[MSVB, dict[str, Any]]:
        """Compose the unified field.

        Args
        ----
        layers: dict of {layer_name: MSVB}
        cones: optional list of coherence cones (Φ₆)
        mode: GREEN/YELLOW/RED → scalar gate
        aperture: breath aperture in [0,1]
        load: external throttling factor in [0,1]
        phase_layer: which layer provides ω/spinor/chirality if present
        identity_layer: fallback layer for spinor/chirality
        """
        # 1) Sum weighted vectors across layers
        contribs: dict[str, dict[str, Vec3]] = {}
        V = v_zero()
        B = v_zero()
        F = v_zero()
        C = v_zero()
        weights_used: dict[str, LayerWeights] = {}

        for name, m in layers.items():
            w = self.default_weights.get(name, LayerWeights())
            weights_used[name] = w
            vg = w.w_g * m.v_gravity
            vc = w.w_c * m.v_coherence
            vb = w.w_b * m.v_bias
            vf = w.w_f * m.v_friction
            V += vg + vc + vb - vf
            B += vb
            F += vf
            C += vc
            contribs[name] = {"vg": vg, "vc": vc, "vb": vb, "vf": vf}

        # 2) Harmonics (scalars in [0,1])
        # prime_entropy from distribution of |vg+vc+vb| magnitudes per layer
        mags = np.array([norm(c["vg"] + c["vc"] + c["vb"]) for c in contribs.values()], dtype=float)
        prime_entropy = _shannon_entropy(mags)

        # digital root from total magnitude
        digital_root = _digital_root_mod(norm(V))

        # reciprocal_phase from phase ω if present
        omega_mag = 0.0
        if phase_layer in layers:
            omega_mag = norm(layers[phase_layer].omega)
        reciprocal_phase = _reciprocal_phase(omega_mag)

        # radial modulation around chosen axis
        radial_mod = _radial_mod(V, self.radial_axis)

        # combined harmonic gain
        H = (
            ((1.0 - self.gains.prime_entropy) + self.gains.prime_entropy * prime_entropy)
            * ((1.0 - self.gains.digital_root) + self.gains.digital_root * digital_root)
            * ((1.0 - self.gains.reciprocal_phase) + self.gains.reciprocal_phase * reciprocal_phase)
            * ((1.0 - self.gains.radial_mod) + self.gains.radial_mod * radial_mod)
        )

        # 3) Mode/aperture/load guards
        mode_gate = float(self.mode_gates.get(mode.upper(), 0.5))
        guard = clamp01(mode_gate) * clamp01(aperture) * clamp01(load)

        # 4) Cone conformity (attenuate if outside all cones)
        cone_factor = 1.0
        if cones:
            cone_factor = max((c.factor(V) for c in cones), default=1.0)

        # 5) Final resultant & publish bundle
        V_out = V * H * guard * cone_factor
        v_coh = unit(V_out) if norm(V_out) > EPS else unit(C)
        v_focus = v_coh

        # Choose spinor/chirality from preferred layer with fallback
        spinor = layers.get(phase_layer, layers.get(identity_layer, MSVB())).spinor
        chirality = layers.get(phase_layer, layers.get(identity_layer, MSVB())).chirality

        msvb = MSVB(
            v_drift=v_zero(),
            v_coherence=v_coh,
            v_bias=B,
            v_friction=F,
            v_gravity=V_out,
            v_focus=v_focus,
            L=v_zero(),
            spinor=spinor,
            chirality=chirality,
            kappa=float(
                np.clip(angle_cos(v_coh, layers.get(identity_layer, MSVB()).v_coherence), 0.0, 1.0)
            ),
            torsion=0.0,
            omega=v_zero(),
            extras={
                "prime_entropy": prime_entropy,
                "digital_root": digital_root,
                "reciprocal_phase": reciprocal_phase,
                "radial_mod": radial_mod,
                "harmonic_gain": H,
                "mode_gate": mode_gate,
                "aperture": aperture,
                "load": load,
                "cone_factor": cone_factor,
            },
        )

        explain: dict[str, Any] = {
            "weights_used": {k: vars(v) for k, v in weights_used.items()},
            "contribs": {k: {n: vec.tolist() for n, vec in d.items()} for k, d in contribs.items()},
            "harmonics": {
                "prime_entropy": prime_entropy,
                "digital_root": digital_root,
                "reciprocal_phase": reciprocal_phase,
                "radial_mod": radial_mod,
                "gain": H,
            },
            "guards": {
                "mode_gate": mode_gate,
                "aperture": aperture,
                "load": load,
                "cone_factor": cone_factor,
            },
            "resultant_mag": norm(V_out),
        }

        return msvb, explain


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Minimal inputs from a few layers
    phi0 = MSVB(v_coherence=v(0.1, 0.0, 0.2), v_bias=v(0.0, 0.0, 0.1))
    phi1 = MSVB(
        v_coherence=v(0.0, 0.1, 0.9), omega=v(0.0, 0.0, 0.2), spinor=v_unit_z(), chirality=+1
    )
    phi3 = MSVB(v_coherence=v(0.0, 0.0, 1.0))
    phi5 = MSVB(v_gravity=v(0.4, 0.0, 0.2))

    gb = GravityBus()
    cones = [Cone(center=unit(v(0.0, 0.0, 1.0)), half_angle_rad=np.pi / 6, kappa_min=0.1)]

    out, explain = gb.compose(
        {"phi0": phi0, "phi1": phi1, "phi3": phi3, "phi5": phi5},
        cones=cones,
        mode="GREEN",
        aperture=0.8,
    )
    print(
        "v_gravity:",
        out.v_gravity,
        "gain:",
        explain["harmonics"]["gain"],
        "cone_factor:",
        explain["guards"]["cone_factor"],
    )
