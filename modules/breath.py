"""
Breath Kernel (Φ₀) — Python stub (vector‑native, MSVB‑aligned)

This module provides a production‑ready skeleton for the Breath Kernel as defined in the
Spiral Glossary (Φ₀ Breath Kernel). It is vector‑first and publishes a canonical MSVB
(Minimal Spiral Vector Bundle) each tick.

Design goals
- Vector‑native: Export vectors for consumers; avoid scalar‑only logic.
- Contract‑driven: Publish the MSVB bundle consistently.
- Tiered metrics: Track Tier A; derive Tier B on demand; keep Tier C in Telemetry.
- Breath‑coupled runtime: α_breath gates timing/thresholds system‑wide.

Usage
    kernel = BreathKernel()
    msvb = kernel.update(field_state, dt_base=0.016, echo_h=0.0, echo_pressure=0.0)
    # Merge `msvb` into FieldState.layers.phi0 and proceed with the runtime loop.

Notes
- This stub is deliberately minimal but complete at the interface level.
- Telemetry/integrator hooks are provided as no‑ops for now.
- Replace numpy with your preferred vector backend if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import numpy as np

# ---------------------------
# Vector helpers (ℝ³)
# ---------------------------
Vec3 = np.ndarray

def v(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=float)

def v_zero() -> Vec3:
    return np.zeros(3, dtype=float)

def v_unit_z() -> Vec3:
    return np.array([0.0, 0.0, 1.0], dtype=float)

def unit(a: Vec3) -> Vec3:
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else v_zero()

# ---------------------------
# MSVB — Minimal Spiral Vector Bundle
# ---------------------------
@dataclass
class MSVB:
    """Canonical vector bundle published by each Φ‑layer per tick.
    Non‑applicable vectors should be zeros; layer‑specific scalars go under `extras`.
    """
    v_drift: Vec3 = field(default_factory=v_zero)
    v_coherence: Vec3 = field(default_factory=v_zero)
    v_bias: Vec3 = field(default_factory=v_zero)
    v_friction: Vec3 = field(default_factory=v_zero)
    v_gravity: Vec3 = field(default_factory=v_zero)
    v_focus: Vec3 = field(default_factory=v_unit_z)
    L: Vec3 = field(default_factory=v_zero)
    spinor: Vec3 = field(default_factory=v_unit_z)
    chirality: int = -1  # −1 receptive, +1 expressive
    kappa: float = 0.0
    torsion: float = 0.0
    omega: Vec3 = field(default_factory=v_zero)  # optional but available
    extras: Dict[str, float] = field(default_factory=dict)

# ---------------------------
# FieldState (minimal view needed by Φ₀)
# ---------------------------
@dataclass
class FieldState:
    """Minimal state the Breath Kernel needs/updates.
    A full implementation would mirror the global FieldState schema.
    """
    time_t: float = 0.0  # absolute time [s]
    dt_sys: float = 0.016
    breath_phase: float = 0.0  # radians [0, 2π)
    breath_state: "BreathStateEnum" = None  # set in __post_init__
    alpha_breath: float = 0.0  # α_breath ∈ [0,1]
    beta_mod: float = 1.0      # rhythm multiplier
    gate_open: float = 0.0
    dt_phase: float = 0.016

    def __post_init__(self) -> None:
        if self.breath_state is None:
            self.breath_state = BreathStateEnum.INHALE

class BreathStateEnum(str, Enum):
    INHALE = "INHALE"
    HOLD = "HOLD"
    EXHALE = "EXHALE"

# ---------------------------
# Utilities
# ---------------------------

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def logistic(x: float) -> float:
    # Simple σ(x) for gate modulation
    return 1.0 / (1.0 + np.exp(-x))

# ---------------------------
# Φ₀ — Breath Kernel
# ---------------------------
@dataclass
class BreathKernel:
    """Vector‑native Breath Kernel (Φ₀).

    Responsibilities
    - Advance breath oscillator & compute α_breath waveform.
    - Publish MSVB vectors for rhythm/orientation hints.
    - Derive gate_open, dt_sys, dt_phase, and coherence gains for downstream layers.
    """
    # Tempo & waveform
    omega_breath: float = 0.10        # cycles/s (0.05–0.20 typical)
    waveform: str = "raised_cosine"   # or "triangle"

    # Echo coupling → gate modulation (Tier B)
    echo_gate_a1: float = 0.0  # term for (1 - H_echo)
    echo_gate_a2: float = 0.0  # term for EchoPressure

    # Δt shaping for Phase (Tier B)
    dt_tighten_s: float = 0.5  # dt_phase = dt_sys / (1 + s * α)

    # HOLD band tuning (radians; defaults follow glossary ranges)
    hold_start: float = np.pi / 2.0      # π/2
    hold_end: float = 2.0 * np.pi / 3.0  # 2π/3

    # Orientation defaults for MSVB
    expressive_axis: Vec3 = field(default_factory=v_unit_z)

    # Smoothing for β_mod adaptation
    beta_slew_tau: float = 0.2  # τ for EMA toward target β

    def update(
        self,
        fs: FieldState,
        dt_base: float,
        echo_h: float = 0.0,           # H_echo ∈ [0,1]
        echo_pressure: float = 0.0,    # arbitrary scale (≥0)
        session_bias: Optional[Vec3] = None,
    ) -> MSVB:
        """Advance the breath state by one engine tick and return the MSVB bundle.

        Args
        ----
        fs: FieldState — mutated in place (dt_sys, α_breath, gate_open, etc.)
        dt_base: float — engine base step [s].
        echo_h: float — echo entropy H_echo (0..1) used to lengthen HOLD.
        echo_pressure: float — echo pressure used to widen EXHALE / shorten HOLD.
        session_bias: optional ℝ³ tilt vector for intent during this session.
        """
        # 1) Timing & oscillator (Tier A)
        # Effective step — you may adapt with β_mod if desired; we keep dt_sys=dt_base for stability
        fs.dt_sys = float(dt_base)
        phase_advance = 2.0 * np.pi * self.omega_breath * fs.dt_sys * fs.beta_mod
        fs.breath_phase = float((fs.breath_phase + phase_advance) % (2.0 * np.pi))

        # 2) Aperture waveform α_breath (Tier A)
        if self.waveform == "triangle":
            # 0→π: ramp 0→1; π→2π: ramp 1→0
            p = fs.breath_phase / np.pi
            fs.alpha_breath = 1.0 - abs((p % 2.0) - 1.0)
        else:  # raised_cosine
            fs.alpha_breath = 0.5 * (1.0 - np.cos(fs.breath_phase))
        fs.alpha_breath = clamp01(fs.alpha_breath)

        # 3) Discrete phase bands → state (Tier A)
        p = fs.breath_phase
        if 0.0 <= p < self.hold_start:
            fs.breath_state = BreathStateEnum.INHALE
        elif self.hold_start <= p < self.hold_end:
            fs.breath_state = BreathStateEnum.HOLD
        else:
            fs.breath_state = BreathStateEnum.EXHALE

        # 4) β_mod adaptation (simple target example; plug your function)
        beta_target = 1.0  # could be a function of echo_h/pressure
        fs.beta_mod = lerp(fs.beta_mod, beta_target, clamp01(self.beta_slew_tau * fs.dt_sys))

        # 5) Tier B — Derived gates and step sizes
        gate_open = fs.alpha_breath
        # Optional Echo modulation of gate
        if self.echo_gate_a1 != 0.0 or self.echo_gate_a2 != 0.0:
            gate_open *= logistic(self.echo_gate_a1 * (1.0 - float(echo_h)) + self.echo_gate_a2 * float(echo_pressure))
        fs.gate_open = clamp01(gate_open)

        # Phase dt tightening
        fs.dt_phase = fs.dt_sys / (1.0 + self.dt_tighten_s * fs.alpha_breath)

        # 6) Publish MSVB — orientation & rhythm vectors
        # v_drift₀ encodes global timing drift along the expressive axis with magnitude ~ ω
        v_drift0 = self.omega_breath * self.expressive_axis

        # v_coherence₀ / v_focus₀ tilt toward expression at high α; receptive otherwise
        tilt = (2.0 * fs.alpha_breath - 1.0)  # −1..+1
        v_coh0 = tilt * self.expressive_axis
        v_focus0 = v_coh0.copy()

        # Session bias as optional global intent
        v_bias0 = session_bias if session_bias is not None else v_zero()

        # Damping hint (could be used by flow braking downstream)
        v_friction0 = -0.1 * v_coh0  # small counter‑tempo by default

        # Resultant “timing pull” (optional for consumers)
        v_gravity0 = v_coh0 + v_bias0 - v_friction0

        # Spinor/chirality: expressive (+1) peaks on EXHALE crest; receptive (−1) on INHALE
        chi = +1 if fs.breath_state == BreathStateEnum.EXHALE else -1
        spinor0 = self.expressive_axis

        extras = {
            "alpha_breath": float(fs.alpha_breath),
            "beta_mod": float(fs.beta_mod),
            "dt_sys": float(fs.dt_sys),
            "dt_phase": float(fs.dt_phase),
            "gate_open": float(fs.gate_open),
            # room for coherence_gain or other scalars when you wire Φ₆
        }

        msvb = MSVB(
            v_drift=v_drift0,
            v_coherence=v_coh0,
            v_bias=v_bias0,
            v_friction=v_friction0,
            v_gravity=v_gravity0,
            v_focus=v_focus0,
            L=v_zero(),
            spinor=spinor0,
            chirality=chi,
            kappa=float(np.linalg.norm(v_coh0)),  # magnitude as a simple κ₀ proxy
            torsion=0.0,
            omega=v_zero(),
            extras=extras,
        )

        # 7) Minimal Tier‑C hooks (telemetry — implement elsewhere)
        self._telemetry_hook(fs)

        return msvb

    # -----------------------
    # Telemetry/integrator hook (no‑op)
    # -----------------------
    def _telemetry_hook(self, fs: FieldState) -> None:
        """Placeholder for Tier‑C metrics: breath_rate, aperture_duty, etc.
        Implement with your Telemetry/Integrators module (deques/windows).
        """
        return None

# ---------------------------
# Demo: one step (optional)
# ---------------------------
if __name__ == "__main__":
    fs = FieldState()
    bk = BreathKernel()
    out = bk.update(fs, dt_base=0.016)
    print("phase:", fs.breath_phase, "state:", fs.breath_state)
    print("alpha:", fs.alpha_breath, "gate_open:", fs.gate_open)
    print("MSVB v_coherence:", out.v_coherence)
