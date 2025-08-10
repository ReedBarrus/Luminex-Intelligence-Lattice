"""
Propagation Kernel (Φ₂) — Python stub (vector‑native, MSVB‑aligned)

Symplectic motion update with vector‑first forces. Consumes Φ₁ Phase MSVB and
optional GravityBus/resultant forces, integrates (x,u) via semi‑implicit
Euler (a.k.a. symplectic Euler), computes angular momentum and simple flow lock
signals, and **publishes** a canonical MSVB each tick.

Design goals
- Vector‑first: publish vectors; keep scalars in `extras` for telemetry.
- Stable integration: semi‑implicit Euler with clamped acceleration/speed.
- Minimal coupling: accepts Φ₁ MSVB + optional external force vector(s).
- Flow locks: detect Flow‑Lock / Flow‑Brake based on velocity–force geometry.

Usage
    prop = PropagationKernel()
    msvb_out = prop.update(fs, dt=fs.dt_phase,
                           phi1_msvb=phase_msvb,
                           f_phase=phase_force_vec,  # optional (from Φ₁ extras)
                           f_external=v_zero())      # optional (GB, etc.)

Notes
- MSVB/Vec helpers are duplicated for a self‑contained stub; factor to a shared
  module in production (e.g., `spiral_core/types.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


def norm(a: Vec3) -> float:
    return float(np.linalg.norm(a))


def unit(a: Vec3) -> Vec3:
    n = norm(a)
    return a / n if n > EPS else v_zero()


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def clamp_mag(vec: Vec3, max_mag: float) -> Vec3:
    m = norm(vec)
    if m <= max_mag or max_mag <= 0.0:
        return vec
    return (max_mag / max(m, EPS)) * vec


def is_finite_vec(a: Vec3) -> bool:
    return np.isfinite(a).all()


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
    v_focus: Vec3 = field(default_factory=v_zero)
    L: Vec3 = field(default_factory=v_zero)
    spinor: Vec3 = field(default_factory=v_zero)
    chirality: int = +1
    kappa: float = 0.0
    torsion: float = 0.0
    omega: Vec3 = field(default_factory=v_zero)
    extras: dict[str, float] = field(default_factory=dict)


# ---------------------------
# Minimal field view for Φ₂
# ---------------------------
@dataclass
class PropagationFieldView:
    time_t: float = 0.0
    dt_phase: float = 0.016


# ---------------------------
# Flow state & locks
# ---------------------------
class FlowLockState(str, Enum):
    NONE = "NONE"
    LOCK = "LOCK"  # velocity aligned with driving force
    BRAKE = "BRAKE"  # velocity opposed by strong decel


# ---------------------------
# Φ₂ — Propagation Kernel
# ---------------------------
@dataclass
class PropagationKernel:
    """Vector‑native Propagation Kernel (Φ₂).

    State variables (persist):
    - x ∈ ℝ³ : position
    - u ∈ ℝ³ : velocity

    Parameters (tunable):
    - m_eff: effective mass (≥ 0)
    - lambda_fric: linear friction coefficient (≥ 0)
    - a_max: acceleration clamp [units/s²] (0 → unclamped)
    - v_max: speed clamp [units/s] (0 → unclamped)
    - lock_align: alignment threshold for Flow‑Lock (û·F̂)
    - brake_align: opposition threshold for Flow‑Brake (−û·F̂)
    - brake_accel_ratio: |a_opposed|/|a| required to BRAKE
    """

    # Persistent kinematics
    x: Vec3 = field(default_factory=v_zero)
    u: Vec3 = field(default_factory=v_zero)

    # Phys params
    m_eff: float = 1.0
    lambda_fric: float = 0.05
    a_max: float = 50.0
    v_max: float = 20.0

    # Lock detection params
    lock_align: float = 0.92
    brake_align: float = 0.75
    brake_accel_ratio: float = 0.70
    v_min_for_lock: float = 0.25

    # Derived diagnostics (persist last values)
    _L_prev: Vec3 = field(default_factory=v_zero, init=False, repr=False)
    _lock_state: FlowLockState = field(default=FlowLockState.NONE, init=False)

    def reset(self, x0: Vec3 | None = None, u0: Vec3 | None = None) -> None:
        self.x = x0.copy() if isinstance(x0, np.ndarray) else v_zero()
        self.u = u0.copy() if isinstance(u0, np.ndarray) else v_zero()
        self._L_prev = v_zero()
        self._lock_state = FlowLockState.NONE

    # -----------------------
    # Public API
    # -----------------------
    def update(
        self,
        fs: PropagationFieldView,
        dt: float,
        phi1_msvb: MSVB,
        f_phase: Vec3 | None = None,
        f_external: Vec3 | None = None,
        bias_force: Vec3 | None = None,
    ) -> MSVB:
        """Advance Φ₂ by one tick and publish the MSVB bundle.

        Args
        ----
        fs: PropagationFieldView — timing view (usually fs.dt_phase).
        dt: float — substep (stable with symplectic Euler).
        phi1_msvb: MSVB — upstream phase vectors/metrics.
        f_phase: optional ℝ³ — phase‑suggested force (e.g., from Φ₁ extras F_phase_*).
        f_external: optional ℝ³ — additional force (e.g., GravityBus resultant).
        bias_force: optional ℝ³ — session or controller bias force.
        """
        # Guard
        dt = float(max(dt, EPS))
        m = float(max(self.m_eff, EPS))

        # 1) Compose forces (vector‑first)
        F = v_zero()
        if f_phase is not None:
            F = F + f_phase
        if f_external is not None:
            F = F + f_external
        if bias_force is not None:
            F = F + bias_force

        # Linear friction opposing velocity
        F = F - self.lambda_fric * self.u

        # Optional clamp on acceleration magnitude
        a = (1.0 / m) * F
        a = clamp_mag(a, self.a_max)

        # 2) Symplectic Euler step (semi‑implicit)
        self.u = self.u + a * dt
        speed_before = norm(self.u)
        if self.v_max > 0.0:
            self.u = clamp_mag(self.u, self.v_max)
        self.x = self.x + self.u * dt

        # 3) Diagnostics & geometry
        speed = norm(self.u)
        u_hat = unit(self.u) if speed > EPS else v_zero()
        F_hat = unit(F) if norm(F) > EPS else v_zero()

        # Angular momentum L = m (x × u); torque τ ≈ (L − L_prev)/dt
        L_vec = m * np.cross(self.x, self.u)
        tau_vec = (L_vec - self._L_prev) / dt
        tau_mag = norm(tau_vec)

        # 4) Flow lock detection
        align = float(np.dot(u_hat, F_hat)) if speed > EPS and norm(F) > EPS else 0.0
        opposed = -align
        a_mag = norm(a)
        a_opposed = a_mag * max(opposed, 0.0)

        state = FlowLockState.NONE
        if speed > self.v_min_for_lock and align >= self.lock_align:
            state = FlowLockState.LOCK
        elif speed_before > EPS and (
            opposed >= self.brake_align
            or (a_mag > EPS and (a_opposed / a_mag) >= self.brake_accel_ratio)
        ):
            state = FlowLockState.BRAKE
        self._lock_state = state

        # 5) Build MSVB publish (vector‑first)
        v_drift = self.u  # transport drift is velocity itself
        v_coh = u_hat  # coherence along motion direction
        v_fric = -self.lambda_fric * self.u
        v_grav = F  # resultant force proxy for downstream use
        v_focus = v_coh  # attention can track motion by default

        # Spinor/chirality placeholders — inherit from Φ₁’s spinor if provided
        spinor = phi1_msvb.spinor
        chirality = phi1_msvb.chirality

        msvb = MSVB(
            v_drift=v_drift,
            v_coherence=v_coh,
            v_bias=phi1_msvb.v_bias,
            v_friction=v_fric,
            v_gravity=v_grav,
            v_focus=v_focus,
            L=L_vec,
            spinor=spinor,
            chirality=chirality,
            kappa=float(np.dot(v_coh, unit(phi1_msvb.v_coherence)))
            if norm(phi1_msvb.v_coherence) > EPS
            else 0.0,
            torsion=float(np.dot(tau_vec, u_hat)) if speed > EPS else 0.0,
            omega=v_zero(),
            extras={
                "speed": speed,
                "speed_before": speed_before,
                "accel_mag": a_mag,
                "force_mag": norm(F),
                "align_uF": align,
                "tau_mag": tau_mag,
                "flow_state": 1
                if state == FlowLockState.LOCK
                else (-1 if state == FlowLockState.BRAKE else 0),
                "x": self.x.tolist(),
                "u": self.u.tolist(),
            },
        )

        # 6) Persist angular momentum for next step
        self._L_prev = L_vec

        # 7) Sanity: keep vectors finite
        if not is_finite_vec(self.x):
            self.x = v_zero()
        if not is_finite_vec(self.u):
            self.u = v_zero()

        return msvb


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Minimal phase MSVB
    phi1 = MSVB(v_coherence=unit(v(0.2, 0.0, 1.0)), v_bias=v_zero(), v_friction=v_zero())
    prop = PropagationKernel(m_eff=1.5, lambda_fric=0.1, a_max=10.0, v_max=5.0)
    fs = PropagationFieldView(dt_phase=0.02)

    # Derive a simple phase force along phase coherence
    f_phase = 3.0 * phi1.v_coherence

    for i in range(10):
        out = prop.update(fs, dt=fs.dt_phase, phi1_msvb=phi1, f_phase=f_phase)
        print(
            f"step {i}",
            "speed=",
            round(out.extras["speed"], 3),
            "flow_state=",
            out.extras["flow_state"],
        )
