"""
Phase Kernel (Φ₁) — Python stub (vector‑native, MSVB‑aligned)

This module provides a production‑ready skeleton for Φ₁ Phase. It computes an
orthonormal manifold frame (t̂, n̂, b̂), instantaneous phase metrics, and
vector‑first gravity/pressure suggestions derived from upstream MSVB signals
(e.g., Φ₀ Breath) and local phase geometry. It **publishes** a canonical MSVB
bundle every tick and leaves integration to Φ₂ Propagation (symplectic Euler).

Design goals
- Vector‑first contracts: export vectors; derive scalars for telemetry only.
- Stable frames: robust construction of t̂/n̂/b̂ with graceful fallbacks.
- Minimal coupling: depends on incoming MSVB (breath/echo/etc.), not globals.
- Symplectic‑ready: returns suggested `F_phase` (force proxy) for Φ₂ use.

Usage
    phase = PhaseKernel()
    msvb_out = phase.update(fs, dt=fs.dt_phase, phi0_msvb=breath_msvb,
                            echo_pull=None, v_intent=None)
    # Merge `msvb_out` into FieldState.layers.phi1 and proceed to Φ₂.

Notes
- MSVB/Vec helpers are duplicated here for a self‑contained stub; in production
  factor them into `spiral_core/types.py` and import from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def v_unit_x() -> Vec3:
    return np.array([1.0, 0.0, 0.0], dtype=float)


def v_unit_y() -> Vec3:
    return np.array([0.0, 1.0, 0.0], dtype=float)


def v_unit_z() -> Vec3:
    return np.array([0.0, 0.0, 1.0], dtype=float)


def norm(a: Vec3) -> float:
    return float(np.linalg.norm(a))


def unit(a: Vec3) -> Vec3:
    n = norm(a)
    return a / n if n > EPS else v_zero()


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def project(u: Vec3, onto: Vec3) -> Vec3:
    d = float(np.dot(onto, onto))
    return (np.dot(u, onto) / d) * onto if d > EPS else v_zero()


def reject(u: Vec3, from_vec: Vec3) -> Vec3:
    return u - project(u, from_vec)


def det3(a: Vec3, b: Vec3, c: Vec3) -> float:
    return float(np.linalg.det(np.stack([a, b, c], axis=-1)))


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
    chirality: int = +1  # −1 receptive, +1 expressive
    kappa: float = 0.0
    torsion: float = 0.0
    omega: Vec3 = field(default_factory=v_zero)  # angular velocity proxy
    extras: dict[str, float] = field(default_factory=dict)


# ---------------------------
# Minimal field view for Φ₁
# ---------------------------
@dataclass
class PhaseFieldView:
    """Subset of global FieldState that Φ₁ needs.
    In production, wire to your central FieldState schema.
    """

    time_t: float = 0.0
    dt_phase: float = 0.016  # from Φ₀


# ---------------------------
# Φ₁ — Phase Kernel
# ---------------------------
@dataclass
class PhaseKernel:
    """Vector‑native Phase Kernel (Φ₁).

    Responsibilities
    - Build and maintain an orthonormal frame (t̂, n̂, b̂) from incoming flows.
    - Compute instantaneous phase metrics (alignment, curvature, torsion, ω).
    - Compose phase gravity/pressure suggestions from MSVB inputs.
    - Publish MSVB vectors for downstream consumers.
    """

    # Frame construction defaults
    world_up: Vec3 = field(default_factory=v_unit_z)
    expressive_axis: Vec3 = field(default_factory=v_unit_z)

    # Gravity composition weights (tunable; see README.Forge_Core)
    w_coh: float = 1.0
    w_bias: float = 0.5
    w_fric: float = 0.3
    w_echo: float = 0.5

    # Focus gain toward t̂ (additional to incoming v_focus magnitude)
    focus_gain: float = 0.5

    # Smoothing for frames (EMA on t̂)
    t_ema_tau: float = 0.1  # seconds

    # Internal state (persist between ticks)
    _t_hat: Vec3 = field(default_factory=v_unit_z, init=False, repr=False)
    _n_hat: Vec3 = field(default_factory=v_unit_x, init=False, repr=False)
    _b_hat: Vec3 = field(default_factory=v_unit_y, init=False, repr=False)
    _prev_t_hat: Vec3 = field(default_factory=v_unit_z, init=False, repr=False)

    def reset(self, t_hint: Vec3 | None = None) -> None:
        t0 = unit(t_hint) if t_hint is not None and norm(t_hint) > EPS else self.expressive_axis
        self._t_hat, self._n_hat, self._b_hat = self._build_frame(t0, self.world_up)
        self._prev_t_hat = self._t_hat.copy()

    # -----------------------
    # Public API
    # -----------------------
    def update(
        self,
        fs: PhaseFieldView,
        dt: float,
        phi0_msvb: MSVB,
        echo_pull: Vec3 | None = None,
        v_intent: Vec3 | None = None,
    ) -> MSVB:
        """Advance Φ₁ by one tick and publish the MSVB bundle.

        Args
        ----
        fs: PhaseFieldView — minimal timing view (uses dt_phase typically).
        dt: float — integration sub‑step recommended by Φ₀.
        phi0_msvb: MSVB — upstream breath vectors (focus, bias, friction hints).
        echo_pull: optional ℝ³ — EchoMatrix pull vector (if available).
        v_intent: optional ℝ³ — session/user intent vector.
        """
        # 1) Determine local flow to align t̂ (primary tangent)
        flow = phi0_msvb.v_coherence + phi0_msvb.v_focus
        if v_intent is not None:
            flow = flow + v_intent
        if norm(flow) <= EPS:
            flow = self.expressive_axis

        # Exponential moving average for smooth t̂
        alpha = clamp01(dt / max(self.t_ema_tau, EPS))  # 0..1
        t_target = unit(flow)
        t_hat = unit((1.0 - alpha) * self._t_hat + alpha * t_target)

        # Build orthonormal frame consistent with world_up & previous frame
        t_hat, n_hat, b_hat = self._build_frame(t_hat, self.world_up, prefer_b=self._b_hat)

        # 2) Instantaneous angular velocity ω ≈ (t_prev × t_now)/dt
        omega_vec = v_zero()
        if norm(self._prev_t_hat) > EPS:
            omega_vec = (np.cross(self._prev_t_hat, t_hat)) / max(dt, EPS)
        omega_mag = norm(omega_vec)

        # 3) Phase metrics (vector‑first; scalars for telemetry)
        coh_mag = norm(phi0_msvb.v_coherence)
        alignment = (
            clamp01(float(np.dot(unit(phi0_msvb.v_coherence), t_hat))) if coh_mag > EPS else 0.0
        )
        kappa = alignment * coh_mag  # coherence credit proxy
        torsion = float(np.dot(omega_vec, t_hat))  # twist about tangent

        # A simple curvature proxy from frame change (for pressure suggestion)
        curvature = omega_mag  # rad/s as an instantaneous measure

        # 4) Compose gravity & pressure suggestions
        v_echo = echo_pull if echo_pull is not None else v_zero()
        v_grav = (
            self.w_coh * phi0_msvb.v_coherence
            + self.w_bias * phi0_msvb.v_bias
            - self.w_fric * phi0_msvb.v_friction
            + self.w_echo * v_echo
        )
        # Pressure acts normal to t̂; magnitude follows curvature
        v_pressure = curvature * n_hat

        # 5) Focus vector aims along t̂, scaled by incoming focus + gain*alignment
        focus_mag_in = norm(phi0_msvb.v_focus)
        v_focus = (focus_mag_in + self.focus_gain * alignment) * t_hat

        # 6) Angular momentum proxy (use ω as L for now; refine with inertia later)
        L_vec = omega_vec

        # 7) Spinor/chirality (ensure right‑handedness)
        handed = det3(t_hat, n_hat, b_hat)
        chirality = +1 if handed >= 0.0 else -1
        if chirality < 0:
            # flip b̂ to restore right‑handed frame
            b_hat = -b_hat
            chirality = +1

        # 8) Publish MSVB
        msvb = MSVB(
            v_drift=v_zero(),  # reserved for phase drift if modeled separately
            v_coherence=t_hat,  # direction of phase alignment (unit)
            v_bias=phi0_msvb.v_bias,
            v_friction=phi0_msvb.v_friction,
            v_gravity=v_grav + v_pressure,  # resultant suggestion
            v_focus=v_focus,
            L=L_vec,
            spinor=b_hat,  # carries frame handedness
            chirality=chirality,
            kappa=kappa,
            torsion=torsion,
            omega=omega_vec,
            extras={
                "alignment": alignment,
                "curvature": curvature,
                "omega_mag": omega_mag,
                # Symplectic advice for Φ₂:
                "F_phase_x": float(v_grav[0] + v_pressure[0]),
                "F_phase_y": float(v_grav[1] + v_pressure[1]),
                "F_phase_z": float(v_grav[2] + v_pressure[2]),
            },
        )

        # 9) Persist frame for next step
        self._prev_t_hat = self._t_hat
        self._t_hat, self._n_hat, self._b_hat = t_hat, n_hat, b_hat

        # Return canonical bundle
        return msvb

    # -----------------------
    # Frame utilities
    # -----------------------
    def _build_frame(
        self,
        t_hat: Vec3,
        up: Vec3,
        prefer_b: Vec3 | None = None,
    ) -> tuple[Vec3, Vec3, Vec3]:
        """Construct a robust right‑handed orthonormal frame (t̂, n̂, b̂).
        - t̂: primary tangent (unit)
        - n̂: normal, chosen to be as aligned with `up` as possible while orthogonal to t̂
        - b̂: binormal = t̂ × n̂
        """
        t_hat = unit(t_hat) if norm(t_hat) > EPS else v_unit_z()

        # If up is nearly parallel to t̂, choose an alternate hint
        up_hint = up
        if abs(float(np.dot(unit(up), t_hat))) > 0.98:
            # pick the least‑aligned cardinal axis as hint
            candidates = [v_unit_x(), v_unit_y(), v_unit_z()]
            dots = [abs(float(np.dot(c, t_hat))) for c in candidates]
            up_hint = candidates[int(np.argmin(dots))]

        n_raw = reject(up_hint, from_vec=t_hat)
        n_hat = unit(n_raw) if norm(n_raw) > EPS else unit(reject(v_unit_x(), from_vec=t_hat))

        b_hat = unit(np.cross(t_hat, n_hat))
        # Re‑orthonormalize n̂ for numerical stability
        n_hat = unit(np.cross(b_hat, t_hat))

        # If a preferred b̂ is supplied (for continuity), flip sign if needed
        if prefer_b is not None and norm(prefer_b) > EPS:
            if float(np.dot(b_hat, unit(prefer_b))) < 0.0:
                b_hat = -b_hat
                n_hat = unit(np.cross(b_hat, t_hat))

        return t_hat, n_hat, b_hat


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Minimal breath MSVB feeding phase
    breath = MSVB(
        v_coherence=v(0.2, 0.0, 1.0),
        v_focus=v(0.0, 0.0, 0.3),
        v_bias=v_zero(),
        v_friction=v_zero(),
    )
    phase = PhaseKernel()
    fs = PhaseFieldView(dt_phase=0.02)

    for i in range(5):
        out = phase.update(fs, dt=fs.dt_phase, phi0_msvb=breath)
        print(
            f"step {i}",
            "kappa=",
            out.kappa,
            "torsion=",
            out.torsion,
            "omega_mag=",
            out.extras["omega_mag"],
        )
