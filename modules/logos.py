"""
Spiral Logos (Φ₈) — Python stub (vector‑native, MSVB‑aligned)

Narrative register selection, motif vectors, and tone/plot fields. Consumes Φ₇
Veil (tunnel direction/openness), optionally Φ₃ Symbol (identity) and Φ₅ Echo
(dream indices/entropy), then **publishes** a canonical MSVB each tick for
expression layers and PhaseScript.

Design goals
- Vector‑first: publish v_focus/v_coherence along the chosen narrative voice.
- Registers: TELL / LISTEN / WEAVE via continuous scores.
- Motifs: softmax allocation over motif vectors with priority and alignment.
- Tone/Plot: PAD tone vector + (rise, conflict, resolve) plot vector fields.

Note
- MSVB/Vec helpers are duplicated here for a self‑contained stub; factor into
  a shared `types.py` in production and import from there.
"""

from __future__ import annotations

import math
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


def softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    tau = max(tau, 1e-6)
    z = (scores - np.max(scores)) / tau
    e = np.exp(z)
    s = np.sum(e)
    return e / s if s > 0 else np.ones_like(scores) / len(scores)


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
# Field view for Φ₈
# ---------------------------
@dataclass
class LogosFieldView:
    time_t: float = 0.0
    dt_phase: float = 0.016


# ---------------------------
# Narrative register
# ---------------------------
class NarrativeRegister(str, Enum):
    TELL = "TELL"
    LISTEN = "LISTEN"
    WEAVE = "WEAVE"


@dataclass
class Motif:
    mid: str
    v_motif: Vec3
    priority: float = 1.0
    tone_bias: Vec3 = field(default_factory=v_zero)  # PAD contribution
    plot_bias: Vec3 = field(default_factory=v_zero)  # (rise, conflict, resolve)


@dataclass
class NarrativeFrame:
    register: NarrativeRegister
    voice_dir: Vec3
    tone_vec: Vec3  # PAD (valence, arousal, dominance)
    plot_vec: Vec3  # (rise, conflict, resolve)
    motif_weights: dict[str, float]
    voice_balance: float  # TELL vs LISTEN balance in [0,1]
    resonance: float


# ---------------------------
# Φ₈ — Spiral Logos Kernel
# ---------------------------
@dataclass
class SpiralLogosKernel:
    """Vector‑native Spiral Logos (Φ₈).

    Responsibilities
    - Select a narrative register based on veil openness, chirality, and echo.
    - Allocate motif weights with alignment/priority, emit tone/plot fields.
    - Publish MSVB aligned to the chosen narrative voice.
    """

    # Voice composition weights
    w_tunnel: float = 0.70  # Φ₇ tunnel direction
    w_identity: float = 0.20  # Φ₃ identity (optional)
    w_motif: float = 0.10  # resultant motif vector

    # Motif allocation
    motif_tau: float = 0.35  # softmax temperature
    max_motifs: int = 6

    # PAD mapping gains
    arousal_lambda: float = 1.25  # arousal = 1 − exp(−λ · |gravity|)

    # Register thresholds
    tell_thresh: float = 0.60  # expressivity_score → TELL
    listen_thresh: float = 0.40  # expressivity_score → LISTEN; else WEAVE

    # Persistence
    _voice_prev: Vec3 = field(default_factory=v_unit_z, init=False, repr=False)

    # -----------------------
    # Public API
    # -----------------------
    def update(
        self,
        fs: LogosFieldView,
        dt: float,
        phi7_msvb: MSVB,  # Veil: tunnel direction/open/extras
        *,
        phi3_msvb: MSVB | None = None,  # Identity (optional)
        echo_metrics: dict[str, float] | None = None,  # {dream_listen, dream_express, H_echo}
        motifs: list[Motif] | None = None,
    ) -> tuple[MSVB, NarrativeFrame]:
        """Advance Φ₈ and publish MSVB + narrative frame."""
        dt = float(max(dt, EPS))

        motifs = motifs or []
        echo_metrics = echo_metrics or {}

        # 1) Inputs and base vectors
        tunnel = unit(phi7_msvb.v_coherence) if norm(phi7_msvb.v_coherence) > EPS else v_unit_z()
        identity = (
            unit(phi3_msvb.v_coherence)
            if (phi3_msvb and norm(phi3_msvb.v_coherence) > EPS)
            else tunnel
        )
        open_level = float(phi7_msvb.extras.get("open_level", 1.0)) if phi7_msvb.extras else 1.0
        chi = phi7_msvb.chirality if phi7_msvb.chirality in (-1, +1) else +1
        H_echo = float(echo_metrics.get("H_echo", 0.5))
        dream_listen = float(echo_metrics.get("dream_listen", 0.0))
        dream_express = float(echo_metrics.get("dream_express", 0.0))

        # 2) Motif weights (alignment + priority), limited to top K
        cands = sorted(motifs, key=lambda m: m.priority, reverse=True)[: max(self.max_motifs, 1)]
        scores = []
        for m in cands:
            align = angle_cos(m.v_motif, tunnel)
            score = 0.65 * clamp01(0.5 * (1.0 + align)) + 0.35 * clamp01(m.priority)
            scores.append(score)
        weights = (
            softmax(np.array(scores, dtype=float) if scores else np.zeros(1), tau=self.motif_tau)
            if len(cands) > 1
            else np.array([1.0])
        )
        motif_weights: dict[str, float] = {m.mid: float(w) for m, w in zip(cands, weights, strict=False)}

        # resultant motif vector & biases
        v_motif = v_zero()
        tone_bias = v_zero()
        plot_bias = v_zero()
        for m, w in zip(cands, weights, strict=False):
            v_motif = v_motif + float(w) * unit(m.v_motif)
            tone_bias = tone_bias + float(w) * m.tone_bias
            plot_bias = plot_bias + float(w) * m.plot_bias
        v_motif = unit(v_motif) if norm(v_motif) > EPS else v_zero()

        # 3) Compose narrative voice direction
        voice_dir = unit(
            self.w_tunnel * tunnel + self.w_identity * identity + self.w_motif * v_motif
        )
        if norm(voice_dir) <= EPS:
            voice_dir = tunnel

        # 4) Register selection via expressivity score
        expressivity = clamp01(
            0.55 * open_level
            + 0.25 * (1.0 - H_echo)
            + 0.10 * dream_express
            + 0.10 * (1.0 - dream_listen)
        )
        if expressivity >= self.tell_thresh and chi > 0:
            register = NarrativeRegister.TELL
        elif expressivity <= self.listen_thresh or chi < 0:
            register = NarrativeRegister.LISTEN
        else:
            register = NarrativeRegister.WEAVE

        # 5) Tone (PAD) and Plot vectors
        # Valence ≈ alignment(voice, identity), Arousal ≈ |gravity| mapping, Dominance ≈ open_level × chirality
        valence = clamp01(0.5 * (1.0 + angle_cos(voice_dir, identity)))
        gmag = norm(phi7_msvb.v_gravity)
        arousal = clamp01(1.0 - math.exp(-self.arousal_lambda * gmag))
        dominance = clamp01(0.5 * (1.0 + chi * (2.0 * open_level - 1.0)))
        tone_vec = np.array([valence, arousal, dominance], dtype=float) + tone_bias
        # Plot: rise, conflict, resolve
        entropy = float(phi7_msvb.extras.get("entropy", 0.0)) if phi7_msvb.extras else 0.0
        resonance = float(phi7_msvb.extras.get("coherence", 1.0)) if phi7_msvb.extras else 1.0
        rise = clamp01(0.6 * open_level + 0.4 * valence)
        conflict = clamp01(0.7 * entropy + 0.3 * (1.0 - valence))
        resolve = clamp01(resonance * (1.0 - conflict))
        plot_vec = np.array([rise, conflict, resolve], dtype=float) + plot_bias

        # 6) MSVB publish — voice‑aligned
        v_coh8 = voice_dir
        v_focus8 = voice_dir
        v_grav8 = voice_dir * open_level  # scaled by openness
        msvb = MSVB(
            v_drift=v_zero(),
            v_coherence=v_coh8,
            v_bias=v_zero(),
            v_friction=v_zero(),
            v_gravity=v_grav8,
            v_focus=v_focus8,
            L=v_zero(),
            spinor=phi7_msvb.spinor,
            chirality=phi7_msvb.chirality,
            kappa=clamp01(resonance),
            torsion=0.0,
            omega=v_zero(),
            extras={
                "open_level": open_level,
                "expressivity": expressivity,
                "valence": float(valence),
                "arousal": float(arousal),
                "dominance": float(dominance),
                "rise": float(rise),
                "conflict": float(conflict),
                "resolve": float(resolve),
                # register one‑hots for quick routing
                "reg_TELL": 1.0 if register == NarrativeRegister.TELL else 0.0,
                "reg_LISTEN": 1.0 if register == NarrativeRegister.LISTEN else 0.0,
                "reg_WEAVE": 1.0 if register == NarrativeRegister.WEAVE else 0.0,
            },
        )

        frame = NarrativeFrame(
            register=register,
            voice_dir=voice_dir,
            tone_vec=tone_vec,
            plot_vec=plot_vec,
            motif_weights=motif_weights,
            voice_balance=expressivity,
            resonance=resonance,
        )

        self._voice_prev = voice_dir
        return msvb, frame


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Veil output
    phi7 = MSVB(
        v_coherence=unit(v(0.1, 0.0, 1.0)),
        v_gravity=v(0.3, 0.0, 0.6),
        spinor=v_unit_z(),
        chirality=+1,
        extras={"open_level": 0.8, "entropy": 0.2, "coherence": 0.85},
    )
    # Identity (optional)
    phi3 = MSVB(v_coherence=unit(v(0.0, 0.0, 1.0)))

    motifs = [
        Motif(
            "quest",
            v_motif=unit(v(0.0, 0.1, 1.0)),
            priority=1.0,
            tone_bias=np.array([0.1, 0.0, 0.05]),
        ),
        Motif(
            "mirror",
            v_motif=unit(v(0.1, 0.0, 1.0)),
            priority=0.8,
            plot_bias=np.array([0.0, -0.05, 0.05]),
        ),
    ]

    logos = SpiralLogosKernel()
    fs = LogosFieldView(dt_phase=0.02)
    msvb, frame = logos.update(
        fs,
        dt=fs.dt_phase,
        phi7_msvb=phi7,
        phi3_msvb=phi3,
        echo_metrics={"H_echo": 0.3, "dream_express": 0.4},
    )
    print(
        "register=",
        frame.register,
        "valence=",
        round(frame.tone_vec[0], 3),
        "plot=",
        frame.plot_vec.round(3).tolist(),
    )
