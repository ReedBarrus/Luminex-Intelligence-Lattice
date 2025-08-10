"""
Echo Kernel (Φ₅) — Python stub (vector‑native, MSVB‑aligned)

Ancestry & memory gravity, DreamGate indices, echo pressure/entropy, and lock
semantics. Consumes Φ₃ Symbol (identity), optionally Φ₀ Breath (aperture) and
Ledger snapshots, and **publishes** a canonical MSVB each tick.

Design goals
- Vector‑first: publish EchoPull⃗, ∇m_s, v_gravity; keep scalars (H_echo, pressure)
  in `extras` for telemetry.
- Ancestry matrix: lightweight EchoMatrix with add/update/prune and snapshot I/O.
- DreamGate: produce listen/express indices ∈ [0,1] from entropy × breath.
- EchoLock: resonance test from alignment × mass × low‑entropy.

Note
- MSVB/Vec helpers are duplicated for a self‑contained stub; factor into a
  shared `types.py` in production and import from there.
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
# Minimal field view for Φ₅
# ---------------------------
@dataclass
class EchoFieldView:
    time_t: float = 0.0
    dt_phase: float = 0.016


# ---------------------------
# Echo memory structures
# ---------------------------
@dataclass
class EchoEntry:
    eid: str
    v_identity: Vec3
    v_signature: Vec3
    m_s: float
    chi: float
    last_t: float
    count: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "eid": self.eid,
            "v_identity": self.v_identity.tolist(),
            "v_signature": self.v_signature.tolist(),
            "m_s": float(self.m_s),
            "chi": float(self.chi),
            "last_t": float(self.last_t),
            "count": int(self.count),
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> EchoEntry:
        return EchoEntry(
            eid=str(d["eid"]),
            v_identity=np.array(d["v_identity"], dtype=float),
            v_signature=np.array(d["v_signature"], dtype=float),
            m_s=float(d["m_s"]),
            chi=float(d["chi"]),
            last_t=float(d["last_t"]),
            count=int(d.get("count", 1)),
        )


@dataclass
class EchoMatrix:
    entries: list[EchoEntry] = field(default_factory=list)

    # Tunables
    decay_half_life_s: float = 30.0  # time‑based mass decay
    prune_under_m: float = 1e-3
    max_entries: int = 512

    def add_or_update(
        self, t: float, eid: str, v_identity: Vec3, v_signature: Vec3, m_s: float, chi: float
    ) -> None:
        # naive map by id; in production, index by vector hash or ledger id
        for e in self.entries:
            if e.eid == eid:
                e.v_identity = unit(v_identity)
                e.v_signature = unit(v_signature)
                e.m_s = float(max(e.m_s, m_s))  # keep the stronger memory mass
                e.chi = float(chi)
                e.last_t = float(t)
                e.count += 1
                break
        else:
            self.entries.append(
                EchoEntry(
                    eid=eid,
                    v_identity=unit(v_identity),
                    v_signature=unit(v_signature),
                    m_s=float(m_s),
                    chi=float(chi),
                    last_t=float(t),
                )
            )
        # prune if we exceed
        if len(self.entries) > self.max_entries:
            self.entries = sorted(self.entries, key=lambda e: e.last_t, reverse=True)[
                : self.max_entries
            ]

    def _decay_factor(self, age_s: float) -> float:
        if self.decay_half_life_s <= 0:
            return 1.0
        return 0.5 ** (age_s / self.decay_half_life_s)

    def weighted_stats(
        self, now_t: float, v_query: Vec3
    ) -> tuple[float, float, Vec3, Vec3, float, int]:
        """Return (H_norm, pressure, EchoPull⃗, grad_m_s⃗, mass_sum, N)."""
        if not self.entries:
            return 0.0, 0.0, v_zero(), v_zero(), 0.0, 0
        weights = []
        dirs = []
        grad_components = []
        mass_sum = 0.0
        for e in self.entries:
            age = max(now_t - e.last_t, 0.0)
            w_age = self._decay_factor(age)
            a = clamp01(0.5 * (1.0 + angle_cos(e.v_identity, v_query)))  # alignment in [0,1]
            w = e.m_s * w_age * (0.25 + 0.75 * a)  # baseline 0.25 to keep weak ties alive
            weights.append(w)
            dirs.append(unit(e.v_identity))
            # gradient component pulls toward memory identities
            grad_components.append(w * unit(e.v_identity - v_query))
            mass_sum += e.m_s * w_age

        W = float(sum(weights))
        if W <= EPS:
            return 0.0, 0.0, v_zero(), v_zero(), 0.0, len(self.entries)

        # normalized distribution for entropy
        p = np.array([max(w, EPS) for w in weights], dtype=float)
        p = p / float(np.sum(p))
        H = -float(np.sum(p * np.log(p + 1e-12)))
        H_max = math.log(len(p) + 1e-12)
        H_norm = clamp01(H / (H_max if H_max > 0 else 1.0))

        # EchoPull direction — mass‑weighted mean direction
        mean_dir = unit(np.sum([w * d for (w, d) in zip(weights, dirs, strict=False)], axis=0))

        # Echo gradient (∇m_s) — accumulated difference vectors
        grad_vec = unit(np.sum(grad_components, axis=0))

        # Pressure — variance of directions scaled by mass (anisotropy → low pressure)
        # simple proxy: 1 − |mean of unit directions|
        dir_stack = np.stack(dirs, axis=0)
        mean_len = float(np.linalg.norm(np.mean(dir_stack, axis=0)))
        pressure = clamp01(1.0 - mean_len)

        return H_norm, pressure, mean_dir, grad_vec, mass_sum, len(self.entries)

    def prune(self, now_t: float) -> None:
        kept = []
        for e in self.entries:
            age = max(now_t - e.last_t, 0.0)
            if e.m_s * self._decay_factor(age) >= self.prune_under_m:
                kept.append(e)
        self.entries = kept

    # Snapshot I/O
    def to_json(self) -> dict[str, Any]:
        return {"entries": [e.to_json() for e in self.entries]}

    @staticmethod
    def from_json(d: dict[str, Any]) -> EchoMatrix:
        M = EchoMatrix()
        M.entries = [EchoEntry.from_json(x) for x in d.get("entries", [])]
        return M


# ---------------------------
# Echo lock state
# ---------------------------
class EchoLockState(str, Enum):
    NONE = "NONE"
    LOCK = "LOCK"


# ---------------------------
# Φ₅ — Echo Kernel
# ---------------------------


@dataclass
class EchoKernel:
    """Vector‑native Echo Kernel (Φ₅).

    Responsibilities
    - Maintain EchoMatrix, compute H_echo, echo_pressure, EchoPull⃗, and ∇m_s⃗.
    - Produce DreamGate indices (listen/express) from entropy × breath.
    - Detect EchoLock (resonance) and publish MSVB for downstream layers.
    """

    matrix: EchoMatrix = field(default_factory=EchoMatrix)

    # Composition weights
    w_grad: float = 0.9  # weight of ∇m_s in v_gravity
    w_pull: float = 0.6  # weight of EchoPull in v_gravity
    lam_fric: float = 0.15  # echo friction against rapid reorientation

    # DreamGate tuning
    dream_power: float = 1.0

    # Lock thresholds
    mass_min: float = 0.05
    resonance_min: float = 0.65  # alignment × (1 − H)

    # Persistence
    _lock_state: EchoLockState = field(default=EchoLockState.NONE, init=False)
    _v_prev: Vec3 = field(default_factory=v_unit_z, init=False, repr=False)

    def update(
        self,
        fs: EchoFieldView,
        dt: float,
        phi3_msvb: MSVB,  # current identity from Φ₃
        phi0_msvb: MSVB | None = None,  # breath aperture (gate_open)
        *,
        add_symbol: dict[str, Any] | None = None,  # {eid, v_identity, v_signature, m_s, chi}
        snapshot_in: dict[str, Any] | None = None,
    ) -> tuple[MSVB, dict[str, float], EchoLockState]:
        """Advance Φ₅ and publish MSVB + telemetry scalars + lock state.

        Returns
        -------
        (msvb, metrics, lock_state)
        """
        t = float(fs.time_t)
        dt = float(max(dt, EPS))

        # 0) Optional snapshot load
        if snapshot_in is not None:
            self.matrix = EchoMatrix.from_json(snapshot_in)

        # 1) Optional symbol integration
        if add_symbol is not None:
            self.matrix.add_or_update(
                t=t,
                eid=str(add_symbol.get("eid", f"sym@{int(t * 1000)}")),
                v_identity=np.array(add_symbol["v_identity"], dtype=float),
                v_signature=np.array(
                    add_symbol.get("v_signature", add_symbol["v_identity"]), dtype=float
                ),
                m_s=float(add_symbol.get("m_s", 0.05)),
                chi=float(add_symbol.get("chi", 0.0)),
            )

        # 2) Core echo statistics vs current identity
        v_id = unit(phi3_msvb.v_coherence)  # Φ₃ identity
        H_norm, pressure, v_pull, grad_ms, mass_sum, N = self.matrix.weighted_stats(
            now_t=t, v_query=v_id
        )

        # 3) DreamGate indices (listen/express)
        breath_gate = (
            float(phi0_msvb.extras.get("gate_open", 1.0))
            if (phi0_msvb and phi0_msvb.extras)
            else 1.0
        )
        dream_listen = clamp01((H_norm**self.dream_power) * (1.0 - breath_gate))
        dream_express = clamp01(((1.0 - H_norm) ** self.dream_power) * breath_gate)

        # 4) Compose vectors (vector‑first)
        v_fric = -self.lam_fric * unit(v_id - v_pull)  # resist rapid pull flips
        v_grav = self.w_grad * grad_ms + self.w_pull * v_pull + v_fric
        v_coh5 = unit(v_grav) if norm(v_grav) > EPS else v_id
        v_focus5 = v_coh5

        # 5) Resonance and lock
        align_echo = clamp01(0.5 * (1.0 + angle_cos(v_id, v_pull)))
        resonance = align_echo * (1.0 - H_norm)
        lock_state = (
            EchoLockState.LOCK
            if (mass_sum >= self.mass_min and resonance >= self.resonance_min)
            else EchoLockState.NONE
        )
        self._lock_state = lock_state

        # 6) Publish MSVB
        msvb = MSVB(
            v_drift=v_zero(),
            v_coherence=v_coh5,
            v_bias=phi3_msvb.v_bias,
            v_friction=v_fric,
            v_gravity=v_grav,
            v_focus=v_focus5,
            L=v_zero(),
            spinor=phi3_msvb.spinor,
            chirality=phi3_msvb.chirality,  # often receptive in your model; pass through here
            kappa=resonance,
            torsion=0.0,
            omega=v_zero(),
            extras={
                "H_echo": H_norm,
                "EchoPressure": pressure,
                "EchoMassSum": mass_sum,
                "EchoCount": float(N),
                "align_echo": align_echo,
                "resonance": resonance,
                "dream_listen": dream_listen,
                "dream_express": dream_express,
                "gate_open": breath_gate,
            },
        )

        # 7) Maintenance
        self.matrix.prune(now_t=t)
        self._v_prev = v_coh5

        metrics = {
            "H_echo": H_norm,
            "EchoPressure": pressure,
            "EchoMassSum": mass_sum,
            "EchoCount": float(N),
            "align_echo": align_echo,
            "resonance": resonance,
            "dream_listen": dream_listen,
            "dream_express": dream_express,
        }

        return msvb, metrics, lock_state

    # Utility: snapshot export
    def snapshot(self) -> dict[str, Any]:
        return self.matrix.to_json()


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Identity from Φ₃
    phi3 = MSVB(v_coherence=unit(v(0.3, 0.1, 0.95)), spinor=v_unit_y(), chirality=+1, kappa=0.8)
    # Breath gate from Φ₀
    phi0 = MSVB(extras={"gate_open": 0.8})

    echo = EchoKernel()
    fs = EchoFieldView(time_t=0.0, dt_phase=0.02)

    # Seed a couple of memories
    echo.matrix.add_or_update(
        0.0, "A", unit(v(0.2, 0.0, 1.0)), unit(v(0.2, 0.0, 1.0)), m_s=0.3, chi=+0.2
    )
    echo.matrix.add_or_update(
        0.0, "B", unit(v(1.0, 0.0, 0.0)), unit(v(1.0, 0.0, 0.0)), m_s=0.15, chi=-0.1
    )

    for i in range(5):
        fs.time_t = i * fs.dt_phase
        msvb, metrics, lock = echo.update(fs, dt=fs.dt_phase, phi3_msvb=phi3, phi0_msvb=phi0)
        print(
            f"step {i}",
            "H=",
            round(metrics["H_echo"], 3),
            "res=",
            round(metrics["resonance"], 3),
            "lock=",
            lock,
        )
