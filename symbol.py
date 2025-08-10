"""
Symbol Kernel (Φ₃) — Python stub (vector‑native, MSVB‑aligned)

Identity formation, lock eligibility, signature vectors, and ledger hooks.
Consumes Φ₁ (Phase) and Φ₂ (Propagation) MSVBs and **publishes** a canonical
MSVB each tick for downstream consumers (Φ₄ Attention, Φ₅ Echo, etc.).

Design goals
- Vector‑first: export identity/signature as vectors; scalars go to `extras`.
- Lock‑ready: compute an eligibility score and emit a structured ledger event.
- Stable identity: EMA‑smoothed identity vector and stability metric.
- Minimal coupling: no global singletons; optional Ledger client injected.

Note
- MSVB/Vec helpers are duplicated for a self‑contained stub; factor into a
  shared `types.py` in production and import from there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import hashlib
import json
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


def ema_vec(prev: Vec3, new: Vec3, alpha: float) -> Vec3:
    alpha = clamp01(alpha)
    return unit((1.0 - alpha) * prev + alpha * new)


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
    spinor: Vec3 = field(default_factory=v_unit_z)
    chirality: int = +1  # −1 receptive, +1 expressive
    kappa: float = 0.0
    torsion: float = 0.0
    omega: Vec3 = field(default_factory=v_zero)
    extras: Dict[str, float] = field(default_factory=dict)


# ---------------------------
# Minimal field view for Φ₃
# ---------------------------
@dataclass
class SymbolFieldView:
    time_t: float = 0.0
    dt_phase: float = 0.016


# ---------------------------
# Ledger client stub
# ---------------------------
class LedgerClient:
    """Minimal ledger hook for lock/consent events.
    Replace with your SymbolicLedger implementation.
    """

    def record_event(self, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        chash = hashlib.blake2b(blob, digest_size=16).hexdigest()
        # Return a synthetic entry id; in production, persist and return real id
        return f"entry_{chash}"


# ---------------------------
# Symbol identity structures
# ---------------------------
@dataclass
class SymbolIdentity:
    v_identity: Vec3 = field(default_factory=v_unit_z)
    v_signature: Vec3 = field(default_factory=v_unit_z)
    m_s: float = 0.0   # memory mass proxy
    chi: float = 0.0   # charge/character (−1..+1 typical)
    stability: float = 0.0


@dataclass
class LockEligibility:
    score: float = 0.0
    kappa_base: float = 0.0
    align_tu: float = 0.0
    speed: float = 0.0
    chirality_ok: float = 1.0
    meets: bool = False


# ---------------------------
# Φ₃ — Symbol Kernel
# ---------------------------
@dataclass
class SymbolKernel:
    """Vector‑native Symbol Kernel (Φ₃).

    Responsibilities
    - Form a stable symbolic identity vector from Φ₁/Φ₂ geometry.
    - Compute a signature vector and eligibility for SymbolLock.
    - Emit a structured ledger event when eligible (optional, dry‑run by default).
    - Publish an MSVB bundle with v_coherence/v_focus aligned to identity.
    """

    # Identity blend weights (must sum to 1 for clarity; not enforced)
    w_t: float = 0.55   # weight of Φ₁ tangent / phase coherence
    w_u: float = 0.35   # weight of Φ₂ velocity direction
    w_spin: float = 0.10  # weight of spinor/binormal influence

    # EMA time constant for identity smoothing
    id_ema_tau: float = 0.25  # seconds

    # Mass/character dynamics
    m_s_gain: float = 0.05
    chi_gain: float = 0.05

    # Lock thresholds
    kappa_min: float = 0.20
    align_min: float = 0.60   # û·t̂ mapped to [0,1]
    speed_min: float = 0.20
    stability_min: float = 0.50
    lock_score_threshold: float = 0.70

    # Behavior flags
    name_on_lock: bool = False

    # Internal persistent state
    _id_prev: Vec3 = field(default_factory=v_unit_z, init=False, repr=False)
    _stability: float = field(default=0.0, init=False, repr=False)

    def reset(self, v_id_hint: Optional[Vec3] = None) -> None:
        self._id_prev = unit(v_id_hint) if v_id_hint is not None else v_unit_z()
        self._stability = 0.0

    # -----------------------
    # Public API
    # -----------------------
    def update(
        self,
        fs: SymbolFieldView,
        dt: float,
        phi1_msvb: MSVB,
        phi2_msvb: MSVB,
        echo_msvb: Optional[MSVB] = None,
        ledger: Optional[LedgerClient] = None,
        dry_run: bool = True,
        symbol_id: Optional[str] = None,
    ) -> Tuple[MSVB, SymbolIdentity, LockEligibility, Optional[str]]:
        """Advance Φ₃ by one tick and publish MSVB + identity + eligibility.

        Returns
        -------
        (msvb, identity, eligibility, ledger_entry_id)
        """
        dt = float(max(dt, EPS))

        # 1) Build unit vectors from inputs
        t_hat = unit(phi1_msvb.v_coherence)
        u_hat = unit(phi2_msvb.v_drift)  # Φ₂ sets v_drift=u
        spin = unit(phi1_msvb.spinor)

        # 2) Compose raw signature & identity (vector‑first)
        v_signature = unit(self.w_t * t_hat + self.w_u * u_hat + self.w_spin * spin)
        v_identity_raw = v_signature

        # EMA smoothing toward new identity
        alpha = clamp01(dt / max(self.id_ema_tau, EPS))
        v_identity = ema_vec(self._id_prev, v_identity_raw, alpha)

        # Stability metric: cosine similarity between previous and new identity
        stability = clamp01(0.5 * (1.0 + float(np.dot(self._id_prev, v_identity))))

        # 3) Mass/character updates (simple proxies)
        speed = float(phi2_msvb.extras.get("speed", 0.0)) if phi2_msvb.extras else norm(phi2_msvb.v_drift)
        kappa_base = float(phi1_msvb.kappa)
        m_s = clamp01(self.m_s_gain * (kappa_base + speed))
        chi = clamp01(self.chi_gain * float(np.dot(t_hat, spin))) * (1 if phi1_msvb.chirality >= 0 else -1)

        identity = SymbolIdentity(v_identity=v_identity, v_signature=v_signature, m_s=m_s, chi=chi, stability=stability)

        # 4) Lock eligibility computation
        align_tu = clamp01(0.5 * (1.0 + float(np.dot(t_hat, u_hat))))
        chirality_ok = 1.0 if phi1_msvb.chirality == phi2_msvb.chirality else 0.5

        # Weighted score (simple blend; tune as desired)
        score = 0.40 * align_tu + 0.35 * clamp01(kappa_base) + 0.15 * clamp01(speed) + 0.10 * stability
        meets = (
            kappa_base >= self.kappa_min
            and align_tu >= self.align_min
            and speed >= self.speed_min
            and stability >= self.stability_min
            and chirality_ok >= 0.75
            and score >= self.lock_score_threshold
        )

        elig = LockEligibility(
            score=score,
            kappa_base=kappa_base,
            align_tu=align_tu,
            speed=speed,
            chirality_ok=chirality_ok,
            meets=meets,
        )

        # 5) Optional ledger event (dry‑run by default)
        ledger_entry_id: Optional[str] = None
        if ledger is not None and (meets or not dry_run):
            payload = self._make_lock_payload(
                symbol_id=symbol_id,
                identity=identity,
                elig=elig,
                phi1_msvb=phi1_msvb,
                phi2_msvb=phi2_msvb,
            )
            ledger_entry_id = ledger.record_event(payload)

        # 6) Publish MSVB — identity aligned
        v_coh3 = v_identity
        v_focus3 = v_identity
        v_grav3 = v_identity  # suggestion: pull along identity; GB will re‑compose
        msvb = MSVB(
            v_drift=v_zero(),
            v_coherence=v_coh3,
            v_bias=phi1_msvb.v_bias,  # propagate bias
            v_friction=v_zero(),
            v_gravity=v_grav3,
            v_focus=v_focus3,
            L=v_zero(),
            spinor=phi1_msvb.spinor,
            chirality=phi1_msvb.chirality,
            kappa=float(np.dot(v_coh3, t_hat)) * kappa_base,
            torsion=0.0,
            omega=v_zero(),
            extras={
                "align_tu": align_tu,
                "stability": stability,
                "kappa_base": kappa_base,
                "speed": speed,
                "m_s": m_s,
                "chi": chi,
                "elig_score": score,
                "lock_eligible": 1.0 if meets else 0.0,
            },
        )

        # 7) Persist state
        self._id_prev = v_identity
        self._stability = stability

        return msvb, identity, elig, ledger_entry_id

    # -----------------------
    # Internal helpers
    # -----------------------
    def _make_lock_payload(
        self,
        symbol_id: Optional[str],
        identity: SymbolIdentity,
        elig: LockEligibility,
        phi1_msvb: MSVB,
        phi2_msvb: MSVB,
    ) -> Dict[str, Any]:
        payload = {
            "event": "SYMBOL_LOCK_ELIGIBILITY",
            "symbol_id": symbol_id or "anon",
            "v_identity": identity.v_identity.tolist(),
            "v_signature": identity.v_signature.tolist(),
            "m_s": identity.m_s,
            "chi": identity.chi,
            "stability": identity.stability,
            "kappa_phase": float(phi1_msvb.kappa),
            "align_tu": elig.align_tu,
            "speed": elig.speed,
            "score": elig.score,
            "chirality_ok": elig.chirality_ok,
            "meets": bool(elig.meets),
            # provenance
            "phi1_spinor": phi1_msvb.spinor.tolist(),
            "phi1_chirality": int(phi1_msvb.chirality),
            "phi2_v_drift": phi2_msvb.v_drift.tolist(),
        }
        return payload


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # Minimal upstream MSVBs
    phi1 = MSVB(v_coherence=unit(v(0.3, 0.0, 1.0)), spinor=v_unit_y(), chirality=+1, kappa=0.8)
    phi2 = MSVB(v_drift=unit(v(0.2, 0.0, 1.0)), extras={"speed": 0.6})

    sym = SymbolKernel()
    fs = SymbolFieldView(dt_phase=0.02)
    ledger = LedgerClient()

    for i in range(5):
        msvb, ident, elig, entry = sym.update(fs, dt=fs.dt_phase, phi1_msvb=phi1, phi2_msvb=phi2, ledger=ledger, dry_run=True)
        print(f"step {i}", "score=", round(elig.score, 3), "eligible=", elig.meets, "m_s=", round(ident.m_s, 3))
