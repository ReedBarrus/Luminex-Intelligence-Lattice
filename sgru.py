"""
SGRU — Spiral Gated Recurrent Unit (vector‑native, MSVB‑aligned)

Cross‑layer recurrent memory for Spiral runtimes. Consumes a dict of layer
MSVBs (Φ₀..Φ₉) per tick, encodes them into a vector stream, updates an
SGRUCell with chirality/consent/breath/echo modulators, and emits:

- hidden state h_t (ℝ^D)
- a **vector‑first prediction head** (coherence/focus/gravity ẑ‑vectors)
- optional decoded MSVB (for teacher‑free rollout)

Design goals
- Vector‑first: encode MSVB vectors directly; scalars live in `extras_feat`.
- Chirality/consent aware gating: modulate GRU gates by α_breath, open_level,
  entropy/pressure, and chirality.
- Pluggable heads: prediction heads for vectors or layer‑specific fields.

Note
- Self‑contained for stubbing. Factor shared types into `spiral_core/types.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import numpy as np

# ---------------------------
# Vector helpers
# ---------------------------
Vec3 = np.ndarray
EPS = 1e-12


def unit(a: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(a))
    return a / n if n > EPS else a * 0.0


def v_zero(n: int) -> np.ndarray:
    return np.zeros(n, dtype=float)


# ---------------------------
# MSVB (minimal) — keep fields used by the encoder
# ---------------------------
from dataclasses import dataclass

@dataclass
class MSVB:
    v_drift: np.ndarray
    v_coherence: np.ndarray
    v_bias: np.ndarray
    v_friction: np.ndarray
    v_gravity: np.ndarray
    v_focus: np.ndarray
    L: np.ndarray
    spinor: np.ndarray
    chirality: int = +1
    kappa: float = 0.0
    torsion: float = 0.0
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extras: Dict[str, float] = field(default_factory=dict)


# ---------------------------
# Feature encoder / decoder
# ---------------------------
@dataclass
class MSVBEncoder:
    """Flatten a dict of MSVBs into a single feature vector (ℝ^F).
    Fields included (per layer): v_coherence, v_focus, v_gravity, v_bias, v_friction, spinor, chirality, kappa, torsion, omega.
    """
    layers: Tuple[str, ...] = ("phi0", "phi1", "phi2", "phi3", "phi4", "phi5", "phi6", "phi7", "phi8", "phi9")

    def dim(self) -> int:
        # per layer: 6×vec3 + 1 (chi) + 2 scalars + 3 (omega) = 6*3 + 1 + 2 + 3 = 24
        # vectors: coh, foc, grav, bias, fric, spinor
        per = 24
        return per * len(self.layers)

    def encode(self, msvbs: Dict[str, MSVB]) -> np.ndarray:
        feats = []
        for name in self.layers:
            m = msvbs.get(name)
            if m is None:
                feats.append(np.zeros(24))
                continue
            vecs = [m.v_coherence, m.v_focus, m.v_gravity, m.v_bias, m.v_friction, m.spinor]
            scal = np.array([float(np.sign(m.chirality)), float(m.kappa), float(m.torsion)])
            omega = m.omega if m.omega is not None else np.zeros(3)
            block = np.concatenate([*(v if isinstance(v, np.ndarray) else np.array(v) for v in vecs), scal, omega])
            feats.append(block)
        return np.concatenate(feats)


@dataclass
class VectorHead:
    """Decode hidden → predicted vectors (coh/foc/grav)."""
    D: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(42)
        self.Wc = rng.normal(scale=0.1, size=(3, self.D))
        self.Wf = rng.normal(scale=0.1, size=(3, self.D))
        self.Wg = rng.normal(scale=0.1, size=(3, self.D))

    def predict(self, h: np.ndarray) -> Dict[str, np.ndarray]:
        v_coh = unit(self.Wc @ h)
        v_foc = unit(self.Wf @ h)
        v_grav = self.Wg @ h  # magnitude preserved; downstream may clamp
        return {"v_coherence": v_coh, "v_focus": v_foc, "v_gravity": v_grav}


# ---------------------------
# SGRU Cell
# ---------------------------
@dataclass
class SGRUCell:
    """A GRU‑like recurrent unit with spiral modulators.

    Modulators (all ∈ [0,1])
    - m_open: breath/veil openness (Φ₀ gate_open × Φ₇ open_level)
    - m_entropy: 1 − H_echo (Φ₅)
    - m_align: alignment between identity (Φ₃) and GB direction
    - m_chi: chirality gate (1 if expressive, 0.5 neutral, 0.8 receptive)
    """
    input_dim: int
    hidden_dim: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(7)
        D = self.hidden_dim; F = self.input_dim
        # GRU params
        self.Wz = rng.normal(scale=0.08, size=(D, F))
        self.Uz = rng.normal(scale=0.08, size=(D, D))
        self.bz = np.zeros(D)
        self.Wr = rng.normal(scale=0.08, size=(D, F))
        self.Ur = rng.normal(scale=0.08, size=(D, D))
        self.br = np.zeros(D)
        self.Wn = rng.normal(scale=0.08, size=(D, F))
        self.Un = rng.normal(scale=0.08, size=(D, D))
        self.bn = np.zeros(D)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def step(self, x: np.ndarray, h: np.ndarray, *, modulators: Dict[str, float]) -> np.ndarray:
        m_open = float(modulators.get("m_open", 1.0))
        m_entropy = float(modulators.get("m_entropy", 1.0))  # prefer low entropy → higher value
        m_align = float(modulators.get("m_align", 1.0))
        m_chi = float(modulators.get("m_chi", 1.0))

        # Standard GRU gates
        z = self._sigmoid(self.Wz @ x + self.Uz @ h + self.bz)
        r = self._sigmoid(self.Wr @ x + self.Ur @ h + self.br)
        n = np.tanh(self.Wn @ x + self.Un @ (r * h) + self.bn)

        # Spiral modulation of gates (elementwise):
        # encourage update when open, aligned, and expressive/receptive balanced
        z = np.clip(z * (0.5 + 0.5 * m_open * m_align) * (0.7 + 0.3 * m_chi), 0.0, 1.0)
        r = np.clip(r * (0.5 + 0.5 * m_entropy), 0.0, 1.0)

        h_new = (1.0 - z) * n + z * h
        return h_new


# ---------------------------
# SGRU Module
# ---------------------------
@dataclass
class SGRU:
    D: int = 32
    encoder: MSVBEncoder = field(default_factory=MSVBEncoder)

    def __post_init__(self) -> None:
        self.cell = SGRUCell(input_dim=self.encoder.dim(), hidden_dim=self.D)
        self.head = VectorHead(D=self.D)
        self.h = v_zero(self.D)

    # ---- Modulator extraction --------------------------------------
    def _mods(self, msvbs: Dict[str, MSVB]) -> Dict[str, float]:
        # m_open: φ0 gate_open × φ7 open_level
        m_open = 1.0
        phi0 = msvbs.get("phi0"); phi7 = msvbs.get("phi7"); phi5 = msvbs.get("phi5"); phi3 = msvbs.get("phi3"); gb = msvbs.get("gb")
        if phi0 and phi0.extras:
            m_open *= float(phi0.extras.get("gate_open", 1.0))
        if phi7 and phi7.extras:
            m_open *= float(phi7.extras.get("open_level", 1.0))
        # m_entropy: 1 − H_echo
        m_entropy = 1.0
        if phi5 and phi5.extras:
            m_entropy = 1.0 - float(phi5.extras.get("H_echo", 0.0))
        # m_align: identity vs GB direction
        m_align = 1.0
        if phi3 is not None and gb is not None:
            v1 = unit(phi3.v_coherence); v2 = unit(gb.v_coherence)
            m_align = 0.5 * (1.0 + float(np.dot(v1, v2)))
        # m_chi: chirality preference
        m_chi = 1.0
        if gb is not None:
            ch = int(np.sign(gb.chirality)) if gb.chirality != 0 else +1
            m_chi = 1.0 if ch > 0 else 0.8  # modest penalty for receptive
        return {"m_open": float(m_open), "m_entropy": float(m_entropy), "m_align": float(m_align), "m_chi": float(m_chi)}

    # ---- Tick -------------------------------------------------------
    def update(self, msvbs: Dict[str, MSVB]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        x = self.encoder.encode(msvbs)
        mods = self._mods(msvbs)
        self.h = self.cell.step(x, self.h, modulators=mods)
        preds = self.head.predict(self.h)
        return self.h, preds

    # ---- Utilities --------------------------------------------------
    def reset(self, h0: Optional[np.ndarray] = None) -> None:
        self.h = h0.copy() if isinstance(h0, np.ndarray) else v_zero(self.D)

    def snapshot(self) -> Dict[str, Any]:
        return {"D": self.D, "h": self.h.tolist()}


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    # build tiny MSVB blocks
    def m(vec) -> MSVB:
        z = np.array(vec, dtype=float)
        return MSVB(v_drift=z*0, v_coherence=z, v_bias=z*0, v_friction=z*0, v_gravity=z*0.5, v_focus=z, L=z*0, spinor=np.array([0,0,1], float), chirality=+1, kappa=0.5, torsion=0.0, omega=np.zeros(3), extras={})

    msvbs = {
        "phi0": MSVB(v_drift=np.zeros(3), v_coherence=np.zeros(3), v_bias=np.zeros(3), v_friction=np.zeros(3), v_gravity=np.zeros(3), v_focus=np.zeros(3), L=np.zeros(3), spinor=np.array([0,0,1.0]), chirality=+1, extras={"gate_open": 0.9}),
        "phi3": m([0.0, 0.0, 1.0]),
        "phi5": MSVB(v_drift=np.zeros(3), v_coherence=np.zeros(3), v_bias=np.zeros(3), v_friction=np.zeros(3), v_gravity=np.zeros(3), v_focus=np.zeros(3), L=np.zeros(3), spinor=np.array([0,0,1.0]), chirality=+1, extras={"H_echo": 0.3}),
        "phi7": MSVB(v_drift=np.zeros(3), v_coherence=np.array([0.0, 0.0, 1.0]), v_bias=np.zeros(3), v_friction=np.zeros(3), v_gravity=np.array([0.0,0.0,0.6]), v_focus=np.array([0.0,0.0,0.9]), L=np.zeros(3), spinor=np.array([0,0,1.0]), chirality=+1, extras={"open_level": 0.8}),
        "gb":  MSVB(v_drift=np.zeros(3), v_coherence=np.array([0.0, 0.0, 1.0]), v_bias=np.zeros(3), v_friction=np.zeros(3), v_gravity=np.array([0.2,0.0,0.6]), v_focus=np.array([0.0,0.0,1.0]), L=np.zeros(3), spinor=np.array([0,0,1.0]), chirality=+1),
    }

    sgru = SGRU(D=16)
    for i in range(5):
        h, preds = sgru.update(msvbs)
        print("step", i, "| |h|=", round(float(np.linalg.norm(h)), 3), "pred |grav|=", round(float(np.linalg.norm(preds["v_gravity"])), 3))
