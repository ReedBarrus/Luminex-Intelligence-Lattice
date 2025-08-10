# propagation.py (v0 vector-native)
from dataclasses import dataclass
import numpy as np

@dataclass
class PropState:
    x: np.ndarray      # (3,) position
    u: np.ndarray      # (3,) velocity
    m_eff: float = 1.0
    # optional cached vectors for downstream layers (filled each tick)
    v_coherence: np.ndarray | None = None
    v_bias:      np.ndarray | None = None
    v_friction:  np.ndarray | None = None
    v_gravity:   np.ndarray | None = None
    v_focus:     np.ndarray | None = None
    a:           np.ndarray | None = None

def lift2D(v2: np.ndarray) -> np.ndarray:
    # R² -> R³ lift (z=0)
    return np.array([v2[0], v2[1], 0.0], dtype=float)

def compute_vectors(phase, prop: PropState, echo_pull_3d=np.zeros(3),
                    k_coh=0.6, k_bias=0.2, k_fric=0.1, gamma=0.1):
    # Get spiral frames from Phase (2D): t̂, n̂ and lift them
    t2, n2 = phase.basis_tn()   # returns R² unit vectors
    t = lift2D(t2); n = lift2D(n2)

    # Map coherence & bias from Phase
    vcoh2 = phase.v_coherence[:2]          # ensure R²
    vbias2 = phase.v_bias[:2]
    # project coherence along t̂ and n̂
    v_coh = (np.dot(vcoh2, t2) * t) + (np.dot(vcoh2, n2) * n)

    # optional curvature “keep-to-track”
    kappa_curv = phase.curvature()
    v_coh += gamma * kappa_curv * n

    v_bias = lift2D(vbias2)

    v_fric = -k_fric * prop.u
    v_grav = k_coh * v_coh + k_bias * v_bias + echo_pull_3d + v_fric
    v_focus = prop.u / (np.linalg.norm(prop.u) + 1e-9)

    a = v_grav / max(prop.m_eff, 1e-6)

    # cache into state for next layers
    prop.v_coherence = v_coh
    prop.v_bias = v_bias
    prop.v_friction = v_fric
    prop.v_gravity = v_grav
    prop.v_focus = v_focus
    prop.a = a

def step_symplectic(phase, prop: PropState, dt: float, echo_pull_3d=np.zeros(3)):
    compute_vectors(phase, prop, echo_pull_3d)
    prop.u = prop.u + prop.a * dt
    prop.x = prop.x + prop.u * dt
    return prop

# Derived helpers
def L_vec(prop: PropState) -> np.ndarray:
    return prop.m_eff * np.cross(prop.x, prop.u)

def speed(prop: PropState) -> float:
    return float(np.linalg.norm(prop.u))
