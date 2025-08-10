# symbol.py
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SymbolState:
    id: str
    theta: float
    v_identity: np.ndarray        # (3,)
    v_signature: np.ndarray       # (3,)
    m_s: float = 0.0              # symbolic mass
    chi: float = 1.0              # consent integrity [0,1]
    depth_rho: int = 0
    spinor: np.ndarray = field(default_factory=lambda: np.array([0,0,1.0]))
    symbol_lock: bool = False
    attention_lock: bool = False

    # cached derived this tick
    v_coh3: np.ndarray | None = None
    v_grav3: np.ndarray | None = None
    kappa_inst: float = 0.0
    tau_sig: float = 0.0

def blend(a: np.ndarray, b: np.ndarray, w: float=0.5) -> np.ndarray:
    v = (1-w)*a + w*b
    n = np.linalg.norm(v)
    return v if n==0 else v/n

def compute_symbol_vectors(sym: SymbolState, phase, prop,
                           a=0.6, b=0.3, g=0.1):
    # bases from Phase (lifted to 3D already in Prop)
    t_hat = prop.v_focus  # ≈ û
    # map Φ₂ coherence into identity pull
    v_coh3 = a*prop.v_coherence + b*(np.dot(t_hat, prop.v_focus) * t_hat) + g*sym.v_signature
    v_grav3 = v_coh3  # v0 keep simple; add echo/style later

    sym.v_coh3 = v_coh3
    sym.v_grav3 = v_grav3
    sym.kappa_inst = float(np.clip(np.linalg.norm(v_coh3), 0.0, 1.0))
    # rough signature torsion proxy (v0): orthogonality to identity
    sym.tau_sig = float(1.0 - abs(np.dot(unit(sym.v_signature), unit(sym.v_identity))))

def unit(v):
    n = np.linalg.norm(v); 
    return v if n==0 else v/n

def update_identity(sym: SymbolState, phase, prop, w=0.5):
    # default identity heading: blend phase tangent & propagation velocity
    v_phase_t = np.append(phase.basis_tn()[0], 0.0)        # lift t̂
    v_prop_u  = unit(prop.u)
    sym.v_identity = blend(v_phase_t, v_prop_u, w=w)

def eligibility_lock(sym: SymbolState, stab_thresh=0.9, chi_min=0.8):
    ok = (sym.kappa_inst >= stab_thresh) and (sym.chi >= chi_min)
    sym.symbol_lock = bool(ok)
    return sym.symbol_lock
