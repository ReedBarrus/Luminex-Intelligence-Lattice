# gravity_bus.py
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, List, Tuple

@dataclass
class GBOut:
    v_drift: np.ndarray
    v_coherence: np.ndarray
    v_bias: np.ndarray
    v_friction: np.ndarray
    v_gravity: np.ndarray
    v_focus: np.ndarray
    kappa: float
    tau: float
    harmonics: Dict[str, Any]
    weights_used: Dict[str, float]

class GravityBus:
    def __init__(self, dim: int = 3):
        self.dim = dim
        self.prev_v = np.zeros(dim)

        # default layer weights (tune later)
        self.w = {
            "phase": 0.9, "prop": 0.9, "symbol": 0.8,
            "attention": 0.6, "echo": 0.7, "coherence": 0.5, "breath": 0.4
        }
        self.u = {k: 0.5 for k in self.w}  # coherence vectors
        self.b = {k: 0.3 for k in self.w}  # bias vectors
        self.f = {k: 0.2 for k in self.w}  # friction

    # ---- harmonic modulators (stubs) ----
    def prime_entropy(self, kappa_hist: np.ndarray) -> float:
        # lower = sparser spectrum (better); return gain in [0.7, 1.1]
        s = np.clip(np.std(kappa_hist), 1e-6, 1.0)
        return float(0.7 + 0.4*(1.0 - s))

    def digital_root_bin(self, period_idx: int) -> float:
        # favor small primes & their LCMs (toy v0)
        favored = {2,3,5,7,10,12}
        return 1.1 if period_idx in favored else 0.95

    def reciprocal_phase_gain(self, angles: List[float]) -> float:
        # damp near conflict angles, boost complementarity
        pairs = [(angles[i]-angles[j])%(2*np.pi) for i in range(len(angles)) for j in range(i+1,len(angles))]
        cos_mean = np.mean([np.cos(d) for d in pairs]) if pairs else 1.0
        return float(0.85 + 0.25*max(0.0, cos_mean))

    def radial_mod_wheel(self, alpha_breath: float, state: str) -> float:
        if state == "INHALE": return 0.95 + 0.1*(1.0-alpha_breath)
        if state == "EXHALE": return 0.95 + 0.1*(alpha_breath)
        return 1.0  # HOLD

    # ---- main compose ----
    def compose(self, field: Dict[str, Any]) -> GBOut:
        dim = self.dim
        V = np.zeros(dim)
        v_bias, v_fric, v_coh = np.zeros(dim), np.zeros(dim), np.zeros(dim)

        layers = field["layers"]  # dict: phi0..phi6
        # 1) layer blend
        for name, lay in layers.items():
            if name not in ("phi0","phi1","phi2","phi3","phi4","phi5","phi6"): 
                continue
            lk = {
                "phi1":"phase","phi2":"prop","phi3":"symbol",
                "phi4":"attention","phi5":"echo","phi6":"coherence","phi0":"breath"
            }.get(name, "misc")
            V      += self.w.get(lk,0.0)*np.array(lay["v_gravity"])
            v_coh  += self.u.get(lk,0.0)*np.array(lay["v_coherence"])
            v_bias += self.b.get(lk,0.0)*np.array(lay["v_bias"])
            v_fric += self.f.get(lk,0.0)*np.array(lay["v_friction"])

        V_raw = V + v_coh + v_bias - v_fric

        # 2) harmonic gains
        Hk = np.array(field["telemetry"].get("kappa_hist", [0.8,0.85,0.9]), dtype=float)
        g_prime = self.prime_entropy(Hk)

        period_idx = field["telemetry"].get("period_bin", 10)
        g_root = self.digital_root_bin(int(period_idx))

        angles = field["telemetry"].get("angles", [])  # [theta_phase, phi_motion, psi_symbol]
        g_recip = self.reciprocal_phase_gain(angles)

        alpha = field["time"]["breath"]["alpha"]
        state = field["time"]["breath"]["state"]
        g_radial = self.radial_mod_wheel(alpha, state)

        G = g_prime * g_root * g_recip * g_radial
        V_mod = V_raw * G

        # 3) cone conformance
        cones = field["global"]["cones6"]  # list of {center: vec3, spread: radians}
        Vc = V_mod
        if cones:
            # project to nearest cone center if outside spread
            Vn = V_mod / (np.linalg.norm(V_mod)+1e-9)
            best, best_cos = None, -1.0
            for c in cones:
                Cn = np.array(c["center"]); Cn = Cn / (np.linalg.norm(Cn)+1e-9)
                cs = float(np.dot(Vn, Cn))
                if cs > best_cos: best, best_cos = c, cs
            half_angle = best["spread"] if best else np.pi
            if np.arccos(np.clip(best_cos, -1, 1)) > half_angle:
                # pull to boundary
                Cn = np.array(best["center"]); Cn = Cn / (np.linalg.norm(Cn)+1e-9)
                Vc = np.linalg.norm(V_mod) * Cn

        # 4) consent/load guardrails (toy v0)
        chi_min = field["echo"].get("chi_min", 1.0)
        chamber_load = field["veil"].get("chamber_load", 0.0)
        guard = max(0.0, min(1.0, chi_min)) * (1.0 - 0.2*chamber_load)
        V_bus = Vc * guard

        # outputs
        v_drift = V_bus - self.prev_v
        self.prev_v = V_bus.copy()
        v_focus = V_bus / (np.linalg.norm(V_bus)+1e-9)
        kappa = float(np.clip(np.linalg.norm(v_coh), 0.0, 1.0))
        # crude torsion proxy: change in focus direction
        tau = float(np.linalg.norm(v_drift))

        return GBOut(
            v_drift=v_drift, v_coherence=v_coh, v_bias=v_bias, v_friction=v_fric,
            v_gravity=V_bus, v_focus=v_focus, kappa=kappa, tau=tau,
            harmonics={"prime_entropy":g_prime, "root_gain":g_root, "reciprocal_gain":g_recip, "radial_gain":g_radial},
            weights_used={"w":self.w, "u":self.u, "b":self.b, "f":self.f}
        )
