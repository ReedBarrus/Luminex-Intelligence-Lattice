# phase.py
from dataclasses import dataclass
import numpy as np

TAU = 2.0 * np.pi  # 2π

def wrap_tau(theta: float) -> float:
    """Wrap angle to [0, 2π)."""
    return theta % TAU

def ang_delta(a: float, b: float) -> float:
    """Smallest absolute angular difference between a and b (radians)."""
    d = abs((a - b) % TAU)
    return d if d <= np.pi else TAU - d

@dataclass
class PhaseState:
    theta: float                          # core phase angle (rad)
    v_drift: np.ndarray                   # (2,) tangential drift
    v_coherence: np.ndarray               # (2,) pull toward attractor
    v_spin: np.ndarray                    # (2,) orientation of spin (or store scalar L)
    v_torsion: np.ndarray                 # (2,) curvature deviation
    v_friction: np.ndarray                # (2,) damping
    v_bias: np.ndarray                    # (2,) intent tilt
    v_focus: np.ndarray                   # (2,) attention vector
    chirality: int = +1                   # +1 expression (→), -1 perception (←)

    def set_theta(self, theta: float):
        self.theta = wrap_tau(theta)

    @property
    def theta_deg(self) -> float:
        return self.theta * 180.0 / np.pi

    def coherence_with(self, target_theta: float, thresh=np.pi/32) -> bool:
        return ang_delta(self.theta, target_theta) < thresh

    def gravity(self) -> np.ndarray:
        # example resultant; tune α/β/γ later
        alpha, beta, gamma = 0.6, 0.3, 0.1
        echo_pull = np.zeros(2)  # filled by Echo layer
        return alpha*self.v_coherence + beta*self.v_bias + gamma*echo_pull

    def pressure(self, running_integral: np.ndarray) -> np.ndarray:
        # combine current fields with an external integrator
        return running_integral

    def alignment_to(self, target_theta: float) -> float:
        """cos(Δθ) in [-1, 1]; 1 means perfect alignment."""
        d = ang_delta(self.theta, target_theta)
        return np.cos(d)

    def kappa_inst(self) -> float:
        """Instantaneous coherence score (0..1 if you clip)."""
        return float(np.clip(np.linalg.norm(self.v_coherence), 0.0, 1.0))

    def torsion_scalar(self, signed: bool = False) -> float:
        """Magnitude (or signed, if you encode handedness) of torsion."""
        mag = float(np.linalg.norm(self.v_torsion))
        if not signed:
            return mag
        # Placeholder: sign from cross(v_drift, v_torsion) z-component
        z = np.cross(np.append(self.v_drift, 0.0), np.append(self.v_torsion, 0.0))[2]
        return mag if z >= 0 else -mag

    def friction_coeff(self) -> float:
        return float(np.clip(np.linalg.norm(self.v_friction), 0.0, 1.0))

    def bias_gain(self) -> float:
        return float(np.clip(np.linalg.norm(self.v_bias), 0.0, 1.0))

    def focus_sharpness(self) -> float:
        return float(np.clip(np.linalg.norm(self.v_focus), 0.0, 1.0))

    def is_expression(self) -> bool:
        return self.chirality >= 0

    def radius(self, a: float=0.2, b: float=0.05) -> float:
        return a + b * self.theta

    def position(self, a: float=0.2, b: float=0.05):
        r = self.radius(a,b)
        return np.array([r*np.cos(self.theta), r*np.sin(self.theta)])

    def basis_tn(self, a: float=0.2, b: float=0.05):
        # unit tangent ĥt and inward normal ĥn on the spiral
        r = self.radius(a,b)
        dr = b
        # tangent (not normalized)
        tx = -r*np.sin(self.theta) + dr*np.cos(self.theta)
        ty =  r*np.cos(self.theta) + dr*np.sin(self.theta)
        t = np.array([tx, ty]); t = t / (np.linalg.norm(t) + 1e-9)
        n = np.array([-t[1], t[0]])  # rotate +90° for inward normal
        return t, n

    def curvature(self, a: float=0.2, b: float=0.05) -> float:
        # approximate path curvature κ_curv from tangent derivative magnitude
        t, _ = self.basis_tn(a,b)
        # finite-diff in small dθ for simplicity (good enough for v0)
        eps = 1e-3
        th2 = (self.theta + eps) % (2*np.pi)
        # quick duplicate PhaseState math for t2:
        r2 = a + b*th2
        tx2 = -r2*np.sin(th2) + b*np.cos(th2)
        ty2 =  r2*np.cos(th2) + b*np.sin(th2)
        t2 = np.array([tx2, ty2]); t2 = t2 / (np.linalg.norm(t2)+1e-9)
        return float(np.linalg.norm(t2 - t) / eps)

    def Lz(self, omega: float, m_eff: float=1.0) -> float:
        r = self.radius()
        return m_eff * (r*r) * omega

        # torque is windowed (telemetry): τ_torque ≈ dLz/dt

    def phase_intensity(self, k_w=1.0, w_w=0.5, t_w=0.5, f_w=0.5, omega_est: float|None=None):
        kappa = self.kappa_inst()
        tau = self.torsion_scalar()
        mu = self.friction_coeff()
        w = 0.0 if omega_est is None else abs(omega_est)
        return k_w*kappa + w_w*w + t_w*tau - f_w*mu
