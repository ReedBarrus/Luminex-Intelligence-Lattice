# telemetry.py
from collections import deque
import numpy as np
from phase import ang_delta, TAU

class PhaseTelemetry:
    def __init__(self, window:int=32, dt:float=1.0):
        self.dt = dt
        self.window = window
        self.theta_hist = deque(maxlen=window+1)
        self.torsion_hist = deque(maxlen=window)

    def push(self, theta: float, torsion_scalar: float):
        self.theta_hist.append(theta)
        self.torsion_hist.append(torsion_scalar)

    def omega(self) -> float | None:
        if len(self.theta_hist) < 2: return None
        a, b = self.theta_hist[-2], self.theta_hist[-1]
        d = (b - a) % TAU
        if d > np.pi: d -= TAU  # shortest signed delta
        return d / self.dt

    def omega_rms(self) -> float | None:
        if len(self.theta_hist) < 3: return None
        vals = []
        for i in range(1, len(self.theta_hist)):
            a, b = self.theta_hist[i-1], self.theta_hist[i]
            d = (b - a) % TAU
            if d > np.pi: d -= TAU
            vals.append(d / self.dt)
        return float(np.sqrt(np.mean(np.square(vals)))) if vals else None

    def torsion_rms(self) -> float | None:
        if not self.torsion_hist: return None
        arr = np.array(self.torsion_hist, dtype=float)
        return float(np.sqrt(np.mean(np.square(arr))))
