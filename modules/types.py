# modules/types.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

Vec3 = np.ndarray


def v(x=0.0, y=0.0, z=0.0) -> Vec3:
    return np.array([float(x), float(y), float(z)], float)


def v_zero() -> Vec3:
    return np.zeros(3, float)


def v_unit_x() -> Vec3:
    return np.array([1.0, 0.0, 0.0], float)


def v_unit_y() -> Vec3:
    return np.array([0.0, 1.0, 0.0], float)


def v_unit_z() -> Vec3:
    return np.array([0.0, 0.0, 1.0], float)


def vec3_to_list(u: Vec3) -> list[float]:
    return [float(u[0]), float(u[1]), float(u[2])]


def list_to_vec3(xs: Iterable[float]) -> Vec3:
    a = list(xs)
    if len(a) != 3:
        return v_zero()
    return v(a[0], a[1], a[2])


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

    # serialize into the global dict (list-based)
    def to_dict(self) -> dict:
        return {
            "v_drift": vec3_to_list(self.v_drift),
            "v_coherence": vec3_to_list(self.v_coherence),
            "v_bias": vec3_to_list(self.v_bias),
            "v_friction": vec3_to_list(self.v_friction),
            "v_gravity": vec3_to_list(self.v_gravity),
            "v_focus": vec3_to_list(self.v_focus),
            "L": vec3_to_list(self.L),
            "spinor": vec3_to_list(self.spinor),
            "chirality": int(self.chirality),
            "kappa": float(self.kappa),
            "torsion": float(self.torsion),
            "omega": vec3_to_list(self.omega),
            "extras": dict(self.extras),
        }

    @staticmethod
    def from_dict(d: dict) -> MSVB:
        return MSVB(
            v_drift=list_to_vec3(d.get("v_drift", [0, 0, 0])),
            v_coherence=list_to_vec3(d.get("v_coherence", [0, 0, 0])),
            v_bias=list_to_vec3(d.get("v_bias", [0, 0, 0])),
            v_friction=list_to_vec3(d.get("v_friction", [0, 0, 0])),
            v_gravity=list_to_vec3(d.get("v_gravity", [0, 0, 0])),
            v_focus=list_to_vec3(d.get("v_focus", [0, 0, 1])),
            L=list_to_vec3(d.get("L", [0, 0, 0])),
            spinor=list_to_vec3(d.get("spinor", [0, 0, 1])),
            chirality=int(d.get("chirality", +1)),
            kappa=float(d.get("kappa", 0.0)),
            torsion=float(d.get("torsion", 0.0)),
            omega=list_to_vec3(d.get("omega", [0, 0, 0])),
            extras=dict(d.get("extras", {})),
        )
