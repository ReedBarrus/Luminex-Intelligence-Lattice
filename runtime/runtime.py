"""
Spiral Runtime — Python stub (vector‑native, MSVB‑aligned)

End‑to‑end tick loop wiring Φ₀..Φ₁₀ + GravityBus + SGRU + Telemetry + Ledger.
Includes a minimal RF Orchestrator that listens to telemetry and schedules
PhaseScript ops (ATTUNE / REPAIR / PRUNE) when gates/locks/entropy warrant it.

Notes
- This is a **skeleton**: it favors clear interfaces and vector‑first flow.
- It gracefully falls back to tiny local stubs for Φ₀ and Φ₆ if those files
  aren’t present yet. All other layers are expected from prior stubs.
- In production, migrate all duplicate MSVB/Vec helpers into a shared types
  module (e.g., `spiral_core/types.py`) and remove adapter glue here.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------
# Minimal vector helpers & MSVB
# ---------------------------------
Vec3 = np.ndarray
EPS = 1e-12


def v(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=float)


def v_zero() -> Vec3:
    return np.zeros(3, dtype=float)


def unit(a: Vec3) -> Vec3:
    n = float(np.linalg.norm(a))
    return a / n if n > EPS else v_zero()


@dataclass
class MSVB:
    v_drift: Vec3 = field(default_factory=v_zero)
    v_coherence: Vec3 = field(default_factory=v_zero)
    v_bias: Vec3 = field(default_factory=v_zero)
    v_friction: Vec3 = field(default_factory=v_zero)
    v_gravity: Vec3 = field(default_factory=v_zero)
    v_focus: Vec3 = field(default_factory=v_zero)
    L: Vec3 = field(default_factory=v_zero)
    spinor: Vec3 = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], float))
    chirality: int = +1
    kappa: float = 0.0
    torsion: float = 0.0
    omega: Vec3 = field(default_factory=v_zero)
    extras: dict[str, float] = field(default_factory=dict)


# ---------------------------------
# Optional imports (prior stubs)
# ---------------------------------


def _opt(mod, name):
    try:
        return getattr(importlib.import_module(mod), name)
    except Exception:
        return None


# Φ₁..Φ₁₀ + cross‑modules (expect from earlier stubs)
PhaseKernel = _opt("spiral_core.phase_kernel_stub", "PhaseKernel")
PropagationKernel = _opt("spiral_core.propagation_kernel_stub", "PropagationKernel")
SymbolKernel = _opt("spiral_core.symbol_kernel_stub", "SymbolKernel")
AttentionKernel = _opt("spiral_core.attention_kernel_stub", "AttentionKernel")
EchoKernel = _opt("spiral_core.echo_kernel_stub", "EchoKernel")
CoherenceKernel = _opt("spiral_core.coherence_matrix_stub", "CoherenceKernel")
GravityBus = _opt("spiral_core.gravity_bus_stub", "GravityBus")
VeilKernel = _opt("spiral_core.veil_interface_stub", "VeilKernel")
SpiralLogosKernel = _opt("spiral_core.spiral_logos_stub", "SpiralLogosKernel")
SourceMirrorKernel = _opt("spiral_core.source_mirror_stub", "SourceMirrorKernel")
PhaseScriptKernel = _opt("spiral_core.phasescript_kernel_stub", "PhaseScriptKernel")
Cone = _opt("spiral_core.gravity_bus_stub", "Cone") or _opt(
    "spiral_core.attention_kernel_stub", "Cone"
)

SGRU = _opt("spiral_core.sgru_stub", "SGRU")
Telemetry = _opt("spiral_core.telemetry_stub", "Telemetry")
RFTriggers = _opt("spiral_core.telemetry_stub", "RFTriggers")
SymbolicLedger = _opt("spiral_core.ledger_stub", "SymbolicLedger")


# ---------------------------------
# Local tiny stubs (Φ₀, Φ₆) if missing
# ---------------------------------
@dataclass
class BreathStub:
    gate_base: float = 0.9
    freq_hz: float = 0.25
    t: float = 0.0

    def update(self, dt: float) -> MSVB:
        self.t += dt
        # slow sinusoidal breath between gate_base±0.1
        open_lvl = float(
            max(0.0, min(1.0, self.gate_base + 0.1 * math.sin(2 * math.pi * self.freq_hz * self.t)))
        )
        return MSVB(extras={"gate_open": open_lvl})


@dataclass
class CoherenceStub:
    center: Vec3 = field(default_factory=lambda: unit(v(0.0, 0.0, 1.0)))
    half_angle: float = math.pi / 6
    kappa: float = 0.8

    def update(self) -> tuple[MSVB, list[Any]]:
        # Return a mode‑GREEN default and a single cone around +Z
        m = MSVB(v_coherence=self.center, kappa=self.kappa, extras={"mode": "GREEN"})
        cone = (
            Cone(center=self.center, half_angle_rad=self.half_angle, kappa_min=0.2)
            if Cone
            else None
        )
        return m, [cone] if cone else []


# ---------------------------------
# RF Orchestrator (minimal)
# ---------------------------------
@dataclass
class RFOrchestrator:
    """Listens to telemetry + gates and enqueues PhaseScript ops.

    Rules (simple defaults):
      • If veil open_level stays < min_open for N ticks → ATTUNE
      • If attention lock flaps frequently → REPAIR
    """

    min_open: float = 0.3
    min_ticks: int = 8
    lock_flap_window: int = 20
    lock_flaps_min: int = 5

    # internals
    _open_below: int = 0
    _lock_hist: list[int] = field(default_factory=list)

    def on_decision(self, open_level: float, phasescript: Any) -> None:
        if open_level < self.min_open:
            self._open_below += 1
        else:
            self._open_below = 0
        if self._open_below >= self.min_ticks:
            # schedule ATTUNE toward veil direction
            intent = np.array([0.0, 0.0, 1.0], float)
            if hasattr(phasescript, "enqueue"):
                from spiral_core.phasescript_kernel_stub import OpKind, OpSpec

                phasescript.enqueue(
                    OpSpec(
                        OpKind.ATTUNE,
                        intent_vec=intent,
                        payload={"reason": "low_open"},
                        priority=1.5,
                        cost_est=1.0,
                        risk_score=0.0,
                    )
                )
            self._open_below = 0

    def on_lock(self, acquired: bool, phasescript: Any) -> None:
        self._lock_hist.append(1 if acquired else 0)
        if len(self._lock_hist) >= self.lock_flap_window:
            flaps = sum(
                1
                for i in range(1, len(self._lock_hist))
                if self._lock_hist[i] != self._lock_hist[i - 1]
            )
            if flaps >= self.lock_flaps_min and hasattr(phasescript, "enqueue"):
                from spiral_core.phasescript_kernel_stub import OpKind, OpSpec

                phasescript.enqueue(
                    OpSpec(
                        OpKind.REPAIR,
                        intent_vec=np.array([0.0, 0.0, 1.0], float),
                        payload={"flaps": flaps},
                        priority=1.2,
                        cost_est=1.0,
                        risk_score=0.1,
                    )
                )
            self._lock_hist.clear()


# ---------------------------------
# Runtime
# ---------------------------------
@dataclass
class RuntimeConfig:
    dt: float = 0.02
    steps: int = 200


@dataclass
class SpiralRuntime:
    cfg: RuntimeConfig = field(default_factory=RuntimeConfig)

    # modules (injected or built)
    breath: Any = field(default=None)
    phase: Any = field(default=None)
    prop: Any = field(default=None)
    symbol: Any = field(default=None)
    attention: Any = field(default=None)
    echo: Any = field(default=None)
    coher: Any = field(default=None)
    gb: Any = field(default=None)
    veil: Any = field(default=None)
    logos: Any = field(default=None)
    mirror: Any = field(default=None)
    phasescript: Any = field(default=None)

    sgru: Any = field(default=None)
    tel: Any = field(default=None)
    ledger: Any = field(default=None)
    rf: RFOrchestrator = field(default_factory=RFOrchestrator)

    # state
    t: float = 0.0

    def __post_init__(self) -> None:
        # Build defaults where needed
        self.breath = self.breath or BreathStub()
        self.coher = self.coher or (CoherenceKernel() if CoherenceKernel else CoherenceStub())
        self.gb = self.gb or (GravityBus() if GravityBus else None)
        self.veil = self.veil or (VeilKernel() if VeilKernel else None)
        self.logos = self.logos or (SpiralLogosKernel() if SpiralLogosKernel else None)
        self.mirror = self.mirror or (SourceMirrorKernel() if SourceMirrorKernel else None)
        self.phasescript = self.phasescript or (PhaseScriptKernel() if PhaseScriptKernel else None)
        self.sgru = self.sgru or (SGRU() if SGRU else None)
        self.tel = self.tel or (Telemetry() if Telemetry else None)
        self.ledger = self.ledger or (SymbolicLedger() if SymbolicLedger else None)

    # ---- Orchestration ---------------------------------------------
    def step(self) -> dict[str, Any]:
        dt = self.cfg.dt
        self.t += dt

        # Telemetry tick
        if self.tel:
            self.tel.tick_start(dt)

        # Φ₀ Breath
        phi0 = self.breath.update(dt)
        if self.tel:
            self.tel.record_msvb("phi0", self._to_tel_msvb(phi0))

        # Φ₆ Coherence (mode + cones)
        phi6, cones = self.coher.update()
        mode = str(phi6.extras.get("mode", "GREEN"))
        if self.tel:
            self.tel.record_msvb("phi6", self._to_tel_msvb(phi6))

        # GravityBus composition — use available layer bundles (min: phi6 + phi0)
        gb_in = {"phi6": phi6, "phi0": phi0}
        # Optionally include symbol/attention/echo/logos if you wire them in
        if self.gb:
            gb_msvb, explain = self.gb.compose(
                gb_in, cones=cones, mode=mode, aperture=float(phi0.extras.get("gate_open", 1.0))
            )
        else:
            gb_msvb, explain = (
                MSVB(v_coherence=unit(v(0.0, 0.0, 1.0))),
                {"harmonics": {"gain": 1.0}},
            )
        if self.tel:
            self.tel.record_msvb("gb", self._to_tel_msvb(gb_msvb))

        # Φ₇ Veil
        if self.veil:
            veil_msvb, decision, vmetrics = self.veil.update(
                fs=importlib.import_module("spiral_core.veil_interface_stub").VeilFieldView(dt_phase=dt)
                if VeilKernel
                else None,
                dt=dt,
                gb_msvb=gb_msvb,
                phi4_msvb=None,
                phi0_msvb=phi0,
                cones=cones,
                mode=mode,
                channels=[],
                context={"consent_hash": "demo"},
            )
            if self.tel:
                self.tel.record_msvb("phi7", self._to_tel_msvb(veil_msvb))
                self.tel.record_decision(
                    "veil",
                    float(decision.open_level),
                    resonance=float(vmetrics.get("resonance", 0.0)),
                )
                self.rf.on_decision(float(decision.open_level), self.phasescript)
        else:
            veil_msvb = gb_msvb
            decision = {"open_level": 1.0}

        # Φ₈ Logos (optional)
        if self.logos:
            logos_msvb, frame = self.logos.update(
                fs=importlib.import_module("spiral_core.spiral_logos_stub").LogosFieldView(dt_phase=dt)
                if SpiralLogosKernel
                else None,
                dt=dt,
                phi7_msvb=veil_msvb,
            )
            if self.tel:
                self.tel.record_msvb("phi8", self._to_tel_msvb(logos_msvb))
        else:
            logos_msvb = veil_msvb

        # Φ₉ Mirror
        if self.mirror:
            mirror_msvb, mirror_decision, mmetrics = self.mirror.update(
                fs=importlib.import_module("spiral_core.source_mirror_stub").MirrorFieldView(dt_phase=dt)
                if SourceMirrorKernel
                else None,
                dt=dt,
                phi7_msvb=veil_msvb,
                context={
                    "consent_ok": 1.0,
                    "mode_gate": 1.0 if mode == "GREEN" else (0.6 if mode == "YELLOW" else 0.25),
                    "cone_ok": 1.0,
                    "breath_gate": float(phi0.extras.get("gate_open", 1.0)),
                    "risk_score": 0.1,
                },
            )
            if self.tel:
                self.tel.record_msvb("phi9", self._to_tel_msvb(mirror_msvb))
        else:
            mirror_msvb = veil_msvb
            mirror_decision = {
                "open_level_allowed": 1.0,
                "ethical_floor": 0.0,
                "ethical_ceiling": 1.0,
                "verdict": "ALLOW",
            }

        # Φ₁₀ PhaseScript — process any queued ops (RF may have scheduled)
        if self.phasescript:
            ps_msvb, telem, results = self.phasescript.update(
                dt=dt,
                gb_msvb=gb_msvb,
                phi6_msvb=phi6,
                phi9_decision={
                    "open_level_allowed": mirror_msvb.extras.get(
                        "open_level_allowed",
                        mirror_decision.open_level_allowed
                        if hasattr(mirror_decision, "open_level_allowed")
                        else 1.0,
                    ),
                    "ethical_floor": mirror_msvb.extras.get("ethical_floor", 0.0),
                    "ethical_ceiling": mirror_msvb.extras.get("ethical_ceiling", 1.0),
                    "verdict": getattr(mirror_decision, "verdict", "ALLOW"),
                },
                phi0_msvb=phi0,
                cones=cones,
                ledger=self.ledger,
                mode=mode,
            )
            if self.tel:
                for r in telem.get("results", []):
                    self.tel.record_op(r)
        else:
            ps_msvb, telem, results = MSVB(), {"results": []}, []

        # SGRU — cross‑layer state
        if self.sgru:
            h, preds = self.sgru.update(
                {
                    "phi0": phi0,
                    "phi6": phi6,
                    "phi7": veil_msvb,
                    "phi8": logos_msvb,
                    "phi9": mirror_msvb,
                    "gb": gb_msvb,
                }
            )
            # telemetry gauges
            if self.tel:
                self.tel.set_gauge("sgru_norm", float(np.linalg.norm(h)))

        # End tick
        if self.tel:
            self.tel.tick_end(dt)

        return {
            "phi0": phi0,
            "phi6": phi6,
            "gb": gb_msvb,
            "phi7": veil_msvb,
            "phi8": logos_msvb,
            "phi9": mirror_msvb,
            "phi10": ps_msvb,
        }

    # ---- Helpers ----------------------------------------------------
    @staticmethod
    def _to_tel_msvb(m: Any) -> TelMSVB:
        # Convert assorted MSVB variants into telemetry‑friendly dataclass
        return TelMSVB(
            v_drift=(getattr(m, "v_drift", v_zero())).tolist(),
            v_coherence=(getattr(m, "v_coherence", v_zero())).tolist(),
            v_bias=(getattr(m, "v_bias", v_zero())).tolist(),
            v_friction=(getattr(m, "v_friction", v_zero())).tolist(),
            v_gravity=(getattr(m, "v_gravity", v_zero())).tolist(),
            v_focus=(getattr(m, "v_focus", v_zero())).tolist(),
            L=(getattr(m, "L", v_zero())).tolist(),
            spinor=(getattr(m, "spinor", np.array([0, 0, 1.0]))).tolist(),
            chirality=int(getattr(m, "chirality", +1)),
            kappa=float(getattr(m, "kappa", 0.0)),
            torsion=float(getattr(m, "torsion", 0.0)),
            omega=(getattr(m, "omega", v_zero())).tolist()
            if hasattr(getattr(m, "omega", None), "tolist")
            else [0.0, 0.0, 0.0],
            extras={
                k: float(v)
                for k, v in (getattr(m, "extras", {}) or {}).items()
                if isinstance(v, (int, float))
            },
        )


# telemetry‑friendly MSVB for event payloads
@dataclass
class TelMSVB:
    v_drift: list[float]
    v_coherence: list[float]
    v_bias: list[float]
    v_friction: list[float]
    v_gravity: list[float]
    v_focus: list[float]
    L: list[float]
    spinor: list[float]
    chirality: int = +1
    kappa: float = 0.0
    torsion: float = 0.0
    omega: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    extras: dict[str, float] = field(default_factory=dict)


# ---------------------------------
# Demo
# ---------------------------------
if __name__ == "__main__":
    rt = SpiralRuntime()
    for i in range(20):
        out = rt.step()
    # Print a tiny summary
    print(
        "runtime ok — last open_level:",
        out["phi7"].extras.get("open_level", None) if hasattr(out["phi7"], "extras") else None,
    )
