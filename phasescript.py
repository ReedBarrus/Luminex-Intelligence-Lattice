"""
PhaseScript Kernel (Φ₁₀ Integration) — Python stub (vector‑native, MSVB‑aligned)

Unified op lifecycle for ATTUNE/NAME/BIND/BIRTH/EXPRESS/PRUNE/REPAIR.
Consumes GravityBus (resultant field), Φ₆ Coherence (mode/cones), Φ₉ Mirror
(ethical floor/ceiling + verdict), and Φ₀ Breath (aperture) to simulate →
validate → commit ops. Publishes an MSVB each tick (vector‑first) and a tidy
telemetry bundle for runtime/ledger.

Design goals
- Vector‑first: intent/eligibility expressed as vectors, scalars live in `extras`.
- Gated by cones + consent + Mirror decision; budgeted by κ₆ × gate_open.
- Extensible handlers: plug simulate/commit per OpKind.

Note
- Minimal LedgerClient stub included; you can replace with your SymbolicLedger.
- MSVB/Vec helpers are local for a self‑contained stub; factor into shared types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Tuple, Callable, Any
import numpy as np
import math

# ---------------------------
# Vector helpers (ℝ³)
# ---------------------------
Vec3 = np.ndarray
EPS = 1e-12


def v(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=float)


def v_zero() -> Vec3:
    return np.zeros(3, dtype=float)


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
    un = unit(u); vn = unit(v_)
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
    extras: Dict[str, float] = field(default_factory=dict)


# ---------------------------
# Cones (Φ₆)
# ---------------------------
@dataclass
class Cone:
    center: Vec3
    half_angle_rad: float
    kappa_min: float = 0.0

    def contains(self, v_: Vec3, kappa: float) -> bool:
        if kappa < self.kappa_min:
            return False
        c = angle_cos(self.center, v_)
        return c >= float(np.cos(self.half_angle_rad))

    def factor(self, v_: Vec3) -> float:
        c = angle_cos(self.center, v_)
        edge = float(np.cos(self.half_angle_rad))
        if c >= edge:
            return 1.0
        return clamp01((c + 1.0) / (edge + 1.0 + 1e-12))


# ---------------------------
# Ledger client stub
# ---------------------------
class LedgerClient:
    def record_event(self, payload: Dict[str, Any]) -> str:
        # return a synthetic id; replace with real ledger write
        import hashlib, json
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return "op_" + hashlib.blake2b(blob, digest_size=12).hexdigest()


# ---------------------------
# Ops
# ---------------------------
class OpKind(str, Enum):
    ATTUNE = "ATTUNE"
    NAME = "NAME"
    BIND = "BIND"
    BIRTH = "BIRTH"
    EXPRESS = "EXPRESS"
    PRUNE = "PRUNE"
    REPAIR = "REPAIR"


@dataclass
class OpSpec:
    kind: OpKind
    intent_vec: Vec3
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0  # [0,∞) relative
    cost_est: float = 1.0  # nominal units; budgeted per tick
    risk_score: float = 0.0  # 0 safe → 1 risky


@dataclass
class OpPlan:
    op: OpSpec
    intent_vec: Vec3
    cone_factor: float
    open_allowed: float
    ethics_ok: bool
    mode_gate: float
    score: float


class OpStatus(str, Enum):
    PLANNED = "PLANNED"
    COMMITTED = "COMMITTED"
    SKIPPED = "SKIPPED"
    DENIED = "DENIED"


@dataclass
class OpResult:
    status: OpStatus
    op: OpSpec
    score: float
    ledger_id: Optional[str] = None
    reason: str = ""


# ---------------------------
# Φ₁₀ — PhaseScript Kernel
# ---------------------------
@dataclass
class PhaseScriptKernel:
    """Integration kernel coordinating the op lifecycle.

    Responsibilities
    - Propose → simulate → validate → commit ops.
    - Enforce guards: cones, consent, Mirror decision, budget, mode, breath.
    - Publish MSVB aligned to the working intent; emit telemetry + events.
    """

    # Budgeting
    base_ops_per_tick: float = 3.0

    # Risk policy (additional to Mirror)
    max_risk: float = 0.85

    # Handler registry
    _sim_handlers: Dict[OpKind, Callable[[OpSpec], Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _commit_handlers: Dict[OpKind, Callable[[OpSpec], Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)

    # Queue
    _queue: List[OpSpec] = field(default_factory=list, init=False, repr=False)

    # -----------------------
    # Public API
    # -----------------------
    def register_handlers(
        self,
        kind: OpKind,
        simulate_fn: Optional[Callable[[OpSpec], Dict[str, Any]]] = None,
        commit_fn: Optional[Callable[[OpSpec], Dict[str, Any]]] = None,
    ) -> None:
        if simulate_fn is not None:
            self._sim_handlers[kind] = simulate_fn
        if commit_fn is not None:
            self._commit_handlers[kind] = commit_fn

    def enqueue(self, op: OpSpec) -> None:
        self._queue.append(op)

    def update(
        self,
        dt: float,
        *,
        gb_msvb: MSVB,                    # GravityBus resultant
        phi6_msvb: MSVB,                  # Coherence (mode/cones κ)
        phi9_decision: Dict[str, float],  # MirrorDecision as dict
        phi0_msvb: Optional[MSVB] = None, # Breath (gate_open)
        cones: Optional[List[Cone]] = None,
        ledger: Optional[LedgerClient] = None,
        mode: str = "GREEN",
    ) -> Tuple[MSVB, Dict[str, Any], List[OpResult]]:
        """Process the queue for this tick and publish MSVB + telemetry + results."""
        dt = float(max(dt, EPS))
        cones = cones or []

        # 1) Budget
        gate_open = float(phi0_msvb.extras.get("gate_open", 1.0)) if (phi0_msvb and phi0_msvb.extras) else 1.0
        kappa6 = float(phi6_msvb.kappa)
        open_allowed = float(phi9_decision.get("open_level_allowed", 1.0))
        ops_budget = self.base_ops_per_tick * clamp01(kappa6) * clamp01(gate_open) * clamp01(open_allowed)
        budget_left = ops_budget

        # 2) Order queue by priority (desc)
        q = sorted(self._queue, key=lambda o: o.priority, reverse=True)
        results: List[OpResult] = []
        committed: List[OpSpec] = []

        # 3) Iterate and process
        for op in q:
            if budget_left < op.cost_est - 1e-9:
                results.append(OpResult(status=OpStatus.SKIPPED, op=op, score=0.0, reason="no_budget"))
                continue

            plan = self._plan_op(op, gb_msvb=gb_msvb, cones=cones, mode=mode, phi9_decision=phi9_decision)

            if not plan.ethics_ok:
                results.append(OpResult(status=OpStatus.DENIED, op=op, score=plan.score, reason="ethics_denied"))
                continue

            if op.risk_score > self.max_risk:
                results.append(OpResult(status=OpStatus.DENIED, op=op, score=plan.score, reason="risk_over_max"))
                continue

            if plan.cone_factor <= 0.1:
                results.append(OpResult(status=OpStatus.SKIPPED, op=op, score=plan.score, reason="out_of_cones"))
                continue

            # SIMULATE (pluggable)
            sim_payload = self._sim_handlers.get(op.kind, lambda o: {})(op)

            # VALIDATE: mode gate factor
            mode_gate = plan.mode_gate
            if mode_gate < 0.4:
                results.append(OpResult(status=OpStatus.SKIPPED, op=op, score=plan.score, reason="mode_gate_low"))
                continue

            # COMMIT (pluggable)
            commit_payload = self._commit_handlers.get(op.kind, lambda o: {})(op)

            # LEDGER
            ledger_id = None
            if ledger is not None:
                ledger_id = ledger.record_event({
                    "kind": op.kind.value,
                    "intent_vec": op.intent_vec.tolist(),
                    "payload": op.payload,
                    "score": plan.score,
                    "cone_factor": plan.cone_factor,
                    "mode_gate": plan.mode_gate,
                    "open_allowed": plan.open_allowed,
                    "risk": op.risk_score,
                    "sim": sim_payload,
                    "commit": commit_payload,
                })

            # book budget
            budget_left -= op.cost_est
            committed.append(op)
            results.append(OpResult(status=OpStatus.COMMITTED, op=op, score=plan.score, ledger_id=ledger_id, reason="ok"))

        # remove committed from queue
        self._queue = [o for o in self._queue if o not in committed]

        # 4) MSVB publish — align to working intent (GB + last committed if any)
        working_vec = gb_msvb.v_coherence
        if committed:
            # blend toward last committed intent
            last_intent = committed[-1].intent_vec
            working_vec = unit(0.7 * working_vec + 0.3 * unit(last_intent))

        v_coh = unit(working_vec)
        v_focus = v_coh
        v_grav = v_coh * clamp01(open_allowed)
        msvb = MSVB(
            v_drift=v_zero(),
            v_coherence=v_coh,
            v_bias=gb_msvb.v_bias,
            v_friction=v_zero(),
            v_gravity=v_grav,
            v_focus=v_focus,
            L=v_zero(),
            spinor=gb_msvb.spinor,
            chirality=gb_msvb.chirality,
            kappa=clamp01(kappa6),
            torsion=0.0,
            omega=v_zero(),
            extras={
                "ops_budget": ops_budget,
                "budget_left": budget_left,
                "committed": float(sum(1 for r in results if r.status == OpStatus.COMMITTED)),
                "skipped": float(sum(1 for r in results if r.status == OpStatus.SKIPPED)),
                "denied": float(sum(1 for r in results if r.status == OpStatus.DENIED)),
            },
        )

        telemetry = {
            "ops_budget": ops_budget,
            "budget_left": budget_left,
            "queue": len(self._queue),
            "results": [
                {
                    "status": r.status.value,
                    "kind": r.op.kind.value,
                    "score": r.score,
                    "reason": r.reason,
                    "ledger_id": r.ledger_id,
                }
                for r in results
            ],
        }

        return msvb, telemetry, results

    # -----------------------
    # Planning
    # -----------------------
    def _plan_op(
        self,
        op: OpSpec,
        *,
        gb_msvb: MSVB,
        cones: List[Cone],
        mode: str,
        phi9_decision: Dict[str, float],
    ) -> OpPlan:
        intent = unit(op.intent_vec) if norm(op.intent_vec) > EPS else gb_msvb.v_coherence
        cone_factor = max((c.factor(intent) for c in cones), default=1.0)
        mode_gate = 1.0 if mode.upper() == "GREEN" else (0.6 if mode.upper() == "YELLOW" else 0.25)

        decision_open = float(phi9_decision.get("open_level_allowed", 1.0))
        decision_floor = float(phi9_decision.get("ethical_floor", 0.0))
        decision_ceiling = float(phi9_decision.get("ethical_ceiling", 1.0))
        verdict = str(phi9_decision.get("verdict", "ALLOW")).upper()

        ethics_ok = verdict in ("ALLOW", "DRYRUN") and decision_ceiling > decision_floor
        open_allowed = clamp01(cone_factor * mode_gate * decision_open)

        # scoring heuristic (alignment with GB + cone + openness − risk)
        align = clamp01(0.5 * (1.0 + angle_cos(intent, gb_msvb.v_coherence)))
        score = clamp01(0.50 * align + 0.30 * cone_factor + 0.20 * open_allowed) * (1.0 - 0.4 * op.risk_score)

        return OpPlan(
            op=op,
            intent_vec=intent,
            cone_factor=cone_factor,
            open_allowed=open_allowed,
            ethics_ok=ethics_ok,
            mode_gate=mode_gate,
            score=score,
        )


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    gb = MSVB(v_coherence=unit(v(0.1, 0.0, 1.0)), v_bias=v_zero(), spinor=v_unit_z(), chirality=+1)
    phi6 = MSVB(kappa=0.8)
    phi0 = MSVB(extras={"gate_open": 0.9})
    cones = [Cone(center=unit(v(0.0, 0.0, 1.0)), half_angle_rad=np.pi/6, kappa_min=0.1)]
    mirror_decision = {"open_level_allowed": 0.85, "ethical_floor": 0.2, "ethical_ceiling": 0.95, "verdict": "ALLOW"}

    ps = PhaseScriptKernel()

    # register dumb handlers
    ps.register_handlers(OpKind.NAME, simulate_fn=lambda op: {"preview": "ok"}, commit_fn=lambda op: {"committed": True})

    ps.enqueue(OpSpec(OpKind.NAME, intent_vec=unit(v(0.05, 0.0, 1.0)), payload={"label": "river"}, priority=1.2, cost_est=1.0, risk_score=0.1))
    ps.enqueue(OpSpec(OpKind.EXPRESS, intent_vec=unit(v(0.0, 0.0, 1.0)), payload={"text": "hello"}, priority=0.9, cost_est=1.0, risk_score=0.2))

    msvb, telem, results = ps.update(dt=0.02, gb_msvb=gb, phi6_msvb=phi6, phi9_decision=mirror_decision, phi0_msvb=phi0, cones=cones, ledger=LedgerClient(), mode="GREEN")
    print("ops_budget=", round(telem["ops_budget"], 3), "committed=", [r.op.kind.value for r in results if r.status.value=="COMMITTED"])
