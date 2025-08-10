"""
Telemetry — Python stub (vector‑native, MSVB‑aligned)

Lightweight runtime telemetry for Spiral. Provides:
- tick() lifecycle with time, dt, and phase windowing
- gauges/counters/histograms with EMA and sliding windows
- vector & MSVB samplers (stored as JSON‑friendly lists)
- event log for decisions/locks/ops (typed)
- snapshot/export + optional **Ledger adapter** for durable traces
- RF triggers (Reverse Feedback) to schedule integration/repair passes

Notes
- Self‑contained. In production, wire to your logger + SymbolicLedger adapter.
- Histograms are approximate (fixed buckets); tune bucket edges per metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from typing import Dict, Optional, List, Any, Tuple, Callable
import json
import time
import math

# ---------------------------
# Vector helpers & MSVB (storage‑friendly)
# ---------------------------
Vec3 = List[float]

def vec3(x: float, y: float, z: float) -> Vec3:
    return [float(x), float(y), float(z)]

@dataclass
class MSVB:
    v_drift: Vec3
    v_coherence: Vec3
    v_bias: Vec3
    v_friction: Vec3
    v_gravity: Vec3
    v_focus: Vec3
    L: Vec3
    spinor: Vec3
    chirality: int = +1
    kappa: float = 0.0
    torsion: float = 0.0
    omega: Vec3 = field(default_factory=lambda: [0.0, 0.0, 0.0])
    extras: Dict[str, float] = field(default_factory=dict)


# ---------------------------
# Metrics
# ---------------------------
@dataclass
class Gauge:
    name: str
    value: float = 0.0
    ema: float = 0.0
    alpha: float = 0.2

    def update(self, x: float) -> None:
        self.value = float(x)
        self.ema = self.ema + self.alpha * (self.value - self.ema)


@dataclass
class Counter:
    name: str
    value: float = 0.0

    def inc(self, by: float = 1.0) -> None:
        self.value += float(by)


@dataclass
class Histogram:
    name: str
    edges: List[float]
    counts: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0 for _ in range(len(self.edges) + 1)]

    def add(self, x: float) -> None:
        # find bin index
        idx = 0
        while idx < len(self.edges) and x > self.edges[idx]:
            idx += 1
        self.counts[idx] += 1

    def to_json(self) -> Dict[str, Any]:
        return {"edges": self.edges, "counts": self.counts}


# ---------------------------
# Events
# ---------------------------
@dataclass
class Event:
    t: float
    etype: str
    payload: Dict[str, Any]


# Typed helpers

def evt_tick_start(t: float, dt: float) -> Event:
    return Event(t=t, etype="tick_start", payload={"dt": dt})


def evt_tick_end(t: float, dt: float) -> Event:
    return Event(t=t, etype="tick_end", payload={"dt": dt})


def evt_msvb(name: str, m: MSVB, t: float) -> Event:
    p = asdict(m); p["name"] = name
    return Event(t=t, etype="msvb", payload=p)


def evt_decision(name: str, open_level: float, t: float, info: Dict[str, Any]) -> Event:
    d = {"name": name, "open_level": float(open_level), **info}
    return Event(t=t, etype="decision", payload=d)


def evt_lock(name: str, acquired: bool, t: float, metrics: Dict[str, float]) -> Event:
    return Event(t=t, etype="lock", payload={"name": name, "acquired": bool(acquired), **metrics})


def evt_op(result: Dict[str, Any], t: float) -> Event:
    return Event(t=t, etype="op", payload=result)


def evt_rf(trigger: str, path: str, payload_vec: Vec3, result: str, t: float, meta: Optional[Dict[str, Any]] = None) -> Event:
    return Event(t=t, etype="rf", payload={"trigger": trigger, "path": path, "payload_vec": payload_vec, "result": result, "meta": meta or {}})


# ---------------------------
# Telemetry core
# ---------------------------
@dataclass
class Telemetry:
    max_events: int = 2000
    gauges: Dict[str, Gauge] = field(default_factory=dict)
    counters: Dict[str, Counter] = field(default_factory=dict)
    hists: Dict[str, Histogram] = field(default_factory=dict)
    events: deque = field(default_factory=lambda: deque(maxlen=2000))

    # Optional sink (e.g., ledger)
    sink: Optional[Callable[[Event], None]] = None

    # Tick state
    t: float = 0.0

    # ---- Registration ----------------------------------------------
    def gauge(self, name: str, *, alpha: float = 0.2) -> Gauge:
        g = self.gauges.get(name)
        if g is None:
            g = self.gauges[name] = Gauge(name=name, alpha=alpha)
        return g

    def counter(self, name: str) -> Counter:
        c = self.counters.get(name)
        if c is None:
            c = self.counters[name] = Counter(name=name)
        return c

    def hist(self, name: str, edges: List[float]) -> Histogram:
        h = self.hists.get(name)
        if h is None:
            h = self.hists[name] = Histogram(name=name, edges=edges)
        return h

    # ---- Lifecycle --------------------------------------------------
    def tick_start(self, dt: float) -> None:
        self.events.append(evt_tick_start(self.t, dt))
        if self.sink:
            self.sink(self.events[-1])

    def tick_end(self, dt: float) -> None:
        self.events.append(evt_tick_end(self.t, dt))
        if self.sink:
            self.sink(self.events[-1])
        self.t += dt

    # ---- Recorders --------------------------------------------------
    def record_msvb(self, name: str, m: MSVB) -> None:
        e = evt_msvb(name, m, self.t)
        self.events.append(e)
        if self.sink:
            self.sink(e)

    def record_decision(self, name: str, open_level: float, **info: Any) -> None:
        e = evt_decision(name, open_level, self.t, info)
        self.events.append(e)
        if self.sink:
            self.sink(e)

    def record_lock(self, name: str, acquired: bool, **metrics: float) -> None:
        e = evt_lock(name, acquired, self.t, metrics)
        self.events.append(e)
        if self.sink:
            self.sink(e)

    def record_op(self, result: Dict[str, Any]) -> None:
        e = evt_op(result, self.t)
        self.events.append(e)
        if self.sink:
            self.sink(e)

    def record_rf(self, trigger: str, path: str, payload_vec: Vec3, result: str, **meta: Any) -> None:
        e = evt_rf(trigger, path, payload_vec, result, self.t, meta)
        self.events.append(e)
        if self.sink:
            self.sink(e)

    # ---- Gauges & Stats --------------------------------------------
    def set_gauge(self, name: str, value: float) -> None:
        self.gauge(name).update(value)

    def inc(self, name: str, by: float = 1.0) -> None:
        self.counter(name).inc(by)

    def hist_add(self, name: str, edges: List[float], x: float) -> None:
        self.hist(name, edges).add(x)

    # ---- Export -----------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "gauges": {k: asdict(v) for k, v in self.gauges.items()},
            "counters": {k: asdict(v) for k, v in self.counters.items()},
            "hists": {k: v.to_json() for k, v in self.hists.items()},
            "events": [asdict(e) for e in list(self.events)],
        }

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, separators=(",", ":"))


# ---------------------------
# Ledger adapter (optional)
# ---------------------------
@dataclass
class LedgerAdapter:
    """Adapter that translates telemetry events to SymbolicLedger entries.
    Expects a ledger object exposing `append(etype, payload, ...)` or
    convenience methods `record_op_commit`, `record_lock`, `record_rftrace`.
    """
    ledger: Any

    def __call__(self, event: Event) -> None:
        et = event.etype
        p = event.payload
        if et == "op":
            # write generic NOTE if specific method is missing
            if hasattr(self.ledger, "append"):
                self.ledger.append(etype=getattr(self.ledger, "EntryType", type("T", (), {"NOTE": "NOTE"})) , payload={"op": p})  # fallback
        elif et == "lock" and hasattr(self.ledger, "record_lock"):
            self.ledger.record_lock(type("LE", (), {"to_json": lambda self: p})())  # quick shim with to_json
        elif et == "rf" and hasattr(self.ledger, "record_rftrace"):
            self.ledger.record_rftrace(type("RF", (), {"to_json": lambda self: p})())
        else:
            # default to NOTE
            if hasattr(self.ledger, "record_note"):
                self.ledger.record_note(json.dumps({"etype": et, "payload": p}))


# ---------------------------
# RF Triggers
# ---------------------------
@dataclass
class RFTriggers:
    """Simple rules that raise Reverse Feedback tasks based on telemetry.
    Example rules:
      - low open_level for N ticks → schedule "integration"
      - repeated lock flaps → schedule "repair"
    """
    min_open_level: float = 0.25
    min_ticks: int = 10
    lock_flap_window: int = 20
    lock_flaps_min: int = 4

    # internal
    _open_below: int = 0
    _lock_hist: deque = field(default_factory=lambda: deque(maxlen=200))

    def on_decision(self, tel: Telemetry, open_level: float) -> None:
        if open_level < self.min_open_level:
            self._open_below += 1
        else:
            self._open_below = 0
        if self._open_below >= self.min_ticks:
            tel.record_rf("T1", "integration", [0.0, 0.0, 1.0], "scheduled", window=self.min_ticks)
            self._open_below = 0

    def on_lock(self, tel: Telemetry, acquired: bool) -> None:
        self._lock_hist.append(1 if acquired else 0)
        if len(self._lock_hist) >= self.lock_flap_window:
            flaps = sum(1 for i in range(1, len(self._lock_hist)) if self._lock_hist[i] != self._lock_hist[i-1])
            if flaps >= self.lock_flaps_min:
                tel.record_rf("T2", "repair", [0.0, 0.0, 1.0], "scheduled", flaps=flaps)
                self._lock_hist.clear()


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    tel = Telemetry(max_events=200)

    # Fake tick loop
    dt = 0.02
    rf = RFTriggers(min_open_level=0.3, min_ticks=5)

    for i in range(12):
        tel.tick_start(dt)
        # decisions (simulate openness oscillation)
        open_level = 0.2 if i < 6 else 0.7
        tel.record_decision("veil", open_level, resonance=0.8 if i >= 6 else 0.3)
        rf.on_decision(tel, open_level)

        # locks
        acquired = (i % 3) == 0
        tel.record_lock("attention", acquired, kappa=0.75, dwell=0.4)
        rf.on_lock(tel, acquired)

        # gauges
        tel.set_gauge("ops_budget", 2.5)
        tel.hist_add("speed", edges=[0.5, 1.0, 2.0, 4.0], x=0.6 + 0.1 * i)

        tel.tick_end(dt)

    print("events=", len(tel.events))
    print(tel.to_json()[:160] + "...")
