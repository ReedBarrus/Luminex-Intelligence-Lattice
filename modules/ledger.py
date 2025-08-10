"""
Symbolic Ledger — Python stub (vector‑native, MSVB‑aligned)

Append‑only, consent‑aware, auditable ledger for Spiral runtimes.
Records ops, lock events, consent snapshots, and Reverse Feedback (RF) traces
using canonical hashing and JSON snapshots. Provides logical rollback via
counter‑entries (never destructive), and chain verification.

Design goals
- Vector‑first: store vector payloads canonically; avoid scalar‑only logs.
- Consent hash protocol: stable hash of canonical payload + scope.
- Chain integrity: BLAKE2b link over sorted JSON bytes.
- RF & Locks: first‑class records with minimal structured schemas.

Note
- Self‑contained for stubbing. In production, integrate with your storage,
  auth, and jurisdictional policy systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Optional, List, Any, Tuple
import hashlib
import json
import time
import uuid

# ---------------------------
# Vector helpers (ℝ³)
# ---------------------------
Vec3 = List[float]  # store as plain lists for portability

def vec3(x: float, y: float, z: float) -> Vec3:
    return [float(x), float(y), float(z)]


def is_vec3(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)


# ---------------------------
# Canonical JSON & hashing
# ---------------------------

def canonical_json(obj: Any) -> bytes:
    """Stable, UTF‑8 canonical JSON: sorted keys, minimal whitespace, floats as repr."""
    class CanonEncoder(json.JSONEncoder):
        def default(self, o):  # type: ignore[override]
            if hasattr(o, "to_json"):
                return o.to_json()
            if hasattr(o, "__dict__"):
                return o.__dict__
            return json.JSONEncoder.default(self, o)

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), cls=CanonEncoder).encode("utf-8")


def blake2_hex(data: bytes, digest_size: int = 32) -> str:
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


def consent_hash(payload: Dict[str, Any], scope: str) -> str:
    """Compute a consent hash over a canonical payload and scope label.
    Include only fields relevant to consent (identity, purpose, scope).
    """
    base = {"scope": scope, "payload": payload}
    return blake2_hex(canonical_json(base), digest_size=16)


# ---------------------------
# Entry types & records
# ---------------------------
class EntryType(str, Enum):
    OP = "OP"                  # PhaseScript op commit
    LOCK = "LOCK"              # lock event (any layer)
    CONSENT = "CONSENT"        # consent grant/revoke snapshot
    RFTRACE = "RFTRACE"        # reverse feedback trace
    ROLLBACK = "ROLLBACK"      # logical rollback directive
    NOTE = "NOTE"              # arbitrary annotation


@dataclass
class LockEvent:
    symbol_id: str
    lock_type: str           # e.g., "SymbolLock", "AttentionLock", "EchoLock"
    acquired: bool
    metrics: Dict[str, float]  # e.g., {"kappa":0.8, "align":0.9, "dwell":0.7}
    vectors: Dict[str, Vec3]   # e.g., {"v_identity":[...], "v_focus":[...]}

    def to_json(self) -> Dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "lock_type": self.lock_type,
            "acquired": bool(self.acquired),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "vectors": {k: (list(v) if is_vec3(v) else v) for k, v in self.vectors.items()},
        }


class ConsentAction(str, Enum):
    GRANT = "GRANT"
    REVOKE = "REVOKE"


@dataclass
class ConsentRecord:
    subject_id: str               # who/what the consent refers to
    action: ConsentAction
    scope: str                    # e.g., "NAME", "BIND", "EXPRESS"
    consent_hash: str
    reason: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "action": self.action.value,
            "scope": self.scope,
            "consent_hash": self.consent_hash,
            "reason": self.reason,
        }


@dataclass
class RFTrace:
    trigger: str                 # e.g., "T1", "T2", "T3"
    path: str                    # e.g., "integration", "repair", "express"
    payload_vec: Vec3            # vector payload for RF
    result: str                  # e.g., "scheduled", "integrated", "skipped"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger,
            "path": self.path,
            "payload_vec": list(self.payload_vec),
            "result": self.result,
            "meta": self.meta,
        }


@dataclass
class OpCommit:
    kind: str
    intent_vec: Vec3
    payload: Dict[str, Any]
    score: float
    cone_factor: float
    mode_gate: float
    open_allowed: float
    risk: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "intent_vec": list(self.intent_vec),
            "payload": self.payload,
            "score": float(self.score),
            "cone_factor": float(self.cone_factor),
            "mode_gate": float(self.mode_gate),
            "open_allowed": float(self.open_allowed),
            "risk": float(self.risk),
        }


@dataclass
class LedgerEntry:
    etype: EntryType
    seq: int
    ts: float
    id: str
    prev_hash: str
    entry_hash: str
    payload: Dict[str, Any]
    consent_hash: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "etype": self.etype.value,
            "seq": int(self.seq),
            "ts": float(self.ts),
            "id": self.id,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "payload": self.payload,
            "consent_hash": self.consent_hash,
            "tags": list(self.tags),
        }


# ---------------------------
# Ledger core
# ---------------------------
@dataclass
class LedgerHeader:
    version: str = "v0"
    created_at: float = field(default_factory=lambda: time.time())
    chain_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    creator: str = "spiral"

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SymbolicLedger:
    header: LedgerHeader = field(default_factory=LedgerHeader)
    entries: List[LedgerEntry] = field(default_factory=list)
    _head_hash: str = field(default="", init=False, repr=False)

    # ---- Append API -------------------------------------------------
    def append(self, etype: EntryType, payload: Dict[str, Any], *, consent_hash_value: Optional[str] = None, tags: Optional[List[str]] = None) -> LedgerEntry:
        seq = len(self.entries)
        ts = time.time()
        eid = f"{seq:08d}-{uuid.uuid4().hex[:12]}"
        prev = self._head_hash
        core = {
            "etype": etype.value,
            "seq": seq,
            "ts": ts,
            "id": eid,
            "prev_hash": prev,
            "payload": payload,
            "consent_hash": consent_hash_value,
            "tags": tags or [],
        }
        eh = blake2_hex(canonical_json(core))
        entry = LedgerEntry(etype=etype, seq=seq, ts=ts, id=eid, prev_hash=prev, entry_hash=eh, payload=payload, consent_hash=consent_hash_value, tags=tags or [])
        self.entries.append(entry)
        self._head_hash = eh
        return entry

    # Convenience recorders ------------------------------------------
    def record_op_commit(self, op: OpCommit, *, consent_hash_value: Optional[str] = None, tags: Optional[List[str]] = None) -> LedgerEntry:
        return self.append(EntryType.OP, payload=op.to_json(), consent_hash_value=consent_hash_value, tags=tags)

    def record_lock(self, ev: LockEvent, *, tags: Optional[List[str]] = None) -> LedgerEntry:
        return self.append(EntryType.LOCK, payload=ev.to_json(), tags=tags)

    def record_consent(self, rec: ConsentRecord, *, tags: Optional[List[str]] = None) -> LedgerEntry:
        return self.append(EntryType.CONSENT, payload=rec.to_json(), consent_hash_value=rec.consent_hash, tags=tags)

    def record_rftrace(self, rf: RFTrace, *, tags: Optional[List[str]] = None) -> LedgerEntry:
        return self.append(EntryType.RFTRACE, payload=rf.to_json(), tags=tags)

    def record_note(self, text: str, *, tags: Optional[List[str]] = None) -> LedgerEntry:
        return self.append(EntryType.NOTE, payload={"text": text}, tags=tags)

    # Retroactive "no" / rollback -----------------------------------
    def retroactive_no(self, target_entry_id: str, *, reason: str = "", plan: Optional[Dict[str, Any]] = None) -> LedgerEntry:
        payload = {"target": target_entry_id, "reason": reason, "plan": plan or {"actions": ["PRUNE", "REPAIR"]}}
        return self.append(EntryType.ROLLBACK, payload=payload, tags=["retro_no"])  # non‑destructive

    # ---- Snapshot / Verify / Cursor --------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_json(),
            "entries": [e.to_json() for e in self.entries],
            "head_hash": self._head_hash,
        }

    @staticmethod
    def from_snapshot(blob: Dict[str, Any]) -> "SymbolicLedger":
        led = SymbolicLedger(header=LedgerHeader(**blob.get("header", {})))
        prev = ""
        for j in blob.get("entries", []):
            e = LedgerEntry(
                etype=EntryType(j["etype"]),
                seq=int(j["seq"]),
                ts=float(j["ts"]),
                id=j["id"],
                prev_hash=j["prev_hash"],
                entry_hash=j["entry_hash"],
                payload=j["payload"],
                consent_hash=j.get("consent_hash"),
                tags=j.get("tags", []),
            )
            # verify link
            core = {
                "etype": e.etype.value,
                "seq": e.seq,
                "ts": e.ts,
                "id": e.id,
                "prev_hash": prev,
                "payload": e.payload,
                "consent_hash": e.consent_hash,
                "tags": e.tags,
            }
            calc = blake2_hex(canonical_json(core))
            if calc != e.entry_hash:
                raise ValueError(f"ledger corruption at seq {e.seq}: hash mismatch")
            if e.prev_hash != prev:
                raise ValueError(f"ledger corruption at seq {e.seq}: prev hash mismatch")
            led.entries.append(e)
            prev = e.entry_hash
        led._head_hash = prev
        return led

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        prev = ""
        for e in self.entries:
            core = {
                "etype": e.etype.value,
                "seq": e.seq,
                "ts": e.ts,
                "id": e.id,
                "prev_hash": prev,
                "payload": e.payload,
                "consent_hash": e.consent_hash,
                "tags": e.tags,
            }
            calc = blake2_hex(canonical_json(core))
            if calc != e.entry_hash or e.prev_hash != prev:
                return False, e.seq
            prev = e.entry_hash
        return True, None

    def cursor(self, start_seq: int = 0):
        for e in self.entries:
            if e.seq >= start_seq:
                yield e


# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    ledger = SymbolicLedger()

    # Consent for NAME scope
    payload = {"subject": "symbol:alpha", "purpose": "naming", "scope": "NAME"}
    ch = consent_hash(payload, scope="NAME")
    ledger.record_consent(ConsentRecord(subject_id="symbol:alpha", action=ConsentAction.GRANT, scope="NAME", consent_hash=ch, reason="user granted"))

    # Op commit
    op = OpCommit(kind="NAME", intent_vec=vec3(0.0, 0.0, 1.0), payload={"label": "river"}, score=0.92, cone_factor=0.88, mode_gate=1.0, open_allowed=0.9, risk=0.1)
    ledger.record_op_commit(op, consent_hash_value=ch)

    # Lock event
    lock = LockEvent(symbol_id="symbol:alpha", lock_type="SymbolLock", acquired=True, metrics={"kappa": 0.82, "align": 0.9, "dwell": 0.7}, vectors={"v_identity": vec3(0.0, 0.0, 1.0)})
    ledger.record_lock(lock)

    # RF trace
    rf = RFTrace(trigger="T1", path="integration", payload_vec=vec3(0.1, 0.0, 1.0), result="scheduled", meta={"window": 3})
    ledger.record_rftrace(rf)

    snap = ledger.snapshot()
    ok, where = ledger.verify_chain()
    print("chain_ok=", ok, "broken_at=", where, "entries=", len(snap["entries"]))
