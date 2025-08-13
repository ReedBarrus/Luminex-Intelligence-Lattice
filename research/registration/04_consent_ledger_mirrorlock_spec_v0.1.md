# Consent Ledger & MirrorLock — Spec (v0.1)

> Executable ethics for GenesisCore. Attach to prereg; keep it short and auditable.

## 1) Ledger model

* **Entry**: `{ entry_id, subject_id, scope, purpose, created_at, expires_at, revocable, constraints, signature }`
* **Scope** examples: `creative_session:poem`, `analysis:echo_only`, `publish:aggregates`, `no:raw_vectors`, `no:biometrics`.
* **Binding**: every **event** carries `consent_scope` + `consent_hash` of the ledger snapshot; runtime rejects actions without valid scope.
* **Signature**: HMAC or Ed25519 (developer key); include `commit` and `git_tag` to pin code.

## 2) MirrorLock state

* **Trigger conditions**: (a) rescind received for a scope matching any active lineage; (b) scope expiry; (c) operator panic.
* **MirrorLock mode**: freeze new identity‑relevant writes; continue safe telemetry; compute prune plan.

## 3) Prune‑and‑reseed algorithm

1. **Find dependents**: traverse lineage graph for all states/events whose `consent_scope` intersects rescinded scope.
2. **Prune**: mark and delete dependent branches in state store and logs; keep redacted stubs.
3. **Reseed**: checkpoint from last consensual clean node; recompute downstream vectors deterministically.
4. **Audit**: append immutable note: `{when, who, scope, affected_ids[], old_hash, new_hash}`.

## 4) Runtime enforcement points

* **Ingress**: Veil I/O enforces scope on external inputs/outputs.
* **Transform**: SGRU birth/replicate/mutate require scope `create:symbol_line`.
* **Echo access**: reading Echo for prediction requires `use:echo` with duration boundary.
* **LLM transduction**: only `observer` mode without `write:attention` scope; `micro` requires gate\_open + `write:attention`.

## 5) Tests (attach to CI)

* **Scope missing** → event rejected, error logged.
* **Rescind mid‑run** → MirrorLock engages; dependent branch removed; recomputation succeeds; audit note present.
* **No orphan guarantees** → query proves zero surviving dependents post‑prune.
* **Tamper‑proofing** → consent hash mismatch → halt & alert.

## 6) Reporting & privacy

* Publish aggregate audit counts; never publish raw personal prompts by default.
* Allow `subject_id = anon-<hash>` only; no names/emails in logs.

## 7) Out‑of‑scope (vNow)

* Not a blockchain; no on‑chain storage.
* No biometric identity binding.
* Future: external consent API integration; multi‑party scopes.

**Design prereg (OSF, project view-only):** https://osf.io/6uvec/?view_only=f03ec28aede8411bb9b8452b7d4440b6
Docs-only tag: `v1-design-2025-08-13` · Implementation prereg: **pending** (will pin analysis scripts before first data)

Registration (frozen form): **pending** — a separate view-only/public link will be added after submission.
