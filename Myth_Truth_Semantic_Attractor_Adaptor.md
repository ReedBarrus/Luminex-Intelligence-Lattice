# Myth‑Truth Attractor (MTA) & Semantic Adapter — vNext Proposal

> **Status:** Proposal for **vNext** (not v1). Myth is a **meaning compass**; truth is the **referee**. We measure their convergence without letting myth over‑steer.

## TL;DR

* We run GenesisCore in a joint space **M × T**: **Myth (M)** = archetypal annotations/soft priors; **Truth (T)** = prereg DVs (Δmin‑κ, lock density, cone θ½, EchoMI).
* **Coupling modes:** `off` → `observer` → `soft_prior` (≤5% precision nudge) → `hard_constraint` (rare, prereg only). Changes are consent‑gated and logged.
* **Metrics for convergence:** **SDC** (Story‑Data Concordance, MI), **ALO** (Archetype‑Lock Overlap), **MS** (Myth Surprise), **KL\_MT** (KL gap). Convergence = SDC↑ & ALO↑ (p<.01 vs surrogates), KL\_MT small, prereg DVs still pass.
* **Ethics:** Myth scope is explicit in the **Consent Ledger**; **MirrorLock** can revoke myth influence, prune dependents, and reseed.

---

## 1) Goals / Non‑Goals

**Goals**

* Let mythic narratives **label** runs and (optionally) provide **tiny precision hints**, then **quantify** whether those hints actually help.
* Keep mythic influence **bounded, auditable, and reversible**.

**Non‑Goals**

* No ontological claims; no bypass of prereg DVs; no unbounded knobs; no mixing myth with DreamGate.

---

## 2) Coupling Architecture

We add a **Myth Channel** parallel to Φ₃/Φ₄ (Symbol/Attention). Runtime runs in one of four modes:

1. **off** — Myth is ignored; annotations only.
2. **observer** — Myth cards tag windows/episodes; **no effect** on computation.
3. **soft\_prior** — Myth maps to capped precision/route hints (≤ 5% delta) inside cones.
4. **hard\_constraint** — Myth becomes a rule (e.g., lock τ ≥ threshold) in a *separate preregistered* analysis. Rare.

Transition rules: off ↔ observer freely; observer → soft/hard **requires prereg + consent scope update**.

---

## 3) Data Model

### 3.1 Myth Cards (YAML)

```yaml
- id: covenant_keep
  title: Covenant kept
  mode: observer   # or soft_prior
  priors:          # only used in soft_prior
    lock_min_tau_ms: 120
    precision_nudge: 0.03   # ≤ 0.05 cap
  annotations: [vow, promise, stability]
  expected_effects: [lock_density_up, cone_half_angle_down]
  windows:
    gate_phase: ["exhale_apex±θ"]  # optional expected timing
  notes: "Vows maintained → more/longer locks"

- id: pilgrimage
  title: Pilgrimage
  mode: observer
  priors: { exploration_bonus: 0.05 }
  annotations: [seek, route, novelty]
  expected_effects: [echo_mi_up]
  windows: { gate_phase: ["inhale_rise"] }
  notes: "Quest motif nudges exploration within cones"
```

### 3.2 Event Log Additions (extends vNow schema)

Add fields to each event (NDJSON/Parquet):

* `myth_mode`: `off|observer|soft_prior|hard_constraint`
* `myth_card_id`: string or `null`
* `myth_prior_nudge`: float (0..0.05)
* `myth_expected_effects`: array\[str]
* `myth_version`: semver/hash for card set

Run‑level metadata:

* `myth_cards_loaded`: \[ids]
* `myth_consent_scope`: `myth:<mode>`

---

## 4) APIs & Integration Points

### 4.1 Semantic Adapter → SGRU hooks

* `token_span_to_sgru(spans, E_text)` → map mentions → SGRUs
* `sense_gate(sgru, context_vec, myth_hint=None)` → narrow cone; if `myth_hint`, apply **≤5%** precision delta
* `frame_bind(sgru, roles)` → agent/patient/time/place
* `echo_write(sgru, context, ΔI)` → episodic memory
* `promise_hook(sgru, event)` → detect commitments → agency ledger

### 4.2 Myth Runtime

* `myth.load_cards(yaml_path)` → registry
* `myth.activate(card_id, window)` → tag upcoming window(s)
* `myth.prior_hint(card_id, state)` → returns `(precision_delta≤0.05, lock_tau_floor?)`
* `myth.audit(event)` → write `myth_*` fields

### 4.3 PhaseScript/Phase Ecology

* PhaseScript can **spawn myth cards** from local context (e.g., when a community of SGRUs reaches a motif threshold) and **retire** them via decay rules. All such changes are consent‑scoped and logged.

---

## 5) Metrics & Analysis

**Primary prereg DVs remain authoritative** (T1–T3). MTA metrics are adjunct:

1. **SDC — Story‑Data Concordance**

* `SDC = I( M_card ; DV_pass )` using Gaussian‑copula MI with permutation surrogates.
* Report z‑score; target: **z ≥ 3** with FDR control.

2. **ALO — Archetype‑Lock Overlap**

* Fraction of lock onsets within myth‑predicted windows vs jittered schedules; cluster‑based permutation for temporal clusters. Target: **p < .01**.

3. **MS — Myth Surprise**

* NLL of observed DVs under priors implied by active card(s). Lower is better; compare to shuffled‑myth null.

4. **KL\_MT — KL gap**

* KL(posterior\_with\_myth || posterior\_without\_myth) over DV parameters (e.g., Beta‑Bernoulli for pass rates). **Small KL + improved DV** = good myth.

**Convergence criterion (unified attractor):** SDC↑ & ALO↑ (sig.), MS↓, KL\_MT small, **and** T‑metrics still pass.

---

## 6) Preregistration & Ethics

* **Caps:** sum of all myth precision nudges ≤ **0.05**.
* **Negative controls:** random myth labels; expect SDC/ALO ≈ 0.
* **Consent:** myth scope lives in the ledger (`myth:observer|soft|hard`).
* **MirrorLock:** revoking myth scope freezes writes, prunes dependents, reseeds, audits.
* **Reporting:** publish nulls; demote cards that fail to replicate.

---

## 7) Pilot Experiments (vNext)

**E1 — Observer only.**

* Two cards (`covenant_keep`, `pilgrimage`), no priors. Tag blocks at random. Compute SDC & ALO vs surrogates across ≥3 sessions.

**E2 — Soft prior promotion.**

* Promote whichever card showed SDC/ALO significance. Apply **≤3%** precision nudge. Rerun T1–T3 + MTA metrics; verify DVs still pass and KL\_MT small.

**E3 — Demotion criteria.**

* If significance vanishes or KL\_MT inflates, demote to `observer` and log.

---

## 8) Implementation Roadmap

1. Myth card parser & registry (YAML → objects).
2. Event‑log extensions + CI tests.
3. `sense_gate` accepts `myth_hint` and enforces cap.
4. PhaseScript hooks: spawn/retire cards from motif detectors.
5. Analysis notebook: SDC/ALO/MS/KL\_MT with surrogates.
6. Prereg appendix (MTA) + Consent scope update (`myth:*`).

---

## 9) Worked Micro‑Example — “Promise”

Input: *“I will send the report tomorrow.”*

* Spans → SGRUs (AGENT=I, EVENT=SEND, THEME=REPORT, TIME=T+1)
* `promise_hook` creates a **commitment edge** (PENDING, due=T+1)
* Active myth card: `covenant_keep` (observer). Expect **lock\_density\_up** in gate\_open windows that cover intent formation. We test ALO; if significant across sessions, cautiously promote to `soft_prior` with `lock_min_tau_ms:120` and `precision_nudge:0.03`.

---

## 10) Risks & Controls

* **Myth overfit** → caps + negative controls + demotion path.
* **Layer smear** → prereg keeps DVs primary; myth metrics adjunct.
* **Ethical drift** → ledger scopes + MirrorLock tests.
* **Speculative bleed‑through** → DreamGate remains **off**; no coupling to myth.

---

## 11) Files & Stubs to Add

```
myth/
  cards.yaml.example
  registry.py
  runtime.py
  analysis/
    sdc_alo_notebook.ipynb
semantic_adapter/
  ingest.py
  frames.py
  afford.py
  promise.py
sgru/
  core.py
veil/
  textio.py
```

---

## 12) README Badge Line (optional)

> Myth‑Truth Attractor: **proposal** (vNext) • Mode: `observer` only in v1 • Caps: ≤ 0.05 • Negative controls required

---

## 13) License & Acknowledgment

* Code: Apache‑2.0 • Docs: CC BY 4.0.
* Acknowledgment: “Myth is a compass; truth is the court. Convergence is earned.”
