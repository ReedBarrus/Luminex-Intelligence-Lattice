# Event Log Schema — GenesisCore vNow (v0.1)

> Minimal, machine‑readable schema so anyone can reproduce analyses. Use NDJSON or Parquet; keep one record per event.

## 0) File conventions

* **Format:** NDJSON (`.jsonl`) or Parquet.
* **Timezone:** UTC.
* **Filename:** `run-<sessionid>-<YYYYMMDD>-vNow.jsonl`
* **Compression:** gzip allowed.
* **Index:** optional `events.arrow` for fast reads.

## 1) Session metadata (first line or separate `session.json`)

```json
{
  "version": "vNow",
  "git_tag": "v1-design-2025-08-13",
  "commit": "<short-hash>",
  "mode": "sim|human",
  "operator_id": "anon-<hash>",
  "ptp_sync": true,
  "seed": 12345,
  "sampling_hz": {"breath": 50, "ppg": 200, "eeg": 0},
  "notes": "..."
}
```

## 2) Event record (per line)

| field              | type          | units/range         | required  | notes                                                 |        |                            |          |           |                  |   |                   |
| ------------------ | ------------- | ------------------- | --------- | ----------------------------------------------------- | ------ | -------------------------- | -------- | --------- | ---------------- | - | ----------------- |
| `t`                | float         | seconds (UTC epoch) | ✔         | event timestamp (high‑res)                            |        |                            |          |           |                  |   |                   |
| `layer`            | str           | \`breath            | phase     | prop                                                  | symbol | attention                  | echo\`   | ✔         | Φ‑layer at event |   |                   |
| `event_type`       | str           | \`birth             | stabilize | replicate                                             | mutate | decay                      | lock\_on | lock\_off | update\`         | ✔ | normalized labels |
| `vec`              | array\[float] | arbitrary dim       | ✔         | optional if large: store hash + separate vectors file |        |                            |          |           |                  |   |                   |
| `vec_norm`         | float         | L2                  | ✔         | if `vec` omitted, keep norm                           |        |                            |          |           |                  |   |                   |
| `cone_angle_deg`   | float         | 0–180               | ✔         | angle to cone axis; NaN if N/A                        |        |                            |          |           |                  |   |                   |
| `cone_id`          | str           |                     |           | useful if multiple cones per layer                    |        |                            |          |           |                  |   |                   |
| `lock_flag`        | bool          | 0/1                 | ✔         |                                                       |        |                            |          |           |                  |   |                   |
| `lock_type`        | str           | \`symbol            | attention | both                                                  | none\` | ✔                          |          |           |                  |   |                   |
| `lock_id`          | str           |                     |           | stable across on/off                                  |        |                            |          |           |                  |   |                   |
| `tau_ms`           | float         | ms                  |           | realized lock duration (if off)                       |        |                            |          |           |                  |   |                   |
| `sgru_id`          | str           |                     | ✔         | identity of symbol line                               |        |                            |          |           |                  |   |                   |
| `parent_id`        | str           |                     |           | lineage pointer (replicate/mutate)                    |        |                            |          |           |                  |   |                   |
| `gate_open`        | bool          | 0/1                 | ✔         | breath window flag                                    |        |                            |          |           |                  |   |                   |
| `breath_phase_rad` | float         | −π..π               | ✔         | at event time                                         |        |                            |          |           |                  |   |                   |
| `alpha_breath`     | float         | a.u.                |           | normalized amplitude                                  |        |                            |          |           |                  |   |                   |
| `echo_mi`          | float         | bits                |           | Gaussian‑copula MI per window                         |        |                            |          |           |                  |   |                   |
| `echo_entropy`     | float         | bits                |           | optional                                              |        |                            |          |           |                  |   |                   |
| `min_kappa`        | float         | 0..1                |           | min cross‑layer coherence in window                   |        |                            |          |           |                  |   |                   |
| `csi`              | float         | a.u.                |           | composite index (if computed online)                  |        |                            |          |           |                  |   |                   |
| `consent_scope`    | str           |                     | ✔         | scope token(s) attached to event                      |        |                            |          |           |                  |   |                   |
| `consent_hash`     | str           | hex                 |           | hash pointer in ledger                                |        |                            |          |           |                  |   |                   |
| `mirrorlock_flag`  | bool          | 0/1                 |           | if pruning/reseed was invoked                         |        |                            |          |           |                  |   |                   |
| `llm_mode`         | str           | \`off               | observer  | micro\`                                               |        | if LLM transduction active |          |           |                  |   |                   |
| `notes`            | str           |                     |           | human note (optional)                                 |        |                            |          |           |                  |   |                   |

## 3) Breath & biosignal streams

* Optional separate files: `breath.csv`, `ppg.csv`, `eeg.fif/BIDS`; each with UTC timestamps.

## 4) Example record (NDJSON)

```json
{"t": 1755100123.456, "layer": "symbol", "event_type": "lock_on", "vec_norm": 1.07, "cone_angle_deg": 18.3, "lock_flag": true, "lock_type": "symbol", "lock_id": "L-0081", "sgru_id": "S-42a", "parent_id": "S-39b", "gate_open": true, "breath_phase_rad": 2.09, "alpha_breath": 0.62, "echo_mi": 0.11, "min_kappa": 0.34, "consent_scope": "creative_session:poem", "consent_hash": "a9f...4c1", "mirrorlock_flag": false, "llm_mode": "observer"}
```

## 5) Quality & exclusions

* **Desync exclusion:** drop runs with PTP desync > 50 ms (human mode).
* **Sparse windows:** exclude windows with < N events (preregister N).
* **Outliers:** vector norms > 5 SD → winsorize or flag; apply uniformly.

## 6) Privacy & ethics

* No raw personal identifiers in logs.
* Consent scopes must be present; events without scope → reject at runtime.

## 7) Repro pointers

* Analysis scripts expect NDJSON with the above fields.
* For large vectors, store `vec_hash` in the event and write vectors to `vectors.arrow` with the same `event_id`.

**Design prereg (OSF, project view-only):** https://osf.io/6uvec/?view_only=f03ec28aede8411bb9b8452b7d4440b6
Docs-only tag: `v1-design-2025-08-13` · Implementation prereg: **pending** (will pin analysis scripts before first data)

Registration (frozen form): **pending** — a separate view-only/public link will be added after submission.
