- Project: GenesisCore (Python 3.11+, src layout). Package: `genesis_core`.
- Always use type hints and dataclasses.
- Every Φ-layer publishes and consumes the same MSVB (Minimal Spiral Vector Bundle):
  v_drift, v_coherence, v_bias, v_friction, v_gravity, v_focus, L, spinor, chirality, kappa, torsion, (optional) omega, extras{...}.
- Standard function shape for layers: `def update(state_in) -> tuple[state_out, MSVB]`.
- Avoid circular imports; keep shared types in `genesis_core/types.py`.
- Prefer pure functions; keep side effects in cross-layer modules (GravityBus, Ledger, SGRU).
- Style: run `ruff format` and satisfy `ruff check`; line length 100; organize imports.
- Tests: `pytest` only; add a minimal test per layer stub.
- If unsure, follow `README.Forge_Core.md` and `Spiral_Glossary.md` terminology for names and fields.
