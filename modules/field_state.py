# modules/field_state.py
from __future__ import annotations
from typing import Dict, Any
from modules.types import MSVB, list_to_vec3

class FieldState:
    """
    Thin wrapper around the global dict FieldState used by runtime.genesis_loop.
    Provides helpers to read/write a layer's MSVB and to build minimal FieldViews.
    """
    def __init__(self, backing: Dict[str, Any]):
        self._fs = backing

    # --- time/breath ---
    @property
    def dt_sys(self) -> float: return float(self._fs["time"]["dt_sys"])
    @property
    def dt_phase(self) -> float: return float(self._fs["time"]["breath"].get("dt_phase", self.dt_sys))
    @property
    def alpha_breath(self) -> float: return float(self._fs["time"]["breath"]["alpha"])

    # --- global/coherence ---
    @property
    def mode6(self) -> str: return str(self._fs["global"].get("mode6","YELLOW"))
    @property
    def cones6(self): return self._fs["global"].get("cones6", [])

    # --- per-layer MSVB ---
    def get_layer(self, key: str) -> MSVB:
        return MSVB.from_dict(self._fs["layers"][key])

    def set_layer(self, key: str, msvb: MSVB) -> None:
        self._fs["layers"][key].update(msvb.to_dict())

    # --- specific layer views (if you want to keep the tiny FieldViews) ---
    def breath_view(self):
        from modules.breath import FieldState as BreathFS  # your Φ0 local view
        bfs = BreathFS()
        b = self._fs["time"]["breath"]
        bfs.dt_sys = float(self._fs["time"]["dt_sys"])
        bfs.breath_phase = float(b["phase"])
        bfs.alpha_breath = float(b["alpha"])
        bfs.beta_mod = float(b["beta_mod"])
        bfs.gate_open = float(self._fs["global"]["gate_open"])
        bfs.dt_phase = float(b.get("dt_phase", bfs.dt_sys))
        return bfs

    def phase_view(self):
        from modules.phase import PhaseFieldView
        return PhaseFieldView(dt_phase=self.dt_phase)

    def propagation_view(self):
        from modules.propagation import PropagationFieldView
        return PropagationFieldView(dt_phase=self.dt_phase)

    def symbol_view(self):
        from modules.symbol import SymbolFieldView
        return SymbolFieldView(dt_phase=self.dt_phase)
