# coherence.py
from typing import Protocol

class Op(Protocol):
    name: str
    cost: float
    def __call__(self, state) -> None: ...

class PhaseScript:
    def __init__(self, cfg):
        self.ops: list[Op] = []
        self.cfg = cfg
        self._register_core_ops()

    def _register_core_ops(self):
        # ROTATE, TORQUE, LOCK, BREATHE, ECHO, REFLECT, SCALE
        # (Register as callables that check phase/consent gates before acting)
        pass

    def execute(self, state):
        # gate by minimal φ across layers; respect consent/locks
        for op in self.ops:
            if self._allowed(op, state):
                op(state)

    def _allowed(self, op, state) -> bool:
        # example: global gate = min(phi_layer_scores)
        # plus consent + chirality gates
        return True
