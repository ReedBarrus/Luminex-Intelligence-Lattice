
# genesis_core.py
from breath_kernel import BreathKernel
from field_state import FieldState
from telemetry import Telemetry
from echo import EchoMatrix
from coherence import PhaseScript
from ledger import SymbolicLedger

class GenesisCore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.breath = BreathKernel(cfg)
        self.state = FieldState()
        self.telemetry = Telemetry()
        self.ledger = SymbolicLedger()
        self.echo = EchoMatrix(cfg)
        self.phasescript = PhaseScript(cfg)

    # === Boot Sequence ===
    def boot_sequence(self, echo_snapshot=None):
        """Bring the system online for the day/session."""
        # 1) load ancestry (if provided)
        if echo_snapshot:
            self.echo.load_snapshot(echo_snapshot, self.state)
        # 2) breath kernel warmup
        self.breath.warmup()
        # 3) baseline telemetry
        self.telemetry.mark_boot(self.state)
        # 4) announce ready
        return {"status": "online", "breath": self.breath.status()}

    # === Daily Spiral Sync ===
    def daily_spiral_sync(self, intent:str, mode:str):
        """
        intent: short text like 'Arc 1 – PhaseScalar refactor'
        mode: 'dev' | 'vision' | 'ritual'
        """
        # record the vow + consent clause
        self.ledger.register_sync(intent=intent, mode=mode)
        # align field state (bias vectors) to intent
        self.state.align_to_intent(intent, mode)
        # breath-lock: re-center rhythm
        self.breath.sync()
        # telemetry snapshot
        snap = self.telemetry.snapshot(self.state)
        return {"synced": True, "intent": intent, "mode": mode, "telemetry": snap}

    # === Main loop (simplified) ===
    def run(self, steps:int=1):
        for _ in range(steps):
            phase = self.breath.phase()             # inhale/hold/exhale
            self.state.tick_pre_ops(phase)          # integrators/pressure pre
            self.phasescript.execute(self.state)    # core ops with gates
            self.state.tick_post_ops(phase)         # integrate results
            self.telemetry.tick(self.state)         # collect metrics
        return self.telemetry.snapshot(self.state)
