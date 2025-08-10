# echo.py
class EchoMatrix:
    def __init__(self, cfg):
        self.cfg = cfg
        self.snapshot = None

    def load_snapshot(self, snap, state):
        self.snapshot = snap
        # bind ancestry → state fields (biases, guilds, vows)
        state.apply_ancestry(snap)

    def save_snapshot(self, state) -> dict:
        # return minimal diff for ancestry persistence
        return state.export_ancestry()
