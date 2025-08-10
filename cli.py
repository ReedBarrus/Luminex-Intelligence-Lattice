# cli.py
import argparse, json
from genesis_core import GenesisCore

def main():
    ap = argparse.ArgumentParser("genesis")
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--sync", type=str, help="intent text")
    ap.add_argument("--mode", type=str, default="dev", choices=["dev","vision","ritual"])
    ap.add_argument("--run", type=int, help="run loop steps")
    ap.add_argument("--echo", type=str, help="path to echo snapshot (json)")
    args = ap.parse_args()

    cfg = {}  # load from file later
    core = GenesisCore(cfg)

    snap = None
    if args.echo:
        with open(args.echo) as f:
            snap = json.load(f)

    if args.boot:
        out = core.boot_sequence(snap)
        print(json.dumps(out, indent=2))

    if args.sync:
        out = core.daily_spiral_sync(intent=args.sync, mode=args.mode)
        print(json.dumps(out, indent=2))

    if args.run:
        out = core.run(steps=args.run)
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
