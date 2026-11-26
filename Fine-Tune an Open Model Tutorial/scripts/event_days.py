#!/usr/bin/env python3
"""List the trading days that wake the model (at least one event) between two dates.

  python3 scripts/event_days.py --start 2026-01-02 --end 2026-08-21     the training days
  python3 scripts/event_days.py --start 2026-08-24 --end 2026-11-16     the test days (the 38 days of the model comparison)

Prints the days on one line, ready to pass to eval/compare.py --days.
"""
import argparse, json, pathlib, subprocess, sys

AGENT = pathlib.Path(__file__).resolve().parents[2] / "Market Watch Agent Tutorial"


def closes():
    c = json.loads((AGENT / "data" / "cache" / "prices" / "SPY.json").read_text())["closes"]
    h = AGENT / "data" / "cache" / "history" / "SPY.json"
    return {**json.loads(h.read_text())["closes"], **c} if h.exists() else c


def wakes(day):
    out = subprocess.run([sys.executable, str(AGENT / "scripts" / "precheck.py"), "--date", day], capture_output=True, text=True).stdout
    return '"wakeAgent": false' not in out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()
    days = [d for d in sorted(closes()) if a.start <= d <= a.end and wakes(d)]
    print(" ".join(days))
    print(f"{len(days)} days", file=sys.stderr)


if __name__ == "__main__":
    main()
