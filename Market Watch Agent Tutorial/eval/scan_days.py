#!/usr/bin/env python3
"""Run the pre-check over the last N trading days and count what it would wake the model for.

  python3 eval/scan_days.py            last 60 trading days
  python3 eval/scan_days.py --days 20
  python3 eval/scan_days.py --end 2026-11-16      the 60 days used in this tutorial
"""
import argparse, collections, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=60)
ap.add_argument("--end", help="last day to include (YYYY-MM-DD); default: the latest in the price data")
a = ap.parse_args()

closes = json.loads((ROOT / "data" / "cache" / "prices" / "SPY.json").read_text())["closes"]
days = [d for d in sorted(closes) if not a.end or d <= a.end][-a.days:]
kinds, forms, skipped, wake = collections.Counter(), collections.Counter(), collections.Counter(), 0
for d in days:
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "precheck.py"), "--date", d],
                         capture_output=True, text=True).stdout
    r = json.loads(out.split('\n{"wakeAgent"')[0])
    wake += bool(r["events"])
    for e in r["events"]:
        kinds[e["type"]] += 1
        if e["type"] == "filing":
            forms[e["form"]] += 1
    skipped.update(r["routine_filings_skipped"])
print(f"{days[0]} to {days[-1]}: {len(days)} trading days")
print(f"  days that wake the model:  {wake} of {len(days)}")
print(f"  events: {kinds['price_move']} price moves + {kinds['filing']} filings = {sum(kinds.values())}")
print(f"  top filings kept:  {', '.join(f'{k} {v}' for k, v in forms.most_common(6))}")
print(f"  routine filings skipped for free: {sum(skipped.values())}  ({', '.join(f'{k} {v}' for k, v in skipped.most_common(4))})")
