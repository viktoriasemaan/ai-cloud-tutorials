#!/usr/bin/env python3
"""What did everything cost? Totals from the gateway's own log (data/calls.jsonl), at list prices.

  python3 eval/costs.py              by model
  python3 eval/costs.py --by tag     by run tag (e.g. judge)
"""
import argparse, collections, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--by", default="model", choices=["model", "provider", "tag"])
a = ap.parse_args()

agg = collections.defaultdict(lambda: [0, 0, 0, 0.0])
for line in (ROOT / "data" / "calls.jsonl").open(encoding="utf-8"):
    c = json.loads(line)
    u = c.get("usage") or {}
    k = c.get(a.by) or "(none)"
    r = agg[k]
    r[0] += 1; r[1] += u.get("prompt_tokens") or 0; r[2] += u.get("completion_tokens") or 0; r[3] += c.get("cost_usd") or 0
print(f"{a.by:<46}{'calls':>7}{'in tokens':>12}{'out tokens':>12}{'cost':>10}")
for k, (n, i, o, usd) in sorted(agg.items(), key=lambda kv: -kv[1][3]):
    print(f"{k[:45]:<46}{n:>7}{i:>12,}{o:>12,}{'$' + format(usd, '.3f'):>10}")
tot = [sum(v[j] for v in agg.values()) for j in range(4)]
print(f"{'total':<46}{tot[0]:>7}{tot[1]:>12,}{tot[2]:>12,}{'$' + format(tot[3], '.3f'):>10}")
spend = ROOT / "data" / "spend.json"
if spend.exists():
    print(f"\nbudget ledger (CoreWeave, this month): {json.loads(spend.read_text())}")
