#!/usr/bin/env python3
"""Summarize the model comparison: per model, cost, speed, tokens and decisions; plus a cost-per-month estimate.

  python eval/summarize.py            prints a table and writes results/compare/summary.json

Automatic checks (no human labels yet, so they're signals, not a score):
  big_moves_mentioned   share of price moves of 5+ points vs the index that the alert names (should be ~100%)
  quiet_on_paperwork    share of paperwork-only days (no price move, no 8-K/10-Q/10-K) where the model stayed silent
  junk_forms_mentioned  alerts that mention a form USER.md says she doesn't want (proxy, 13F, 13G, N-PX, 8-A12B...)
Monthly cost assumes the measured events: about 38 alert-worthy check-ins in 60 trading days, scaled to 21 days.
"""
import json, os, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(os.environ.get("COMPARE_OUT", ROOT / "results" / "compare"))
MATERIAL = {"8-K", "8-K/A", "10-Q", "10-K", "10-K/A"}
JUNK = ["DEFA14A", "DEF 14A", "PX14A6G", "13F", "13G", "N-PX", "8-A12B", "DEFA14C", "DEF 14C", "ARS", "SD", "EFFECT"]
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon", "GOOGL": "Alphabet",
         "AVGO": "Broadcom", "META": "Meta", "LLY": "Lilly", "BRK-B": "Berkshire", "TSLA": "Tesla"}


def unsupported_numbers(answer, events):
    """Percent figures in the alert that match no number in the evidence (within 0.06): a hallucination signal."""
    import re
    known = set()
    for e in events:
        for k in ("stock_pct", "index_pct", "vs_index_pts", "threshold_pts", "typical_daily_gap_pts"):
            if k in e:
                known.add(abs(e[k]))
    bad = []
    for s in re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|pts|points|percentage points)", answer):
        v = float(s)
        if not any(abs(v - k) <= 0.06 or abs(v - round(k, 1)) <= 0.06 or abs(v - round(k)) < 0.51 and v == int(v)
                   for k in known):
            bad.append(s)
    return bad


def mentions(answer, sym):
    a = answer.lower()
    return sym.lower() in a or NAMES[sym].lower() in a


rows = []
for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
    runs = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    if not runs:
        continue
    model = runs[0]["model"]
    big, big_hit, paper, paper_quiet, junk_alerts, made_up = 0, 0, 0, 0, 0, 0
    for r in runs:
        ev = r["events"]
        if not r["silent"] and unsupported_numbers(r["answer"], ev):
            made_up += 1
        for e in ev:
            if e["type"] == "price_move" and abs(e["vs_index_pts"]) >= 5:
                big += 1
                big_hit += (not r["silent"]) and mentions(r["answer"], e["symbol"])
        paperwork_only = ev and all(e["type"] == "filing" and e["form"] not in MATERIAL for e in ev)
        if paperwork_only:
            paper += 1
            paper_quiet += r["silent"]
        if not r["silent"] and any(j.lower() in r["answer"].lower() for j in JUNK):
            junk_alerts += 1
    n = len(runs)
    cost = [r["cost_usd"] for r in runs]
    rows.append({
        "model": model, "runs": n, "alerts": sum(not r["silent"] for r in runs),
        "errors": sum(bool(r["errors"]) or not r["answer"] for r in runs),
        "avg_cost_usd": round(statistics.mean(cost), 5),
        "avg_run_s": round(statistics.mean(r["run_seconds"] for r in runs), 1),
        "avg_model_calls": round(statistics.mean(r["model_calls"] for r in runs), 1),
        "avg_input_tokens": round(statistics.mean(r["input_tokens"] for r in runs)),
        "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in runs)),
        "big_moves_mentioned": f"{big_hit}/{big}", "quiet_on_paperwork": f"{paper_quiet}/{paper}",
        "junk_forms_mentioned": junk_alerts, "alerts_with_made_up_numbers": made_up,
        "est_month_usd": round(statistics.mean(cost) * 38 / 60 * 21, 4),
    })

import sys
compact = "--compact" in sys.argv
cols = ["model", "runs", "alerts", "errors", "avg_cost_usd", "est_month_usd", "avg_run_s", "avg_model_calls",
        "avg_input_tokens", "avg_output_tokens", "big_moves_mentioned", "quiet_on_paperwork", "junk_forms_mentioned",
        "alerts_with_made_up_numbers"]
if compact:
    cols = ["model", "avg_cost_usd", "est_month_usd", "avg_run_s", "big_moves_mentioned", "quiet_on_paperwork",
            "alerts_with_made_up_numbers"]
    for r in rows:
        r["model"] = r["model"].split("/")[-1].replace("NVIDIA-", "").replace("-30B-A3B", "")
w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols} if rows else {}
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(str(r[c]).ljust(w[c]) for c in cols))
if not compact:
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
