#!/usr/bin/env python3
"""Compare the two AI judges on the days both scored validly: overall score, right decisions, and rank.

  python3 eval/judges_table.py
"""
import collections, json, os, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
A, B = ROOT / "results" / "judge", ROOT / "results" / "judge-kimi-k2.6"
days = sorted(p.name for p in B.glob("2*.json") if (A / p.name).exists())
table = {}
for name, d in (("Opus judge", A), ("Kimi judge", B)):
    scores, right = collections.defaultdict(list), collections.Counter()
    for day in days:
        for v in json.loads((d / day).read_text())["verdicts"]:
            m = v["model"].split("/")[-1].replace("NVIDIA-", "").replace("-30B-A3B", "")
            scores[m].append(v["overall"]); right[m] += v["right_decision"]
    ranked = sorted(scores, key=lambda m: -statistics.mean(scores[m]))
    table[name] = {m: (statistics.mean(scores[m]), right[m], ranked.index(m) + 1) for m in scores}
same = sum(json.loads((A / d).read_text())["best_model"] == json.loads((B / d).read_text())["best_model"] for d in days)
print(f"{len(days)} days scored validly by both judges; same best answer on {same} of them\n")
print(f"{'model':<24}{'Opus judge':>18}{'Kimi judge':>18}{'avg rank':>10}")
rows = sorted(table["Opus judge"], key=lambda m: (table["Opus judge"][m][2] + table["Kimi judge"][m][2]) / 2)
for m in rows:
    o, k = table["Opus judge"][m], table["Kimi judge"][m]
    print(f"{m:<24}{o[0]:>8.2f} #{o[2]} ({o[1]}/{len(days)}){k[0]:>8.2f} #{k[2]} ({k[1]}/{len(days)}){(o[2]+k[2])/2:>9.1f}")
