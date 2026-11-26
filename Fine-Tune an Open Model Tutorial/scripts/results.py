#!/usr/bin/env python3
"""Step 8 of the fine-tuning tutorial: the before-and-after table on the 38 test days.

For every model it prints the wrong calls (alert when it should stay silent, or the reverse) and the average score from
each judge, how often each judge picked it as the best answer, and what a run cost in tokens, seconds and dollars.

  python3 scripts/results.py
      --runs   ../Market Watch Agent Tutorial/results/compare-test
      --judges ../Market Watch Agent Tutorial/results/judge-test-opus ../Market Watch Agent Tutorial/results/judge-test-glm-5.3
"""
import argparse, glob, json, pathlib, statistics

AGENT = pathlib.Path(__file__).resolve().parents[2] / "Market Watch Agent Tutorial"
SHORT = {"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B": "Nemotron (base)",
         "nemotron-ft-e1": "Nemotron + LoRA, 1 epoch", "nemotron-ft-e3": "Nemotron + LoRA, 3 epochs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(AGENT / "results" / "compare-test"))
    ap.add_argument("--judges", nargs="+", default=[str(AGENT / "results" / "judge-test-opus"),
                                                    str(AGENT / "results" / "judge-test-glm-5.3")])
    a = ap.parse_args()
    runs = {}
    for f in glob.glob(f"{a.runs}/*/*.json"):
        r = json.load(open(f, encoding="utf-8"))
        runs.setdefault(r["model"], []).append(r)
    models = sorted(runs, key=lambda m: list(SHORT).index(m) if m in SHORT else 9)
    names = [pathlib.Path(j).name.replace("judge-test-", "") for j in a.judges]
    verdicts = {n: {} for n in names}
    best = {n: {} for n in names}
    days = {n: 0 for n in names}
    for n, j in zip(names, a.judges):
        for f in sorted(glob.glob(f"{j}/2*.json")):
            d = json.load(open(f, encoding="utf-8"))
            days[n] += 1
            best[n][d["best_model"]] = best[n].get(d["best_model"], 0) + 1
            for v in d["verdicts"]:
                verdicts[n].setdefault(v["model"], []).append(v)
    print(f"{'model':28}" + "".join(f"{'wrong · ' + n:>16}{'score':>7}{'best':>6}" for n in names)
          + f"{'out tokens':>12}{'seconds':>9}{'$ / 38 runs':>13}")
    for m in models:
        row = f"{SHORT.get(m, m):28}"
        for n in names:
            vs = verdicts[n].get(m, [])
            wrong = sum(not v["right_decision"] for v in vs)
            row += f"{f'{wrong} of {len(vs)}':>16}{statistics.mean(v['overall'] for v in vs) if vs else 0:>7.2f}{best[n].get(m, 0):>6}"
        rs = runs[m]
        row += (f"{statistics.mean(r['output_tokens'] or 0 for r in rs):>12.0f}{statistics.mean(r['run_seconds'] for r in rs):>9.1f}"
                f"{sum(r['cost_usd'] or 0 for r in rs):>13.4f}")
        print(row)
    print("judged days: " + ", ".join(f"{n} {days[n]}" for n in names))


if __name__ == "__main__":
    main()
