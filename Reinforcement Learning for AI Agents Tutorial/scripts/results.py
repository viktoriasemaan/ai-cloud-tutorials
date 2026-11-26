#!/usr/bin/env python3
"""Step 7 of the RL tutorial: before vs after, side by side.

Reads results/summary.json (written by evaluate.py) and the training log (written by train_rl.py).

  python3 scripts/results.py --labels base step0 rl
"""
import argparse, json, pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]
NAMES = {"base": "base (Serverless\nInference)", "step0": "before (step 0)", "rl": "after (step 20)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["base", "step0", "rl"])
    ap.add_argument("--run", default="store-agent-rl")
    a = ap.parse_args()
    s = json.loads((HERE / "results" / "summary.json").read_text())
    cols = [s[k] for k in a.labels]
    log = HERE / "results" / f"train-{a.run}.jsonl"
    if log.exists():
        steps = [json.loads(line) for line in open(log, encoding="utf-8")]
        print("training: average reward of the tries at each step (1.0 = every try right)")
        for st in steps:
            bar = "█" * round(st["reward"] * 40)
            val = f"   held-out check {st['val_reward']:.0%}" if "val_reward" in st else ""
            print(f"  step {st['step']:>2}  {st['reward']:.2f} {bar}{val}")
        print(f"  {sum(st['train_s'] for st in steps) / 60:.0f} min of training updates, "
              f"{sum(st['rollout_s'] for st in steps) / 60:.0f} min of tries\n")
    n = cols[0]["episodes"]
    print(f"held-out test: {n} tries on {n // 2} tasks the model never trained on\n")
    heads = [k.replace("base", "base*") for k in a.labels]
    print(f"{'task kind':24}" + "".join(f"{h:>12}" for h in heads))
    for k in cols[0]["by_kind"]:
        row = "".join(f"{c['by_kind'][k][0]:>8.0f} of {c['by_kind'][k][1]:<2}" for c in cols)
        print(f"{k:24}{row}")
    print(f"\n{'correct':24}" + "".join(f"{c['accuracy']:>12.0%}" for c in cols))
    for k, c in zip(a.labels, cols):
        print(f"  {k:6} {c['model']}")
    if "base" in a.labels:
        print("* base runs on CoreWeave Serverless Inference; step0 and the trained checkpoints run on the training endpoint,\n"
              "  so step0 -> rl is the like-for-like before and after.")


if __name__ == "__main__":
    main()
