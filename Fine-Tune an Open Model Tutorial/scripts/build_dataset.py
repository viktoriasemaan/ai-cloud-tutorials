#!/usr/bin/env python3
"""Step 4 of the fine-tuning tutorial: turn judged answers into training examples.

For every training day it keeps the answers the judge scored 5 out of 5 with the right decision (alert or stay
silent), and writes one example per kept answer:

  input:  the exact request the agent sent to Nemotron that day (system prompt, the pre-check report, the tools),
          taken from the gateway's log, so the model learns from what it will really see in production
  target: the kept answer (from any of the four open models)

  python3 scripts/build_dataset.py
      --runs   ../Market Watch Agent Tutorial/results/compare-train       answers per model and day
      --judge  ../Market Watch Agent Tutorial/results/judge-train-glm-5.3 the judge's verdicts per day
      --min    5                                                          lowest overall score to keep

Writes data/train.jsonl (one {"messages": [...], "tools": [...]} per line, last message from the assistant) and
data/dataset_report.json (how many examples, from which models, how many silent).
"""
import argparse, collections, json, pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]
AGENT = HERE.parent / "Market Watch Agent Tutorial"
NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"


def requests_by_day(calls_file):
    """The first request Nemotron received for each replayed day: [system, user] messages and the tool list."""
    out = {}
    for line in open(calls_file, encoding="utf-8"):
        try:
            c = json.loads(line)
        except ValueError:
            continue
        req = c.get("request") or {}
        if c.get("model") != NEMOTRON or not isinstance(req, dict):
            continue
        msgs = req.get("messages") or []
        if len(msgs) != 2 or msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
            continue  # only the opening request, not the follow-ups after a tool call
        text = msgs[1].get("content")
        text = text if isinstance(text, str) else json.dumps(text)
        for key in ('"checked_day": "', '\\"checked_day\\": \\"'):
            i = text.find(key)
            if i >= 0:
                day = text[i + len(key): i + len(key) + 10]
                out[day] = {"messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                            "tools": req.get("tools") or []}
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(AGENT / "results" / "compare-train"))
    ap.add_argument("--judge", default=str(AGENT / "results" / "judge-train-glm-5.3"))
    ap.add_argument("--calls", default=str(AGENT / "data" / "calls.jsonl"))
    ap.add_argument("--min", type=int, default=5)
    a = ap.parse_args()
    reqs = requests_by_day(a.calls)
    examples, by_model, silent, days_used, missing = [], collections.Counter(), 0, set(), []
    for f in sorted(pathlib.Path(a.judge).glob("2*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        day = j["day"]
        if day not in reqs:
            missing.append(day)
            continue
        for v in j["verdicts"]:
            if v["overall"] < a.min or not v["right_decision"] or "claude" in v["model"]:
                continue
            answer = (v.get("answer") or "").strip()
            if not answer:
                continue
            examples.append({"messages": reqs[day]["messages"] + [{"role": "assistant", "content": answer}],
                             "tools": reqs[day]["tools"]})
            by_model[v["model"]] += 1
            silent += answer.startswith("[SILENT]")
            days_used.add(day)
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    report = {"examples": len(examples), "days": len(days_used), "silent_examples": silent,
              "by_model": dict(by_model), "days_without_a_logged_request": missing}
    (out / "dataset_report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(examples)} examples from {len(days_used)} days ({silent} are [SILENT])")
    for m, n in by_model.most_common():
        print(f"  {n:4}  {m}")
    if missing:
        print(f"  skipped {len(missing)} days with no logged Nemotron request")
    print(f"saved {out / 'train.jsonl'}")


if __name__ == "__main__":
    main()
