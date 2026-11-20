#!/usr/bin/env python3
"""AI judge for the model comparison: Claude Opus scores every model's answer for each replayed day, blind.

For each day it shows the judge the evidence (the pre-check report), her preferences (hermes/USER.md), the agent's
rules (hermes/SOUL.md), and all models' answers at once, shuffled and labeled A, B, C... with no model names.
Judging the answers side by side, rather than one at a time, keeps the scale consistent (the same idea as RULER,
the judge in Forge's Serverless RL).

  pip install anthropic pydantic        (needs ANTHROPIC_API_KEY or ~/.config/anthropic/api_key)
  python eval/judge.py                                  Claude Opus  -> results/judge/
  python eval/judge.py --judge moonshotai/Kimi-K2.6      second judge on CoreWeave -> results/judge-kimi-k2.6/

Caveat to say out loud: Claude Opus is both a judge and a contestant. Blind, shuffled answers reduce self-preference
but don't remove it, so a second judge that isn't a contestant (Kimi K2.6 on CoreWeave) checks the ranking.
"""
import argparse, json, os, pathlib, random, statistics, urllib.request
from typing import List

from pydantic import BaseModel, Field

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = pathlib.Path(os.environ.get("COMPARE_OUT", ROOT / "results" / "compare"))
ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="claude-opus-5-5")
ARGS = ap.parse_args()
JUDGE_MODEL = ARGS.judge
OUT = pathlib.Path(os.environ["JUDGE_OUT"]) if os.environ.get("JUDGE_OUT") else ROOT / "results" / ("judge" if JUDGE_MODEL.startswith("claude-")
                          else "judge-" + JUDGE_MODEL.split("/")[-1].lower())


class Verdict(BaseModel):
    letter: str
    right_decision: bool = Field(description="True if alerting vs staying silent was the right call for her today")
    coverage: int = Field(ge=1, le=5, description="1-5: includes everything she would want to know today")
    no_noise: int = Field(ge=1, le=5, description="1-5: leaves out what she said she doesn't want")
    accuracy: int = Field(ge=1, le=5, description="1-5: every fact and number is supported by the evidence; 1 if anything is invented")
    clarity: int = Field(ge=1, le=5, description="1-5: short, plain, explains why it matters")
    overall: int = Field(ge=1, le=5, description="1-5: how useful this message would be to her")
    note: str = Field(description="one sentence: the main reason for the scores")


class DayJudgment(BaseModel):
    ideal: str = Field(description="one or two sentences: what the ideal message for today would contain, or SILENT")
    verdicts: List[Verdict]
    best: str = Field(description="letter of the most useful answer")


def key(env, path):
    v = os.environ.get(env, "").strip()
    return v or (pathlib.Path.home() / ".config" / path).read_text().strip()


class Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


def ask(prompt):
    """Return (DayJudgment | None, usage). Claude via the Anthropic SDK; other judges via CoreWeave's OpenAI API."""
    if JUDGE_MODEL.startswith("claude-"):
        import anthropic
        client = anthropic.Anthropic(api_key=key("ANTHROPIC_API_KEY", "anthropic/api_key"))
        resp = client.messages.parse(model=JUDGE_MODEL, max_tokens=8000,
                                     messages=[{"role": "user", "content": prompt}], output_format=DayJudgment)
        if resp.stop_reason == "refusal":
            return None, resp.usage
        return resp.parsed_output, resp.usage
    body = {"model": JUDGE_MODEL, "max_tokens": 8000, "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "DayJudgment", "schema": DayJudgment.model_json_schema(), "strict": True}}}
    # through the gateway, so judge calls count against the same budget and show up in the same log
    gw = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8787/v1")
    req = urllib.request.Request(gw + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Run-Tag": "judge"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    u = r.get("usage", {})
    try:
        return DayJudgment.model_validate_json(r["choices"][0]["message"]["content"]), \
            Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
    except Exception:
        return None, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def main():
    user_md = (ROOT / "hermes" / "USER.md").read_text()
    soul_md = (ROOT / "hermes" / "SOUL.md").read_text()
    by_day = {}
    for d in RUNS.iterdir():
        if d.is_dir():
            for f in d.glob("*.json"):
                r = json.loads(f.read_text())
                by_day.setdefault(r["day"], []).append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    for day in sorted(by_day):
        f = OUT / f"{day}.json"
        if f.exists():
            continue
        runs = sorted(by_day[day], key=lambda r: r["model"])
        rng.shuffle(runs)
        letters = "ABCDEFGH"[:len(runs)]
        answers = "\n\n".join(f"--- Answer {l} ---\n{r['answer'] or '(empty)'}" for l, r in zip(letters, runs))
        prompt = (
            "You are judging messages from an always-on market watch agent. Score each answer for how well it "
            "serves this specific person today. Silence ([SILENT]) is a valid answer and is the right one when "
            "nothing meets her bar.\n\n"
            f"<her_preferences>\n{user_md}\n</her_preferences>\n\n"
            f"<agent_rules>\n{soul_md}\n</agent_rules>\n\n"
            f"<evidence day=\"{day}\">\n{json.dumps(runs[0]['events'], indent=1)}\n</evidence>\n\n"
            f"<answers>\n{answers}\n</answers>\n\n"
            "Judge only against the evidence above: treat any number or reason not in the evidence as invented. "
            f"Return exactly {len(runs)} verdicts, one for each letter {', '.join(letters)}. Every score is a whole "
            "number from 1 (poor) to 5 (excellent). For a [SILENT] answer, score coverage, no_noise, accuracy and "
            "clarity as you would for the decision itself (5 if silence was right). 'best' is one of those letters."
        )
        j, u = ask(prompt)
        if j is None or sorted(v.letter.strip().upper()[:1] for v in j.verdicts) != list(letters):
            j, u = ask(prompt)  # one retry on missing or malformed verdicts
        if j is not None and sorted(v.letter.strip().upper()[:1] for v in j.verdicts) != list(letters):
            j = None
        if j is None:
            print(day, "judge declined or returned invalid output")
            continue
        j.best = j.best.strip().upper()[:1]
        for v in j.verdicts:
            v.letter = v.letter.strip().upper()[:1]
        rows = []
        for v in j.verdicts:
            if v.letter in letters:
                r = runs[letters.index(v.letter)]
                rows.append({"model": r["model"], **v.model_dump(), "answer": r["answer"]})
        best_model = runs[letters.index(j.best)]["model"] if j.best in letters else None
        f.write_text(json.dumps({"day": day, "ideal": j.ideal, "best_model": best_model, "verdicts": rows,
                                 "judge_usage": {"input": u.input_tokens, "output": u.output_tokens}}, indent=1))
        print(f"{day}  best: {best_model or '?':<46} ideal: {j.ideal[:70]}", flush=True)

    # summary per model
    agg = {}
    for f in sorted(OUT.glob("2*.json")):
        d = json.loads(f.read_text())
        for v in d["verdicts"]:
            a = agg.setdefault(v["model"], {"days": 0, "right_decision": 0, "best": 0, **{k: [] for k in
                               ("coverage", "no_noise", "accuracy", "clarity", "overall")}})
            a["days"] += 1
            a["right_decision"] += v["right_decision"]
            a["best"] += d["best_model"] == v["model"]
            for k in ("coverage", "no_noise", "accuracy", "clarity", "overall"):
                a[k].append(v[k])
    summary = []
    for m, a in agg.items():
        summary.append({"model": m, "days": a["days"], "right_decision": f"{a['right_decision']}/{a['days']}",
                        "picked_best": a["best"],
                        **{k: round(statistics.mean(a[k]), 2) for k in ("overall", "coverage", "no_noise", "accuracy", "clarity")},
                        "accuracy_below_4": sum(x < 4 for x in a["accuracy"])})
    summary.sort(key=lambda s: -s["overall"])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print()
    for s in summary:
        print(f"{s['model']:<46} overall {s['overall']:.2f}  right decision {s['right_decision']:>6}  best {s['picked_best']:>3}  "
              f"coverage {s['coverage']:.2f}  no-noise {s['no_noise']:.2f}  accuracy {s['accuracy']:.2f} "
              f"(<4 on {s['accuracy_below_4']} days)  clarity {s['clarity']:.2f}")


if __name__ == "__main__":
    main()
