#!/usr/bin/env python3
"""Steps 3 and 6 of the RL tutorial: score a model on the held-out test tasks, before and after training.

Every call goes through the AI gateway from the Market Watch Agent tutorial (http://127.0.0.1:8787/v1), so it counts
against your monthly budget and shows up in Agent Lens. The gateway also knows how to call a model you trained:
pass its wandb-artifact:/// name, or a short name from data/trained_models.json.

  python3 scripts/evaluate.py --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B --label base
  python3 scripts/evaluate.py --model nemotron-rl --label rl
"""
import argparse, asyncio, collections, json, os, pathlib, sys, time

from openai import AsyncOpenAI

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from agent import HERE, play  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", default=str(HERE / "data" / "tasks_test.jsonl"))
    ap.add_argument("--tries", type=int, default=2, help="episodes per task (the score is the average)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--show", type=int, default=0, help="print the queries of the first N episodes")
    a = ap.parse_args()
    tasks = [json.loads(line) for line in open(a.tasks, encoding="utf-8")]
    if a.limit:
        tasks = tasks[:a.limit]
    client = AsyncOpenAI(base_url=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8787/v1"), api_key="gateway",
                         default_headers={"X-Run-Tag": f"rl-eval-{a.label}"}, timeout=600, max_retries=2)
    sem = asyncio.Semaphore(a.parallel)
    out = HERE / "results" / f"eval-{a.label}.jsonl"
    out.parent.mkdir(exist_ok=True)
    t0 = time.time()

    async def one(task, k):
        async with sem:
            try:
                convo, ans, r, m, qs = await play(client, a.model, task, temperature=a.temperature)
                err = None
            except Exception as e:  # count a failed call as a wrong answer, but say so
                ans, r, m, qs, err = None, 0.0, {"correct": 0.0}, [], f"{type(e).__name__}: {e}"[:200]
            rec = {"task": task["id"], "template": task["template"], "try": k, "answer": ans, "gold": task["answer"],
                   "reward": r, "metrics": m, "queries": qs, "error": err}
            done.append(rec)
            mark = "✓" if r else "✗"
            print(f"  {len(done):>3}/{len(tasks) * a.tries} {mark} {task['template']:24} answer {str(ans)[:28]:28} right {task['answer']}"
                  + (f"  ({err[:60]})" if err else ""), flush=True)
            return rec

    done = []
    print(f"{a.model} · {len(tasks)} test tasks x {a.tries} tries · temperature {a.temperature}", flush=True)
    recs = await asyncio.gather(*(one(t, k) for t in tasks for k in range(a.tries)))
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    by = collections.defaultdict(list)
    for r in recs:
        by[r["template"]].append(r["reward"])
    acc = sum(r["reward"] for r in recs) / len(recs)
    print(f"\n{'task kind':24} correct")
    for k in sorted(by):
        print(f"{k:24} {sum(by[k]):>3.0f} of {len(by[k])}")
    errs = sum(1 for r in recs if r["error"])
    print(f"\n{a.label}: {sum(r['reward'] for r in recs):.0f} of {len(recs)} correct = {acc:.0%}"
          f" · {errs} failed calls · {time.time() - t0:.0f} s · saved {out.relative_to(HERE)}")
    summary = HERE / "results" / "summary.json"
    s = json.loads(summary.read_text()) if summary.exists() else {}
    s[a.label] = {"model": a.model, "correct": sum(r["reward"] for r in recs), "episodes": len(recs), "accuracy": round(acc, 4),
                  "by_kind": {k: [sum(v), len(v)] for k, v in sorted(by.items())}, "failed_calls": errs}
    summary.write_text(json.dumps(s, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
