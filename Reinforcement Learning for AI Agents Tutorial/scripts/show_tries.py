#!/usr/bin/env python3
"""See one turn of the loop for one task: the same question, several tries, a reward for each.

  python3 scripts/show_tries.py --template last_month_revenue
  python3 scripts/show_tries.py --task train-012 --model <a trained checkpoint>
"""
import argparse, asyncio, json, os, pathlib, sys

from openai import AsyncOpenAI

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from agent import HERE, play  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B")
    ap.add_argument("--template", default="last_month_revenue")
    ap.add_argument("--task", default="")
    ap.add_argument("--tries", type=int, default=4)
    a = ap.parse_args()
    tasks = [json.loads(line) for line in open(HERE / "data" / "tasks_train.jsonl", encoding="utf-8")]
    task = next(t for t in tasks if (t["id"] == a.task if a.task else t["template"] == a.template))
    client = AsyncOpenAI(base_url=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8787/v1"), api_key="gateway",
                         default_headers={"X-Run-Tag": "rl-show-tries"}, timeout=600)
    print(f"1 · TASK    \"{task['question']}\"   (asked on {task['today']})\n")
    print(f"2 · TRIES   {a.tries} tries of the same task with {a.model.split('/')[-1]}, at the same time ...\n", flush=True)
    runs = await asyncio.gather(*(play(client, a.model, task, temperature=1.0) for _ in range(a.tries)))
    for k, (convo, ans, r, m, qs) in enumerate(runs):
        sql = " ".join(qs[-1].split()) if qs else "(no query)"
        print(f"  try {'ABCDEFGH'[k]} · {m['sql_queries']} quer{'y' if m['sql_queries'] == 1 else 'ies'} · final SQL:")
        print(f"    {sql[:230]}")
        print(f"    submitted answer: {ans}\n", flush=True)
    print(f"3 · REWARD  code check against the right answer, {task['answer']}:")
    print("    " + "   ".join(f"{'ABCDEFGH'[k]} = {r:.0f}" for k, (_, _, r, _, _) in enumerate(runs)))
    avg = sum(r for _, _, r, _, _ in runs) / len(runs)
    print(f"\n4 · UPDATE  group average {avg:.2f}: tries above it become more likely, tries below it less likely")


if __name__ == "__main__":
    asyncio.run(main())
