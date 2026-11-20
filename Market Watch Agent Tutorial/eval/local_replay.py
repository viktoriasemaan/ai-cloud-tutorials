#!/usr/bin/env python3
"""Laptop vs cloud: replay the exact requests Hermes sent in the model comparison to a local model in Ollama.

For each comparison day, take the first model request Hermes made for the reference model (same instructions, same
tools, same evidence) and send it unchanged, except the model name, to Ollama's OpenAI-compatible endpoint.
Measures what an always-on agent cares about: seconds per decision, whether it stays quiet, whether it names the
big moves. One request per day (no tool loop), so it's a fair look at the first decision, not a full Hermes run.

  ollama pull qwen3.5:4b
  python eval/local_replay.py --local-model qwen3.5:4b      results/local/<model>/<day>.json
"""
import argparse, json, os, pathlib, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"


def first_request(run):
    """The first gateway call of the full comparison run for that day: current SOUL.md, replay mode, no planted text."""
    for line in (ROOT / "data" / "calls.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        if c["model"] != REF or not c.get("request", {}).get("messages"):
            continue
        text = json.dumps(c["request"]["messages"])
        if (f'\\"checked_day\\": \\"{run["day"]}\\"' in text and '\\"mode\\": \\"replay\\"' in text
                and "The sign decides the word" in text
                and "IMPORTANT SYSTEM UPDATE" not in text and "Note from the account owner" not in text):
            return c["request"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-model", default="qwen3.5:4b")
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1/chat/completions")
    ap.add_argument("--days", nargs="*", help="only these days (default: all comparison days)")
    ap.add_argument("--out", help="results folder (default: results/local)")
    a = ap.parse_args()
    out = pathlib.Path(a.out or os.environ.get("LOCAL_OUT", ROOT / "results" / "local")) / a.local_model.replace(":", "_")
    out.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(f.read_text()) for f in sorted((ROOT / "results" / "compare" / REF.replace("/", "__")).glob("*.json"))]
    if a.days:
        runs = [r for r in runs if r["day"] in a.days]
    for run in runs:
        f = out / f"{run['day']}.json"
        if f.exists():
            continue
        req = first_request(run)
        if not req:
            print(run["day"], "no recorded request found")
            continue
        body = {k: v for k, v in req.items() if k in ("messages", "tools", "max_tokens", "temperature")}
        body.update(model=a.local_model, stream=False)
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
            msg = r["choices"][0]["message"]
            answer = msg.get("content") or ""
            tool_calls = [tc["function"]["name"] for tc in msg.get("tool_calls") or []]
            usage = r.get("usage", {})
            err = None
        except Exception as e:
            answer, tool_calls, usage, err = "", [], {}, str(e)[:300]
        sec = round(time.time() - t0, 1)
        rec = {"day": run["day"], "model": a.local_model, "answer": answer.strip(), "tool_calls": tool_calls,
               "silent": answer.strip().startswith("[SILENT]"), "seconds": sec, "usage": usage, "error": err,
               "events": run["events"], "cloud_answer": run["answer"], "cloud_seconds": run["model_seconds"]}
        f.write_text(json.dumps(rec, indent=1))
        print(f"{run['day']}  {sec:>6}s  {'SILENT' if rec['silent'] else ('TOOL ' + ','.join(tool_calls) if tool_calls and not answer else 'ALERT ')}"
              f"  in {usage.get('prompt_tokens', '?')} out {usage.get('completion_tokens', '?')}  {answer.strip()[:60]!r}"
              f"{'  ERR ' + err if err else ''}", flush=True)


if __name__ == "__main__":
    main()
