#!/usr/bin/env python3
"""Model comparison: the same agent (Hermes), the same instructions and the same replayed days; only the model changes.

For every (day, model) pair it:
  1. resets the agent's memory to the starting USER.md (so no run learns from another),
  2. points the "model-compare" cron job at that day and that model,
  3. runs the job once through Hermes, delivered locally (nothing goes to Telegram),
  4. reads the answer from Hermes's run record and the model calls from the gateway's log (data/calls.jsonl).

  python3 eval/compare.py --models openai/gpt-oss-120b claude-opus-5-5 --days 2026-10-29 2026-11-05
  python3 eval/compare.py --all-event-days            (every replay day that had at least one event)

Results: results/compare/<model>/<day>.json, plus results/compare/summary.json.
Run inside the environment where Hermes is installed (WSL here), with the gateway running.
"""
import argparse, glob, json, os, pathlib, re, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROFILE = os.environ.get("HERMES_PROFILE", "marketwatch")
HOME = pathlib.Path.home() / ".hermes" / "profiles" / PROFILE
CALLS = ROOT / "data" / "calls.jsonl"
OUT = pathlib.Path(os.environ.get("COMPARE_OUT", ROOT / "results" / "compare"))
JOB = "model-compare"
PROMPT = "Read the market report below and decide whether to alert. Follow your instructions."
DATE_FILE = HOME / "scripts" / "replay_date.txt"


def hermes(*args, check=True):
    r = subprocess.run(["hermes", "-p", PROFILE, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"hermes {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def job_id():
    for f in glob.glob(str(HOME / "cron" / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for j in (d.get("jobs", []) if isinstance(d, dict) else d if isinstance(d, list) else []):
            if j.get("name") == JOB:
                return j["id"]
    return None


def setup():
    script = HOME / "scripts" / "precheck_compare.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'exec python3 "{ROOT}/scripts/precheck.py" --date "$(cat {DATE_FILE})"\n')
    script.chmod(0o755)
    if not job_id():
        hermes("cron", "create", "every 1000d", PROMPT, "--name", JOB, "--script", "precheck_compare.sh",
               "--deliver", "local", "--no-wrap-response", "--paused")
    return job_id()


def reset_memory():
    mem = HOME / "memories"
    shutil.copy(ROOT / "hermes" / "USER.md", mem / "USER.md")
    (mem / "MEMORY.md").unlink(missing_ok=True)


def latest_output(after):
    docs = [p for p in glob.glob(str(HOME / "cron" / "output" / "*" / "*.md")) if os.path.getmtime(p) >= after]
    if not docs:
        return None, ""
    p = max(docs, key=os.path.getmtime)
    t = open(p, encoding="utf-8").read()
    m = re.search(r"^## Response\s*\n(.*)", t, re.S | re.M)
    return p, (m.group(1).strip() if m else "")


def calls_between(t0, t1, model):
    out = []
    if CALLS.exists():
        for line in CALLS.open(encoding="utf-8"):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t0 <= c["t0"] <= t1 and c["model"] == model:
                out.append(c)
    return out


def run_one(jid, day, model):
    DATE_FILE.write_text(day)
    reset_memory()
    hermes("cron", "edit", jid, "--model", model)
    hermes("cron", "resume", jid)
    t0 = time.time()
    log = hermes("cron", "run", jid, check=False)
    t1 = time.time()
    hermes("cron", "pause", jid, check=False)
    time.sleep(1.5)  # let the gateway finish writing its log line
    doc, answer = latest_output(t0 - 1)
    calls = calls_between(t0 - 1, time.time(), model)
    u = lambda k: sum(((c.get("usage") or {}).get(k) or 0) for c in calls)
    rep_events = []
    try:
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "precheck.py"), "--date", day],
                             capture_output=True, text=True).stdout
        rep_events = json.loads(out.split('\n{"wakeAgent"')[0])["events"]
    except Exception:
        pass
    silent = answer.strip() == "[SILENT]" or answer.strip().startswith("[SILENT]")
    return {
        "day": day, "model": model, "answer": answer, "silent": silent, "run_seconds": round(t1 - t0, 1),
        "model_calls": len(calls), "input_tokens": u("prompt_tokens"), "output_tokens": u("completion_tokens"),
        "cost_usd": round(sum(c.get("cost_usd") or 0 for c in calls), 6),
        "model_seconds": round(sum(c.get("latency_s", 0) for c in calls), 1),
        "errors": [c.get("error", "")[:300] for c in calls if c.get("error")],
        "events": rep_events, "run_log": log[-400:], "output_doc": doc.replace(str(pathlib.Path.home()), "~") if doc else doc,
    }


def event_days():
    c = json.load(open(ROOT / "data" / "cache" / "prices" / "SPY.json"))["closes"]
    days = []
    for d in sorted(c)[-60:]:
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "precheck.py"), "--date", d],
                             capture_output=True, text=True).stdout
        if '"wakeAgent": false' not in out:
            days.append(d)
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--days", nargs="*")
    ap.add_argument("--all-event-days", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    days = event_days() if a.all_event_days else a.days
    jid = setup()
    print(f"comparison job {jid}: {len(days)} days x {len(a.models)} models", flush=True)
    for day in days:
        for model in a.models:
            f = OUT / model.replace("/", "__") / f"{day}.json"
            if a.skip_existing and f.exists():
                continue
            r = run_one(jid, day, model)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(r, indent=1))
            first = (r["answer"].splitlines() or [""])[0][:70]
            print(f"{day} {model:<46} {'SILENT' if r['silent'] else 'ALERT '} calls {r['model_calls']} "
                  f"in {r['input_tokens']:>6} out {r['output_tokens']:>5} {r['run_seconds']:>5}s ${r['cost_usd']:.5f}"
                  f"{'  ERR' if r['errors'] else ''}  {first}", flush=True)
    reset_memory()


if __name__ == "__main__":
    main()
