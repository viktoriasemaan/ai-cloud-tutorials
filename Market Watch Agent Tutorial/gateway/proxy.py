#!/usr/bin/env python3
"""The gateway: one OpenAI-compatible address in front of every model, and one referee for every call.

The agent (Hermes) talks to http://127.0.0.1:8787/v1 as if it were one provider. The gateway:
  1. routes the call by model name: "claude-*" to Anthropic, everything else to CoreWeave Serverless Inference,
  2. writes every request and answer to data/calls.jsonl (the raw material for training later),
  3. counts tokens and prices the call,
  4. sends an OpenTelemetry span per call to CoreWeave Agent Lens, so every step shows up there.

  python gateway/proxy.py                 (port 8787; set GATEWAY_PORT to change)

Keys (environment variable, or the file next to it):
  FORGE_API_KEY      ~/.config/coreweave/forge_key   CoreWeave Forge: inference + Agent Lens
  ANTHROPIC_API_KEY  ~/.config/anthropic/api_key     optional, only for claude-* models
Agent Lens project: AGENT_LENS_PROJECT ("<team>/<project>"), or set FORGE_TEAM (your Forge team name) for "<FORGE_TEAM>/market-agent".
"""
import hashlib, json, os, pathlib, sys, threading, time, urllib.error, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "calls.jsonl"
LOCK = threading.Lock()

COREWEAVE = "https://api.inference.wandb.ai/v1"
COREWEAVE_TRAINED = "https://api.training.wandb.ai/v1"  # where Serverless SFT/RL checkpoints are served
TRAINED_BASE = os.environ.get("TRAINED_BASE_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B")
ANTHROPIC = "https://api.anthropic.com/v1"  # Anthropic's OpenAI-compatible endpoint

# USD per 1M tokens: input, cached input, output (list prices checked 2026-11-17)
PRICES = {
    "openai/gpt-oss-120b": (0.03, None, 0.17), "openai/gpt-oss-20b": (0.03, None, 0.13),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B": (0.07, None, 0.20),
    "Qwen/Qwen3.6-35B-A3B": (0.10, None, 0.89), "Qwen/Qwen3.8-27B": (0.40, None, 3.00),
    "google/gemma-4-26B-A4B-it": (0.10, None, 0.30), "deepseek-ai/DeepSeek-V4.1-Flash": (0.13, 0.002, 0.65),
    "zai-org/GLM-5.3-Flash": (0.12, 0.02, 0.38), "zai-org/GLM-5.3": (0.75, None, 3.00), "MiniMaxAI/MiniMax-M3": (0.23, None, 0.96), "moonshotai/Kimi-K2.6": (0.65, None, 3.41),
    "claude-opus-5-5": (4.00, 0.20, 20.00),
}


def secret(env, path, required=True):
    v = os.environ.get(env, "").strip()
    if v:
        return v
    p = pathlib.Path(path).expanduser()
    if p.exists():
        return p.read_text().strip()
    if required:
        sys.exit(f"missing {env} (or {path})")
    return ""


FORGE_KEY = secret("FORGE_API_KEY", "~/.config/coreweave/forge_key")
ANTHROPIC_KEY = secret("ANTHROPIC_API_KEY", "~/.config/anthropic/api_key", required=False)
PROJECT = os.environ.get("AGENT_LENS_PROJECT", f"{os.environ.get('FORGE_TEAM', 'your-forge-team')}/market-agent")


def trained_aliases():
    """Short names for fine-tuned checkpoints: {"nemotron-ft": "wandb-artifact:///team/project/name:step3"}."""
    f = ROOT / "data" / "trained_models.json"
    return json.loads(f.read_text()) if f.exists() else {}


def cost(model, usage):
    # a fine-tuned adapter is priced here at its base model's rate (no separate list price published, Nov 2026)
    p = PRICES.get(model) or (PRICES.get(TRAINED_BASE) if model.startswith("wandb-artifact:") or model in trained_aliases() else None)
    if not p or not usage:
        return None
    inp = usage.get("prompt_tokens", 0) or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    out = usage.get("completion_tokens", 0) or 0
    cin = p[1] if p[1] is not None else p[0]
    return round(((inp - cached) * p[0] + cached * cin + out * p[2]) / 1e6, 6)


# ---------- Agent Lens (OpenTelemetry) ----------
TRACER = None
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    entity, project = PROJECT.split("/", 1)
    provider = TracerProvider(resource=Resource.create({
        "service.name": "market-agent-gateway", "wandb.entity": entity, "wandb.project": project}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
        endpoint=os.environ.get("AGENT_LENS_ENDPOINT", "https://forge.coreweave.com/api/trace/agents/otel/v1/traces"),
        headers={"wandb-api-key": FORGE_KEY})))
    trace.set_tracer_provider(provider)
    TRACER = trace.get_tracer("market-agent-gateway")
except ImportError:
    print("note: opentelemetry not installed, so no Agent Lens spans (pip install -r gateway/requirements.txt)", file=sys.stderr)


AGENT_NAME = os.environ.get("AGENT_NAME", "market-watch-agent")


def _text(c):
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def to_parts(msg):
    """OpenAI chat message -> GenAI semantic-convention message (role + typed parts)."""
    parts = []
    if msg.get("role") == "tool":
        parts.append({"type": "tool_call_response", "id": msg.get("tool_call_id", ""), "response": _text(msg.get("content"))})
    elif _text(msg.get("content")):
        parts.append({"type": "text", "content": _text(msg.get("content"))})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = fn.get("arguments")
        parts.append({"type": "tool_call", "id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args})
    return {"role": msg.get("role", "user"), "parts": parts}


def span(rec):
    """One Agent Lens turn (invoke_agent) holding one LLM call (chat), shaped for the Conversations tab."""
    if not TRACER:
        return
    t0, t1 = int(rec["t0"] * 1e9), int(rec["t1"] * 1e9)
    msgs = [to_parts(m) for m in rec["request"].get("messages", [])]
    out = [dict(to_parts((c or {}).get("message", {})), finish_reason=(c or {}).get("finish_reason") or "stop")
           for c in (rec.get("choices") or [])]
    u = rec.get("usage") or {}
    common = {"gen_ai.conversation.id": rec["conversation"], "gen_ai.agent.name": AGENT_NAME,
              "gen_ai.input.messages": json.dumps(msgs)[:60000], "gen_ai.output.messages": json.dumps(out)[:60000]}
    turn = TRACER.start_span(f"invoke_agent {AGENT_NAME}", start_time=t0,
                             attributes={**common, "gen_ai.operation.name": "invoke_agent"})
    ctx = trace.set_span_in_context(turn)
    chat = TRACER.start_span(f"chat {rec['model']}", context=ctx, start_time=t0, attributes={
        **common, "gen_ai.operation.name": "chat", "gen_ai.provider.name": rec["provider"],
        "gen_ai.request.model": rec["model"],
        "gen_ai.usage.input_tokens": u.get("prompt_tokens", 0) or 0,
        "gen_ai.usage.output_tokens": u.get("completion_tokens", 0) or 0,
        "market_agent.cost_usd": rec.get("cost_usd") or 0.0, "market_agent.latency_s": rec["latency_s"],
        "market_agent.tag": rec.get("tag", "")})
    if rec["request"].get("tools"):
        chat.set_attribute("gen_ai.tool.definitions", json.dumps(rec["request"]["tools"])[:20000])
    if rec.get("error"):
        chat.set_attribute("error.type", rec["error"][:200])
    chat.end(end_time=t1)
    turn.end(end_time=t1)


# ---------- budget: a hard stop on CoreWeave spend, kept by the gateway itself ----------
# Forge's own cap on the Free plan is fixed at $100/month beyond the allowance; this one is yours to set.
# Our estimate uses list prices, so set it below the real limit you want (default $40 for a $50 stop).
MONTHLY_LIMIT = float(os.environ.get("GATEWAY_MONTHLY_LIMIT_USD", "40"))
LEDGER = ROOT / "data" / "spend.json"


def spent_this_month():
    month = time.strftime("%Y-%m")
    try:
        d = json.loads(LEDGER.read_text())
    except Exception:
        d = {}
    return d.get(month, 0.0)


def add_spend(usd):
    month = time.strftime("%Y-%m")
    with LOCK:
        try:
            d = json.loads(LEDGER.read_text())
        except Exception:
            d = {}
        d[month] = round(d.get(month, 0.0) + usd, 6)
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(d))


def record(rec):
    rec["cost_usd"] = cost(rec["model"], rec.get("usage"))
    if rec["provider"] == "coreweave" and rec["cost_usd"]:
        add_spend(rec["cost_usd"])
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    try:
        span(rec)
    except Exception as e:  # tracing must never break the agent
        print("span error:", e, file=sys.stderr)
    u = rec.get("usage") or {}
    print(f"{time.strftime('%H:%M:%S')} {rec['model']:<46} in {u.get('prompt_tokens', '?'):>6} out "
          f"{u.get('completion_tokens', '?'):>5}  {rec['latency_s']:>5.1f}s  ${rec['cost_usd'] or 0:.5f}  {rec.get('tag', '')}",
          flush=True)


def fix_request(body, provider):
    """Small compatibility fixes between what Hermes sends and what a provider accepts (found while comparing models)."""
    fixes = []
    model = body.get("model", "")
    if provider == "coreweave":
        if model.startswith("deepseek-ai/") and "reasoning_effort" in body:
            body.pop("reasoning_effort")  # DeepSeek on CoreWeave returns 400 "Invalid request" with this field
            fixes.append("dropped reasoning_effort")
        for m in body.get("messages", []):
            if m.get("role") == "assistant" and "reasoning" in m and "reasoning_content" in m:
                m.pop("reasoning")  # both names map to one field there: "duplicate field `reasoning_content`"
                fixes.append("dropped duplicate reasoning")
    return sorted(set(fixes))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _upstream(self, model):
        if model.startswith("claude-"):
            if not ANTHROPIC_KEY:
                raise ValueError("no ANTHROPIC_API_KEY for a claude-* model")
            return "anthropic", ANTHROPIC, ANTHROPIC_KEY
        if model.startswith("wandb-artifact:"):  # a model you fine-tuned with Serverless SFT or RL (W&B artifact)
            return "coreweave-trained", COREWEAVE_TRAINED, FORGE_KEY
        return "coreweave", COREWEAVE, FORGE_KEY

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            req = urllib.request.Request(COREWEAVE + "/models", headers={"Authorization": "Bearer " + FORGE_KEY})
            body = urllib.request.urlopen(req, timeout=60).read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self.send_error(404, "only /v1/chat/completions is relayed")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        model = body.get("model", "")
        trained = trained_aliases().get(model)
        if trained:  # a short name for your fine-tuned checkpoint, e.g. "nemotron-ft" (see data/trained_models.json)
            body["model"] = trained
        try:
            provider, base, key = self._upstream(body["model"] if trained else model)
        except ValueError as e:
            return self.send_error(400, str(e))
        if provider.startswith("coreweave") and spent_this_month() >= MONTHLY_LIMIT:
            msg = json.dumps({"error": {"code": "gateway_budget_reached", "message":
                  f"Gateway budget reached: ${spent_this_month():.2f} of ${MONTHLY_LIMIT:.2f} this month. "
                  "Raise GATEWAY_MONTHLY_LIMIT_USD to continue."}}).encode()
            self.send_response(402)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            print(f"{time.strftime('%H:%M:%S')} BLOCKED {model}: monthly budget reached", flush=True)
            return
        fixes = fix_request(body, provider)
        stream = bool(body.get("stream"))
        if stream:
            body.setdefault("stream_options", {})["include_usage"] = True
        first_user = next((m.get("content") for m in body.get("messages", []) if m.get("role") == "user"), "")
        rec = {"id": str(uuid.uuid4()), "t0": time.time(), "provider": provider, "model": model,
               "tag": self.headers.get("X-Run-Tag", ""),
               "conversation": self.headers.get("X-Conversation-Id")
               or hashlib.sha1(json.dumps(first_user).encode()).hexdigest()[:16],
               "request": body, "fixes": fixes}
        req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                              "User-Agent": "market-watch-gateway/1.0"})  # Cloudflare blocks the default Python-urllib agent on the training endpoint (error 1010)
        try:
            up = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            err = e.read()
            rec.update(t1=time.time(), latency_s=round(time.time() - rec["t0"], 2), error=err.decode()[:2000])
            record(rec)
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        if not stream:
            data = up.read()
            ans = json.loads(data)
            rec.update(t1=time.time(), latency_s=round(time.time() - rec["t0"], 2),
                       choices=ans.get("choices"), usage=ans.get("usage"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            record(rec)
            return
        # streaming: pass every server-sent event through as it arrives, and rebuild the answer on the side
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        text, reasoning, tool_calls, usage, finish = [], [], {}, None, None
        for line in up:
            self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()
            s = line.decode("utf-8", "ignore").strip()
            if not s.startswith("data:") or s == "data: [DONE]":
                continue
            try:
                ev = json.loads(s[5:])
            except json.JSONDecodeError:
                continue
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                finish = ch.get("finish_reason") or finish
                text.append(d.get("content") or "")
                reasoning.append(d.get("reasoning_content") or d.get("reasoning") or "")
                for tc in d.get("tool_calls") or []:
                    t = tool_calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    t["id"] = tc.get("id") or t["id"]
                    fn = tc.get("function") or {}
                    t["name"] += fn.get("name") or ""
                    t["arguments"] += fn.get("arguments") or ""
        self.wfile.write(b"0\r\n\r\n")
        msg = {"role": "assistant", "content": "".join(text)}
        if any(reasoning):
            msg["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            msg["tool_calls"] = [{"id": t["id"], "type": "function",
                                  "function": {"name": t["name"], "arguments": t["arguments"]}}
                                 for _, t in sorted(tool_calls.items())]
        rec.update(t1=time.time(), latency_s=round(time.time() - rec["t0"], 2),
                   choices=[{"message": msg, "finish_reason": finish}], usage=usage)
        record(rec)


if __name__ == "__main__":
    port = int(os.environ.get("GATEWAY_PORT", "8787"))
    print(f"budget: ${spent_this_month():.2f} of ${MONTHLY_LIMIT:.2f} used this month (CoreWeave, list prices)", flush=True)
    print(f"gateway on http://127.0.0.1:{port}/v1 · CoreWeave Serverless Inference"
          f"{' + Anthropic' if ANTHROPIC_KEY else ''} · Agent Lens project {PROJECT if TRACER else '(off)'}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
