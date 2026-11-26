"""The store analytics agent and its reward check, shared by evaluate.py and train_rl.py.

One episode = one task: the model gets the question and the database schema, calls run_sql as many times as it
needs (up to MAX_TURNS model turns), and ends by calling submit_answer. The reward is a code check: 1 if the
submitted answer matches the answer of the reference query, 0 if not. That's all the model is ever told.
"""
import json, pathlib, re, sqlite3

HERE = pathlib.Path(__file__).resolve().parents[1]
DB = HERE / "data" / "store.db"
MAX_TURNS = 6

SYSTEM = """You are a data analyst agent for an online coffee and tea store. Answer the question by querying the store's SQLite database with the run_sql tool, then call submit_answer with the final answer.

Today is {today}.

Tables:
customers(id, name, country, signup_date)
products(id, name, category, price)
orders(id, customer_id, order_date, status)
order_items(order_id, product_id, quantity, unit_price, discount)

Business rules:
- Revenue is quantity * unit_price * (1 - discount), counted on completed orders only.
- Give money rounded to 2 decimals, percents rounded to 1 decimal, and product names exactly as stored."""

TOOLS = [
    {"type": "function", "function": {
        "name": "run_sql", "description": "Run one read-only SQLite query on the store database and get up to 20 rows back.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "submit_answer", "description": "Submit the final answer to the question. This ends the task.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}},
]


def run_sql(query: str) -> str:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        cur = con.execute(query)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(21)
        more = len(rows) > 20
        out = json.dumps({"columns": cols, "rows": rows[:20]}, default=str)
        return (out[:3000] + (" ... (more rows)" if more else "")) or "(no rows)"
    except Exception as e:
        return f"Error: {e}"
    finally:
        con.close()


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def check(answer, gold, kind) -> float:
    """The reward: 1.0 if the answer matches the reference answer, else 0.0."""
    if answer is None:
        return 0.0
    s = str(answer).strip()
    if kind == "text":
        return float(str(gold).lower() in s.lower() and len(s) <= len(str(gold)) + 40)
    m = NUM.search(s)
    if not m:
        return 0.0
    try:
        v = float(m.group().replace(",", ""))
    except ValueError:
        return 0.0
    tol = {"money": 0.011, "count": 0.0, "percent": 0.11}[kind]
    return float(abs(v - float(gold)) <= tol)


def messages_for(task):
    return [{"role": "system", "content": SYSTEM.format(today=task["today"])},
            {"role": "user", "content": task["question"]}]


async def play(client, model_name, task, temperature=1.0, max_tokens=4096, on_choice=None):
    """Run one episode. Returns (conversation, answer, reward, metrics).

    on_choice(choice) lets the trainer keep the raw Choice objects (with token logprobs) for ART."""
    msgs = messages_for(task)
    convo = list(msgs)
    answer, sql_errors, queries, out_tokens = None, 0, [], 0
    turn = 0
    for turn in range(1, MAX_TURNS + 1):
        resp = await client.chat.completions.create(model=model_name, messages=msgs, tools=TOOLS,
                                                    temperature=temperature, max_tokens=max_tokens)
        choice = resp.choices[0]
        out_tokens += getattr(resp.usage, "completion_tokens", 0) or 0
        convo.append(on_choice(choice) if on_choice else choice.message.model_dump(exclude_none=True))
        m = choice.message
        msgs.append({"role": "assistant", "content": m.content or "",
                     **({"tool_calls": [tc.model_dump() for tc in m.tool_calls]} if m.tool_calls else {})})
        if not m.tool_calls:
            break
        done = False
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc.function.name == "submit_answer":
                answer = args.get("answer")
                result = "Answer submitted."
                done = True
            elif tc.function.name == "run_sql":
                q = args.get("query", "")
                queries.append(q)
                result = run_sql(q)
                sql_errors += result.startswith("Error")
            else:
                result = f"Error: unknown tool {tc.function.name}"
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            msgs.append(tool_msg)
            convo.append(tool_msg)
        if done:
            break
    reward = check(answer, task["answer"], task["kind"])
    metrics = {"correct": reward, "turns": turn, "sql_queries": len(queries), "sql_errors": sql_errors,
               "submitted": float(answer is not None), "completion_tokens": out_tokens}
    return convo, answer, reward, metrics, queries
