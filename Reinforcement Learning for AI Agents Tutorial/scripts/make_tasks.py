#!/usr/bin/env python3
"""Step 2 of the RL tutorial: write the tasks the agent practices on.

Each task is a business question plus the date the question is asked ("today"). The right answer comes from a
reference SQL query run on data/store.db, so the reward can be checked in code. Nobody writes example answers
for the model: it only ever sees the question, and the check only says right or wrong.

  python3 scripts/make_tasks.py          writes data/tasks_train.jsonl and data/tasks_test.jsonl
"""
import calendar, datetime as dt, json, pathlib, random, sqlite3

HERE = pathlib.Path(__file__).resolve().parents[1]
DB = HERE / "data" / "store.db"
TODAY = dt.date(2026, 11, 15)
REV = "SUM(oi.quantity * oi.unit_price * (1 - oi.discount))"
JOIN = "FROM orders o JOIN order_items oi ON oi.order_id = o.id WHERE o.status = 'completed'"
CATS = ["Coffee", "Tea", "Equipment", "Mugs", "Snacks"]
COUNTRIES = ["United States", "Canada", "United Kingdom", "Germany", "France", "Australia", "Japan", "Brazil"]


def month_bounds(y, m):
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y}-{m:02d}-01", f"{ny}-{nm:02d}-01"


def mname(y, m):
    return f"{calendar.month_name[m]} {y}"


def months(a=(2025, 1), b=(2026, 10)):
    y, m = a
    while (y, m) <= b:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def prev_month(y, m, k=1):
    for _ in range(k):
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return y, m


def candidates(rnd):
    """(template, question, today, reference SQL, answer type)."""
    out = []
    for y, m in months():
        lo, hi = month_bounds(y, m)
        rng = f"AND o.order_date >= '{lo}' AND o.order_date < '{hi}'"
        out.append(("revenue_month", f"What was the total revenue in {mname(y, m)}?", TODAY,
                    f"SELECT ROUND({REV}, 2) {JOIN} {rng}", "money"))
        out.append(("refunded_orders", f"How many orders were refunded in {mname(y, m)}?", TODAY,
                    f"SELECT COUNT(*) FROM orders o WHERE o.status = 'refunded' {rng}", "count"))
        out.append(("top_product", f"Which product brought in the most revenue in {mname(y, m)}?", TODAY,
                    f"SELECT p.name {JOIN.replace('WHERE', 'JOIN products p ON p.id = oi.product_id WHERE')} {rng} "
                    f"GROUP BY p.id ORDER BY {REV} DESC LIMIT 1", "text"))
        out.append(("avg_order_value", f"What was the average order value of completed orders in {mname(y, m)}?", TODAY,
                    f"SELECT ROUND({REV} * 1.0 / COUNT(DISTINCT o.id), 2) {JOIN} {rng}", "money"))
        cat = rnd.choice(CATS)
        out.append(("units_category", f"How many units of {cat} products were sold in {mname(y, m)}?", TODAY,
                    f"SELECT SUM(oi.quantity) {JOIN.replace('WHERE', 'JOIN products p ON p.id = oi.product_id WHERE')} "
                    f"{rng} AND p.category = '{cat}'", "count"))
        out.append(("new_customer_revenue",
                    f"How much revenue came from customers who signed up in {mname(y, m)}, on orders they placed that same month?",
                    TODAY, f"SELECT ROUND(COALESCE({REV}, 0), 2) {JOIN.replace('WHERE', 'JOIN customers c ON c.id = o.customer_id WHERE')} "
                    f"{rng} AND c.signup_date >= '{lo}' AND c.signup_date < '{hi}'", "money"))
    for (y1, m1), (y2, m2) in zip(list(months())[:-1], list(months())[1:]):
        r = lambda y, m: "(SELECT {} {} AND o.order_date >= '{}' AND o.order_date < '{}')".format(REV, JOIN, *month_bounds(y, m))
        out.append(("month_over_month", f"By what percent did revenue change from {mname(y1, m1)} to {mname(y2, m2)}?", TODAY,
                    f"SELECT ROUND(({r(y2, m2)} - {r(y1, m1)}) * 100.0 / {r(y1, m1)}, 1)", "percent"))
    for y in (2025, 2026):
        for q in (1, 2, 3):
            for cat in CATS:
                lo, _ = month_bounds(y, 3 * q - 2)
                _, hi = month_bounds(y, 3 * q)
                out.append(("category_quarter", f"How much revenue did the {cat} category make in Q{q} {y}?", TODAY,
                            f"SELECT ROUND({REV}, 2) {JOIN.replace('WHERE', 'JOIN products p ON p.id = oi.product_id WHERE')} "
                            f"AND o.order_date >= '{lo}' AND o.order_date < '{hi}' AND p.category = '{cat}'", "money"))
        for c in COUNTRIES:
            out.append(("country_customers", f"How many customers from {c} placed at least one completed order in {y}?", TODAY,
                        f"SELECT COUNT(DISTINCT o.customer_id) {JOIN.replace('WHERE', 'JOIN customers c ON c.id = o.customer_id WHERE')} "
                        f"AND o.order_date >= '{y}-01-01' AND o.order_date < '{y + 1}-01-01' AND c.country = '{c}'", "count"))
    # relative dates: the same question asked on different days
    start = dt.date(2025, 2, 3)
    for k in range(0, (TODAY - start).days, 9):
        today = start + dt.timedelta(days=k)
        y, m = prev_month(today.year, today.month)
        lo, hi = month_bounds(y, m)
        out.append(("last_month_revenue", "What was last month's revenue?", today,
                    f"SELECT ROUND({REV}, 2) {JOIN} AND o.order_date >= '{lo}' AND o.order_date < '{hi}'", "money"))
        if today >= dt.date(2025, 4, 1):
            y2, m2 = prev_month(today.year, today.month, 2)
            lo2, hi2 = month_bounds(y2, m2)
            out.append(("last_month_orders", "How many completed orders did we have last month?", today,
                        f"SELECT COUNT(*) FROM orders o WHERE o.status = 'completed' AND o.order_date >= '{lo}' AND o.order_date < '{hi}'", "count"))
            out.append(("two_months_ago_revenue", "What was the revenue two months ago?", today,
                        f"SELECT ROUND({REV}, 2) {JOIN} AND o.order_date >= '{lo2}' AND o.order_date < '{hi2}'", "money"))
        if today >= dt.date(2025, 4, 1):
            q = (today.month - 1) // 3
            qy, qn = (today.year - 1, 4) if q == 0 else (today.year, q)
            lo, _ = month_bounds(qy, 3 * qn - 2)
            _, hi = month_bounds(qy, 3 * qn)
            out.append(("last_quarter_revenue", "What was last quarter's revenue?", today,
                        f"SELECT ROUND({REV}, 2) {JOIN} AND o.order_date >= '{lo}' AND o.order_date < '{hi}'", "money"))
        out.append(("year_to_date_revenue", "What is our revenue so far this year, up to yesterday?", today,
                    f"SELECT ROUND(COALESCE({REV}, 0), 2) {JOIN} AND o.order_date >= '{today.year}-01-01' "
                    f"AND o.order_date < '{today.isoformat()}'", "money"))
    return out


def main():
    rnd = random.Random(11)
    con = sqlite3.connect(DB)
    rows = []
    for tpl, q, today, sql, kind in candidates(rnd):
        ans = con.execute(sql).fetchone()[0]
        if ans is None:
            continue
        rows.append({"template": tpl, "question": q, "today": today.isoformat(), "answer": ans, "kind": kind, "sql": sql})
    by = {}
    for r in rows:
        by.setdefault(r["template"], []).append(r)
    train, test = [], []
    for tpl, rs in sorted(by.items()):
        rnd.shuffle(rs)
        n_test = max(3, round(len(rs) * 0.2))
        test += rs[:n_test][:5]
        train += rs[n_test:][:14]
    rnd.shuffle(train)
    rnd.shuffle(test)
    for i, r in enumerate(train):
        r["id"] = f"train-{i:03d}"
    for i, r in enumerate(test):
        r["id"] = f"test-{i:03d}"
    for name, rs in (("tasks_train.jsonl", train), ("tasks_test.jsonl", test)):
        with open(HERE / "data" / name, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({"id": r["id"], **{k: r[k] for k in r if k != "id"}}) + "\n")
    print(f"{len(rows)} possible tasks in {len(by)} kinds -> {len(train)} for training, {len(test)} held out for testing")
    for tpl in sorted(by):
        print(f"  {tpl:24} train {sum(r['template'] == tpl for r in train):>2} · test {sum(r['template'] == tpl for r in test)}")
    ex = next(r for r in train if r["template"] == "last_month_revenue")
    print(f"\nexample: today {ex['today']} · \"{ex['question']}\" -> {ex['answer']}")


if __name__ == "__main__":
    main()
