#!/usr/bin/env python3
"""Step 1 of the RL tutorial: build the store database the agent will query.

A small online store, generated from a fixed seed so everyone gets the same numbers:
customers, products, orders (with a status) and order items (with a discount).
The data ends the day before "today" (2026-11-15), so "last month" means October 2026.

  python3 scripts/make_store_db.py          writes data/store.db
"""
import datetime as dt, pathlib, random, sqlite3

HERE = pathlib.Path(__file__).resolve().parents[1]
DB = HERE / "data" / "store.db"
TODAY = dt.date(2026, 11, 15)

CATEGORIES = {
    "Coffee": [("House Blend 1kg", 24.0), ("Single Origin Ethiopia", 19.5), ("Decaf Colombia", 17.0), ("Espresso Roast", 21.0),
               ("Cold Brew Pack", 15.5), ("Coffee Sampler", 29.0), ("Dark Roast Sumatra", 20.0), ("Breakfast Blend", 16.5)],
    "Tea": [("Earl Grey", 9.5), ("Sencha Green", 12.0), ("Chamomile", 8.0), ("Masala Chai", 11.0), ("Jasmine Pearls", 18.0),
            ("Rooibos", 8.5), ("Matcha Ceremonial", 32.0)],
    "Equipment": [("Pour-Over Dripper", 28.0), ("Burr Grinder", 129.0), ("Gooseneck Kettle", 69.0), ("French Press", 39.0),
                  ("Espresso Machine", 449.0), ("Milk Frother", 49.0), ("Digital Scale", 35.0), ("Aeropress", 42.0)],
    "Mugs": [("Ceramic Mug", 14.0), ("Travel Tumbler", 26.0), ("Double-Wall Glass", 18.0), ("Espresso Cups Set", 22.0),
             ("Camp Mug", 16.0)],
    "Snacks": [("Almond Biscotti", 7.5), ("Dark Chocolate Bar", 5.0), ("Shortbread Tin", 12.5), ("Granola Bites", 6.0),
               ("Stroopwafels", 8.0), ("Ginger Cookies", 6.5)],
}
COUNTRIES = [("United States", 40), ("Canada", 12), ("United Kingdom", 12), ("Germany", 10), ("France", 8), ("Australia", 8),
             ("Japan", 5), ("Brazil", 5)]
FIRST = ["Ava", "Liam", "Mia", "Noah", "Emma", "Leo", "Sofia", "Lucas", "Chloe", "Ethan", "Nora", "Mateo", "Zoe", "Kai",
         "Ivy", "Owen", "Lena", "Hugo", "Maya", "Felix", "Aria", "Theo", "Yuki", "Rafael"]
LAST = ["Smith", "Garcia", "Chen", "Muller", "Martin", "Brown", "Silva", "Tanaka", "Wilson", "Dubois", "Kim", "Taylor",
        "Rossi", "Novak", "Lopez", "Evans", "Sato", "Clark"]


def main():
    rnd = random.Random(7)
    DB.parent.mkdir(exist_ok=True)
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT, signup_date TEXT);
    CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), order_date TEXT, status TEXT);
    CREATE TABLE order_items (order_id INTEGER REFERENCES orders(id), product_id INTEGER REFERENCES products(id),
                              quantity INTEGER, unit_price REAL, discount REAL);
    """)
    products = []
    for cat, items in CATEGORIES.items():
        for name, price in items:
            products.append((len(products) + 1, name, cat, price))
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    start = dt.date(2025, 1, 1)
    days = (TODAY - start).days
    countries = [c for c, w in COUNTRIES for _ in range(w)]
    customers = []
    for i in range(1, 521):
        signup = start + dt.timedelta(days=int(rnd.random() ** 0.8 * (days - 1)))
        customers.append((i, f"{rnd.choice(FIRST)} {rnd.choice(LAST)}", rnd.choice(countries), signup.isoformat()))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)

    orders, items = [], []
    for d in range(days):
        day = start + dt.timedelta(days=d)
        season = 1.6 if day.month in (11, 12) else 1.0
        growth = 1 + d / days
        for _ in range(int(rnd.gauss(7 * season * growth, 2)) if rnd.random() > 0.02 else 0):
            eligible = [c for c in customers if c[3] <= day.isoformat()]
            if not eligible:
                continue
            cust = rnd.choice(eligible)
            ts = dt.datetime.combine(day, dt.time()) + dt.timedelta(seconds=rnd.randint(0, 86399))
            r = rnd.random()
            recent = (TODAY - day).days <= 4
            status = ("pending" if recent and r < 0.5 else
                      "refunded" if r < 0.07 else "cancelled" if r < 0.12 else "completed")
            oid = len(orders) + 1
            orders.append((oid, cust[0], ts.strftime("%Y-%m-%d %H:%M:%S"), status))
            for pid in rnd.sample(range(1, len(products) + 1), rnd.choice([1, 1, 1, 2, 2, 3, 4])):
                p = products[pid - 1]
                qty = rnd.choice([1, 1, 1, 1, 2, 2, 3]) if p[2] != "Equipment" else 1
                items.append((oid, pid, qty, p[3], rnd.choice([0, 0, 0, 0, 0.1, 0.1, 0.2])))
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
    con.commit()
    for t in ("customers", "products", "orders", "order_items"):
        print(f"{t:12} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>6,} rows")
    print("orders from", con.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone())
    print(f"wrote {DB.relative_to(HERE)} ({DB.stat().st_size / 1e6:.1f} MB) · today is {TODAY}")


if __name__ == "__main__":
    main()
