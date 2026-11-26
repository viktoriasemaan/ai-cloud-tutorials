#!/usr/bin/env python3
"""Step 1 of the fine-tuning tutorial: get more market days to learn from.

The market agent's pre-check uses Alpha Vantage's free plan, which returns the last 100 trading days. That gave
38 days with news, which is enough to compare models but too few to train one. This script downloads about two years
of daily closing prices for the watchlist and the S&P 500 from Yahoo Finance's public chart data (free, no key) and
saves them next to the agent's cache. The pre-check then uses them for any day older than its own 100-day window.

  python3 scripts/fetch_history.py                 2 years for every symbol in the watchlist + SPY
  python3 scripts/fetch_history.py --years 1

Saves: ../Market Watch Agent Tutorial/data/cache/history/<SYMBOL>.json  ({"source": ..., "closes": {date: close}})
"""
import argparse, datetime as dt, json, pathlib, sys, time, urllib.request

AGENT = pathlib.Path(__file__).resolve().parents[2] / "Market Watch Agent Tutorial"
OUT = AGENT / "data" / "cache" / "history"
UA = {"User-Agent": "Mozilla/5.0 (tutorial data download)"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def yahoo(symbol, start):
    """Daily closes from Yahoo Finance's chart endpoint (dates in New York time)."""
    t0 = int(dt.datetime.fromisoformat(start).timestamp())
    d = json.loads(get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={t0}&period2={int(time.time())}&interval=1d"))
    res = d["chart"]["result"][0]
    days = [dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat() for t in res["timestamp"]]
    closes = res["indicators"]["quote"][0]["close"]
    return {day: round(c, 4) for day, c in zip(days, closes) if c is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2)
    a = ap.parse_args()
    cfg = json.loads((AGENT / "config" / "watchlist.json").read_text())
    symbols = [cfg["baseline"]["symbol"]] + [c["symbol"] for c in cfg["companies"]]
    start = (dt.date.today() - dt.timedelta(days=int(a.years * 365))).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in symbols:
        try:
            closes = yahoo(s, start)
        except Exception as e:
            sys.exit(f"{s}: no history ({e})")
        (OUT / f"{s}.json").write_text(json.dumps({"source": "Yahoo Finance", "closes": closes}))
        print(f"{s:6} {len(closes):4} trading days  {min(closes)} to {max(closes)}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
