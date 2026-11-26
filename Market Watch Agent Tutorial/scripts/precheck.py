#!/usr/bin/env python3
"""The pre-check: decide, without any model, whether anything happened on the watchlist.

It runs on every scheduler tick and costs nothing but a few free API calls. If nothing happened, it prints
{"wakeAgent": false} as its last line and Hermes skips the model for that tick. If something did, it prints the
evidence as JSON for the agent to read.

  python precheck.py                  live: the latest trading day, and new filings since the last run
  python precheck.py --date 2026-10-02   replay: that day's price moves and filings (no state is changed)

Data (both free):
  prices   Alpha Vantage TIME_SERIES_DAILY (key: ALPHAVANTAGE_API_KEY or ~/.config/alphavantage/api_key).
           Cached per symbol per day, so a day costs at most 11 of the free plan's 25 calls.
  filings  SEC EDGAR submissions (User-Agent: SEC_USER_AGENT or ~/.config/sec/user_agent, "Name email").

A price move counts when the stock's return minus the index's return is at least max(2.0 points, 2 standard
deviations of that difference over the previous 60 trading days). Routine filings (insider-trade forms 3/4/5/144
and similar, listed in config/watchlist.json) are counted and skipped.
"""
import argparse, datetime as dt, json, os, pathlib, statistics, sys, time, urllib.parse, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "watchlist.json").read_text())
DATA = ROOT / "data"
CACHE = DATA / "cache"


def secret(env, path):
    v = os.environ.get(env, "").strip()
    if v:
        return v
    p = pathlib.Path(path).expanduser()
    if p.exists():
        return p.read_text().strip()
    sys.exit(f"missing {env} (or {path})")


def get_json(url, headers=None, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "market-agent-tutorial"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def daily_closes(symbol, today):
    """{date: close}: the ~100 recent trading days, plus older days from data/cache/history if that exists
    (the fine-tuning tutorial downloads it with fetch_history.py). Recent prices always win."""
    closes = _recent_closes(symbol, today)
    h = CACHE / "history" / f"{symbol}.json"
    if h.exists():
        first = min(closes)
        older = {d: c for d, c in json.loads(h.read_text())["closes"].items() if d < first}
        closes = {**older, **closes}
    return closes


def _recent_closes(symbol, today):
    """{date: close} for ~100 trading days, cached once per calendar day."""
    f = CACHE / "prices" / f"{symbol}.json"
    if f.exists():
        c = json.loads(f.read_text())
        # The free plan allows 25 calls a day, so refresh at most twice: once a day, and once after the close
        # (4:45 pm New York time) to pick up that day's closing price.
        ny = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5)  # EST; good enough for a once-a-day gate
        after_close = ny.hour * 60 + ny.minute >= 16 * 60 + 45 and ny.weekday() < 5
        close_cutoff = ny.replace(hour=16, minute=45, second=0, microsecond=0).timestamp() + 5 * 3600
        has_today_close = max(c["closes"]) >= ny.date().isoformat()
        stale_after_close = after_close and not has_today_close and c.get("fetched_at", 0) < close_cutoff
        if c.get("fetched_on") == today and not stale_after_close:
            return c["closes"]
    key = secret("ALPHAVANTAGE_API_KEY", "~/.config/alphavantage/api_key")
    q = urllib.parse.urlencode({"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "compact", "apikey": key})
    for attempt in range(3):
        time.sleep(1.5)  # the free plan allows 1 request per second
        d = get_json("https://www.alphavantage.co/query?" + q)
        ts = d.get("Time Series (Daily)")
        if ts or "per second" not in str(d.get("Information", "")):
            break
        time.sleep(3)
    if not ts:
        msg = d.get("Note") or d.get("Information") or d.get("Error Message") or str(d)[:200]
        if f.exists():  # rate-limited: fall back to the cache, and don't retry until tomorrow
            print(f"note: {symbol} prices not refreshed ({msg[:80]})", file=sys.stderr)
            c = json.loads(f.read_text())
            c.update(fetched_on=today, fetched_at=time.time(), refresh_failed=msg[:120])
            f.write_text(json.dumps(c))
            return c["closes"]
        sys.exit(f"Alpha Vantage gave no data for {symbol}: {msg}")
    closes = {k: float(v["4. close"]) for k, v in ts.items()}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"fetched_on": today, "fetched_at": time.time(), "closes": closes}))
    return closes


def price_events(day, today):
    rules = CFG["rules"]
    spy = daily_closes(CFG["baseline"]["symbol"], today)
    days = sorted(spy)
    if day not in spy:
        return [], f"{day} is not a trading day in the price data ({days[0]}..{days[-1]})"
    i = days.index(day)
    if i < 21:
        return [], f"not enough history before {day}"
    ret = lambda c, a, b: c[b] / c[a] - 1
    out = []
    for co in CFG["companies"]:
        c = daily_closes(co["symbol"], today)
        if day not in c:
            continue
        hist = [ret(c, days[j - 1], days[j]) - ret(spy, days[j - 1], days[j])
                for j in range(max(1, i - 60), i) if days[j] in c and days[j - 1] in c]
        sigma = statistics.pstdev(hist) * 100 if len(hist) > 10 else 0.0
        stock = ret(c, days[i - 1], day) * 100
        index = ret(spy, days[i - 1], day) * 100
        excess = stock - index
        bar = max(rules["price_move_min_pct"], rules["price_move_sigma"] * sigma)
        if abs(excess) >= bar:
            out.append({"type": "price_move", "symbol": co["symbol"], "company": co["name"], "date": day,
                        "close": round(c[day], 2), "stock_pct": round(stock, 2), "index_pct": round(index, 2),
                        "vs_index_pts": round(excess, 2), "threshold_pts": round(bar, 2),
                        "typical_daily_gap_pts": round(sigma, 2)})
    return out, None


# What 8-K item numbers mean, so the agent can say "earnings" instead of "item 2.02" (SEC Form 8-K instructions).
ITEMS_8K = {
    "1.01": "entered a material agreement", "1.02": "ended a material agreement", "1.03": "bankruptcy",
    "1.05": "material cybersecurity incident", "2.01": "completed an acquisition or sale of assets",
    "2.02": "quarterly or annual results (earnings)", "2.03": "took on a material financial obligation",
    "2.05": "restructuring or exit costs", "2.06": "material impairment", "3.01": "delisting notice",
    "3.02": "unregistered sale of shares", "3.03": "change to shareholder rights", "4.01": "changed auditor",
    "4.02": "earlier financial statements can't be relied on", "5.01": "change in control",
    "5.02": "executive or director change", "5.03": "bylaws or charter amended", "5.07": "shareholder vote results",
    "7.01": "Regulation FD disclosure (often a press release or presentation)", "8.01": "other events",
    "9.01": "financial statements and exhibits attached",
}


def sec_ciks(ua):
    f = CACHE / "sec_tickers.json"
    if not f.exists() or time.time() - f.stat().st_mtime > 7 * 86400:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(get_json("https://www.sec.gov/files/company_tickers.json", {"User-Agent": ua})))
    return {v["ticker"].upper(): v["cik_str"] for v in json.loads(f.read_text()).values()}


def filing_events(day, live, state):
    ua = secret("SEC_USER_AGENT", "~/.config/sec/user_agent")
    ciks = sec_ciks(ua)
    skip = set(CFG["rules"]["routine_forms_skipped"])
    events, skipped = [], {}
    for co in CFG["companies"]:
        cik = ciks.get(co["sec_ticker"].upper())
        if not cik:
            continue
        sf = CACHE / "sec" / f"{cik}.json"
        if live or not sf.exists() or time.time() - sf.stat().st_mtime > 6 * 3600:
            s = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", {"User-Agent": ua})
            sf.parent.mkdir(parents=True, exist_ok=True)
            sf.write_text(json.dumps(s))
            time.sleep(0.15)  # SEC fair access: well under 10 requests a second
        else:
            s = json.loads(sf.read_text())
        r = s["filings"]["recent"]
        seen = set(state.get("seen", {}).get(co["symbol"], []))
        first_run = co["symbol"] not in state.get("seen", {})
        new_seen = []
        for k in range(len(r["form"])):
            form, fdate, acc = r["form"][k], r["filingDate"][k], r["accessionNumber"][k]
            if live:
                if acc in seen:
                    continue
                if fdate < (dt.date.fromisoformat(day) - dt.timedelta(days=3)).isoformat():
                    continue  # older than 3 days: not news any more
                new_seen.append(acc)
                if first_run and fdate < dt.date.today().isoformat():
                    continue  # the first live run records older filings without alerting; today's still count
            elif fdate != day:
                continue
            if form in skip:
                skipped[form] = skipped.get(form, 0) + 1
                continue
            doc = r["primaryDocument"][k]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
            items = r.get("items", [""] * len(r["form"]))[k]
            meaning = [ITEMS_8K[i.strip()] for i in items.split(",") if i.strip() in ITEMS_8K and i.strip() != "9.01"]
            events.append({"type": "filing", "symbol": co["symbol"], "company": co["name"], "date": fdate,
                           "form": form, "items": items, "what_it_is": "; ".join(meaning),
                           "description": r.get("primaryDocDescription", [""] * len(r["form"]))[k], "url": url})
        if live:
            state.setdefault("seen", {})[co["symbol"]] = sorted(seen | set(new_seen))[-400:]
    return events, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="replay a past trading day (YYYY-MM-DD)")
    ap.add_argument("--summary", action="store_true", help="print a readable table instead of JSON")
    a = ap.parse_args()
    today = dt.date.today().isoformat()
    live = a.date is None
    state_f = DATA / "state.json"
    state = json.loads(state_f.read_text()) if (live and state_f.exists()) else {}
    if live:
        day = sorted(daily_closes(CFG["baseline"]["symbol"], today))[-1]
    else:
        day = a.date
    prices, note = price_events(day, today)
    if live and state.get("last_price_day") == day:
        prices = []  # this day's moves were already reported
    filings, skipped = filing_events(day, live, state)
    if live:
        state["last_price_day"] = day
        state["last_run"] = dt.datetime.now().isoformat(timespec="seconds")
        DATA.mkdir(exist_ok=True)
        state_f.write_text(json.dumps(state, indent=1))
    events = prices + filings
    report = {"checked_day": day, "mode": "live" if live else "replay", "events": events,
              "routine_filings_skipped": skipped}
    if note:
        report["note"] = note
    if a.summary:
        print_summary(report)
        return
    print(json.dumps(report, indent=1))
    if not events:
        print(json.dumps({"wakeAgent": False}))


def print_summary(r):
    """The same report, readable: one line per event."""
    print(f"{r['checked_day']} ({r['mode']}): {len(r['events'])} events")
    for e in r["events"]:
        if e["type"] == "price_move":
            print(f"  {e['symbol']:<6} price   {e['stock_pct']:+.2f}% vs S&P 500 {e['index_pct']:+.2f}%  "
                  f"({e['vs_index_pts']:+.2f} pts, bar {e['threshold_pts']:.2f})")
        else:
            what = e.get("what_it_is") or e.get("description") or ""
            print(f"  {e['symbol']:<6} filing  {e['form']:<16} {what[:48]}")
    skipped = sum(r["routine_filings_skipped"].values())
    print(f"  routine filings skipped: {skipped} {dict(r['routine_filings_skipped']) if skipped else ''}")
    if not r["events"]:
        print('  -> {"wakeAgent": false}: the model is not called')


if __name__ == "__main__":
    main()
