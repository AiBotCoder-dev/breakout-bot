"""
alpaca_today.py - report every trade FILLED today on the Alpaca paper account, with stats.

WHY THIS EXISTS
---------------
"Show me what the bot did today." This pulls the actual fills from Alpaca (ground truth,
not the journal's estimates) for the current US/Eastern market day and prints:
  - every fill (symbol, side, qty, avg price, notional, time)
  - per-symbol round-trips + realized P&L (FIFO matched within the day)
  - day totals (buys/sells $, net, # trades) and the account's today P&L (equity - last_equity)

It reuses broker.py's AlpacaPaperBroker for auth, so it needs the same env vars the bot uses:
    ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET   (or ALPACA_API_KEY / ALPACA_API_SECRET)

RUN
---
    python alpaca_today.py            # today (US/Eastern)
    python alpaca_today.py 2026-06-15 # a specific date
"""
from __future__ import annotations

import os
import sys
from collections import deque, defaultdict
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def _load_dotenv():
    """Load KEY=VALUE lines from a local .env (gitignored) into os.environ.

    Lets you keep Alpaca keys out of the shell history / this chat: drop them in
    a .env next to this file and they never get committed (see .gitignore).
    """
    env = Path(__file__).with_name(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
from broker import AlpacaPaperBroker  # noqa: E402  (after .env is loaded)

ET = ZoneInfo("America/New_York")


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _mult(symbol: str) -> float:
    # OCC option symbols are long (e.g. AAPL260116C00150000); contracts are x100.
    return 100.0 if len(symbol or "") > 8 else 1.0


def fetch_fills(b: AlpacaPaperBroker, day_et) -> list[dict]:
    """All filled orders whose fill timestamp lands on `day_et` (US/Eastern)."""
    start_utc = datetime.combine(day_et, time(0, 0), ET).astimezone(timezone.utc)
    end_utc = datetime.combine(day_et, time(23, 59, 59), ET).astimezone(timezone.utc)
    rows = b._get("/v2/orders", {
        "status": "closed",
        "after": start_utc.isoformat(),
        "until": end_utc.isoformat(),
        "limit": 500,
        "direction": "asc",
    }) or []
    fills = []
    for o in rows:
        if str(o.get("status")) != "filled":
            continue
        fq = _f(o.get("filled_qty"))
        fp = _f(o.get("filled_avg_price"))
        if fq <= 0 or fp <= 0:
            continue
        ts = o.get("filled_at") or o.get("submitted_at") or ""
        when = ""
        if ts:
            try:
                when = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ET).strftime("%H:%M:%S")
            except ValueError:
                when = ts[:19]
        sym = o.get("symbol", "")
        fills.append({
            "sym": sym, "side": o.get("side"), "qty": fq, "price": fp,
            "mult": _mult(sym), "notional": fp * fq * _mult(sym), "time": when,
        })
    return fills


def fifo_realized(fills: list[dict]) -> dict:
    """Per-symbol FIFO match of today's fills -> realized P&L on round-tripped qty."""
    lots: dict[str, deque] = defaultdict(deque)   # open buy lots (price, qty)
    short: dict[str, deque] = defaultdict(deque)  # open sell lots (price, qty)
    realized: dict[str, float] = defaultdict(float)
    for fl in fills:
        sym, qty, price, mult = fl["sym"], fl["qty"], fl["price"], fl["mult"]
        if fl["side"] == "buy":
            q = qty
            while q > 0 and short[sym]:                 # cover existing shorts first
                sp, sq = short[sym][0]
                m = min(q, sq)
                realized[sym] += (sp - price) * m * mult
                q -= m
                if m >= sq:
                    short[sym].popleft()
                else:
                    short[sym][0] = (sp, sq - m)
            if q > 0:
                lots[sym].append((price, q))
        else:  # sell
            q = qty
            while q > 0 and lots[sym]:                  # close existing longs first
                bp, bq = lots[sym][0]
                m = min(q, bq)
                realized[sym] += (price - bp) * m * mult
                q -= m
                if m >= bq:
                    lots[sym].popleft()
                else:
                    lots[sym][0] = (bp, bq - m)
            if q > 0:
                short[sym].append((price, q))
    open_long = {s: sum(q for _, q in d) for s, d in lots.items() if sum(q for _, q in d)}
    open_short = {s: sum(q for _, q in d) for s, d in short.items() if sum(q for _, q in d)}
    return {"realized": dict(realized), "open_long": open_long, "open_short": open_short}


def run(day_et=None):
    b = AlpacaPaperBroker()
    if not b.available():
        print("No Alpaca paper credentials found in this environment.\n"
              "Set ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET (or ALPACA_API_KEY/SECRET),\n"
              "then re-run. This must run on the machine/account that places the trades.")
        return

    if day_et is None:
        day_et = datetime.now(ET).date()

    print(f"ALPACA PAPER - trades filled on {day_et} (US/Eastern)\n")
    fills = fetch_fills(b, day_et)
    if not fills:
        print("  No fills today.")
    else:
        print(f"  {'time':>8} | {'side':<4} | {'symbol':<22} | {'qty':>6} | "
              f"{'avg px':>10} | {'notional':>12}")
        print("  " + "-" * 78)
        for fl in fills:
            print(f"  {fl['time']:>8} | {fl['side']:<4} | {fl['sym']:<22} | "
                  f"{fl['qty']:>6.0f} | {fl['price']:>10.2f} | {fl['notional']:>12,.2f}")

    # --- aggregate stats ---
    buys = [f for f in fills if f["side"] == "buy"]
    sells = [f for f in fills if f["side"] == "sell"]
    bought = sum(f["notional"] for f in buys)
    sold = sum(f["notional"] for f in sells)
    syms = sorted({f["sym"] for f in fills})
    fr = fifo_realized(fills)
    day_realized = sum(fr["realized"].values())

    print("\n" + "=" * 70)
    print(f" DAY STATS - {day_et}")
    print("=" * 70)
    print(f"  Fills            : {len(fills)}  ({len(buys)} buys / {len(sells)} sells)")
    print(f"  Symbols traded   : {len(syms)}  {', '.join(syms) if syms else ''}")
    print(f"  Bought / Sold    : ${bought:,.2f} / ${sold:,.2f}")
    print(f"  Net cash flow    : ${sold - bought:+,.2f}  (sells minus buys)")
    print(f"  Realized P&L day : ${day_realized:+,.2f}  (FIFO on round-trips closed today)")

    if fr["realized"]:
        print("\n  Per-symbol realized (closed round-trips today):")
        for sym, pnl in sorted(fr["realized"].items(), key=lambda kv: kv[1]):
            print(f"    {sym:<22} ${pnl:+,.2f}")
    if fr["open_long"] or fr["open_short"]:
        print("\n  Still open from today's fills (not yet round-tripped):")
        for sym, q in fr["open_long"].items():
            print(f"    {sym:<22} long  {q:.0f}")
        for sym, q in fr["open_short"].items():
            print(f"    {sym:<22} short {q:.0f}")

    # --- account snapshot (today's total P&L incl. unrealized) ---
    try:
        a = b._get("/v2/account")
        equity = _f(a.get("equity"))
        last_equity = _f(a.get("last_equity"))
        cash = _f(a.get("cash"))
        print("\n  --- account snapshot ---")
        print(f"  Equity           : ${equity:,.2f}   Cash: ${cash:,.2f}")
        print(f"  Today total P&L  : ${equity - last_equity:+,.2f}  "
              f"(equity - last_equity; includes unrealized on open positions)")
    except Exception as e:
        print(f"\n  (account snapshot unavailable: {e})")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    day = datetime.strptime(arg, "%Y-%m-%d").date() if arg else None
    run(day)
