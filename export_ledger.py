"""
export_ledger.py — durable, authoritative trade ledger (no log scraping, ever).

Pulls every fill from the Alpaca paper account, FIFO-matches them into closed
round-trips with exact P&L, and writes:
  data/ledger_trades.csv     — one row per closed round-trip (the full blotter)
  data/ledger_open.csv       — current open positions with unrealized P&L
  data/ledger_summary.json   — headline stats (win rate, realized, churn, by-source)

Run daily from the ledger_export workflow (committed back to the repo) so the full
history is always one file-read away. Reads Alpaca keys (and optionally DATABASE_URL
for the journal's per-source attribution) from the environment; writes only the data/
files. SELECT-only on the journal.
"""
import os
import csv
import json
import sys
from datetime import datetime, timezone
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from broker import AlpacaPaperBroker

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)


def occ(s):
    try:
        s = s.strip().upper()
        K = int(s[-8:]) / 1000
        ty = "C" if s[-9] == "C" else "P"
        ymd = s[-15:-9]
        und = s[:-15]
        return und, K, ty, f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}"
    except Exception:
        return s, None, None, None


def main():
    b = AlpacaPaperBroker()
    t = b.test_connection()
    if not t.get("ok"):
        print("export_ledger: broker not connected —", t.get("error"))
        return 1
    acct = b.get_account()

    orders = b._get("/v2/orders", {"status": "closed", "limit": 500,
                                   "direction": "asc"})
    fills = defaultdict(list)
    for o in orders:
        if str(o.get("status")) != "filled":
            continue
        fp = float(o.get("filled_avg_price", 0) or 0)
        fq = float(o.get("filled_qty", 0) or 0)
        if fp <= 0 or fq <= 0:
            continue
        ts = o.get("filled_at") or o.get("submitted_at") or ""
        fills[o.get("symbol", "")].append((ts, o.get("side"), fq, fp))

    trades = []
    for sym, fs in fills.items():
        fs.sort(key=lambda x: x[0])
        mult = 100 if len(sym) > 8 else 1
        longs = deque()
        for ts, side, qty, px in fs:
            if side == "buy":
                longs.append([ts, qty, px])
            else:
                remain = qty
                while remain > 1e-9 and longs:
                    lot = longs[0]
                    m = min(remain, lot[1])
                    und, K, ty, exp = occ(sym)
                    trades.append({
                        "open": lot[0][:19], "close": ts[:19], "symbol": sym,
                        "underlying": und, "strike": K, "type": ty, "expiry": exp,
                        "qty": m, "entry": round(lot[2], 2), "exit": round(px, 2),
                        "pnl": round((px - lot[2]) * m * mult, 2),
                    })
                    lot[1] -= m
                    remain -= m
                    if lot[1] <= 1e-9:
                        longs.popleft()
    trades.sort(key=lambda x: x["close"])

    # write the full blotter
    with open(os.path.join(DATA, "ledger_trades.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["open", "close", "symbol", "underlying",
                                          "strike", "type", "expiry", "qty",
                                          "entry", "exit", "pnl"])
        w.writeheader()
        w.writerows(trades)

    # open positions
    pos = b.get_option_positions()
    with open(os.path.join(DATA, "ledger_open.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "qty", "avg_entry", "current", "unrealized_pct",
                    "unrealized_pnl"])
        for p in pos:
            w.writerow([p["symbol"], p["qty"], p["avg_entry"], p["current"],
                        p["unrealized_pct"], round(float(p["unrealized_pnl"]), 2)])

    wins = [x for x in trades if x["pnl"] > 0]
    losses = [x for x in trades if x["pnl"] <= 0]
    n = len(trades)
    tot = round(sum(x["pnl"] for x in trades), 2)

    # journal per-source attribution (optional, SELECT-only)
    by_source = {}
    if os.environ.get("DATABASE_URL"):
        try:
            import psycopg2
            import psycopg2.extras
            cx = psycopg2.connect(os.environ["DATABASE_URL"], sslmode="require")
            cx.set_session(readonly=True)
            cur = cx.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT setup, COUNT(*) n, "
                        "SUM(CASE WHEN pnl_pct>0 THEN 1 ELSE 0 END) wins, "
                        "ROUND(SUM(pnl_dollars)::numeric,0) pnl "
                        "FROM broker_trade_journal WHERE status='CLOSED' "
                        "AND pnl_pct IS NOT NULL GROUP BY setup ORDER BY n DESC")
            for r in cur.fetchall():
                by_source[r["setup"]] = {
                    "n": int(r["n"]), "wins": int(r["wins"] or 0),
                    "win_rate": round((r["wins"] or 0) / r["n"] * 100, 0) if r["n"] else 0,
                    "pnl": float(r["pnl"] or 0)}
            cx.close()
        except Exception as e:
            print("export_ledger: journal pull skipped —", str(e)[:80])

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "equity": round(float(acct.get("equity", 0) or 0), 2),
        "cash": round(float(acct.get("cash", 0) or 0), 2),
        "closed_round_trips": n,
        "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / n * 100, 1) if n else 0,
        "realized_pnl": tot,
        "avg_win": round(sum(x["pnl"] for x in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(x["pnl"] for x in losses) / len(losses), 2) if losses else 0,
        "expectancy_per_trade": round(tot / n, 2) if n else 0,
        "open_positions": len(pos),
        "unrealized_pnl": round(sum(float(p.get("unrealized_pnl", 0) or 0) for p in pos), 2),
        "by_source": by_source,
    }
    with open(os.path.join(DATA, "ledger_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"export_ledger: {n} round-trips, win {summary['win_rate_pct']}%, "
          f"realized ${tot:+,.2f}, {len(pos)} open. Wrote data/ledger_*.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
