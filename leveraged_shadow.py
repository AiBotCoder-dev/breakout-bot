"""
leveraged_shadow.py — READ-ONLY paper shadow of the leveraged-ETF strategy.

Why this exists
---------------
Six weeks of live options trading and a stack of reviewer-driven tests
(reviewer_tests.py, leveraged_momentum_backtest.py) converged on one conclusion:
the bot's real edge is a WEAK-BUT-REAL directional/momentum tilt, and the
survival-optimal way to express a directional bet is NOT naked calls (a
tail-driven lottery, 41% of trades < -70%) but TREND-FILTERED LEVERAGED SHARES
(0% catastrophic tail, positive Sortino).

WEALTHSIMPLE-REALISTIC (the user copies signals into Wealthsimple with real $):
  Vehicles must be Wealthsimple-tradeable AND priced with real fees. Wealthsimple
  charges 1.5% currency-conversion EACH way on US securities (basic plan), which
  destroys a weekly-rotation strategy (US 3x rotation Sortino 1.12 -> 0.53 with FX).
  So we use CAD-listed 2x ETFs (TSX, NO FX fee): HQU/HSU/HXU + HEU/HFU/HGU. Canada's
  leveraged ETFs are all 2x — which is also the leverage-sweep optimum (3x is past
  the Sortino peak). Real .TO prices already contain the ETF's MER + decay.

The backtest of that idea (wealthsimple_backtest.py, 8y, real fees):
  CAD 2x rotation (WS, no FX)  ->  +2306%  Sortino 1.17  maxDD -51%
  HQU buy&hold                     +525%   Sortino 0.73  maxDD -65%
  QQQ buy&hold                     +289%   Sortino 0.86  maxDD -35%
(Headline % is survivorship-flattered; the Sortino EDGE over the benchmarks and
the tamed drawdown vs -65% for 2x buy-hold are the honest signals. The -51% DD is
real — 2x leverage — and is the risk this shadow exists to make visible.)

What this module does
---------------------
Runs that strategy in PAPER, tracked in its OWN Supabase tables, completely
READ-ONLY with respect to the live options bot and the Alpaca account:

  * It NEVER calls the broker and NEVER places an order.
  * It keeps a virtual $10k book, marks it to market from FREE yfinance prices,
    rebalances weekly into the strongest 3x sector ETFs that are in an uptrend,
    and snapshots its equity curve daily.
  * monitor.py calls step(conn) once per cycle; step() self-gates to run at most
    once per UTC calendar day, so it adds ~one yfinance pull per day and nothing
    to the trading path.

After enough days this gives a LIVE, side-by-side answer to the reviewer's Q2 —
is the leveraged-shares expression actually worth trading vs just holding the
index — without risking a cent of the real book.

Tables (all prefixed lev_shadow_, isolated from everything else):
  lev_shadow_state   one row: cash, holdings JSON, last_step/last_rebalance dates
  lev_shadow_trades  every rebalance BUY/SELL with price + reason
  lev_shadow_equity  daily (date, equity) for the dashboard curve
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, date, timedelta

# CAD-listed 2x ETFs — Wealthsimple-tradeable, NO 1.5% FX fee (they trade in CAD
# on the TSX), and 2x matches the leverage-sweep optimum. yfinance uses .TO.
#   HQU 2x Nasdaq-100 · HSU 2x S&P500 · HXU 2x TSX60 ·
#   HEU 2x energy · HFU 2x financials · HGU 2x gold miners
UNIVERSE = ["HQU.TO", "HSU.TO", "HXU.TO", "HEU.TO", "HFU.TO", "HGU.TO"]
TOP_K = 2                 # equal-weight the top-2 in an uptrend
START_EQUITY = 10_000.0   # virtual book size, CAD (paper; unrelated to real account)
REBAL_DAYS = 7            # weekly rebalance cadence
# Wealthsimple cost: $0 commission, and CAD-listed => NO FX fee. The only real
# per-trade cost is the bid/ask spread (~0.08%/side on a leveraged ETF). The ETF's
# MER + decay are already inside its real .TO price history.
COST_BPS = 8.0            # bid/ask spread per side (Wealthsimple, CAD-listed)
MOM_LOOKBACK = 63         # 3-month momentum, matches the backtest


# ── tables ──────────────────────────────────────────────────────────────────
def _ensure(conn):
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lev_shadow_state (
                id INTEGER PRIMARY KEY,
                cash REAL, holdings TEXT,
                last_step TEXT, last_rebalance TEXT,
                started_at TEXT, start_equity REAL
            )""")
    except Exception:
        pass
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lev_shadow_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, ticker TEXT, action TEXT,
                shares REAL, price REAL, reason TEXT
            )""")
    except Exception:
        pass
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lev_shadow_equity (
                d TEXT PRIMARY KEY, equity REAL, spy REAL, qqq REAL
            )""")
    except Exception:
        pass


def _state(conn):
    """Load the single state row, initialising it on first ever call."""
    _ensure(conn)
    try:
        row = conn.execute("SELECT * FROM lev_shadow_state WHERE id=1").fetchone()
    except Exception:
        row = None
    if row:
        d = dict(row) if hasattr(row, "keys") else {}
        try:
            holdings = json.loads(d.get("holdings") or "{}")
        except Exception:
            holdings = {}
        return {
            "cash": float(d.get("cash") or 0.0),
            "holdings": holdings,                      # {ticker: {shares, entry}}
            "last_step": d.get("last_step"),
            "last_rebalance": d.get("last_rebalance"),
            "started_at": d.get("started_at"),
            "start_equity": float(d.get("start_equity") or START_EQUITY),
        }
    # first run — all cash
    now = datetime.now(timezone.utc).isoformat()
    st = {"cash": START_EQUITY, "holdings": {}, "last_step": None,
          "last_rebalance": None, "started_at": now, "start_equity": START_EQUITY}
    try:
        conn.execute(
            "INSERT INTO lev_shadow_state (id, cash, holdings, last_step, "
            "last_rebalance, started_at, start_equity) VALUES (1,?,?,?,?,?,?) "
            "ON CONFLICT (id) DO NOTHING",
            (st["cash"], json.dumps(st["holdings"]), None, None, now, START_EQUITY))
    except Exception:
        pass
    return st


def _save_state(conn, st):
    try:
        conn.execute(
            "UPDATE lev_shadow_state SET cash=?, holdings=?, last_step=?, "
            "last_rebalance=? WHERE id=1",
            (float(st["cash"]), json.dumps(st["holdings"]),
             st.get("last_step"), st.get("last_rebalance")))
    except Exception:
        pass


# ── market data (free yfinance only) ─────────────────────────────────────────
def _history(tickers, days=420):
    """Download recent daily closes for a set of tickers. Returns {t: DataFrame}
    with Close/sma50/sma200/mom columns. Never raises."""
    out = {}
    try:
        import yfinance as yf
        import numpy as np  # noqa: F401
        import pandas as pd
        end = datetime.now()
        start = end - timedelta(days=days)
        raw = yf.download(list(tickers), start=start, end=end, progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
        for t in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw[t].dropna(subset=["Close"]).copy()
                else:
                    df = raw.dropna(subset=["Close"]).copy()
                if df is None or df.empty or len(df) < 210:
                    continue
                c = df["Close"]
                df["sma50"] = c.rolling(50).mean()
                df["sma200"] = c.rolling(200).mean()
                df["mom"] = c / c.shift(MOM_LOOKBACK) - 1
                out[t] = df
            except Exception:
                continue
    except Exception:
        pass
    return out


def _last(df, col="Close"):
    try:
        v = float(df[col].iloc[-1])
        return v if v == v else None      # NaN guard
    except Exception:
        return None


# ── the daily step ────────────────────────────────────────────────────────────
def step(conn, force=False, notify=None):
    """Advance the shadow at most once per UTC day. READ-ONLY: never touches the
    broker. Returns a short summary dict on the day it runs, else None.

    `notify` — optional callable(str) for a once-in-a-while Telegram line.
    """
    st = _state(conn)
    today = date.today().isoformat()
    if not force and st.get("last_step") == today:
        return None                       # already ran today

    # Need history for the union of current holdings + universe (rebalance may
    # rotate into any of them) + the broad-gate indices. One pull per day is cheap.
    tickers = sorted(set(UNIVERSE) | set(st["holdings"].keys())
                     | {"SPY", "QQQ", "XIU.TO"})
    hist = _history(tickers)
    if not hist:
        return None                       # data outage; try again next cycle

    # 1) mark current holdings to market
    equity = st["cash"]
    for t, pos in list(st["holdings"].items()):
        px = _last(hist[t]) if t in hist else None
        if px is None:
            continue
        equity += float(pos.get("shares") or 0) * px

    # 2) BROAD-GATE AIRBAG (reviewer-validated). The per-ETF filter misses the
    # correlation-spike / late-cycle case where "strong" sectors gap down together
    # (17% of weeks were divergent; 4 of the 5 worst weeks were gate-in-cash). So
    # require SPY AND XIU both above their 200 SMA to hold ANYTHING. Checked DAILY,
    # not just on rebalance days, so a mid-week regime break exits immediately.
    # Scenario C in broad_gate_test.py: maxDD -50% -> -23%, Calmar 1.20 -> 2.05.
    broad = _broad_long(hist)
    rebalanced = False
    if not broad:
        if st["holdings"]:                # airbag: liquidate to cash right now
            rebalanced = _liquidate(conn, st, hist, "broad_gate_risk_off")
    else:
        # 3) rebalance if due (weekly, or if we hold nothing yet)
        due = force or st.get("last_rebalance") is None
        if not due and st.get("last_rebalance"):
            try:
                due = (date.today() - date.fromisoformat(st["last_rebalance"])).days >= REBAL_DAYS
            except Exception:
                due = True
        if due:
            rebalanced = _rebalance(conn, st, hist, equity)

    # 3) recompute equity post-rebalance and snapshot the curve
    equity = st["cash"]
    for t, pos in list(st["holdings"].items()):
        px = _last(hist[t]) if t in hist else None
        if px is None:
            continue
        equity += float(pos.get("shares") or 0) * px

    spy_px = _last(hist["SPY"]) if "SPY" in hist else None
    qqq_px = _last(hist["QQQ"]) if "QQQ" in hist else None
    try:
        conn.execute(
            "INSERT INTO lev_shadow_equity (d, equity, spy, qqq) VALUES (?,?,?,?) "
            "ON CONFLICT (d) DO UPDATE SET equity=EXCLUDED.equity, "
            "spy=EXCLUDED.spy, qqq=EXCLUDED.qqq",
            (today, round(equity, 2), spy_px, qqq_px))
    except Exception:
        pass

    st["last_step"] = today
    _save_state(conn, st)

    holdings_txt = ", ".join(sorted(st["holdings"].keys())) or "CASH"
    ret = (equity / st["start_equity"] - 1) * 100
    summary = {"equity": round(equity, 2), "return_pct": round(ret, 1),
               "holdings": sorted(st["holdings"].keys()), "rebalanced": rebalanced,
               "broad_long": broad}
    if rebalanced and notify:
        try:
            gate = "" if broad else "\n🛡️ Broad gate RISK-OFF — forced to cash."
            notify(f"🧪 <b>Leveraged shadow rebalance</b> (paper, read-only)\n"
                   f"Now holding: {holdings_txt}\n"
                   f"Shadow equity: ${equity:,.0f} ({ret:+.1f}% since start){gate}")
        except Exception:
            pass
    return summary


def _broad_long(hist) -> bool:
    """Broad-index risk gate: LONG only when BOTH SPY and XIU (TSX 60) are above
    their 200 SMA. Risk-off if EITHER breaks (the conservative airbag). Fails
    SAFE — if the gate data is missing, return False (go to cash)."""
    ok = True
    for idx in ("SPY", "XIU.TO"):
        df = hist.get(idx)
        if df is None:
            return False
        px = _last(df); s200 = _last(df, "sma200")
        if px is None or s200 is None:
            return False
        ok = ok and (px > s200)
    return ok


def _liquidate(conn, st, hist, reason):
    """Sell every holding to cash at current marks and log it. Used by the broad-
    gate airbag. Mutates st. Returns True if anything was sold."""
    now = datetime.now(timezone.utc).isoformat()
    sold = False
    for t in list(st["holdings"].keys()):
        px = _last(hist[t]) if t in hist else None
        pos = st["holdings"].pop(t)
        sh = float(pos.get("shares") or 0)
        if px is None or sh <= 0:
            continue
        st["cash"] += sh * px * (1 - COST_BPS / 10_000)
        try:
            conn.execute(
                "INSERT INTO lev_shadow_trades (ts, ticker, action, shares, "
                "price, reason) VALUES (?,?,?,?,?,?)",
                (now, t, "SELL", round(sh, 4), round(float(px), 4), reason))
        except Exception:
            pass
        sold = True
    return sold


def _rebalance(conn, st, hist, equity):
    """Rank the universe, pick the top-K in an uptrend, move the book to an
    equal-weight of them (or cash). Mutates st in place. Returns True if it ran."""
    # rank uptrending names by 63d momentum
    ranked = []
    for t in UNIVERSE:
        df = hist.get(t)
        if df is None:
            continue
        px = _last(df); s50 = _last(df, "sma50"); s200 = _last(df, "sma200")
        mom = _last(df, "mom")
        if None in (px, s50, s200, mom):
            continue
        if px > s50 and px > s200:          # uptrend filter tames 3x decay
            ranked.append((mom, t, px))
    ranked.sort(reverse=True)
    target = {t: px for _, t, px in ranked[:TOP_K]}   # {ticker: price}

    now = datetime.now(timezone.utc).isoformat()

    def _log(ticker, action, shares, price, reason):
        try:
            conn.execute(
                "INSERT INTO lev_shadow_trades (ts, ticker, action, shares, "
                "price, reason) VALUES (?,?,?,?,?,?)",
                (now, ticker, action, round(float(shares), 4),
                 round(float(price), 4), reason))
        except Exception:
            pass

    # SELL anything not in the new target (mark to current price -> cash)
    for t in list(st["holdings"].keys()):
        if t in target:
            continue
        px = _last(hist[t]) if t in hist else None
        pos = st["holdings"].pop(t)
        sh = float(pos.get("shares") or 0)
        if px is None or sh <= 0:
            continue
        proceeds = sh * px
        st["cash"] += proceeds * (1 - COST_BPS / 10_000)   # friction
        _log(t, "SELL", sh, px, "rotate_out")

    # If nothing qualifies, sit in cash.
    if not target:
        st["last_rebalance"] = date.today().isoformat()
        return True

    # BUY / rebalance into equal weight of the target set
    per_name = equity / len(target)
    for t, px in target.items():
        if px <= 0:
            continue
        want_shares = per_name / px
        held = float(st["holdings"].get(t, {}).get("shares") or 0)
        delta = want_shares - held
        if abs(delta) * px < 1.0:            # ignore sub-$1 rebalances
            # keep existing position, just refresh entry mark
            st["holdings"][t] = {"shares": held, "entry": px}
            continue
        notional = abs(delta) * px
        st["cash"] -= (delta * px) + notional * (COST_BPS / 10_000)
        st["holdings"][t] = {"shares": want_shares, "entry": px}
        _log(t, "BUY" if delta > 0 else "TRIM", abs(delta), px, "rotate_in")

    st["last_rebalance"] = date.today().isoformat()
    return True


# ── reporting (dashboard / telegram) ──────────────────────────────────────────
def report(conn) -> dict:
    """Current shadow status + performance vs SPY/QQQ since inception."""
    _ensure(conn)
    st = _state(conn)
    try:
        rows = conn.execute(
            "SELECT d, equity, spy, qqq FROM lev_shadow_equity ORDER BY d").fetchall()
        curve = [dict(r) if hasattr(r, "keys") else
                 {"d": r[0], "equity": r[1], "spy": r[2], "qqq": r[3]} for r in rows]
    except Exception:
        curve = []

    out = {"holdings": sorted(st["holdings"].keys()) or ["CASH"],
           "cash": round(st["cash"], 2),
           "start_equity": st["start_equity"],
           "started_at": st.get("started_at"),
           "last_rebalance": st.get("last_rebalance"),
           "n_days": len(curve)}

    if curve:
        eq0 = st["start_equity"]
        eq1 = float(curve[-1]["equity"] or eq0)
        out["equity"] = round(eq1, 2)
        out["return_pct"] = round((eq1 / eq0 - 1) * 100, 1)
        # benchmark returns over the SAME observed window
        def _bench(key):
            vs = [c[key] for c in curve if c.get(key)]
            if len(vs) >= 2 and vs[0]:
                return round((float(vs[-1]) / float(vs[0]) - 1) * 100, 1)
            return None
        out["spy_pct"] = _bench("spy")
        out["qqq_pct"] = _bench("qqq")
        # max drawdown of the shadow curve
        peak = eq0; mdd = 0.0
        for c in curve:
            e = float(c["equity"] or eq0)
            peak = max(peak, e)
            mdd = min(mdd, (e - peak) / peak)
        out["max_dd_pct"] = round(mdd * 100, 1)
    else:
        out["equity"] = st["start_equity"]
        out["return_pct"] = 0.0
    return out


if __name__ == "__main__":
    # Local smoke test against the live DB (still read-only w.r.t. the broker).
    import os
    try:
        import psycopg2
        url = os.environ.get("DATABASE_URL", "").strip()
        if not url:
            raise SystemExit("Set DATABASE_URL to smoke-test against Supabase.")
        raw = psycopg2.connect(url, sslmode="require")
        raw.autocommit = True
        # minimal sqlite-ish shim so this file runs standalone
        import monitor
        conn = monitor.PgAdapter(raw)
        s = step(conn, force=True)
        print("step:", s)
        print("report:", json.dumps(report(conn), indent=2, default=str))
    except SystemExit:
        raise
    except Exception as e:
        print("smoke test error:", e)
