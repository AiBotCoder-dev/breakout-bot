"""
leveraged_shadow.py — READ-ONLY paper shadow of the leveraged-ETF strategy.

PRICING (rebuilt 2026-08): the tradeable products are CAD-listed 2x ETFs, but
yfinance cannot quote .TO tickers reliably (their feed froze for 3 weeks and the
book silently marked a dead price). So every CAD ETF is now SYNTHESISED as a 2x
daily-rebalanced series off a fresh US underlying — see PROXY / _history(). The
signals still name the CAD ticker you'd buy on Wealthsimple; only the marking
changed. Two of the six mappings are approximations (HEU/HFU track CANADIAN sector
indices) and are flagged as such in report().

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
# on the TSX), and 2x matches the leverage-sweep optimum.
#   HQU 2x Nasdaq-100 · HSU 2x S&P500 · HXU 2x TSX60 ·
#   HEU 2x energy · HFU 2x financials · HGU 2x gold miners
UNIVERSE = ["HQU.TO", "HSU.TO", "HXU.TO", "HEU.TO", "HFU.TO", "HGU.TO"]

# ── PRICING VIA US PROXIES (2026-08 rebuild) ────────────────────────────────
# yfinance CANNOT price .TO tickers reliably — their data froze at 2026-07-17 while
# US tickers stayed current, so the shadow silently marked a dead feed for 12 days
# (equity stuck at $9,991.99 to the cent). Fix: never quote .TO directly. Instead
# rebuild each CAD 2x ETF SYNTHETICALLY from a fresh US underlying:
#     synthetic[i] = synthetic[i-1] * (1 + LEV*underlying_daily_return - drag)
# Daily compounding means volatility decay is modelled inherently, and the drag
# models MER + borrow + tracking error that a pure 2x math series would ignore.
#
# MAPPING ACCURACY — be honest about this:
#   HQU -> QQQ  EXACT   (HQU tracks the Nasdaq-100, CAD-hedged)
#   HSU -> SPY  EXACT   (HSU tracks the S&P 500, CAD-hedged)
#   HXU -> EWC  CLOSE   (EWC = MSCI Canada vs HXU's S&P/TSX 60 — highly correlated)
#   HGU -> GDX  CLOSE   (both global/Canadian-heavy gold miners)
#   HEU -> XLE  APPROX  (HEU tracks CANADIAN energy — oil sands/pipelines — vs US energy)
#   HFU -> XLF  APPROX  (HFU tracks CANADIAN banks vs US financials)
# The two APPROX names are correlated but NOT the same index, so their live signals
# are indicative rather than exact. Flagged in report() so it's never mistaken for
# a true quote of the tradeable CAD product.
PROXY = {"HQU.TO": "QQQ", "HSU.TO": "SPY", "HXU.TO": "EWC",
         "HEU.TO": "XLE", "HFU.TO": "XLF", "HGU.TO": "GDX"}
APPROX = {"HEU.TO", "HFU.TO"}      # index mismatch — signals are indicative only
LEV = 2.0                          # the CAD products are 2x daily
SYNTH_ANNUAL_DRAG = 0.015          # ~1.15% MER + borrow/tracking (synthetic flatters)
BROAD_GATE_TICKERS = ("SPY", "EWC")  # US-listed, always fresh (XIU.TO was stale)

TOP_K = 2                 # equal-weight the top-2 in an uptrend
START_EQUITY = 10_000.0   # virtual book size, CAD (paper; unrelated to real account)
REBAL_DAYS = 7            # weekly rebalance cadence
# Wealthsimple cost: $0 commission, and CAD-listed => NO FX fee. The only real
# per-trade cost is the bid/ask spread (~0.08%/side on a leveraged ETF).
COST_BPS = 8.0            # bid/ask spread per side (Wealthsimple, CAD-listed)
MOM_LOOKBACK = 63         # 3-month momentum, matches the backtest
SYNTH_BASE = 100.0        # synthetic series is an INDEX LEVEL, not a CAD quote


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


# ── market data (free yfinance, US tickers only — .TO is unreliable) ─────────
def _history(cad_tickers, days=520):
    """Build the price history the shadow trades on.

    Returns {ticker: DataFrame(Close, sma50, sma200, mom)} containing BOTH:
      * each CAD 2x ETF, SYNTHESISED as a 2x daily-rebalanced series off its fresh
        US underlying (see PROXY) with an MER/borrow drag, and
      * the raw US index tickers used for benchmarking and the broad gate.
    Never raises; a ticker that fails to download is simply absent.
    """
    out = {}
    try:
        import yfinance as yf
        import pandas as pd
        cad_tickers = list(cad_tickers)
        proxies = {PROXY[t] for t in cad_tickers if t in PROXY}
        raws = sorted(proxies | set(BROAD_GATE_TICKERS) | {"SPY", "QQQ"})
        end = datetime.now()
        start = end - timedelta(days=days)
        raw = yf.download(raws, start=start, end=end, progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)

        def _grab(sym):
            try:
                df = (raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw)
                df = df.dropna(subset=["Close"]).copy()
                return df if len(df) >= 210 else None
            except Exception:
                return None

        def _decorate(df):
            c = df["Close"]
            df["sma50"] = c.rolling(50).mean()
            df["sma200"] = c.rolling(200).mean()
            df["mom"] = c / c.shift(MOM_LOOKBACK) - 1
            return df

        # raw US tickers (benchmarks + broad gate) — used as-is
        under = {}
        for sym in raws:
            d = _grab(sym)
            if d is not None:
                under[sym] = d
                out[sym] = _decorate(d.copy())

        # CAD 2x ETFs — synthesised from their underlying's daily returns
        daily_drag = SYNTH_ANNUAL_DRAG / 252.0
        for t in cad_tickers:
            sym = PROXY.get(t)
            d = under.get(sym)
            if d is None:
                continue
            r = d["Close"].pct_change().fillna(0.0)
            lev_r = LEV * r - daily_drag          # 2x daily + cost => decay is inherent
            synth = (1.0 + lev_r).cumprod() * SYNTH_BASE
            sdf = pd.DataFrame({"Close": synth}, index=d.index)
            out[t] = _decorate(sdf)
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

    # Universe + anything still held (a delisted/removed name must still be marked).
    # _history() resolves each CAD ticker to its fresh US underlying and also
    # returns the raw index tickers. One pull per day is cheap.
    tickers = sorted(set(UNIVERSE) | set(st["holdings"].keys()))
    hist = _history(tickers)
    if not hist:
        return None                       # data outage; try again next cycle

    # STALENESS GUARD — kept as a safety net. Prices now come from US underlyings
    # (always fresh), but if a feed ever freezes again the shadow must refuse to
    # mark rather than repeat the 2026-07 failure, where .TO data stuck at
    # 2026-07-17 and the book silently held $9,991.99 to the cent for 12 days.
    _cad = [t for t in (set(st["holdings"].keys()) | set(UNIVERSE)) if t in hist]
    _fresh = None
    for t in _cad:
        df = hist.get(t)
        if df is not None and len(df):
            try:
                ld = df.index[-1].date()
                _fresh = ld if _fresh is None else max(_fresh, ld)
            except Exception:
                pass
    _stale_days = (date.today() - _fresh).days if _fresh else 999
    if _stale_days > 4:
        st["last_step"] = today
        _save_state(conn, st)
        return {"stale": True, "stale_days": _stale_days,
                "last_data": str(_fresh), "holdings": sorted(st["holdings"].keys())}

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
    """Broad-index risk gate: LONG only when BOTH the US (SPY) and Canada (EWC,
    standing in for the TSX — XIU.TO's feed is unreliable) are above their 200
    SMA. Risk-off if EITHER breaks (the conservative airbag). Fails SAFE — if the
    gate data is missing, return False (go to cash)."""
    ok = True
    for idx in BROAD_GATE_TICKERS:
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

    # BUY / rebalance into equal weight of the target set.
    # Reserve a cost buffer FIRST: allocating the full equity and then deducting
    # fees pushed cash negative (a small margin overdraft that compounds a little
    # at every rebalance). Sizing off investable equity keeps the book fully funded.
    investable = equity * (1 - 2 * COST_BPS / 10_000)   # covers a full round trip
    per_name = investable / len(target)
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

    _held = sorted(st["holdings"].keys())
    out = {"holdings": _held or ["CASH"],
           "priced_via": {t: PROXY.get(t) for t in _held if t in PROXY},
           "approx": [t for t in _held if t in APPROX],
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
