"""
ignition_shadow.py — READ-ONLY paper hunter for igniting pumps & crashes.

Two validated convex sleeves (backtests: ignition_detector.py, convex_put_backtest.py),
run in PAPER so we watch their LIVE expectancy before risking a cent:

  PUMP  — breakout near 20d-highs (px>=98% of 20d-high, price>EMA20>EMA50, 10d>+6%)
          OR a sharp 5-day thrust >+10%, with RVOL>1.2  ->  buy a 2% OTM CALL, ~10
          DTE, hold to expiry. Rebuilt to catch GRINDING runs, not just explosive
          days (5y catch rate 36% vs 9% for the old RVOL+5%-day trigger).
  CRASH — early breakdown (close<EMA20 AND 5-day return<-4%) OR a sharp -6% down-day
          ->  buy a 5% OTM PUT, ~7 DTE, hold to expiry (5y catch rate 86%). The
          EARLY breakdown, not the panic-day reaction (which is -EV).

Both are tail-hunters: ~70-75% of trades expire worthless, a rare few pay big, and
the whole edge is hold-to-expiry (a -50% stop INVERTS it). So this NEVER uses a
stop — it marks each paper option with Black-Scholes and settles at intrinsic on
expiry. It NEVER touches the broker. Fixed tiny $100 paper ticket per signal so a
long losing streak is survivable (the real-world constraint for tail strategies).

monitor.py calls step(conn) each cycle; it self-gates to once per UTC day.
Tables: ignition_shadow (one row per paper trade).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, date, timedelta

UNIVERSE = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","AMD","MU","SMCI",
            "MRVL","ARM","QCOM","INTC","TSLA","PLTR","COIN","MSTR","HOOD","SOFI",
            "AFRM","SHOP","NET","CRWD","DDOG","SNOW","RBLX","U","ABNB","UBER",
            "DASH","GME","MARA","RIOT","RDDT","CVNA","BABA","PDD","NIO","DELL",
            "AI","PATH","SOUN","CELH","ANF","DKNG","PANW","NFLX","MRNA","ROKU"]

RVOL_MIN = 3.0; PUMP_MOVE = 0.05; CRASH_RET5 = -0.04
PUMP_OTM = 0.02; PUMP_DTE = 10
CRASH_OTM = 0.05; CRASH_DTE = 7
TICKET = 100.0          # fixed tiny paper ticket per signal
IVP = 0.95              # entry IV = realized vol * this
RET_CAP = 1200.0        # realistic exit-liquidity cap on a single win
R = 0.04
_N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs(S, K, T, sig, put=False):
    if T <= 0 or sig <= 0:
        return max((K - S) if put else (S - K), 0.0)
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return (K * math.exp(-R * T) * _N(-d2) - S * _N(-d1)) if put \
        else (S * _N(d1) - K * math.exp(-R * T) * _N(d2))
def _buyf(m): return m + max(m * 0.03, 0.02) + 0.0065
def _sellf(m): return max(0.0, m - max(m * 0.03, 0.02) - 0.0065)


def _ensure(conn):
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ignition_shadow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, ticker TEXT, sleeve TEXT, direction TEXT,
                strike REAL, entry_underlying REAL, entry_premium REAL,
                iv0 REAL, expiry TEXT, ticket REAL,
                status TEXT DEFAULT 'OPEN',
                exit_underlying REAL, exit_premium REAL,
                pnl_pct REAL, pnl_dollars REAL, closed_at TEXT
            )""")
    except Exception:
        pass


# ── market data ───────────────────────────────────────────────────────────────
def _history(tickers, days=160):
    out = {}
    try:
        import yfinance as yf
        import numpy as np
        import pandas as pd
        raw = yf.download(list(tickers), period=None,
                          start=datetime.now() - timedelta(days=days),
                          end=datetime.now(), progress=False, auto_adjust=True,
                          group_by="ticker", threads=True)
        for t in tickers:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(subset=["Close"]).copy()
                if len(df) < 30:
                    continue
                c = df["Close"]
                df["ema20"] = c.ewm(span=20).mean()
                df["ema50"] = c.ewm(span=50).mean()
                df["rv20"] = np.log(c / c.shift()).rolling(20).std() * np.sqrt(252)
                df["ret1"] = c.pct_change()
                df["ret5"] = c / c.shift(5) - 1
                df["ret10"] = c / c.shift(10) - 1
                df["hi20"] = c.rolling(20).max()
                if "Volume" in df:
                    df["rvol"] = df["Volume"] / df["Volume"].rolling(20).mean()
                else:
                    df["rvol"] = 0.0
                out[t] = df.dropna()
            except Exception:
                continue
    except Exception:
        pass
    return out


def _f(df, col):
    try:
        v = float(df[col].iloc[-1]); return v if v == v else None
    except Exception:
        return None


# ── the daily step ────────────────────────────────────────────────────────────
def step(conn, force=False, notify=None):
    """Once per UTC day: detect new ignition signals, open paper trades, mark &
    settle existing ones. READ-ONLY; never touches the broker."""
    _ensure(conn)
    _ensure_state(conn)
    today = date.today().isoformat()
    if not force and _last_step(conn) == today:
        return None

    hist = _history(UNIVERSE)
    if not hist:
        return None

    # open tickers per sleeve (avoid stacking duplicates)
    open_keys = _open_keys(conn)
    new_signals = []
    for t, df in hist.items():
        px = _f(df, "Close")
        if px is None:
            continue
        rvol = _f(df, "rvol") or 0.0; ret1 = _f(df, "ret1") or 0.0
        ret5 = _f(df, "ret5") or 0.0; ret10 = _f(df, "ret10") or 0.0
        ema20 = _f(df, "ema20"); ema50 = _f(df, "ema50"); hi20 = _f(df, "hi20")
        rv = _f(df, "rv20") or 0.4
        # PUMP (rebuilt — catches grinding runs, not just explosive days; 5y catch
        # 36% vs 9% for the old RVOL+5%-day trigger). Breakout near 20d-highs in an
        # uptrend with 10d thrust, OR a sharp 5-day thrust; RVOL>1.2 filters drift.
        pump = (rvol > 1.2 and (
            (None not in (ema20, ema50, hi20) and px >= 0.98 * hi20 and px > ema20
             and ema20 > ema50 and ret10 > 0.06) or ret5 > 0.10))
        # CRASH (rebuilt — 5y catch 86%): early breakdown below EMA20 + weak week,
        # OR a sharp -6% down-day.
        crash = ((ema20 is not None and px < ema20 and ret5 < -0.04) or ret1 < -0.06)
        if pump and (t, "pump") not in open_keys:
            new_signals.append(_open(conn, t, "pump", "call", px, rv))
        elif crash and (t, "crash") not in open_keys:
            new_signals.append(_open(conn, t, "crash", "put", px, rv))

    marked = _mark_and_settle(conn, hist)
    _set_last_step(conn, today)

    new_signals = [s for s in new_signals if s]
    if new_signals and notify:
        try:
            lines = [f"🎯 <b>Ignition shadow</b> (paper) — {len(new_signals)} new signal(s):"]
            for s in new_signals:
                lines.append(f"{'🚀 PUMP' if s['sleeve']=='pump' else '💥 CRASH'} "
                             f"{s['ticker']}: would buy {s['direction'].upper()} "
                             f"${s['strike']:.0f} (~{s['dte']}DTE) @ ~${s['entry_premium']:.2f}")
            notify("\n".join(lines))
        except Exception:
            pass
    return {"new": len(new_signals), "settled": marked,
            "signals": [(s["ticker"], s["sleeve"]) for s in new_signals]}


def _open(conn, ticker, sleeve, direction, px, rv):
    otm = PUMP_OTM if sleeve == "pump" else CRASH_OTM
    dte = PUMP_DTE if sleeve == "pump" else CRASH_DTE
    put = direction == "put"
    K = px * (1 + otm) if not put else px * (1 - otm)
    iv0 = max(0.20, min(2.5, rv * IVP))
    mid = _bs(px, K, dte / 365, iv0, put)
    entry = _buyf(mid)
    if mid < 0.15 or entry <= 0.01:       # un-buyable — skip
        return None
    now = datetime.now(timezone.utc)
    expiry = (date.today() + timedelta(days=dte)).isoformat()
    try:
        conn.execute(
            "INSERT INTO ignition_shadow (ts, ticker, sleeve, direction, strike, "
            "entry_underlying, entry_premium, iv0, expiry, ticket, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?, 'OPEN')",
            (date.today().isoformat(), ticker, sleeve, direction, round(K, 2),
             round(px, 2), round(entry, 3), round(iv0, 3), expiry, TICKET))
    except Exception:
        return None
    return {"ticker": ticker, "sleeve": sleeve, "direction": direction,
            "strike": K, "dte": dte, "entry_premium": entry}


def _mark_and_settle(conn, hist):
    """Settle any paper option whose expiry has passed, at intrinsic vs the current
    underlying. (We don't store daily marks — only the final realised outcome.)"""
    try:
        rows = conn.execute("SELECT * FROM ignition_shadow WHERE status='OPEN'").fetchall()
    except Exception:
        return 0
    n = 0; today = date.today()
    for r in rows:
        d = dict(r) if hasattr(r, "keys") else {}
        try:
            exp = date.fromisoformat(str(d.get("expiry"))[:10])
        except Exception:
            continue
        if exp > today:
            continue                       # still live
        t = d.get("ticker"); df = hist.get(t)
        S1 = _f(df, "Close") if df is not None else None
        if S1 is None:
            S1 = float(d.get("entry_underlying") or 0)   # fallback
        K = float(d.get("strike") or 0); put = d.get("direction") == "put"
        entry = float(d.get("entry_premium") or 0)
        intrinsic = max((K - S1) if put else (S1 - K), 0.0)
        pct = min((_sellf(intrinsic) / entry - 1) * 100, RET_CAP) if entry > 0 else -100.0
        pct = max(-100.0, pct)
        dollars = round(pct / 100.0 * float(d.get("ticket") or TICKET), 2)
        try:
            conn.execute(
                "UPDATE ignition_shadow SET status='CLOSED', exit_underlying=?, "
                "exit_premium=?, pnl_pct=?, pnl_dollars=?, closed_at=? WHERE id=?",
                (round(S1, 2), round(intrinsic, 3), round(pct, 1), dollars,
                 today.isoformat(), d.get("id")))
            n += 1
        except Exception:
            pass
    return n


# ── tiny state helpers (once-per-day gate) ────────────────────────────────────
def _ensure_state(conn):
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS ignition_shadow_state "
                     "(id INTEGER PRIMARY KEY, last_step TEXT)")
    except Exception:
        pass
def _last_step(conn):
    try:
        r = conn.execute("SELECT last_step FROM ignition_shadow_state WHERE id=1").fetchone()
        return (dict(r).get("last_step") if hasattr(r, "keys") else r[0]) if r else None
    except Exception:
        return None
def _set_last_step(conn, d):
    try:
        conn.execute("INSERT INTO ignition_shadow_state (id, last_step) VALUES (1,?) "
                     "ON CONFLICT (id) DO UPDATE SET last_step=EXCLUDED.last_step", (d,))
    except Exception:
        pass
def _open_keys(conn):
    keys = set()
    try:
        for r in conn.execute("SELECT ticker, sleeve FROM ignition_shadow "
                              "WHERE status='OPEN'").fetchall():
            d = dict(r) if hasattr(r, "keys") else {"ticker": r[0], "sleeve": r[1]}
            keys.add((d.get("ticker"), d.get("sleeve")))
    except Exception:
        pass
    return keys


# ── reporting (dashboard) ─────────────────────────────────────────────────────
def report(conn) -> dict:
    _ensure(conn)
    out = {"sleeves": {}, "open": 0, "recent": []}
    for sleeve in ("pump", "crash"):
        try:
            rows = conn.execute(
                "SELECT pnl_pct, pnl_dollars FROM ignition_shadow "
                "WHERE status='CLOSED' AND sleeve=?", (sleeve,)).fetchall()
            vals = [(float((dict(r) if hasattr(r,"keys") else {"pnl_pct":r[0],"pnl_dollars":r[1]}).get("pnl_pct") or 0),
                     float((dict(r) if hasattr(r,"keys") else {"pnl_pct":r[0],"pnl_dollars":r[1]}).get("pnl_dollars") or 0))
                    for r in rows]
        except Exception:
            vals = []
        if vals:
            n = len(vals); wins = sum(1 for p, _ in vals if p > 0)
            out["sleeves"][sleeve] = {
                "n": n, "win": round(100 * wins / n, 0),
                "expectancy": round(sum(p for p, _ in vals) / n, 1),
                "total_pnl": round(sum(d for _, d in vals), 2)}
        else:
            out["sleeves"][sleeve] = {"n": 0, "win": 0, "expectancy": 0, "total_pnl": 0}
    try:
        r = conn.execute("SELECT COUNT(*) c FROM ignition_shadow WHERE status='OPEN'").fetchone()
        out["open"] = int((dict(r).get("c") if hasattr(r, "keys") else r[0]) or 0)
    except Exception:
        pass
    try:
        rows = conn.execute(
            "SELECT ts, ticker, sleeve, direction, strike, status, pnl_pct "
            "FROM ignition_shadow ORDER BY id DESC LIMIT 12").fetchall()
        out["recent"] = [dict(r) if hasattr(r, "keys") else {} for r in rows]
    except Exception:
        pass
    tot = sum(s["total_pnl"] for s in out["sleeves"].values())
    out["total_pnl"] = round(tot, 2)
    return out


if __name__ == "__main__":
    import os, json
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit("Set DATABASE_URL to smoke-test.")
    import psycopg2, monitor
    raw = psycopg2.connect(url, sslmode="require"); raw.autocommit = True
    conn = monitor.PgAdapter(raw)
    print("step:", step(conn, force=True))
    print("report:", json.dumps(report(conn), indent=1, default=str))
