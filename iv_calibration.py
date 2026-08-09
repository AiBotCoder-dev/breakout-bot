"""
iv_calibration.py — MEASURE the real option surface instead of assuming it.

Every backtest this session hit the same wall: we assume implied volatility rather
than observe it. That single unobservable decided several results —
  * premium selling looked -4.6% ... but that depends on the real IV/RV gap
  * "buy cheap IV" looked like a 1.19x edge ... and turned out to be an artifact
  * the earnings edge is 1.25x on a flat surface but ~1.06x if the earnings smile
    is steep — and price history CANNOT tell us which
This tool ends the guessing by pulling live option chains from Alpaca and measuring:

  atm_iv        implied vol at the money (front expiry ~10 DTE)
  iv_rv_ratio   atm_iv / 20-day realized vol  -> the VARIANCE RISK PREMIUM, directly
  smile_slope   how much richer 5% OTM strikes are, in the SAME parameterisation the
                backtests used:  iv(K) = iv_atm * (1 + slope*|ln(K/S)|)
  put/call skew separately, since equity smiles are lopsided
  spread_pct    real bid/ask width -> validates (or corrects) our friction assumptions
  earnings_in   whether earnings falls inside the option's life

Persisted daily to `iv_calibration`, so the surface is measured across regimes and
earnings vs non-earnings names — which finally settles the earnings-smile question
with data instead of a sensitivity range.

Runs read-only inside the monitor cycle; never trades.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone, date, timedelta

# names we actually trade / study — a spread of mega-cap and high-beta
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "TSLA",
            "NFLX", "COIN", "PLTR", "MU", "CRWD", "SHOP", "UBER", "SPY", "QQQ"]
DTE_TARGET = 10          # match the backtests
OTM_PROBE = 0.05         # measure the smile at 5% OTM (what the studies priced)
R = 0.04
_N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ── Black-Scholes + implied-vol inversion (fallback when the feed omits IV) ──
def _bs(S, K, T, sig, put=False):
    if T <= 0 or sig <= 0:
        return max((K - S) if put else (S - K), 0.0)
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return (K * math.exp(-R * T) * _N(-d2) - S * _N(-d1)) if put \
        else (S * _N(d1) - K * math.exp(-R * T) * _N(d2))


def implied_vol(price, S, K, T, put=False):
    """Bisection inversion — robust where Newton can diverge on wide/edge quotes."""
    if price <= 0 or T <= 0:
        return None
    intrinsic = max((K - S) if put else (S - K), 0.0)
    if price < intrinsic - 1e-6:
        return None
    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _bs(S, K, T, mid, put) > price:
            hi = mid
        else:
            lo = mid
    iv = (lo + hi) / 2
    return iv if 0.02 < iv < 4.9 else None


def _ensure(conn):
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS iv_calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                d TEXT, ticker TEXT, spot REAL, expiry TEXT, dte INTEGER,
                atm_iv REAL, rv20 REAL, iv_rv_ratio REAL,
                call_otm_iv REAL, put_otm_iv REAL,
                call_slope REAL, put_slope REAL, smile_slope REAL,
                atm_spread_pct REAL, otm_spread_pct REAL,
                earnings_in_life INTEGER, days_to_earnings INTEGER
            )""")
    except Exception:
        pass


def _chain(broker, underlying, target_dte=DTE_TARGET):
    """Fetch option snapshots for the expiry closest to target_dte."""
    import requests
    from broker import DATA_BASE
    lo = (date.today() + timedelta(days=max(2, target_dte - 6))).isoformat()
    hi = (date.today() + timedelta(days=target_dte + 10)).isoformat()
    try:
        r = requests.get(f"{DATA_BASE}/v1beta1/options/snapshots/{underlying}",
                         headers=broker._headers(),
                         params={"expiration_date_gte": lo,
                                 "expiration_date_lte": hi, "limit": 1000},
                         timeout=25)
        if r.status_code != 200:
            return {}
        return r.json().get("snapshots", {}) or {}
    except Exception:
        return {}


def _spot_and_rv(ticker):
    try:
        import yfinance as yf
        import numpy as np
        import pandas as pd
        h = yf.Ticker(ticker).history(period="3mo")["Close"].dropna()
        if len(h) < 25:
            return None, None
        lr = np.log(h / h.shift()).dropna()
        return float(h.iloc[-1]), float(lr.iloc[-20:].std() * math.sqrt(252))
    except Exception:
        return None, None


def _days_to_earnings(ticker):
    try:
        import yfinance as yf
        import pandas as pd
        ed = yf.Ticker(ticker).get_earnings_dates(limit=12)
        if ed is None or len(ed) == 0:
            return None
        today = pd.Timestamp(date.today())
        fut = [d for d in pd.to_datetime(ed.index).tz_localize(None).normalize() if d >= today]
        return int((min(fut) - today).days) if fut else None
    except Exception:
        return None


def measure(broker, ticker):
    """Measure the surface for one name. Returns a dict or None."""
    from broker import AlpacaPaperBroker
    spot, rv20 = _spot_and_rv(ticker)
    if not spot:
        return None
    snaps = _chain(broker, ticker)
    if not snaps:
        return None

    # group by expiry, keep the one closest to DTE_TARGET
    rows = []
    for sym, s in snaps.items():
        p = AlpacaPaperBroker.parse_occ_symbol(sym)
        if not p:
            continue
        q = s.get("latestQuote") or {}
        bid = float(q.get("bp", 0) or 0); ask = float(q.get("ap", 0) or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        dte = (p["expiry"] - date.today()).days
        if dte <= 1:
            continue
        mid = (bid + ask) / 2
        iv = s.get("impliedVolatility")
        try:
            iv = float(iv) if iv else None
        except Exception:
            iv = None
        if not iv:                       # feed omitted IV -> invert it ourselves
            iv = implied_vol(mid, spot, p["strike"], dte / 365, p["type"] == "put")
        if not iv:
            continue
        rows.append({"dte": dte, "strike": p["strike"], "type": p["type"],
                     "iv": iv, "mid": mid,
                     "spread_pct": (ask - bid) / mid * 100 if mid > 0 else 999,
                     "expiry": p["expiry"]})
    if not rows:
        return None
    best_dte = min({r["dte"] for r in rows}, key=lambda d: abs(d - DTE_TARGET))
    rows = [r for r in rows if r["dte"] == best_dte]
    if len(rows) < 4:
        return None

    def _nearest(kind, target_strike):
        c = [r for r in rows if r["type"] == kind]
        return min(c, key=lambda r: abs(r["strike"] - target_strike)) if c else None

    atm_c = _nearest("call", spot); atm_p = _nearest("put", spot)
    otm_c = _nearest("call", spot * (1 + OTM_PROBE))
    otm_p = _nearest("put", spot * (1 - OTM_PROBE))
    if not (atm_c and atm_p and otm_c and otm_p):
        return None
    atm_iv = (atm_c["iv"] + atm_p["iv"]) / 2

    def _slope(otm, atm_iv_):
        m = abs(math.log(otm["strike"] / spot))
        if m < 1e-4 or atm_iv_ <= 0:
            return None
        return (otm["iv"] / atm_iv_ - 1.0) / m       # matches iv=atm*(1+slope*|ln K/S|)

    cs = _slope(otm_c, atm_iv); ps = _slope(otm_p, atm_iv)
    dte_e = _days_to_earnings(ticker)
    return {
        "d": date.today().isoformat(), "ticker": ticker, "spot": round(spot, 2),
        "expiry": str(rows[0]["expiry"]), "dte": best_dte,
        "atm_iv": round(atm_iv, 4), "rv20": round(rv20, 4) if rv20 else None,
        "iv_rv_ratio": round(atm_iv / rv20, 3) if rv20 else None,
        "call_otm_iv": round(otm_c["iv"], 4), "put_otm_iv": round(otm_p["iv"], 4),
        "call_slope": round(cs, 3) if cs is not None else None,
        "put_slope": round(ps, 3) if ps is not None else None,
        "smile_slope": round((cs + ps) / 2, 3) if (cs is not None and ps is not None) else None,
        "atm_spread_pct": round((atm_c["spread_pct"] + atm_p["spread_pct"]) / 2, 1),
        "otm_spread_pct": round((otm_c["spread_pct"] + otm_p["spread_pct"]) / 2, 1),
        "earnings_in_life": 1 if (dte_e is not None and dte_e <= best_dte) else 0,
        "days_to_earnings": dte_e,
    }


def step(conn, broker=None, force=False):
    """Measure the whole universe once per day. Read-only; never trades."""
    _ensure(conn)
    today = date.today().isoformat()
    if not force:
        try:
            r = conn.execute("SELECT COUNT(*) c FROM iv_calibration WHERE d=?",
                             (today,)).fetchone()
            if int((dict(r).get("c") if hasattr(r, "keys") else r[0]) or 0) > 0:
                return None
        except Exception:
            pass
    if broker is None:
        try:
            from broker import AlpacaPaperBroker
            broker = AlpacaPaperBroker()
        except Exception:
            return None
    if not broker.available():
        return None

    n = 0
    for t in UNIVERSE:
        m = measure(broker, t)
        if not m:
            continue
        try:
            conn.execute(
                "INSERT INTO iv_calibration (d, ticker, spot, expiry, dte, atm_iv, "
                "rv20, iv_rv_ratio, call_otm_iv, put_otm_iv, call_slope, put_slope, "
                "smile_slope, atm_spread_pct, otm_spread_pct, earnings_in_life, "
                "days_to_earnings) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (m["d"], m["ticker"], m["spot"], m["expiry"], m["dte"], m["atm_iv"],
                 m["rv20"], m["iv_rv_ratio"], m["call_otm_iv"], m["put_otm_iv"],
                 m["call_slope"], m["put_slope"], m["smile_slope"],
                 m["atm_spread_pct"], m["otm_spread_pct"], m["earnings_in_life"],
                 m["days_to_earnings"]))
            n += 1
        except Exception:
            pass
    return {"measured": n}


def report(conn) -> dict:
    """Summarise everything measured so far — the answers to our open questions."""
    _ensure(conn)
    out = {}
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM iv_calibration WHERE atm_iv IS NOT NULL").fetchall()]
    except Exception:
        rows = []
    out["n"] = len(rows)
    if not rows:
        return out

    def _avg(vals):
        v = [float(x) for x in vals if x is not None]
        return round(sum(v) / len(v), 3) if v else None

    out["iv_rv_ratio"] = _avg([r.get("iv_rv_ratio") for r in rows])
    out["smile_slope"] = _avg([r.get("smile_slope") for r in rows])
    out["put_slope"] = _avg([r.get("put_slope") for r in rows])
    out["call_slope"] = _avg([r.get("call_slope") for r in rows])
    out["atm_spread_pct"] = _avg([r.get("atm_spread_pct") for r in rows])
    out["otm_spread_pct"] = _avg([r.get("otm_spread_pct") for r in rows])
    e = [r for r in rows if r.get("earnings_in_life")]
    ne = [r for r in rows if not r.get("earnings_in_life")]
    out["earnings"] = {"n": len(e), "smile_slope": _avg([r.get("smile_slope") for r in e]),
                       "iv_rv_ratio": _avg([r.get("iv_rv_ratio") for r in e])}
    out["no_earnings"] = {"n": len(ne), "smile_slope": _avg([r.get("smile_slope") for r in ne]),
                          "iv_rv_ratio": _avg([r.get("iv_rv_ratio") for r in ne])}
    return out


if __name__ == "__main__":
    import json
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit("Set DATABASE_URL (and Alpaca keys) to run.")
    import psycopg2, monitor
    raw = psycopg2.connect(url, sslmode="require"); raw.autocommit = True
    conn = monitor.PgAdapter(raw)
    print("step:", step(conn, force=True))
    print(json.dumps(report(conn), indent=1, default=str))
