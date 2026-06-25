"""
dipbuy.py — DIP-BUY mean-reversion call strategy (backtested 63.6% win rate).

Signal:    a short-term OVERSOLD dip (RSI(2) < 10) in a confirmed UPTREND
           (price > 50-day SMA > 200-day SMA). Oversold names in uptrends bounce
           more often than not, so the directional hit rate is high.
Structure: a ~7% IN-THE-MONEY call (delta ~0.75), ~12 DTE — high delta so the option
           win rate tracks the directional win rate (far-OTM throws that away).
Exit:      handled by broker.manage_option_exits' DIPBUY branch — +20% profit target /
           -45% stop / ~7-trading-day time cap (taking the bounce is what realises the
           wins). See mean_reversion_strategy.py for the full backtest + walk-forward.

This is OPT-IN via the DIPBUY env flag and runs alongside the existing scanner.
"""
from __future__ import annotations
import os
import math
from datetime import date, timedelta

# ── validated parameters (env-overridable) ──────────────────────────────────
ITM_DEPTH    = float(os.environ.get("DIPBUY_ITM_DEPTH", "0.07"))   # strike = price*(1-this)
DTE_IDEAL    = int(os.environ.get("DIPBUY_DTE", "12"))
RSI2_MAX     = float(os.environ.get("DIPBUY_RSI2_MAX", "10"))
TARGET_PCT   = float(os.environ.get("DIPBUY_TARGET_PCT", "20"))    # +20% take-profit
STOP_PCT     = float(os.environ.get("DIPBUY_STOP_PCT", "-45"))     # -45% stop
MAXHOLD_CAL  = int(os.environ.get("DIPBUY_MAXHOLD_DAYS", "10"))    # ≈7 trading days

UNIVERSE = ["NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","AMD","AVGO","NFLX",
            "JPM","BAC","XOM","UNH","V","MA","COST","WMT","DIS","CRM","ORCL","ADBE",
            "QCOM","MU","INTC","PLTR","COIN","SOFI","MARA","RIOT","SMCI","ARM","SNOW",
            "UBER","ABNB","SHOP","RBLX","CVNA","DKNG","HD","NKE","PYPL","BA","CAT","GS",
            "SPY","QQQ","IWM","SMH","XLK","XLF","XLE"]


def _rsi(c, p):
    import numpy as np
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = np.full_like(c, 50.0, float); ag = al = 0.0
    for i in range(1, len(c)):
        if i <= p:
            ag = (ag*(i-1)+g[i])/i; al = (al*(i-1)+l[i])/i
        else:
            ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+l[i])/p
        out[i] = 100.0 if al < 1e-12 else 100.0 - 100.0/(1.0+ag/al)
    return out


def _pick_expiry(broker, ticker):
    """Nearest listed Alpaca expiry to DTE_IDEAL within a sensible weekly window."""
    today = date.today()
    lo = (today + timedelta(days=max(2, DTE_IDEAL - 6))).isoformat()
    hi = (today + timedelta(days=DTE_IDEAL + 12)).isoformat()
    try:
        data = broker._get("/v2/options/contracts", {
            "underlying_symbols": ticker.upper(), "type": "call",
            "status": "active", "expiration_date_gte": lo,
            "expiration_date_lte": hi, "limit": 500})
        exps = sorted({c.get("expiration_date")
                       for c in (data.get("option_contracts") or [])
                       if c.get("expiration_date")})
        if not exps:
            return None
        return min(exps, key=lambda e: abs((date.fromisoformat(e) - today).days - DTE_IDEAL))
    except Exception:
        return None


def find_dipbuy_setups(broker, universe=None, max_results=2):
    """Return candidate dicts (compatible with monitor._attempt_entry) for names that
    are oversold (RSI2<RSI2_MAX) inside an uptrend, each as a ~7% ITM call ~12 DTE.
    Ranked most-oversold first. Defensive: returns [] on any failure."""
    try:
        import numpy as np
        import yfinance as yf
    except Exception:
        return []
    universe = universe or UNIVERSE
    today = date.today()
    try:
        data = yf.download(universe, period="14mo", auto_adjust=True,
                           group_by="ticker", threads=True, progress=False)
    except Exception:
        return []
    cands = []
    for tk in universe:
        try:
            c = data[tk]["Close"].dropna()
            if len(c) < 210:
                continue
            cv = c.values
            price = float(cv[-1])
            s50 = float(c.rolling(50).mean().iloc[-1])
            s200 = float(c.rolling(200).mean().iloc[-1])
            if not (price > s50 > s200):
                continue
            r2 = float(_rsi(cv, 2)[-1])
            if r2 >= RSI2_MAX:
                continue
            cands.append((r2, tk, price))
        except Exception:
            continue
    cands.sort(key=lambda x: x[0])           # most oversold first
    out = []
    for r2, tk, price in cands:
        if len(out) >= max_results:
            break
        expiry = _pick_expiry(broker, tk)
        if not expiry:
            continue
        dte = (date.fromisoformat(expiry) - today).days
        strike = round(price * (1 - ITM_DEPTH), 2)
        out.append({
            "ticker": tk, "expiry": expiry, "strike": strike,
            "option_type": "call", "dte": dte,
            "otm_pct": strike / price - 1.0,        # negative = ITM
            "quality_score": round(min(100.0, (RSI2_MAX - r2) * 4 + 55), 0),
            "sources": ["dipbuy"], "rsi2": round(r2, 1), "price": price,
        })
    return out


if __name__ == "__main__":
    # offline self-test of the signal logic (no broker needed)
    import numpy as np, yfinance as yf
    h = yf.Ticker("AAPL").history(period="14mo")["Close"].dropna()
    print("AAPL RSI2 =", round(float(_rsi(h.values, 2)[-1]), 1),
          "| params:", dict(ITM_DEPTH=ITM_DEPTH, DTE=DTE_IDEAL, RSI2_MAX=RSI2_MAX,
                            TARGET=TARGET_PCT, STOP=STOP_PCT, MAXHOLD=MAXHOLD_CAL))
