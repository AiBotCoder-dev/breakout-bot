"""
short_term_options.py — Short-DTE call strategy on BACKTESTED reversal setups.

THE EDGE (measured, not assumed)
--------------------------------
short_term_backtest.py tested six 1-3 day setups across 34 liquid names over 3y.
Only the capitulation/reversal setups showed real edge with a FAT RIGHT TAIL —
which is exactly what long options want (capped downside, uncapped upside):

  POST_PANIC_BOUNCE   3d edge +1.03% vs base, win 51%, p90 +11.9%   (n=447)
  GAP_DOWN_REVERSAL   3d edge +1.29% vs base, win 54%, p90 +12.0%   (n=458)

Plus the market-wide panic signals from panic_backtest.py (the strongest):
  SPY -5% day  -> +3.39% NEXT DAY, 83% win
  VIX >= 40    -> +8% over 20d, 90% win

Setups that were REJECTED by the backtest (noise/negative) and are deliberately
NOT traded here: momentum-thrust chasing, inside-day breakouts, oversold bounces.

DESIGN FOR SHORT HOLDS
----------------------
  • Contract: ATM to slightly ITM (delta 0.45-0.65). On a 2-3 day hold you have
    no time for a far-OTM lottery to come good — you need delta NOW. (This is the
    key difference from the momentum_options swing strategy, which buys OTM.)
  • DTE: 1-7 (next 1-2 weekly expiries). Short, but not 0DTE — 0DTE theta is a
    casino. We want enough time for the 2-3 day reversal to play out.
  • Exit: +75% take profit OR -40% stop OR hard time-stop at 3 trading days
    (the edge is measured over 1-3 days; holding longer just bleeds theta).
  • Sizing: defined-risk lottery — small fixed $ per ticket. Expect frequent
    full losses; the fat-tail winners carry the strategy.

HONEST EXPECTANCY
-----------------
This is NOT a money printer. The underlying edge is ~1-1.3% over baseline; the
"crazy returns" live only in the right tail and only on the rare high-conviction
days. Most tickets will lose. Forward P&L (tracked via options_performance) is
the only thing that proves whether the leverage converts the tail into profit.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except Exception:                       # pragma: no cover
    np = pd = yf = None


# ── Backtested stats attached to each setup (for the UI + honest disclosure) ──
SETUP_STATS = {
    "MARKET_PANIC_SPY5":  {"label": "Market capitulation (SPY -5% day)",
                           "horizon": "1-2d", "win": 83, "fwd": "+3.4%/day",
                           "tail": "index move levers to +200-400% on calls"},
    "MARKET_VIX40":       {"label": "Extreme fear (VIX >= 40)",
                           "horizon": "5-20d", "win": 90, "fwd": "+8% (20d)",
                           "tail": "highest-conviction buy signal we have"},
    "GAP_DOWN_REVERSAL":  {"label": "Gap-down reversal (gap<-2%, RSI<40)",
                           "horizon": "2-3d", "win": 54, "fwd": "+1.3% vs base",
                           "tail": "p90 +12% underlying"},
    "POST_PANIC_BOUNCE":  {"label": "Post-panic bounce (prior day -4%, today green)",
                           "horizon": "2-3d", "win": 51, "fwd": "+1.0% vs base",
                           "tail": "p90 +12% underlying"},
}

# Tunables
DTE_MIN, DTE_MAX, DTE_IDEAL = 1, 7, 4
DELTA_MIN, DELTA_MAX        = 0.45, 0.70    # ATM-to-slightly-ITM for short holds
COST_CAP_PER_CONTRACT       = 8.00          # ITM short-DTE costs more than OTM swings
MIN_VOL, MIN_OI             = 50, 100

TAKE_PROFIT_PCT  = 75.0
STOP_LOSS_PCT    = -40.0
TIME_STOP_DAYS   = 3
STRATEGY_LABEL   = "short_term_call"


def _rsi(s: pd.Series, p: int) -> float:
    d = s.diff()
    up = d.clip(lower=0).rolling(p).mean()
    dn = (-d.clip(upper=0)).rolling(p).mean()
    rs = up / dn.replace(0, np.nan)
    v = (100 - 100 / (1 + rs)).iloc[-1]
    return float(v) if pd.notna(v) else 50.0


# ══════════════════════════════════════════════════════════════════════════════
# SETUP DETECTION (single ticker)
# ══════════════════════════════════════════════════════════════════════════════
def detect_setups(ticker: str) -> list:
    """Return list of firing setup dicts for `ticker` based on today's bar."""
    if yf is None:
        return []
    try:
        raw = yf.download(ticker, period="1y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 60:
            return []
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"])
    except Exception:
        return []

    c = df["Close"]; o = df["Open"]
    last_close = float(c.iloc[-1]); prev_close = float(c.iloc[-2])
    last_open  = float(o.iloc[-1])
    gap = last_open / prev_close - 1 if prev_close else 0
    ret_today = last_close / prev_close - 1 if prev_close else 0
    ret_prev  = prev_close / float(c.iloc[-3]) - 1 if len(c) >= 3 else 0
    rsi14 = _rsi(c, 14)

    fired = []
    if gap <= -0.02 and rsi14 < 40:
        fired.append("GAP_DOWN_REVERSAL")
    if ret_prev <= -0.04 and ret_today > 0:
        fired.append("POST_PANIC_BOUNCE")
    return [{"ticker": ticker.upper(), "setup": s, "price": round(last_close, 2),
             "stats": SETUP_STATS[s]} for s in fired]


def market_panic_signals() -> list:
    """Market-wide capitulation signals (strongest short-term edge)."""
    if yf is None:
        return []
    out = []
    try:
        spy = yf.download("SPY", period="1mo", interval="1d",
                          progress=False, auto_adjust=True)
        vix = yf.download("^VIX", period="5d", interval="1d",
                          progress=False, auto_adjust=True)
        for d in (spy, vix):
            if d is not None and isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
        if spy is not None and not spy.empty:
            r1 = float(spy["Close"].iloc[-1] / spy["Close"].iloc[-2] - 1)
            if r1 <= -0.05:
                out.append({"setup": "MARKET_PANIC_SPY5", "ticker": "SPY",
                            "price": round(float(spy["Close"].iloc[-1]), 2),
                            "stats": SETUP_STATS["MARKET_PANIC_SPY5"]})
        if vix is not None and not vix.empty:
            v = float(vix["Close"].iloc[-1])
            if v >= 40:
                out.append({"setup": "MARKET_VIX40", "ticker": "SPY",
                            "price": round(float(spy["Close"].iloc[-1]), 2)
                                     if spy is not None and not spy.empty else 0,
                            "stats": SETUP_STATS["MARKET_VIX40"]})
    except Exception:
        pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT SELECTION (ATM/ITM for short holds)
# ══════════════════════════════════════════════════════════════════════════════
def _pick_expiry(expiries: list) -> str | None:
    today = date.today(); best, bd = None, 10**9
    for e in expiries:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        except Exception:
            continue
        if DTE_MIN <= dte <= DTE_MAX and abs(dte - DTE_IDEAL) < bd:
            best, bd = e, abs(dte - DTE_IDEAL)
    return best


def select_short_dte_call(ticker: str, spot: float) -> dict | None:
    """ATM-to-slightly-ITM call, 1-7 DTE — high delta for a 2-3 day hold."""
    if yf is None or spot <= 0:
        return None
    try:
        tk = yf.Ticker(ticker)
        exp = _pick_expiry(list(tk.options or []))
        if not exp:
            return None
        calls = tk.option_chain(exp).calls
        if calls is None or calls.empty:
            return None
    except Exception:
        return None

    today = date.today()
    try:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
    except Exception:
        dte = DTE_IDEAL

    # Target strikes from ~3% ITM to ATM (high delta, affordable-ish)
    lo, hi = spot * 0.97, spot * 1.005
    cand = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
    if cand.empty:
        cand = calls.iloc[(calls["strike"] - spot).abs().argsort()[:3]]

    best, best_score = None, -1e9
    for _, r in cand.iterrows():
        def _f(v, d=0.0):
            try:
                x = float(v); return x if x == x else d
            except Exception:
                return d
        b, a = _f(r.get("bid")), _f(r.get("ask"))
        prem = (a + b) / 2 if (b > 0 and a > 0) else _f(r.get("lastPrice"))
        if prem <= 0 or prem > COST_CAP_PER_CONTRACT:
            continue
        vol, oi = _f(r.get("volume")), _f(r.get("openInterest"))
        if vol < MIN_VOL and oi < MIN_OI:
            continue
        strike = _f(r.get("strike"))
        iv = _f(r.get("impliedVolatility"))
        # Prefer closest-to-ATM with decent liquidity, cheaper premium
        score = -abs(strike - spot) + min(oi / 5000, 1) - prem * 0.2
        if score > best_score:
            best_score, best = score, {
                "ticker": ticker.upper(), "contract_symbol": str(r.get("contractSymbol", "")),
                "strike": strike, "expiry": exp, "dte": dte,
                "premium": round(prem, 2), "iv": round(iv, 3),
                "volume": int(vol), "open_interest": int(oi),
                "moneyness": round((strike / spot - 1) * 100, 1),
            }
    return best


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
class ShortTermOptionsStrategy:
    def __init__(self, conn, universe: list | None = None):
        self.conn = conn
        from momentum_strategy import LIQUID_UNIVERSE
        self.universe = [t.upper() for t in (universe or LIQUID_UNIVERSE) if "." not in t]

    def scan(self, progress=None) -> list:
        """Find all firing short-term setups + best short-DTE call for each."""
        plays = []
        # 1) Market-wide panic (strongest) — applies to SPY/QQQ
        for sig in market_panic_signals():
            for proxy in ("SPY", "QQQ"):
                c = select_short_dte_call(proxy, sig["price"] or 0)
                if c:
                    plays.append({**sig, "proxy": proxy, "contract": c}); break
        # 2) Single-name reversal setups
        total = len(self.universe)
        for i, t in enumerate(self.universe):
            if progress:
                try: progress(i + 1, total, t)
                except Exception: pass
            for s in detect_setups(t):
                c = select_short_dte_call(t, s["price"])
                if c:
                    plays.append({**s, "proxy": t, "contract": c})
        return plays


if __name__ == "__main__":
    print("Scanning for short-term reversal option setups (no DB writes)...\n")
    mkt = market_panic_signals()
    print(f"Market panic signals firing: {len(mkt)}")
    for m in mkt:
        print(f"  {m['setup']}: {m['stats']['label']}  ({m['stats']['win']}% win)")
    print("\nChecking a few names for single-name setups:")
    for t in ["NVDA", "AAPL", "TSLA", "AMD", "META", "PLTR"]:
        s = detect_setups(t)
        if s:
            for x in s:
                print(f"  {t}: {x['setup']}  @ ${x['price']}")
        else:
            print(f"  {t}: no setup")
