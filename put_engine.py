"""
put_engine.py — Bearish/put counterpart to the call engines, GATED to where puts work.

WHAT THE BACKTEST FORCED (bearish_backtest.py, 34 names, 3y)
-----------------------------------------------------------
Most "obvious" bearish setups LOSE money because the market drifts up and
everything bounces:
  DOWNTREND_BREAKDOWN  3d +0.84%  (bounces — anti-edge)
  RELATIVE_WEAKNESS    3d +0.70%  (bounces — anti-edge)
  DEATH_CROSS_FRESH    5d +1.45%  (contrarian BUY, not a put signal!)
Only two showed weak put edge:
  OVERBOUGHT_DOWNTREND 5d bear_edge -0.52%  (n=452)  KEEP
  GAP_UP_FADE          1d bear_edge -0.64%  (n=63)   KEEP (1-day only)

CONCLUSION: puts are NOT a standalone trend strategy. They work in 3 conditions,
and this engine fires ONLY in them:
  1. BEARISH REGIME   — SPY below its 200-SMA / regime BEAR or NEUTRAL. Never
                        fight a confirmed uptrend with puts.
  2. CATALYST         — negative VIP post (Trump/tariffs), high-impact bad news.
                        ("Concerning economic news tanks the market" — real, but
                        event-driven, not setup-driven.)
  3. VALIDATED SETUPS — overbought-in-downtrend (5d) + gap-up fade (1d).

Mirrors the call stack (rank → contract → score → short-term → monitor alerts)
but bearish, with a regime gate that keeps it holstered in bull markets.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except Exception:                       # pragma: no cover
    np = pd = yf = None

# Reuse the liquid universe + scoring brain from the call side
from momentum_strategy import LIQUID_UNIVERSE


# ── Tunables ──────────────────────────────────────────────────────────────────
OTM_PCT_MIN, OTM_PCT_MAX = 0.03, 0.12     # puts: 3-12% below spot
DTE_MIN, DTE_MAX, DTE_IDEAL = 7, 35, 18
COST_CAP_PER_CONTRACT = 6.00
MIN_VOL, MIN_OI = 50, 100
IV_HARD_CEILING = 1.40                     # puts carry skew → allow a touch higher
TAKE_PROFIT_PCT, STOP_LOSS_PCT, TIME_STOP_DAYS = 100.0, -50.0, 10
STRATEGY_LABEL = "bearish_put"

# Backtested stats for honest UI disclosure
SETUP_STATS = {
    "OVERBOUGHT_DOWNTREND": {"label": "Overbought in downtrend (RSI>70, <200SMA)",
                             "horizon": "5d", "bear_edge": "-0.52%", "n": 452},
    "GAP_UP_FADE":          {"label": "Failed gap-up (gap>2%, RSI>60, closes red)",
                             "horizon": "1d", "bear_edge": "-0.64%", "n": 63},
    "CATALYST_NEGATIVE":    {"label": "Negative catalyst (VIP/bad news)",
                             "horizon": "1-3d", "bear_edge": "event-driven", "n": "—"},
    "REGIME_BEAR":          {"label": "Bearish market regime (SPY < 200SMA)",
                             "horizon": "swing", "bear_edge": "regime tailwind", "n": "—"},
}


def _rsi(s: pd.Series, p: int = 14) -> float:
    d = s.diff(); up = d.clip(lower=0).rolling(p).mean(); dn = (-d.clip(upper=0)).rolling(p).mean()
    v = (100 - 100 / (1 + up / dn.replace(0, np.nan))).iloc[-1]
    return float(v) if pd.notna(v) else 50.0


# ══════════════════════════════════════════════════════════════════════════════
# REGIME GATE — puts only allowed when NOT in a confirmed bull
# ══════════════════════════════════════════════════════════════════════════════
def market_allows_puts() -> dict:
    """
    Returns {allowed: bool, regime: str, spy: float, sma200: float, reason: str}.
    Puts allowed when SPY is at/below its 200-SMA (BEAR/NEUTRAL). In a confirmed
    bull (SPY comfortably above 200-SMA) puts are holstered — the backtest is
    unambiguous that shorting into an uptrend loses.
    """
    if yf is None:
        return {"allowed": False, "regime": "?", "reason": "no data"}
    try:
        spy = yf.download("SPY", period="1y", interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        c = spy["Close"].dropna()
        price = float(c.iloc[-1]); sma200 = float(c.iloc[-200:].mean())
        sma50 = float(c.iloc[-50:].mean())
        # BULL if > 2% above 200SMA AND 50>200; else puts allowed
        above = (price / sma200 - 1) * 100
        if above > 2 and sma50 > sma200:
            return {"allowed": False, "regime": "BULL", "spy": round(price, 2),
                    "sma200": round(sma200, 2),
                    "reason": f"SPY {above:+.1f}% above 200-SMA — confirmed bull, "
                              f"puts holstered (shorting uptrends loses)"}
        regime = "BEAR" if (price < sma200 and sma50 < sma200) else "NEUTRAL"
        return {"allowed": True, "regime": regime, "spy": round(price, 2),
                "sma200": round(sma200, 2),
                "reason": f"SPY {above:+.1f}% vs 200-SMA — {regime}, puts active"}
    except Exception as e:
        return {"allowed": False, "regime": "?", "reason": f"err: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# BEARISH RANKING — weakest names in confirmed downtrend (put candidates)
# ══════════════════════════════════════════════════════════════════════════════
def _download(ticker: str) -> pd.DataFrame | None:
    if yf is None:
        return None
    try:
        raw = yf.download(ticker, period="1y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 210:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw.dropna(subset=["Close"])
    except Exception:
        return None


def bearish_signal(df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < 210:
        return None
    c = df["Close"]
    price = float(c.iloc[-1])
    if price <= 0:
        return None
    sma50 = float(c.iloc[-50:].mean()); sma200 = float(c.iloc[-200:].mean())
    mom_6m = (price / float(c.iloc[-127]) - 1) if len(c) > 127 else 0
    mom_3m = (price / float(c.iloc[-64]) - 1) if len(c) > 64 else 0
    rsi = _rsi(c)
    in_downtrend = price < sma50 and price < sma200 and sma50 < sma200
    return {"price": round(price, 2), "sma50": round(sma50, 2),
            "sma200": round(sma200, 2), "mom_6m": round(mom_6m, 4),
            "mom_3m": round(mom_3m, 4), "rsi": round(rsi, 1),
            "in_downtrend": in_downtrend}


def bearish_rank(conn=None, top_n: int = 10, progress=None) -> list:
    """Weakest names in a confirmed downtrend — ranked most-bearish first."""
    rows = []
    uni = [t for t in LIQUID_UNIVERSE if "." not in t]
    for i, t in enumerate(uni):
        if progress:
            try: progress(i + 1, len(uni), t)
            except Exception: pass
        df = _download(t)
        sig = bearish_signal(df)
        if not sig or not sig["in_downtrend"]:
            continue
        # ATR for stop sizing
        high, low, cl = df["High"], df["Low"], df["Close"]
        tr = pd.concat([(high-low), (high-cl.shift()).abs(), (low-cl.shift()).abs()],
                       axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1] or sig["price"]*0.02)
        price = sig["price"]
        rows.append({
            "ticker": t, "score": -sig["mom_6m"],   # most negative momentum = top
            "mom_6m": sig["mom_6m"], "mom_3m": sig["mom_3m"], "rsi": sig["rsi"],
            "price": price,
            "stop": round(price + 2*atr, 2),          # stop ABOVE for a put
            "target": round(price * 0.85, 2),         # 15% down target
            "sma50": sig["sma50"], "sma200": sig["sma200"],
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# BEARISH SHORT-TERM SETUPS (the 2 that survived the backtest)
# ══════════════════════════════════════════════════════════════════════════════
def detect_bearish_setups(ticker: str) -> list:
    df = _download(ticker)
    if df is None or len(df) < 210:
        return []
    c = df["Close"]; o = df["Open"]
    last = float(c.iloc[-1]); prev = float(c.iloc[-2]); lopen = float(o.iloc[-1])
    gap = lopen/prev - 1 if prev else 0
    ret_today = last/prev - 1 if prev else 0
    rsi = _rsi(c); sma200 = float(c.iloc[-200:].mean())
    fired = []
    if rsi > 70 and last < sma200:
        fired.append("OVERBOUGHT_DOWNTREND")
    if gap > 0.02 and _rsi(c) > 60 and ret_today < 0:
        fired.append("GAP_UP_FADE")
    return [{"ticker": ticker.upper(), "setup": s, "price": round(last, 2),
             "stats": SETUP_STATS[s]} for s in fired]


# ══════════════════════════════════════════════════════════════════════════════
# PUT CONTRACT SELECTION
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


def select_put_contract(ticker: str, spot: float) -> dict | None:
    if yf is None or spot <= 0:
        return None
    try:
        tk = yf.Ticker(ticker)
        exp = _pick_expiry(list(tk.options or []))
        if not exp:
            return None
        puts = tk.option_chain(exp).puts
        if puts is None or puts.empty:
            return None
    except Exception:
        return None
    today = date.today()
    try:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
    except Exception:
        dte = DTE_IDEAL
    lo, hi = spot*(1-OTM_PCT_MAX), spot*(1-OTM_PCT_MIN)   # strikes BELOW spot
    cand = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]
    if cand.empty:
        return None
    best, best_score = None, -1e9
    for _, r in cand.iterrows():
        def _f(v, d=0.0):
            try:
                x = float(v); return x if x == x else d
            except Exception:
                return d
        b, a = _f(r.get("bid")), _f(r.get("ask"))
        prem = (a+b)/2 if (b > 0 and a > 0) else _f(r.get("lastPrice"))
        if prem <= 0 or prem > COST_CAP_PER_CONTRACT:
            continue
        vol, oi = _f(r.get("volume")), _f(r.get("openInterest"))
        if vol < MIN_VOL and oi < MIN_OI:
            continue
        iv = _f(r.get("impliedVolatility"))
        if iv and iv > IV_HARD_CEILING:
            continue
        strike = _f(r.get("strike"))
        otm = (spot - strike)/spot
        score = (1-otm)*2 + min(oi/5000, 1) - prem*0.5 - (iv or 0)*0.3
        if score > best_score:
            best_score, best = score, {
                "ticker": ticker.upper(), "contract_symbol": str(r.get("contractSymbol", "")),
                "strike": strike, "expiry": exp, "dte": dte, "otm_pct": round(otm, 4),
                "premium": round(prem, 2), "iv": round(iv, 3),
                "volume": int(vol), "open_interest": int(oi)}
    return best


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
class PutEngine:
    def __init__(self, conn):
        self.conn = conn

    def scan(self, top_n: int = 8, progress=None, force: bool = False) -> dict:
        """
        Returns {allowed, regime, reason, plays}. Plays empty unless puts allowed
        (regime gate) OR force=True (manual override on the dashboard).
        """
        gate = market_allows_puts()
        if not gate["allowed"] and not force:
            return {**gate, "plays": []}

        plays = []
        ranked = bearish_rank(self.conn, top_n=top_n + 4, progress=progress)
        seen = set()
        for r in ranked:
            if len(plays) >= top_n:
                break
            tk = r["ticker"]
            if tk in seen:
                continue
            c = select_put_contract(tk, r["price"])
            if not c:
                continue
            c.update({"option_type": "put", "underlying_price": r["price"]})
            # Quality via the shared scoring brain (bearish thesis)
            try:
                from options_analytics import options_trade_score
                thesis = max(8.0, abs(r["mom_6m"]) / 3.0 * 100)
                qs = options_trade_score(self.conn, tk, {**c, "iv": c.get("iv", 0)},
                                         thesis_move_pct=thesis)
                c["quality_score"] = qs["score"]; c["quality_grade"] = qs["grade"]
                c["quality_decision"] = qs["decision"]
            except Exception:
                c["quality_score"] = None; c["quality_grade"] = "?"
            seen.add(tk)
            plays.append({**r, "contract": c})
        return {**gate, "plays": plays}

    def short_term_bearish(self, universe: list | None = None, progress=None) -> list:
        """The 2 validated short-term bearish setups → put plays (regime-gated)."""
        gate = market_allows_puts()
        if not gate["allowed"]:
            return []
        out = []
        uni = [t for t in (universe or LIQUID_UNIVERSE) if "." not in t]
        for i, t in enumerate(uni):
            if progress:
                try: progress(i + 1, len(uni), t)
                except Exception: pass
            for s in detect_bearish_setups(t):
                c = select_put_contract(t, s["price"])
                if c:
                    out.append({**s, "contract": c})
        return out


if __name__ == "__main__":
    print("Put engine — regime gate + bearish scan (no DB writes)\n")
    g = market_allows_puts()
    print(f"Regime: {g['regime']}  allowed={g['allowed']}")
    print(f"  {g['reason']}")
    if g["allowed"]:
        eng = PutEngine(None)
        res = eng.scan(top_n=5)
        print(f"\nPut plays: {len(res['plays'])}")
        for p in res["plays"]:
            c = p["contract"]
            print(f"  {p['ticker']:6s} 6m={p['mom_6m']*100:+.0f}% RSI={p['rsi']:.0f}  "
                  f"${c['strike']:.0f}P {c['expiry']} ${c['premium']:.2f} "
                  f"Q={c.get('quality_score')}")
    else:
        print("\n  Puts holstered by regime gate (correct in a bull market).")
        print("  Use force=True on the dashboard to override for a specific catalyst.")
