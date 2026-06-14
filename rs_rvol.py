"""
rs_rvol.py — Relative-Strength + Relative-Volume score bonus for MOMENTUM setups.

Backtest (rs_rvol_backtest.py, 57 names, 6y, fwd 10d) verdict:
  MOMENTUM/uptrend longs:
    RVOL >= 1.3x        -> +1.7pp win (55.1->56.8), mean +1.00->+1.59%   USE
    RS>0 AND RVOL>=1.3x -> +2.0pp win (57.2%), mean +1.68%                BEST
    RS alone            -> negligible (momentum already beats SPY)
  DIP/bottom-fisher longs:
    RS and RVOL both HURT (-4 to -10pp) — you WANT laggards on quiet volume.
    => DO NOT call this for bottom_fisher / reversal sources.

So this returns a small additive score BONUS (not a hard filter — keeps the data
sample intact) to be applied ONLY to momentum-type candidates.
"""

from __future__ import annotations

import time

try:
    import yfinance as yf
    import pandas as pd
except Exception:                       # pragma: no cover
    yf = pd = None

_spy_cache = {"ts": 0.0, "ret63": None}
_MAX_BONUS = 10


def _spy_3m_return() -> float | None:
    """Cached SPY 63-trading-day return (refresh hourly)."""
    if yf is None:
        return None
    if _spy_cache["ret63"] is not None and (time.time() - _spy_cache["ts"]) < 3600:
        return _spy_cache["ret63"]
    try:
        df = yf.download("SPY", period="6mo", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = df["Close"].dropna()
        ret = float(c.iloc[-1] / c.iloc[-64] - 1) if len(c) >= 64 else 0.0
        _spy_cache.update(ts=time.time(), ret63=ret)
        return ret
    except Exception:
        return None


def compute(ticker: str) -> dict:
    """
    Returns {rvol, rs_diff_pct, bonus, label} for `ticker`.
      rvol      = today's volume / 20-day average
      rs_diff   = name 3-mo return minus SPY 3-mo return (percentage points)
      bonus     = additive score points (0..+10) for momentum setups
    Safe on any failure (returns zero bonus).
    """
    out = {"rvol": 0.0, "rs_diff_pct": 0.0, "bonus": 0, "label": "RS/RVOL n/a"}
    if yf is None:
        return out
    try:
        df = yf.download(ticker, period="6mo", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if len(df) < 64:
            return out
        c = df["Close"]; v = df["Volume"]
        rvol = float(v.iloc[-1] / v.iloc[-20:].mean()) if v.iloc[-20:].mean() > 0 else 0.0
        name_ret = float(c.iloc[-1] / c.iloc[-64] - 1)
        spy_ret = _spy_3m_return() or 0.0
        rs_diff = (name_ret - spy_ret) * 100

        bonus = 0
        # RVOL is the dominant driver (+1.7pp win in the backtest)
        if rvol >= 2.0:
            bonus += 9
        elif rvol >= 1.3:
            bonus += 6
        elif rvol < 0.7:
            bonus -= 3                       # dead-volume move — downweight
        # RS is a small confirming bonus on top
        if rs_diff > 5:
            bonus += 4
        elif rs_diff > 0:
            bonus += 2
        bonus = max(-3, min(_MAX_BONUS, bonus))

        out.update(rvol=round(rvol, 2), rs_diff_pct=round(rs_diff, 1), bonus=bonus,
                   label=f"RVOL {rvol:.1f}x · RS {rs_diff:+.0f}pp vs SPY")
        return out
    except Exception:
        return out


# sources that are UPTREND/momentum-type (RS/RVOL helps). Dip sources excluded.
MOMENTUM_SOURCES = {"momentum", "pead", "whale", "vip", "mover", "panic"}
DIP_SOURCES      = {"bottom_fisher", "reversal", "put_engine", "news_put",
                    "exhaustion_put"}


def applies_to(sources) -> bool:
    """True if the candidate is momentum-type (so the bonus should apply)."""
    s = set(sources or [])
    if s & DIP_SOURCES:          # any dip source present -> don't apply
        return False
    return bool(s & MOMENTUM_SOURCES)


if __name__ == "__main__":
    for t in ["NVDA", "AMD", "SOFI", "INTC"]:
        print(t, compute(t))
