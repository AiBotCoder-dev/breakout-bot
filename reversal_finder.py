"""
reversal_finder.py — Live "downtrend -> base -> uptrend" watchlist.

Backtest (reversal_backtest.py, 88 names, 5y) verdict:
  BASE_BREAKOUT  -> UNDERPERFORMS (breakouts from bases fail).  REJECTED.
  RECLAIM_50SMA  -> 62.6% win @60d, median +5.1%, controlled downside.  USE.
  SMA200_SLOPE   -> real edge but rare (n=22).  Used as a confirming bonus.

So this finder uses the VALIDATED criteria — fell hard, based tightly without new
lows, then reclaimed a flattening/rising 50-SMA (NOT a naive base breakout), with
the 200-SMA slope improving as a confirmation bonus.

It is a WATCHLIST / idea generator, not an auto-trader — the edge is modest. Each
candidate is stage-classified so you can act with discipline:
  BASING     — in the base, downtrend arrested, NOT yet turned up (watch)
  TRIGGERED  — just reclaimed the 50-SMA (the validated entry)
  EXTENDED   — triggered a while ago, already run (chase risk)
"""

from __future__ import annotations

from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except Exception:                       # pragma: no cover
    np = pd = yf = None

from momentum_strategy import LIQUID_UNIVERSE

BASE_LEN = 40
BACKTEST_STATS = {"win_60d": 62.6, "median_60d": 5.1, "n": 92,
                  "note": "RECLAIM_50SMA variant — validated; base-breakout was rejected"}


def _load(t: str):
    if yf is None:
        return None
    try:
        raw = yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 300:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        df["sma50"] = df["Close"].rolling(50).mean()
        df["sma200"] = df["Close"].rolling(200).mean()
        return df.dropna()
    except Exception:
        return None


def classify(df) -> dict | None:
    """Classify the current reversal stage for one ticker, or None if not a setup."""
    if df is None or len(df) < 260:
        return None
    C = df["Close"].values; H = df["High"].values; L = df["Low"].values
    sma50 = df["sma50"].values; sma200 = df["sma200"].values
    i = len(C) - 1
    price = float(C[i])

    # 1) prior downtrend — >=18% below high from 5-12 months ago
    prior_high = H[i-252:i-105].max() if i-252 >= 0 else H[:i-105].max()
    if prior_high <= 0:
        return None
    drawdown = (price / prior_high - 1) * 100
    if drawdown > -18:
        return None

    # 2) base — tight range, no fresh lows in 2nd half
    base = C[i-BASE_LEN:i]
    if len(base) < BASE_LEN:
        return None
    bmin, bmax = float(base.min()), float(base.max())
    range_pct = (bmax / bmin - 1) * 100 if bmin > 0 else 999
    if range_pct > 22:
        return None
    half = BASE_LEN // 2
    if L[i-half:i].min() < L[i-BASE_LEN:i-half].min() * 0.99:
        return None

    # 200-SMA slope (improving = bonus)
    slope200 = (sma200[i] / sma200[i-20] - 1) * 100 if sma200[i-20] > 0 else 0
    if slope200 < -3:                     # still strongly downtrending — too early
        return None
    slope_improving = slope200 > (sma200[i-1]/sma200[i-21]-1)*100 if sma200[i-21] > 0 else False

    # 3) trigger state — 50-SMA reclaim
    rising50 = sma50[i] >= sma50[i-10]
    above50_now = price > sma50[i]
    # how many bars ago did it cross above 50SMA?
    crossed_ago = None
    for k in range(1, 25):
        if i-k-1 >= 0 and C[i-k] > sma50[i-k] and C[i-k-1] <= sma50[i-k-1]:
            crossed_ago = k
            break
    just_crossed = (price > sma50[i] and C[i-1] <= sma50[i-1])

    if just_crossed and rising50:
        stage = "TRIGGERED"
    elif above50_now and crossed_ago is not None and crossed_ago <= 8 and rising50:
        stage = "TRIGGERED"
    elif above50_now and crossed_ago is not None and crossed_ago > 8:
        stage = "EXTENDED"
    elif not above50_now:
        stage = "BASING"
    else:
        stage = "BASING"

    # entry/stop/target
    atr = float(pd.concat([
        (df["High"]-df["Low"]),
        (df["High"]-df["Close"].shift()).abs(),
        (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
        or price*0.02)
    stop = round(min(bmin * 0.98, price - 1.5*atr), 2)     # below the base
    target = round(price * 1.25, 2)

    return {
        "stage": stage,
        "price": round(price, 2),
        "drawdown_pct": round(drawdown, 1),
        "base_range_pct": round(range_pct, 1),
        "sma50": round(float(sma50[i]), 2),
        "sma200": round(float(sma200[i]), 2),
        "slope200_20d": round(slope200, 2),
        "slope_improving": bool(slope_improving),
        "above_50": above50_now,
        "crossed_ago": crossed_ago,
        "stop": stop, "target": target,
        "entry": round(price, 2),
    }


class ReversalFinder:
    def __init__(self, conn=None, universe=None):
        self.conn = conn
        self.universe = [t for t in (universe or LIQUID_UNIVERSE) if "." not in t]

    def scan(self, progress=None) -> list:
        out = []
        def _one(t):
            df = _load(t)
            c = classify(df)
            return (t, c)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_one, t): t for t in self.universe}
            done = 0
            for fut in as_completed(futs):
                done += 1
                if progress:
                    try: progress(done, len(futs), futs[fut])
                    except Exception: pass
                t, c = fut.result()
                if c:
                    out.append({"ticker": t, **c})
        # Order: TRIGGERED first, then BASING, then EXTENDED; within, best slope
        order = {"TRIGGERED": 0, "BASING": 1, "EXTENDED": 2}
        out.sort(key=lambda r: (order.get(r["stage"], 9), -r["slope200_20d"]))
        return out


if __name__ == "__main__":
    print("Scanning liquid universe for downtrend->base->uptrend reversals...\n")
    res = ReversalFinder().scan()
    by = {}
    for r in res:
        by.setdefault(r["stage"], []).append(r)
    for stage in ["TRIGGERED", "BASING", "EXTENDED"]:
        rows = by.get(stage, [])
        print(f"\n=== {stage} ({len(rows)}) ===")
        for r in rows[:12]:
            print(f"  {r['ticker']:6s} ${r['price']:>8.2f}  DD {r['drawdown_pct']:>6.1f}%  "
                  f"base {r['base_range_pct']:>4.1f}%  200slope {r['slope200_20d']:>+5.1f}%  "
                  f"{'improving' if r['slope_improving'] else ''}")
