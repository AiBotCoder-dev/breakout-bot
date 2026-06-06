"""
reversal_backtest.py — Does "downtrend -> base -> uptrend" actually have edge?

The user's setup is Weinstein Stage 1->2: a stock that FELL, then STABILISED into
a low-volatility base, then TURNED UP. The discipline (vs catching a falling knife)
is requiring the base + the upturn trigger — you don't buy the decline, you buy
the base breaking upward.

We define it objectively and test 3 trigger variants across a broad universe over
~5y, measuring forward 20/40/60-day returns vs baseline. Ship a live finder ONLY
if it shows real edge.

PATTERN (evaluated at bar i):
  1. PRIOR DOWNTREND  — price is >= 18% below its high from [i-252, i-105]
                        (it fell hard from a high made 5-12 months ago)
  2. STABILISED BASE  — over the last BASE_LEN bars: range (max/min - 1) < 0.22
                        (tight consolidation) AND the 2nd-half low >= 1st-half low
                        (stopped making new lows) AND price near/above flattening
                        200-SMA-ish (declining slope arrested)
  3. UPTURN TRIGGER (variant):
       A. BASE_BREAKOUT  close > max(high of base window)
       B. RECLAIM_50SMA  close crosses above a flattening/rising 50-SMA
       C. SMA200_SLOPE   200-SMA slope flips from negative to positive
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = [
    "SPY","QQQ","IWM","AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","TSLA",
    "AVGO","NFLX","CRM","ADBE","INTC","CSCO","QCOM","TXN","MU","AMAT","ORCL",
    "PLTR","COIN","SHOP","UBER","PYPL","SQ","ROKU","SNAP","PINS","DKNG","RBLX",
    "JPM","BAC","WFC","GS","MS","C","V","MA","AXP","SCHW",
    "UNH","LLY","JNJ","ABBV","MRK","PFE","TMO","ABT","BMY","GILD","MRNA","BIIB",
    "WMT","COST","HD","MCD","NKE","SBUX","DIS","KO","PG","TGT","LOW",
    "CAT","BA","GE","HON","UPS","XOM","CVX","COP","SLB","FCX","NEM",
    "F","GM","DAL","AAL","CCL","NCLH","RIVN","LCID","CHPT","PLUG","SOFI","HOOD",
]
FWD = [20, 40, 60]
YEARS = 5
BASE_LEN = 40


def _load(t):
    try:
        end = datetime.now(); start = end - timedelta(days=int(YEARS*365.25)+400)
        raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 400:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        df["sma50"] = df["Close"].rolling(50).mean()
        df["sma200"] = df["Close"].rolling(200).mean()
        return df.dropna()
    except Exception:
        return None


def _signals(df, variant):
    """Return boolean array: reversal trigger fired at bar i."""
    C = df["Close"].values; H = df["High"].values; L = df["Low"].values
    sma50 = df["sma50"].values; sma200 = df["sma200"].values
    n = len(C)
    sig = np.zeros(n, dtype=bool)
    for i in range(260, n):
        # 1) prior downtrend: >=18% below high from [i-252, i-105]
        hi_window = H[i-252:i-105]
        if len(hi_window) == 0:
            continue
        prior_high = hi_window.max()
        if prior_high <= 0 or (C[i] / prior_high - 1) > -0.18:
            continue
        # 2) stabilised base over last BASE_LEN bars (excluding today)
        base = C[i-BASE_LEN:i]
        if len(base) < BASE_LEN:
            continue
        bmin, bmax = base.min(), base.max()
        if bmin <= 0 or (bmax / bmin - 1) > 0.22:      # tight range
            continue
        half = BASE_LEN // 2
        if L[i-half:i].min() < L[i-BASE_LEN:i-half].min() * 0.99:  # still making lows
            continue
        # 200-SMA slope arrested (not steeply falling)
        slope200 = (sma200[i] / sma200[i-20] - 1) if sma200[i-20] > 0 else 0
        if slope200 < -0.03:                            # still strongly downtrending
            continue
        # 3) trigger variant
        fired = False
        if variant == "BASE_BREAKOUT":
            fired = C[i] > bmax and C[i-1] <= bmax
        elif variant == "RECLAIM_50SMA":
            rising50 = sma50[i] >= sma50[i-10]
            fired = (C[i] > sma50[i] and C[i-1] <= sma50[i-1] and rising50)
        elif variant == "SMA200_SLOPE":
            slope_prev = (sma200[i-1]/sma200[i-21]-1) if sma200[i-21] > 0 else 0
            fired = (slope200 > 0 and slope_prev <= 0)
        if fired:
            sig[i] = True
    return sig


def _forward(df, sig):
    C = df["Close"].values; n = len(C)
    out = {w: [] for w in FWD}
    for i in np.where(sig)[0]:
        for w in FWD:
            if i+w < n and C[i] > 0:
                out[w].append((C[i+w]/C[i]-1)*100)
    return out


def run():
    print(f"Loading {len(UNIVERSE)} tickers, {YEARS}y...")
    data = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_load, t): t for t in UNIVERSE}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None:
                data[futs[fut]] = df
    print(f"  loaded {len(data)}\n")

    base = {w: [] for w in FWD}
    for df in data.values():
        C = df["Close"].values; n = len(C)
        for i in range(260, n):
            for w in FWD:
                if i+w < n and C[i] > 0:
                    base[w].append((C[i+w]/C[i]-1)*100)
    print("BASELINE (random entry):")
    for w in FWD:
        a = np.array(base[w]); print(f"  {w}d: mean {a.mean():+.2f}%  win {100*(a>0).mean():.1f}%")

    print("\n" + "="*86)
    print(" REVERSAL SETUP EDGE  (downtrend -> base -> uptrend)")
    print("="*86)
    for variant in ["BASE_BREAKOUT","RECLAIM_50SMA","SMA200_SLOPE"]:
        pooled = {w: [] for w in FWD}
        for df in data.values():
            sig = _signals(df, variant)
            f = _forward(df, sig)
            for w in FWD: pooled[w].extend(f[w])
        n1 = len(pooled[FWD[0]])
        print(f"\n  {variant}  (n={n1} signals)")
        if n1 < 20:
            print("    too few"); continue
        for w in FWD:
            a = np.array(pooled[w]); b = np.array(base[w])
            print(f"    {w}d: mean {a.mean():+.2f}%  median {np.median(a):+.2f}%  "
                  f"win {100*(a>0).mean():.1f}%  edge {a.mean()-b.mean():+.2f}%  "
                  f"p90 {np.percentile(a,90):+.1f}%  p10 {np.percentile(a,10):+.1f}%")


if __name__ == "__main__":
    run()
