"""
bearish_backtest.py — Find which bearish setups actually have put edge.

The user believes puts have a strong edge. The data will settle it. We test six
bearish setups across 34 liquid names over 3y. For PUTS we want forward returns
that are NEGATIVE (price falls) with a fat LEFT tail (p10 deeply negative — that's
where a put 5-10x's). A setup only has put edge if its forward return is clearly
BELOW the universe's (upward) baseline drift AND the left tail is fat.

Setups (all bearish):
  DOWNTREND_BREAKDOWN  close < 50 & 200 SMA, breaks below 20-day low
  RELATIVE_WEAKNESS    6m return < -10% AND below 200-SMA (persistent weakness)
  GAP_UP_FADE          gaps up > 2%, RSI14 > 60, closes RED (failed pop)
  OVERBOUGHT_DOWNTREND RSI14 > 70 while below 200-SMA (bear-rally exhaustion)
  DEATH_CROSS_FRESH    50-SMA crossed below 200-SMA within last 5 days
  LOWER_HIGH_REJECT    new 10-day-lower high, closes in bottom 25% of range
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = [
    "SPY","QQQ","IWM","AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","TSLA",
    "AVGO","NFLX","CRM","ORCL","COIN","PLTR","MU","SMCI","MRVL","SHOP","UBER",
    "BAC","JPM","XOM","CVX","CAT","BA","DIS","SBUX","NKE","COST","LLY","UNH",
]
FWD = [1, 3, 5]
YEARS = 3


def _rsi(s: pd.Series, p: int) -> pd.Series:
    d = s.diff(); up = d.clip(lower=0).rolling(p).mean(); dn = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _load(ticker: str) -> pd.DataFrame | None:
    try:
        end = datetime.now(); start = end - timedelta(days=int(YEARS*365.25)+260)
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 260:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        df["prev_close"] = df["Close"].shift(1)
        df["ret_1d"] = df["Close"].pct_change()
        df["ret_6m"] = df["Close"].pct_change(126)
        df["gap"] = df["Open"] / df["prev_close"] - 1
        df["rsi14"] = _rsi(df["Close"], 14)
        df["sma50"] = df["Close"].rolling(50).mean()
        df["sma200"] = df["Close"].rolling(200).mean()
        df["low20"] = df["Low"].rolling(20).min()
        df["high10"] = df["High"].rolling(10).max()
        df["range"] = df["High"] - df["Low"]
        df["close_pos"] = (df["Close"] - df["Low"]) / df["range"].replace(0, np.nan)
        return df.dropna()
    except Exception:
        return None


def _triggers(df: pd.DataFrame) -> dict:
    t = {}
    t["DOWNTREND_BREAKDOWN"] = ((df["Close"] < df["sma50"]) & (df["Close"] < df["sma200"])
                                & (df["Close"] < df["low20"].shift(1)))
    t["RELATIVE_WEAKNESS"]   = (df["ret_6m"] < -0.10) & (df["Close"] < df["sma200"])
    t["GAP_UP_FADE"]         = (df["gap"] > 0.02) & (df["rsi14"] > 60) & (df["ret_1d"] < 0)
    t["OVERBOUGHT_DOWNTREND"]= (df["rsi14"] > 70) & (df["Close"] < df["sma200"])
    t["DEATH_CROSS_FRESH"]   = ((df["sma50"] < df["sma200"])
                                & (df["sma50"].shift(5) >= df["sma200"].shift(5)))
    t["LOWER_HIGH_REJECT"]   = ((df["High"] < df["high10"].shift(1)) & (df["close_pos"] < 0.25)
                                & (df["Close"] < df["sma50"]))
    return t


def _forward(df: pd.DataFrame, mask: pd.Series) -> dict:
    out = {w: [] for w in FWD}; idx = np.where(mask.values)[0]; cl = df["Close"].values; n = len(cl)
    for i in idx:
        base = cl[i]
        for w in FWD:
            j = i + w
            if j < n and base > 0:
                out[w].append((cl[j]/base - 1) * 100)
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
        cl = df["Close"].values; n = len(cl)
        for i in range(n):
            for w in FWD:
                if i+w < n and cl[i] > 0:
                    base[w].append((cl[i+w]/cl[i]-1)*100)
    print("BASELINE (random entry):")
    for w in FWD:
        a = np.array(base[w]); print(f"  {w}d: mean {a.mean():+.2f}%  down-rate {100*(a<0).mean():.1f}%")

    setups = ["DOWNTREND_BREAKDOWN","RELATIVE_WEAKNESS","GAP_UP_FADE",
              "OVERBOUGHT_DOWNTREND","DEATH_CROSS_FRESH","LOWER_HIGH_REJECT"]
    pooled = {s:{w:[] for w in FWD} for s in setups}
    for df in data.values():
        trig = _triggers(df)
        for s in setups:
            f = _forward(df, trig[s].fillna(False))
            for w in FWD: pooled[s][w].extend(f[w])

    print("\n" + "="*82)
    print(" BEARISH SETUP EDGE  (forward return NEGATIVE = good for puts; fat p10 = put 5-10x)")
    print("="*82)
    for s in setups:
        n1 = len(pooled[s][FWD[0]])
        print(f"\n  {s}  (n={n1})")
        if n1 < 20:
            print("    too few"); continue
        for w in FWD:
            a = np.array(pooled[s][w]); b = np.array(base[w])
            edge = a.mean() - b.mean()  # negative = bearish edge
            print(f"    {w}d: mean {a.mean():+.2f}%  down-rate {100*(a<0).mean():.1f}%  "
                  f"bear_edge {edge:+.2f}%  p10 {np.percentile(a,10):+.1f}%  p90 {np.percentile(a,90):+.1f}%")


if __name__ == "__main__":
    run()
