"""
short_term_backtest.py — Find which 1-3 day setups actually have edge.

The user wants short-horizon, explosive options trades. Before building any such
strategy we test the candidate setups on 3 years of real data across a liquid
universe + index ETFs, and only the ones with measurable edge become live signals.

Setups tested (all long/bullish, since options-call focused):
  GAP_DOWN_REVERSAL   open gaps down > 2% vs prev close, RSI < 40 (panic dip-buy)
  OVERSOLD_BOUNCE     RSI(2) < 10 while above 200-SMA (pullback in uptrend)
  MOMENTUM_THRUST     close in top 15% of day's range + up day + vol > 1.5x avg
  INSIDE_DAY_BREAK    today's range inside yesterday's, then breaks above prior high
  HIGH_VOL_GAP_UP     gaps up > 3% on > 2x volume (continuation/FOMO)
  POST_PANIC_BOUNCE   prior day fell > 4% then today green (capitulation reversal)

For each trigger we record forward 1/2/3-day close-to-close returns and compute
win rate, mean, median, and the tail. Edge = mean forward return clearly above
the universe's baseline drift AND win rate > ~55%.
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
FWD = [1, 2, 3]
YEARS = 3


def _rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(period).mean()
    dn = (-d.clip(upper=0)).rolling(period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _load(ticker: str) -> pd.DataFrame | None:
    try:
        end = datetime.now(); start = end - timedelta(days=int(YEARS * 365.25) + 60)
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 220:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        df["prev_close"] = df["Close"].shift(1)
        df["prev_high"]  = df["High"].shift(1)
        df["prev_low"]   = df["Low"].shift(1)
        df["prev_range_hi"] = df["High"].shift(1)
        df["prev_range_lo"] = df["Low"].shift(1)
        df["ret_1d"]     = df["Close"].pct_change()
        df["gap"]        = df["Open"] / df["prev_close"] - 1
        df["rsi2"]       = _rsi(df["Close"], 2)
        df["rsi14"]      = _rsi(df["Close"], 14)
        df["sma200"]     = df["Close"].rolling(200).mean()
        df["vol_avg20"]  = df["Volume"].rolling(20).mean()
        df["range"]      = df["High"] - df["Low"]
        df["close_pos"]  = (df["Close"] - df["Low"]) / df["range"].replace(0, np.nan)
        return df.dropna()
    except Exception:
        return None


def _triggers(df: pd.DataFrame) -> dict:
    """Return boolean Series per setup."""
    t = {}
    t["GAP_DOWN_REVERSAL"] = (df["gap"] <= -0.02) & (df["rsi14"] < 40)
    t["OVERSOLD_BOUNCE"]   = (df["rsi2"] < 10) & (df["Close"] > df["sma200"])
    t["MOMENTUM_THRUST"]   = ((df["close_pos"] > 0.85) & (df["ret_1d"] > 0.01)
                              & (df["Volume"] > 1.5 * df["vol_avg20"]))
    t["INSIDE_DAY_BREAK"]  = ((df["prev_high"].shift(1) > df["High"].shift(1))
                              & (df["prev_low"].shift(1) < df["Low"].shift(1))
                              & (df["Close"] > df["prev_high"]))
    t["HIGH_VOL_GAP_UP"]   = (df["gap"] >= 0.03) & (df["Volume"] > 2 * df["vol_avg20"])
    t["POST_PANIC_BOUNCE"] = (df["ret_1d"].shift(1) <= -0.04) & (df["ret_1d"] > 0)
    return t


def _forward(df: pd.DataFrame, mask: pd.Series) -> list:
    """Forward close-to-close returns at FWD horizons for each trigger day."""
    out = {w: [] for w in FWD}
    idx = np.where(mask.values)[0]
    closes = df["Close"].values
    n = len(df)
    for i in idx:
        base = closes[i]
        for w in FWD:
            j = i + w
            if j < n and base > 0:
                out[w].append((closes[j] / base - 1) * 100)
    return out


def run():
    print(f"Loading {len(UNIVERSE)} tickers, {YEARS}y each...")
    data = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_load, t): t for t in UNIVERSE}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None:
                data[futs[fut]] = df
    print(f"  loaded {len(data)} tickers\n")

    # Baseline drift (random day forward returns across all names)
    base = {w: [] for w in FWD}
    for df in data.values():
        closes = df["Close"].values; n = len(closes)
        for i in range(n):
            for w in FWD:
                if i + w < n and closes[i] > 0:
                    base[w].append((closes[i + w] / closes[i] - 1) * 100)
    print("BASELINE (random entry, all names):")
    for w in FWD:
        a = np.array(base[w])
        print(f"  {w}d: mean {a.mean():+.2f}%  win {100*(a>0).mean():.1f}%  (n={len(a)})")

    # Each setup pooled across universe
    setups = ["GAP_DOWN_REVERSAL","OVERSOLD_BOUNCE","MOMENTUM_THRUST",
              "INSIDE_DAY_BREAK","HIGH_VOL_GAP_UP","POST_PANIC_BOUNCE"]
    pooled = {s: {w: [] for w in FWD} for s in setups}
    for df in data.values():
        trig = _triggers(df)
        for s in setups:
            fwd = _forward(df, trig[s].fillna(False))
            for w in FWD:
                pooled[s][w].extend(fwd[w])

    print("\n" + "=" * 78)
    print(" SHORT-TERM SETUP EDGE  (pooled across universe, 3y)")
    print("=" * 78)
    for s in setups:
        n1 = len(pooled[s][1])
        print(f"\n  {s}   (n={n1} triggers)")
        if n1 < 20:
            print("    too few triggers to judge")
            continue
        for w in FWD:
            a = np.array(pooled[s][w])
            if len(a) == 0:
                continue
            b = np.array(base[w])
            edge = a.mean() - b.mean()
            print(f"    {w}d: mean {a.mean():+.2f}%  median {np.median(a):+.2f}%  "
                  f"win {100*(a>0).mean():.1f}%  edge_vs_base {edge:+.2f}%  "
                  f"p90 {np.percentile(a,90):+.1f}%  p10 {np.percentile(a,10):+.1f}%")


if __name__ == "__main__":
    run()
