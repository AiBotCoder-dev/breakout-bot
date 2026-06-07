"""
overextended_backtest.py — Do overextended stocks actually mean-revert (puts)?

Shorting parabolic/overbought names is dangerous (overbought gets more overbought),
so we TEST it before building. For PUTS we want forward returns that are NEGATIVE
(price pulls back) with a fat LEFT tail (p10 deeply negative = where a put 3-10x's).

Setups tested (overextension flavors):
  PARABOLIC        up > 20% over the last 10 days
  RSI2_EXTREME     RSI(2) > 98  (extreme short-term overbought)
  STRETCHED_EMA    close > 15% above the 20-EMA
  CONSEC_UP        5+ consecutive up days (buying climax)
  UPPER_BB_PIERCE  close > upper Bollinger(20,2) by > 3% of price
  GAP_UP_FAIL      gap up > 3% then closes red (failed pop)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# High-beta names that actually get parabolic + broad large caps
UNIVERSE = [
    "COIN","SMCI","ARM","TSLA","PLTR","NVDA","AMD","MU","MARA","RIOT","MSTR",
    "SHOP","NFLX","META","AMAT","QCOM","CRWD","NET","DDOG","SNOW","RBLX","HOOD",
    "UBER","SOFI","AFRM","DKNG","ROKU","SQ","PYPL","ABNB","INTC","AVGO","ORCL",
    "AAPL","MSFT","AMZN","GOOGL","JPM","BAC","XOM","CVX","CAT","BA","DIS","WMT",
    "COST","HD","UNH","LLY","MRK","IONQ","RGTI","AI","SOUN","BBAI","RCAT","AVAV",
]
FWD = [1, 3, 5]
YEARS = 4


def _rsi(s, p):
    d = s.diff(); up = d.clip(lower=0).rolling(p).mean(); dn = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))


def _load(t):
    try:
        end = datetime.now(); start = end - timedelta(days=int(YEARS*365.25)+60)
        raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 120:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        c = df["Close"]
        df["ret1"] = c.pct_change()
        df["ret10"] = c.pct_change(10)
        df["rsi2"] = _rsi(c, 2)
        df["ema20"] = c.ewm(span=20, adjust=False).mean()
        df["stretch"] = c / df["ema20"] - 1
        df["gap"] = df["Open"]/c.shift(1) - 1
        sma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
        df["bb_up"] = sma20 + 2*sd20
        up = (df["ret1"] > 0).astype(int)
        df["consec_up"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
        return df.dropna()
    except Exception:
        return None


def _triggers(df):
    t = {}
    t["PARABOLIC"]       = df["ret10"] > 0.20
    t["RSI2_EXTREME"]    = df["rsi2"] > 98
    t["STRETCHED_EMA"]   = df["stretch"] > 0.15
    t["CONSEC_UP"]       = df["consec_up"] >= 5
    t["UPPER_BB_PIERCE"] = df["Close"] > df["bb_up"] * 1.03
    t["GAP_UP_FAIL"]     = (df["gap"] > 0.03) & (df["ret1"] < 0)
    return t


def _fwd(df, mask):
    cl = df["Close"].values; n = len(cl); out = {w: [] for w in FWD}
    for i in np.where(mask.fillna(False).values)[0]:
        for w in FWD:
            if i+w < n and cl[i] > 0:
                out[w].append((cl[i+w]/cl[i]-1)*100)
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
    print("BASELINE (random entry, high-beta universe):")
    for w in FWD:
        a = np.array(base[w]); print(f"  {w}d: mean {a.mean():+.2f}%  down-rate {100*(a<0).mean():.1f}%")

    setups = ["PARABOLIC","RSI2_EXTREME","STRETCHED_EMA","CONSEC_UP","UPPER_BB_PIERCE","GAP_UP_FAIL"]
    pooled = {s:{w:[] for w in FWD} for s in setups}
    for df in data.values():
        trig = _triggers(df)
        for s in setups:
            f = _fwd(df, trig[s])
            for w in FWD: pooled[s][w].extend(f[w])

    print("\n" + "="*92)
    print(" OVEREXTENDED SHORT EDGE  (forward NEGATIVE = mean-reverts = good for puts)")
    print("="*92)
    for s in setups:
        n1 = len(pooled[s][1])
        print(f"\n  {s}  (n={n1})")
        if n1 < 25:
            print("    too few"); continue
        for w in FWD:
            a = np.array(pooled[s][w]); b = np.array(base[w])
            print(f"    {w}d: mean {a.mean():+.2f}%  down-rate {100*(a<0).mean():.1f}%  "
                  f"bear_edge {a.mean()-b.mean():+.2f}%  p10 {np.percentile(a,10):+.1f}%  "
                  f"p90 {np.percentile(a,90):+.1f}%")


if __name__ == "__main__":
    run()
