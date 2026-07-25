"""
leveraged_etf_backtest.py — Express the DIRECTIONAL edges via 3x ETFs, not options?

The reviewer's #5: for directional edges, leveraged ETFs give leverage WITHOUT
theta/IV/spread. The catch they glossed: 3x ETFs have VOLATILITY DECAY from daily
rebalancing. We use the REAL TQQQ/SQQQ price history, so that decay is already
baked in — no assumption. Two of our directional edges are index-expressible:

  1. OVERNIGHT EDGE  — buy the close, sell the open. Test QQQ vs TQQQ (3x).
  2. TREND FOLLOWING — hold when QQQ > rising 50 & 200 SMA (the momentum regime),
                       else cash. Test QQQ vs TQQQ, and report MAX DRAWDOWN so the
                       decay/whipsaw cost is visible, not hidden.

Honest metrics: total return, per-trade/overnight stats, AND max drawdown +
volatility (leverage cuts both ways — the point is the risk-adjusted picture).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

YEARS = 6


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(YEARS*365.25)+260)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy()
    df["sma50"]=df["Close"].rolling(50).mean(); df["sma200"]=df["Close"].rolling(200).mean()
    return df.dropna()


def _maxdd(equity):
    e=np.asarray(equity,float); peak=np.maximum.accumulate(e)
    return float(((e-peak)/peak).min()*100)


def overnight(df):
    """Buy close, sell next open. Returns array of overnight % returns."""
    o=df["Open"].values; c=df["Close"].values
    return (o[1:]/c[:-1]-1)*100


def trend_follow(df):
    """Long when Close>sma50>sma200 (yesterday's signal, no lookahead); else flat.
    Returns (total_ret%, maxdd%, pct_time_in, ann_vol%)."""
    c=df["Close"].values; s50=df["sma50"].values; s200=df["sma200"].values
    dr=c[1:]/c[:-1]-1                      # daily returns
    sig=(c[:-1]>s50[:-1]) & (s50[:-1]>s200[:-1])   # in-market flag for each next-day return
    strat=np.where(sig, dr, 0.0)
    eq=np.cumprod(1+strat)
    return (eq[-1]-1)*100, _maxdd(eq), 100*sig.mean(), float(np.std(strat)*np.sqrt(252)*100)


def run():
    print(f"Loading QQQ, TQQQ, SQQQ, {YEARS}y...")
    q=_load("QQQ"); t=_load("TQQQ"); s=_load("SQQQ")
    if q is None or t is None:
        print("load failed"); return
    # align
    common = q.index.intersection(t.index)
    q2=q.loc[common]; t2=t.loc[common]
    print(f"  aligned {len(common)} days\n")

    print("="*70)
    print(" 1) OVERNIGHT EDGE — buy close / sell open")
    print("="*70)
    for name,df in [("QQQ (1x)",q2),("TQQQ (3x)",t2)]:
        on=overnight(df)
        cum=(np.prod(1+on/100)-1)*100
        eq=np.cumprod(1+on/100)
        print(f"  {name:11s}: overnight total {cum:+.0f}%  win {100*(on>0).mean():.1f}%  "
              f"avg {on.mean():+.3f}%/night  maxDD {_maxdd(eq):.0f}%")

    print("\n"+"="*70)
    print(" 2) TREND FOLLOWING — hold in uptrend (>50>200 SMA), else cash")
    print("="*70)
    for name,df in [("QQQ (1x)",q2),("TQQQ (3x)",t2)]:
        r,dd,ti,vol=trend_follow(df)
        print(f"  {name:11s}: total {r:+.0f}%  maxDD {dd:.0f}%  vol {vol:.0f}%  "
              f"in-market {ti:.0f}%  return/DD {abs(r/dd) if dd else 0:.2f}")
    # buy-and-hold benchmarks
    bhq=(q2['Close'].iloc[-1]/q2['Close'].iloc[0]-1)*100
    bht=(t2['Close'].iloc[-1]/t2['Close'].iloc[0]-1)*100
    eqq=q2['Close'].values/q2['Close'].values[0]; eqt=t2['Close'].values/t2['Close'].values[0]
    print(f"  {'QQQ buy&hold':11s}: total {bhq:+.0f}%  maxDD {_maxdd(eqq):.0f}%")
    print(f"  {'TQQQ buy&hold':11s}: total {bht:+.0f}%  maxDD {_maxdd(eqt):.0f}%  "
          f"(<- the decay/drawdown reality of raw 3x)")

    print("\n  READ: leverage helps total return but the maxDD tells the honest story.")
    print("  Trend-following the 3x (vs holding it) is the test of whether the timing")
    print("  tames the decay. Compare return/DD across rows.")


if __name__ == "__main__":
    run()
