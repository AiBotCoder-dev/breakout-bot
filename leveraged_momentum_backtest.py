"""
leveraged_momentum_backtest.py — Does momentum via 3x SECTOR ETFs preserve edge?

The reviewer's Q1: our edge is a cross-sectional STOCK tilt, but the tradeable
leveraged vehicles are index/sector-level. Does rotating into the strongest 3x
SECTOR ETF (ride the leader) keep alpha, or wash into beta? And does the trend
filter tame the 3x decay/drawdown?

STRATEGY (weekly rebalance, real 3x ETF history so decay is inherent):
  • Universe of liquid 3x SECTOR ETFs.
  • Each week: rank by 63-day momentum; HOLD the top-K that are ALSO above their
    own 50 & 200 SMA (uptrend filter). Equal-weight. Cash if none qualify.
  • Compare to SPY buy-hold and QQQ buy-hold on total return, maxDD, Sortino,
    and % time in cash (the defensive value).

This is the exact "leveraged-shares expression of the directional edge" the
instrument test (reviewer_tests.py) said should beat options on survival.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Liquid 3x sector/index ETFs (decay is inherent in their real price history)
ETFS = ["TQQQ","SOXL","SPXL","FNGU","TECL","FAS","TNA","LABU","CURE","UDOW","RETL","DPST"]
TOP_K = 2
YEARS = 8


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(YEARS*365.25)+260)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty or len(raw)<300: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["sma50"]=c.rolling(50).mean(); df["sma200"]=c.rolling(200).mean()
    df["mom63"]=c/c.shift(63)-1
    return df.dropna()


def _maxdd(eq):
    e=np.asarray(eq,float); pk=np.maximum.accumulate(e); return float(((e-pk)/pk).min()*100)
def _sortino(dr):
    a=np.asarray(dr,float); dn=a[a<0]; dd=np.sqrt(np.mean(dn**2)) if len(dn) else 1e-9
    return float(a.mean()/dd*np.sqrt(252)) if dd>0 else 0.0


def run():
    print(f"Loading {len(ETFS)} 3x ETFs + SPY/QQQ, {YEARS}y...")
    data={}
    for t in ETFS+["SPY","QQQ"]:
        d=_load(t)
        if d is not None: data[t]=d
    spy=data.pop("SPY"); qqq=data.pop("QQQ")
    etfs={k:v for k,v in data.items()}
    print(f"  loaded {len(etfs)} ETFs\n")
    # common trading calendar = SPY index
    cal=spy.index
    # build daily return of the rotation strategy
    eq=1.0; curve=[]; dr=[]; in_cash=0; days=0
    # weekly rebalance: pick holdings on Mondays (or every 5th bar)
    holdings=[]
    for k in range(210,len(cal)):
        d=cal[k]
        if k%5==0:   # weekly rebalance
            ranked=[]
            for t,df in etfs.items():
                if d in df.index:
                    i=df.index.get_loc(d)
                    if i>200:
                        c=df["Close"].values; s50=df["sma50"].values; s200=df["sma200"].values; m=df["mom63"].values
                        if c[i]>s50[i] and c[i]>s200[i] and m[i]==m[i]:
                            ranked.append((m[i],t))
            ranked.sort(reverse=True)
            holdings=[t for _,t in ranked[:TOP_K]]
        # daily return of current holdings
        if not holdings:
            r=0.0; in_cash+=1
        else:
            rs=[]
            for t in holdings:
                df=etfs[t]
                if d in df.index:
                    i=df.index.get_loc(d)
                    if i>0: rs.append(df["Close"].values[i]/df["Close"].values[i-1]-1)
            r=float(np.mean(rs)) if rs else 0.0
        eq*=(1+r); curve.append(eq); dr.append(r); days+=1

    tot=(eq-1)*100
    print("="*72); print(" LEVERAGED SECTOR-MOMENTUM ROTATION (top-2 3x ETFs in uptrend, weekly)"); print("="*72)
    print(f"  total return {tot:+.0f}%   maxDD {_maxdd(curve):.0f}%   Sortino {_sortino(dr):.2f}   "
          f"in-cash {100*in_cash/days:.0f}% of days")
    # benchmarks over same window
    def bench(df,name):
        c=df["Close"]; sub=c[c.index>=cal[210]]
        t=(sub.iloc[-1]/sub.iloc[0]-1)*100; e=(sub/sub.iloc[0]).values
        d=sub.pct_change().dropna().values
        print(f"  {name:14s} total {t:+.0f}%   maxDD {_maxdd(e):.0f}%   Sortino {_sortino(d):.2f}")
    bench(spy,"SPY buy&hold"); bench(qqq,"QQQ buy&hold")
    print("\n  READ: compare Sortino + maxDD. If the rotation's Sortino beats SPY/QQQ")
    print("  with a survivable drawdown, the leveraged-ETF expression preserves edge.")


if __name__ == "__main__":
    run()
