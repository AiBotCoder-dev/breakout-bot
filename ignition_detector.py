"""
ignition_detector.py — Can we DETECT a crash/pump igniting and profit from it?

You can't predict the monster movers in advance. But big moves start with a
signature: a volume EXPLOSION + a range-expansion day. This tests whether reacting
to that signature — buying a directional option the moment it fires — has real
expectancy, or just bleeds on false alarms.

IGNITION SIGNAL (per day):
  RVOL = volume / 20-day avg volume  >  3.0      (unusual participation)
  AND  |1-day return|  >  5%                      (a real move, not noise)
  direction = sign of the move (up = pump -> CALL, down = crash -> PUT)

For each fire we measure TWO things:
  1. CONTINUATION — the underlying's next-7-day return in the move's direction
     (does a volume-backed 5% move keep going, or mean-revert?).
  2. OPTION EXPECTANCY — buy a slightly-OTM option, ~10 DTE, hold to expiry, net of
     friction, realistic (skip un-buyable <$0.20 puts/calls, cap gain +1200%).
Split by PUMP vs CRASH, because equities crash differently than they rally.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","AMD","MU","SMCI",
            "MRVL","ARM","QCOM","INTC","TSLA","PLTR","COIN","MSTR","HOOD","SOFI",
            "AFRM","SHOP","NET","CRWD","DDOG","SNOW","RBLX","U","ABNB","UBER",
            "DASH","GME","MARA","RIOT","RDDT","CVNA","BABA","PDD","NIO","DELL",
            "AI","PATH","SOUN","CELH","ANF","DKNG","PANW","NFLX","MRNA","ROKU"]
HOLD = 7; DTE0 = 10; OTM = 0.02
RVOL_MIN = 3.0; MOVE_MIN = 0.05
R = 0.04; N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
HALF_SPREAD_PCT=0.03; MIN_HALF_SPREAD=0.02; COMM=0.0065
MIN_ENTRY=0.20; RET_CAP=1200.0; IVP=0.95


def bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*N(-d2)-S*N(-d1)) if put else (S*N(d1)-K*math.exp(-R*T)*N(d2))
def buyf(m): return m+max(m*HALF_SPREAD_PCT,MIN_HALF_SPREAD)+COMM
def sellf(m): return max(0.0,m-max(m*HALF_SPREAD_PCT,MIN_HALF_SPREAD)-COMM)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(6*365.25)+120)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty or "Volume" not in raw: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["rv"]=np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["ret1"]=c.pct_change()
    df["rvol"]=df["Volume"]/df["Volume"].rolling(20).mean()
    return df.dropna()


def run():
    print(f"Scanning {len(UNIVERSE)} names for ignition signals (RVOL>{RVOL_MIN}, |move|>{MOVE_MIN*100:.0f}%), 6y...\n")
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r

    buckets={"PUMP (up)":{"cont":[],"opt":[]}, "CRASH (down)":{"cont":[],"opt":[]}}
    for t,df in data.items():
        C=df["Close"].values; rv=df["rv"].values; ret1=df["ret1"].values
        rvol=df["rvol"].values; n=len(C); open_until=-1
        for i in range(25,n-HOLD-1):
            if i<=open_until: continue
            if rvol[i]<RVOL_MIN or abs(ret1[i])<MOVE_MIN: continue
            up = ret1[i]>0
            open_until=i+HOLD
            S0=C[i]; S1=C[i+HOLD]
            cont=(S1/S0-1)*(1 if up else -1)*100          # return in the move's direction
            iv0=max(0.20,min(2.5,rv[i]*IVP)); put=not up
            K=S0*(1+OTM) if up else S0*(1-OTM)
            mid=bs(S0,K,DTE0/365,iv0,put); entry=buyf(mid)
            if mid<MIN_ENTRY or entry<=0.01: continue
            exitv=max((K-S1) if put else (S1-K),0.0)
            r=min((sellf(exitv)/entry-1)*100, RET_CAP)
            b=buckets["PUMP (up)" if up else "CRASH (down)"]
            b["cont"].append(cont); b["opt"].append(r)

    print("="*96)
    print(f" IGNITION-DETECTION EDGE — react to the signature, buy the direction, hold {DTE0}d")
    print("="*96)
    print(f"  {'bucket':14s} {'signals':>8} {'contin.7d':>10} {'cont win':>9}   {'OPT win':>8} {'OPT exp':>9} {'OPT med':>8}")
    for name,b in buckets.items():
        c=np.array(b["cont"]); o=np.array(b["opt"])
        if len(c)==0: print(f"  {name:14s} n=0"); continue
        print(f"  {name:14s} {len(c):>8} {c.mean():+9.1f}% {100*(c>0).mean():>8.0f}%   "
              f"{100*(o>0).mean():>7.0f}% {o.mean():+8.1f}% {np.median(o):+7.0f}%")
    print("\n  READ: 'contin.7d' = does the underlying keep moving the same way after the")
    print("  ignition (positive = momentum continues; negative = it mean-reverts and the")
    print("  signal is a fade). 'OPT exp' = option expectancy net of friction. Positive")
    print("  OPT exp on a bucket = a detectable, tradeable ignition edge worth building.")


if __name__ == "__main__":
    run()
