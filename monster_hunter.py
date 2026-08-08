"""
monster_hunter.py — What separates the 100%+ option winners from everything else?

GOAL (the user's actual objective): don't try to make average option trades
profitable — instead RECOGNISE, out of hundreds of candidates, the rare few that
can return 100%+. If we can beat the base rate meaningfully, convexity does the rest.

WHY THIS STUDY IS MORE TRUSTWORTHY THAN THE EXPECTANCY ONES: it's a RANKING
question. A wrong IV assumption shifts every bucket in roughly the same direction,
so the *relative lift* of one setup over another survives even if the absolute
numbers are off. That's exactly what we need for a selector.

METHOD
  Sample entries across a liquid universe every few days, 7y. At each one compute a
  feature vector, then price a lottery-style contract (5% OTM, 10 DTE) in BOTH
  directions and hold to expiry. Label MONSTER = return >= +100%.
  Then measure, per feature bucket: P(monster) and the LIFT vs the base rate.
  Finally combine the best separators into a rule and report catch rate, precision,
  and the expectancy of the selected subset.

Features tested (all computable live, no paid data):
  ivrank    percentile of RV20 in trailing year  (cheap vol -> expansion?)
  rv20      absolute realized vol                 (does the name even move?)
  rvol      volume surge vs 20d average
  ret5      recent 5-day thrust
  rng_pos   position in the 20-day range
  trend     price vs sma50/sma200
  dist52    distance below the 52-week high
  atr_exp   short-vol / long-vol ratio (vol already expanding?)
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
            "AI","PATH","SOUN","CELH","ANF","DKNG","PANW","NFLX","MRNA","ROKU",
            "SPY","QQQ","IWM","XLE","XLF"]
OTM = 0.05; DTE = 10; STEP = 3          # sample every 3 trading days
MONSTER = 100.0                          # >= +100% return
R = 0.04; _N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
MIN_MID = 0.15; RET_CAP = 1500.0; IVP = 0.95


def bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*_N(-d2)-S*_N(-d1)) if put else (S*_N(d1)-K*math.exp(-R*T)*_N(d2))
def buyf(m): return m+max(m*0.03,0.02)+0.0065
def sellf(m): return max(0.0,m-max(m*0.03,0.02)-0.0065)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(7*365.25)+200)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty or "Volume" not in raw or len(raw)<400: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    lr=np.log(c/c.shift())
    df["rv20"]=lr.rolling(20).std()*np.sqrt(252)
    df["rv60"]=lr.rolling(60).std()*np.sqrt(252)
    df["ivrank"]=df["rv20"].rolling(252).rank(pct=True)*100
    df["sma50"]=c.rolling(50).mean(); df["sma200"]=c.rolling(200).mean()
    df["ret5"]=c/c.shift(5)-1
    df["rvol"]=df["Volume"]/df["Volume"].rolling(20).mean()
    df["hi20"]=c.rolling(20).max(); df["lo20"]=c.rolling(20).min()
    df["hi252"]=c.rolling(252).max()
    return df.dropna()


def _collect():
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    rows=[]
    for t,df in data.items():
        c=df["Close"].values; rv=df["rv20"].values; rv60=df["rv60"].values
        ivr=df["ivrank"].values; s50=df["sma50"].values; s200=df["sma200"].values
        r5=df["ret5"].values; rvol=df["rvol"].values
        hi20=df["hi20"].values; lo20=df["lo20"].values; hi252=df["hi252"].values
        n=len(c)
        for i in range(260,n-DTE-1,STEP):
            S0=c[i]; S1=c[i+DTE]
            iv=max(0.15,min(2.5,rv[i]*IVP))
            rng=(S0-lo20[i])/(hi20[i]-lo20[i]) if hi20[i]>lo20[i] else 0.5
            base={"ivrank":ivr[i],"rv20":rv[i],"rvol":rvol[i],"ret5":r5[i],
                  "rng":rng,"trend":1.0 if (S0>s50[i]>s200[i]) else 0.0,
                  "dist52":(S0/hi252[i]-1)*100,
                  "atr_exp":(rv[i]/rv60[i]) if rv60[i]>0 else 1.0}
            for put in (False,True):
                K=S0*(1-OTM) if put else S0*(1+OTM)
                mid=bs(S0,K,DTE/365,iv,put)
                if mid<MIN_MID: continue
                entry=buyf(mid)
                exitv=max((K-S1) if put else (S1-K),0.0)
                ret=min((sellf(exitv)/entry-1)*100, RET_CAP)
                rows.append({**base,"put":1.0 if put else 0.0,"ret":ret})
    return pd.DataFrame(rows)


def _lift(df, col, edges, labels, direction=None):
    sub = df if direction is None else df[df["put"]==direction]
    base = 100*(sub["ret"]>=MONSTER).mean()
    print(f"\n  {col}   (base rate {base:.1f}% monsters, n={len(sub):,})")
    for (lo,hi),lab in zip(edges,labels):
        s=sub[(sub[col]>=lo)&(sub[col]<hi)]
        if len(s)<200: continue
        p=100*(s["ret"]>=MONSTER).mean()
        print(f"    {lab:22s} n={len(s):>6,}  P(monster) {p:5.1f}%  "
              f"lift {p/base if base>0 else 0:4.2f}x   exp {s['ret'].mean():+7.1f}%")


def run():
    print(f"Sampling {len(UNIVERSE)} names, 7y, every {STEP}d, {int(OTM*100)}% OTM / {DTE} DTE...")
    df=_collect()
    print(f"  {len(df):,} option samples  ({100*(df['ret']>=MONSTER).mean():.1f}% were 100%+ monsters)\n")
    print("="*100)
    print(" WHICH FEATURES SEPARATE THE 100%+ WINNERS?   (lift = vs base rate)")
    print("="*100)
    _lift(df,"ivrank",[(0,20),(20,40),(40,60),(60,80),(80,101)],
          ["IV rank 0-20 (cheap)","IV rank 20-40","IV rank 40-60","IV rank 60-80","IV rank 80+ (rich)"])
    _lift(df,"rv20",[(0,.30),(.30,.45),(.45,.65),(.65,.90),(.90,9)],
          ["vol <30%","vol 30-45%","vol 45-65%","vol 65-90%","vol 90%+"])
    _lift(df,"atr_exp",[(0,.8),(.8,1.0),(1.0,1.2),(1.2,1.5),(1.5,9)],
          ["vol contracting <0.8","0.8-1.0","1.0-1.2","1.2-1.5","vol expanding 1.5+"])
    _lift(df,"rvol",[(0,.8),(.8,1.2),(1.2,2.0),(2.0,3.0),(3.0,99)],
          ["volume quiet <0.8","0.8-1.2","1.2-2.0","2.0-3.0","volume surge 3x+"])
    _lift(df,"ret5",[(-9,-.10),(-.10,-.03),(-.03,.03),(.03,.10),(.10,9)],
          ["5d thrust <-10%","-10..-3%","flat","+3..+10%","5d thrust >+10%"])
    _lift(df,"rng",[(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01)],
          ["at 20d LOW","low-mid","middle","mid-high","at 20d HIGH"])
    _lift(df,"dist52",[(-99,-40),(-40,-20),(-20,-10),(-10,-3),(-3,1)],
          ["-40%+ below 52wk hi","-40..-20%","-20..-10%","-10..-3%","at 52wk high"])
    print("\n" + "="*100)
    print(" SPLIT BY DIRECTION (0=calls, 1=puts)")
    print("="*100)
    for d,lab in [(0.0,"CALLS"),(1.0,"PUTS")]:
        s=df[df["put"]==d]
        print(f"  {lab}: n={len(s):,}  P(monster) {100*(s['ret']>=MONSTER).mean():.1f}%  "
              f"expectancy {s['ret'].mean():+.1f}%")
    df.to_pickle("monster_samples.pkl")
    print("\n  saved monster_samples.pkl for rule-building")


if __name__ == "__main__":
    run()
