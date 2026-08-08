"""
monster_iv_robustness.py — Is the "buy cheap IV" edge REAL, or an artifact?

monster_hunter.py found low IV rank -> 1.19x more 100%+ winners. But it priced
options at IV = RV20 x 0.95, which is CIRCULAR: when RV20 is at the bottom of its
range and then mean-reverts up, the model has underpriced the option, manufacturing
"monsters" for free. Real markets know vol mean-reverts and quote IV ABOVE RV at the
lows (a vol floor), so the naive model could invent the entire edge.

This re-tests the finding under three IV models, from naive to realistic:
  M1 naive       iv = rv20 * 0.95                    (what monster_hunter used)
  M2 mean-revert iv = (0.5*rv20 + 0.5*rv252) * 0.95  (market prices toward long-run vol)
  M3 vol floor   iv = max(rv20, 0.85*rv252) * 0.95   (IV rarely prints far below long-run)

If the low-IV-rank lift SURVIVES M2/M3, the edge is real and tradeable.
If it collapses, monster_hunter's headline was an artifact of the pricing assumption.
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
            "SHOP","NET","CRWD","DDOG","SNOW","RBLX","U","ABNB","UBER","DASH",
            "GME","MARA","RIOT","CVNA","BABA","DELL","CELH","DKNG","NFLX","ROKU",
            "SPY","QQQ","IWM","XLE","XLF"]
OTM=0.05; DTE=10; STEP=3; MONSTER=100.0
R=0.04; _N=lambda x:0.5*(1+math.erf(x/math.sqrt(2)))
MIN_MID=0.15; RET_CAP=1500.0


def bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*_N(-d2)-S*_N(-d1)) if put else (S*_N(d1)-K*math.exp(-R*T)*_N(d2))
def buyf(m): return m+max(m*0.03,0.02)+0.0065
def sellf(m): return max(0.0,m-max(m*0.03,0.02)-0.0065)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(7*365.25)+200)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty or len(raw)<500: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    lr=np.log(c/c.shift())
    df["rv20"]=lr.rolling(20).std()*np.sqrt(252)
    df["rv252"]=lr.rolling(252).std()*np.sqrt(252)
    df["ivrank"]=df["rv20"].rolling(252).rank(pct=True)*100
    return df.dropna()


def run():
    print(f"Loading {len(UNIVERSE)} names, 7y...")
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")

    models={
        "M1 naive  iv=rv20":            lambda r20,r252: r20*0.95,
        "M2 mean-revert 50/50":         lambda r20,r252: (0.5*r20+0.5*r252)*0.95,
        "M3 vol floor max(rv20,.85rv252)": lambda r20,r252: max(r20,0.85*r252)*0.95,
    }
    results={k:[] for k in models}
    for t,df in data.items():
        c=df["Close"].values; r20=df["rv20"].values; r252=df["rv252"].values
        ivr=df["ivrank"].values; n=len(c)
        for i in range(260,n-DTE-1,STEP):
            S0=c[i]; S1=c[i+DTE]
            for name,fn in models.items():
                iv=max(0.10,min(2.5,fn(r20[i],r252[i])))
                for put in (False,True):
                    K=S0*(1-OTM) if put else S0*(1+OTM)
                    mid=bs(S0,K,DTE/365,iv,put)
                    if mid<MIN_MID: continue
                    entry=buyf(mid)
                    exitv=max((K-S1) if put else (S1-K),0.0)
                    ret=min((sellf(exitv)/entry-1)*100,RET_CAP)
                    results[name].append((ivr[i],ret))

    print("="*104)
    print(" DOES THE 'BUY CHEAP IV' EDGE SURVIVE A REALISTIC IV MODEL?")
    print("="*104)
    for name in models:
        arr=np.array(results[name])
        if len(arr)==0: continue
        rank=arr[:,0]; ret=arr[:,1]
        base=100*(ret>=MONSTER).mean()
        print(f"\n  {name}   (base {base:.1f}% monsters, overall exp {ret.mean():+.1f}%, n={len(ret):,})")
        for lo,hi,lab in [(0,20,"IV rank 0-20 CHEAP"),(20,40,"20-40"),(40,60,"40-60"),
                          (60,80,"60-80"),(80,101,"80+ RICH")]:
            m=(rank>=lo)&(rank<hi)
            if m.sum()<200: continue
            p=100*(ret[m]>=MONSTER).mean()
            print(f"    {lab:20s} n={m.sum():>6,}  P(monster) {p:5.1f}%  "
                  f"lift {p/base if base else 0:4.2f}x   exp {ret[m].mean():+7.1f}%")

    print("\n" + "="*104)
    print(" VERDICT")
    print("="*104)
    print("  Compare the CHEAP-vs-RICH gap across models. If M2/M3 flatten it, the")
    print("  'buy cheap vol' edge was an artifact of pricing off RV20 alone. If the")
    print("  gap persists, it is a genuine, tradeable selector.")


if __name__ == "__main__":
    run()
