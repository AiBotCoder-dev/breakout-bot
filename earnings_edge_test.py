"""
earnings_edge_test.py — Is there a REAL earnings edge: implied move vs the move the
stock actually delivers?

The previous test showed earnings lift monsters 1.12-1.31x, but the whole result hung
on a UNIFORM event-IV premium. That is circular: applying one multiplier to every name
underprices the high-move names (NVDA) and overprices the sleepy ones (KO), which
manufactures exactly the cross-sectional "edge" we'd be looking for. Real markets price
each name's event roughly in line with its own history.

SO THIS TEST REMOVES THAT ARTIFACT: the option's implied move is CALIBRATED PER NAME to
that stock's OWN historical average earnings move (point-in-time — only earnings that had
already happened). Total implied move is combined properly:
      implied_total = sqrt( (base_vol_move)^2 + (hist_avg_earnings_move * k)^2 )
      iv = implied_total / sqrt(T)
k sweeps how richly the market prices the event: 0.9 (cheap), 1.0 (fair), 1.2 (rich).

With the market pricing each name fairly, ANY remaining edge must come from predicting
when a move will EXCEED its own history. We test three candidate predictors:
  trend2   — are this name's last 2 earnings moves bigger than its long-run average?
  rvrank   — pre-earnings realized-vol percentile
  runup    — absolute 10-day price run into the event

If none of them separate, then earnings magnitude is unpredictable and there is no
selectable edge beyond simply having a catalyst.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","AMD","MU","SMCI",
            "MRVL","ARM","QCOM","INTC","TSLA","PLTR","COIN","HOOD","SOFI","SHOP",
            "NET","CRWD","DDOG","SNOW","RBLX","U","ABNB","UBER","DASH","CVNA",
            "BABA","DELL","CELH","DKNG","NFLX","ROKU","MRNA","PANW","ANF","GME",
            "JPM","V","MA","UNH","LLY","WMT","COST","HD","CAT","BA","XOM","CVX",
            "ORCL","CRM","ADBE","PYPL","SQ","TTD","ZS","OKTA"]
OTM=0.05; DTE=10; MONSTER=100.0
R=0.04; _N=lambda x:0.5*(1+math.erf(x/math.sqrt(2)))
MIN_MID=0.15; RET_CAP=1500.0
K_SWEEP=(0.9,1.0,1.2)
MIN_PRIOR=4          # need >=4 prior earnings to estimate the name's typical move


def bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*_N(-d2)-S*_N(-d1)) if put else (S*_N(d1)-K*math.exp(-R*T)*_N(d2))
def buyf(m): return m+max(m*0.03,0.02)+0.0065
def sellf(m): return max(0.0,m-max(m*0.03,0.02)-0.0065)


def _load(t):
    try:
        end=datetime.now(); start=end-timedelta(days=int(8*365.25)+200)
        raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
        if raw is None or raw.empty or len(raw)<500: return None
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
        df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
        lr=np.log(c/c.shift())
        df["rv20"]=lr.rolling(20).std()*np.sqrt(252)
        df["rv252"]=lr.rolling(252).std()*np.sqrt(252)
        df["ivrank"]=df["rv20"].rolling(252).rank(pct=True)*100
        df["ret1"]=c.pct_change()
        df["run10"]=c/c.shift(10)-1
        df=df.dropna()
        try:
            ed=yf.Ticker(t).get_earnings_dates(limit=60)
            ed=pd.to_datetime(ed.index).tz_localize(None).normalize() if ed is not None and len(ed) else None
        except Exception:
            ed=None
        if ed is None or len(ed)==0: return None
        return df, sorted(set(ed))
    except Exception:
        return None


def run():
    print(f"Loading {len(UNIVERSE)} names + earnings calendars, 8y...")
    data={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")

    samples=[]     # dicts: k, ret, trend2, rvrank, runup, cheapness
    nonearn=[]     # baseline: same structure, no earnings in life
    for t,(df,edates) in data.items():
        idx=df.index.normalize(); c=df["Close"].values
        r20=df["rv20"].values; r252=df["rv252"].values
        ivr=df["ivrank"].values; ret1=df["ret1"].values; run10=df["run10"].values
        n=len(c)
        pos={d:i for i,d in enumerate(idx)}
        # realized move for each earnings date: biggest 1-day move in [d, d+1]
        emoves=[]
        for d in edates:
            i=pos.get(d)
            if i is None or i+1>=n: continue
            emoves.append((i, max(abs(ret1[i]), abs(ret1[i+1]))))
        emoves.sort()
        # walk events, using only PRIOR earnings for the estimate (point-in-time)
        for j,(ei,realized) in enumerate(emoves):
            if j<MIN_PRIOR: continue
            prior=[m for _,m in emoves[:j]]
            hist=float(np.mean(prior))
            if hist<=0: continue
            entry_i=ei-3                      # buy ~3 days before the event
            if entry_i<260 or entry_i+DTE>=n: continue
            S0=c[entry_i]; S1=c[entry_i+DTE]
            base_move=max(0.10,min(2.5,(0.5*r20[entry_i]+0.5*r252[entry_i])*0.95))*math.sqrt(DTE/365)
            trend2=float(np.mean(prior[-2:]))/hist
            for k in K_SWEEP:
                implied=math.sqrt(base_move**2 + (hist*k)**2)
                iv=implied/math.sqrt(DTE/365)
                for put in (False,True):
                    K=S0*(1-OTM) if put else S0*(1+OTM)
                    mid=bs(S0,K,DTE/365,iv,put)
                    if mid<MIN_MID: continue
                    e=buyf(mid); x=max((K-S1) if put else (S1-K),0.0)
                    samples.append({"k":k,"ret":min((sellf(x)/e-1)*100,RET_CAP),
                                    "trend2":trend2,"rvrank":ivr[entry_i],
                                    "runup":abs(run10[entry_i]),
                                    "realized_vs_hist":realized/hist})
        # non-earnings baseline
        eset={i for i,_ in emoves}
        for i in range(260,n-DTE-1,7):
            if any((i+d) in eset or (i+d-1) in eset for d in range(1,DTE+1)): continue
            S0=c[i]; S1=c[i+DTE]
            iv=max(0.10,min(2.5,(0.5*r20[i]+0.5*r252[i])*0.95))
            for put in (False,True):
                K=S0*(1-OTM) if put else S0*(1+OTM)
                mid=bs(S0,K,DTE/365,iv,put)
                if mid<MIN_MID: continue
                e=buyf(mid); x=max((K-S1) if put else (S1-K),0.0)
                nonearn.append(min((sellf(x)/e-1)*100,RET_CAP))

    sdf=pd.DataFrame(samples); nb=np.array(nonearn)
    print("="*104)
    print(" BASELINE (no earnings in the option's life)")
    print("="*104)
    print(f"  n={len(nb):,}  P(monster) {100*(nb>=MONSTER).mean():.1f}%  expectancy {nb.mean():+.1f}%")

    print("\n" + "="*104)
    print(" EARNINGS PRICED AT EACH NAME'S OWN HISTORY (no cross-sectional artifact)")
    print("="*104)
    for k in K_SWEEP:
        s=sdf[sdf["k"]==k]
        tag={0.9:"market underprices (cheap)",1.0:"market prices it FAIRLY",1.2:"market overprices (rich)"}[k]
        print(f"  k={k:.1f}  {tag:28s} n={len(s):>5,}  "
              f"P(monster) {100*(s['ret']>=MONSTER).mean():5.1f}%  exp {s['ret'].mean():+7.1f}%")

    print("\n" + "="*104)
    print(" CAN ANYTHING PREDICT AN ABOVE-AVERAGE EARNINGS MOVE?  (k=1.0, fairly priced)")
    print("="*104)
    fair=sdf[sdf["k"]==1.0]
    base=100*(fair["ret"]>=MONSTER).mean()
    print(f"  (base within fairly-priced earnings: {base:.1f}% monsters)")
    for col,edges,labs in [
        ("trend2",[(0,.8),(.8,1.0),(1.0,1.25),(1.25,9)],
         ["last 2 moves SMALL","slightly small","slightly big","last 2 moves BIG"]),
        ("rvrank",[(0,30),(30,60),(60,85),(85,101)],
         ["vol rank low","mid","high","vol rank very high"]),
        ("runup",[(0,.03),(.03,.07),(.07,.15),(.15,9)],
         ["quiet into event","mild run","big run","huge run into event"]),
    ]:
        print(f"\n  {col}")
        for (lo,hi),lab in zip(edges,labs):
            g=fair[(fair[col]>=lo)&(fair[col]<hi)]
            if len(g)<150: continue
            p=100*(g["ret"]>=MONSTER).mean()
            print(f"    {lab:22s} n={len(g):>5,}  P(monster) {p:5.1f}%  lift {p/base if base else 0:4.2f}x"
                  f"   exp {g['ret'].mean():+7.1f}%")

    print("\n" + "="*104)
    print(" VERDICT")
    print("="*104)
    print("  k=1.0 vs baseline answers: is a CATALYST alone worth it once fairly priced?")
    print("  The predictor table answers: can we pick WHICH earnings will over-deliver?")
    print("  Lifts near 1.0x mean earnings magnitude is unpredictable -> no selectable edge.")


if __name__ == "__main__":
    run()
