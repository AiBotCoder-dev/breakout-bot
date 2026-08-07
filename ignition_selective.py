"""
ignition_selective.py — Make it MANUALLY EXECUTABLE: take only the top-N signals/day.

The rebuilt triggers fire thousands of times — impossible to copy by hand. But this
is a tail strategy (a few big winners carry it), so the real question is: if you rank
every day's signals by conviction and take ONLY the strongest 1-3, do you keep the
expectancy while cutting volume to a handful a week?

Conviction score (bigger = stronger ignition, more likely to be a real monster):
  pump  = 10-day thrust  x  RVOL
  crash = (5-day drop, or the down-day)  x  RVOL

For N in {take-all, 3/day, 2/day, 1/day} we report: trades PER WEEK (executable?),
expectancy, and what share of the year's massive runs still get caught. Option:
pump 2% OTM call 10DTE / crash 5% OTM put 7DTE, hold to expiry, net of friction.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

UNIVERSE = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","AMD","MU","SMCI",
            "MRVL","ARM","QCOM","INTC","TSLA","PLTR","COIN","MSTR","HOOD","SOFI",
            "AFRM","SHOP","NET","CRWD","DDOG","SNOW","RBLX","U","ABNB","UBER",
            "DASH","GME","MARA","RIOT","RDDT","CVNA","BABA","PDD","NIO","DELL",
            "AI","PATH","SOUN","CELH","ANF","DKNG","PANW","NFLX","MRNA","ROKU"]
R=0.04; _N=lambda x:0.5*(1+math.erf(x/math.sqrt(2)))
MIN_ENTRY=0.15; RET_CAP=1200.0; IVP=0.95; FWD=15; UP_THR=0.30; DN_THR=-0.25


def _bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*_N(-d2)-S*_N(-d1)) if put else (S*_N(d1)-K*math.exp(-R*T)*_N(d2))
def _buyf(m): return m+max(m*0.03,0.02)+0.0065
def _sellf(m): return max(0.0,m-max(m*0.03,0.02)-0.0065)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(6*365.25)+120)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty or "Volume" not in raw or len(raw)<300: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["ema20"]=c.ewm(span=20).mean(); df["ema50"]=c.ewm(span=50).mean()
    df["rv"]=np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["ret1"]=c.pct_change(); df["ret5"]=c/c.shift(5)-1; df["ret10"]=c/c.shift(10)-1
    df["rvol"]=df["Volume"]/df["Volume"].rolling(20).mean(); df["hi20"]=c.rolling(20).max()
    return df.dropna()


def _optret(S0,S1,rv,put,otm,dte):
    iv=max(0.20,min(2.5,rv*IVP)); K=S0*(1-otm) if put else S0*(1+otm)
    mid=_bs(S0,K,dte/365,iv,put); entry=_buyf(mid)
    if mid<MIN_ENTRY or entry<=0.01: return None
    return min((_sellf(max((K-S1) if put else (S1-K),0.0))/entry-1)*100, RET_CAP)


def run():
    print(f"Loading {len(UNIVERSE)} names, 6y...")
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")

    # collect all signals (non-overlapping per name+sleeve), tag if it caught a run
    sigs=[]          # dict(date, name, sleeve, score, ret, caught_run)
    runs_up=0; runs_dn=0
    for t,df in data.items():
        c=df["Close"].values; ema20=df["ema20"].values; ema50=df["ema50"].values
        rv=df["rv"].values; ret1=df["ret1"].values; ret5=df["ret5"].values
        ret10=df["ret10"].values; rvol=df["rvol"].values; hi20=df["hi20"].values
        idx=df.index; n=len(c); pu=-99; cu=-99
        # mark massive-run start days (for catch measurement)
        up_starts=set(); dn_starts=set(); lu=-99; ld=-99
        for i in range(60,n-FWD):
            fwd=c[i+FWD]/c[i]-1
            if fwd>UP_THR and i-lu>FWD: up_starts.add(i); lu=i;
            if fwd<DN_THR and i-ld>FWD: dn_starts.add(i); ld=i
        runs_up+=len(up_starts); runs_dn+=len(dn_starts)
        for i in range(60,n-11):
            pump = rvol[i]>1.2 and ((c[i]>=0.98*hi20[i] and c[i]>ema20[i]
                    and ema20[i]>ema50[i] and ret10[i]>0.06) or ret5[i]>0.10)
            crash = (c[i]<ema20[i] and ret5[i]<-0.04) or ret1[i]<-0.06
            if pump and i>pu:
                r=_optret(c[i],c[i+10],rv[i],False,0.02,10)
                if r is not None:
                    caught=any(s in up_starts for s in range(i-5,i+3))
                    sigs.append({"d":idx[i],"t":t,"sleeve":"pump","score":ret10[i]*min(rvol[i],6),
                                 "ret":r,"caught":caught}); pu=i+10
            if crash and i>cu:
                r=_optret(c[i],c[i+7],rv[i],True,0.05,7)
                if r is not None:
                    sev=max(-ret5[i],-ret1[i]); caught=any(s in dn_starts for s in range(i-5,i+3))
                    sigs.append({"d":idx[i],"t":t,"sleeve":"crash","score":sev*min(rvol[i],6),
                                 "ret":r,"caught":caught}); cu=i+7

    # total span in weeks
    dates=sorted(set(s["d"] for s in sigs)); weeks=max(1,(dates[-1]-dates[0]).days/7)

    def _eval(sel):
        by={"pump":[],"crash":[]}; caught={"pump":0,"crash":0}
        for s in sel:
            by[s["sleeve"]].append(s["ret"])
            if s["caught"]: caught[s["sleeve"]]+=1
        return by, caught, len(sel)

    def _pick_topN(perday):
        byday=defaultdict(list)
        for s in sigs: byday[s["d"]].append(s)
        out=[]
        for d,lst in byday.items():
            out.extend(sorted(lst,key=lambda x:x["score"],reverse=True)[:perday])
        return out

    print("="*104)
    print(" SELECTIVITY — take only the top-N conviction signals per day (manual-executable?)")
    print("="*104)
    print(f"  {'strategy':16s} {'trades/wk':>9} {'pump n':>7} {'pump exp':>9} {'crash n':>8} "
          f"{'crash exp':>10} {'runs caught':>18}")
    for name,perday in [("take ALL",9999),("top 3/day",3),("top 2/day",2),("top 1/day",1)]:
        sel = sigs if perday==9999 else _pick_topN(perday)
        by,caught,ntot=_eval(sel)
        p=np.array(by["pump"]); c_=np.array(by["crash"])
        pexp=p.mean() if len(p) else 0; cexp=c_.mean() if len(c_) else 0
        cr=f"P {100*caught['pump']//max(runs_up,1)}% / C {100*caught['crash']//max(runs_dn,1)}%"
        print(f"  {name:16s} {ntot/weeks:>8.1f} {len(p):>7} {pexp:>+8.1f}% {len(c_):>8} "
              f"{cexp:>+9.1f}% {cr:>18}")
    print(f"\n  ({runs_up} massive pumps + {runs_dn} massive crashes exist in the 6y window.)")
    print("  READ: 'trades/wk' must be small enough to place by hand. If expectancy stays")
    print("  high as N shrinks, the CONVICTION RANK works — a few strong picks keep the edge")
    print("  while cutting volume to something you can actually copy. Catch rate drops (you")
    print("  skip weaker setups), but for a manual trader executable > exhaustive.")


if __name__ == "__main__":
    run()
