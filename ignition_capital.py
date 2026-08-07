"""
ignition_capital.py — How much capital does the tail strategy need to SURVIVE?

A tail-hunter loses ~65-75% of trades (each a ~-100% premium loss) and only makes
money on rare big winners that may not land for weeks. The danger isn't the edge —
it's going BROKE during a losing streak before a winner arrives. And options have a
floor: the cheapest real bet is ~1 contract (~$100), so a small account is FORCED to
bet a big fraction each time -> ruin.

This bootstraps the strategy's own per-trade returns (top-2/day selective, pump+crash
pooled, realistic caps) and simulates a year of trading (~8 trades/wk) at each starting
capital, betting a fixed ~$100/contract. Reports risk of RUIN (can't place a bet) and
the outcome spread — so 'how much capital' is answered with the strategy's real numbers.
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
MIN_ENTRY=0.15; RET_CAP=1200.0; IVP=0.95
TICKET=100.0; TRADES_PER_YR=400; N_PATHS=20000
np.random.seed(11)


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


def _selective_returns(data):
    sigs=[]
    for t,df in data.items():
        c=df["Close"].values; ema20=df["ema20"].values; ema50=df["ema50"].values
        rv=df["rv"].values; ret1=df["ret1"].values; ret5=df["ret5"].values
        ret10=df["ret10"].values; rvol=df["rvol"].values; hi20=df["hi20"].values
        idx=df.index; n=len(c); pu=-99; cu=-99
        for i in range(60,n-11):
            pump=rvol[i]>1.2 and ((c[i]>=0.98*hi20[i] and c[i]>ema20[i] and ema20[i]>ema50[i] and ret10[i]>0.06) or ret5[i]>0.10)
            crash=(c[i]<ema20[i] and ret5[i]<-0.04) or ret1[i]<-0.06
            if pump and i>pu:
                r=_optret(c[i],c[i+10],rv[i],False,0.02,10)
                if r is not None: sigs.append((idx[i],ret10[i]*min(rvol[i],6),r)); pu=i+10
            if crash and i>cu:
                r=_optret(c[i],c[i+7],rv[i],True,0.05,7)
                if r is not None: sigs.append((idx[i],max(-ret5[i],-ret1[i])*min(rvol[i],6),r)); cu=i+7
    byday=defaultdict(list)
    for d,sc,r in sigs: byday[d].append((sc,r))
    sel=[]
    for d,lst in byday.items():
        for sc,r in sorted(lst,reverse=True)[:2]: sel.append(r)
    return np.array(sel)


def run():
    print(f"Loading {len(UNIVERSE)} names, 6y...")
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    rets=_selective_returns(data)
    print(f"  {len(rets)} top-2/day trades  ·  win {100*(rets>0).mean():.0f}%  "
          f"expectancy {rets.mean():+.1f}%  (bet size ${TICKET:.0f})\n")

    print("="*84)
    print(f" RUIN & OUTCOMES over ~1 year ({TRADES_PER_YR} trades, ${TICKET:.0f}/bet, {N_PATHS:,} paths)")
    print("="*84)
    print(f"  {'capital':>9} {'bet as %':>9} {'RUIN risk':>10} {'median end':>11} {'bad(P10)':>10} {'good(P90)':>10}")
    for C0 in (500,1000,2500,5000,10000,25000):
        ends=np.empty(N_PATHS); ruin=0
        for p in range(N_PATHS):
            C=C0; dead=False
            draws=rets[np.random.randint(0,len(rets),TRADES_PER_YR)]
            for r in draws:
                if C<TICKET: dead=True; break
                C+=TICKET*r/100.0
            if dead or C<TICKET: ruin+=1
            ends[p]=max(C,0)
        p10,p50,p90=np.percentile(ends,[10,50,90])
        print(f"  ${C0:>8,} {100*TICKET/C0:>8.0f}% {100*ruin/N_PATHS:>9.0f}% "
              f"${p50:>10,.0f} ${p10:>9,.0f} ${p90:>9,.0f}")
    print("\n  RUIN = fell below one bet ($100) and can no longer play. NOTE: expectancy is")
    print("  positive, so with ENOUGH capital the median grows — but too little capital gets")
    print("  wiped by the losing streak BEFORE the tail winner lands. That gap is the answer.")
    print("  (Backtest magnitude is optimistic re: real pump/crash-day IV, so treat required")
    print("  capital as a FLOOR, not a ceiling.)")


if __name__ == "__main__":
    run()
