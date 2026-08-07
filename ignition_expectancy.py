"""
ignition_expectancy.py — Do the REBUILT (looser) triggers keep positive expectancy?

Higher catch rate came from a wider net that fires far more often — so this checks
the thing that actually matters for money: per-trade expectancy net of friction.
If the extra (junkier) signals drag expectancy toward/through zero, the wider net
isn't worth it. Compares OLD vs NEW triggers on the SAME option/exit rules.

Per signal (non-overlapping per sleeve per name):
  PUMP  -> 2% OTM CALL, 10 DTE, hold to expiry (intrinsic)
  CRASH -> 5% OTM PUT,  7 DTE, hold to expiry
Realistic: skip un-buyable mids (<$0.15), cap a single win at +1200% (exit liquidity),
friction = spread + commission each side. Reports win%, EXPECTANCY, median, worst,
tail concentration (top-5% share of gains), and total P&L on a fixed $100 ticket.
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
R = 0.04; _N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
MIN_ENTRY = 0.15; RET_CAP = 1200.0; IVP = 0.95; TICKET = 100.0


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
    exitv=max((K-S1) if put else (S1-K),0.0)
    return min((_sellf(exitv)/entry-1)*100, RET_CAP)


def _sim(data, which):
    """which in {'old','new'}. Returns {'pump':[...], 'crash':[...]} per-trade %."""
    res={"pump":[], "crash":[]}
    for t,df in data.items():
        c=df["Close"].values; ema20=df["ema20"].values; ema50=df["ema50"].values
        rv=df["rv"].values; ret1=df["ret1"].values; ret5=df["ret5"].values
        ret10=df["ret10"].values; rvol=df["rvol"].values; hi20=df["hi20"].values
        n=len(c); pu=-99; cu=-99
        for i in range(60,n-11):
            if which=="old":
                pump = rvol[i]>3 and ret1[i]>0.05
                crash = c[i]<ema20[i] and ret5[i]<-0.04
            else:
                pump = rvol[i]>1.2 and ((c[i]>=0.98*hi20[i] and c[i]>ema20[i]
                        and ema20[i]>ema50[i] and ret10[i]>0.06) or ret5[i]>0.10)
                crash = (c[i]<ema20[i] and ret5[i]<-0.04) or ret1[i]<-0.06
            if pump and i>pu:
                r=_optret(c[i],c[i+10],rv[i],False,0.02,10)
                if r is not None: res["pump"].append(r); pu=i+10
            if crash and i>cu:
                r=_optret(c[i],c[i+7],rv[i],True,0.05,7)
                if r is not None: res["crash"].append(r); cu=i+7
    return res


def _line(name, a):
    a=np.array(a)
    if len(a)==0: print(f"  {name:22s} n=0"); return
    gains=a[a>0]; top=np.sort(gains)[::-1][:max(1,int(len(a)*0.05))]
    tail=100*top.sum()/gains.sum() if gains.sum()>0 else 0
    total=a.mean()/100*TICKET*len(a)     # total $ if every signal took a $100 ticket
    print(f"  {name:22s} n={len(a):<5} win {100*(a>0).mean():3.0f}%  "
          f"EXPECTANCY {a.mean():+6.1f}%  med {np.median(a):+5.0f}%  "
          f"worst {a.min():+.0f}%  top5%={tail:3.0f}%  total ${total:+,.0f}")


def run():
    print(f"Loading {len(UNIVERSE)} names, 6y...")
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")
    old=_sim(data,"old"); new=_sim(data,"new")
    print("="*104)
    print(" EXPECTANCY — OLD (tight) vs NEW (wider net) triggers, hold-to-expiry, net of friction")
    print("="*104)
    _line("PUMP  old", old["pump"]); _line("PUMP  NEW", new["pump"])
    print()
    _line("CRASH old", old["crash"]); _line("CRASH NEW", new["crash"])
    print("\n  READ: EXPECTANCY is the number that matters. If NEW stays clearly positive,")
    print("  the wider net pays — higher catch rate WITHOUT diluting the edge to zero. If")
    print("  NEW expectancy collapses vs OLD, the extra signals are junk and we tighten.")
    print("  'total $' = paper P&L if every single signal took a $100 ticket (shows scale).")


if __name__ == "__main__":
    run()
