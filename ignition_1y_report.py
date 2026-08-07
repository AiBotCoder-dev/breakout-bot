"""
ignition_1y_report.py — Over the PAST YEAR, how many massive runs did the ignition
system actually flag, and what would the flagged contracts have returned?

Defines a "massive run" event objectively, then checks whether the live ignition
signals fired near its start (a CATCH), and prices the contract the shadow would
have bought.

  MASSIVE UP (pump)   : a day whose forward 15-day return > +30%
  MASSIVE DOWN (crash): a day whose forward 15-day return < -25%
  (non-overlapping — each run counted once)

  DETECTION (must fire within [start-1, start+2] trading days):
    pump  = RVOL>3 AND up-day>5%
    crash = close<EMA20 AND 5-day return < -4%   (the early breakdown)

  FLAGGED CONTRACT: pump->2% OTM call 10DTE ; crash->5% OTM put 7DTE ; hold to
  expiry, net of friction, realistic (skip <$0.15 mid, cap gain +1200%).

Reports: events found, how many flagged (catch rate), the option P&L on the flagged
runs, and the notable catches/misses by name.
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
FWD = 15; UP_THR = 0.30; DN_THR = -0.25
R = 0.04; _N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
MIN_ENTRY = 0.15; RET_CAP = 1200.0; IVP = 0.95


def _bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*_N(-d2)-S*_N(-d1)) if put else (S*_N(d1)-K*math.exp(-R*T)*_N(d2))
def _buyf(m): return m+max(m*0.03,0.02)+0.0065
def _sellf(m): return max(0.0,m-max(m*0.03,0.02)-0.0065)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=500)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty or "Volume" not in raw or len(raw)<120: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["ema20"]=c.ewm(span=20).mean()
    df["rv"]=np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["ret1"]=c.pct_change(); df["ret5"]=c/c.shift(5)-1
    df["rvol"]=df["Volume"]/df["Volume"].rolling(20).mean()
    return df.dropna()


def _opt_ret(S0,S1,rv,put,otm,dte):
    iv=max(0.20,min(2.5,rv*IVP)); K=S0*(1-otm) if put else S0*(1+otm)
    mid=_bs(S0,K,dte/365,iv,put); entry=_buyf(mid)
    if mid<MIN_ENTRY or entry<=0.01: return None,K,None
    exitv=max((K-S1) if put else (S1-K),0.0)
    return min((_sellf(exitv)/entry-1)*100,RET_CAP),K,entry


def run():
    print(f"Loading {len(UNIVERSE)} names (~year + lookback)...")
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r

    up_events=[]; dn_events=[]        # each: dict(ticker, move, caught, opt_ret)
    total_pump_sig=0; total_crash_sig=0; pump_opt=[]; crash_opt=[]
    for t,df in data.items():
        C=df["Close"].values; ema=df["ema20"].values; rv=df["rv"].values
        ret1=df["ret1"].values; ret5=df["ret5"].values; rvol=df["rvol"].values
        n=len(C)
        lo=max(25, n-252-FWD); hi=n-FWD
        pump=np.array([rvol[i]>3 and ret1[i]>0.05 for i in range(n)])
        crash=np.array([ema[i]==ema[i] and C[i]<ema[i] and ret5[i]<-0.04 for i in range(n)])
        total_pump_sig+=int(pump[lo:hi].sum()); total_crash_sig+=int(crash[lo:hi].sum())
        # per-signal option outcomes (for overall expectancy incl. false alarms)
        for i in range(lo,hi):
            if pump[i]:
                r,_,_=_opt_ret(C[i],C[i+10 if i+10<n else -1],rv[i],False,0.02,10)
                if r is not None: pump_opt.append(r)
            if crash[i]:
                r,_,_=_opt_ret(C[i],C[i+7 if i+7<n else -1],rv[i],True,0.05,7)
                if r is not None: crash_opt.append(r)
        # massive-run events, non-overlapping
        last_up=-99; last_dn=-99
        for i in range(lo,hi):
            fwd=C[i+FWD]/C[i]-1
            if fwd>UP_THR and i-last_up>FWD:
                last_up=i
                caught=any(pump[max(0,i-1):i+3])
                r=None
                if caught:
                    j=next((k for k in range(max(0,i-1),i+3) if pump[k]),i)
                    r,_,_=_opt_ret(C[j],C[min(j+10,n-1)],rv[j],False,0.02,10)
                up_events.append({"t":t,"move":fwd*100,"caught":caught,"r":r})
            if fwd<DN_THR and i-last_dn>FWD:
                last_dn=i
                caught=any(crash[max(0,i-1):i+3])
                r=None
                if caught:
                    j=next((k for k in range(max(0,i-1),i+3) if crash[k]),i)
                    r,_,_=_opt_ret(C[j],C[min(j+7,n-1)],rv[j],True,0.05,7)
                dn_events.append({"t":t,"move":fwd*100,"caught":caught,"r":r})

    def _summ(name, ev):
        ntot=len(ev); nc=sum(1 for e in ev if e["caught"])
        rate=100*nc/ntot if ntot else 0
        rs=[e["r"] for e in ev if e["caught"] and e["r"] is not None]
        print(f"  {name}: {ntot} massive runs, FLAGGED {nc} ({rate:.0f}%)"
              + (f", flagged-contract avg {np.mean(rs):+.0f}%  median {np.median(rs):+.0f}%" if rs else ""))
        return ev

    print("\n"+"="*92)
    print(f" PAST-YEAR MASSIVE RUNS vs IGNITION DETECTION  ({len(data)} names)")
    print("="*92)
    _summ("PUMPS  (+30% in 15d)", up_events)
    _summ("CRASHES(-25% in 15d)", dn_events)

    print("\n  --- biggest runs and whether the system flagged them ---")
    allev=[("UP",e) for e in up_events]+[("DN",e) for e in dn_events]
    allev.sort(key=lambda x: abs(x[1]["move"]), reverse=True)
    for tag,e in allev[:16]:
        flag = (f"FLAGGED → contract {e['r']:+.0f}%" if e["caught"] and e["r"] is not None
                else "FLAGGED (contract n/a)" if e["caught"] else "missed")
        print(f"    {e['t']:6s} {tag} {e['move']:+5.0f}% over 15d   {flag}")

    print("\n"+"="*92)
    print(" OVERALL SLEEVE EXPECTANCY over the year (INCLUDING every false alarm)")
    print("="*92)
    for nm,o in [("PUMP call",pump_opt),("CRASH put",crash_opt)]:
        a=np.array(o)
        if len(a): print(f"  {nm:10s} n={len(a):<4} win {100*(a>0).mean():.0f}%  expectancy {a.mean():+.1f}%  median {np.median(a):+.0f}%")
    print("\n  READ: 'FLAGGED %' = share of the year's massive runs the detector caught")
    print("  near the start. The overall expectancy is the honest cost/benefit of ALL")
    print("  signals (the false alarms are the price of catching the real runs).")


if __name__ == "__main__":
    run()
