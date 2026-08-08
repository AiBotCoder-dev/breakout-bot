"""
earnings_monster_test.py — Do options SPANNING EARNINGS produce more 100%+ winners?

The strongest untested monster-generator. Earnings are the one scheduled event that
reliably gaps a stock 5-20% overnight — exactly what turns a 5% OTM option into a
100%+ winner. And unlike vol regimes, earnings dates are KNOWN IN ADVANCE, so this
is directly selectable: out of hundreds of candidates, buy the ones with a catalyst.

THE HONEST CATCH (and why the last finding was an artifact): the market PRICES the
event. A 10-DTE option spanning earnings trades at a big IV premium, then crushes
after. If we price earnings options off ordinary realized vol we'd underpay for them
and manufacture a fake edge — the exact trap that killed the "buy cheap IV" result.
So we SWEEP the earnings IV premium: 1.0x (naive/free), 1.3x, 1.6x (realistic).
If earnings still produce excess monsters after paying a 60% IV premium, it is real.

Exits are at EXPIRY (intrinsic), so post-event IV crush cannot distort the exit —
only the entry price carries the event premium, which is the honest treatment.
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
            "BABA","DELL","CELH","DKNG","NFLX","ROKU","MRNA","PANW","ANF","GME"]
OTM=0.05; DTE=10; STEP=3; MONSTER=100.0
R=0.04; _N=lambda x:0.5*(1+math.erf(x/math.sqrt(2)))
MIN_MID=0.15; RET_CAP=1500.0
EARN_MULTS=(1.0,1.3,1.6)          # sweep the event IV premium


def bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*_N(-d2)-S*_N(-d1)) if put else (S*_N(d1)-K*math.exp(-R*T)*_N(d2))
def buyf(m): return m+max(m*0.03,0.02)+0.0065
def sellf(m): return max(0.0,m-max(m*0.03,0.02)-0.0065)


def _load(t):
    try:
        end=datetime.now(); start=end-timedelta(days=int(7*365.25)+200)
        raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
        if raw is None or raw.empty or len(raw)<500: return None
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
        df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
        lr=np.log(c/c.shift())
        df["rv20"]=lr.rolling(20).std()*np.sqrt(252)
        df["rv252"]=lr.rolling(252).std()*np.sqrt(252)
        df=df.dropna()
        # earnings dates
        try:
            ed=yf.Ticker(t).get_earnings_dates(limit=60)
            edates=set(pd.to_datetime(ed.index).tz_localize(None).normalize()) if ed is not None and len(ed) else set()
        except Exception:
            edates=set()
        if not edates: return None
        return df, edates
    except Exception:
        return None


def run():
    print(f"Loading {len(UNIVERSE)} names + earnings calendars, 7y...")
    data={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)} with earnings data\n")

    # rows: (spans_earnings, mult_index, return)
    res={m:{True:[],False:[]} for m in EARN_MULTS}
    for t,(df,edates) in data.items():
        c=df["Close"].values; r20=df["rv20"].values; r252=df["rv252"].values
        idx=df.index.normalize(); n=len(c)
        eflag=np.array([d in edates for d in idx])
        for i in range(260,n-DTE-1,STEP):
            spans=bool(eflag[i+1:i+DTE+1].any())
            S0=c[i]; S1=c[i+DTE]
            base_iv=max(0.10,min(2.5,(0.5*r20[i]+0.5*r252[i])*0.95))   # realistic M2
            for m in EARN_MULTS:
                iv=base_iv*(m if spans else 1.0)
                for put in (False,True):
                    K=S0*(1-OTM) if put else S0*(1+OTM)
                    mid=bs(S0,K,DTE/365,iv,put)
                    if mid<MIN_MID: continue
                    entry=buyf(mid)
                    exitv=max((K-S1) if put else (S1-K),0.0)
                    res[m][spans].append(min((sellf(exitv)/entry-1)*100,RET_CAP))

    print("="*100)
    print(" DO EARNINGS-SPANNING OPTIONS PRODUCE MORE 100%+ WINNERS?")
    print("="*100)
    for m in EARN_MULTS:
        e=np.array(res[m][True]); ne=np.array(res[m][False])
        if len(e)==0 or len(ne)==0: continue
        pe=100*(e>=MONSTER).mean(); pn=100*(ne>=MONSTER).mean()
        tag={1.0:"NAIVE (event priced free)",1.3:"realistic +30% IV",1.6:"conservative +60% IV"}[m]
        print(f"\n  earnings IV premium x{m:.1f}  — {tag}")
        print(f"    SPANS earnings   n={len(e):>6,}  P(monster) {pe:5.1f}%  exp {e.mean():+7.1f}%  "
              f"median {np.median(e):+6.0f}%")
        print(f"    no earnings      n={len(ne):>6,}  P(monster) {pn:5.1f}%  exp {ne.mean():+7.1f}%  "
              f"median {np.median(ne):+6.0f}%")
        print(f"    --> LIFT {pe/pn if pn>0 else 0:.2f}x monsters,  "
              f"expectancy edge {e.mean()-ne.mean():+.1f}pp")

    print("\n" + "="*100)
    print(" VERDICT")
    print("="*100)
    print("  Read the x1.6 row — that pays a realistic-to-conservative event premium.")
    print("  If earnings STILL show a clear monster lift there, it is a genuine,")
    print("  schedulable selector: out of hundreds of candidates, prefer the ones with")
    print("  a known catalyst inside the option's life.")
    print("  If the lift only exists at x1.0, the market fully prices the event and")
    print("  there is no free monster in earnings either.")


if __name__ == "__main__":
    run()
