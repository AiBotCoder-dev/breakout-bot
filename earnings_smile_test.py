"""
earnings_smile_test.py — Does the earnings edge survive a realistic VOLATILITY SMILE?

earnings_edge_test found a 1.26x monster lift for options spanning earnings, even when
each name's event was priced at its OWN historical average move. The mechanism was that
earnings distributions are fat-tailed, so pricing the average move still underprices the
TAILS that OTM options pay on.

THE THREAT: that argument assumes a FLAT implied-vol surface. Real markets don't quote
flat — they quote a SMILE, charging more for OTM strikes precisely because tails are fat.
And around earnings the smile is typically STEEPER (the market knows a jump is coming).
If the smile is steep enough, the market has already taken the edge and 1.26x evaporates.

MODEL: iv(K) = iv_atm * (1 + slope * |ln(K/S)|)
  At 5% OTM, |ln(K/S)| ~ 0.049, so slope 2 => ~+10% IV, slope 4 => ~+20%, slope 6 => ~+29%.
SCENARIOS (the last two are the real stress tests — earnings tails bid up EXTRA):
  1 flat                      slope 0 everywhere            (the original result)
  2 uniform smile             slope 2 everywhere
  3 earnings smile steeper    slope 2 normal / 4 earnings
  4 earnings smile much steeper slope 2 normal / 6 earnings

If the lift holds up in 3-4, the edge is real and tradeable. If it collapses, the market
has already priced it and there is no free monster in earnings.
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
OTM=0.05; DTE=10; MONSTER=100.0; MIN_PRIOR=4
R=0.04; _N=lambda x:0.5*(1+math.erf(x/math.sqrt(2)))
MIN_MID=0.15; RET_CAP=1500.0

# (label, slope_normal, slope_earnings)
SCENARIOS=[("1 flat (original)",0.0,0.0),
           ("2 uniform smile",2.0,2.0),
           ("3 earnings steeper",2.0,4.0),
           ("4 earnings much steeper",2.0,6.0)]


def bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*_N(-d2)-S*_N(-d1)) if put else (S*_N(d1)-K*math.exp(-R*T)*_N(d2))
def buyf(m): return m+max(m*0.03,0.02)+0.0065
def sellf(m): return max(0.0,m-max(m*0.03,0.02)-0.0065)
def smile_iv(iv_atm,S,K,slope):
    return iv_atm*(1.0+slope*abs(math.log(K/S)))


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
        df["ret1"]=c.pct_change()
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
    print(f"Loading {len(UNIVERSE)} names + earnings, 8y...")
    data={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")

    # results[scenario][is_earnings] = list of returns
    res={lab:{True:[],False:[]} for lab,_,_ in SCENARIOS}
    for t,(df,edates) in data.items():
        idx=df.index.normalize(); c=df["Close"].values
        r20=df["rv20"].values; r252=df["rv252"].values; ret1=df["ret1"].values
        n=len(c); pos={d:i for i,d in enumerate(idx)}
        emoves=[]
        for d in edates:
            i=pos.get(d)
            if i is None or i+1>=n: continue
            emoves.append((i,max(abs(ret1[i]),abs(ret1[i+1]))))
        emoves.sort()
        eset={i for i,_ in emoves}

        # ── earnings samples (priced at each name's own history, k=1.0) ──
        for j,(ei,_r) in enumerate(emoves):
            if j<MIN_PRIOR: continue
            hist=float(np.mean([m for _,m in emoves[:j]]))
            if hist<=0: continue
            i=ei-3
            if i<260 or i+DTE>=n: continue
            S0=c[i]; S1=c[i+DTE]
            base_move=max(0.10,min(2.5,(0.5*r20[i]+0.5*r252[i])*0.95))*math.sqrt(DTE/365)
            iv_atm=math.sqrt(base_move**2+hist**2)/math.sqrt(DTE/365)
            for lab,_sn,se in SCENARIOS:
                for put in (False,True):
                    K=S0*(1-OTM) if put else S0*(1+OTM)
                    iv=smile_iv(iv_atm,S0,K,se)
                    mid=bs(S0,K,DTE/365,iv,put)
                    if mid<MIN_MID: continue
                    e=buyf(mid); x=max((K-S1) if put else (S1-K),0.0)
                    res[lab][True].append(min((sellf(x)/e-1)*100,RET_CAP))

        # ── non-earnings baseline ──
        for i in range(260,n-DTE-1,7):
            if any((i+d) in eset or (i+d-1) in eset for d in range(1,DTE+1)): continue
            S0=c[i]; S1=c[i+DTE]
            iv_atm=max(0.10,min(2.5,(0.5*r20[i]+0.5*r252[i])*0.95))
            for lab,sn,_se in SCENARIOS:
                for put in (False,True):
                    K=S0*(1-OTM) if put else S0*(1+OTM)
                    iv=smile_iv(iv_atm,S0,K,sn)
                    mid=bs(S0,K,DTE/365,iv,put)
                    if mid<MIN_MID: continue
                    e=buyf(mid); x=max((K-S1) if put else (S1-K),0.0)
                    res[lab][False].append(min((sellf(x)/e-1)*100,RET_CAP))

    print("="*104)
    print(" DOES THE EARNINGS EDGE SURVIVE THE VOLATILITY SMILE?")
    print("="*104)
    print(f"  {'scenario':26s} {'earnings':>22s}   {'no earnings':>22s}   {'LIFT':>6s}")
    for lab,_sn,_se in SCENARIOS:
        e=np.array(res[lab][True]); nb=np.array(res[lab][False])
        if len(e)==0 or len(nb)==0: continue
        pe=100*(e>=MONSTER).mean(); pn=100*(nb>=MONSTER).mean()
        print(f"  {lab:26s} {pe:5.1f}% mon / {e.mean():+7.1f}% exp   "
              f"{pn:5.1f}% mon / {nb.mean():+7.1f}% exp   {pe/pn if pn else 0:5.2f}x")

    print("\n" + "="*104)
    print(" VERDICT")
    print("="*104)
    print("  Scenarios 3-4 are the real test: the market bids earnings tails up MORE than")
    print("  ordinary tails. If the lift stays clearly above 1.0x there, buying a catalyst")
    print("  is a genuine selector. If it converges to ~1.0x, the smile has already priced")
    print("  the fat tail and there is no free monster in earnings.")


if __name__ == "__main__":
    run()
