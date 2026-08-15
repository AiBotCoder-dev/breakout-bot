"""
dead_option_exit_test.py — Can we cut MATHEMATICALLY DEAD options without
re-creating the stop-loss damage?

THE PROBLEM (user's observation): policy C holds losers to expiry. Those positions
occupy slots under the concentration cap (10), so a corpse that will never recover
can block a fresh signal that might run 100%+. That is a real opportunity cost.

THE TRAP TO AVOID: the obvious fix is a P&L stop, and we PROVED that is harmful —
a -50% stop inverted the tail edge (+20% -> -26%) and did its worst damage in chop,
because price whipsaws through the stop and then recovers. Cutting on PRICE is the
mistake.

THE DIFFERENT IDEA: cut on PROBABILITY OF RECOVERY, not on P&L. An option is dead
when the move it still needs is huge relative to the move the stock can still make
in the time left:
      reachability = (distance to strike) / (expected move over remaining days)
                   = |K/S - 1| / (sigma * sqrt(days_left / 252))
Reachability of 1.0 means the strike is one expected-move away — very much alive.
At 3.0 it needs a 3-sigma move to even break even. Crucially this is NOT a P&L
rule: a position down 70% with time and proximity is KEPT, while one down 40% that
has gone far OTM with two days left is CUT. That is the opposite selection to a stop.

Measured on both axes, because the whole point is slot efficiency:
  * expectancy  — does cutting cost us return? (it must not, or it is just a stop)
  * position-days freed — how much capacity does it release for new signals?
Priced with the CALIBRATED inputs (IV/RV ~1.02, real ~12% OTM spread).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","AMD","MU","TSLA",
            "NFLX","COIN","PLTR","CRWD","SHOP","UBER","QCOM","INTC","MRVL","ARM",
            "SNOW","NET","DDOG","ABNB","DASH","RBLX","HOOD","SOFI","DELL","PANW"]
OTM = 0.03          # the newly tightened band
DTE = 10
R = 0.04
_N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
IV_MULT = 1.02                 # calibrated 2026-08-10
HALF_SPREAD_PCT = 0.06         # half the measured ~12% OTM width
MIN_HALF_SPREAD = 0.02
COMM = 0.0065
MIN_MID = 0.15
RET_CAP = 1200.0
ACTIVATION = 30.0              # policy C: arm at +30%
TRAIL = 0.30                   # then trail 30% off the peak


def bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*_N(-d2)-S*_N(-d1)) if put else (S*_N(d1)-K*math.exp(-R*T)*_N(d2))
def _hs(m): return max(m*HALF_SPREAD_PCT, MIN_HALF_SPREAD)
def buyf(m): return m+_hs(m)+COMM
def sellf(m): return max(0.0, m-_hs(m)-COMM)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(7*365.25)+200)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty or len(raw)<400: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["rv20"]=np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["sma50"]=c.rolling(50).mean(); df["sma200"]=c.rolling(200).mean()
    df["mom6"]=c/c.shift(126)-1
    return df.dropna()


def _simulate(c, i, iv, put, dead_thresh):
    """Walk one trade under policy C, optionally cutting when reachability exceeds
    dead_thresh. Returns (return_pct, days_held)."""
    S0=c[i]; K=S0*(1-OTM) if put else S0*(1+OTM)
    mid=bs(S0,K,DTE/365,iv,put)
    if mid<MIN_MID: return None
    entry=buyf(mid)
    peak=mid
    for d in range(1,DTE+1):
        S=c[i+d]; left=DTE-d; T=left/365
        val = max((K-S) if put else (S-K),0.0) if d==DTE else bs(S,K,T,iv,put)
        peak=max(peak,val)
        pct=(sellf(val)/entry-1)*100
        peakpct=(sellf(peak)/entry-1)*100
        # policy C trailing exit (armed at +30%)
        if peakpct>=ACTIVATION and val<=peak*(1-TRAIL):
            return min(pct,RET_CAP), d
        # DEAD-OPTION test — probability of recovery, not P&L
        if dead_thresh and left>=1 and pct<0:
            exp_move = iv*math.sqrt(max(left,1)/252.0)
            need = abs(K/S - 1.0)
            if exp_move>0 and (need/exp_move) > dead_thresh:
                return min(pct,RET_CAP), d
        if d==DTE:
            return min(pct,RET_CAP), d
    return None


def run():
    print(f"Loading {len(UNIVERSE)} names, 7y...")
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")

    variants={"policy C (hold to expiry)": None, "C + dead @1.5σ": 1.5,
              "C + dead @2.0σ": 2.0, "C + dead @2.5σ": 2.5, "C + dead @3.0σ": 3.0}
    res={k:{"rets":[], "days":[]} for k in variants}

    for t,df in data.items():
        c=df["Close"].values; rv=df["rv20"].values
        s50=df["sma50"].values; s200=df["sma200"].values; mom=df["mom6"].values
        n=len(c)
        for i in range(210,n-DTE-1,3):
            iv=max(0.10,min(2.5,rv[i]*IV_MULT))
            bullish = c[i]>s50[i]>s200[i] and mom[i]>0
            put = not bullish
            for lab,thr in variants.items():
                out=_simulate(c,i,iv,put,thr)
                if out is None: continue
                r_,d_=out
                res[lab]["rets"].append(r_); res[lab]["days"].append(d_)

    base_days=np.mean(res["policy C (hold to expiry)"]["days"])
    base_exp=np.mean(res["policy C (hold to expiry)"]["rets"])
    print("="*104)
    print(" CUTTING DEAD OPTIONS — does it cost return, and how much capacity does it free?")
    print("="*104)
    print(f"  {'variant':28s} {'n':>7} {'win':>6} {'expectancy':>12} {'vs C':>9} "
          f"{'avg days held':>14} {'slot-days saved':>16}")
    for lab in variants:
        a=np.array(res[lab]["rets"]); d=np.array(res[lab]["days"])
        if len(a)==0: continue
        saved=100*(1-d.mean()/base_days)
        print(f"  {lab:28s} {len(a):>7,} {100*(a>0).mean():>5.0f}% {a.mean():>+11.1f}% "
              f"{a.mean()-base_exp:>+8.1f}pp {d.mean():>13.1f} {saved:>15.0f}%")

    print("\n  READ: the dead-exit is only worth it if expectancy is roughly UNCHANGED")
    print("  (within ~1-2pp) while slot-days fall materially. If expectancy drops a lot")
    print("  it is just a stop-loss in disguise and we should not ship it.")
    print("  Slot-days saved matter because the book is capped at 10 concurrent")
    print("  positions — every freed day is capacity for a fresh signal.")


if __name__ == "__main__":
    run()
