"""
convex_put_backtest.py — Is a TAIL-HUNTING put strategy actually +EV?

The TSLA +1788% put was a convex bet on a crash: negative catalyst + price already
breaking down -> cheap OTM put -> a rare huge win pays for many -100% losses. The
user wants to build a bot that systematically hunts these. Win rate is irrelevant
here; the ONLY question is expectancy: over hundreds of breakdown signals, does the
fat left tail (crashes) pay for the frequent worthless expirations, net of friction?

We can't backtest the NEWS trigger historically, but we CAN backtest its price
component — a confirmed breakdown — which is the tradeable core:
  SIGNAL: close < EMA20  AND  5-day return < -4%   (a confirmed break, not a dip)
  Non-overlapping per name. On signal, buy an OTM put, DTE ~7.

Two things this settles:
  1. STRUCTURE — which OTM level (5/8/10%) maximises expectancy.
  2. EXIT — the crucial one. A -50% stop is fine for directional calls but LETHAL
     for tail-hunting: it cuts the position before the crash matures. We compare
     HOLD-TO-EXPIRY vs the -50% stop to prove whether the stop kills the tail.

Also reports TAIL CONCENTRATION (how much of all profit comes from the top 5% of
trades) and the strategy's equity-curve drawdown — because a real tail-hunter has
to survive a long drought of losses before the payoff lands.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = ["AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","AVGO","NFLX","CRM",
            "INTC","QCOM","MU","ORCL","PLTR","COIN","SHOP","UBER","JPM","V","MA",
            "UNH","LLY","WMT","COST","HD","CAT","BA","XOM","CVX","TSLA","SOFI",
            "HOOD","SMCI","MRVL","SNOW","NET","CRWD","DDOG","DIS","PYPL","SQ"]
DTE0 = 7                    # days to expiry at entry
R = 0.04
N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
HALF_SPREAD_PCT=0.03; MIN_HALF_SPREAD=0.02; COMM=0.0065; ENTRY_IV_PREMIUM=0.95


def bs_put(S,K,T,sig):
    if T<=0 or sig<=0: return max(K-S,0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); return K*math.exp(-R*T)*N(-(d1-sig*math.sqrt(T)))-S*N(-d1)
def _hs(m): return max(m*HALF_SPREAD_PCT,MIN_HALF_SPREAD)
def buyf(m): return m+_hs(m)+COMM
def sellf(m): return max(0.0,m-_hs(m)-COMM)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(6*365.25)+120)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["ema20"]=c.ewm(span=20).mean()
    df["rv20"]=np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["ret5"]=c/c.shift(5)-1
    return df.dropna()


MIN_ENTRY = 0.15     # can't realistically buy a put cheaper than ~$0.15
RET_CAP = 1200.0     # cap single-trade return: real exit liquidity on a deep-ITM
                     # penny put won't let you bank +150,000% on size


def _simulate(otm, stop_pct=None, realistic=False):
    """Return list of per-trade net % results for the given OTM level + exit rule.
    stop_pct=None -> hold to expiry (intrinsic). stop_pct=-50 -> exit if the put's
    marked value falls to -50% before expiry (tail-cutting stop).
    realistic=True -> skip un-buyable cheap puts (< MIN_ENTRY) and cap the per-trade
    gain at RET_CAP, so BS pricing of penny options can't fabricate the edge."""
    data={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    res=[]
    for t,df in data.items():
        C=df["Close"].values; ema=df["ema20"].values; rv=df["rv20"].values
        ret5=df["ret5"].values; n=len(C); open_until=-1
        for i in range(30,n-DTE0-1):
            if i<=open_until: continue
            if not (C[i]<ema[i] and ret5[i]<-0.04):    # confirmed breakdown
                continue
            open_until=i+DTE0
            S0=C[i]; K=S0*(1-otm)
            iv0=max(0.15,min(2.0,rv[i]*ENTRY_IV_PREMIUM))
            _mid=bs_put(S0,K,DTE0/365,iv0)
            entry=buyf(_mid)
            if entry<=0.01: continue
            if realistic and _mid < MIN_ENTRY: continue    # un-buyable penny put
            # walk each day; apply stop or ride to expiry
            exitval=None
            for d in range(1,DTE0+1):
                S=C[i+d]; T=(DTE0-d)/365
                val = max(0.0,K-S) if d==DTE0 else bs_put(S,K,T,iv0)
                markpct=(sellf(val)/entry-1)*100
                if stop_pct is not None and markpct<=stop_pct and d<DTE0:
                    exitval=sellf(val); break
                if d==DTE0:
                    exitval=max(0.0,K-S)      # settle at intrinsic
            r=(sellf(exitval)/entry-1)*100
            if realistic: r=min(r, RET_CAP)
            res.append(r)
    return res


def _report(name, r):
    a=np.array(r)
    if len(a)==0: print(f"  {name}: n=0"); return
    win=100*(a>0).mean(); exp=a.mean(); med=np.median(a)
    # tail concentration: profit from top 5% of trades vs total gross profit
    gains=a[a>0]; top5=np.sort(gains)[::-1][:max(1,int(len(a)*0.05))]
    tail_share=100*top5.sum()/gains.sum() if gains.sum()>0 else 0
    # equity curve on equal $1 bets (compounding-agnostic): cumulative mean path DD
    eq=np.cumsum(a/100.0); peak=np.maximum.accumulate(eq); dd=(eq-peak).min()
    print(f"  {name:26s} n={len(a):<5} win {win:3.0f}%  EXPECTANCY {exp:+6.1f}%  "
          f"med {med:+5.0f}%  top5%→{tail_share:3.0f}% of gains  worstDD {dd*100:+.0f}u")


def run():
    print(f"Backtesting confirmed-breakdown puts, {len(UNIVERSE)} names, 6y, DTE {DTE0}...\n")
    print("="*104)
    print(" STRUCTURE SWEEP — hold to expiry (let the tail run)")
    print("="*104)
    for otm in (0.05,0.08,0.10):
        _report(f"{int(otm*100)}% OTM · hold-to-exp", _simulate(otm, None))
    print("\n"+"="*104)
    print(" EXIT COMPARISON @ 8% OTM — does the -50% stop kill the tail?")
    print("="*104)
    _report("8% OTM · hold-to-exp", _simulate(0.08, None))
    _report("8% OTM · -50% stop",   _simulate(0.08, -50))
    print("\n"+"="*104)
    print(" REALISTIC — skip un-buyable penny puts (<$0.15) + cap gain at +1200% (real exit liquidity)")
    print("="*104)
    for otm in (0.05,0.08,0.10):
        _report(f"{int(otm*100)}% OTM · realistic", _simulate(otm, None, realistic=True))
    _report("8% OTM · realistic +stop", _simulate(0.08, -50, realistic=True))

    print("\n  READ: EXPECTANCY is the whole game (win% is expected to be low — that's fine).")
    print("  Positive expectancy = the crashes pay for the worthless expirations -> a real")
    print("  tail edge worth building. 'top5%→X% of gains' shows how tail-dependent it is")
    print("  (very high = a few crashes carry everything; fragile, needs many small bets).")
    print("  If the -50% stop CUTS expectancy vs hold-to-expiry, the stop is killing the")
    print("  tail and a tail-hunter must NOT use it.")


if __name__ == "__main__":
    run()
