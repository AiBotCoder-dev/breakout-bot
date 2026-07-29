"""
capital_model.py — What can ~$1k REALISTICALLY do with our strategy?

Grounds expectations in the actual backtested returns of the strategy we're
running (CAD 2x rotation + broad-gate airbag, Scenario C), then DE-RATES them
honestly (backtests always overstate) and runs a Monte Carlo so you see the full
range of outcomes — median, good case, bad case — plus the drawdown you'd have to
stomach, and how monthly contributions change the picture.

Method:
  1. Reconstruct Scenario-C monthly returns from real ETF history.
  2. Build 3 honest scenarios by scaling the drift (vol/drawdowns kept full):
       OPTIMISTIC  = backtest as-is (the ceiling; unlikely to repeat live)
       BASE        = half the backtested edge (survivorship + overfit + regime haircut)
       PESSIMISTIC = edge mostly gone (flat drift, full volatility)
  3. Block-bootstrap (3-month blocks, preserves drawdown clustering) 20k paths
     over 12 and 36 months, with optional monthly contributions.
  4. Report P10/P50/P90 balances, worst drawdown, and prob(end below start).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

YEARS = 8; TOP_K = 2; MOM = 63
CAD = ["HQU.TO","HSU.TO","HXU.TO","HEU.TO","HFU.TO","HGU.TO"]
START = 1000.0
N_PATHS = 20000
np.random.seed(7)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(YEARS*365.25)+260)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty or len(raw)<300: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["sma50"]=c.rolling(50).mean(); df["sma200"]=c.rolling(200).mean(); df["mom"]=c/c.shift(MOM)-1
    return df.dropna()


def _scenario_c_daily():
    data={}
    for t in CAD+["SPY","XIU.TO"]:
        d=_load(t)
        if d is not None: data[t]=d
    cad={t:data[t] for t in CAD if t in data}
    base=max(cad.values(),key=len); cal=base.index
    def a200(df): return (df["Close"]>df["sma200"]).reindex(cal,method="ffill").fillna(False)
    broad=(a200(data["SPY"])&a200(data["XIU.TO"])).values
    dr=[]; idx=[]; tgt=[]
    for k in range(210,len(cal)):
        d=cal[k]; allow=bool(broad[k])
        if k%5==0:
            rk=[]
            for t,df in cad.items():
                if d in df.index:
                    i=df.index.get_loc(d)
                    if i>200:
                        c=df["Close"].values; s50=df["sma50"].values; s200=df["sma200"].values; m=df["mom"].values
                        if m[i]==m[i] and c[i]>s50[i] and c[i]>s200[i]: rk.append((m[i],t))
            rk.sort(reverse=True); tgt=[t for _,t in rk[:TOP_K]]
        held=tgt if allow else []
        if held:
            rs=[cad[t]["Close"].values[cad[t].index.get_loc(d)]/cad[t]["Close"].values[cad[t].index.get_loc(d)-1]-1 for t in held if d in cad[t].index and cad[t].index.get_loc(d)>0]
            r=float(np.mean(rs)) if rs else 0.0
        else: r=0.0
        dr.append(r); idx.append(d)
    return pd.Series(dr,index=pd.DatetimeIndex(idx))


def _monthly(daily):
    return (daily+1).resample("ME").prod()-1


def _bootstrap(monthly_rets, months, n_paths, contrib=0.0, block=3):
    m=np.asarray(monthly_rets,float); L=len(m)
    out_bal=np.empty(n_paths); out_dd=np.empty(n_paths)
    for p in range(n_paths):
        seq=[]
        while len(seq)<months:
            s=np.random.randint(0,L-block); seq.extend(m[s:s+block])
        seq=seq[:months]
        bal=START; peak=START; mdd=0.0
        for r in seq:
            bal=bal*(1+r)+contrib
            peak=max(peak,bal); mdd=min(mdd,(bal-peak)/peak)
        out_bal[p]=bal; out_dd[p]=mdd
    return out_bal, out_dd


def run():
    print("Reconstructing Scenario-C monthly returns from real ETF history...")
    daily=_scenario_c_daily(); mo=_monthly(daily).values
    mu=mo.mean(); sd=mo.std()
    ann=(np.prod(1+mo)**(12/len(mo))-1)*100
    print(f"  backtest: {len(mo)} months, mean {mu*100:+.2f}%/mo, vol {sd*100:.1f}%/mo, "
          f"~{ann:+.0f}%/yr (SURVIVORSHIP-FLATTERED CEILING)\n")

    scen={
        "OPTIMISTIC (backtest holds)": mo,
        "BASE (half the edge)":        mo-(mu*0.5),
        "PESSIMISTIC (edge decays)":   mo-mu,      # flat drift, full vol
    }
    for horizon in (12, 36):
        print("="*84)
        print(f" {horizon}-MONTH OUTCOMES on ${START:.0f} start  (20k Monte Carlo paths)")
        print("="*84)
        for contrib in (0.0, 100.0):
            tag = "no contributions" if contrib==0 else f"+${contrib:.0f}/mo added"
            invested = START + contrib*horizon
            print(f"\n  -- {tag}  (total you put in: ${invested:.0f}) --")
            print(f"     {'scenario':30s} {'bad(P10)':>10} {'median':>10} {'good(P90)':>10} {'worst DD':>9} {'P(loss)':>8}")
            for name, rets in scen.items():
                bal, dd = _bootstrap(rets, horizon, N_PATHS, contrib)
                p10,p50,p90=np.percentile(bal,[10,50,90])
                ddmed=np.percentile(dd,50)*100
                ploss=100*np.mean(bal<invested)
                print(f"     {name:30s} ${p10:8.0f} ${p50:8.0f} ${p90:9.0f} "
                      f"{ddmed:7.0f}% {ploss:6.0f}%")
    print("\n  READ: 'bad/median/good' are the 10th/50th/90th percentile ending balances.")
    print("  'worst DD' is the typical deepest drop along the way (you WILL live through")
    print("  these). 'P(loss)' = chance you end with less than you put in. Notice how much")
    print("  more the CONTRIBUTIONS move the needle than the strategy does at this size —")
    print("  that's the whole point: at $1k, savings rate > return rate.")
    print(f"\n  For context, $100/day = ~$25,000/yr. Even the OPTIMISTIC 36-mo median")
    print(f"  doesn't approach that — because $100/day is a CAPITAL goal, not a strategy goal.")


if __name__ == "__main__":
    run()
