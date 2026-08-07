"""
monster_movers.py — Which stocks ran hardest in the last month, and which option
contract would have benefited the MOST from each move?

Scans a broad liquid/volatile universe, ranks by absolute ~1-month move, and for
the top movers finds the best REALISTIC option (call for up-moves, put for down):
the strike that maximises return among contracts you could actually have bought
(entry premium >= $0.20 — no fantasy penny strikes). Priced Black-Scholes: bought
~30 DTE at the start of the window, held to expiry (intrinsic at the end).

This is a HINDSIGHT exercise — the honest lesson is you can't know the mover in
advance — but it shows the leverage a single well-placed contract can carry.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","AMD","MU","SMCI",
            "MRVL","ARM","QCOM","INTC","ON","TSLA","PLTR","COIN","MSTR","HOOD",
            "SOFI","AFRM","SHOP","NET","CRWD","DDOG","SNOW","RBLX","U","ABNB",
            "UBER","DASH","GME","AMC","MARA","RIOT","CLSK","RDDT","CVNA","BABA",
            "PDD","NIO","DELL","AI","PATH","SOUN","LLY","UNH","BA","XOM","CELH",
            "ANF","DKNG","PANW","NFLX","MRNA","ENPH","FSLR","TTD","ROKU"]
WINDOW = 22          # ~1 trading month
R = 0.04
N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
MIN_ENTRY = 0.20     # only contracts you could actually buy


def bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    if put: return K*math.exp(-R*T)*N(-d2)-S*N(-d1)
    return S*N(d1)-K*math.exp(-R*T)*N(d2)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=120)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty or len(raw)<40: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["rv"]=np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    return df.dropna()


def _best_contract(t, df):
    S0=float(df["Close"].iloc[-WINDOW]); S1=float(df["Close"].iloc[-1])
    move=S1/S0-1
    rv=float(df["rv"].iloc[-WINDOW]); iv=max(0.20,min(2.5,rv*0.90))
    put = move<0
    T0=30/365
    best=None
    # sweep strikes from ATM out to 60% OTM in the move's direction
    for otm in np.arange(0.0,0.60,0.01):
        K = S0*(1-otm) if put else S0*(1+otm)
        entry=bs(S0,K,T0,iv,put)
        if entry<MIN_ENTRY: continue
        exitv=max((K-S1) if put else (S1-K),0.0)      # intrinsic at expiry
        ret=(exitv/entry-1)*100 if entry>0 else 0
        if best is None or ret>best["ret"]:
            best={"K":K,"otm":otm,"entry":entry,"exitv":exitv,"ret":ret,
                  "put":put,"S0":S0,"S1":S1,"move":move}
    return best


def _occ(t,put,K):
    exp=(datetime.now()).strftime("%y%m%d")
    return f"{t}{exp}{'P' if put else 'C'}{int(round(K*1000)):08d}"


def run():
    print(f"Scanning {len(UNIVERSE)} names for the biggest ~1-month moves...\n")
    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    rows=[]
    for t,df in data.items():
        b=_best_contract(t,df)
        if b: rows.append((t,b))
    rows.sort(key=lambda x: abs(x[1]["move"]), reverse=True)

    print("="*100)
    print(" TOP MOVERS (last ~1 month) + the single option contract that paid the most")
    print("="*100)
    print(f"  {'stock':6s} {'move':>7} {'dir':>4}  {'best strike':>12}  {'entry':>7} {'→ worth':>8}  {'return':>9}   $100→")
    for t,b in rows[:14]:
        d="CALL" if not b["put"] else "PUT"
        arrow=f"${b['S0']:.0f}→${b['S1']:.0f}"
        val100=100*(1+b["ret"]/100)
        print(f"  {t:6s} {b['move']*100:+6.0f}% {d:>4}  ${b['K']:>7.0f}({b['otm']*100:.0f}%O) "
              f"${b['entry']:6.2f} ${b['exitv']:7.2f}  {b['ret']:+8.0f}%  ${val100:,.0f}   [{arrow}]")
    print()
    # spotlight the single best payoff
    top=max(rows[:14], key=lambda x: x[1]["ret"])
    t,b=top
    print("="*100)
    print(f" BIGGEST OPTION PAYOFF: {t} {'PUT' if b['put'] else 'CALL'}")
    print("="*100)
    print(f"  contract  : {_occ(t,b['put'],b['K'])}  (${b['K']:.0f} strike, ~30 DTE at entry)")
    print(f"  stock move : ${b['S0']:.2f} -> ${b['S1']:.2f}  ({b['move']*100:+.0f}%)")
    print(f"  the option : bought ~${b['entry']:.2f}  ->  worth ~${b['exitv']:.2f} at expiry")
    print(f"  return     : {b['ret']:+.0f}%   ($100 -> ${100*(1+b['ret']/100):,.0f})")
    print("\n  NOTE: idealized (BS pricing, realized-vol IV, held to expiry). Real fills")
    print("  cross a spread and real IV on these names was likely higher, so treat the")
    print("  % as a ceiling. The point isn't the number — it's that ONE cheap directional")
    print("  contract carries the whole move. The catch: knowing WHICH name, in advance.")


if __name__ == "__main__":
    run()
