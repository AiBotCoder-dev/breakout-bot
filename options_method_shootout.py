"""
options_method_shootout.py — Find an options STRUCTURE with a higher hit-rate.

Our current model buys short-dated, slightly-OTM CALLS on a directional signal.
Live result: 22% win rate, expectancy -41%/trade. The structure is the problem —
a cheap OTM call needs a fast, large move just to beat theta + spread. This tests
the SAME entry signal expressed six different ways and asks which one actually
wins more often, net of real friction (bid/ask + commission each leg + IV crush).

Entry signal (long/bullish): price > rising 50SMA > rising 200SMA AND 6-mo
momentum > +10% (our MCPT-validated momentum entry). Hold 15 trading days.
Entry options written 30 DTE, exit ~9 DTE.

Structures (all priced with Black-Scholes + friction):
  1 OTM_CALL_SHORT  current model: +3% OTM call, short-dated (the baseline)
  2 ATM_CALL_30D    at-the-money call, 30 DTE (more time, less gamma-starved)
  3 ITM_CALL_30D    5% in-the-money call (delta ~0.7): stock-replacement, low theta
  4 DEBIT_SPREAD    long ATM / short +7% OTM call: caps payoff, cuts cost
  5 PUT_CREDIT_SPR  sell -5% put / buy -10% put: WINS unless price falls >5%
  6 CASH_SEC_PUT    sell -3% put: WINS unless price falls >3% (return on collateral)

For each: WIN RATE (the "chance to succeed"), mean/median return, expectancy,
worst trade, and how often it's a near-total loss. Credit trades report P&L on
capital-at-risk (spread width / cash collateral) so the numbers are comparable.
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
            "UNH","LLY","WMT","COST","HD","CAT","BA","XOM","CVX","TSLA","SOFI"]
HOLD = 15
R = 0.04
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
HALF_SPREAD_PCT = 0.025; MIN_HALF_SPREAD = 0.02; COMM = 0.0065
IV_CRUSH = 0.10; ENTRY_IV_PREMIUM = 0.90
T0 = 30/365; T1 = 9/365


def bs_call(S,K,T,sig):
    if T<=0 or sig<=0: return max(S-K,0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); return S*N(d1)-K*math.exp(-R*T)*N(d1-sig*math.sqrt(T))
def bs_put(S,K,T,sig):
    if T<=0 or sig<=0: return max(K-S,0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); return K*math.exp(-R*T)*N(-(d1-sig*math.sqrt(T)))-S*N(-d1)
def _hs(m): return max(m*HALF_SPREAD_PCT, MIN_HALF_SPREAD)
def buyf(m): return m+_hs(m)+COMM          # pay to open a long / close a short
def sellf(m): return max(0.0, m-_hs(m)-COMM)  # collect to open a short / close a long


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(6*365.25)+260)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["sma50"]=c.rolling(50).mean(); df["sma200"]=c.rolling(200).mean()
    df["rv20"]=np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["mom6"]=c/c.shift(126)-1
    return df.dropna()


def run():
    print(f"Loading {len(UNIVERSE)} names, 6y...")
    data={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")

    res={k:[] for k in ["1 OTM_CALL_SHORT","2 ATM_CALL_30D","3 ITM_CALL_30D",
                        "4 DEBIT_SPREAD","5 PUT_CREDIT_SPR","6 CASH_SEC_PUT"]}
    for t,df in data.items():
        C=df["Close"].values; s50=df["sma50"].values; s200=df["sma200"].values
        mom=df["mom6"].values; rv=df["rv20"].values; n=len(C)
        for i in range(210,n-HOLD):
            fire=(C[i]>s50[i]>s200[i] and s50[i]>s50[i-10] and s200[i]>s200[i-20] and mom[i]>0.10)
            if not fire: continue
            S0=C[i]; S1=C[i+HOLD]
            iv0=max(0.15,min(1.5,rv[i]*ENTRY_IV_PREMIUM)); iv1=iv0*(1-IV_CRUSH)

            # 1 current: +3% OTM call, short-dated (14->3 DTE)
            K=S0*1.03; e=bs_call(S0,K,14/365,iv0); x=bs_call(S1,K,3/365,iv1)
            res["1 OTM_CALL_SHORT"].append((sellf(x)/buyf(e)-1)*100 if e>0.01 else 0.0)
            # 2 ATM call 30D
            K=S0; e=bs_call(S0,K,T0,iv0); x=bs_call(S1,K,T1,iv1)
            res["2 ATM_CALL_30D"].append((sellf(x)/buyf(e)-1)*100 if e>0.01 else 0.0)
            # 3 ITM call (5% ITM) 30D
            K=S0*0.95; e=bs_call(S0,K,T0,iv0); x=bs_call(S1,K,T1,iv1)
            res["3 ITM_CALL_30D"].append((sellf(x)/buyf(e)-1)*100 if e>0.01 else 0.0)
            # 4 debit spread: long ATM / short +7%
            Kl=S0; Ks=S0*1.07
            de=buyf(bs_call(S0,Kl,T0,iv0))-sellf(bs_call(S0,Ks,T0,iv0))
            dx=sellf(bs_call(S1,Kl,T1,iv1))-buyf(bs_call(S1,Ks,T1,iv1))
            res["4 DEBIT_SPREAD"].append((max(0.0,dx)/de-1)*100 if de>0.01 else 0.0)
            # 5 put credit spread: short -5% / long -10%; P&L on margin (width-credit)
            Ks=S0*0.95; Kl=S0*0.90
            credit=sellf(bs_put(S0,Ks,T0,iv0))-buyf(bs_put(S0,Kl,T0,iv0))
            close=buyf(bs_put(S1,Ks,T1,iv1))-sellf(bs_put(S1,Kl,T1,iv1))
            width=(Ks-Kl); margin=width-credit
            res["5 PUT_CREDIT_SPR"].append((credit-close)/margin*100 if margin>0.01 else 0.0)
            # 6 cash-secured put: short -3% put; P&L on collateral (strike)
            Kp=S0*0.97
            cr=sellf(bs_put(S0,Kp,T0,iv0)); cl=buyf(bs_put(S1,Kp,T1,iv1))
            res["6 CASH_SEC_PUT"].append((cr-cl)/Kp*100)   # return on collateral

    print("="*100)
    print(" OPTIONS STRUCTURE SHOOTOUT — same momentum signal, six expressions (net of friction)")
    print("="*100)
    print(f"  {'structure':20s} {'n':>5} {'WIN%':>6} {'mean':>8} {'median':>8} "
          f"{'expectancy':>11} {'worst':>7} {'%<-50%':>7}")
    for k,v in res.items():
        a=np.array(v)
        if len(a)==0: continue
        win=100*(a>0).mean(); mean=a.mean(); med=np.median(a); worst=a.min()
        blowup=100*(a<=-50).mean()
        print(f"  {k:20s} {len(a):5d} {win:5.0f}% {mean:+7.1f}% {med:+7.1f}% "
              f"{mean:+10.1f}% {worst:+6.0f}% {blowup:6.0f}%")
    print("\n  READ: 'WIN%' is the chance-to-succeed the user asked about; 'expectancy'")
    print("  is mean return per trade (the real test). Credit trades (5,6) report P&L on")
    print("  capital-at-risk. A high win% with positive expectancy AND a survivable worst")
    print("  case is the upgrade. Note credit structures cap the upside — the trade-off.")


if __name__ == "__main__":
    run()
