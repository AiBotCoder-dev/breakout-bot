"""
reviewer_tests.py — Answers the reviewer's sharpened challenges with real numbers.

PART 1 — Alpha vs Beta (their #1):
  Beta-adjusted alpha of momentum-SIGNAL entries vs UNCONDITIONAL (hold same names),
  with an Information Ratio, PLUS a regime decomposition (Bull/Correction/Bear/Panic).
  alpha_i = r_stock,i − beta_i · r_spy,i   (beta = rolling 90d vs SPY)
  IR = mean(alpha)/std(alpha) × sqrt(252/HOLD)   (annualized)

PART 2 — Instrument comparison (their #3 "real test"):
  For every momentum signal, price three expressions of the SAME directional bet
  and compare Sortino / median / win / tail (their ghost decision metrics):
    A  near-money CALL (bought)   — net of spread+commission (the current structure)
    B  OTM put SPREAD (sold)      — −5%/−10% put credit spread, net
    C  synthetic 3x DAILY-REBAL shares — decay is INHERENT (product of 1+3·dailyret),
       no assumed term needed; this is the honest leveraged-shares expression
  Sortino uses downside deviation only (option returns are left-skewed).
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
            "HOOD","SMCI","MRVL","SNOW","NET","CRWD","DDOG"]
HOLD = 10
R = 0.04
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
HALF_SPREAD_PCT = 0.025; MIN_HALF_SPREAD = 0.02; COMM = 0.0065; IV_CRUSH = 0.10
ENTRY_IV_PREMIUM = 0.90


def bs_call(S,K,T,sig):
    if T<=0 or sig<=0: return max(S-K,0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); return S*N(d1)-K*math.exp(-R*T)*N(d1-sig*math.sqrt(T))
def bs_put(S,K,T,sig):
    if T<=0 or sig<=0: return max(K-S,0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); return K*math.exp(-R*T)*N(-(d1-sig*math.sqrt(T)))-S*N(-d1)
def _hs(m): return max(m*HALF_SPREAD_PCT, MIN_HALF_SPREAD)
def buyf(m): return m+_hs(m)+COMM
def sellf(m): return max(0.0, m-_hs(m)-COMM)


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(6*365.25)+260)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["sma50"]=c.rolling(50).mean(); df["sma200"]=c.rolling(200).mean()
    df["rv20"]=np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["mom6"]=c/c.shift(126)-1; df["ret"]=c.pct_change()
    return df.dropna()


def _sortino(v):
    a=np.asarray(v,float)
    if len(a)==0: return 0.0
    dn=a[a<0]; dd=np.sqrt(np.mean(dn**2)) if len(dn) else 1e-9
    return a.mean()/dd if dd>0 else 0.0


def run():
    print(f"Loading {len(UNIVERSE)} names + SPY + ^VIX, 6y...")
    data={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE+["SPY","^VIX"]}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    spy=data.pop("SPY"); vix=data.pop("^VIX",None)
    spy_c=spy["Close"]; spy_ret=spy["Close"].pct_change()
    spy200=spy["Close"].rolling(200).mean()
    print(f"  loaded {len(data)}\n")

    sig_alpha=[]; unc_alpha=[]; regime_alpha={"Bull":[],"Correction":[],"Bear":[],"Panic":[]}
    A=[]; B=[]; C=[]
    for t,df in data.items():
        C_=df["Close"].values; s50=df["sma50"].values; s200=df["sma200"].values
        mom6=df["mom6"].values; rv=df["rv20"].values; ret=df["ret"].values; idx=df.index; n=len(C_)
        spy_al=spy_c.reindex(idx).values; spyret_al=spy_ret.reindex(idx).values
        spy200_al=spy200.reindex(idx).values
        vix_al=vix["Close"].reindex(idx).values if vix is not None else np.full(n,20.0)
        # rolling 90d beta of stock vs spy
        sr=pd.Series(ret,index=idx); mr=pd.Series(spyret_al,index=idx)
        beta=(sr.rolling(90).cov(mr)/mr.rolling(90).var()).values
        for i in range(210,n-HOLD):
            if C_[i]<=0: continue
            rs=C_[i+HOLD]/C_[i]-1
            si=spy_c.index.get_indexer([idx[i]],method="nearest")[0]
            if not(0<=si<len(spy_c)-HOLD): continue
            rm=float(spy_c.iloc[si+HOLD])/float(spy_c.iloc[si])-1
            b=beta[i] if beta[i]==beta[i] else 1.0
            a_unc=(rs-b*rm)*100
            unc_alpha.append(a_unc)
            fire=(C_[i]>s50[i]>s200[i] and s50[i]>s50[i-10] and s200[i]>s200[i-20] and mom6[i]>0.10)
            if not fire: continue
            sig_alpha.append(a_unc)
            # regime at signal
            sp=spy_al[i]; s2=spy200_al[i]; vx=vix_al[i]
            if vx>35: reg="Panic"
            elif sp<s2: reg="Bear"
            elif vx>=20: reg="Correction"
            else: reg="Bull"
            regime_alpha[reg].append(a_unc)
            # instruments
            S0=C_[i]; S1=C_[i+HOLD]; iv0=max(0.15,min(1.5,rv[i]*ENTRY_IV_PREMIUM)); iv1=iv0*(1-IV_CRUSH)
            # A near-money call
            K=S0*1.03; e=bs_call(S0,K,21/365,iv0); x=bs_call(S1,K,11/365,iv1)
            A.append((sellf(x)/buyf(e)-1)*100 if e>0.01 else 0.0)
            # B sold put spread (-5%/-10%): credit = short put - long put; profit if stays up
            Ks=S0*0.95; Kl=S0*0.90
            cr_e=bs_put(S0,Ks,21/365,iv0)-bs_put(S0,Kl,21/365,iv0)
            cr_x=bs_put(S1,Ks,11/365,iv1)-bs_put(S1,Kl,11/365,iv1)
            width=(Ks-Kl)
            if cr_e>0.01:
                # seller P&L on margin (width-credit); net friction ~4 legs
                credit_net=sellf(bs_put(S0,Ks,21/365,iv0))-buyf(bs_put(S0,Kl,21/365,iv0))
                close_net=buyf(bs_put(S1,Ks,11/365,iv1))-sellf(bs_put(S1,Kl,11/365,iv1))
                pnl=credit_net-close_net; margin=width-credit_net
                B.append(pnl/margin*100 if margin>0.01 else 0.0)
            # C synthetic 3x daily-rebalanced shares (decay inherent)
            dr=ret[i+1:i+HOLD+1]
            lev=np.prod(1+3*dr)-1
            C.append(lev*100)

    print("="*74); print(" PART 1 — ALPHA vs BETA (beta-adjusted, Information Ratio)"); print("="*74)
    s=np.array(sig_alpha); u=np.array(unc_alpha)
    ann=math.sqrt(252/HOLD)
    ir_s=(s.mean()/s.std()*ann) if s.std()>0 else 0
    ir_u=(u.mean()/u.std()*ann) if u.std()>0 else 0
    print(f"  SIGNAL entries  beta-adj alpha: mean {s.mean():+.2f}%/10d  IR(annual) {ir_s:+.2f}  (n={len(s)})")
    print(f"  UNCOND (hold)   beta-adj alpha: mean {u.mean():+.2f}%/10d  IR(annual) {ir_u:+.2f}  (n={len(u)})")
    print(f"  SELECTION alpha (signal − uncond): {s.mean()-u.mean():+.2f}%/10d")
    print(f"  --> reviewer's decision rule: pivot to factor ETF if IR < 0.4  (signal IR = {ir_s:+.2f})")
    print("\n  REGIME DECOMPOSITION (signal beta-adj alpha):")
    for reg in ["Bull","Correction","Bear","Panic"]:
        v=np.array(regime_alpha[reg])
        if len(v): print(f"    {reg:11s} n={len(v):<5} alpha {v.mean():+.2f}%/10d  win {100*(v>0).mean():.0f}%")
        else: print(f"    {reg:11s} n=0")

    print("\n"+"="*74); print(" PART 2 — INSTRUMENT COMPARISON (same signal, 3 expressions)"); print("="*74)
    for name,v in [("A near-money CALL (buy)",A),("B put SPREAD (sell)",B),("C 3x shares (daily-rebal)",C)]:
        a=np.array(v)
        if len(a)==0: continue
        worst=a.min(); tail=100*(a<=-70).mean()
        print(f"  {name:26s} n={len(a):<5} win {100*(a>0).mean():4.0f}%  mean {a.mean():+6.1f}%  "
              f"med {np.median(a):+6.1f}%  Sortino {_sortino(a):+.2f}  worst {worst:+.0f}%  tail<-70%: {tail:.0f}%")
    print("\n  (Sortino = mean / downside-deviation; the reviewer's ghost decision metric.)")


if __name__ == "__main__":
    run()
