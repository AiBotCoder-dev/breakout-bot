"""
chop_filter_test.py — Does sitting out CHOPPY markets improve option results?

THE OBSERVATION: on 2026-08-07 the bot closed 8 trades and was stopped on all 8 —
calls AND puts. That is not a signal failure, it is a rangebound market punishing
every directional bet. Nothing in the bot currently detects that state.

THE MEASURE — Kaufman's Efficiency Ratio on SPY:
      ER = |close[t] - close[t-n]| / sum(|daily change|) over the same n days
  ER near 1.0 = clean directional trend (price went somewhere)
  ER near 0.0 = chop (lots of movement, no progress)
Unlike ADX it needs no smoothing constants and is scale-free, so there is nothing
to curve-fit.

THE TEST: simulate directional option buys across a liquid universe, bucket every
entry by the MARKET's efficiency ratio that day, and compare outcomes. If expectancy
is materially worse in low-ER (choppy) regimes, a gate that sits them out is free
money — it removes trades rather than needing to predict anything.

Both directions are measured, because the point is that chop hurts calls AND puts.
Priced with the CALIBRATED inputs measured on 2026-08-10 (IV/RV ~1.02, real OTM
spread ~12%), not the old optimistic assumptions.
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
ER_WINDOW = 10          # efficiency ratio lookback
OTM = 0.03              # matches the newly tightened strike band
DTE = 10
R = 0.04
_N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
# calibrated 2026-08-10: IV ~= realized vol, and OTM round-trip spread ~12%
IV_MULT = 1.02
HALF_SPREAD_PCT = 0.06   # half of the measured ~12% OTM width
MIN_HALF_SPREAD = 0.02
COMM = 0.0065
MIN_MID = 0.15
RET_CAP = 1200.0


def bs(S,K,T,sig,put=False):
    if T<=0 or sig<=0: return max((K-S) if put else (S-K),0.0)
    d1=(math.log(S/K)+(R+sig*sig/2)*T)/(sig*math.sqrt(T)); d2=d1-sig*math.sqrt(T)
    return (K*math.exp(-R*T)*_N(-d2)-S*_N(-d1)) if put else (S*_N(d1)-K*math.exp(-R*T)*_N(d2))
def _hs(m): return max(m*HALF_SPREAD_PCT, MIN_HALF_SPREAD)
def buyf(m): return m+_hs(m)+COMM
def sellf(m): return max(0.0, m-_hs(m)-COMM)


def efficiency_ratio(close, n=ER_WINDOW):
    """Kaufman ER: net directional progress / total path travelled."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return (net / path).replace([np.inf, -np.inf], np.nan)


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


def run():
    print(f"Loading SPY + {len(UNIVERSE)} names, 7y...")
    spy=_load("SPY")
    if spy is None:
        print("  FATAL: SPY failed"); return
    spy_er = efficiency_ratio(spy["Close"])

    data={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")

    rows=[]
    for t,df in data.items():
        c=df["Close"].values; rv=df["rv20"].values
        s50=df["sma50"].values; s200=df["sma200"].values; mom=df["mom6"].values
        er=spy_er.reindex(df.index, method="ffill").values
        n=len(c)
        for i in range(210,n-DTE-1,3):
            e=er[i]
            if e!=e: continue
            iv=max(0.10,min(2.5,rv[i]*IV_MULT))
            S0=c[i]; S1=c[i+DTE]
            # trade the direction the trend implies (what the bot does)
            bullish = c[i]>s50[i]>s200[i] and mom[i]>0
            for put in ([False] if bullish else [True]):
                K=S0*(1+OTM) if not put else S0*(1-OTM)
                mid=bs(S0,K,DTE/365,iv,put)
                if mid<MIN_MID: continue
                entry=buyf(mid); x=max((K-S1) if put else (S1-K),0.0)
                held=min((sellf(x)/entry-1)*100,RET_CAP)
                # THE INTERACTION TEST: same entries, but walked daily with the
                # retired -50% hard stop. Chop should punish a STOPPED strategy
                # (price whipsaws through the stop then recovers) while leaving a
                # hold-to-expiry strategy alone — which would explain both the
                # 8-of-8 stopped day AND this backtest's chop-is-fine result.
                stopped=None
                for dd in range(1,DTE+1):
                    Sd=c[i+dd]; Td=(DTE-dd)/365
                    val=bs(Sd,K,Td,iv,put) if dd<DTE else max((K-Sd) if put else (Sd-K),0.0)
                    if (sellf(val)/entry-1)*100 <= -50 and dd<DTE:
                        stopped=-50.0; break
                if stopped is None: stopped=held
                rows.append({"er":e,"ret":held,"ret_stop":stopped,
                             "dir":"put" if put else "call"})
    d=pd.DataFrame(rows)
    print("="*94)
    print(f" OPTION OUTCOMES BY MARKET EFFICIENCY RATIO (SPY, {ER_WINDOW}d) — chop vs trend")
    print("="*94)
    base=d["ret"].mean()
    print(f"  {'regime':26s} {'n':>7} {'win':>6} {'expectancy':>12} {'median':>9}")
    buckets=[(0,.20,"0.00-0.20  deep chop"),(.20,.35,"0.20-0.35  choppy"),
             (.35,.50,"0.35-0.50  mixed"),(.50,.70,"0.50-0.70  trending"),
             (.70,1.01,"0.70+      strong trend")]
    for lo,hi,lab in buckets:
        s=d[(d["er"]>=lo)&(d["er"]<hi)]
        if len(s)<200: continue
        print(f"  {lab:26s} {len(s):>7,} {100*(s['ret']>0).mean():>5.0f}% "
              f"{s['ret'].mean():>+11.1f}% {s['ret'].median():>+8.0f}%")
    print(f"\n  overall expectancy {base:+.1f}%")

    print("\n" + "="*94)
    print(" WHAT A GATE WOULD DO (skip entries below the ER threshold)")
    print("="*94)
    print(f"  {'gate':22s} {'trades kept':>12} {'% skipped':>10} {'expectancy':>12} {'vs no gate':>11}")
    for thr in (0.0,0.20,0.30,0.35,0.40,0.50):
        s=d[d["er"]>=thr]
        if len(s)<100: continue
        lab="no gate" if thr==0 else f"ER >= {thr:.2f}"
        print(f"  {lab:22s} {len(s):>12,} {100*(1-len(s)/len(d)):>9.0f}% "
              f"{s['ret'].mean():>+11.1f}% {s['ret'].mean()-base:>+10.1f}pp")

    print("\n" + "="*94)
    print(" THE INTERACTION — does chop only hurt when a HARD STOP is in play?")
    print("="*94)
    print(f"  {'regime':26s} {'hold-to-expiry':>16} {'with -50% stop':>16} {'stop costs':>12}")
    for lo,hi,lab in buckets:
        s=d[(d["er"]>=lo)&(d["er"]<hi)]
        if len(s)<200: continue
        h=s["ret"].mean(); st=s["ret_stop"].mean()
        print(f"  {lab:26s} {h:>+15.1f}% {st:>+15.1f}% {st-h:>+11.1f}pp")
    print("\n  READ: if the stop's damage is WORST in the chop rows, then chop was never")
    print("  the problem by itself — the -50% stop was, and removing it (policy C, live")
    print("  2026-08-11) already fixed what a chop gate would have been patching.")


if __name__ == "__main__":
    run()
