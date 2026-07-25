"""
beta_test.py — Is the momentum edge ALPHA or just bull-market BETA?

The reviewer's #9, done correctly. "Short the signals" is the wrong test (in a
bull market shorting anything loses). The right test: do the momentum-SIGNAL-TIMED
entries beat (a) just holding the SAME names unconditionally, and (b) the market
(SPY) over identical windows? If signal-timed ≈ unconditional ≈ SPY, the "edge"
is beta — you'd do as well buying and holding.

For each name we compare forward HOLD-day returns:
  SIGNAL      — only bars where the momentum entry fires (price>rising 50&200 SMA,
                6-mo mom > +10%)
  UNCONDITIONAL — every bar (buy-and-hold-the-name baseline)
  SPY         — the market's forward return over the same dates (pure beta)

Alpha = SIGNAL mean − UNCONDITIONAL mean (does the timing add anything?).
Also reports the信号's excess over SPY (beta-adjusted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = [
    "AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","AVGO","NFLX","CRM","ADBE",
    "INTC","QCOM","MU","AMAT","ORCL","PLTR","COIN","SHOP","UBER","JPM","BAC","GS",
    "V","MA","UNH","LLY","WMT","COST","HD","MCD","NKE","DIS","CAT","BA","XOM","CVX",
    "SOFI","HOOD","SMCI","MRVL","SNOW","NET","CRWD","DDOG","TSLA","AXP","PYPL",
]
HOLD = 10
YEARS = 5


def _rsi(c, p=14):
    d = np.diff(c, prepend=c[0]); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = np.full_like(c, 50.0, float); ag = al = 0.0
    for i in range(1, len(c)):
        if i <= p: ag=(ag*(i-1)+g[i])/i; al=(al*(i-1)+l[i])/i
        else: ag=(ag*(p-1)+g[i])/p; al=(al*(p-1)+l[i])/p
        out[i] = 100.0 if al < 1e-12 else 100.0-100.0/(1.0+ag/al)
    return out


def _load(t):
    end=datetime.now(); start=end-timedelta(days=int(YEARS*365.25)+260)
    raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    df=raw.dropna(subset=["Close"]).copy(); c=df["Close"]
    df["sma50"]=c.rolling(50).mean(); df["sma200"]=c.rolling(200).mean()
    df["mom6"]=c/c.shift(126)-1
    return df.dropna()


def run():
    print(f"Loading {len(UNIVERSE)} names + SPY, {YEARS}y...")
    data={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE+["SPY"]}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    spy=data.pop("SPY",None)
    if spy is None: print("no SPY"); return
    spy_c=spy["Close"]
    print(f"  loaded {len(data)}\n")

    sig_rets=[]; uncond_rets=[]; spy_at_sig=[]
    for t,df in data.items():
        C=df["Close"].values; s50=df["sma50"].values; s200=df["sma200"].values
        mom6=df["mom6"].values; idx=df.index; n=len(C)
        spy_al=spy_c.reindex(idx).values
        for i in range(210,n-HOLD):
            fwd=(C[i+HOLD]/C[i]-1)*100
            uncond_rets.append(fwd)
            if (C[i]>s50[i]>s200[i] and s50[i]>s50[i-10] and s200[i]>s200[i-20] and mom6[i]>0.10):
                sig_rets.append(fwd)
                # SPY forward over the same window
                si=spy_c.index.get_indexer([idx[i]],method="nearest")[0]
                if 0<=si<len(spy_c)-HOLD:
                    spy_at_sig.append((float(spy_c.iloc[si+HOLD])/float(spy_c.iloc[si])-1)*100)

    s=np.array(sig_rets); u=np.array(uncond_rets); sp=np.array(spy_at_sig)
    print("="*72)
    print(f" ALPHA vs BETA — momentum entries, forward {HOLD}-day underlying return")
    print("="*72)
    print(f"  SIGNAL-timed entries  : n={len(s):<6} mean {s.mean():+.2f}%  win {100*(s>0).mean():.1f}%")
    print(f"  UNCONDITIONAL (hold)  : n={len(u):<6} mean {u.mean():+.2f}%  win {100*(u>0).mean():.1f}%")
    print(f"  SPY over signal dates : n={len(sp):<6} mean {sp.mean():+.2f}%")
    print()
    alpha_vs_hold = s.mean()-u.mean()
    alpha_vs_spy  = s.mean()-sp.mean()
    print(f"  ALPHA vs buy-and-hold the same names : {alpha_vs_hold:+.2f}%/10d")
    print(f"  ALPHA vs SPY (beta-adjusted)         : {alpha_vs_spy:+.2f}%/10d")
    print()
    # verdict
    if alpha_vs_hold < 0.3 and alpha_vs_spy < 0.3:
        print("  VERDICT: mostly BETA — the timing adds little over holding / the market.")
        print("           The 'edge' is largely exposure to a rising market.")
    elif alpha_vs_hold >= 0.3:
        print("  VERDICT: real timing ALPHA — signal entries beat holding the same names.")
    else:
        print("  VERDICT: mixed — beats the market but not much better than holding the names.")
    # annualized rough scale
    print(f"\n  (rough scale: {alpha_vs_hold:+.2f}%/10d ≈ {alpha_vs_hold*25:+.0f}%/yr of timing alpha)")


if __name__ == "__main__":
    run()
