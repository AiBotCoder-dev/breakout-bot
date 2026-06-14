"""
exhaustion_reversal_v2.py — refined, ACTIVELY-MANAGED put reversal at exhausted highs.

V1 verdict: hold-to-expiry puts at exhausted highs are <50% win (momentum wins),
but the MFE (drop within 20d) is real (median 3.7%, p90 9.3%). So V2 tests a
TRADEABLE managed version and tries to lift conviction:

  STRONGER SETUP (top / puts):
    • within 3% of the 252-day high
    • EXTENDED: close >= 12% above the 50-SMA (stretched, room to revert)
    • EXHAUSTION: bearish RSI divergence OR drying volume into the high
    • DECISIVE TRIGGER: close breaks BELOW the 10-day low (not a 1-day wobble)
      on a down day

  MANAGED EXIT (realistic — you don't hold a reversal forever):
    entry = close at trigger
    stop  = just above the recent 5-bar swing high (tight; thesis = breakdown)
    target= entry * (1 - RR * risk)                [RR = 2.5]
    walk forward up to MAXHOLD; first touch wins (ties -> stop); else exit at close.
    non-overlapping. Reports realized win rate, R:R, expectancy.

  Split: ALL names vs HIGH-BETA subset (the ones that actually crash).
  Then MCPT on the better bucket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

HIGH_BETA = ["COIN","MSTR","CVNA","SMCI","ARM","AFRM","UPST","RIVN","LCID","PLTR",
             "TSLA","NVDA","SNAP","ROKU","DKNG","RBLX","ENPH","NET","CRWD","SHOP",
             "MARA","RIOT","SOFI","HOOD","DDOG","NFLX","AMD","MU"]
MEGA = ["SPY","QQQ","AAPL","MSFT","GOOGL","AMZN","META","JPM","V","MA","UNH","LLY",
        "JNJ","WMT","COST","HD","PG","KO","XOM","CVX","CAT","BA","DIS","MCD","NKE"]
UNIVERSE = sorted(set(HIGH_BETA + MEGA))

YEARS = 9
RR = 2.5
MAXHOLD = 15
MAX_RISK = 0.06


def _rsi(close, period=14):
    d=np.diff(close,prepend=close[0]); g=np.where(d>0,d,0.0); l=np.where(d<0,-d,0.0)
    out=np.full_like(close,50.0,dtype=float); ag=al=0.0
    for i in range(1,len(close)):
        if i<=period: ag=(ag*(i-1)+g[i])/i; al=(al*(i-1)+l[i])/i
        else: ag=(ag*(period-1)+g[i])/period; al=(al*(period-1)+l[i])/period
        out[i]=100.0 if al<1e-12 else 100.0-100.0/(1.0+ag/al)
    return out


def _load(t):
    try:
        end=datetime.now(); start=end-timedelta(days=int(YEARS*365.25)+300)
        raw=yf.download(t,start=start,end=end,progress=False,auto_adjust=True)
        if raw is None or raw.empty or len(raw)<400: return None
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
        df=raw.dropna(subset=["Close"]).copy()
        df["rsi"]=_rsi(df["Close"].values,14)
        df["sma50"]=df["Close"].rolling(50).mean()
        df["vol20"]=df["Volume"].rolling(20).mean()
        df["hi252"]=df["High"].rolling(252).max()
        return df.dropna()
    except Exception:
        return None


def _signals(df):
    C=df["Close"].values;O=df["Open"].values;H=df["High"].values;L=df["Low"].values
    rsi=df["rsi"].values; s50=df["sma50"].values; vol=df["Volume"].values
    v20=df["vol20"].values; hi=df["hi252"].values
    n=len(C); sig=np.zeros(n,dtype=bool); PEAK_LB=10
    for i in range(260,n):
        # made a 252-day high within the last PEAK_LB bars (recent peak), and was
        # extended above the 50-SMA at that peak (stretched) — then reverses.
        recent_peak = H[i-PEAK_LB:i].max()
        if not (hi[i]>0 and recent_peak >= 0.995*hi[i-PEAK_LB]): continue  # peaked recently
        peak_j = i-PEAK_LB + int(np.argmax(H[i-PEAK_LB:i]))
        if not (s50[peak_j]>0 and C[peak_j] >= 1.12*s50[peak_j]): continue # was extended
        # exhaustion visible around the peak: bearish RSI divergence or drying volume
        rsi_lh = rsi[peak_j] < rsi[max(0,peak_j-15):peak_j].max()-3
        vol_dry = v20[peak_j]>0 and vol[peak_j-5:peak_j].mean() < 1.0*v20[peak_j]
        if not (rsi_lh or vol_dry): continue
        # decisive trigger NOW: close breaks below the 10-day low on a down day,
        # and we're still within a sensible distance of the peak (early in the break)
        if not (C[i] < L[i-10:i].min() and C[i] < O[i]): continue
        if C[i] < 0.88*recent_peak: continue        # not too late (already crashed)
        sig[i]=True
    return sig


def _trades(df):
    sig=_signals(df); C=df["Close"].values; H=df["High"].values; L=df["Low"].values
    n=len(C); out=[]; i=260
    while i < n-1:
        if not sig[i]: i+=1; continue
        entry=C[i]
        swing_hi=H[max(0,i-5):i+1].max()
        stop=swing_hi*1.001
        if entry<=0 or stop<=entry: i+=1; continue
        risk=(stop-entry)/entry
        if risk>MAX_RISK or risk<0.005: i+=1; continue
        target=entry*(1-RR*risk)
        realized=None; exit_i=min(i+MAXHOLD,n-1)
        for j in range(i+1,exit_i+1):
            if H[j]>=stop: realized=-risk; exit_i=j; break       # stopped (rallied)
            if L[j]<=target: realized=RR*risk; exit_i=j; break   # target (dropped)
        if realized is None:
            realized=(entry-C[exit_i])/entry                      # short P&L at close
        out.append(realized)
        i=exit_i+1
    return out


def _stats(tr):
    a=np.asarray(tr,float); n=len(a)
    if n==0: return None
    wins=a[a>0]; losses=a[a<0]
    aw=wins.mean() if len(wins) else 0; al=losses.mean() if len(losses) else 0
    pf=(wins.sum()/-losses.sum()) if losses.sum()<0 else float('inf')
    return dict(n=n, win=100*(a>0).mean(), exp=a.mean()*100,
                aw=aw*100, al=al*100, rr=(aw/-al if al<0 else float('inf')), pf=pf)


def run():
    print(f"Loading {len(UNIVERSE)} names, {YEARS}y...")
    data={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")

    print("="*84)
    print(" EXHAUSTION-REVERSAL PUTS V2 — managed exit (RR 2.5, stop@swing-high, 15d)")
    print("="*84)
    for name, names in [("ALL names", list(data.keys())),
                        ("HIGH-BETA only", [t for t in HIGH_BETA if t in data])]:
        tr=[]
        for t in names:
            tr += _trades(data[t])
        s=_stats(tr)
        print(f"\n  ▶ {name}  ({len(names)} tickers)")
        if not s or s["n"]<15:
            print(f"    only {s['n'] if s else 0} trades — too few."); continue
        print(f"    n={s['n']}  win {s['win']:.1f}%  expectancy {s['exp']:+.2f}%/trade  "
              f"R:R {s['rr']:.2f}  PF {s['pf']:.2f}")
        print(f"    avg win {s['aw']:+.2f}%   avg loss {s['al']:+.2f}%   "
              f"(~{s['n']/YEARS:.0f} trades/yr)")

    # MCPT on the full bucket
    try:
        from permutation_test import get_permutation, metric_from_trades
        real=[]
        for df in data.values(): real += _trades(df)
        rm=metric_from_trades(real)
        null=[]
        for p in range(150):
            pt=[]
            for ti,df in enumerate(data.values()):
                need=["Open","High","Low","Close","Volume"]
                perm=get_permutation(df[need],start_index=0,seed=p*1000+ti).copy()
                c=perm["Close"]
                perm["rsi"]=_rsi(c.values,14); perm["sma50"]=c.rolling(50).mean()
                perm["vol20"]=perm["Volume"].rolling(20).mean()
                perm["hi252"]=perm["High"].rolling(252).max()
                perm=perm.dropna()
                if len(perm)>300: pt += _trades(perm)
            null.append(metric_from_trades(pt)["profit_factor"])
        null=np.array(null)
        p_pf=(1+int((null>=rm["profit_factor"]).sum()))/(151)
        print("\n" + "="*84)
        print(f"  MCPT (ALL): real PF {rm['profit_factor']} (n={rm['n']}) · "
              f"null mean {null.mean():.2f}/95th {np.percentile(null,95):.2f} · p={round(p_pf,4)}")
        print("  VERDICT:", "REAL edge" if p_pf<=0.05 else "weak/none" if p_pf<=0.2 else "NO edge")
    except Exception as e:
        print("  MCPT skipped:", e)


if __name__ == "__main__":
    run()
