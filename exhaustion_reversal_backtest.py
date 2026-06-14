"""
exhaustion_reversal_backtest.py — "Max-profit reversal at extremes" (the user's idea).

THE IDEA: wait for a stock at/near its extreme (52-week / all-time high or low),
look for SLOWING pressure (momentum exhaustion: divergence + drying volume), and
enter ONLY once the reversal actually TRIGGERS. Rare setups → should be high win
rate. For options: at a complacent high IV is cheap, and the reversal expands IV
hard, so a put gains on BOTH delta and vega ("max profit").

THE HONEST RISK: shorting new highs is a known LOSER (momentum continues). So the
whole question this tests is: does requiring EXHAUSTION + a TRIGGER flip a
near-extreme from negative-edge into positive-edge? We test it as a clean 2x2:

  A) near-high + trigger ONLY      (no exhaustion filter)   -> baseline reversal
  B) near-high + trigger + EXHAUSTION (divergence/volume)   -> does it add edge?

If B clearly beats A and the baseline, the "charts indicate it before it happens"
thesis holds. We measure the PUT side (tops) and CALL side (bottoms), forward
returns we WANT (down for puts, up for calls), win rate, magnitude, and MCPT.

SIGNALS (evaluated at bar i, using only data up to i):
  NEAR HIGH : close >= 0.97 * max(high, last 252 bars)   (within 3% of the high)
  NEAR LOW  : close <= 1.03 * min(low,  last 252 bars)
  EXHAUSTION (top): bearish RSI divergence (price HH, RSI LH over ~15 bars)
                    OR the recent push to the high came on FALLING volume
  EXHAUSTION (bot): bullish RSI divergence (price LL, RSI HL)
                    OR capitulation volume spike then fade
  TRIGGER (top): close < min(low[i-3:i])  (breaks the short-term floor) AND
                 close < open (down day)
  TRIGGER (bot): close > max(high[i-3:i]) AND close > open
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Universe: liquid + high-beta names that actually reverse hard, + indices
UNIVERSE = [
    "SPY","QQQ","IWM","AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","TSLA",
    "AVGO","NFLX","CRM","ADBE","INTC","QCOM","MU","AMAT","ORCL","PLTR","COIN",
    "SHOP","UBER","PYPL","ROKU","DKNG","RBLX","SNAP","JPM","BAC","GS","V","MA",
    "UNH","LLY","JNJ","ABBV","MRK","PFE","WMT","COST","HD","MCD","NKE","DIS",
    "CAT","BA","GE","XOM","CVX","FCX","SOFI","HOOD","SMCI","MRVL","SNOW","NET",
    "CRWD","DDOG","ARM","MSTR","ENPH","CVNA","AFRM","UPST","RIVN","LCID",
]
YEARS = 9
FWD = [5, 10, 20]
HIGH_PROX = 0.03
NEAR_LEN = 252


def _rsi(close, period=14):
    d = np.diff(close, prepend=close[0])
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = np.full_like(close, 50.0, dtype=float); ag = al = 0.0
    for i in range(1, len(close)):
        if i <= period:
            ag = (ag*(i-1)+g[i])/i; al = (al*(i-1)+l[i])/i
        else:
            ag = (ag*(period-1)+g[i])/period; al = (al*(period-1)+l[i])/period
        out[i] = 100.0 if al < 1e-12 else 100.0 - 100.0/(1.0+ag/al)
    return out


def _load(t):
    try:
        end = datetime.now(); start = end - timedelta(days=int(YEARS*365.25)+300)
        raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 400:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        df["rsi"] = _rsi(df["Close"].values, 14)
        df["vol20"] = df["Volume"].rolling(20).mean()
        df["hi252"] = df["High"].rolling(NEAR_LEN).max()
        df["lo252"] = df["Low"].rolling(NEAR_LEN).min()
        return df.dropna()
    except Exception:
        return None


def _signals(df, side, mode):
    """side='top'(puts) or 'bot'(calls); mode='trigger_only' or 'exhaustion'."""
    C=df["Close"].values; O=df["Open"].values; H=df["High"].values; L=df["Low"].values
    rsi=df["rsi"].values; vol=df["Volume"].values; v20=df["vol20"].values
    hi=df["hi252"].values; lo=df["lo252"].values
    n=len(C); sig=np.zeros(n,dtype=bool)
    for i in range(260, n):
        if side == "top":
            near = hi[i] > 0 and C[i] >= (1-HIGH_PROX)*hi[i]
            if not near: continue
            trig = C[i] < L[i-3:i].min() and C[i] < O[i]
            if not trig: continue
            if mode == "exhaustion":
                # bearish RSI divergence: price higher-high vs ~15 bars ago, RSI lower-high
                hh = C[i] > C[i-15:i].max()*0.999
                rsi_lh = rsi[i] < rsi[i-15:i].max() - 3
                vol_dry = v20[i] > 0 and vol[i-5:i].mean() < 0.95 * v20[i]
                if not ((hh and rsi_lh) or vol_dry):
                    continue
            sig[i] = True
        else:  # bottom (calls)
            near = lo[i] > 0 and C[i] <= (1+HIGH_PROX)*lo[i]
            if not near: continue
            trig = C[i] > H[i-3:i].max() and C[i] > O[i]
            if not trig: continue
            if mode == "exhaustion":
                ll = C[i] < C[i-15:i].min()*1.001
                rsi_hl = rsi[i] > rsi[i-15:i].min() + 3
                capit = v20[i] > 0 and vol[i-5:i].max() > 1.6 * v20[i]
                if not ((ll and rsi_hl) or capit):
                    continue
            sig[i] = True
    return sig


def _forward(df, sig, side):
    """Return list of 'profit-direction' forward returns (positive = trade worked)."""
    C=df["Close"].values; H=df["High"].values; L=df["Low"].values; n=len(C)
    out={w:[] for w in FWD}; mfe=[]
    for i in np.where(sig)[0]:
        for w in FWD:
            if i+w < n and C[i] > 0:
                raw=(C[i+w]/C[i]-1)*100
                out[w].append(-raw if side=="top" else raw)  # puts profit on down
        # max favorable excursion over 20d (how big the move got)
        if i+20 < n and C[i] > 0:
            if side=="top":
                mfe.append((C[i] - L[i+1:i+21].min())/C[i]*100)   # biggest drop
            else:
                mfe.append((H[i+1:i+21].max() - C[i])/C[i]*100)   # biggest pop
    return out, mfe


def run():
    print(f"Loading {len(UNIVERSE)} names, {YEARS}y...")
    data={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_load,t):t for t in UNIVERSE}
        for f in as_completed(futs):
            r=f.result()
            if r is not None: data[futs[f]]=r
    print(f"  loaded {len(data)}\n")

    # baseline forward (random) for the profit direction
    base={"top":{w:[] for w in FWD}, "bot":{w:[] for w in FWD}}
    for df in data.values():
        C=df["Close"].values; n=len(C)
        for i in range(260,n):
            for w in FWD:
                if i+w<n and C[i]>0:
                    raw=(C[i+w]/C[i]-1)*100
                    base["top"][w].append(-raw); base["bot"][w].append(raw)

    for side, label in [("top","TOP reversal → BUY PUTS (exhausted highs)"),
                        ("bot","BOTTOM reversal → BUY CALLS (capitulated lows)")]:
        print("="*90); print(f" {label}"); print("="*90)
        bmean={w:np.mean(base[side][w]) for w in FWD}
        print(f"  baseline random (profit-dir): " +
              "  ".join(f"{w}d {bmean[w]:+.2f}%/win{100*(np.array(base[side][w])>0).mean():.0f}%" for w in FWD))
        for mode,mlabel in [("trigger_only","A) near-extreme + trigger ONLY"),
                            ("exhaustion","B) + EXHAUSTION filter (divergence/volume)")]:
            pooled={w:[] for w in FWD}; allmfe=[]
            for df in data.values():
                s=_signals(df,side,mode); f,mfe=_forward(df,s,side)
                for w in FWD: pooled[w].extend(f[w])
                allmfe.extend(mfe)
            nn=len(pooled[FWD[0]])
            print(f"\n  {mlabel}   (n={nn} signals over {YEARS}y across {len(data)} names)")
            if nn<15:
                print("    too few signals"); continue
            for w in FWD:
                a=np.array(pooled[w])
                print(f"    {w:>2}d: win {100*(a>0).mean():5.1f}%  mean {a.mean():+.2f}%  "
                      f"median {np.median(a):+.2f}%  edge {a.mean()-bmean[w]:+.2f}%  "
                      f"p75 {np.percentile(a,75):+.1f}%")
            mfe=np.array(allmfe)
            print(f"    max-favorable-excursion over 20d: median {np.median(mfe):.1f}%  "
                  f"p75 {np.percentile(mfe,75):.1f}%  p90 {np.percentile(mfe,90):.1f}%")
        print()


if __name__ == "__main__":
    run()
