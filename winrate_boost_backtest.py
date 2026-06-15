"""
winrate_boost_backtest.py — Which changes actually push OPTION win rate > 55%?

The gap: the bot's call signals are ~61% directionally right but the near-money
option only wins ~42% (theta + strike distance eat the small moves). This tests
the concrete levers to lift the OPTION win rate, on the momentum signals, with
DAILY PATH repricing (so profit-taking exits are modeled honestly):

  A. Near-money (3% OTM), hold 10d                — current
  B. ITM 5% (delta ~0.6), hold 10d                — higher delta
  C. Deep ITM 10% (delta ~0.7), hold 10d          — stock-like
  D. Near-money + take-profit at +50%             — exit rule only
  E. ITM 5% + take-profit at +40%                 — delta + profit-taking
  F. Debit spread (ITM long / ATM short), hold 10d

Reports win rate + mean + median for each so you can see which clears 55% and
what it costs in upside. Honest note printed: win rate is NOT profit.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

UNIVERSE = ["AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","AVGO","NFLX","CRM",
            "JPM","V","MA","COST","HD","CAT","UNH","LLY","WMT","ORCL"]
HOLD = 10
R = 0.04
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return S * N(d1) - K * math.exp(-R * T) * N(d1 - sig * math.sqrt(T))


def _load(t):
    end = datetime.now(); start = end - timedelta(days=int(5*365.25)+260)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy()
    c = df["Close"]
    df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
    df["rv20"] = np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["mom6"] = c/c.shift(126) - 1
    return df.dropna()


def _trade(path, iv, K_mult, tp=None, spread=False):
    """path = [S0..S_HOLD]; return option % return with optional take-profit."""
    S0 = path[0]; Te = 21/365
    if spread:
        kl, ks = S0*0.97, S0*1.03
        entry = bs_call(S0,kl,Te,iv) - bs_call(S0,ks,Te,iv)
        def val(S,T): return bs_call(S,kl,T,iv) - bs_call(S,ks,T,iv)
    else:
        K = S0*K_mult
        entry = bs_call(S0,K,Te,iv)
        def val(S,T): return bs_call(S,K,T,iv)
    if entry <= 0.01:
        return None
    for d in range(1, len(path)):
        T = (21-d)/365
        v = val(path[d], T)
        r = (v/entry - 1)*100
        if tp is not None and r >= tp:
            return tp                     # booked the profit target
    return (val(path[-1], (21-HOLD)/365)/entry - 1)*100


def run():
    print(f"Loading {len(UNIVERSE)} names, 5y...")
    data = {}
    for t in UNIVERSE:
        try:
            data[t] = _load(t)
        except Exception:
            pass
    print(f"  loaded {len(data)}\n")
    configs = {
        "A near-money (3% OTM) hold":      dict(K_mult=1.03),
        "B ITM 5% hold":                   dict(K_mult=0.95),
        "C deep ITM 10% hold":             dict(K_mult=0.90),
        "D near-money + TP+50%":           dict(K_mult=1.03, tp=50),
        "E ITM 5% + TP+40%":               dict(K_mult=0.95, tp=40),
        "F debit spread (ITM/ATM) hold":   dict(spread=True, K_mult=1.0),
    }
    res = {k: [] for k in configs}
    for df in data.values():
        C=df["Close"].values; s50=df["sma50"].values; s200=df["sma200"].values
        rv=df["rv20"].values; mom6=df["mom6"].values; n=len(C); i=210
        while i < n-HOLD:
            if (C[i]>s50[i]>s200[i] and s50[i]>s50[i-10] and s200[i]>s200[i-20] and mom6[i]>0.10):
                iv=max(0.15,min(1.5,rv[i]*1.1)); path=C[i:i+HOLD+1]
                for k,kw in configs.items():
                    r=_trade(path, iv, **kw)
                    if r is not None: res[k].append(r)
                i += HOLD
            else:
                i += 1
    print("="*74)
    print(" OPTION WIN-RATE LEVERS (momentum signals, daily-path repriced)")
    print("="*74)
    for k in configs:
        a=np.array(res[k])
        if len(a)==0: continue
        win=100*(a>0).mean()
        flag = "  ✅>55" if win>=55 else ""
        print(f"  {k:<32} n={len(a):<5} win {win:5.1f}%  mean {a.mean():+6.1f}%  "
              f"median {np.median(a):+6.1f}%{flag}")
    print("\n  NOTE: win rate and profit are NOT the same. Higher-win configs cap")
    print("  the home runs (lower mean). Read win rate + mean together.")


if __name__ == "__main__":
    run()
