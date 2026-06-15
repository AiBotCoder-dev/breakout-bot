"""
full_bot_backtest.py — Replay the BOT'S call decision process over 5y / 5 stocks.

Marks every bar where the bot WOULD buy a call (its two validated call edges) and
measures accuracy two ways:
  1. Directional  — did the underlying rise over the 10-day hold? (the thesis)
  2. Option       — would a NEAR-MONEY call (the bot's new default structure)
                    actually have MADE money? (BS-priced, the real test)

The bot's CALL process (validated edges only — the ones it actually trades):
  MOMENTUM  : price > rising 50SMA > rising 200SMA AND 6-mo momentum > +10%
              (the MCPT p=0.004 cross-sectional momentum entry)
  BOTTOM-FISHER : RSI(14) < 30 AND within 6% of the 60-day low AND today closes
              green (the MCPT p=0.005 oversold-at-support entry)

Non-overlapping (one trade at a time per name). Lists every signal so you can
eyeball the points, then reports per-stock + aggregate accuracy.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

STOCKS = ["NVDA", "AAPL", "TSLA", "AMD", "JPM"]
HOLD = 10           # trading days (≈ the bot's 2-week option window)
R = 0.04
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * N(d1) - K * math.exp(-R * T) * N(d2)


def _rsi(c, p=14):
    d = np.diff(c, prepend=c[0]); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = np.full_like(c, 50.0, float); ag = al = 0.0
    for i in range(1, len(c)):
        if i <= p: ag = (ag*(i-1)+g[i])/i; al = (al*(i-1)+l[i])/i
        else: ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+l[i])/p
        out[i] = 100.0 if al < 1e-12 else 100.0 - 100.0/(1.0+ag/al)
    return out


def _load(t):
    end = datetime.now(); start = end - timedelta(days=int(5*365.25)+260)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy()
    c = df["Close"]
    df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
    df["rv20"] = np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["rsi"] = _rsi(c.values, 14)
    df["low60"] = df["Low"].rolling(60).min()
    df["mom6"] = c/c.shift(126) - 1
    return df.dropna()


def run():
    print(f"Replaying the bot's CALL process — {len(STOCKS)} stocks, 5y, {HOLD}d hold\n")
    agg_dir, agg_opt, agg_setup = [], [], []
    per_stock = {}
    examples = {}

    for t in STOCKS:
        try:
            df = _load(t)
        except Exception as e:
            print(f"{t}: load failed {e}"); continue
        C = df["Close"].values; H = df["High"].values; L = df["Low"].values
        s50 = df["sma50"].values; s200 = df["sma200"].values
        rsi = df["rsi"].values; low60 = df["low60"].values
        mom6 = df["mom6"].values; rv = df["rv20"].values
        idx = df.index
        n = len(C); i = 210
        sigs = []
        while i < n - HOLD:
            setup = None
            if (C[i] > s50[i] > s200[i] and s50[i] > s50[i-10]
                    and s200[i] > s200[i-20] and mom6[i] > 0.10):
                setup = "MOMENTUM"
            elif (rsi[i] < 30 and low60[i] > 0 and (C[i]/low60[i]-1) <= 0.06
                  and C[i] > C[i-1]):
                setup = "BOTTOM_FISHER"
            if setup:
                S0 = C[i]; S1 = C[i+HOLD]
                dir_ret = (S1/S0 - 1) * 100
                iv = max(0.15, min(1.5, rv[i]*1.1))
                K = S0 * 1.03            # near-money call (the bot's default band)
                e = bs_call(S0, K, 21/365, iv); x = bs_call(S1, K, 11/365, iv)
                opt_ret = (x/e - 1) * 100 if e > 0.01 else 0.0
                sigs.append((idx[i].date(), setup, round(S0, 2),
                             round(dir_ret, 1), round(opt_ret, 0)))
                agg_dir.append(dir_ret); agg_opt.append(opt_ret); agg_setup.append(setup)
                i += HOLD               # non-overlapping
            else:
                i += 1
        per_stock[t] = sigs
        examples[t] = sigs
        d = np.array([s[3] for s in sigs]); o = np.array([s[4] for s in sigs])
        if len(sigs):
            print(f"=== {t}: {len(sigs)} call signals ===")
            print(f"   directional win {100*(d>0).mean():.0f}%  ·  "
                  f"option win {100*(o>0).mean():.0f}%  ·  "
                  f"avg option {o.mean():+.0f}%  median {np.median(o):+.0f}%")
            # show a handful of example signal points
            for s in sigs[:6]:
                tag = "WIN " if s[4] > 0 else "loss"
                print(f"     {s[0]}  {s[1]:<13} ${s[2]:>8.2f}  10d {s[3]:+5.1f}%  "
                      f"-> call {s[4]:+4.0f}%  [{tag}]")
            print()

    d = np.array(agg_dir); o = np.array(agg_opt)
    print("=" * 70)
    print(f" AGGREGATE — {len(d)} call signals across {len(STOCKS)} stocks, 5y")
    print("=" * 70)
    print(f"  Directional accuracy (stock up in 10d) : {100*(d>0).mean():.1f}%")
    print(f"  OPTION win rate (near-money call green) : {100*(o>0).mean():.1f}%")
    print(f"  Option mean / median return            : {o.mean():+.0f}% / {np.median(o):+.0f}%")
    print(f"  Option expectancy per trade            : {o.mean():+.1f}%")
    for st in ("MOMENTUM", "BOTTOM_FISHER"):
        m = np.array([agg_opt[k] for k in range(len(agg_setup)) if agg_setup[k] == st])
        if len(m):
            print(f"    {st:<14} n={len(m):<4} option win {100*(m>0).mean():.0f}%  "
                  f"avg {m.mean():+.0f}%")


if __name__ == "__main__":
    run()
