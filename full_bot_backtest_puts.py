"""
full_bot_backtest_puts.py — Replay the BOT'S PUT decision process (5y / 5 names).

Mirror of full_bot_backtest.py but for the bot's validated PUT edges, with the
REGIME GATE the bot actually enforces (it will NOT buy puts in a confirmed bull —
so most of a 2021-2026 bull sample is correctly EXCLUDED; that's the point).

The bot's PUT process (validated edges, price-based so we can replay them):
  PUT_BREAKDOWN  : price < 50SMA < 200SMA (confirmed downtrend) AND 6-mo momentum
                   < 0 AND realized vol >= 35% (crash fuel — puts are dead money
                   on low-vol names). This is put_engine.bearish_signal + the vol
                   filter the engine requires.
  EXHAUSTION     : peaked near the 252-high while extended >12% over 50SMA with
                   exhaustion (RSI divergence / volume fade), then broke the
                   10-day low on a down day (the p=0.013 reversal we validated).

REGIME GATE (applied per date): puts allowed only when SPY is NOT a confirmed
bull — i.e. SPY <= 2% above its 200SMA OR 50SMA < 200SMA. Faithful to
put_engine.market_allows_puts(). Universe = high-crash-fuel names (where the bot
actually trades puts), not low-vol names.

Accuracy two ways: directional (stock DOWN in 10d) and a near-money PUT (BS-priced).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

STOCKS = ["TSLA", "AMD", "COIN", "NVDA", "NFLX"]   # high-vol crash-fuel names
HOLD = 10
R = 0.04
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-R * T) * N(-d2) - S * N(-d1)


def _rsi(c, p=14):
    d = np.diff(c, prepend=c[0]); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = np.full_like(c, 50.0, float); ag = al = 0.0
    for i in range(1, len(c)):
        if i <= p: ag = (ag*(i-1)+g[i])/i; al = (al*(i-1)+l[i])/i
        else: ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+l[i])/p
        out[i] = 100.0 if al < 1e-12 else 100.0 - 100.0/(1.0+ag/al)
    return out


def _load(t, years=5):
    end = datetime.now(); start = end - timedelta(days=int(years*365.25)+300)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy()
    c = df["Close"]
    df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
    df["rv20"] = np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["rsi"] = _rsi(c.values, 14)
    df["hi252"] = df["High"].rolling(252).max()
    df["mom6"] = c/c.shift(126) - 1
    return df.dropna()


def _spy_put_ok():
    """Date-indexed bool: would the regime gate ALLOW puts on each date?"""
    spy = _load("SPY", years=6)
    c = spy["Close"]; s200 = spy["sma200"]; s50 = spy["sma50"]
    above = (c / s200 - 1) * 100
    bull = (above > 2) & (s50 > s200)
    return (~bull)            # puts allowed when NOT a confirmed bull


def run():
    print(f"Replaying the bot's PUT process — {len(STOCKS)} names, 5y, {HOLD}d hold")
    print("(REGIME-GATED: puts only when SPY is NOT a confirmed bull)\n")
    put_ok = _spy_put_ok()
    agg_dir, agg_opt, agg_setup = [], [], []

    for t in STOCKS:
        try:
            df = _load(t)
        except Exception as e:
            print(f"{t}: load failed {e}"); continue
        C = df["Close"].values; H = df["High"].values; L = df["Low"].values; O = df["Open"].values
        s50 = df["sma50"].values; s200 = df["sma200"].values
        rsi = df["rsi"].values; rv = df["rv20"].values
        hi = df["hi252"].values; mom6 = df["mom6"].values
        idx = df.index
        # align the SPY regime gate to this name's dates
        ok = put_ok.reindex(idx).fillna(False).values
        n = len(C); i = 260; sigs = []
        while i < n - HOLD:
            if not ok[i]:
                i += 1; continue
            setup = None
            # PUT_BREAKDOWN
            if (C[i] < s50[i] < s200[i] and mom6[i] < 0 and rv[i]*100 >= 35):
                setup = "BREAKDOWN"
            else:
                # EXHAUSTION peak-then-break
                rp = H[i-10:i].max()
                if rp > 0 and rp >= 0.995*hi[i-10]:
                    pj = i-10 + int(np.argmax(H[i-10:i]))
                    if s50[pj] > 0 and C[pj] >= 1.12*s50[pj]:
                        rsi_lh = rsi[pj] < rsi[max(0,pj-15):pj].max()-3
                        if rsi_lh and C[i] < L[i-10:i].min() and C[i] < O[i] and C[i] >= 0.88*rp:
                            setup = "EXHAUSTION"
            if setup:
                S0, S1 = C[i], C[i+HOLD]
                dir_ret = (S1/S0 - 1) * 100         # negative = put thesis right
                iv = max(0.18, min(1.6, rv[i]*1.1))
                K = S0 * 0.97                         # near-money put
                e = bs_put(S0, K, 21/365, iv); x = bs_put(S1, K, 11/365, iv)
                opt = (x/e - 1) * 100 if e > 0.01 else 0.0
                sigs.append((idx[i].date(), setup, round(S0,2), round(dir_ret,1), round(opt,0)))
                agg_dir.append(dir_ret); agg_opt.append(opt); agg_setup.append(setup)
                i += HOLD
            else:
                i += 1
        d = np.array([s[3] for s in sigs]); o = np.array([s[4] for s in sigs])
        if len(sigs):
            print(f"=== {t}: {len(sigs)} put signals ===")
            print(f"   directional win (stock DOWN) {100*(d<0).mean():.0f}%  ·  "
                  f"option win {100*(o>0).mean():.0f}%  ·  avg option {o.mean():+.0f}%  "
                  f"median {np.median(o):+.0f}%")
            for s in sigs[:6]:
                tag = "WIN " if s[4] > 0 else "loss"
                print(f"     {s[0]}  {s[1]:<10} ${s[2]:>8.2f}  10d {s[3]:+5.1f}%  "
                      f"-> put {s[4]:+4.0f}%  [{tag}]")
            print()
        else:
            print(f"=== {t}: 0 put signals (regime gate kept puts holstered) ===\n")

    d = np.array(agg_dir); o = np.array(agg_opt)
    print("=" * 70)
    if len(d) == 0:
        print(" AGGREGATE — 0 put signals. The regime gate correctly refused puts")
        print(" across a mostly-bull 5y window. That IS the bot working as designed.")
        return
    print(f" AGGREGATE — {len(d)} put signals across {len(STOCKS)} names, 5y (regime-gated)")
    print("=" * 70)
    print(f"  Directional accuracy (stock DOWN in 10d): {100*(d<0).mean():.1f}%")
    print(f"  OPTION win rate (near-money put green)   : {100*(o>0).mean():.1f}%")
    print(f"  Option mean / median return              : {o.mean():+.0f}% / {np.median(o):+.0f}%")
    print(f"  Option expectancy per trade              : {o.mean():+.1f}%")
    for st in ("BREAKDOWN", "EXHAUSTION"):
        m = np.array([agg_opt[k] for k in range(len(agg_setup)) if agg_setup[k] == st])
        if len(m):
            print(f"    {st:<11} n={len(m):<4} option win {100*(m>0).mean():.0f}%  avg {m.mean():+.0f}%")


if __name__ == "__main__":
    run()
