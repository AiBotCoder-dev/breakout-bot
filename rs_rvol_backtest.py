"""
rs_rvol_backtest.py — Do Relative Strength + Relative Volume actually lift win rate?

Before wiring RS/RVOL into the setup score, prove they earn it. We take the two
validated long setups and check whether adding each filter improves the forward
win rate vs the unfiltered baseline:

  MOMENTUM proxy  — price > rising 50SMA & 200SMA (an "in an uptrend" long)
  DIP proxy       — RSI14<35 near a 60d low that ticks green (bottom-fisher-like)

FILTERS TESTED:
  RS   — 3-month return of the NAME minus 3-month return of SPY > 0 (and a
         stronger >5% variant). "Is this name leading the market?"
  RVOL — today's volume >= 1.3x its 20-day average. "Is the move real?"

For each setup we measure forward 10-day return, win rate, and the LIFT each
filter adds. A filter ships only if it raises win rate without gutting the
sample. Honest baselines, real data, ~6y, liquid universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = [
    "AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","TSLA","AVGO","NFLX",
    "CRM","ADBE","INTC","CSCO","QCOM","TXN","MU","AMAT","ORCL","PLTR","COIN",
    "SHOP","UBER","PYPL","ROKU","DKNG","RBLX","JPM","BAC","GS","V","MA","AXP",
    "UNH","LLY","JNJ","ABBV","MRK","PFE","WMT","COST","HD","MCD","NKE","DIS",
    "CAT","BA","GE","XOM","CVX","FCX","SOFI","HOOD","SMCI","MRVL","SNOW","NET",
]
FWD = 10


def _load(t):
    try:
        end = datetime.now(); start = end - timedelta(days=int(6*365.25)+260)
        raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 300:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        c = df["Close"]
        df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
        df["vol20"] = df["Volume"].rolling(20).mean()
        df["ret63"] = c / c.shift(63) - 1                # 3-month return
        d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - 100/(1 + up/dn)
        df["low60"] = df["Low"].rolling(60).min()
        return df.dropna()
    except Exception:
        return None


def run():
    print(f"Loading {len(UNIVERSE)} names + SPY, 6y...")
    data = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_load, t): t for t in UNIVERSE + ["SPY"]}
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                data[futs[f]] = r
    spy = data.pop("SPY", None)
    if spy is None:
        print("no SPY"); return
    spy_ret63 = spy["ret63"]
    print(f"  loaded {len(data)}\n")

    # collect forward returns per setup, split by filter pass/fail
    cats = ["MOMENTUM", "DIP"]
    res = {c: {"base": [], "rs": [], "rs_strong": [], "rvol": [], "rs+rvol": []}
           for c in cats}

    for t, df in data.items():
        C = df["Close"].values; H = df["High"].values; L = df["Low"].values
        s50 = df["sma50"].values; s200 = df["sma200"].values
        vol = df["Volume"].values; v20 = df["vol20"].values
        rsi = df["rsi"].values; low60 = df["low60"].values
        r63 = df["ret63"].values
        idx = df.index
        # align SPY 3m return by date
        spy_aligned = spy_ret63.reindex(idx).values
        n = len(C)
        for i in range(210, n - FWD):
            fwd = (C[i + FWD] / C[i] - 1) * 100
            rs_diff = (r63[i] - spy_aligned[i]) * 100 if spy_aligned[i] == spy_aligned[i] else 0
            rvol = vol[i] / v20[i] if v20[i] > 0 else 0
            setups = []
            if C[i] > s50[i] > s200[i] and s50[i] > s50[i-10] and s200[i] > s200[i-20]:
                setups.append("MOMENTUM")
            if rsi[i] < 35 and low60[i] > 0 and (C[i]/low60[i]-1) <= 0.06 and C[i] > C[i-1]:
                setups.append("DIP")
            for cat in setups:
                res[cat]["base"].append(fwd)
                if rs_diff > 0:       res[cat]["rs"].append(fwd)
                if rs_diff > 5:       res[cat]["rs_strong"].append(fwd)
                if rvol >= 1.3:       res[cat]["rvol"].append(fwd)
                if rs_diff > 0 and rvol >= 1.3: res[cat]["rs+rvol"].append(fwd)

    def stat(a):
        a = np.asarray(a, dtype=float)
        if len(a) == 0: return (0, 0, 0)
        return (len(a), 100*(a > 0).mean(), a.mean())

    print("="*82)
    print(f" RS / RVOL FILTER LIFT — forward {FWD}-day win rate & mean return")
    print("="*82)
    for cat in cats:
        print(f"\n  ▶ {cat} setup")
        b_n, b_w, b_m = stat(res[cat]["base"])
        print(f"    {'baseline (no filter)':<26} n={b_n:<6} win {b_w:.1f}%  mean {b_m:+.2f}%")
        for key, lbl in [("rs", "RS > 0 (beating SPY)"),
                         ("rs_strong", "RS > 5% (clearly leading)"),
                         ("rvol", "RVOL >= 1.3x"),
                         ("rs+rvol", "RS>0 AND RVOL>=1.3x")]:
            nn, ww, mm = stat(res[cat][key])
            lift = ww - b_w
            flag = "  ✅" if (lift > 1.5 and nn >= 200) else ("  ~" if lift > 0 else "  ❌")
            print(f"    {lbl:<26} n={nn:<6} win {ww:.1f}%  mean {mm:+.2f}%  "
                  f"(lift {lift:+.1f}pp){flag}")


if __name__ == "__main__":
    run()
