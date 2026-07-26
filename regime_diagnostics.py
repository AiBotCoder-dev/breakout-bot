"""
regime_diagnostics.py — The reviewer's "one hard pushback" on timed_QQQ.

Two worries about the trend filter that the headline Sortino hides:
  1. WHIPSAW: in a grinding sideways market (2022), does the 50/200-SMA filter
     flip in/out repeatedly, bleeding slippage + missed re-entry gaps each time?
  2. IS IT JUST "SMARTER SPY"? If timed_QQQ's returns are ~perfectly correlated
     with SPY, the "alpha" is really beta-timing, and it whipsaws in chop.

This measures, over the 8y sample and zoomed into 2022:
  * flip count (cash<->long transitions) per calendar year
  * average return in the 5 days AFTER each cash->long flip (the "re-entry tax":
    negative means we tend to re-enter into a bounce that fades)
  * 90-day rolling correlation of timed_QQQ daily returns vs SPY, summarised
  * timed_QQQ's 2022 return vs QQQ buy-hold 2022 (did the filter actually help
    in the exact regime the reviewer fears?)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

YEARS = 8
MOM = 63


def _load(t):
    end = datetime.now(); start = end - timedelta(days=int(YEARS * 365.25) + 260)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy(); c = df["Close"]
    df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
    df["mom"] = c / c.shift(MOM) - 1
    df["ret"] = c.pct_change()
    return df.dropna()


def run():
    print(f"Loading QQQ + SPY, {YEARS}y...")
    qqq = _load("QQQ"); spy = _load("SPY")
    if qqq is None or spy is None:
        print("  FATAL: load failed"); return

    c = qqq["Close"]; s50 = qqq["sma50"]; s200 = qqq["sma200"]; m = qqq["mom"]
    ret = qqq["ret"]; idx = qqq.index
    long = ((c > s50) & (c > s200) & (m > 0)).astype(int)   # regime state per day
    strat_ret = (ret * long).fillna(0.0)                     # timed_QQQ daily return

    # ── 1. whipsaw: flips per year + re-entry tax ────────────────────────────
    flips = long.diff().fillna(0)
    up_flips = idx[flips == 1]       # cash -> long
    dn_flips = idx[flips == -1]      # long -> cash
    print("\n" + "=" * 68)
    print(" WHIPSAW — regime flips per year (cash<->long transitions)")
    print("=" * 68)
    yrs = sorted(set(idx.year))
    for y in yrs:
        n_up = sum(1 for d in up_flips if d.year == y)
        n_dn = sum(1 for d in dn_flips if d.year == y)
        frac_long = long[idx.year == y].mean()
        tag = "  <-- the grinding bear" if y == 2022 else ""
        print(f"    {y}   flips: {n_up+n_dn:2d}  (in {n_up}, out {n_dn})   "
              f"time long {100*frac_long:3.0f}%{tag}")

    # re-entry tax: mean 5-day forward return after each cash->long flip
    taxes = []
    arr = ret.values
    for d in up_flips:
        i = idx.get_loc(d)
        if i + 5 < len(arr):
            taxes.append(float(np.prod(1 + arr[i+1:i+6]) - 1) * 100)
    if taxes:
        print(f"\n  Re-entry tax: after a cash->long flip, the next-5-day return "
              f"averaged {np.mean(taxes):+.2f}%")
        print(f"    (n={len(taxes)} flips; negative => we re-enter into fades. "
              f"share negative: {100*np.mean([t<0 for t in taxes]):.0f}%)")

    # ── 2. is it just "smarter SPY"? 90d rolling correlation ─────────────────
    spy_ret = spy["ret"].reindex(idx).fillna(0.0)
    roll = strat_ret.rolling(90).corr(spy_ret).dropna()
    print("\n" + "=" * 68)
    print(" BETA-TIMING CHECK — 90d rolling corr(timed_QQQ, SPY)")
    print("=" * 68)
    print(f"    median {roll.median():.2f}   mean {roll.mean():.2f}   "
          f"min {roll.min():.2f}   max {roll.max():.2f}")
    print(f"    share of days corr>0.85: {100*(roll>0.85).mean():.0f}%")
    print("    (high & stable corr => it's a timed SPY/beta play, not independent "
          "alpha;\n     the VALUE is then the timing/defensive overlay, not selection.)")

    # ── 3. did the filter actually help in 2022? ─────────────────────────────
    def _yr_ret(series, y):
        sub = series[series.index.year == y]
        return float(np.prod(1 + sub.values) - 1) * 100 if len(sub) else 0.0
    print("\n" + "=" * 68)
    print(" 2022 STRESS — timed_QQQ vs QQQ buy-hold in the exact feared regime")
    print("=" * 68)
    print(f"    timed_QQQ 2022 return : {_yr_ret(strat_ret, 2022):+.1f}%")
    print(f"    QQQ buy-hold 2022     : {_yr_ret(ret, 2022):+.1f}%")
    print(f"    timed_QQQ time-in-cash 2022: {100*(1-long[idx.year==2022].mean()):.0f}%")
    print("\n  READ: if 2022 flips are few AND the re-entry tax is small AND 2022")
    print("  return beats buy-hold, the whipsaw fear doesn't bite and the filter is")
    print("  real. If corr>0.85 dominates, call it 'timed beta' honestly — still")
    print("  useful, but it's a market-timing overlay, not stock/sector selection.")


if __name__ == "__main__":
    run()
