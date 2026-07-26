"""
overnight_regime_test.py — The reviewer's overnight x regime interaction test.

Known: QQQ's return lives overnight (close->open), not intraday. Open question:
is that overnight premium a MOMENTUM signal or an independent RISK/FEAR premium?

Reviewer's hypothesis: it's a fear premium, so it should be AT LEAST as strong —
maybe stronger — when the trend filter is in CASH (market below its SMAs). If so,
the counterintuitive play is to harvest overnight QQQ specifically when the regime
filter has us out of the daytime trend.

TEST (10y): split every session by regime state and compare the overnight
(prev-close -> today-open) return:
  * UPTREND   : QQQ above its 50 & 200 SMA
  * DOWNTREND : QQQ below (the filter would be in cash for the day trade)
  * also: the day right AFTER a cash->long flip
Report mean overnight return, win rate, and annualised contribution per bucket,
plus intraday (open->close) for contrast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

YEARS = 10
MOM = 63


def _load(t):
    end = datetime.now(); start = end - timedelta(days=int(YEARS * 365.25) + 260)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close", "Open"]).copy(); c = df["Close"]
    df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
    df["mom"] = c / c.shift(MOM) - 1
    # overnight = prev close -> today open ; intraday = today open -> today close
    df["overnight"] = df["Open"] / c.shift(1) - 1
    df["intraday"] = df["Close"] / df["Open"] - 1
    return df.dropna()


def _bucket(name, ov, intr):
    ov = np.asarray(ov, float); intr = np.asarray(intr, float)
    if len(ov) == 0:
        print(f"    {name:24s} n=0"); return
    ann_ov = (np.prod(1 + ov) ** (252 / len(ov)) - 1) * 100
    print(f"    {name:24s} n={len(ov):<5} overnight mean {ov.mean()*100:+.3f}%  "
          f"win {100*(ov>0).mean():.0f}%  ann {ann_ov:+.0f}%   | intraday mean "
          f"{intr.mean()*100:+.3f}%")


def run():
    print(f"Loading QQQ (OHLC), {YEARS}y...")
    df = _load("QQQ")
    if df is None:
        print("  FATAL"); return
    c = df["Close"]; s50 = df["sma50"]; s200 = df["sma200"]; m = df["mom"]
    up = ((c > s50) & (c > s200) & (m > 0))
    ov = df["overnight"]; intr = df["intraday"]
    idx = df.index
    long = up.astype(int)
    flip_up = long.diff().fillna(0) == 1     # cash->long today
    # "day after a cash->long flip" -> overnight into the next session
    after_flip = flip_up.shift(1).fillna(False)

    print("\n" + "=" * 84)
    print(" OVERNIGHT PREMIUM x REGIME  (QQQ close->open, split by trend state)")
    print("=" * 84)
    _bucket("ALL days", ov, intr)
    _bucket("UPTREND (filter long)", ov[up], intr[up])
    _bucket("DOWNTREND (filter cash)", ov[~up], intr[~up])
    _bucket("day after cash->long", ov[after_flip], intr[after_flip])

    ann_all = (np.prod(1 + ov.values) ** (252 / len(ov)) - 1) * 100
    ann_up = (np.prod(1 + ov[up].values) ** (252 / max(up.sum(), 1)) - 1) * 100
    ann_dn = (np.prod(1 + ov[~up].values) ** (252 / max((~up).sum(), 1)) - 1) * 100
    print("\n" + "=" * 84)
    print(" VERDICT")
    print("=" * 84)
    print(f"    overnight annualised:  UPTREND {ann_up:+.0f}%   DOWNTREND {ann_dn:+.0f}%"
          f"   (all {ann_all:+.0f}%)")
    if ann_dn >= ann_up * 0.9:
        print("    -> HYPOTHESIS HOLDS: overnight premium is at least as strong in")
        print("       DOWNTREND/cash. It's a fear/risk premium, independent of trend.")
        print("       Play: harvest overnight QQQ WHEN the day-filter is in cash.")
    else:
        print("    -> Overnight premium is weaker in downtrends; it tracks the trend")
        print("       more than fear. Keep overnight aligned with the long regime.")
    print("\n  (Mind survivorship of the regime window; treat as directional, not exact.)")


if __name__ == "__main__":
    run()
