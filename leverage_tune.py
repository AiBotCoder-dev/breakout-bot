"""
leverage_tune.py — Where does the trend-filtered edge's Sortino PEAK vs leverage?

The decomposition (aggregation_test.py) showed the trend FILTER is the real MVP
(timed_QQQ Sortino 1.94) and that real 3x ETFs lose ~half their theoretical
leverage to decay. So the open question: applied to a trend-filtered exposure,
how much leverage is optimal before decay + drawdown overwhelm the return?

This sweeps L in {1, 1.5, 2, 2.5, 3} over TWO trend-filtered strategies, using
SYNTHETIC daily-rebalanced leverage (curve = product of 1 + L*daily_ret) so
volatility decay is modelled inherently. A small annual leverage DRAG models the
expense-ratio + borrow cost that real leveraged ETFs add on top of decay
(synthetic-only would flatter the high-L lines, as the real-vs-synth 3x gap in
aggregation_test proved).

  timed_QQQ   : long QQQ while it's above its 50 & 200 SMA and mom>0, else cash
  sector_rot  : weekly top-2 momentum rotation across 1x sector ETFs (already
                trend-filtered by the >50/>200 SMA entry condition)

Read the Sortino column: the peak L is the risk-adjusted sweet spot. Prior: well
below 3x.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

YEARS = 8
TOP_K = 2
MOM = 63
LEVS = [1.0, 1.5, 2.0, 2.5, 3.0]
ANNUAL_DRAG = 0.010          # ~1%/yr expense+borrow ABOVE the modelled decay, scaled by (L-1)

SECTORS = ["QQQ", "SOXX", "SPY", "FNGS", "XLK", "XLF", "IWM", "XBI",
           "XLV", "DIA", "XRT", "KRE"]


def _load(t):
    end = datetime.now(); start = end - timedelta(days=int(YEARS * 365.25) + 260)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty or len(raw) < 300:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy(); c = df["Close"]
    df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
    df["mom"] = c / c.shift(MOM) - 1
    return df.dropna()


def _maxdd(eq):
    e = np.asarray(eq, float); pk = np.maximum.accumulate(e)
    return float(((e - pk) / pk).min() * 100)


def _sortino(dr):
    a = np.asarray(dr, float); dn = a[a < 0]
    dd = np.sqrt(np.mean(dn ** 2)) if len(dn) else 1e-9
    return float(a.mean() / dd * np.sqrt(252)) if dd > 0 else 0.0


def _cagr(eq, days):
    yrs = days / 252.0
    return float((eq ** (1 / yrs) - 1) * 100) if yrs > 0 and eq > 0 else 0.0


def _lever(base_dr, L):
    """Synthetic daily-rebalanced leverage with an annual drag scaled by (L-1)."""
    daily_drag = ANNUAL_DRAG * (L - 1) / 252.0
    dr = [L * r - daily_drag for r in base_dr]
    eq = 1.0; curve = []
    for r in dr:
        eq *= (1 + r); curve.append(eq)
    return eq, curve, dr


def _timed_qqq(qqq, cal):
    dr = []
    c = qqq["Close"]; s50 = qqq["sma50"]; s200 = qqq["sma200"]; m = qqq["mom"]
    for k in range(210, len(cal)):
        d = cal[k]
        if d not in qqq.index:
            continue
        i = qqq.index.get_loc(d)
        if i < 1:
            continue
        long = (c.iloc[i] > s50.iloc[i] and c.iloc[i] > s200.iloc[i] and m.iloc[i] > 0)
        dr.append((c.iloc[i] / c.iloc[i - 1] - 1) if long else 0.0)
    return dr


def _sector_rot(data, cal):
    dr = []; holdings = []
    for k in range(210, len(cal)):
        d = cal[k]
        if k % 5 == 0:
            ranked = []
            for t, df in data.items():
                if d in df.index:
                    i = df.index.get_loc(d)
                    if i > 200:
                        c = df["Close"].values; s50 = df["sma50"].values
                        s200 = df["sma200"].values; m = df["mom"].values
                        if c[i] > s50[i] and c[i] > s200[i] and m[i] == m[i]:
                            ranked.append((m[i], t))
            ranked.sort(reverse=True)
            holdings = [t for _, t in ranked[:TOP_K]]
        if not holdings:
            dr.append(0.0); continue
        rs = []
        for t in holdings:
            df = data[t]
            if d in df.index:
                i = df.index.get_loc(d)
                if i > 0:
                    rs.append(df["Close"].values[i] / df["Close"].values[i - 1] - 1)
        dr.append(float(np.mean(rs)) if rs else 0.0)
    return dr


def _sweep(name, base_dr):
    print(f"\n  {name}  (base Sortino {_sortino(base_dr):+.2f})")
    print(f"    {'lev':>5}  {'total':>9}  {'CAGR':>7}  {'maxDD':>7}  {'Sortino':>8}  {'Calmar':>7}")
    best_s = (-1e9, None); best_c = (-1e9, None)
    for L in LEVS:
        eq, curve, dr = _lever(base_dr, L)
        so = _sortino(dr); cagr = _cagr(eq, len(dr)); mdd = _maxdd(curve)
        calmar = cagr / abs(mdd) if mdd != 0 else 0.0   # CAGR / |maxDD|
        if so > best_s[0]:
            best_s = (so, L)
        if calmar > best_c[0]:
            best_c = (calmar, L)
        print(f"    {L:5.1f}  {(eq-1)*100:+8.0f}%  {cagr:+6.1f}%  "
              f"{mdd:6.0f}%  {so:+7.2f}  {calmar:6.2f}")
    print(f"    -> Sortino peaks at {best_s[1]:.1f}x   ·   Calmar peaks at {best_c[1]:.1f}x")
    return best_s[1], best_c[1]


def run():
    print(f"Loading {len(SECTORS)} sector ETFs, {YEARS}y...")
    data = {}
    for t in SECTORS:
        d = _load(t)
        if d is not None:
            data[t] = d
    if "QQQ" not in data or "SPY" not in data:
        print("  FATAL: QQQ/SPY failed"); return
    cal = data["SPY"].index
    print(f"  loaded {len(data)}")

    print("\n" + "=" * 72)
    print(" LEVERAGE SWEEP — trend-filtered strategies, synthetic daily-rebal + drag")
    print("=" * 72)
    p1s, p1c = _sweep("timed_QQQ", _timed_qqq(data["QQQ"], cal))
    rot = {t: data[t] for t in SECTORS if t in data}
    p2s, p2c = _sweep("sector_rot", _sector_rot(rot, cal))

    print("\n" + "=" * 72)
    print(f"  Sortino-optimal:  timed_QQQ {p1s:.1f}x   ·   sector_rot {p2s:.1f}x")
    print(f"  Calmar-optimal :  timed_QQQ {p1c:.1f}x   ·   sector_rot {p2c:.1f}x")
    print("  (Calmar = CAGR/|maxDD| — the reviewer's survival metric. If it peaks")
    print("  LOWER than Sortino, drawdown says use even less leverage.)")
    print("  NOTE: synthetic leverage is OPTIMISTIC vs real leveraged ETFs (the")
    print("  real-vs-synth 3x gap in aggregation_test was ~half). Treat the PEAK as")
    print("  an upper bound; the real-vehicle optimum is likely a notch lower.")


if __name__ == "__main__":
    run()
