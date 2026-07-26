"""
broad_gate_test.py — Does a broad-index RISK-OFF gate beat the per-ETF filter?

The reviewer's late-cycle-divergence argument: the per-ETF 200-SMA filter is
backward-looking at each sector in ISOLATION. It misses the correlation spike —
defensive sectors that "look strong" (still above their own 200-SMA) until a
credit/geopolitical shock gaps everything down together. A broad-index gate is
the airbag the per-ETF filter can't be.

Three scenarios, IDENTICAL weekly top-2 momentum rotation on the CAD 2x universe:
  A  per-ETF filter only  — hold a name only if it's > its own 50 & 200 SMA
  B  broad gate only      — if SPY & XIU both > 200-SMA, hold top-2 by momentum
                            (no per-ETF filter); else ALL cash
  C  BOTH must agree       — per-ETF filter AND broad gate long; else cash

Broad gate = LONG only when SPY AND XIU are both above their 200-SMA (risk-off if
EITHER breaks — the conservative airbag).

Beyond CAGR/Sortino/Calmar, the deciding diagnostics:
  * DIVERGENCE COUNT — % of weeks A is invested while the broad gate is in cash
    (>10% = a real late-cycle risk window)
  * WORST-WEEK OVERLAP — were A's worst weeks ones the broad gate would've dodged?
  * ENTRY CORRELATION — avg pairwise corr of the 2 held ETFs at entry; when >0.7
    the per-ETF filter is really a concentrated single bet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

YEARS = 8
TOP_K = 2
MOM = 63
CAD_2X = ["HQU.TO", "HSU.TO", "HXU.TO", "HEU.TO", "HFU.TO", "HGU.TO"]
SPREAD_BPS = 8.0


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
    df["ret"] = c.pct_change()
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


def run():
    print(f"Loading CAD 2x + SPY + XIU, {YEARS}y...")
    data = {}
    for t in CAD_2X + ["SPY", "XIU.TO"]:
        d = _load(t)
        if d is not None:
            data[t] = d
    cad = {t: data[t] for t in CAD_2X if t in data}
    if not cad or "SPY" not in data or "XIU.TO" not in data:
        print("  FATAL: missing data"); return
    base = max(cad.values(), key=lambda x: len(x))
    cal = base.index

    # broad-gate boolean per calendar day (ffill the two indices onto CAD calendar)
    def _above200(df):
        s = (df["Close"] > df["sma200"])
        return s.reindex(cal, method="ffill").fillna(False)
    spy_ok = _above200(data["SPY"]); xiu_ok = _above200(data["XIU.TO"])
    broad_long = (spy_ok & xiu_ok)

    # per-scenario daily returns + bookkeeping
    dr = {"A": [], "B": [], "C": []}
    hold = {"A": [], "B": [], "C": []}
    diverge_weeks = 0; total_weeks = 0
    entry_corrs = []
    weekA_ret = []              # weekly return of A, with the week's broad state
    wk_accum = 0.0; wk_broad = True

    rets = {t: cad[t]["ret"] for t in cad}
    for k in range(210, len(cal)):
        d = cal[k]
        bl = bool(broad_long.iloc[k])
        if k % 5 == 0:                                       # weekly rebalance
            # rank uptrend-or-not by momentum
            ranked = []
            passing = []
            for t, df in cad.items():
                if d in df.index:
                    i = df.index.get_loc(d)
                    if i > 200:
                        c = df["Close"].values; s50 = df["sma50"].values
                        s200 = df["sma200"].values; m = df["mom"].values
                        if m[i] == m[i]:
                            ranked.append((m[i], t))
                            if c[i] > s50[i] and c[i] > s200[i]:
                                passing.append((m[i], t))
            ranked.sort(reverse=True); passing.sort(reverse=True)
            hold["A"] = [t for _, t in passing[:TOP_K]]
            hold["B"] = ([t for _, t in ranked[:TOP_K]] if bl else [])
            hold["C"] = ([t for _, t in passing[:TOP_K]] if bl else [])
            # divergence: A invested while broad gate says cash
            total_weeks += 1
            if hold["A"] and not bl:
                diverge_weeks += 1
            # entry correlation of A's two names (trailing 60d returns)
            if len(hold["A"]) == 2:
                a, b = hold["A"]
                ra = cad[a]["ret"].reindex(cal).iloc[max(0, k-60):k]
                rb = cad[b]["ret"].reindex(cal).iloc[max(0, k-60):k]
                cc = ra.corr(rb)
                if cc == cc:
                    entry_corrs.append(cc)
            # close out the prior week's A bucket
            if total_weeks > 1:
                weekA_ret.append((wk_accum, wk_broad))
            wk_accum = 0.0; wk_broad = bl

        for sc in ("A", "B", "C"):
            h = hold[sc]
            if not h:
                r = 0.0
            else:
                rs = []
                for t in h:
                    if d in cad[t].index:
                        i = cad[t].index.get_loc(d)
                        if i > 0:
                            rs.append(cad[t]["Close"].values[i] / cad[t]["Close"].values[i-1] - 1)
                r = float(np.mean(rs)) if rs else 0.0
            dr[sc].append(r)
        wk_accum = (1 + wk_accum) * (1 + dr["A"][-1]) - 1

    def _curve(x):
        eq = 1.0; c = []
        for r in x:
            eq *= (1 + r); c.append(eq)
        return eq, c

    print("\n" + "=" * 88)
    print(" THREE-SCENARIO GATE TEST  (CAD 2x, weekly top-2, 8y)")
    print("=" * 88)
    print(f"    {'scenario':32s} {'CAGR':>7} {'maxDD':>7} {'Sortino':>8} {'Calmar':>7}")
    for sc, label in [("A", "A per-ETF filter only"),
                      ("B", "B broad gate only"),
                      ("C", "C both must agree")]:
        eq, curve = _curve(dr[sc])
        cagr = _cagr(eq, len(dr[sc])); mdd = _maxdd(curve)
        calmar = cagr / abs(mdd) if mdd else 0.0
        print(f"    {label:32s} {cagr:+6.1f}% {mdd:6.0f}% {_sortino(dr[sc]):+7.2f} {calmar:6.2f}")

    print("\n" + "=" * 88)
    print(" DECIDING DIAGNOSTICS")
    print("=" * 88)
    print(f"  DIVERGENCE: A invested while broad gate in CASH — {diverge_weeks}/"
          f"{total_weeks} weeks = {100*diverge_weeks/max(total_weeks,1):.0f}%")
    print("    (>10% => a real late-cycle window the per-ETF filter can't see)")
    if entry_corrs:
        ec = np.array(entry_corrs)
        print(f"  ENTRY CORRELATION of A's 2 names: mean {ec.mean():.2f}  "
              f"share>0.70: {100*(ec>0.7).mean():.0f}%")
        print("    (high => the 'diversified' pair is really one concentrated bet)")
    # worst-week overlap
    if weekA_ret:
        wk = sorted(weekA_ret, key=lambda x: x[0])[:5]
        n_dodge = sum(1 for r, b in wk if not b)
        print(f"  WORST 5 WEEKS for A: returns "
              f"{', '.join(f'{r*100:+.0f}%' for r,_ in wk)}")
        print(f"    of these, {n_dodge}/5 occurred while the broad gate was in CASH "
              f"(i.e. C/B would have dodged them)")

    print("\n  READ (reviewer's adoption rule): adopt C if its CAGR is within ~5pts")
    print("  of A, maxDD improves >5pts, and bull-market cash drag doesn't balloon.")


if __name__ == "__main__":
    run()
