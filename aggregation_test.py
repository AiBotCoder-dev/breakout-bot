"""
aggregation_test.py — Answers the reviewer's Q1: is the leveraged-rotation edge
SELECTION ALPHA, or just 3x BETA on a bull market?

The leveraged_shadow returns +2513% over 8y. But that number confounds two
things: (1) picking the strongest SECTOR each week (selection), and (2) applying
3x leverage to a rising market (beta). This test decomposes them by running the
IDENTICAL weekly top-2 momentum rotation across matched vehicle sets:

  1x_rotation      rotation on the 1x sector underlyings   -> SELECTION alpha, unlevered
  synth3x_rotation the 1x rotation's daily returns × 3     -> PURE leverage on the selection
  3x_rotation      rotation on the real 3x sector ETFs     -> selection + real-vehicle leverage
  QQQ / SPY hold   buy & hold                              -> the BETA baseline
  timed_QQQ        hold QQQ only while QQQ itself is in an uptrend, else cash
                                                           -> TIMING alpha at the index level

DECISION RULE (reviewer's):
  * If 1x_rotation's Sortino/CAGR BEATS QQQ buy-hold  -> selection alpha is REAL;
    leverage merely amplifies a genuine edge (the shadow is legitimate).
  * If 1x_rotation ~= QQQ buy-hold                    -> the +2513% is purely
    leverage×beta with NO selection alpha, and the right move is the reviewer's
    regime-switching architecture (hold MTUM/QQQ, deploy 3x only on regime).
  * timed_QQQ vs QQQ isolates whether TIMING (as opposed to selection) adds
    anything on the aggregate.

Matched 3x -> 1x underlyings (same sector bets, no leverage):
  TQQQ->QQQ  SOXL->SOXX  SPXL->SPY  FNGU->FNGS  TECL->XLK  FAS->XLF
  TNA->IWM   LABU->XBI   CURE->XLV  UDOW->DIA   RETL->XRT  DPST->KRE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

YEARS = 8
TOP_K = 2
MOM = 63

# matched pairs: (3x ETF, its 1x sector underlying)
PAIRS = [("TQQQ", "QQQ"), ("SOXL", "SOXX"), ("SPXL", "SPY"), ("FNGU", "FNGS"),
         ("TECL", "XLK"), ("FAS", "XLF"), ("TNA", "IWM"), ("LABU", "XBI"),
         ("CURE", "XLV"), ("UDOW", "DIA"), ("RETL", "XRT"), ("DPST", "KRE")]
ETF3X = [p[0] for p in PAIRS]
ETF1X = [p[1] for p in PAIRS]


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


def _rotation_returns(universe_data, cal):
    """Daily return series of the weekly top-K uptrend momentum rotation over the
    shared calendar `cal`. Returns (daily_returns list, in_cash_fraction)."""
    dr = []; holdings = []; in_cash = 0; days = 0
    for k in range(210, len(cal)):
        d = cal[k]
        if k % 5 == 0:                                   # weekly rebalance
            ranked = []
            for t, df in universe_data.items():
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
            r = 0.0; in_cash += 1
        else:
            rs = []
            for t in holdings:
                df = universe_data[t]
                if d in df.index:
                    i = df.index.get_loc(d)
                    if i > 0:
                        rs.append(df["Close"].values[i] / df["Close"].values[i - 1] - 1)
            r = float(np.mean(rs)) if rs else 0.0
        dr.append(r); days += 1
    return dr, (in_cash / days if days else 0.0)


def _timed_qqq_returns(qqq, cal):
    """Hold QQQ only while it is above its own 50 & 200 SMA and mom>0, else cash."""
    dr = []; in_cash = 0; days = 0
    c = qqq["Close"]; s50 = qqq["sma50"]; s200 = qqq["sma200"]; m = qqq["mom"]
    for k in range(210, len(cal)):
        d = cal[k]
        if d not in qqq.index:
            continue
        i = qqq.index.get_loc(d)
        if i < 1:
            continue
        long = (c.iloc[i] > s50.iloc[i] and c.iloc[i] > s200.iloc[i] and m.iloc[i] > 0)
        r = (c.iloc[i] / c.iloc[i - 1] - 1) if long else 0.0
        if not long:
            in_cash += 1
        dr.append(r); days += 1
    return dr, (in_cash / days if days else 0.0)


def _curve(dr):
    eq = 1.0; c = []
    for r in dr:
        eq *= (1 + r); c.append(eq)
    return eq, c


def _report_line(name, dr, cash_frac=None):
    eq, curve = _curve(dr)
    tot = (eq - 1) * 100
    line = (f"  {name:20s} total {tot:+8.0f}%  CAGR {_cagr(eq, len(dr)):+6.1f}%  "
            f"maxDD {_maxdd(curve):6.0f}%  Sortino {_sortino(dr):+.2f}")
    if cash_frac is not None:
        line += f"  in-cash {100*cash_frac:.0f}%"
    print(line)
    return eq, curve


def run():
    print(f"Loading {len(ETF3X)} 3x + {len(ETF1X)} 1x + SPY/QQQ, {YEARS}y...")
    need = sorted(set(ETF3X) | set(ETF1X) | {"SPY", "QQQ"})
    data = {}
    for t in need:
        d = _load(t)
        if d is not None:
            data[t] = d
    spy = data.get("SPY"); qqq = data.get("QQQ")
    if spy is None or qqq is None:
        print("  FATAL: SPY/QQQ failed to load"); return
    cal = spy.index

    d1 = {t: data[t] for t in ETF1X if t in data}
    d3 = {t: data[t] for t in ETF3X if t in data}
    print(f"  1x universe: {sorted(d1)} ({len(d1)})")
    print(f"  3x universe: {sorted(d3)} ({len(d3)})\n")

    dr1, cash1 = _rotation_returns(d1, cal)
    dr3, cash3 = _rotation_returns(d3, cal)
    drt, casht = _timed_qqq_returns(qqq, cal)
    synth = [3 * r for r in dr1]                          # pure 3x leverage on 1x rotation

    # align QQQ/SPY buy-hold to the same [210:] window
    def _bh(df):
        c = df["Close"]; sub = c[c.index >= cal[210]]
        return sub.pct_change().dropna().tolist()
    qqq_bh = _bh(qqq); spy_bh = _bh(spy)

    print("=" * 92)
    print(" AGGREGATION / SELECTION-vs-LEVERAGE DECOMPOSITION  (weekly top-2 momentum rotation)")
    print("=" * 92)
    _report_line("1x_rotation", dr1, cash1)
    _report_line("QQQ buy&hold", qqq_bh)
    _report_line("SPY buy&hold", spy_bh)
    print("  " + "-" * 88)
    _report_line("synth3x_rotation", synth)
    _report_line("3x_rotation", dr3, cash3)
    print("  " + "-" * 88)
    _report_line("timed_QQQ", drt, casht)

    # ── the verdicts ──────────────────────────────────────────────────────────
    e1, _ = _curve(dr1); eq, _ = _curve(qqq_bh)
    s1 = _sortino(dr1); sq = _sortino(qqq_bh)
    et, _ = _curve(drt)
    print("\n" + "=" * 92)
    print(" VERDICT")
    print("=" * 92)
    print(f"  SELECTION alpha  : 1x_rotation Sortino {s1:+.2f} vs QQQ buy-hold {sq:+.2f}  "
          f"(CAGR {_cagr(e1,len(dr1)):+.1f}% vs {_cagr(eq,len(qqq_bh)):+.1f}%)")
    if s1 > sq * 1.10 and _cagr(e1, len(dr1)) > _cagr(eq, len(qqq_bh)):
        print("    -> SELECTION ALPHA IS REAL. Rotation beats the index unlevered; "
              "leverage amplifies a genuine edge. The shadow is legitimate.")
    elif s1 >= sq * 0.95:
        print("    -> MARGINAL. Selection roughly matches the index risk-adjusted; "
              "most of the 3x number is leverage. Lean regime-switching.")
    else:
        print("    -> NO SELECTION ALPHA. Unlevered rotation does NOT beat QQQ; the "
              "+2513% is leverage×beta. Pivot to the regime-switching architecture.")
    st = _sortino(drt)
    print(f"  TIMING alpha     : timed_QQQ Sortino {st:+.2f} vs QQQ buy-hold {sq:+.2f}  "
          f"(CAGR {_cagr(et,len(drt)):+.1f}% vs {_cagr(eq,len(qqq_bh)):+.1f}%)")
    print("    -> if timed_QQQ does not beat QQQ risk-adjusted, index-level TIMING "
          "adds nothing; any edge is SELECTION, not timing.")
    print("\n  READ: the honest question isn't the +2513% headline — it's whether "
          "1x_rotation\n  clears QQQ. If it does, the edge survives; if not, it was "
          "leverage all along.")


if __name__ == "__main__":
    run()
