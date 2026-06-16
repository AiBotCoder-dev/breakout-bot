"""
walk_forward.py - out-of-sample (walk-forward) validation of the MOMENTUM entry.

WHY THIS EXISTS
---------------
full_bot_backtest.py tests its thresholds (e.g. `mom6 > 0.10`) on the SAME data they
were chosen on - the textbook way to fool yourself. This harness does it honestly:

  1. Slide a ~2-year IN-SAMPLE window across history.
  2. On each IS window, pick the momentum threshold that maximised average forward
     return (the optimiser only ever sees the past).
  3. Apply that threshold to the next ~6-month OUT-OF-SAMPLE window - data the
     optimiser never touched - and record those trades.
  4. Roll forward 6 months and repeat.

The pooled OOS trades are the honest estimate of live performance. Two things to look
for: (a) OOS results far below the IS-selected results = overfitting; (b) the "best"
threshold jumping around fold-to-fold = the parameter is noise, not an edge. We also
compare against a FIXED 0.10 threshold on the same OOS windows - if adaptive tuning
doesn't beat a constant, the tuning is adding nothing.

SCOPE / LIMITATIONS (read before trusting it)
---------------------------------------------
- MOMENTUM only. BOTTOM_FISHER produced ~13 trades total - far too few to fold.
- Optimises ONE knob (the 6-month momentum threshold) over a fixed trend filter. Real
  overfitting risk grows with more knobs; this is a FLOOR on the problem, not a ceiling.
- Signals are scanned non-overlapping over full history per threshold, then bucketed by
  entry date into IS/OOS. Boundary effects at window edges are minor.
- Option NET reuses the calibrated friction/IV model from full_bot_backtest.py, which
  (see BACKTEST_CHANGES.md section 6) is dominated by an unknowable IV - so judge this
  test on the DIRECTIONAL numbers first.

RUN
---
    python walk_forward.py
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta

import numpy as np

# Reuse the exact pricing + data layer from the main backtest so the two stay in sync.
from full_bot_backtest import (
    STOCKS, HOLD, ENTRY_IV_PREMIUM, IV_CRUSH,
    bs_call, buy_fill, sell_fill, _load_all,
)

# --- knobs -----------------------------------------------------------------
MOM_GRID   = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]  # 6-month momentum thresholds to try
BASELINE   = 0.10        # the hard-coded threshold we're trying to validate / beat
IS_DAYS    = 730         # in-sample window (calendar days, ~2y)
OOS_DAYS   = 182         # out-of-sample window (~6mo); also the roll step
MIN_TRADES = 8           # ignore an IS threshold with fewer trades than this


def scan_momentum(df, ticker, mom_min):
    """Non-overlapping MOMENTUM signals for one name at one threshold.

    Returns list of (entry_date, ticker, fwd_ret%, opt_net%).
    """
    C = df["Close"].values
    s50 = df["sma50"].values; s200 = df["sma200"].values
    mom6 = df["mom6"].values; rv = df["rv20"].values
    idx = df.index
    n = len(C); i = 210
    out = []
    while i < n - HOLD:
        if (C[i] > s50[i] > s200[i] and s50[i] > s50[i-10]
                and s200[i] > s200[i-20] and mom6[i] > mom_min):
            S0 = C[i]; S1 = C[i+HOLD]
            fwd = (S1/S0 - 1) * 100
            iv0 = max(0.15, min(1.5, rv[i]*ENTRY_IV_PREMIUM))
            iv1 = max(0.05, iv0 * (1.0 - IV_CRUSH))
            K = S0 * 1.03
            mid_e = bs_call(S0, K, 21/365, iv0); mid_x = bs_call(S1, K, 11/365, iv1)
            cost = buy_fill(mid_e); proceeds = sell_fill(mid_x)
            opt_net = (proceeds/cost - 1) * 100 if cost > 0.01 else 0.0
            out.append((idx[i].date(), ticker, fwd, opt_net))
            i += HOLD
        else:
            i += 1
    return out


def make_folds(dmin, dmax):
    """Rolling (IS, OOS) calendar windows over [dmin, dmax)."""
    folds = []
    is_start = dmin
    while True:
        is_end = is_start + timedelta(days=IS_DAYS)
        oos_start = is_end
        oos_end = oos_start + timedelta(days=OOS_DAYS)
        if oos_start >= dmax:
            break
        folds.append((is_start, is_end, oos_start, min(oos_end, dmax)))
        is_start = is_start + timedelta(days=OOS_DAYS)   # roll forward by the OOS length
    return folds


def _between(sigs, lo, hi):
    return [s for s in sigs if lo <= s[0] < hi]


def run():
    print("WALK-FORWARD validation of the MOMENTUM entry (out-of-sample is the truth)\n")
    dfs = _load_all(10)   # pull 10y so there's enough post-warmup history for several folds

    # cache the non-overlapping scan for every threshold, pooled across names
    scans = {p: [] for p in MOM_GRID}
    for t, df in dfs.items():
        for p in MOM_GRID:
            scans[p].extend(scan_momentum(df, t, p))

    alld = [s[0] for p in MOM_GRID for s in scans[p]]
    if not alld:
        print("no signals - aborting"); return
    dmin, dmax = min(alld), max(alld)
    folds = make_folds(dmin, dmax)

    oos_dir, oos_opt = [], []      # walk-forward (adaptive) OOS trades
    base_dir, base_opt = [], []    # fixed-0.10 baseline on the SAME OOS windows
    is_sel_dir = []                # the IS-selected avg returns (optimistic)
    chosen = []

    print(f"  {'fold':>4} | {'OOS window':>23} | {'best p':>6} | "
          f"{'IS n':>5} {'IS avg':>7} | {'OOS n':>5} {'OOS avg':>8} {'OOS win%':>9}")
    print("  " + "-" * 82)
    for k, (is0, is1, oo0, oo1) in enumerate(folds, 1):
        # pick the threshold on IS by best average forward return (>= MIN_TRADES)
        best = None
        for p in MOM_GRID:
            iss = _between(scans[p], is0, is1)
            if len(iss) < MIN_TRADES:
                continue
            obj = float(np.mean([s[2] for s in iss]))
            if best is None or obj > best[1]:
                best = (p, obj, len(iss))
        if best is None:
            continue
        p, is_obj, is_n = best
        chosen.append(p); is_sel_dir.append(is_obj)

        oos = _between(scans[p], oo0, oo1)
        for s in oos:
            oos_dir.append(s[2]); oos_opt.append(s[3])
        base = _between(scans[BASELINE], oo0, oo1)
        for s in base:
            base_dir.append(s[2]); base_opt.append(s[3])

        o = np.array([s[2] for s in oos]) if oos else np.array([])
        owin = 100*(o > 0).mean() if len(o) else float('nan')
        oavg = o.mean() if len(o) else float('nan')
        print(f"  {k:>4} | {str(oo0)} .. {str(oo1)} | {p:>6.2f} | "
              f"{is_n:>5} {is_obj:>+6.1f}% | {len(oos):>5} {oavg:>+7.1f}% {owin:>8.0f}%")

    print("\n" + "=" * 70)
    print(f" WALK-FORWARD AGGREGATE - {len(folds)} folds, MOMENTUM only")
    print("=" * 70)
    od = np.array(oos_dir); oo = np.array(oos_opt)
    bd = np.array(base_dir); bo = np.array(base_opt)
    iss = np.array(is_sel_dir)

    print(f"  In-sample SELECTED avg fwd return (optimistic) : {iss.mean():+.1f}%")
    print( "  --- OUT-OF-SAMPLE (the honest number) ---")
    if len(od):
        print(f"  WF OOS  directional   n={len(od):<4} win {100*(od>0).mean():4.1f}%  "
              f"avg {od.mean():+.1f}%")
        print(f"  WF OOS  option NET    n={len(oo):<4} win {100*(oo>0).mean():4.1f}%  "
              f"avg {oo.mean():+.1f}%")
    print( "  --- FIXED 0.10 baseline on the SAME OOS windows ---")
    if len(bd):
        print(f"  Fixed   directional   n={len(bd):<4} win {100*(bd>0).mean():4.1f}%  "
              f"avg {bd.mean():+.1f}%")
        print(f"  Fixed   option NET    n={len(bo):<4} win {100*(bo>0).mean():4.1f}%  "
              f"avg {bo.mean():+.1f}%")
    print()
    if len(od):
        tax = iss.mean() - od.mean()
        print(f"  >>> OVERFITTING TAX (IS-selected minus OOS) : {tax:+.1f}% avg return <<<")
        if len(bd):
            edge = od.mean() - bd.mean()
            print(f"  >>> ADAPTIVE EDGE vs fixed 0.10 (OOS)       : {edge:+.1f}% avg return <<<")
    print(f"\n  Thresholds picked per fold: {dict(Counter(chosen))}")
    print("  (if this jumps around, the 'best' threshold is noise, not a stable edge)")


if __name__ == "__main__":
    run()
