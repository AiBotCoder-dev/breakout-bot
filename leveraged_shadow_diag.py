"""
leveraged_shadow_diag.py — WHY is the leveraged shadow lagging the market?

The shadow holds HEU/HFU (2x energy/financials) and is flat while SPY/QQQ rose.
Two hypotheses, and this tests which:
  (A) BENIGN: a normal momentum drawdown at a sector-rotation turn — the backtest
      (Sortino 1.17) already includes these stretches; 10 days is noise.
  (B) HARMFUL: the 63-day momentum selection systematically buys yesterday's leaders
      right as they roll over — i.e. it picks short-term LAGGARDS.

Part 1 — SNAPSHOT: for each CAD 2x ETF, its 63d momentum (what the strategy ranks
on) vs its recent 10/21-day return (what's happening now). Shows if it picked high-
momentum names that are now lagging the ones it skipped.

Part 2 — STRUCTURAL TEST: across all history, does the top-2-by-63d-momentum pick
actually BEAT the universe average over the NEXT 10 days, or underperform it? That
distinguishes a real selection edge (A) from buying-the-top (B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

CAD = ["HQU.TO", "HSU.TO", "HXU.TO", "HEU.TO", "HFU.TO", "HGU.TO"]
NAMES = {"HQU.TO": "2x Nasdaq", "HSU.TO": "2x S&P500", "HXU.TO": "2x TSX60",
         "HEU.TO": "2x Energy", "HFU.TO": "2x Financials", "HGU.TO": "2x GoldMiners"}
MOM = 63; FWD = 10


def _load(t, days=800):
    end = datetime.now(); start = end - timedelta(days=days)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy(); c = df["Close"]
    df["mom63"] = c / c.shift(MOM) - 1
    df["ret10"] = c / c.shift(10) - 1
    df["ret21"] = c / c.shift(21) - 1
    return df


def run():
    data = {t: _load(t) for t in CAD + ["QQQ", "SPY"]}
    data = {k: v for k, v in data.items() if v is not None}

    print("=" * 84)
    print(" PART 1 — SNAPSHOT: what it ranks on (63d mom) vs what's happening now")
    print("=" * 84)
    print(f"  {'ETF':13s} {'sector':14s} {'63d mom (rank on)':>17} {'last 21d':>10} {'last 10d':>10}")
    snap = []
    for t in CAD:
        d = data.get(t)
        if d is None:
            continue
        m = float(d["mom63"].iloc[-1]); r21 = float(d["ret21"].iloc[-1]); r10 = float(d["ret10"].iloc[-1])
        snap.append((t, m, r21, r10))
    snap.sort(key=lambda x: -x[1])          # rank by momentum (what the strategy does)
    picked = {s[0] for s in snap[:2]}
    for t, m, r21, r10 in snap:
        tag = "  <- PICKED (top-2 mom)" if t in picked else ""
        print(f"  {t:13s} {NAMES[t]:14s} {m*100:>+15.0f}% {r21*100:>+9.0f}% {r10*100:>+9.0f}%{tag}")
    qm = float(data['QQQ']['ret10'].iloc[-1]) * 100 if 'QQQ' in data else 0
    print(f"\n  For reference: QQQ last 10d {qm:+.0f}%")
    # is the picked set lagging the unpicked recently?
    pick10 = np.mean([r10 for t, m, r21, r10 in snap if t in picked]) * 100
    skip10 = np.mean([r10 for t, m, r21, r10 in snap if t not in picked]) * 100
    print(f"  PICKED last-10d avg {pick10:+.1f}%   vs   SKIPPED avg {skip10:+.1f}%"
          + ("   <-- picked ARE lagging the ones it skipped" if pick10 < skip10 else ""))

    print("\n" + "=" * 84)
    print(" PART 2 — STRUCTURAL TEST: does top-2-momentum BEAT the field over next 10d?")
    print("=" * 84)
    # align on a common calendar (intersection of CAD ETFs)
    idx = None
    for t in CAD:
        if t in data:
            idx = data[t].index if idx is None else idx.intersection(data[t].index)
    idx = idx[MOM + 5:]
    picks_fwd = []; field_fwd = []; qqq_fwd = []
    qc = data['QQQ']['Close'].reindex(idx, method='ffill') if 'QQQ' in data else None
    for k in range(0, len(idx) - FWD, 5):    # weekly, matches rebalance
        d = idx[k]
        moms = []
        for t in CAD:
            if t in data and d in data[t].index:
                m = data[t]["mom63"].loc[d]
                if m == m:
                    moms.append((m, t))
        if len(moms) < 4:
            continue
        moms.sort(reverse=True)
        top2 = [t for _, t in moms[:2]]
        # forward 10d return of picks vs the whole field
        fwd_all = []
        for _, t in moms:
            c = data[t]["Close"]
            if d in c.index:
                i = c.index.get_loc(d)
                if i + FWD < len(c):
                    fwd_all.append((t, c.iloc[i + FWD] / c.iloc[i] - 1))
        if not fwd_all:
            continue
        pf = np.mean([r for t, r in fwd_all if t in top2])
        ff = np.mean([r for t, r in fwd_all])
        picks_fwd.append(pf); field_fwd.append(ff)
        if qc is not None and d in qc.index:
            i = qc.index.get_loc(d)
            if i + FWD < len(qc):
                qqq_fwd.append(qc.iloc[i + FWD] / qc.iloc[i] - 1)

    pf = np.array(picks_fwd); ff = np.array(field_fwd)
    print(f"  n={len(pf)} weekly rebalances")
    print(f"  top-2 momentum picks  avg fwd-10d {pf.mean()*100:+.2f}%   win-vs-field {100*(pf>ff).mean():.0f}%")
    print(f"  universe average       avg fwd-10d {ff.mean()*100:+.2f}%")
    print(f"  SELECTION EDGE (picks - field): {(pf.mean()-ff.mean())*100:+.2f}%/10d")
    if len(qqq_fwd):
        print(f"  QQQ buy-hold          avg fwd-10d {np.mean(qqq_fwd)*100:+.2f}%")
    print("\n  READ: if SELECTION EDGE > 0, momentum picks beat the field -> the current")
    print("  lag is a NORMAL drawdown (hypothesis A). If <= 0, the 63d-momentum rank is")
    print("  buying laggards -> a real selection problem (hypothesis B) worth fixing.")
    print("  BLIND SPOTS: 6-ETF commodity-heavy universe; 63d lookback is one choice;")
    print("  and forward-10d is one horizon — a turning-point regime can differ from avg.")


if __name__ == "__main__":
    run()
