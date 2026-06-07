"""
callout_proxy_backtest.py — Does chasing social-hype names actually work?

We can't backtest historical social callouts (no dataset exists). But we CAN
backtest the closest proxy: the kind of move that GENERATES callouts — a big
single-day pop on heavy volume in a retail-favorite name. If buying after that
pop has edge, following callouts might too; if it's negative, callouts are
chasing tops and should be tracked-not-trusted.

Proxy trigger: +8%+ up day on >= 2x average volume (the classic 'everyone's
posting about it' bar). Measure forward 1/3/5/10-day returns + win rate vs the
universe baseline. Direction = long (most callouts are bullish calls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Retail-favorite / heavily-discussed universe (where callouts cluster)
UNIVERSE = [
    "GME","AMC","TSLA","NVDA","AMD","PLTR","SOFI","NIO","RIVN","LCID","MARA","RIOT",
    "COIN","MSTR","HOOD","RBLX","DKNG","SNAP","PINS","AFRM","UPST","BBBY","BB",
    "NOK","SNDL","CLOV","WISH","SPCE","TLRY","ROKU","SQ","PYPL","SHOP","ABNB",
    "UBER","DASH","AI","SOUN","BBAI","IONQ","RGTI","QUBT","SMCI","ARM","INTC",
    "F","NKLA","CHPT","PLUG","FCEL","GEVO","MULN","FFIE","DJT","RDDT","CVNA",
]
FWD = [1, 3, 5, 10]
YEARS = 4


def _load(t):
    try:
        end = datetime.now(); start = end - timedelta(days=int(YEARS*365.25)+40)
        raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 80:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        df["ret1"] = df["Close"].pct_change()
        df["vol_avg20"] = df["Volume"].rolling(20).mean()
        return df.dropna()
    except Exception:
        return None


def run():
    print(f"Loading {len(UNIVERSE)} retail-favorite names, {YEARS}y...")
    data = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_load, t): t for t in UNIVERSE}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None:
                data[futs[fut]] = df
    print(f"  loaded {len(data)}\n")

    base = {w: [] for w in FWD}
    trig = {w: [] for w in FWD}
    n_trig = 0
    for df in data.values():
        cl = df["Close"].values; n = len(cl)
        pop = ((df["ret1"] >= 0.08) & (df["Volume"] >= 2 * df["vol_avg20"])).values
        for i in range(n):
            for w in FWD:
                if i+w < n and cl[i] > 0:
                    r = (cl[i+w]/cl[i]-1)*100
                    base[w].append(r)
                    if pop[i]:
                        trig[w].append(r)
        n_trig += int(pop.sum())

    print("BASELINE (random entry, retail universe):")
    for w in FWD:
        a = np.array(base[w]); print(f"  {w}d: mean {a.mean():+.2f}%  win {100*(a>0).mean():.1f}%")

    print(f"\n{'='*78}\n CHASE-THE-HYPE  (buy after +8% day on 2x volume — the callout trigger)")
    print(f" n = {n_trig} triggers\n{'='*78}")
    for w in FWD:
        a = np.array(trig[w]); b = np.array(base[w])
        if len(a) == 0:
            continue
        print(f"  {w}d: mean {a.mean():+.2f}%  median {np.median(a):+.2f}%  "
              f"win {100*(a>0).mean():.1f}%  edge {a.mean()-b.mean():+.2f}%  "
              f"p10 {np.percentile(a,10):+.1f}%  p90 {np.percentile(a,90):+.1f}%")

    print("\nINTERPRETATION:")
    e5 = np.array(trig[5]).mean() - np.array(base[5]).mean() if trig[5] else 0
    if e5 > 0.5:
        print("  Positive edge — chasing hype has short-term momentum. Callouts MAY help.")
    elif e5 < -0.5:
        print("  NEGATIVE edge — chasing hype buys tops. Callouts likely LOW signal;")
        print("  track outcomes, don't trust blindly. The win-rate gate is justified.")
    else:
        print("  ~Zero edge — chasing hype is a coin flip. Callouts need outcome-")
        print("  tracking + a win-rate gate before being actionable.")


if __name__ == "__main__":
    run()
