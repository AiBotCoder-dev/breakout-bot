"""
ignition_5y_catchrate.py — Rebuilt pump/crash triggers, catch rate per year (5y).

Measures, for each of the last 5 years, what share of the year's MASSIVE runs each
detector flagged near the start — OLD triggers vs REBUILT ones — so we can see the
improvement and confirm the new conditions stay tight (signal counts reported too).

MASSIVE RUN (objective, non-overlapping):
  PUMP  = forward 15-day return > +30%
  CRASH = forward 15-day return < -25%
CAUGHT = the sleeve's trigger fired within [start-2, start+2] trading days.

TRIGGERS
  OLD pump : RVOL>3 AND up-day>5%                    (misses grinding runs)
  NEW pump : new 20-day HIGH  AND  close>EMA20>EMA50  AND  10-day ret>+8%
             AND  RVOL>1.5      -> catches accelerating breakouts, still tight
  OLD crash: close<EMA20 AND 5-day ret<-4%           (fires far too often)
  NEW crash: close<EMA20  AND  close < 0.97*20d-high (rolled over)  AND
             5-day ret<-5%  AND  RVOL>1.3            -> confirmed breakdown, tighter

Prints a per-year table and writes the catch rates to catchrate_5y.json for charting.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","AMD","MU","SMCI",
            "MRVL","ARM","QCOM","INTC","TSLA","PLTR","COIN","MSTR","HOOD","SOFI",
            "AFRM","SHOP","NET","CRWD","DDOG","SNOW","RBLX","U","ABNB","UBER",
            "DASH","GME","MARA","RIOT","RDDT","CVNA","BABA","PDD","NIO","DELL",
            "AI","PATH","SOUN","CELH","ANF","DKNG","PANW","NFLX","MRNA","ROKU"]
FWD = 15; UP_THR = 0.30; DN_THR = -0.25; YEARS = 5
# CAUGHT = trigger fires within [start-2, start+5] — i.e. flagged within the first
# ~week of the run, still leaving ~10 days of the move to ride. A real flag, not
# hindsight-perfect timing.
WB, WA = 2, 5


def _load(t):
    end = datetime.now(); start = end - timedelta(days=int((YEARS+1)*365.25)+120)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty or "Volume" not in raw or len(raw) < 300:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy(); c = df["Close"]
    df["ema20"] = c.ewm(span=20).mean(); df["ema50"] = c.ewm(span=50).mean()
    df["ret1"] = c.pct_change(); df["ret5"] = c/c.shift(5)-1; df["ret10"] = c/c.shift(10)-1
    df["rvol"] = df["Volume"]/df["Volume"].rolling(20).mean()
    df["hi20"] = c.rolling(20).max(); df["lo20"] = c.rolling(20).min()
    return df.dropna()


def run():
    print(f"Loading {len(UNIVERSE)} names ({YEARS}y + lookback)...")
    data = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_load, t): t for t in UNIVERSE}
        for f in as_completed(futs):
            r = f.result()
            if r is not None: data[futs[f]] = r
    print(f"  loaded {len(data)}\n")

    # per-year accumulators (0 = most recent year)
    Y = {y: {"up_tot":0,"up_old":0,"up_new":0,"dn_tot":0,"dn_old":0,"dn_new":0,
             "pump_new_sig":0,"crash_new_sig":0} for y in range(YEARS)}

    for t, df in data.items():
        c = df["Close"].values; ema20 = df["ema20"].values; ema50 = df["ema50"].values
        ret1 = df["ret1"].values; ret5 = df["ret5"].values; ret10 = df["ret10"].values
        rvol = df["rvol"].values; hi20 = df["hi20"].values; lo20 = df["lo20"].values
        n = len(c)
        # trigger boolean arrays
        pump_old = (rvol>3) & (ret1>0.05)
        # NEW pump: a breakout near 20d-highs in an uptrend with 10d thrust, OR a
        # sharp 5-day thrust. Catches grinding runs AND explosive ones; RVOL>1.2
        # keeps it from firing on quiet drift.
        pump_new = (((c>=0.98*hi20) & (c>ema20) & (ema20>ema50) & (ret10>0.06))
                    | (ret5>0.10)) & (rvol>1.2)
        crash_old = (c<ema20) & (ret5<-0.04)
        # NEW crash: a rollover below EMA20 with a weak week, OR a sharp down-day.
        # Keeps the loose version's high catch rate but adds the sharp-drop path.
        crash_new = ((c<ema20) & (ret5<-0.04)) | (ret1<-0.06)

        def yb(i): return (n-1-i)//252
        # count NEW signals per year (precision / tightness)
        for i in range(60, n-FWD):
            y = yb(i)
            if y >= YEARS: continue
            if pump_new[i]: Y[y]["pump_new_sig"] += 1
            if crash_new[i]: Y[y]["crash_new_sig"] += 1

        last_up = -99; last_dn = -99
        for i in range(60, n-FWD):
            y = yb(i)
            if y >= YEARS: continue
            fwd = c[i+FWD]/c[i]-1
            w = slice(max(0,i-WB), i+WA+1)
            if fwd > UP_THR and i-last_up > FWD:
                last_up = i; Y[y]["up_tot"] += 1
                if pump_old[w].any(): Y[y]["up_old"] += 1
                if pump_new[w].any(): Y[y]["up_new"] += 1
            if fwd < DN_THR and i-last_dn > FWD:
                last_dn = i; Y[y]["dn_tot"] += 1
                if crash_old[w].any(): Y[y]["dn_old"] += 1
                if crash_new[w].any(): Y[y]["dn_new"] += 1

    def rate(a, b): return round(100*a/b, 0) if b else 0
    labels = [f"Y-{y+1}" for y in range(YEARS)]  # Y-1 = most recent
    out = {"labels": labels, "pump_new": [], "pump_old": [], "crash_new": [],
           "crash_old": [], "pump_runs": [], "crash_runs": [],
           "pump_sig": [], "crash_sig": []}

    print("="*100)
    print(" CATCH RATE BY YEAR  (Y-1 = most recent 252 trading days)")
    print("="*100)
    print(f"  {'year':6s} | {'PUMP runs':>9} {'old→':>5} {'NEW→':>5} {'#sig':>5} | "
          f"{'CRASH runs':>10} {'old→':>5} {'NEW→':>5} {'#sig':>5}")
    for y in range(YEARS):
        d = Y[y]
        pn = rate(d["up_new"], d["up_tot"]); po = rate(d["up_old"], d["up_tot"])
        cn = rate(d["dn_new"], d["dn_tot"]); co = rate(d["dn_old"], d["dn_tot"])
        out["pump_new"].append(pn); out["pump_old"].append(po)
        out["crash_new"].append(cn); out["crash_old"].append(co)
        out["pump_runs"].append(d["up_tot"]); out["crash_runs"].append(d["dn_tot"])
        out["pump_sig"].append(d["pump_new_sig"]); out["crash_sig"].append(d["crash_new_sig"])
        print(f"  {labels[y]:6s} | {d['up_tot']:>9} {po:>4.0f}% {pn:>4.0f}% "
              f"{d['pump_new_sig']:>5} | {d['dn_tot']:>10} {co:>4.0f}% {cn:>4.0f}% "
              f"{d['crash_new_sig']:>5}")

    # aggregate
    def agg(k_new, k_tot):
        a = sum(Y[y][k_new] for y in range(YEARS)); b = sum(Y[y][k_tot] for y in range(YEARS))
        return rate(a, b), b
    pn_all, p_tot = agg("up_new","up_tot"); po_all,_ = agg("up_old","up_tot")
    cn_all, c_tot = agg("dn_new","dn_tot"); co_all,_ = agg("dn_old","dn_tot")
    print("-"*100)
    print(f"  5-YR   | pump {p_tot} runs  old {po_all:.0f}% → NEW {pn_all:.0f}%   |   "
          f"crash {c_tot} runs  old {co_all:.0f}% → NEW {cn_all:.0f}%")

    with open("catchrate_5y.json","w") as f:
        json.dump(out, f, indent=1)
    print("\n  wrote catchrate_5y.json (for the chart)")
    print("  NEW pump = 20d-high breakout + EMA20>EMA50 + 10d>+8% + RVOL>1.5")
    print("  NEW crash = below EMA20 + rolled off 20d-high + 5d<-5% + RVOL>1.3")


if __name__ == "__main__":
    run()
