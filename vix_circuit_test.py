"""
vix_circuit_test.py — Does a VIX CIRCUIT BREAKER improve the 200-SMA airbag?

The 200-SMA broad gate is a slow STRATEGIC posture — in a fast crash (Mar 2020) it
exits near the bottom. The reviewer's fix is not another regime switch (SPY<50-SMA
would whipsaw) but a one-way CIRCUIT BREAKER: a VIX-spike tripwire that forces cash
for a MANDATORY cooldown, then hands re-entry back to the 200-SMA gate. It fires
once per event and counts down — so it can't whipsaw.

Trigger (checked daily): VIX_today > THRESH AND VIX_5d_ago < 25  (a 10+pt spike in a
week = panic, not a Fed-day blip). While the cooldown runs, force cash regardless of
the 200-SMA. After it expires, resume the normal gate.

Variants (reviewer's, + a Canada-local one since the book holds TSX 2x sectors):
  C        200-SMA gate only (the current shadow)
  C+V1     + VIX>35 circuit, 10d cooldown
  C+V2     + VIX>35 circuit, 20d cooldown
  C+V3     + VIX>30 circuit, 15d cooldown
  C+V1+TSX + (VIX>35 OR TSX local stress), 10d   [TSX stress = XIU 5d < -5% AND
             20d realized vol > 1.5x 60d vol]

Reports per variant: CAGR / maxDD / Sortino / Calmar, fire count, GENUINE fires
(SPY fell >5% further within the cooldown = protection earned) vs FALSE POSITIVES.
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


def _load(t, need_hl=False):
    end = datetime.now(); start = end - timedelta(days=int(YEARS * 365.25) + 260)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty or len(raw) < 200:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy(); c = df["Close"]
    df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
    df["mom"] = c / c.shift(MOM) - 1
    df["ret"] = c.pct_change()
    return df


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
    print(f"Loading CAD 2x + SPY + XIU + ^VIX, {YEARS}y...")
    data = {}
    for t in CAD_2X + ["SPY", "XIU.TO"]:
        d = _load(t)
        if d is not None:
            data[t] = d
    vix = _load("^VIX")
    cad = {t: data[t] for t in CAD_2X if t in data}
    if not cad or "SPY" not in data or "XIU.TO" not in data or vix is None:
        print("  FATAL: missing data"); return
    base = max(cad.values(), key=lambda x: len(x))
    cal = base.index

    # broad-gate boolean + VIX + TSX stress, all reindexed onto the CAD calendar
    def _above200(df):
        return (df["Close"] > df["sma200"]).reindex(cal, method="ffill").fillna(False)
    broad_long = (_above200(data["SPY"]) & _above200(data["XIU.TO"])).values
    vix_c = vix["Close"].reindex(cal, method="ffill").values
    # TSX local stress: XIU 5d return < -5% AND 20d rv > 1.5x 60d rv
    xiu = data["XIU.TO"]; xr = xiu["ret"]
    xiu5 = (xiu["Close"] / xiu["Close"].shift(5) - 1)
    rv20 = xr.rolling(20).std(); rv60 = xr.rolling(60).std()
    tsx_stress = ((xiu5 < -0.05) & (rv20 > 1.5 * rv60)).reindex(cal, method="ffill").fillna(False).values
    spy_c = data["SPY"]["Close"].reindex(cal, method="ffill").values

    def _simulate(vix_thresh=None, cooldown_n=0, use_tsx=False):
        """Returns (daily_returns, fires[list of k], genuine_count)."""
        dr = []; target = []; cooldown = 0; fires = []
        for k in range(210, len(cal)):
            d = cal[k]
            # circuit-breaker state machine (one-way, counts down)
            circuit = False
            if vix_thresh is not None:
                if cooldown > 0:
                    cooldown -= 1; circuit = True
                else:
                    vix_now = vix_c[k]; vix_5 = vix_c[k-5] if k >= 5 else 30.0
                    trig = (vix_now == vix_now and vix_now > vix_thresh and vix_5 < 25)
                    if use_tsx:
                        trig = trig or bool(tsx_stress[k])
                    if trig:
                        cooldown = cooldown_n - 1; circuit = True; fires.append(k)
            allowed = bool(broad_long[k]) and not circuit
            if k % 5 == 0:                                    # weekly target
                ranked = []
                for t, df in cad.items():
                    if d in df.index:
                        i = df.index.get_loc(d)
                        if i > 200:
                            c = df["Close"].values; s50 = df["sma50"].values
                            s200 = df["sma200"].values; m = df["mom"].values
                            if m[i] == m[i] and c[i] > s50[i] and c[i] > s200[i]:
                                ranked.append((m[i], t))
                ranked.sort(reverse=True)
                target = [t for _, t in ranked[:TOP_K]]
            held = target if allowed else []
            if not held:
                r = 0.0
            else:
                rs = []
                for t in held:
                    if d in cad[t].index:
                        i = cad[t].index.get_loc(d)
                        if i > 0:
                            rs.append(cad[t]["Close"].values[i] / cad[t]["Close"].values[i-1] - 1)
                r = float(np.mean(rs)) if rs else 0.0
            dr.append(r)
        # genuine fire = SPY fell >5% further within the cooldown window
        genuine = 0
        for k in fires:
            end = min(k + cooldown_n, len(cal) - 1)
            seg = spy_c[k:end+1]
            if len(seg) and np.nanmin(seg) / spy_c[k] - 1 < -0.05:
                genuine += 1
        return dr, fires, genuine

    def _curve(x):
        eq = 1.0; c = []
        for r in x:
            eq *= (1 + r); c.append(eq)
        return eq, c

    variants = [
        ("C  (200-SMA gate only)", None, 0, False),
        ("C+V1  VIX>35, 10d",      35, 10, False),
        ("C+V2  VIX>35, 20d",      35, 20, False),
        ("C+V3  VIX>30, 15d",      30, 15, False),
        ("C+V1+TSX  (or local), 10d", 35, 10, True),
    ]
    print("\n" + "=" * 96)
    print(" VIX CIRCUIT BREAKER — variants on the 200-SMA airbag (CAD 2x, 8y)")
    print("=" * 96)
    print(f"    {'variant':30s} {'CAGR':>7} {'maxDD':>7} {'Sortino':>8} {'Calmar':>7} "
          f"{'fires':>6} {'genuine':>8} {'FP':>4}")
    for label, th, cd, tsx in variants:
        dr, fires, genuine = _simulate(th, cd, tsx)
        eq, curve = _curve(dr)
        cagr = _cagr(eq, len(dr)); mdd = _maxdd(curve)
        calmar = cagr / abs(mdd) if mdd else 0.0
        nf = len(fires); fp = nf - genuine
        print(f"    {label:30s} {cagr:+6.1f}% {mdd:6.0f}% {_sortino(dr):+7.2f} "
              f"{calmar:6.2f} {nf:6d} {genuine:8d} {fp:4d}")

    print("\n  READ (reviewer's rule): adopt the circuit if it improves maxDD with an")
    print("  acceptable CAGR give-up AND false-positive rate < ~20%. 'genuine' = SPY")
    print("  fell >5% further inside the cooldown (the leg-down it actually dodged).")
    print("  Fires are few (2-4 in 8y is expected — this is a panic tripwire).")


if __name__ == "__main__":
    run()
