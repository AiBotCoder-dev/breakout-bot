"""
weekly_research.py — MASSIVE backtest of WEEKLY call-option strategies.

Tests a matrix of [9 entry strategies] x [6 call structures], priced in the WEEKLY
regime (≈7-DTE at entry, 5-trading-day hold ≈ 2 DTE at exit), NET of real friction
(bid/ask + commission on every leg) and IV crush — reusing the validated engine in
full_bot_backtest.py. Ranks every combo by expectancy / median / win rate so we can
see which weekly-call strategies actually make money after costs.

Run:  python weekly_research.py            (weekly: 7-DTE entry, 5d hold)
      python weekly_research.py 14 10       (override entry DTE / hold days)
"""
import sys
import math
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

from full_bot_backtest import (bs_call, buy_fill, sell_fill, _rsi,
                               IV_CRUSH, ENTRY_IV_PREMIUM)

# ── weekly regime knobs (overridable on the CLI) ────────────────────────────
DTE_ENTRY = int(sys.argv[1]) if len(sys.argv) > 1 else 7     # ≈ weekly
HOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 5          # trading days held
# calendar days elapsed ≈ trading days x 7/5; exit DTE = entry DTE - elapsed
DTE_EXIT = max(1, DTE_ENTRY - max(1, round(HOLD * 1.4)))

UNIVERSE = ["NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","AMD","AVGO","NFLX",
            "JPM","BAC","XOM","UNH","V","MA","COST","WMT","DIS","CRM","ORCL","ADBE",
            "QCOM","MU","INTC","PLTR","COIN","SOFI","MARA","RIOT","SMCI","ARM","SNOW",
            "UBER","ABNB","SHOP","RBLX","CVNA","DKNG","SPY","QQQ","IWM","SMH","XLE","XLF"]

# ── option structures priced over the weekly hold ───────────────────────────
def _call(S0, S1, K, iv0, iv1):
    Te, Tx = DTE_ENTRY/365.0, DTE_EXIT/365.0
    me = bs_call(S0, K, Te, iv0); mx = bs_call(S1, K, Tx, iv1)
    cost = buy_fill(me); proc = sell_fill(mx)
    return (proc/cost - 1)*100 if cost > 0.01 else 0.0

def _spread(S0, S1, Kl, Ks, iv0, iv1):
    Te, Tx = DTE_ENTRY/365.0, DTE_EXIT/365.0
    le = bs_call(S0, Kl, Te, iv0); se = bs_call(S0, Ks, Te, iv0)
    lx = bs_call(S1, Kl, Tx, iv1); sx = bs_call(S1, Ks, Tx, iv1)
    dn = buy_fill(le) - sell_fill(se)
    vn = max(0.0, sell_fill(lx) - buy_fill(sx))
    return (vn/dn - 1)*100 if dn > 0.01 else 0.0

def structures(S0, S1, iv0, iv1):
    return {
        "ITM -5%":      _call(S0, S1, S0*0.95, iv0, iv1),
        "ATM":          _call(S0, S1, S0*1.00, iv0, iv1),
        "OTM +3%":      _call(S0, S1, S0*1.03, iv0, iv1),
        "OTM +5%":      _call(S0, S1, S0*1.05, iv0, iv1),
        "SPR ATM/+5%":  _spread(S0, S1, S0*1.00, S0*1.05, iv0, iv1),
        "SPR ATM/+10%": _spread(S0, S1, S0*1.00, S0*1.10, iv0, iv1),
    }
STRUCTS = ["ITM -5%", "ATM", "OTM +3%", "OTM +5%", "SPR ATM/+5%", "SPR ATM/+10%"]

# ── entry strategies — each returns the list of strategy names firing at bar i ─
def signals(i, A):
    C = A["C"]; up = C[i] > A["s50"][i] > A["s200"][i]
    out = []
    if up and A["mom6"][i] > 0.10:                                  out.append("MOMENTUM")
    if up and A["rsi"][i] < 40:                                     out.append("PULLBACK_UPTREND")
    if C[i] > A["hi20"][i-1]:                                       out.append("BREAKOUT_20D")
    if C[i] > A["hi55"][i-1]:                                       out.append("BREAKOUT_55D")
    if C[i] > A["s50"][i] and C[i-1] < A["s50"][i-1] and A["s50"][i] > A["s200"][i]:
        out.append("MA50_RECLAIM")
    if A["rsi"][i-1] < 30 <= A["rsi"][i] and C[i] > A["s200"][i]:   out.append("RSI_BOUNCE_UPTR")
    if A["mom1"][i] > 0.10 and C[i] > A["s50"][i]:                  out.append("STRONG_MOM_1M")
    if C[i] > A["ph"][i-1] and C[i] > A["s50"][i]:                  out.append("GAP_CONTINUATION")
    if A["rsi"][i] < 25:                                            out.append("DEEP_OVERSOLD")
    return out

STRATS = ["MOMENTUM","PULLBACK_UPTREND","BREAKOUT_20D","BREAKOUT_55D","MA50_RECLAIM",
          "RSI_BOUNCE_UPTR","STRONG_MOM_1M","GAP_CONTINUATION","DEEP_OVERSOLD"]


def load(tk, data):
    try:
        df = data[tk].dropna(subset=["Close"]).copy()
    except Exception:
        return None
    if len(df) < 260:
        return None
    c = df["Close"]
    A = {
        "C": c.values,
        "s50": c.rolling(50).mean().values,
        "s200": c.rolling(200).mean().values,
        "rsi": _rsi(c.values, 14),
        "mom6": (c/c.shift(126) - 1).values,
        "mom1": (c/c.shift(21) - 1).values,
        "hi20": c.rolling(20).max().values,
        "hi55": c.rolling(55).max().values,
        "ph": df["High"].values,
        "rv": (np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)).values,
    }
    return A


def main():
    print(f"WEEKLY CALL RESEARCH — entry ~{DTE_ENTRY} DTE, {HOLD}d hold "
          f"(exit ~{DTE_EXIT} DTE), NET of friction + {IV_CRUSH*100:.0f}% IV crush, "
          f"entry IV = realized x {ENTRY_IV_PREMIUM:.2f}")
    print(f"universe: {len(UNIVERSE)} names, 5y\n")
    data = yf.download(UNIVERSE, period="5y", auto_adjust=True, group_by="ticker",
                       threads=True, progress=False)
    # acc[strat][struct] = list of NET returns
    acc = {st: {k: [] for k in STRUCTS} for st in STRATS}
    nload = 0
    for tk in UNIVERSE:
        A = load(tk, data)
        if A is None:
            continue
        nload += 1
        C, rv = A["C"], A["rv"]
        n = len(C)
        last = {st: -10**9 for st in STRATS}
        for i in range(210, n - HOLD):
            if not np.isfinite(C[i]) or not np.isfinite(rv[i]) or rv[i] <= 0:
                continue
            fired = signals(i, A)
            if not fired:
                continue
            S0 = C[i]; S1 = C[i+HOLD]
            if not np.isfinite(S1):
                continue
            iv0 = max(0.15, min(2.0, rv[i]*ENTRY_IV_PREMIUM))
            iv1 = max(0.05, iv0*(1.0 - IV_CRUSH))
            sret = structures(S0, S1, iv0, iv1)
            for st in fired:
                if i - last[st] < HOLD:          # non-overlap per strategy
                    continue
                last[st] = i
                for k in STRUCTS:
                    acc[st][k].append(sret[k])
    print(f"loaded {nload}/{len(UNIVERSE)} names\n")

    # ── full matrix: expectancy (mean NET%) per strategy x structure ──────────
    print("="*92)
    print("  EXPECTANCY (mean NET % per trade) — strategy x structure")
    print("="*92)
    hdr = "  " + f"{'strategy':<18}" + "".join(f"{k:>12}" for k in STRUCTS) + f"{'n':>7}"
    print(hdr); print("  " + "-"*(len(hdr)-2))
    combos = []
    for st in STRATS:
        ns = len(acc[st][STRUCTS[0]])
        row = "  " + f"{st:<18}"
        for k in STRUCTS:
            arr = np.array(acc[st][k])
            m = arr.mean() if len(arr) else 0
            row += f"{m:>+11.0f}%"
            if len(arr):
                combos.append((st, k, len(arr), 100*(arr>0).mean(),
                               arr.mean(), float(np.median(arr))))
        row += f"{ns:>7}"
        print(row)

    # ── top combos by expectancy (require a real sample) ──────────────────────
    print("\n" + "="*92)
    print("  TOP 15 STRATEGY x STRUCTURE BY EXPECTANCY (n>=80, NET of costs)")
    print("="*92)
    print(f"  {'strategy':<18}{'structure':<14}{'n':>6}{'win%':>7}{'mean%':>8}{'median%':>9}")
    print("  " + "-"*70)
    for st, k, n, w, mean, med in sorted([c for c in combos if c[2] >= 80],
                                         key=lambda x: -x[4])[:15]:
        print(f"  {st:<18}{k:<14}{n:>6}{w:>6.0f}%{mean:>+7.0f}%{med:>+8.0f}%")

    # ── per-structure aggregate (which wrapper wins across all strategies) ────
    print("\n" + "="*92)
    print("  STRUCTURE SCORECARD (pooled across all strategies)")
    print("="*92)
    print(f"  {'structure':<14}{'n':>7}{'win%':>7}{'mean%':>8}{'median%':>9}")
    print("  " + "-"*48)
    for k in STRUCTS:
        allr = np.array([r for st in STRATS for r in acc[st][k]])
        if len(allr):
            print(f"  {k:<14}{len(allr):>7}{100*(allr>0).mean():>6.0f}%"
                  f"{allr.mean():>+7.0f}%{float(np.median(allr)):>+8.0f}%")

    # ── per-strategy best structure ──────────────────────────────────────────
    print("\n" + "="*92)
    print("  BEST STRUCTURE PER STRATEGY (by expectancy)")
    print("="*92)
    for st in STRATS:
        best = None
        for k in STRUCTS:
            arr = np.array(acc[st][k])
            if len(arr) >= 80 and (best is None or arr.mean() > best[1]):
                best = (k, arr.mean(), 100*(arr>0).mean(), float(np.median(arr)), len(arr))
        if best:
            print(f"  {st:<18} -> {best[0]:<14} mean {best[1]:>+5.0f}%  "
                  f"win {best[2]:>3.0f}%  median {best[3]:>+5.0f}%  (n={best[4]})")
        else:
            print(f"  {st:<18} -> (insufficient sample)")
    print("\nNote: NET returns after bid/ask + commission + IV crush. Expectancy>0 means")
    print("the average weekly trade makes money after costs. Median<0 with mean>0 = a")
    print("fat-tailed profile (a few big winners carry it) — size accordingly.")


if __name__ == "__main__":
    main()
