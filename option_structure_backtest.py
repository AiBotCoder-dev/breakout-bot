"""
option_structure_backtest.py — Does changing the OPTION STRUCTURE raise win rate?

We currently buy naked OTM calls. This tests, via Black-Scholes on our validated
MOMENTUM setups, whether different structures win more often / pay better:

  Entries: momentum/uptrend longs (price>rising 50&200SMA) — the big validated bucket.
  For each entry: take the real forward 10-day underlying move, price each option
  STRUCTURE at entry and at exit (10 days later, same IV — isolates the structure
  effect), and record the option return. Then win rate + mean + median per structure.

STRUCTURES (all ~21 DTE at entry, exit after 10 days):
  A. OTM call  (8% OTM)            — the current "lottery" approach
  B. NTM call  (2% OTM)            — nearer the money, higher delta
  C. ATM call  (at the money)      — highest practical delta long
  D. ITM call  (5% ITM)            — deep delta, most stock-like
  E. Debit spread (ATM long / 10% OTM short) — lower breakeven, theta-hedged
  F. ATM call 45 DTE               — more time, less theta

Entry IV per name = trailing 20-day realized vol x 1.1 (a realistic IV premium).
Honest, large sample, ~6y liquid universe.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = [
    "AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","TSLA","AVGO","NFLX","CRM",
    "ADBE","INTC","QCOM","MU","AMAT","ORCL","PLTR","COIN","SHOP","UBER","ROKU",
    "DKNG","RBLX","JPM","BAC","GS","V","MA","UNH","LLY","WMT","COST","HD","NKE",
    "DIS","CAT","BA","XOM","CVX","SOFI","HOOD","SMCI","MRVL","SNOW","NET","CRWD","DDOG",
]
HOLD = 10
R = 0.04
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, sig):
    if T <= 0:
        return max(S - K, 0.0)
    if sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * N(d1) - K * math.exp(-R * T) * N(d2)


def _load(t):
    try:
        end = datetime.now(); start = end - timedelta(days=int(6 * 365.25) + 260)
        raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 300:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        c = df["Close"]
        df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
        df["rv20"] = np.log(c / c.shift()).rolling(20).std() * np.sqrt(252)
        return df.dropna()
    except Exception:
        return None


def _structures(S0, S1, iv):
    """Return {name: option_return_pct} for each structure, entry->exit(HOLD days)."""
    Te, Tx = 21 / 365, (21 - HOLD) / 365
    out = {}
    defs = {
        "A_OTM8":  ("call", S0 * 1.08),
        "B_NTM2":  ("call", S0 * 1.02),
        "C_ATM":   ("call", S0),
        "D_ITM5":  ("call", S0 * 0.95),
        "F_ATM45": ("call45", S0),
    }
    for name, (kind, K) in defs.items():
        if kind == "call45":
            e = bs_call(S0, K, 45 / 365, iv); x = bs_call(S1, K, 35 / 365, iv)
        else:
            e = bs_call(S0, K, Te, iv); x = bs_call(S1, K, Tx, iv)
        out[name] = (x / e - 1) * 100 if e > 0.01 else 0.0
    # E. debit spread: long ATM, short 10% OTM
    el = bs_call(S0, S0, Te, iv); es = bs_call(S0, S0 * 1.10, Te, iv)
    xl = bs_call(S1, S0, Tx, iv); xs = bs_call(S1, S0 * 1.10, Tx, iv)
    edeb, xdeb = el - es, xl - xs
    out["E_SPREAD"] = (xdeb / edeb - 1) * 100 if edeb > 0.01 else 0.0
    return out


def run():
    print(f"Loading {len(UNIVERSE)} names, 6y...")
    data = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_load, t): t for t in UNIVERSE}
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                data[futs[f]] = r
    print(f"  loaded {len(data)}\n")

    res = {k: [] for k in ["A_OTM8", "B_NTM2", "C_ATM", "D_ITM5", "E_SPREAD", "F_ATM45"]}
    for df in data.values():
        C = df["Close"].values; s50 = df["sma50"].values; s200 = df["sma200"].values
        rv = df["rv20"].values; n = len(C)
        for i in range(210, n - HOLD):
            # momentum/uptrend entry
            if not (C[i] > s50[i] > s200[i] and s50[i] > s50[i-10] and s200[i] > s200[i-20]):
                continue
            iv = max(0.12, min(1.5, rv[i] * 1.1))
            S0, S1 = C[i], C[i + HOLD]
            if S0 <= 0:
                continue
            for k, v in _structures(S0, S1, iv).items():
                res[k].append(v)

    print("=" * 78)
    print(f" OPTION STRUCTURE on momentum longs — {HOLD}d hold, win rate & expectancy")
    print("=" * 78)
    labels = {
        "A_OTM8":  "OTM call (8% OTM)  [CURRENT]",
        "B_NTM2":  "NTM call (2% OTM)",
        "C_ATM":   "ATM call",
        "D_ITM5":  "ITM call (5% ITM)",
        "E_SPREAD":"Debit spread (ATM/+10%)",
        "F_ATM45": "ATM call 45 DTE",
    }
    base_win = None
    for k in ["A_OTM8", "B_NTM2", "C_ATM", "D_ITM5", "E_SPREAD", "F_ATM45"]:
        a = np.array(res[k], dtype=float)
        if len(a) == 0:
            continue
        win = 100 * (a > 0).mean(); mean = a.mean(); med = np.median(a)
        if base_win is None:
            base_win = win
        lift = win - base_win
        print(f"  {labels[k]:<30} n={len(a):<6} win {win:5.1f}%  "
              f"mean {mean:+6.1f}%  median {med:+6.1f}%  (win vs OTM {lift:+.1f}pp)")
    print("\n  NOTE: spreads/ITM raise WIN RATE but cap the home-run mean. The right")
    print("  read is win-rate + median (typical outcome), not just mean (skewed by tails).")


if __name__ == "__main__":
    run()
