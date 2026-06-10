"""
bottom_fisher_backtest.py — Can we enter EARLIER than momentum (catch the dip /
the bottom) and still have a real, tradeable edge?

WHY
---
The momentum engine buys CONFIRMED strength, so it enters after the first leg.
The user wants earlier entries ("catch the bottom"). Bottom-fishing trades
WEAKNESS, which is lower-accuracy by nature — BUT it buys near support with a
TIGHT stop, so the reward:risk can be excellent. This script measures the truth:
win rate, reward:risk, expectancy, forward-return edge vs baseline, and an MCPT
p-value. We ship a live "Bottom Fisher" entry mode ONLY if a variant clears the
bar. No 80%-foolproof fantasy — just the real numbers.

THREE EARLY-ENTRY MODES (evaluated at bar i, using only data up to i):
  A. PULLBACK_UPTREND   — confirmed uptrend (px>rising 200SMA, 50SMA>200SMA),
                          short-term oversold pullback (RSI2<10 or px<=20SMA),
                          today turns back up (close>prev close). Buy the dip.
  B. OVERSOLD_SUPPORT   — RSI14<30 AND price within 6% of its 60-day low (at
                          support), today closes green. Deeper mean reversion,
                          does NOT require a long-term uptrend (higher variance).
  C. CAPITULATION_REV   — 5-day drop < -10%, volume spike > 1.8x avg, reversal
                          bar (close>open, close in top half of range). Panic flush.

EXIT MODEL (realistic tight-stop bottom-fishing):
  entry  = close at signal
  stop   = recent 5-bar swing low (just under support); skip if risk > 8% (not a
           clean, tight bottom). This is the discipline that makes the R:R work.
  target = entry * (1 + RR * risk)              [RR = 2.5 reward-to-risk]
  walk forward up to MAXHOLD bars, first touch wins (ties -> stop, conservative);
  otherwise exit at the close of bar i+MAXHOLD. Non-overlapping (one trade at a
  time per name) so signals don't double-count a single move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
UNIVERSE = [
    "SPY","QQQ","IWM","AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","TSLA",
    "AVGO","NFLX","CRM","ADBE","INTC","CSCO","QCOM","TXN","MU","AMAT","ORCL",
    "PLTR","COIN","SHOP","UBER","PYPL","SQ","ROKU","SNAP","PINS","DKNG","RBLX",
    "JPM","BAC","WFC","GS","MS","C","V","MA","AXP","SCHW",
    "UNH","LLY","JNJ","ABBV","MRK","PFE","TMO","ABT","BMY","GILD","MRNA","BIIB",
    "WMT","COST","HD","MCD","NKE","SBUX","DIS","KO","PG","TGT","LOW",
    "CAT","BA","GE","HON","UPS","XOM","CVX","COP","SLB","FCX","NEM",
    "F","GM","DAL","AAL","CCL","NCLH","RIVN","LCID","CHPT","PLUG","SOFI","HOOD",
]
YEARS      = 6
FWD        = [5, 10, 20]      # forward-return horizons for the directional-edge view
RR         = 2.5             # reward-to-risk target multiple
MAXHOLD    = 20             # max bars to hold a trade
MAX_RISK   = 0.08          # skip signals whose stop is wider than 8% (not tight)
MODES      = ["PULLBACK_UPTREND", "OVERSOLD_SUPPORT", "CAPITULATION_REV"]


# ── Data + indicators ─────────────────────────────────────────────────────────
def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Wilder smoothing
    rsi = np.full_like(close, 50.0, dtype=float)
    ag = al = 0.0
    for i in range(1, len(close)):
        if i <= period:
            ag = (ag * (i - 1) + gain[i]) / i
            al = (al * (i - 1) + loss[i]) / i
        else:
            ag = (ag * (period - 1) + gain[i]) / period
            al = (al * (period - 1) + loss[i]) / period
        rsi[i] = 100.0 if al < 1e-12 else 100.0 - 100.0 / (1.0 + ag / al)
    return rsi


def _load(t: str):
    try:
        end = datetime.now(); start = end - timedelta(days=int(YEARS * 365.25) + 400)
        raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 320:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        c = df["Close"]
        df["sma20"]  = c.rolling(20).mean()
        df["sma50"]  = c.rolling(50).mean()
        df["sma200"] = c.rolling(200).mean()
        df["vol20"]  = df["Volume"].rolling(20).mean()
        df["rsi2"]   = _rsi(c.values, 2)
        df["rsi14"]  = _rsi(c.values, 14)
        df["low60"]  = df["Low"].rolling(60).min()
        return df.dropna()
    except Exception:
        return None


# ── Signal logic per mode (returns boolean array) ─────────────────────────────
def _signals(df: pd.DataFrame, mode: str) -> np.ndarray:
    C = df["Close"].values; O = df["Open"].values
    H = df["High"].values;  L = df["Low"].values
    s20 = df["sma20"].values; s50 = df["sma50"].values; s200 = df["sma200"].values
    rsi2 = df["rsi2"].values; rsi14 = df["rsi14"].values
    vol = df["Volume"].values; vol20 = df["vol20"].values; low60 = df["low60"].values
    n = len(C); sig = np.zeros(n, dtype=bool)

    for i in range(220, n):
        if mode == "PULLBACK_UPTREND":
            # confirmed uptrend
            uptrend = (C[i] > s200[i] and s50[i] > s200[i]
                       and s200[i] > s200[i - 20])           # 200SMA rising
            if not uptrend:
                continue
            pulled = (rsi2[i - 1] < 10) or (C[i - 1] <= s20[i - 1])  # was oversold/dipped
            turn   = C[i] > C[i - 1]                          # turning back up today
            if pulled and turn:
                sig[i] = True

        elif mode == "OVERSOLD_SUPPORT":
            oversold = rsi14[i] < 30
            at_supp  = low60[i] > 0 and (C[i] / low60[i] - 1) <= 0.06   # within 6% of 60d low
            turn     = C[i] > C[i - 1]
            if oversold and at_supp and turn:
                sig[i] = True

        elif mode == "CAPITULATION_REV":
            drop5 = (C[i] / C[i - 5] - 1) < -0.10 if C[i - 5] > 0 else False
            vspike = vol20[i] > 0 and vol[i] > 1.8 * vol20[i]
            rng = H[i] - L[i]
            rev_bar = rng > 0 and (C[i] > O[i]) and ((C[i] - L[i]) / rng > 0.5)
            if drop5 and vspike and rev_bar:
                sig[i] = True
    return sig


# ── Realistic exit model -> realized trade returns (non-overlapping) ──────────
def _trades(df: pd.DataFrame, mode: str) -> list:
    sig = _signals(df, mode)
    C = df["Close"].values; H = df["High"].values; L = df["Low"].values
    n = len(C); out = []
    i = 220
    while i < n - 1:
        if not sig[i]:
            i += 1; continue
        entry = C[i]
        swing_low = L[max(0, i - 5):i + 1].min()
        stop = swing_low * 0.999
        if entry <= 0 or stop <= 0 or stop >= entry:
            i += 1; continue
        risk = (entry - stop) / entry
        if risk > MAX_RISK or risk < 0.005:        # must be a tight, clean bottom
            i += 1; continue
        target = entry * (1 + RR * risk)
        realized = None; exit_i = min(i + MAXHOLD, n - 1)
        for j in range(i + 1, exit_i + 1):
            if L[j] <= stop:                       # stop first (conservative on ties)
                realized = -risk; exit_i = j; break
            if H[j] >= target:
                realized = RR * risk; exit_i = j; break
        if realized is None:
            realized = (C[exit_i] - entry) / entry
        out.append(realized)
        i = exit_i + 1                              # non-overlapping
    return out


def _forward(df: pd.DataFrame, mode: str) -> dict:
    sig = _signals(df, mode); C = df["Close"].values; n = len(C)
    out = {w: [] for w in FWD}
    for i in np.where(sig)[0]:
        for w in FWD:
            if i + w < n and C[i] > 0:
                out[w].append((C[i + w] / C[i] - 1) * 100)
    return out


def _stats(trades: list) -> dict:
    a = np.asarray(trades, dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0}
    wins = a[a > 0]; losses = a[a < 0]
    avg_w = wins.mean() if len(wins) else 0.0
    avg_l = losses.mean() if len(losses) else 0.0
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    return {
        "n": n,
        "win_rate": 100 * (a > 0).mean(),
        "expectancy": a.mean() * 100,           # % per trade
        "avg_win": avg_w * 100,
        "avg_loss": avg_l * 100,
        "reward_risk": (avg_w / -avg_l) if avg_l < 0 else float("inf"),
        "profit_factor": pf,
    }


# ── MCPT (reuse the project's permutation engine) ─────────────────────────────
def _mcpt(data: dict, mode: str, n_perm: int = 150, seed: int = 7) -> dict:
    from permutation_test import get_permutation, metric_from_trades
    real_trades = []
    for df in data.values():
        real_trades += _trades(df, mode)
    real = metric_from_trades(real_trades)
    null_pf = []
    for p in range(n_perm):
        pt = []
        for ti, df in enumerate(data.values()):
            need = ["Open", "High", "Low", "Close", "Volume"]
            perm = get_permutation(df[need], start_index=0, seed=seed + p * 1000 + ti)
            # rebuild indicators on the permuted bars
            perm = perm.copy()
            c = perm["Close"]
            perm["sma20"] = c.rolling(20).mean(); perm["sma50"] = c.rolling(50).mean()
            perm["sma200"] = c.rolling(200).mean(); perm["vol20"] = perm["Volume"].rolling(20).mean()
            perm["rsi2"] = _rsi(c.values, 2); perm["rsi14"] = _rsi(c.values, 14)
            perm["low60"] = perm["Low"].rolling(60).min()
            perm = perm.dropna()
            if len(perm) > 240:
                pt += _trades(perm, mode)
        null_pf.append(metric_from_trades(pt)["profit_factor"])
    null_pf = np.asarray(null_pf, dtype=float)
    p_pf = (1 + int(np.sum(null_pf >= real["profit_factor"]))) / (n_perm + 1)
    return {"real_pf": real["profit_factor"], "real_n": real["n"],
            "null_pf_mean": float(null_pf.mean()),
            "null_pf_95": float(np.percentile(null_pf, 95)), "p_value": round(p_pf, 4)}


def run(do_mcpt: bool = True):
    print(f"Loading {len(UNIVERSE)} tickers, {YEARS}y...")
    data = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_load, t): t for t in UNIVERSE}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None:
                data[futs[fut]] = df
    print(f"  loaded {len(data)}\n")

    # baseline forward returns (random entry)
    base = {w: [] for w in FWD}
    for df in data.values():
        C = df["Close"].values; n = len(C)
        for i in range(220, n):
            for w in FWD:
                if i + w < n and C[i] > 0:
                    base[w].append((C[i + w] / C[i] - 1) * 100)
    print("BASELINE (random entry, forward %):")
    for w in FWD:
        a = np.array(base[w]); print(f"  {w}d: mean {a.mean():+.2f}%  win {100*(a>0).mean():.1f}%")

    print("\n" + "=" * 92)
    print(" BOTTOM-FISHER ENTRY MODES — realistic tight-stop exit "
          f"(RR={RR}:1, stop@swing-low, {MAXHOLD}-bar max)")
    print("=" * 92)
    results = {}
    for mode in MODES:
        trades, fwd = [], {w: [] for w in FWD}
        for df in data.values():
            trades += _trades(df, mode)
            f = _forward(df, mode)
            for w in FWD: fwd[w].extend(f[w])
        s = _stats(trades); results[mode] = s
        print(f"\n  ▶ {mode}")
        if s["n"] < 20:
            print(f"    only {s['n']} trades — too few to judge."); continue
        print(f"    TRADEABLE (stop/target):  n={s['n']}  "
              f"win {s['win_rate']:.1f}%  expectancy {s['expectancy']:+.2f}%/trade  "
              f"R:R {s['reward_risk']:.2f}  PF {s['profit_factor']:.2f}")
        print(f"    avg win {s['avg_win']:+.2f}%   avg loss {s['avg_loss']:+.2f}%")
        for w in FWD:
            a = np.array(fwd[w]); b = np.array(base[w])
            if len(a):
                print(f"    fwd {w}d: mean {a.mean():+.2f}%  win {100*(a>0).mean():.1f}%  "
                      f"edge vs random {a.mean()-b.mean():+.2f}%")

    # MCPT on the best-by-expectancy mode with enough trades
    if do_mcpt:
        cand = [m for m in MODES if results.get(m, {}).get("n", 0) >= 30]
        if cand:
            best = max(cand, key=lambda m: results[m]["expectancy"])
            print("\n" + "=" * 92)
            print(f" MCPT on best mode: {best}  (is the edge real or luck?)")
            print("=" * 92)
            mc = _mcpt(data, best, n_perm=150)
            print(f"  real profit factor : {mc['real_pf']}  (n={mc['real_n']} trades)")
            print(f"  null PF mean/95th  : {mc['null_pf_mean']:.3f} / {mc['null_pf_95']:.3f}")
            print(f"  p-value            : {mc['p_value']}")
            v = ("STRONG real edge (p<=0.01)" if mc['p_value'] <= 0.01 else
                 "Significant edge (p<=0.05)" if mc['p_value'] <= 0.05 else
                 "Weak/marginal (0.05<p<=0.20)" if mc['p_value'] <= 0.20 else
                 "NO edge — consistent with random")
            print(f"  VERDICT            : {v}")
    return results


if __name__ == "__main__":
    run()
