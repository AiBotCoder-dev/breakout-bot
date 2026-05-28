"""
permutation_test.py — Monte Carlo Permutation Test (MCPT) for the breakout strategy.

WHAT THIS ANSWERS
-----------------
"Does the breakout strategy actually have an edge, or did the backtest just get
lucky / overfit the data?"  Adapted from NeuroTrader's MCPT method
(github.com/neurotrader888/mcpt).

HOW IT WORKS
------------
1. Run the REAL strategy (the bot's own PatternDetector + BreakoutProbabilityEngine
   replay, identical to BacktestEngine) on real price history → record a metric
   (profit factor / mean trade return).
2. Permute the price bars many times. `get_permutation()` shuffles the bar-to-bar
   moves in LOG space so each synthetic series keeps the same volatility, gaps and
   bar shapes as the real data but has NO genuine temporal structure to predict.
3. Re-run the exact same strategy on each shuffled series → a null distribution of
   "what the metric looks like when there's nothing real to find."
4. p-value = fraction of permutations whose metric ≥ the real metric.
   p ≤ 0.05  → strong evidence of a real edge.
   p ≈ 0.5  → the strategy is no better than random (overfit / noise).

IMPORTANT SCOPE
---------------
This validates the PRICE/PATTERN side only. Permuting OHLC bars cannot test the
news / congressional / StockTwits / options-flow engines (those signals aren't in
the bars). It is the right tool for the technical patterns and the Master-Score
weighting — not for the whole multi-source stack.
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None

import trading_scanner as ts
from trading_scanner import (
    _norm, _fcol, PatternDetector, BreakoutProbabilityEngine, BreakoutScanner,
)
from datetime import datetime, timedelta


# ══════════════════════════════════════════════════════════════════════════════
# BAR PERMUTATION  (NeuroTrader algorithm, adapted for capitalized OHLCV)
# ══════════════════════════════════════════════════════════════════════════════
def get_permutation(df: pd.DataFrame, start_index: int = 0, seed=None) -> pd.DataFrame:
    """
    Return a permuted copy of an OHLCV DataFrame.

    Preserves each bar's shape (open→high/low/close moves) and the gap distribution
    (open vs previous close) but shuffles them across time, destroying any real
    sequential/predictable structure. Volume is shuffled with the same permutation
    so its distribution is preserved while decorrelating it from time.

    Columns expected: Open, High, Low, Close, Volume (capitalized). Index preserved.
    """
    assert start_index >= 0
    rng = np.random.default_rng(seed)

    cols = ["Open", "High", "Low", "Close"]
    n_bars = len(df)
    perm_index = start_index + 1
    perm_n = n_bars - perm_index
    if perm_n <= 2:
        return df.copy()

    log_bars = np.log(df[cols].to_numpy())          # (n_bars, 4)
    o, h, l, c = (log_bars[:, 0], log_bars[:, 1], log_bars[:, 2], log_bars[:, 3])

    start_bar = log_bars[start_index].copy()

    # relative components
    r_o = o - np.concatenate([[np.nan], c[:-1]])     # open vs previous close (gap)
    r_h = h - o                                      # high vs open
    r_l = l - o                                      # low  vs open
    r_c = c - o                                      # close vs open

    rel_o = r_o[perm_index:]
    rel_h = r_h[perm_index:]
    rel_l = r_l[perm_index:]
    rel_c = r_c[perm_index:]

    idx = np.arange(perm_n)
    perm1 = rng.permutation(idx)                     # intrabar (h/l/c) shuffle
    rel_h, rel_l, rel_c = rel_h[perm1], rel_l[perm1], rel_c[perm1]
    perm2 = rng.permutation(idx)                     # gaps shuffled separately
    rel_o = rel_o[perm2]

    perm_bars = np.zeros((n_bars, 4))
    perm_bars[:start_index] = log_bars[:start_index]
    perm_bars[start_index] = start_bar
    for i in range(perm_index, n_bars):
        k = i - perm_index
        perm_bars[i, 0] = perm_bars[i - 1, 3] + rel_o[k]   # open
        perm_bars[i, 1] = perm_bars[i, 0] + rel_h[k]       # high
        perm_bars[i, 2] = perm_bars[i, 0] + rel_l[k]       # low
        perm_bars[i, 3] = perm_bars[i, 0] + rel_c[k]       # close

    perm_bars = np.exp(perm_bars)
    out = pd.DataFrame(perm_bars, index=df.index, columns=cols)

    # Volume: keep real before start, shuffle the rest by the same intrabar perm.
    vol = df["Volume"].to_numpy(dtype=float).copy()
    vtail = vol[perm_index:][perm1]
    out["Volume"] = np.concatenate([vol[:perm_index], vtail])

    # Guard against any non-finite artefacts
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY REPLAY  (mirrors BacktestEngine._replay_ticker, on a provided df)
# ══════════════════════════════════════════════════════════════════════════════
def strategy_trades(scanner: BreakoutScanner, df_full: pd.DataFrame,
                    hold: int = 20, days: int = 120, min_prob: int = 50) -> list:
    """Run the real breakout strategy over df_full; return realized trade returns."""
    n = len(df_full)
    trades = []
    end_back   = min(days + hold + 20, n - hold)
    start_back = hold + 20
    if end_back < start_back:
        return trades

    for back in range(end_back, start_back - 1, -10):
        cutoff = n - back
        if cutoff < 60:
            continue
        df_slice = df_full.iloc[:cutoff].copy()
        df_fwd   = df_full.iloc[cutoff:cutoff + hold]
        if len(df_fwd) < 5:
            continue
        try:
            df_ind = scanner._indicators(df_slice)
            pats   = PatternDetector(df_ind, None, scanner.cfg).run_all()
            prob   = BreakoutProbabilityEngine(pats, df_ind, None,
                                               scanner.cfg).calculate_probability()
        except Exception:
            continue
        if prob["probability"] < min_prob:
            continue

        entry = float(df_slice["Close"].iloc[-1])
        if entry <= 0:
            continue
        atr_s   = _fcol(df_ind, "ATRr")
        atr_val = float(atr_s.iloc[-1]) if not atr_s.empty else entry * 0.02
        stop    = entry - atr_val * 1.5
        target  = (float(df_slice["High"].values[-252:].max())
                   if cutoff >= 252 else float(df_slice["High"].max()))

        # Walk forward — first touch wins; realized return reflects the actual fill.
        realized = None
        for _, row in df_fwd.iterrows():
            if target > entry and float(row["High"]) >= target:
                realized = (target - entry) / entry
                break
            if float(row["Low"]) <= stop:
                realized = (stop - entry) / entry
                break
        if realized is None:
            realized = (float(df_fwd["Close"].iloc[-1]) - entry) / entry

        trades.append(realized)

    return trades


def metric_from_trades(trades: list) -> dict:
    """Profit factor + mean return + win rate + n."""
    n = len(trades)
    if n == 0:
        return {"profit_factor": 0.0, "mean_return": 0.0, "win_rate": 0.0, "n": 0}
    arr   = np.asarray(trades, dtype=float)
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    pf = (gains / losses) if losses > 1e-9 else (gains if gains > 0 else 0.0)
    return {
        "profit_factor": round(float(pf), 4),
        "mean_return":   round(float(arr.mean()), 5),
        "win_rate":      round(float((arr > 0).mean() * 100), 1),
        "n":             n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MCPT RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def _download(ticker: str) -> pd.DataFrame | None:
    if yf is None:
        return None
    try:
        end   = datetime.now()
        start = end - timedelta(days=500)
        raw   = yf.download(ticker, start=start, end=end,
                            progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 120:
            return None
        df = _norm(raw).dropna()
        return df if len(df) >= 120 else None
    except Exception:
        return None


def run_mcpt(tickers: list, n_perm: int = 200, hold: int = 20, days: int = 120,
             min_prob: int = 50, seed: int = 42, progress=None) -> dict:
    """
    Run the permutation test pooled across `tickers`.
    Returns a dict with the real metrics, null distributions, and p-values.
    """
    scanner = BreakoutScanner()

    # ── load real data once ──────────────────────────────────────────────────
    data = {}
    for t in tickers:
        df = _download(t)
        if df is not None:
            data[t] = df
    if not data:
        raise RuntimeError("No price data could be loaded for any ticker.")
    loaded = list(data.keys())

    # ── real strategy metric (pooled) ─────────────────────────────────────────
    real_trades = []
    for t, df in data.items():
        real_trades += strategy_trades(scanner, df, hold, days, min_prob)
    real = metric_from_trades(real_trades)

    # ── null distribution ──────────────────────────────────────────────────────
    null_pf, null_mean = [], []
    for p in range(n_perm):
        ptrades = []
        for ti, (t, df) in enumerate(data.items()):
            perm_df = get_permutation(df, start_index=0, seed=seed + p * 1000 + ti)
            ptrades += strategy_trades(scanner, perm_df, hold, days, min_prob)
        m = metric_from_trades(ptrades)
        null_pf.append(m["profit_factor"])
        null_mean.append(m["mean_return"])
        if progress:
            try:
                progress(p + 1, n_perm, m)
            except Exception:
                pass

    null_pf   = np.asarray(null_pf, dtype=float)
    null_mean = np.asarray(null_mean, dtype=float)

    # one-sided p-value: how often does random match/beat the real metric?
    p_pf   = (1 + int(np.sum(null_pf   >= real["profit_factor"]))) / (n_perm + 1)
    p_mean = (1 + int(np.sum(null_mean >= real["mean_return"])))  / (n_perm + 1)

    return {
        "tickers": loaded,
        "n_perm": n_perm,
        "hold": hold,
        "days": days,
        "real": real,
        "p_value_profit_factor": round(p_pf, 4),
        "p_value_mean_return":   round(p_mean, 4),
        "null_pf_mean":   round(float(null_pf.mean()), 4),
        "null_pf_pct95":  round(float(np.percentile(null_pf, 95)), 4),
        "null_mean_mean": round(float(null_mean.mean()), 5),
        "null_mean_pct95": round(float(np.percentile(null_mean, 95)), 5),
    }


def print_report(res: dict):
    real = res["real"]
    sep = "=" * 64
    print(f"\n{sep}")
    print("  MONTE CARLO PERMUTATION TEST — Breakout Strategy")
    print(sep)
    print(f"  Tickers     : {', '.join(res['tickers'])}")
    print(f"  Permutations: {res['n_perm']}   Hold: {res['hold']}d   Lookback: {res['days']}d")
    print(f"\n  REAL STRATEGY")
    print(f"    Trades        : {real['n']}")
    print(f"    Win rate      : {real['win_rate']}%")
    print(f"    Mean return   : {real['mean_return']*100:+.2f}% per trade")
    print(f"    Profit factor : {real['profit_factor']}")
    print(f"\n  NULL (random) DISTRIBUTION")
    print(f"    Profit factor : mean {res['null_pf_mean']}  |  95th pct {res['null_pf_pct95']}")
    print(f"    Mean return   : mean {res['null_mean_mean']*100:+.2f}%  |  "
          f"95th pct {res['null_mean_pct95']*100:+.2f}%")
    print(f"\n  P-VALUES  (chance random ≥ real)")
    print(f"    Profit factor : p = {res['p_value_profit_factor']}")
    print(f"    Mean return   : p = {res['p_value_mean_return']}")

    pf_p = res["p_value_profit_factor"]
    verdict = ("STRONG real edge (p ≤ 0.01)"      if pf_p <= 0.01 else
               "Significant edge (p ≤ 0.05)"      if pf_p <= 0.05 else
               "Weak / marginal (0.05 < p ≤ 0.20)" if pf_p <= 0.20 else
               "NO edge — consistent with random / overfit")
    print(f"\n  VERDICT: {verdict}")
    print(sep + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Monte Carlo Permutation Test for the breakout strategy.")
    ap.add_argument("tickers", nargs="*", default=["AAPL", "MSFT", "NVDA"],
                    help="Tickers to test (default: AAPL MSFT NVDA)")
    ap.add_argument("--perms", type=int, default=200, help="Number of permutations")
    ap.add_argument("--hold",  type=int, default=20,  help="Forward hold (days)")
    ap.add_argument("--days",  type=int, default=120, help="Signal lookback window (days)")
    ap.add_argument("--min-prob", type=int, default=50, help="Min breakout probability to take a signal")
    args = ap.parse_args()

    tickers = [t.upper() for t in (args.tickers or ["AAPL", "MSFT", "NVDA"])]
    print(f"Running MCPT on {tickers} with {args.perms} permutations...")

    def _cb(i, total, m):
        if i % max(total // 20, 1) == 0 or i == total:
            print(f"  permutation {i}/{total}  (last null PF={m['profit_factor']}, n={m['n']})")

    res = run_mcpt(tickers, n_perm=args.perms, hold=args.hold, days=args.days,
                   min_prob=args.min_prob, progress=_cb)
    print_report(res)
