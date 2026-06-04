"""
panic_backtest.py — Backtest the psychology of market panic.

THE QUESTION
------------
"When humans panic and the market overshoots fundamentals, do prices systematically
revert — and if so, by how much, how fast, and what does the signature look like?"

WHAT IT DOES
------------
Pulls 12 years of SPY + VIX daily data. Identifies "panic days" via four
objective criteria. For each panic day, measures forward returns at 1/5/10/20/60
days. Cross-references with named historical events (COVID, 2018 Powell, 2011
debt downgrade, etc.) and reports what actually happened.

The output answers, with real numbers:
  • After a VIX > 30 close, what's SPY's avg forward return 20 days later?
  • After a single-day SPY drop > 3%, does it bounce or keep falling?
  • Is "buy when others are fearful" actually profitable, and at what frequency?
  • Where in the panic does the bottom typically form — at peak VIX, or after?
"""

from __future__ import annotations

import math
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


# Named historical events for cross-reference (one date per event, the actual
# panic peak). Most are widely documented; the bot will pull SPY/VIX context
# around each.
NAMED_EVENTS = [
    ("2011-08-08", "S&P US debt downgrade panic"),
    ("2015-08-24", "China devaluation flash crash"),
    ("2018-02-05", "Volmageddon (XIV blew up)"),
    ("2018-12-24", "Powell-pivot bottom"),
    ("2020-03-16", "COVID waterfall day"),
    ("2020-03-23", "COVID bottom"),
    ("2022-09-30", "UK gilts crisis"),
    ("2023-03-13", "SVB bank run weekend"),
    ("2024-08-05", "Yen carry-trade unwind"),
    ("2025-04-04", "Tariff-shock panic"),
]

FORWARD_WINDOWS = [1, 5, 10, 20, 60]
HISTORY_YEARS   = 12


# ==============================================================================
# DATA
# ==============================================================================
def load_data():
    """Pull aligned daily SPY + ^VIX. Compute returns, drawdown, RSI."""
    end = datetime.now()
    start = end - timedelta(days=int(HISTORY_YEARS * 365.25))

    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    if spy is None or spy.empty or vix is None or vix.empty:
        raise RuntimeError("data download failed")

    for df in (spy, vix):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df = pd.DataFrame({
        "spy":     spy["Close"],
        "spy_hi":  spy["High"],
        "spy_lo":  spy["Low"],
        "spy_vol": spy["Volume"],
        "vix":     vix["Close"],
    }).dropna()

    df["spy_ret_1d"]  = df["spy"].pct_change()
    df["spy_ret_5d"]  = df["spy"].pct_change(5)
    df["vix_ret_1d"]  = df["vix"].pct_change()
    df["spy_high_252"] = df["spy"].rolling(252, min_periods=20).max()
    df["spy_drawdown_pct"] = (df["spy"] / df["spy_high_252"] - 1) * 100
    df["spy_sma200"] = df["spy"].rolling(200, min_periods=50).mean()

    # 14-day RSI on SPY
    delta = df["spy"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    df["spy_rsi14"] = 100 - 100 / (1 + rs)

    # Volume Z-score (vs 60-day mean)
    df["vol_z"] = ((df["spy_vol"] - df["spy_vol"].rolling(60).mean())
                   / df["spy_vol"].rolling(60).std())

    return df.dropna()


# ==============================================================================
# PANIC IDENTIFICATION CRITERIA
# ==============================================================================
CRITERIA = {
    "VIX_ABS_30":     lambda r: r["vix"] >= 30,                 # absolute fear
    "VIX_ABS_40":     lambda r: r["vix"] >= 40,                 # extreme fear
    "VIX_SPIKE_25":   lambda r: r["vix_ret_1d"] >= 0.25,        # sudden fear
    "VIX_SPIKE_40":   lambda r: r["vix_ret_1d"] >= 0.40,        # sudden extreme
    "SPY_DROP_3":     lambda r: r["spy_ret_1d"] <= -0.03,       # one-day crash
    "SPY_DROP_5":     lambda r: r["spy_ret_1d"] <= -0.05,       # one-day capitulation
    "RSI_OVERSOLD":   lambda r: r["spy_rsi14"] <= 25,           # technical capitulation
    "VOL_FLOOD":      lambda r: r["vol_z"] >= 3.0,              # capitulation volume
    "COMBO_HARD":     lambda r: (r["vix"] >= 30 and r["spy_ret_1d"] <= -0.02
                                  and r["spy_rsi14"] <= 35),   # textbook panic
}


def find_panic_days(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return rows where the named criterion fires."""
    mask = df.apply(CRITERIA[name], axis=1)
    return df.loc[mask].copy()


def measure_forward(df: pd.DataFrame, panic_idx: pd.DatetimeIndex,
                    windows: list[int]) -> pd.DataFrame:
    """For each panic day, return forward % returns at each window."""
    rows = []
    spy = df["spy"]
    for d in panic_idx:
        try:
            i = df.index.get_loc(d)
        except KeyError:
            continue
        if isinstance(i, slice):
            i = i.start
        base = spy.iloc[i]
        row = {"date": d.date()}
        for w in windows:
            j = i + w
            if j >= len(spy):
                row[f"fwd_{w}d_pct"] = np.nan
                continue
            row[f"fwd_{w}d_pct"] = (spy.iloc[j] / base - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(fwd: pd.DataFrame, windows: list[int]) -> dict:
    """Compute win rate + mean + median + extremes for each forward window."""
    out = {"n": int(len(fwd))}
    for w in windows:
        col = f"fwd_{w}d_pct"
        s = fwd[col].dropna()
        if s.empty:
            out[f"{w}d"] = None
            continue
        out[f"{w}d"] = {
            "mean":   round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "wins":   int((s > 0).sum()),
            "losses": int((s < 0).sum()),
            "win_rate": round(float((s > 0).mean() * 100), 1),
            "p10":    round(float(s.quantile(0.10)), 2),
            "p90":    round(float(s.quantile(0.90)), 2),
            "best":   round(float(s.max()), 2),
            "worst":  round(float(s.min()), 2),
        }
    return out


# ==============================================================================
# REPORTING
# ==============================================================================
def print_criterion(df: pd.DataFrame, name: str, windows: list[int]):
    rows = find_panic_days(df, name)
    if rows.empty:
        print(f"\n  {name}: 0 events found in {HISTORY_YEARS}y window")
        return
    fwd = measure_forward(df, rows.index, windows)
    summ = summarize(fwd, windows)
    print(f"\n  -- {name}  ({summ['n']} historical days)")
    print(f"    {'Window':>7}  {'Win%':>6}  {'Mean%':>7}  {'Median%':>8}  "
          f"{'p10':>7}  {'p90':>7}  {'Best%':>7}  {'Worst%':>7}")
    for w in windows:
        s = summ.get(f"{w}d")
        if not s:
            continue
        print(f"    {w:>4}d   {s['win_rate']:>5.1f}%  {s['mean']:>+6.2f}%  "
              f"{s['median']:>+7.2f}%  {s['p10']:>+6.2f}%  {s['p90']:>+6.2f}%  "
              f"{s['best']:>+6.2f}%  {s['worst']:>+6.2f}%")


def print_named_events(df: pd.DataFrame, windows: list[int]):
    print(f"\n\n  ==========================================================")
    print(f"   NAMED HISTORICAL PANICS — context + forward returns")
    print(f"  ==========================================================")
    rows = []
    for dt_str, label in NAMED_EVENTS:
        try:
            dt = pd.Timestamp(dt_str)
            i_pos = df.index.get_indexer([dt], method="nearest")[0]
            if i_pos < 0 or i_pos >= len(df):
                continue
            r = df.iloc[i_pos]
        except Exception:
            continue
        spy_now = float(r["spy"]); vix_now = float(r["vix"])
        dd = float(r["spy_drawdown_pct"]); rsi = float(r["spy_rsi14"])
        ret1 = float(r["spy_ret_1d"]) * 100
        print(f"\n  {dt_str}  {label}")
        print(f"    Context: SPY ${spy_now:.2f}  drop {ret1:+.1f}%  VIX {vix_now:.1f}  "
              f"drawdown {dd:+.1f}%  RSI {rsi:.0f}")
        fwd_row = {}
        for w in windows:
            j = i_pos + w
            if j < len(df):
                fwd_row[w] = (float(df["spy"].iloc[j]) / spy_now - 1) * 100
        if fwd_row:
            txt = "  ".join(f"+{w}d: {p:+.1f}%" for w, p in fwd_row.items())
            print(f"    Forward: {txt}")


if __name__ == "__main__":
    print(f"Loading {HISTORY_YEARS}y of SPY + VIX data...")
    df = load_data()
    print(f"  {len(df)} trading days  "
          f"({df.index[0].date()} to {df.index[-1].date()})")

    print(f"\n{'='*68}\n  PANIC CRITERIA — FORWARD RETURN DISTRIBUTIONS\n{'='*68}")
    for c in CRITERIA:
        print_criterion(df, c, FORWARD_WINDOWS)

    print_named_events(df, FORWARD_WINDOWS)
