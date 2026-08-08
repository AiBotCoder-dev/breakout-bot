"""
premium_selling_backtest.py — Is SELLING premium (the right side of the variance
risk premium) profitable, conditioned on IV rank?

WHY: the audit showed the bot loses on EVERY slice as a net options BUYER. The
structural reason is that implied vol systematically exceeds subsequent realized
vol — buyers pay that premium, sellers collect it. This tests being on the other
side, with DEFINED risk.

━━ THE METHODOLOGICAL PROBLEM (read this before trusting any number) ━━
We have no real IV history — only prices. If we price options off trailing realized
vol, we ASSUME AWAY the variance risk premium itself, so the test cannot "prove" the
VRP. What it CAN honestly measure is the vol-MEAN-REVERSION component: sell when
trailing vol is historically high, and if vol subsequently falls, the option was
overpriced relative to what actually happened.
Because real IV usually sits ABOVE trailing RV, modelling IV = RV x mult with
mult<=1 means we collect LESS premium than reality -> the test is CONSERVATIVE.
We therefore SWEEP the IV multiplier (0.90 / 1.00 / 1.10) to show sensitivity
rather than hide behind one assumption.

STRUCTURES (defined risk, comparable on return-on-capital-at-risk):
  PUT CREDIT SPREAD — sell 5% OTM put, buy 10% OTM put (bullish/neutral)
  IRON CONDOR       — sell 5% OTM put+call, buy 10% OTM wings (neutral)

CONDITION: IV rank = percentile of today's RV20 within its trailing 252 days.
Test rank>0 (always), >50 (elevated), >70 (rich) to see if selectivity matters.

EXITS: (a) hold to expiry, (b) manage at +50% of max profit (the standard rule).

Reports win%, expectancy on capital-at-risk, and — most importantly for a premium
seller — the LEFT TAIL: worst trade, % of trades worse than -100%, and the equity
curve max drawdown. A high win rate with catastrophic tails is a ruin machine.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
            "AMD", "TSLA", "JPM", "XLE", "XLF", "AVGO", "NFLX", "MU", "COIN",
            "UBER", "DIS", "BA", "CAT", "WMT", "COST", "PLTR"]
DTE0 = 30            # sell 30 DTE
HOLD_MAX = 21        # exit by 21 days held (9 DTE left) if not managed earlier
SHORT_OTM = 0.05     # short strike 5% OTM
LONG_OTM = 0.10      # long wing 10% OTM
R = 0.04
_N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
HALF_SPREAD_PCT = 0.03; MIN_HALF_SPREAD = 0.02; COMM = 0.0065


def bs(S, K, T, sig, put=False):
    if T <= 0 or sig <= 0:
        return max((K - S) if put else (S - K), 0.0)
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return (K * math.exp(-R * T) * _N(-d2) - S * _N(-d1)) if put \
        else (S * _N(d1) - K * math.exp(-R * T) * _N(d2))


def _hs(m): return max(m * HALF_SPREAD_PCT, MIN_HALF_SPREAD)
def sell_at(m): return max(0.0, m - _hs(m) - COMM)    # we receive
def buy_at(m):  return m + _hs(m) + COMM              # we pay


def _load(t):
    end = datetime.now(); start = end - timedelta(days=int(7 * 365.25) + 200)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty or len(raw) < 400:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy(); c = df["Close"]
    df["rv20"] = np.log(c / c.shift()).rolling(20).std() * np.sqrt(252)
    df["ivrank"] = df["rv20"].rolling(252).rank(pct=True) * 100
    return df.dropna()


def _trade(C, i, rv, iv_mult, condor, manage_50):
    """Open a credit structure at bar i; walk forward. Returns pct return on
    capital-at-risk, or None if unpriceable."""
    S0 = C[i]
    iv = max(0.10, min(2.5, rv * iv_mult))
    Kps, Kpl = S0 * (1 - SHORT_OTM), S0 * (1 - LONG_OTM)
    Kcs, Kcl = S0 * (1 + SHORT_OTM), S0 * (1 + LONG_OTM)

    # open: sell shorts (receive bid), buy wings (pay ask)
    credit = sell_at(bs(S0, Kps, DTE0/365, iv, True)) - buy_at(bs(S0, Kpl, DTE0/365, iv, True))
    width = Kps - Kpl
    if condor:
        credit += sell_at(bs(S0, Kcs, DTE0/365, iv, False)) - buy_at(bs(S0, Kcl, DTE0/365, iv, False))
        width = max(width, Kcl - Kcs)      # one side can lose at a time
    if credit <= 0.05:
        return None
    risk = width - credit
    if risk <= 0.05:
        return None

    for d in range(1, HOLD_MAX + 1):
        if i + d >= len(C):
            return None
        S = C[i + d]; T = (DTE0 - d) / 365
        # cost to close now: buy back shorts (pay ask), sell wings (receive bid)
        cost = buy_at(bs(S, Kps, T, iv, True)) - sell_at(bs(S, Kpl, T, iv, True))
        if condor:
            cost += buy_at(bs(S, Kcs, T, iv, False)) - sell_at(bs(S, Kcl, T, iv, False))
        pnl = credit - cost
        if manage_50 and pnl >= 0.50 * credit:
            return pnl / risk * 100
    # exit at HOLD_MAX
    S = C[i + HOLD_MAX]; T = (DTE0 - HOLD_MAX) / 365
    cost = buy_at(bs(S, Kps, T, iv, True)) - sell_at(bs(S, Kpl, T, iv, True))
    if condor:
        cost += buy_at(bs(S, Kcs, T, iv, False)) - sell_at(bs(S, Kcl, T, iv, False))
    return (credit - cost) / risk * 100


def _run(data, iv_mult, rank_min, condor, manage_50):
    out = []
    for t, df in data.items():
        C = df["Close"].values; rv = df["rv20"].values; rk = df["ivrank"].values
        n = len(C); nxt = -99
        for i in range(260, n - HOLD_MAX - 1):
            if i < nxt or rk[i] < rank_min:
                continue
            r = _trade(C, i, rv[i], iv_mult, condor, manage_50)
            if r is None:
                continue
            out.append(r); nxt = i + HOLD_MAX
    return np.array(out)


def _stats(a):
    if len(a) == 0:
        return None
    eq = np.cumsum(a / 100.0); peak = np.maximum.accumulate(eq)
    dd = (eq - peak).min()
    dn = a[a < 0]
    sortino = a.mean() / (np.sqrt(np.mean(dn ** 2)) if len(dn) else 1e-9)
    return {"n": len(a), "win": 100 * (a > 0).mean(), "exp": a.mean(),
            "med": np.median(a), "worst": a.min(),
            "tail": 100 * (a <= -100).mean(), "dd": dd * 100, "sortino": sortino}


def _line(label, a):
    s = _stats(a)
    if not s:
        print(f"  {label:34s} n=0"); return
    print(f"  {label:34s} n={s['n']:<5} win {s['win']:3.0f}%  exp {s['exp']:+6.2f}%  "
          f"med {s['med']:+6.1f}%  worst {s['worst']:+7.0f}%  <-100%: {s['tail']:3.0f}%  "
          f"Sortino {s['sortino']:+.2f}  maxDD {s['dd']:+.0f}u")


def run():
    print(f"Loading {len(UNIVERSE)} names, 7y...")
    data = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_load, t): t for t in UNIVERSE}
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                data[futs[f]] = r
    print(f"  loaded {len(data)}\n")

    print("=" * 118)
    print(" A) IV-RANK SELECTIVITY — put credit spread, hold to 21d  (IV = RV x 1.00)")
    print("=" * 118)
    for rk in (0, 50, 70):
        _line(f"sell always (rank>{rk})" if rk == 0 else f"sell when IV rank > {rk}",
              _run(data, 1.00, rk, condor=False, manage_50=False))

    print("\n" + "=" * 118)
    print(" B) MANAGING WINNERS AT +50% vs HOLDING  (IV rank > 50)")
    print("=" * 118)
    _line("credit spread · hold to 21d", _run(data, 1.00, 50, False, False))
    _line("credit spread · manage at +50%", _run(data, 1.00, 50, False, True))
    _line("iron condor  · hold to 21d", _run(data, 1.00, 50, True, False))
    _line("iron condor  · manage at +50%", _run(data, 1.00, 50, True, True))

    print("\n" + "=" * 118)
    print(" C) SENSITIVITY TO THE IV ASSUMPTION (the number we cannot observe)")
    print("=" * 118)
    for m in (0.90, 1.00, 1.10):
        _line(f"credit spread, manage +50%, IV = RV x {m:.2f}",
              _run(data, m, 50, False, True))
    print("\n  IV x 0.90 = we sell CHEAP (conservative: real IV is usually richer).")
    print("  IV x 1.10 ~ a realistic variance-risk-premium. If the edge only appears")
    print("  at 1.10, it IS the VRP and depends entirely on that premium being real.")
    print("\n  KEY RISK: premium selling wins often and loses big. Watch 'worst',")
    print("  '<-100%' (losses exceeding the capital at risk) and maxDD, not win%.")


if __name__ == "__main__":
    run()
