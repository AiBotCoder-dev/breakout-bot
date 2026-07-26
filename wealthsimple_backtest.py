"""
wealthsimple_backtest.py — Is the trend-filtered rotation viable on WEALTHSIMPLE?

Two real-world constraints the user set:
  1. Every vehicle must be tradeable on Wealthsimple.
  2. Wealthsimple's per-trade cost must be modeled so paper == real.

Wealthsimple reality (2026):
  * $0 commission on stocks/ETFs.
  * 1.5% currency-conversion fee on US-dollar securities EACH way (~3% round trip)
    on the basic plan. Brutal for a weekly-rotation strategy.
  * CAD-listed ETFs (TSX) have NO FX fee. Canada's leveraged ETFs are all 2x
    (Global X / ex-Horizons BetaPro) — which matches our ~1.5-2x leverage optimum.

So this backtests the rotation on CAD-listed 2x ETFs (real .TO price history, so
MER + decay are already inside the prices) with a realistic cost model, and puts
it head-to-head with the SAME rotation on US 3x ETFs paying the 1.5% FX each side.
The point: prove the CAD 2x version is the one that survives real fees.

CAD 2x universe (Wealthsimple-tradeable, no FX):
  HQU 2x Nasdaq-100 · HSU 2x S&P500 · HXU 2x TSX60 ·
  HEU 2x energy · HFU 2x financials · HGU 2x gold miners
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
US_3X = ["TQQQ", "SOXL", "SPXL", "FAS", "CURE", "FNGU"]   # rough sector analogues

# cost per REBALANCE side, as a fraction of traded notional
SPREAD_BPS = 8.0          # bid/ask spread on a leveraged ETF (~0.08%/side)
FX_BPS_US  = 150.0        # Wealthsimple 1.5% currency conversion, US securities only


def _load(t):
    end = datetime.now(); start = end - timedelta(days=int(YEARS * 365.25) + 260)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty or len(raw) < 300:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy(); c = df["Close"]
    df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
    df["mom"] = c / c.shift(MOM) - 1
    return df.dropna()


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


def _rotation(data, cal, cost_side_bps):
    """Weekly top-2 uptrend momentum rotation with a per-rebalance turnover cost.
    Returns (daily_returns, in_cash_frac, n_rebalances)."""
    dr = []; holdings = []; in_cash = 0; days = 0; rebals = 0
    for k in range(210, len(cal)):
        d = cal[k]
        rebal_cost = 0.0
        if k % 5 == 0:
            ranked = []
            for t, df in data.items():
                if d in df.index:
                    i = df.index.get_loc(d)
                    if i > 200:
                        c = df["Close"].values; s50 = df["sma50"].values
                        s200 = df["sma200"].values; m = df["mom"].values
                        if c[i] > s50[i] and c[i] > s200[i] and m[i] == m[i]:
                            ranked.append((m[i], t))
            ranked.sort(reverse=True)
            new_hold = [t for _, t in ranked[:TOP_K]]
            # turnover = names that changed (sell old + buy new); cost both sides
            changed = set(new_hold).symmetric_difference(set(holdings))
            if changed:
                # fraction of book turned over ~ changed/​max(len,1); charge cost per side
                turn = len(changed) / max(len(new_hold) or 1, 1)
                rebal_cost = turn * (cost_side_bps / 10_000)
                rebals += 1
            holdings = new_hold
        if not holdings:
            r = -rebal_cost; in_cash += 1
        else:
            rs = []
            for t in holdings:
                df = data[t]
                if d in df.index:
                    i = df.index.get_loc(d)
                    if i > 0:
                        rs.append(df["Close"].values[i] / df["Close"].values[i - 1] - 1)
            r = (float(np.mean(rs)) if rs else 0.0) - rebal_cost
        dr.append(r); days += 1
    return dr, (in_cash / days if days else 0.0), rebals


def _curve(dr):
    eq = 1.0; c = []
    for r in dr:
        eq *= (1 + r); c.append(eq)
    return eq, c


def _line(name, dr, cash=None, rebals=None):
    eq, curve = _curve(dr)
    extra = ""
    if cash is not None:
        extra += f"  in-cash {100*cash:.0f}%"
    if rebals is not None:
        extra += f"  rebals {rebals}"
    print(f"  {name:34s} total {(eq-1)*100:+8.0f}%  CAGR {_cagr(eq,len(dr)):+6.1f}%  "
          f"maxDD {_maxdd(curve):6.0f}%  Sortino {_sortino(dr):+.2f}{extra}")
    return eq, curve


def run():
    print(f"Loading CAD 2x + US 3x + benchmarks, {YEARS}y...")
    data = {}
    for t in CAD_2X + US_3X + ["QQQ", "XIU.TO"]:
        d = _load(t)
        if d is not None:
            data[t] = d
        print(f"    {t:8s} {'OK  bars='+str(len(d)) if d is not None else 'NO DATA'}")
    cad = {t: data[t] for t in CAD_2X if t in data}
    us = {t: data[t] for t in US_3X if t in data}
    if not cad:
        print("\n  FATAL: no CAD 2x data loaded (check .TO tickers)."); return
    # shared calendar: use a CAD ETF with the longest history
    base = max(cad.values(), key=lambda x: len(x))
    cal = base.index

    print(f"\n  CAD 2x universe: {sorted(cad)} ({len(cad)})")
    print(f"  US 3x universe : {sorted(us)} ({len(us)})\n")

    print("=" * 96)
    print(" WEALTHSIMPLE-REALISTIC ROTATION  (weekly top-2, trend-filtered, real fees)")
    print("=" * 96)
    # CAD 2x: spread only, no FX
    dr_cad, c_cad, r_cad = _rotation(cad, cal, SPREAD_BPS)
    _line("CAD 2x rotation (WS, no FX)", dr_cad, c_cad, r_cad)
    # US 3x on Wealthsimple basic: spread + 1.5% FX each side
    if us:
        dr_us, c_us, r_us = _rotation(us, cal, SPREAD_BPS + FX_BPS_US)
        _line("US 3x rotation (WS basic +1.5% FX)", dr_us, c_us, r_us)
        # US 3x if FX is waived (Premium/USD acct): spread only
        dr_usp, c_usp, r_usp = _rotation(us, cal, SPREAD_BPS)
        _line("US 3x rotation (WS Premium, no FX)", dr_usp, c_usp, r_usp)
    print("  " + "-" * 92)
    # benchmarks (buy-hold)
    def _bh(t):
        if t not in data:
            return
        c = data[t]["Close"]; sub = c[c.index >= cal[210]]
        _line(f"{t} buy&hold", sub.pct_change().dropna().tolist())
    _bh("HQU.TO"); _bh("QQQ"); _bh("XIU.TO")

    print("\n" + "=" * 96)
    print(" READ")
    print("=" * 96)
    print("  The CAD 2x line pays only the bid/ask spread — this is what you can ACTUALLY")
    print("  replicate on Wealthsimple. Compare it to the US-3x-with-FX line: the 1.5%")
    print("  currency fee each rebalance is what makes US 3x unviable on the basic plan.")
    print("  If CAD 2x still beats HQU/QQQ buy-hold on Sortino, the edge survives REAL fees.")


if __name__ == "__main__":
    run()
