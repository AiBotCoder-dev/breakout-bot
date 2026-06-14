"""
exhaustion_reversal.py — Live "max-profit reversal at exhausted highs" PUT scanner.

VALIDATED (exhaustion_reversal_v2.py, 53 names, 9y, managed exit):
  Setup: a stock PEAKED at/near its 252-day high while EXTENDED (>=12% above the
  50-SMA) with EXHAUSTION visible at the peak (bearish RSI divergence OR drying
  volume), then DECISIVELY breaks down (close < 10-day low on a down day), caught
  early (still within ~12% of the peak).
  Result: 60% win, +1.49%/trade, PF 1.92, MCPT p=0.0132. RARE (~2/yr/universe).

OPTION "MAX PROFIT" (Black-Scholes, the user's IV thesis):
  At a complacent high IV is low; the reversal expands IV hard, so a slightly-OTM
  put gains on BOTH delta and vega. Modeled: WIN ~+218% (IV 30->48 on -5.2%),
  LOSS ~-70% -> blended option expectancy ~+103%/trade (conservative ~+53%).
  => prefer LOW IV-rank entries (cheap vol with room to expand).

This is a SECONDARY, rare, high-conviction PUT signal (small sample n=20 — the
forward live record is the real test). Regime-gated like the other put engines.
"""

from __future__ import annotations

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except Exception:                       # pragma: no cover
    np = pd = yf = None

# Names that actually crash hard enough for the put to pay (high crash-fuel) +
# liquid large caps that still produce clean exhaustion reversals.
UNIVERSE = [
    "COIN","MSTR","CVNA","SMCI","ARM","AFRM","UPST","RIVN","LCID","PLTR","TSLA",
    "NVDA","SNAP","ROKU","DKNG","RBLX","ENPH","NET","CRWD","SHOP","MARA","RIOT",
    "SOFI","HOOD","DDOG","NFLX","AMD","MU","AAPL","MSFT","GOOGL","AMZN","META",
    "AVGO","QCOM","ORCL","CAT","BA","NKE","DIS",
]
PEAK_LB = 10
BACKTEST_STATS = {"win": 60.0, "expectancy_pct": 1.49, "profit_factor": 1.92,
                  "mcpt_p": 0.0132, "n": 20, "opt_expectancy_pct": 103,
                  "freq": "~2/yr/universe"}


def _rsi(close, period=14):
    d = np.diff(close, prepend=close[0]); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = np.full_like(close, 50.0, dtype=float); ag = al = 0.0
    for i in range(1, len(close)):
        if i <= period:
            ag = (ag*(i-1)+g[i])/i; al = (al*(i-1)+l[i])/i
        else:
            ag = (ag*(period-1)+g[i])/period; al = (al*(period-1)+l[i])/period
        out[i] = 100.0 if al < 1e-12 else 100.0 - 100.0/(1.0+ag/al)
    return out


def _load(t):
    if yf is None:
        return None
    try:
        raw = yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 300:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.dropna(subset=["Close"]).copy()
        df["rsi"] = _rsi(df["Close"].values, 14)
        df["sma50"] = df["Close"].rolling(50).mean()
        df["vol20"] = df["Volume"].rolling(20).mean()
        df["hi252"] = df["High"].rolling(252).max()
        return df.dropna()
    except Exception:
        return None


def classify(df) -> dict | None:
    """Return the live exhaustion-reversal PUT setup for one name, or None.
    Mirrors the validated V2 signal at the latest bar."""
    if df is None or len(df) < 270:
        return None
    C = df["Close"].values; O = df["Open"].values; H = df["High"].values; L = df["Low"].values
    rsi = df["rsi"].values; s50 = df["sma50"].values; vol = df["Volume"].values
    v20 = df["vol20"].values; hi = df["hi252"].values
    i = len(C) - 1

    recent_peak = float(H[i-PEAK_LB:i].max())
    if not (hi[i] > 0 and recent_peak >= 0.995 * hi[i-PEAK_LB]):
        return None
    peak_j = i - PEAK_LB + int(np.argmax(H[i-PEAK_LB:i]))
    if not (s50[peak_j] > 0 and C[peak_j] >= 1.12 * s50[peak_j]):
        return None
    rsi_lh = rsi[peak_j] < rsi[max(0, peak_j-15):peak_j].max() - 3
    vol_dry = v20[peak_j] > 0 and vol[peak_j-5:peak_j].mean() < 1.0 * v20[peak_j]
    if not (rsi_lh or vol_dry):
        return None
    if not (C[i] < L[i-10:i].min() and C[i] < O[i]):
        return None
    price = float(C[i])
    if price < 0.88 * recent_peak:           # too late — already crashed
        return None

    swing_hi = float(H[max(0, i-5):i+1].max())
    stop = round(swing_hi * 1.005, 2)        # just above the peak/swing high
    risk = (stop - price) / price if price > 0 else 9
    target = round(price * (1 - 2.5 * risk), 2)
    return {
        "price": round(price, 2),
        "peak": round(recent_peak, 2),
        "off_peak_pct": round((price/recent_peak - 1) * 100, 1),
        "rsi": round(float(rsi[i]), 1),
        "exhaustion": "RSI divergence" if rsi_lh else "volume fade",
        "stop": stop, "target": target,
        "risk_pct": round(risk * 100, 1),
        "entry": round(price, 2),
        "direction": "put",
    }


class ExhaustionReversal:
    def __init__(self, conn=None, universe=None):
        self.conn = conn
        self.universe = [t for t in (universe or UNIVERSE) if "." not in t]

    def scan(self, progress=None) -> list:
        out = []
        def _one(t):
            return (t, classify(_load(t)))
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_one, t): t for t in self.universe}
            done = 0
            for fut in as_completed(futs):
                done += 1
                if progress:
                    try: progress(done, len(futs), futs[fut])
                    except Exception: pass
                t, c = fut.result()
                if c:
                    out.append({"ticker": t, **c})
        out.sort(key=lambda r: r["off_peak_pct"])   # freshest breaks first
        return out


if __name__ == "__main__":
    print("Scanning for exhaustion-reversal PUT setups (rare — usually empty)...")
    print(f"(edge: {BACKTEST_STATS['win']}% win, PF {BACKTEST_STATS['profit_factor']}, "
          f"p={BACKTEST_STATS['mcpt_p']}, opt expectancy ~+{BACKTEST_STATS['opt_expectancy_pct']}%)\n")
    res = ExhaustionReversal().scan()
    if not res:
        print("No exhaustion-reversal setups right now (expected — ~2/yr).")
    for r in res:
        print(f"  {r['ticker']:6s} ${r['price']:>8.2f}  {r['off_peak_pct']:+.1f}% off peak "
              f"${r['peak']:.2f}  RSI {r['rsi']:.0f}  ({r['exhaustion']})  "
              f"stop ${r['stop']:.2f} target ${r['target']:.2f}")
