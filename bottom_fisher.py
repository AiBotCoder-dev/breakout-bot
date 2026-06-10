"""
bottom_fisher.py — Live "buy the validated bottom" scanner + option setups.

THE EDGE (validated, not vibes)
-------------------------------
bottom_fisher_backtest.py tested three early-entry / dip-buying modes across 88
liquid names over 6 years with a realistic tight-stop exit and MCPT:

  OVERSOLD_SUPPORT  ->  PF 1.71, n=539, expectancy +0.99%/trade, R:R 2.63,
                        fwd 5-10d bounce win ~59.5%,  MCPT p = 0.005  ✅ SHIP
  CAPITULATION_REV  ->  bigger payoff but n=88, p=0.30 (too rare) -> Panic Detector
  PULLBACK_UPTREND  ->  PF 1.25, no real fwd edge -> rejected

So this finder fires ONLY the validated OVERSOLD_SUPPORT setup:
  • RSI(14) < 30                         (genuinely oversold)
  • price within 6% of its 60-day low    (sitting on support, not mid-air)
  • today closes GREEN (> prior close)    (the bounce is starting — don't catch
                                           the falling knife, buy the first turn)

HONEST FRAMING (surfaced everywhere):
  • This is a SUB-50% win-rate edge (≈39% tradeable). It makes money on
    reward:risk (winners ≈2.6x losers), NOT on accuracy. There is no 80%
    foolproof bottom — the backtest proved it. Size it as a SECONDARY signal.
  • It enters EARLIER than momentum (buys the dip, not the breakout), which is
    exactly its purpose: better entry price, bigger upside if the bounce runs.
  • For options we hold for the bounce with the premium trailing stop (the tight
    underlying stop gets whipsawed — fwd win 59% >> tradeable 39%).
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

from momentum_strategy import LIQUID_UNIVERSE

# ── Validated parameters (mirror the backtest exactly) ────────────────────────
RSI_OVERSOLD     = 30.0
SUPPORT_BAND     = 0.06        # within 6% of the 60-day low
RR_TARGET        = 2.5         # reward:risk used for the suggested target
MAX_RISK         = 0.08        # skip if stop (swing low) is wider than 8%
BACKTEST_STATS   = {
    "mode": "OVERSOLD_SUPPORT", "n": 539, "profit_factor": 1.71,
    "expectancy_pct": 0.99, "reward_risk": 2.63, "fwd5d_win": 59.5,
    "mcpt_p": 0.005, "verdict": "STRONG real edge (p<=0.01)",
}


def _rsi(close: "np.ndarray", period: int = 14) -> "np.ndarray":
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
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
    if yf is None:
        return None
    try:
        raw = yf.download(t, period="1y", interval="1d", progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 80:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw.dropna(subset=["Close"]).copy()
    except Exception:
        return None


def classify(df) -> dict | None:
    """Return the live OVERSOLD_SUPPORT signal for one ticker, or None."""
    if df is None or len(df) < 70:
        return None
    C = df["Close"].values; L = df["Low"].values
    i = len(C) - 1
    price = float(C[i])
    if price <= 0:
        return None

    rsi = _rsi(C, 14)
    rsi_now = float(rsi[i])
    low60 = float(L[max(0, i - 60):i + 1].min())
    dist_from_low = (price / low60 - 1) if low60 > 0 else 9.9
    green_today = C[i] > C[i - 1]

    oversold = rsi_now < RSI_OVERSOLD
    at_support = 0 <= dist_from_low <= SUPPORT_BAND

    # Stage so the user can act with discipline (mirror reversal_finder UX):
    #   TRIGGERED — all three conditions true RIGHT NOW (the validated entry)
    #   WATCH     — oversold + at support but not yet green (knife still falling)
    if oversold and at_support and green_today:
        stage = "TRIGGERED"
    elif oversold and at_support:
        stage = "WATCH"
    else:
        return None

    # Tight stop at the recent 5-bar swing low (the discipline behind the R:R)
    swing_low = float(L[max(0, i - 5):i + 1].min())
    stop = round(swing_low * 0.999, 2)
    risk = (price - stop) / price if price > 0 else 9.9
    if risk > MAX_RISK or risk <= 0:
        return None
    target = round(price * (1 + RR_TARGET * risk), 2)

    return {
        "stage": stage,
        "price": round(price, 2),
        "rsi14": round(rsi_now, 1),
        "dist_from_60d_low_pct": round(dist_from_low * 100, 1),
        "stop": stop,
        "target": target,
        "risk_pct": round(risk * 100, 1),
        "reward_risk": RR_TARGET,
        "entry": round(price, 2),
    }


class BottomFisher:
    """Live scanner for the validated oversold-at-support bottom-buy setup."""

    def __init__(self, conn=None, universe=None):
        self.conn = conn
        self.universe = [t for t in (universe or LIQUID_UNIVERSE) if "." not in t]

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
        # TRIGGERED first (the validated entry), then WATCH; closest to support first
        order = {"TRIGGERED": 0, "WATCH": 1}
        out.sort(key=lambda r: (order.get(r["stage"], 9), r["dist_from_60d_low_pct"]))
        return out

    def find_option_setups(self, max_setups: int = 4, progress=None) -> list:
        """
        For each TRIGGERED underlying, pick a short-dated, near-the-money CALL
        (the bounce play) and return setups in the SAME dict shape the broker
        auto-entry consumes (ticker/expiry/strike/premium/contract_symbol/...).

        Near-the-money (0-8% OTM) — a snap-back bounce needs delta, not a far OTM
        lottery. Reuses momentum_options.select_call_contract gates (cost cap,
        liquidity, IV ceiling) so discipline is identical.
        """
        from momentum_options import select_call_contract, _earnings_within, EARNINGS_AVOID_DAYS
        triggered = [r for r in self.scan(progress=progress) if r["stage"] == "TRIGGERED"]
        setups = []
        for r in triggered:
            if len(setups) >= max_setups:
                break
            tk = r["ticker"]
            if _earnings_within(tk, EARNINGS_AVOID_DAYS):
                continue
            c = select_call_contract(tk, r["price"], otm_min=0.0, otm_max=0.08)
            if not c:
                continue
            # Quality score: discipline gate via options_analytics, thesis = the
            # expected bounce. Oversold names are high-vol; a modest snap-back is
            # a meaningful % move. Floor the thesis so thin moves are skipped.
            try:
                from options_analytics import options_trade_score
                thesis = 9.0   # expected bounce move (%); validated edge is a multi-% snap-back
                qs = options_trade_score(self.conn, tk, {**c, "iv": (c.get("iv") or 0)},
                                         thesis_move_pct=thesis)
                c["quality_score"] = qs["score"]
                c["quality_grade"] = qs["grade"]
                c["quality_decision"] = qs["decision"]
            except Exception:
                c["quality_score"] = None
                c["quality_grade"] = "?"
            c["option_type"] = "call"
            c["underlying_price"] = r["price"]
            c["signal"] = "bottom_fisher"
            c["rsi14"] = r["rsi14"]
            c["dist_from_low"] = r["dist_from_60d_low_pct"]
            c["mom_6m"] = None     # not a momentum trade
            c["mom_3m"] = None
            setups.append(c)
        return setups


if __name__ == "__main__":
    print("Scanning liquid universe for validated OVERSOLD-AT-SUPPORT bottoms...")
    print(f"(edge: PF {BACKTEST_STATS['profit_factor']}, n={BACKTEST_STATS['n']}, "
          f"MCPT p={BACKTEST_STATS['mcpt_p']})\n")
    res = BottomFisher().scan()
    by = {}
    for r in res:
        by.setdefault(r["stage"], []).append(r)
    for stage in ["TRIGGERED", "WATCH"]:
        rows = by.get(stage, [])
        print(f"\n=== {stage} ({len(rows)}) ===")
        for r in rows[:15]:
            print(f"  {r['ticker']:6s} ${r['price']:>8.2f}  RSI {r['rsi14']:>4.1f}  "
                  f"{r['dist_from_60d_low_pct']:>4.1f}% above 60d-low  "
                  f"stop ${r['stop']:.2f} (risk {r['risk_pct']:.1f}%)  "
                  f"target ${r['target']:.2f}")
    if not res:
        print("No oversold-at-support setups right now (market not handing you bottoms).")
