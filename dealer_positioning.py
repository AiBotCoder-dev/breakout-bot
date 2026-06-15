"""
dealer_positioning.py — "Follow the market makers" via options hedging footprints.

When dealers sell options they MUST hedge in the underlying, so big open-interest
strikes act as magnets / walls and price tends to gravitate toward "max pain"
(the strike where the most option premium expires worthless) — especially NEAR
EXPIRY. This is the one sophisticated institutional-flow signal that's computable
from FREE option-chain open interest. It is NOT a precise predictor — it's a
probabilistic context read, strongest the closer to expiry and the higher the OI.

Computes (nearest weekly expiry):
  • max_pain        — the dealer "pin" target (gravity, mostly near expiry)
  • call_wall       — biggest call OI strike above spot = resistance (dealers sell
                      into rallies there to stay hedged)
  • put_wall        — biggest put OI strike below spot = support
  • pc_oi_ratio     — put/call OI; >1 = hedged/defensive (often contrarian bullish),
                      <0.7 = complacent (little downside protection — fragile)

HONEST LIMITS: retail can't see the true dealer gamma SIGN (that needs paid
positioning data); OI-based max pain is an approximation with modest, mostly
near-expiry pull. Use as CONTEXT, not a trade trigger (backtest before trading on it).
"""

from __future__ import annotations

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except Exception:                       # pragma: no cover
    np = pd = yf = None


def get_dealer_positioning(ticker: str) -> dict:
    out = {"ok": False, "signals": [], "max_pain": None, "max_pain_gap_pct": None,
           "call_wall": None, "put_wall": None, "pc_oi_ratio": None, "expiry": None}
    if yf is None:
        return out
    try:
        tk = yf.Ticker(ticker)
        spot = float(tk.fast_info["last_price"])
        exps = list(tk.options or [])
        if not exps or spot <= 0:
            return out
        expiry = exps[0]
        ch = tk.option_chain(expiry)
        calls, puts = ch.calls, ch.puts
        if calls is None or puts is None or calls.empty or puts.empty:
            return out
    except Exception:
        return out

    out["expiry"] = expiry
    coi = float(calls["openInterest"].fillna(0).sum())
    poi = float(puts["openInterest"].fillna(0).sum())
    sig = []

    # max pain
    try:
        strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
        def _pain(K):
            cp = ((K - calls["strike"]).clip(lower=0) * calls["openInterest"].fillna(0)).sum()
            pp = ((puts["strike"] - K).clip(lower=0) * puts["openInterest"].fillna(0)).sum()
            return cp + pp
        mp = float(min(strikes, key=_pain))
        out["max_pain"] = round(mp, 2)
        gap = (mp / spot - 1) * 100
        out["max_pain_gap_pct"] = round(gap, 1)
        if abs(gap) >= 1.5:
            pull = "DOWN toward" if gap < 0 else "UP toward"
            sig.append(f"Max-pain pin at ${mp:.0f} ({gap:+.1f}%) — mild dealer gravity "
                       f"{pull} it into the {expiry} expiry")
    except Exception:
        pass

    # walls
    try:
        cabove = calls[calls["strike"] >= spot]
        if not cabove.empty:
            cw = cabove.loc[cabove["openInterest"].idxmax()]
            out["call_wall"] = round(float(cw["strike"]), 2)
            sig.append(f"Call wall ${float(cw['strike']):.0f} (OI {int(cw['openInterest'] or 0)}) "
                       f"— dealer-hedging RESISTANCE")
        pbelow = puts[puts["strike"] <= spot]
        if not pbelow.empty:
            pw = pbelow.loc[pbelow["openInterest"].idxmax()]
            out["put_wall"] = round(float(pw["strike"]), 2)
            sig.append(f"Put wall ${float(pw['strike']):.0f} (OI {int(pw['openInterest'] or 0)}) "
                       f"— dealer-hedging SUPPORT")
    except Exception:
        pass

    # positioning ratio
    if coi > 0:
        pcr = poi / coi
        out["pc_oi_ratio"] = round(pcr, 2)
        if pcr >= 1.2:
            sig.append(f"Put/Call OI {pcr:.2f} — heavily hedged/defensive (crowded; "
                       f"often a contrarian floor)")
        elif pcr <= 0.6:
            sig.append(f"Put/Call OI {pcr:.2f} — complacent, little downside protection "
                       f"(fragile to a shock)")

    out["ok"] = bool(sig)
    out["signals"] = sig
    return out


if __name__ == "__main__":
    for t in ["NVDA", "SPY", "TSLA", "SOFI"]:
        d = get_dealer_positioning(t)
        print(f"\n{t}: max_pain={d['max_pain']} ({d['max_pain_gap_pct']}%) "
              f"call_wall={d['call_wall']} put_wall={d['put_wall']} P/C={d['pc_oi_ratio']}")
        for s in d["signals"]:
            print(f"   • {s}")
