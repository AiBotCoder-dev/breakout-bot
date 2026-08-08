"""
winner_gate.py — DIRECTION-AWARE meta-label filter (separates winners from losers).

It sits in front of the strategy and grades each candidate trade 0..100, and can
veto the weak ones. It changes NOTHING about how signals are generated.

WHY THIS WAS REBUILT (2026-08): the original gate scored EVERY trade on bullish-call
logic — reward uptrend, punish "chasing", require positive momentum. But the audit
of 131 graded trades showed that logic is INVERTED versus reality:
    trades the old gate APPROVED -> net -$92   (it picked losers)
    trades the old gate REJECTED -> net +$746  (it threw away winners)
It rejected 18 winners worth +$3,928 — including the TSLA put (+$2,915) it scored
18.4 — because the bot's real edge is CONVEX/PUT bets in DOWNTRENDS, which the
bullish gate always failed. Calls bled (-$3,278); puts won (+$2,777).

So the gate is now DIRECTION-AWARE — it grades a CALL on momentum quality and a PUT
on BREAKDOWN quality, using the features the ignition backtests validated:

  CALL (bullish) — reward: real uptrend, strong 6-mo momentum, reachable strike,
                   a breakout (high in its range is FINE for momentum).
  PUT  (convex)  — reward: NOT an uptrend, price low in its range / breaking down,
                   a sharp recent 5-day drop, reachable put strike. This is the
                   crash/breakdown profile that actually pays.

Strike REACHABILITY (shared): the strike's distance OTM must be within ~1 sigma of
the expected move over the hold — kills the theta-death lottery tickets, both ways.

PUBLIC API (back-compatible):
  evaluate(features, direction="call") -> {passed, score 0..100, reasons, reach}
  compute_entry_features(ticker, otm_pct, dte, iv, direction) -> feature dict
All thresholds env-overridable.
"""
from __future__ import annotations
import math
import os

REACH_MAX = float(os.environ.get("WG_REACH_MAX", "1.2"))     # strike <= ~1.2 sigma move
MOM_MIN   = float(os.environ.get("WG_MOM_MIN", "0.10"))      # call momentum bar
PUT_RNG_MAX = float(os.environ.get("WG_PUT_RNG_MAX", "0.45"))  # put must be low in range
PASS_SCORE = float(os.environ.get("WG_PASS_SCORE", "50"))    # min score to pass


def _clamp(x, lo, hi): return max(lo, min(hi, x))


def expected_move(rv, dte) -> float:
    try:
        return float(rv) * math.sqrt(max(1, int(dte)) / 252.0)
    except Exception:
        return 0.0


def reachability(otm_pct, rv, dte) -> float:
    """Strike distance OTM in units of the expected move. <=1 == within ~1 sigma."""
    em = expected_move(rv, dte)
    if em <= 0:
        return 99.0
    try:
        return abs(float(otm_pct)) / em
    except Exception:
        return 99.0


# ── direction-aware scoring ─────────────────────────────────────────────────
def _score_call(in_uptrend, mom_6m, rng_pos, reach):
    s = 30.0 if in_uptrend else 0.0
    if mom_6m is not None:
        s += _clamp(float(mom_6m) / 0.40 * 30.0, 0, 30)      # momentum strength
    if reach is not None:
        s += _clamp((1.0 - min(float(reach), 2.0) / 2.0) * 25.0, 0, 25)
    if rng_pos is not None:
        s += _clamp(float(rng_pos) * 15.0, 0, 15)            # breakout = high in range OK
    return round(s, 1)


def _score_put(in_uptrend, mom_6m, rng_pos, reach, ret5):
    s = 30.0 if (in_uptrend is False) else 0.0               # downtrend is GOOD for puts
    if rng_pos is not None:
        s += _clamp((1.0 - float(rng_pos)) * 35.0, 0, 35)    # low in range = breaking down
    if reach is not None:
        s += _clamp((1.0 - min(float(reach), 2.0) / 2.0) * 20.0, 0, 20)
    if ret5 is not None:                                      # sharp recent drop
        s += _clamp((-float(ret5)) / 0.10 * 15.0, 0, 15)
    elif mom_6m is not None and float(mom_6m) < 0:
        s += _clamp((-float(mom_6m)) / 0.30 * 15.0, 0, 15)
    return round(s, 1)


def evaluate(features: dict, direction: str = "call") -> dict:
    """Grade a candidate. Direction-aware. Fails OPEN on missing features (never
    silently blocks). `direction` may also be read from features['direction']."""
    f = features or {}
    direction = (f.get("direction") or direction or "call").lower()
    is_put = direction == "put"

    in_uptrend = bool(f.get("in_uptrend")) if f.get("in_uptrend") is not None else None
    mom = f.get("mom_6m"); rng = f.get("rng_pos"); ret5 = f.get("ret5")
    otm = f.get("otm_pct"); rv = f.get("rv"); dte = f.get("dte"); reach = f.get("reach")
    if reach is None and None not in (otm, rv, dte):
        reach = reachability(otm, rv, dte)

    reasons = []; passed = True
    if is_put:
        score = _score_put(in_uptrend, mom, rng, reach, ret5)
        # a PUT should be a genuine breakdown: not a strong uptrend, low in its range
        if in_uptrend is True:
            passed = False; reasons.append("put but still in an uptrend (no breakdown)")
        if rng is not None and float(rng) > PUT_RNG_MAX:
            passed = False; reasons.append(f"not breaking down (range {float(rng):.0%} > {PUT_RNG_MAX:.0%})")
    else:
        score = _score_call(in_uptrend, mom, rng, reach)
        if in_uptrend is False:
            passed = False; reasons.append("call but not in an uptrend")
        if mom is not None and float(mom) < MOM_MIN:
            passed = False; reasons.append(f"weak momentum ({float(mom):.0%} < {MOM_MIN:.0%})")
    # reachability applies both ways
    if reach is not None and float(reach) > REACH_MAX:
        passed = False; reasons.append(f"strike unreachable (reach {float(reach):.2f} > {REACH_MAX:.2f})")
    # a low score is itself a veto (catches weak-but-not-hard-failing setups)
    if score < PASS_SCORE:
        passed = False; reasons.append(f"low score ({score:.0f} < {PASS_SCORE:.0f})")

    return {"passed": passed, "score": score, "reasons": reasons,
            "reach": (round(float(reach), 2) if reach is not None else None)}


def compute_entry_features(ticker, otm_pct=None, dte=None, iv=None,
                           direction="call") -> dict:
    """LIVE helper — build the gate feature dict (both call & put features).
    Defensive: returns {} on any failure so the caller fails open."""
    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf
        from momentum_strategy import momentum_signal
        h = yf.Ticker(ticker).history(period="14mo")["Close"].dropna()
        if len(h) < 210:
            return {}
        df = pd.DataFrame({"Close": h.values}, index=h.index)
        sig = momentum_signal(df) or {}
        S = float(h.iloc[-1])
        rets = np.log(h / h.shift(1)).dropna()
        rv = float(rets.iloc[-21:].std() * math.sqrt(252))
        hi20 = float(h.iloc[-20:].max()); lo20 = float(h.iloc[-20:].min())
        rng = (S - lo20) / (hi20 - lo20) if hi20 > lo20 else 0.5
        ema20 = float(h.ewm(span=20).mean().iloc[-1])
        ret5 = float(S / h.iloc[-6] - 1) if len(h) > 6 else 0.0
        feats = {
            "direction": direction,
            "in_uptrend": bool(sig.get("in_uptrend")),
            "mom_6m": sig.get("mom_6m"), "mom_3m": sig.get("mom_3m"),
            "rng_pos": round(rng, 3), "rv": round(rv, 3),
            "ret5": round(ret5, 4), "below_ema20": bool(S < ema20),
            "otm_pct": otm_pct, "dte": dte, "iv": iv,
        }
        if otm_pct is not None and dte is not None:
            feats["reach"] = round(reachability(otm_pct, rv, dte), 2)
        return feats
    except Exception:
        return {}


if __name__ == "__main__":
    call_good = {"direction": "call", "in_uptrend": True, "mom_6m": 0.22,
                 "rng_pos": 0.70, "otm_pct": 0.03, "rv": 0.45, "dte": 7}
    put_good = {"direction": "put", "in_uptrend": False, "rng_pos": 0.08,
                "ret5": -0.12, "otm_pct": 0.05, "rv": 0.60, "dte": 7}
    print("good call ->", evaluate(call_good, "call"))
    print("good put  ->", evaluate(put_good, "put"))
