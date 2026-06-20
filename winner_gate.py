"""
winner_gate.py — META-LABEL filter that separates likely WINNERS from likely
losers, sitting IN FRONT of the existing strategy.

It changes NOTHING about how signals are generated. The primary strategy still
produces the same candidates; the gate only decides which of them to ACT on (and,
later, which to size up). This is the "meta-labeling" approach (board issue #6).

Four pre-trade separators, drawn from the bot's own win/loss history (the trailing
winners were real trends that moved; the losers were unreachable strikes that
theta-died, entries bought at the highs, below-trend signals, and data-floor
explore tickets):

  1. Strike REACHABILITY — the strike's distance OTM must be within ~1 sigma of the
     expected move over the hold. Kills the theta-death lottery tickets (the SMH
     TIME_STOP, the +15%-OTM/9-DTE PLTR call).
  2. Don't CHASE — skip entries at the top of the recent range / the 10-day high.
     Kills the same-day stop-outs (CSCO/PLTR bought at the open high).
  3. Full TREND alignment — price>sma50>sma200 AND mom_6m >= MOM_MIN (the VALIDATED
     bar; the live scan currently uses 0.05, half of it).
  4. No FREE PASS — data-floor "explore" trades face the same gate (they were
     disproportionately losers, e.g. the -$97 UNH).

PUBLIC API:
  evaluate(features) -> {passed: bool, score: float 0..100, reasons: [str], reach}
  compute_entry_features(ticker, otm_pct, dte, iv) -> feature dict (live helper)
  expected_move / reachability — the strike-reachability math (shared with backtest)

All thresholds are env-overridable so they can be tuned/swept without code edits.
"""
from __future__ import annotations
import math
import os

# ── thresholds (env-overridable) ────────────────────────────────────────────
REACH_MAX = float(os.environ.get("WG_REACH_MAX", "1.0"))    # strike <= ~1 sigma move
CHASE_MAX = float(os.environ.get("WG_CHASE_MAX", "0.85"))   # position in 20-day range
MOM_MIN   = float(os.environ.get("WG_MOM_MIN", "0.10"))     # validated momentum bar
REQUIRE_UPTREND = os.environ.get("WG_REQUIRE_UPTREND", "1").strip().lower() \
    not in ("0", "false", "no", "off", "")


def expected_move(rv, dte) -> float:
    """1-sigma expected move (as a fraction) over `dte` calendar days at vol `rv`."""
    try:
        return float(rv) * math.sqrt(max(1, int(dte)) / 252.0)
    except Exception:
        return 0.0


def reachability(otm_pct, rv, dte) -> float:
    """Strike distance OTM in units of the expected move. <=1.0 means the strike is
    within ~1 sigma (reachable); large means the option needs a move it rarely makes."""
    em = expected_move(rv, dte)
    if em <= 0:
        return 99.0
    try:
        return float(otm_pct) / em
    except Exception:
        return 99.0


def _score(in_uptrend, mom_6m, rng_pos, reach) -> float:
    """0..100 'winner score' — also usable later for position sizing."""
    s = 0.0
    s += 30.0 if in_uptrend else 0.0
    if mom_6m is not None:
        s += max(0.0, min(25.0, (float(mom_6m) / 0.40) * 25.0))
    if rng_pos is not None:
        s += max(0.0, min(20.0, (1.0 - float(rng_pos)) * 20.0))
    if reach is not None:
        s += max(0.0, min(25.0, (1.0 - min(float(reach), 2.0) / 2.0) * 25.0))
    return round(s, 1)


def evaluate(features: dict) -> dict:
    """Apply the four separators. Missing features are skipped (fail-open per check)
    so a data hiccup never silently blocks everything; the LIVE path additionally
    fails open if the whole feature dict is empty."""
    f = features or {}
    reasons = []
    passed = True

    in_uptrend = bool(f.get("in_uptrend")) if f.get("in_uptrend") is not None else None
    mom = f.get("mom_6m")
    rng = f.get("rng_pos")
    otm = f.get("otm_pct")
    rv = f.get("rv")
    dte = f.get("dte")
    reach = f.get("reach")
    if reach is None and otm is not None and rv is not None and dte is not None:
        reach = reachability(otm, rv, dte)

    # 1 + 3) trend alignment
    if REQUIRE_UPTREND and in_uptrend is False:
        passed = False
        reasons.append("not in uptrend (need price>sma50>sma200)")
    if mom is not None and float(mom) < MOM_MIN:
        passed = False
        reasons.append(f"weak momentum (mom_6m {float(mom):.0%} < {MOM_MIN:.0%})")
    # 2) don't chase
    if rng is not None and float(rng) > CHASE_MAX:
        passed = False
        reasons.append(f"chasing (range pos {float(rng):.0%} > {CHASE_MAX:.0%})")
    # 4) strike reachability
    if reach is not None and float(reach) > REACH_MAX:
        passed = False
        reasons.append(f"strike unreachable (reach {float(reach):.2f} > {REACH_MAX:.2f})")

    return {"passed": passed,
            "score": _score(bool(in_uptrend), mom, rng, reach),
            "reasons": reasons,
            "reach": (round(float(reach), 2) if reach is not None else None)}


def compute_entry_features(ticker, otm_pct=None, dte=None, iv=None) -> dict:
    """LIVE helper: pull the ticker's history and build the gate feature dict.
    Defensive — returns {} on any failure so the caller can fail open."""
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
        hi20 = float(h.iloc[-20:].max())
        lo20 = float(h.iloc[-20:].min())
        rng = (S - lo20) / (hi20 - lo20) if hi20 > lo20 else 0.5
        feats = {
            "in_uptrend": bool(sig.get("in_uptrend")),
            "mom_6m": sig.get("mom_6m"),
            "mom_3m": sig.get("mom_3m"),
            "rng_pos": round(rng, 3),
            "rv": round(rv, 3),
            "otm_pct": otm_pct,
            "dte": dte,
            "iv": iv,
        }
        if otm_pct is not None and dte is not None:
            feats["reach"] = round(reachability(otm_pct, rv, dte), 2)
        return feats
    except Exception:
        return {}


if __name__ == "__main__":
    # quick self-test
    demo = {"in_uptrend": True, "mom_6m": 0.22, "rng_pos": 0.40,
            "otm_pct": 0.03, "rv": 0.45, "dte": 21}
    print("demo features:", demo)
    print("evaluate     :", evaluate(demo))
    bad = {"in_uptrend": False, "mom_6m": 0.04, "rng_pos": 0.97,
           "otm_pct": 0.15, "rv": 0.30, "dte": 9}
    print("bad features :", bad)
    print("evaluate     :", evaluate(bad))
