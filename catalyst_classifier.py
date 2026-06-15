"""
catalyst_classifier.py — Codifies the catalyst decision-tree into a verdict.

Turns the "how does price react to news" framework into something the bot applies
automatically. Given a ticker, it reads live price action + the calendar + IV and
outputs a structured read of WHY price is doing what it's doing and whether the
move is READABLE, a COIN-FLIP, or one to AVOID — with the reasoning.

The framework (move = (reality − expectations) × positioning × prior-run):
  1. SHOCK vs SCHEDULED — is there a known event (earnings/macro) nearby, or did
     it move on a surprise? Shocks: the move is mostly done; don't chase the gap.
  2. EXPECTATIONS — how much of the option-implied expected move is already used?
     (>80% used = the easy money is gone.)
  3. PRIOR RUN — extended + good move = fade risk (sell-the-news, like PLTR's beat).
     Beaten down + relief = pop (the Bottom-Fisher logic).
  4. IV STATE — high IV-rank = expensive + crush risk (prefer spread/skip);
     low IV-rank = cheap convexity (naked long ok).
  5. REACTION READ — did the gap HOLD (continuation) or FADE (reversal)? Read it,
     don't predict it.

Output: {classification, readability, bias, structure_hint, confidence, reasons[],
         verdict} — a one-glance call the trader (or the bot) can act on.
"""

from __future__ import annotations

from datetime import datetime, date

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except Exception:                       # pragma: no cover
    np = pd = yf = None


def _rsi(c, p=14):
    d = np.diff(c, prepend=c[0]); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = np.full_like(c, 50.0, float); ag = al = 0.0
    for i in range(1, len(c)):
        if i <= p: ag = (ag*(i-1)+g[i])/i; al = (al*(i-1)+l[i])/i
        else: ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+l[i])/p
        out[i] = 100.0 if al < 1e-12 else 100.0 - 100.0/(1.0+ag/al)
    return out


def _earnings_within(conn, ticker, days=4):
    """True (+ days_until) if earnings fall within `days`."""
    try:
        from earnings_engine import EarningsCalendar
        e = EarningsCalendar(conn).get(ticker)
        if not e:
            return False, None
        for k in ("date", "earnings_date", "next_earnings"):
            v = e.get(k) if isinstance(e, dict) else None
            if v:
                try:
                    d = datetime.fromisoformat(str(v)[:10]).date()
                    du = (d - date.today()).days
                    return (0 <= du <= days), du
                except Exception:
                    continue
    except Exception:
        pass
    return False, None


def classify(conn, ticker: str) -> dict | None:
    """Full catalyst read for one ticker, or None if data is missing."""
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period="1y", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) < 60:
            return None
    except Exception:
        return None

    C = df["Close"].values; O = df["Open"].values; H = df["High"].values
    L = df["Low"].values; V = df["Volume"].values
    i = len(C) - 1
    price = float(C[i]); prev = float(C[i-1])
    gap = (O[i]/prev - 1) * 100 if prev else 0
    day = (price/prev - 1) * 100 if prev else 0
    intraday = (price/O[i] - 1) * 100 if O[i] else 0
    rng = H[i] - L[i]
    range_pos = (price - L[i]) / rng if rng > 0 else 0.5      # 1=closed at high
    rsi = float(_rsi(C, 14)[i])
    ret20 = (price/C[i-20] - 1) * 100 if i >= 20 else 0
    hi252 = float(H[max(0, i-251):i+1].max()); lo252 = float(L[max(0, i-251):i+1].min())
    off_high = (price/hi252 - 1) * 100 if hi252 else 0
    vol20 = float(V[max(0, i-20):i].mean()) if i >= 20 else V[i]
    rvol = V[i]/vol20 if vol20 > 0 else 1
    big = abs(day) >= 4.0

    # calendar context
    earn_soon, earn_days = _earnings_within(conn, ticker, days=4)
    macro_high = False
    try:
        from macro_engine import event_risk
        macro_high = event_risk().get("level") == "HIGH"
    except Exception:
        pass

    # expectations: how much of the implied move is already used
    em_pct = used = None
    try:
        from options_analytics import expected_move
        em = expected_move(ticker)
        if em and em.get("exp_move_pct"):
            em_pct = em["exp_move_pct"]
            used = abs(day) / em_pct if em_pct > 0 else None
    except Exception:
        pass

    # IV state
    ivr = None
    try:
        from options_analytics import iv_rank
        r = iv_rank(conn, ticker)
        if r:
            ivr = r["iv_rank_pct"]
    except Exception:
        pass

    reasons = []

    # 1) classification — shock vs scheduled vs technical
    if earn_soon:
        classification = "SCHEDULED_EARNINGS"
        reasons.append(f"Earnings in {earn_days}d — a BINARY event (reaction depends on "
                       f"expectations, not the print; IV will crush after).")
    elif macro_high:
        classification = "SCHEDULED_MACRO"
        reasons.append("Major macro print imminent (CPI/FOMC/Jobs) — index-level "
                       "binary risk; 'good news can be bad news.'")
    elif big and rvol >= 1.8 and abs(gap) >= 3:
        classification = "SHOCK"
        reasons.append(f"Big gap ({gap:+.1f}%) on {rvol:.1f}x volume with no scheduled "
                       f"event — looks like a SURPRISE (news/M&A/legal). The move is "
                       f"largely DONE; don't chase the gap.")
    else:
        classification = "TECHNICAL"
        reasons.append("No imminent scheduled event and no shock-sized gap — this is "
                       "flow/technical, not a hard catalyst.")

    # 2) expectations
    if used is not None:
        if used >= 0.8:
            reasons.append(f"Already moved {abs(day):.1f}% = {used:.1f}x the implied "
                           f"move ({em_pct:.1f}%) — the easy money is largely gone.")
        else:
            reasons.append(f"Moved {abs(day):.1f}% of an implied {em_pct:.1f}% "
                           f"({used:.1f}x) — room left vs what's priced in.")

    # 3) prior run / positioning
    extended = ret20 > 20 or rsi > 72 or off_high > -3
    beaten = ret20 < -20 or rsi < 30
    if extended:
        reasons.append(f"EXTENDED (20d {ret20:+.0f}%, RSI {rsi:.0f}, {off_high:+.0f}% off "
                       f"52w-high) — buyers exhausted; good news can still SELL OFF.")
    elif beaten:
        reasons.append(f"BEATEN DOWN (20d {ret20:+.0f}%, RSI {rsi:.0f}) — sellers "
                       f"exhausted; any relief tends to POP.")

    # 4) IV state
    if ivr is not None:
        if ivr >= 65:
            reasons.append(f"IV-rank {ivr:.0f}% — options EXPENSIVE + crush risk; "
                           f"prefer a debit spread or skip the naked long.")
        elif ivr <= 35:
            reasons.append(f"IV-rank {ivr:.0f}% — vol is CHEAP with room to expand; "
                           f"naked long has real convexity.")

    # 4b) EXPECTATIONS + POSITIONING (free analyst/short data) — the missing half
    #     of (reality − expectations). Priced-for-perfection = fade risk on good
    #     news; rising estimates = tailwind; high short interest = squeeze fuel.
    exp = None
    try:
        from expectations import get_expectations
        exp = get_expectations(ticker)
        for _s in (exp.get("signals") or [])[:3]:
            reasons.append(_s)
    except Exception:
        exp = None

    # 5) reaction read — did the move hold or fade?
    if big:
        if day > 0 and range_pos >= 0.66:
            reasons.append("Up move HELD into the close (strong) — continuation read.")
        elif day > 0 and range_pos <= 0.4:
            reasons.append("Up move FADED off the highs (sell-the-news) — reversal read.")
        elif day < 0 and range_pos <= 0.34:
            reasons.append("Down move HELD weak into the close — breakdown read.")
        elif day < 0 and range_pos >= 0.6:
            reasons.append("Down move REVERSED off the lows (capitulation buy) — bounce read.")

    # ── synthesize verdict ─────────────────────────────────────────────────────
    if classification in ("SCHEDULED_EARNINGS", "SCHEDULED_MACRO"):
        readability, bias, structure = "AVOID", "neutral", "skip / wait for the print"
    elif classification == "SHOCK":
        # don't chase; the second move is the trade
        readability = "COIN_FLIP"
        bias = "neutral"
        structure = "wait for the SECOND move (continuation or reversal), don't chase the gap"
    else:
        # technical/flow — read the prior run + reaction + EXPECTATIONS.
        # "priced for perfection" (above analyst target) amplifies fade risk;
        # rising/falling estimate revisions confirm the directional lean.
        _ppf = bool(exp and exp.get("priced_for_perfection"))
        _exp_bias = (exp or {}).get("positioning_bias", "neutral")
        _fade = extended or _ppf
        if beaten and (day > 0 and range_pos >= 0.6):
            readability, bias = "READABLE", "bullish"
        elif _fade and (day < 0 or range_pos <= 0.4):
            readability, bias = "READABLE", "bearish"
        elif day > 0 and range_pos >= 0.66 and not _fade:
            readability, bias = "READABLE", "bullish"
        elif day < 0 and range_pos <= 0.34 and not beaten:
            readability, bias = "READABLE", "bearish"
        else:
            readability, bias = "COIN_FLIP", "neutral"
        # expectations confirmation: a matching analyst lean upgrades conviction;
        # a contradicting one knocks a READABLE down to a coin-flip.
        if readability == "READABLE" and _exp_bias != "neutral":
            if _exp_bias != bias:
                readability = "COIN_FLIP"
                reasons.append(f"…but analyst positioning leans {_exp_bias} "
                               f"(conflicts with the price read) — downgraded to coin-flip.")
        if ivr is not None and ivr >= 65:
            structure = "debit spread (IV too high for a naked long)"
        elif bias == "bullish":
            structure = "near-money call"
        elif bias == "bearish":
            structure = "near-money put"
        else:
            structure = "wait for confirmation"

    confidence = {"READABLE": 65, "COIN_FLIP": 35, "AVOID": 0}[readability]

    verdict = (f"{ticker}: {classification} → {readability}"
               + (f" ({bias})" if bias != "neutral" else "")
               + f". {structure}.")

    return {
        "ticker": ticker, "price": round(price, 2), "day_pct": round(day, 1),
        "gap_pct": round(gap, 1), "rvol": round(rvol, 1), "rsi": round(rsi, 0),
        "classification": classification, "readability": readability,
        "bias": bias, "structure_hint": structure, "confidence": confidence,
        "iv_rank": ivr, "expected_move_pct": em_pct,
        "reasons": reasons, "verdict": verdict,
    }


if __name__ == "__main__":
    class _NoConn:
        def execute(self, *a, **k):
            class _R:
                def fetchone(self): return None
                def fetchall(self): return []
            return _R()
    for t in ["NVDA", "PLTR", "SMCI", "SOFI"]:
        r = classify(_NoConn(), t)
        if r:
            print(f"\n{r['verdict']}")
            for x in r["reasons"]:
                print(f"   • {x}")
        else:
            print(f"\n{t}: no data")
