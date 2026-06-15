"""
expectations.py — FREE "expectations + positioning" data for the thinking system.

The catalyst framework's core equation is move = (reality − EXPECTATIONS) × positioning.
The bot could read price + news but was blind to the EXPECTATIONS half — what the
Street already assumes. This fills that gap using only free yfinance fields:

  • Analyst price target gap  — current vs consensus target. Far BELOW target =
    analyst upside (room to run). At/ABOVE target = priced for perfection
    (good news can still sell off — the PLTR lesson).
  • Estimate-revision trend    — are forward EPS estimates being RAISED or CUT?
    (eps_trend current vs 30d ago.) Rising = analyst tailwind; falling = headwind.
    This is one of the most predictive free signals there is (PEAD's cousin).
  • Analyst recommendation mean — 1=strong buy … 5=sell (consensus lean).
  • Short interest (% float)    — crowded short = squeeze fuel on a rally AND
    crash fuel on a breakdown; informs both call and put reads.

Output feeds the catalyst classifier (the thinking system) — informational/decision
context, NOT yet a trade-score input (that should be backtested before shipping).
"""

from __future__ import annotations

try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None


def _num(v, d=None):
    try:
        x = float(v)
        return x if x == x else d
    except Exception:
        return d


def get_expectations(ticker: str) -> dict:
    """Free analyst-expectations + positioning read for one ticker."""
    out = {"ok": False, "signals": [], "positioning_bias": "neutral",
           "priced_for_perfection": False, "estimate_trend": None,
           "target_gap_pct": None, "short_pct_float": None, "rec_mean": None}
    if yf is None:
        return out
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception:
        return out

    price = _num(info.get("currentPrice"))
    tgt   = _num(info.get("targetMeanPrice"))
    rec   = _num(info.get("recommendationMean"))
    nanal = _num(info.get("numberOfAnalystOpinions"), 0) or 0
    shortf = _num(info.get("shortPercentOfFloat"))
    sig = []
    bull = bear = 0

    # 1) price target gap
    gap = None
    if price and tgt and tgt > 0:
        gap = (price / tgt - 1) * 100
        out["target_gap_pct"] = round(gap, 1)
        if gap <= -15:
            sig.append(f"{abs(gap):.0f}% BELOW consensus target ${tgt:.0f} — analyst upside (room to run)")
            bull += 1
        elif gap >= 3:
            sig.append(f"At/ABOVE consensus target ${tgt:.0f} ({gap:+.0f}%) — PRICED FOR PERFECTION; good news can still sell off")
            out["priced_for_perfection"] = True
            bear += 1

    # 2) estimate-revision trend (forward EPS now vs 30d ago)
    try:
        et = tk.eps_trend
        if et is not None and not et.empty:
            row = None
            for key in ("+1q", "0y", "+1y", "0q"):
                if key in et.index:
                    row = et.loc[key]; break
            cur = _num(row.get("current")) if row is not None else None
            old = _num(row.get("30daysAgo")) if row is not None else None
            if cur is not None and old is not None and old != 0:
                chg = (cur / old - 1) * 100
                if chg >= 1:
                    out["estimate_trend"] = "rising"
                    sig.append(f"Forward EPS estimates REVISED UP {chg:+.0f}% in 30d — analyst tailwind")
                    bull += 1
                elif chg <= -1:
                    out["estimate_trend"] = "falling"
                    sig.append(f"Forward EPS estimates CUT {chg:.0f}% in 30d — analyst headwind")
                    bear += 1
                else:
                    out["estimate_trend"] = "flat"
    except Exception:
        pass

    # 3) recommendation mean
    if rec and nanal >= 5:
        out["rec_mean"] = round(rec, 2)
        if rec <= 2.0:
            sig.append(f"Analyst consensus STRONG BUY ({rec:.1f}, {int(nanal)} analysts)")
            bull += 0.5
        elif rec >= 3.5:
            sig.append(f"Analyst consensus bearish ({rec:.1f})")
            bear += 0.5

    # 4) short interest
    if shortf is not None:
        out["short_pct_float"] = round(shortf * 100, 1)
        if shortf >= 0.15:
            sig.append(f"HIGH short interest {shortf*100:.0f}% of float — squeeze fuel on a rally / crash fuel on a breakdown")

    if bull - bear >= 1:
        out["positioning_bias"] = "bullish"
    elif bear - bull >= 1:
        out["positioning_bias"] = "bearish"
    out["ok"] = bool(sig)
    out["signals"] = sig
    return out


if __name__ == "__main__":
    for t in ["NVDA", "PLTR", "SOFI", "INTC"]:
        e = get_expectations(t)
        print(f"\n{t}: bias={e['positioning_bias']} "
              f"target_gap={e['target_gap_pct']} est={e['estimate_trend']} "
              f"short={e['short_pct_float']}%")
        for s in e["signals"]:
            print(f"   • {s}")
