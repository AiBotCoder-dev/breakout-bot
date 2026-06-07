"""
macro_engine.py — "Human logic" macro layer: events, interpretation, event-risk.

The market doesn't move on charts alone — it moves on MACRO DATA and what that
data implies for Fed policy. This encodes the cause-and-effect a seasoned trader
carries in their head:

  • Hot inflation (CPI/PCE above expected) -> rate-hike fears -> risk-OFF
    (growth/tech hit hardest because higher rates compress long-duration valuations)
  • Cool inflation -> rate-cut hopes -> risk-ON
  • Strong jobs (NFP hot): in a HAWKISH/high-rate regime this is "good news is bad
    news" -> risk-OFF (Fed stays tight). In an easing regime -> risk-ON.
  • Hawkish Fed/FOMC -> down; dovish -> up.

It does THREE things:
  1. Knows what's coming (economic calendar — jobs, CPI, PCE, FOMC).
  2. Flags EVENT RISK before binary data drops -> "wait, don't size up."
  3. Interprets a release (actual vs consensus) into a likely market reaction +
     which sectors get hurt/helped, with the reasoning spelled out.

HONEST LIMITS: this does NOT predict the data or guarantee the reaction — nobody
can. It encodes the documented relationships and the risk calendar so you act
with awareness instead of getting blindsided (like Friday's jobs report).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta


# ── Known 2026 FOMC decision dates (2nd day = the announcement) ────────────────
# Publicly scheduled. Verify against federalreserve.gov if trading around them.
FOMC_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 16),
]

# High-impact event profiles (impact 1-5; 5 = can move the whole market)
EVENT_PROFILE = {
    "JOBS":   {"name": "Jobs Report (NFP)",      "impact": 5, "time": "8:30 ET"},
    "CPI":    {"name": "CPI Inflation",          "impact": 5, "time": "8:30 ET"},
    "PCE":    {"name": "PCE (Fed's gauge)",      "impact": 4, "time": "8:30 ET"},
    "FOMC":   {"name": "FOMC Rate Decision",     "impact": 5, "time": "14:00 ET"},
    "PPI":    {"name": "PPI (producer prices)",  "impact": 3, "time": "8:30 ET"},
    "RETAIL": {"name": "Retail Sales",           "impact": 3, "time": "8:30 ET"},
    "GDP":    {"name": "GDP",                     "impact": 3, "time": "8:30 ET"},
}


# ══════════════════════════════════════════════════════════════════════════════
# CALENDAR — compute recurring event dates
# ══════════════════════════════════════════════════════════════════════════════
def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)   # weekday 4 = Friday


def _last_business_day(year: int, month: int) -> date:
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() >= 5:        # back up over weekend
        d -= timedelta(days=1)
    return d


def upcoming_events(days_ahead: int = 21, today: date | None = None) -> list:
    """
    Return high-impact events in the next `days_ahead` days, soonest first.
    Recurring events computed; FOMC from the known list. Dates for CPI/PCE/PPI/
    retail are APPROXIMATE (agencies set exact days) — labeled as such.
    """
    today = today or date.today()
    end = today + timedelta(days=days_ahead)
    events = []

    # Iterate the months we might touch
    months = []
    y, m = today.year, today.month
    for _ in range(3):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1; y += 1

    for (yy, mm) in months:
        # Jobs report — first Friday (exact cadence)
        events.append((_first_friday(yy, mm), "JOBS", False))
        # CPI — typically ~10th-14th business; approximate to the 12th
        events.append((date(yy, mm, 12), "CPI", True))
        # PPI — usually a day or two before/after CPI; approximate to 13th
        events.append((date(yy, mm, 13), "PPI", True))
        # Retail sales — ~mid month; approximate to 15th
        events.append((date(yy, mm, 15), "RETAIL", True))
        # PCE — last business day-ish
        events.append((_last_business_day(yy, mm), "PCE", True))

    for d in FOMC_2026:
        events.append((d, "FOMC", False))

    out = []
    for d, etype, approx in sorted(events, key=lambda x: x[0]):
        if today <= d <= end:
            prof = EVENT_PROFILE.get(etype, {})
            out.append({
                "date": d.isoformat(),
                "days_away": (d - today).days,
                "type": etype,
                "name": prof.get("name", etype),
                "impact": prof.get("impact", 3),
                "time": prof.get("time", ""),
                "approx_date": approx,
            })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# EVENT-RISK GATE — "should I wait?"
# ══════════════════════════════════════════════════════════════════════════════
def event_risk(today: date | None = None) -> dict:
    """
    Assess near-term macro event risk.
    Returns {level, next_event, days_away, advice}.
      level: LOW / ELEVATED / HIGH
    """
    today = today or date.today()
    ev = upcoming_events(days_ahead=10, today=today)
    high = [e for e in ev if e["impact"] >= 5]
    if not ev:
        return {"level": "LOW", "next_event": None, "days_away": None,
                "advice": "No major macro events in the next 10 days. Normal conditions."}
    nxt = high[0] if high else ev[0]
    da = nxt["days_away"]
    if nxt["impact"] >= 5 and da <= 1:
        level = "HIGH"
        advice = (f"{nxt['name']} {('tomorrow' if da==1 else 'today')} "
                  f"({nxt['time']}). Binary risk — AVOID new size, hold dry powder. "
                  f"The data can override every technical setup.")
    elif nxt["impact"] >= 5 and da <= 4:
        level = "ELEVATED"
        advice = (f"{nxt['name']} in {da} days. Elevated risk — trade smaller, keep "
                  f"cash for the reaction. Don't chase into the print.")
    else:
        level = "LOW"
        advice = (f"Next major event: {nxt['name']} in {da} days. Conditions normal "
                  f"for now; revisit as it approaches.")
    return {"level": level, "next_event": nxt["name"], "days_away": da,
            "next_date": nxt["date"], "advice": advice}


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION — actual vs consensus -> likely reaction
# ══════════════════════════════════════════════════════════════════════════════
def interpret(event_type: str, surprise: str, regime: str = "hawkish") -> dict:
    """
    surprise: 'hot' (above expected) / 'cool' (below) / 'inline'
    regime:   'hawkish' (rate-fearful / high-rate) or 'easing' (cuts expected)
    Returns {bias, magnitude, sectors_hurt, sectors_helped, reasoning}.
    """
    et = event_type.upper()
    s = surprise.lower()
    hawkish = regime != "easing"

    # Inflation data: hot = bad for stocks (rate fears), cool = good
    if et in ("CPI", "PCE", "PPI"):
        if s == "hot":
            return {"bias": "RISK-OFF", "magnitude": "high",
                    "sectors_hurt": ["Tech/Growth", "Real Estate", "Small caps"],
                    "sectors_helped": ["Energy", "Financials (rates up)", "Cash"],
                    "reasoning": "Hot inflation revives rate-hike fears. Higher rates "
                    "compress long-duration (growth/tech) valuations most. Risk-off."}
        if s == "cool":
            return {"bias": "RISK-ON", "magnitude": "high",
                    "sectors_hurt": ["Defensives (relative)"],
                    "sectors_helped": ["Tech/Growth", "Small caps", "Real Estate"],
                    "reasoning": "Cooler inflation feeds rate-cut hopes. Lower discount "
                    "rate lifts growth/tech most. Broad risk-on relief rally."}
        return {"bias": "NEUTRAL", "magnitude": "low",
                "sectors_hurt": [], "sectors_helped": [],
                "reasoning": "Inflation in line — relief, but no new directional fuel."}

    # Jobs: regime-dependent good-news-is-bad-news
    if et == "JOBS":
        if s == "hot":
            if hawkish:
                return {"bias": "RISK-OFF", "magnitude": "high",
                        "sectors_hurt": ["Tech/Growth", "Rate-sensitive"],
                        "sectors_helped": ["Energy", "Financials"],
                        "reasoning": "Strong jobs in a rate-fearful market = 'good news "
                        "is bad news': a hot labor market keeps the Fed hawkish, so "
                        "stocks SELL OFF despite a strong economy. (This is what hit "
                        "Friday.)"}
            return {"bias": "RISK-ON", "magnitude": "medium",
                    "sectors_hurt": [], "sectors_helped": ["Cyclicals", "Financials"],
                    "reasoning": "Strong jobs with the Fed already easing = healthy "
                    "economy, risk-on."}
        if s == "cool":
            if hawkish:
                return {"bias": "RISK-ON", "magnitude": "medium",
                        "sectors_hurt": [], "sectors_helped": ["Tech/Growth"],
                        "reasoning": "Weak jobs eases Fed-tightening pressure -> rate-cut "
                        "hopes -> relief rally. But watch for 'too weak = recession fear'."}
            return {"bias": "RISK-OFF", "magnitude": "medium",
                    "sectors_hurt": ["Cyclicals"], "sectors_helped": ["Defensives", "Bonds"],
                    "reasoning": "Weak jobs with no tightening to relieve = recession "
                    "worry -> risk-off into defensives."}
        return {"bias": "NEUTRAL", "magnitude": "low", "sectors_hurt": [],
                "sectors_helped": [], "reasoning": "Jobs in line — no surprise to trade."}

    # FOMC
    if et == "FOMC":
        if s in ("hot", "hawkish"):
            return {"bias": "RISK-OFF", "magnitude": "high",
                    "sectors_hurt": ["Tech/Growth", "Real Estate", "Small caps"],
                    "sectors_helped": ["Financials", "Cash"],
                    "reasoning": "Hawkish Fed (hike or hawkish hold/guidance) -> higher "
                    "rates for longer -> valuations compress, risk-off."}
        if s in ("cool", "dovish"):
            return {"bias": "RISK-ON", "magnitude": "high",
                    "sectors_hurt": [], "sectors_helped": ["Tech/Growth", "Small caps"],
                    "reasoning": "Dovish Fed (cut or dovish guidance) -> lower-rate path "
                    "-> growth/tech rip, broad risk-on."}
        return {"bias": "NEUTRAL", "magnitude": "medium", "sectors_hurt": [],
                "sectors_helped": [], "reasoning": "Fed in line with expectations — "
                "reaction depends on the press-conference tone."}

    return {"bias": "NEUTRAL", "magnitude": "low", "sectors_hurt": [],
            "sectors_helped": [], "reasoning": "Lower-impact release."}


def explain_recent(vix: float | None = None, spy_5d_pct: float | None = None,
                   today: date | None = None) -> str:
    """
    Narrate the likely macro driver of recent action — answers 'why did the market
    move?' Heuristic: cross-reference recent high-impact events + price/VIX behavior.
    """
    today = today or date.today()
    # Was there a major event in the last 3 days?
    recent = []
    for off in range(0, 4):
        d = today - timedelta(days=off)
        ev = upcoming_events(days_ahead=0, today=d)  # events ON day d
        for e in upcoming_events(days_ahead=1, today=d):
            if e["days_away"] == 0 and e["impact"] >= 4:
                recent.append((d, e))
    bits = []
    if recent:
        seen = set()
        for d, e in recent:
            if e["type"] in seen:
                continue
            seen.add(e["type"])
            bits.append(f"{e['name']} on {d.strftime('%a %d %b')}")
    move = ""
    if spy_5d_pct is not None:
        move = (f"SPY {spy_5d_pct:+.1f}% over 5 days. " )
    vix_txt = f"VIX {vix:.0f}. " if vix is not None else ""
    if bits:
        return (f"{move}{vix_txt}Likely macro driver: {', '.join(bits)}. "
                f"If the data ran hot, the selloff is the rate-fear / 'good news is "
                f"bad news' mechanism — strong data keeps the Fed hawkish, which "
                f"compresses growth/tech. Check the event-risk gate before adding size.")
    return (f"{move}{vix_txt}No major scheduled macro release in the last few days — "
            f"recent moves are more likely technical/positioning or headline-driven "
            f"than data-driven.")
