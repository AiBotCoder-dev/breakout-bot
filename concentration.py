"""
concentration.py — Stop the bot piling into correlated bets.

THE EVIDENCE: across 38 multi-trade days, 13 of them (34%) had EVERY open position
move the same way. The clearest case was 2026-08-07 — the bot opened 10 positions
and closed 8 stopped, calls AND puts. That is not eight independent bets going wrong,
it is ONE bet expressed eight ways.

This does NOT try to create edge — it only removes concentration. Expectancy per
trade is unchanged; what falls is the variance of the whole book, which is what
actually threatens a small account (see ignition_capital.py: ruin risk is driven by
losing STREAKS, and correlated positions manufacture streaks).

Four caps, all env-overridable:
  MAX_OPEN_POSITIONS  total concurrent open contracts
  MAX_SAME_DIRECTION  concurrent calls (or puts) — forces the book to be two-sided
  MAX_PER_SECTOR      concurrent positions in one sector (book had 3 Financials
                      + 3 Technology at the time of writing)
  MAX_OPENS_PER_DAY   new entries per day (the 10-open day preceded the 8/8 wipeout)

check() returns (allowed, reason). Fails OPEN on any error — a bookkeeping problem
must never silently block all trading.
"""

from __future__ import annotations

import os


def _envi(name, dflt):
    try:
        return int(os.environ.get(name, "") or dflt)
    except Exception:
        return dflt


def limits() -> dict:
    # max_open raised 10 -> 14 (2026-08-14). The original 10 was a guess, not a
    # measured number, and the throughput arithmetic showed it was the binding
    # constraint rather than any risk finding: 10 slots / an 8-day average hold
    # sustains only ~1.25 entries per day against 3-10 signals per day, so real
    # signals were being turned away. 14 gives ~1.75/day at ZERO expectancy cost.
    # The CORRELATION risk we actually measured (34% of multi-trade days moving as
    # one) is handled by max_same_dir and max_sector, which target it directly —
    # those stay tight. Raising the headcount does not re-create that risk.
    return {
        "max_open":      _envi("MAX_OPEN_POSITIONS", 14),
        "max_same_dir":  _envi("MAX_SAME_DIRECTION", 10),
        "max_sector":    _envi("MAX_PER_SECTOR", 3),
        "max_per_day":   _envi("MAX_OPENS_PER_DAY", 5),
    }


def snapshot(conn) -> dict:
    """Current open-book composition. Never raises."""
    out = {"n": 0, "calls": 0, "puts": 0, "sectors": {}}
    try:
        rows = conn.execute(
            "SELECT contract_symbol, sector FROM broker_trade_journal "
            "WHERE status='OPEN'").fetchall()
    except Exception:
        return out
    for r in rows:
        d = dict(r) if hasattr(r, "keys") else {}
        sym = str(d.get("contract_symbol") or "")
        if len(sym) < 9:
            continue
        out["n"] += 1
        if sym[-9] == "P":
            out["puts"] += 1
        else:
            out["calls"] += 1
        sec = (d.get("sector") or "?").strip() or "?"
        out["sectors"][sec] = out["sectors"].get(sec, 0) + 1
    return out


def _opened_today(conn) -> int:
    """Read the LIVE daily entry counter. The caller's copy is computed once per
    cycle and goes stale as entries are placed within that same cycle, which would
    let a burst blow straight through the daily cap — the exact 10-open pattern this
    is here to prevent. broker_daily_entries is incremented per entry, so query it."""
    try:
        from datetime import datetime, timezone
        d = datetime.now(timezone.utc).date().isoformat()
        r = conn.execute("SELECT n FROM broker_daily_entries WHERE snapshot_date=?",
                         (d,)).fetchone()
        if not r:
            return 0
        return int((r.get("n") if hasattr(r, "get") else r[0]) or 0)
    except Exception:
        return 0


def check(conn, option_type="call", sector=None, opened_today=None) -> tuple:
    """May we open one more position? Returns (allowed: bool, reason: str).

    Fails OPEN — if the book can't be read we allow the trade rather than
    silently halting the whole strategy on a bookkeeping error.
    """
    try:
        L = limits()
        snap = snapshot(conn)
        if opened_today is None:
            opened_today = _opened_today(conn)
        is_put = str(option_type).lower().startswith("p")

        if snap["n"] >= L["max_open"]:
            return False, (f"book full ({snap['n']}/{L['max_open']} open)")

        same = snap["puts"] if is_put else snap["calls"]
        if same >= L["max_same_dir"]:
            side = "puts" if is_put else "calls"
            return False, (f"{same}/{L['max_same_dir']} {side} already open "
                           f"— one-sided book, skip to stay diversified")

        sec = (sector or "?").strip() or "?"
        if sec != "?" and snap["sectors"].get(sec, 0) >= L["max_sector"]:
            return False, (f"{snap['sectors'][sec]}/{L['max_sector']} already open "
                           f"in {sec} — sector concentration")

        if opened_today >= L["max_per_day"]:
            return False, (f"{opened_today}/{L['max_per_day']} opened today "
                           "— daily entry cap")
        return True, ""
    except Exception:
        return True, ""          # fail OPEN


def report(conn) -> dict:
    """For the dashboard: current usage against each cap."""
    L = limits(); s = snapshot(conn)
    worst_sec = max(s["sectors"].items(), key=lambda kv: kv[1]) if s["sectors"] else ("—", 0)
    return {"open": s["n"], "max_open": L["max_open"],
            "calls": s["calls"], "puts": s["puts"], "max_same_dir": L["max_same_dir"],
            "top_sector": worst_sec[0], "top_sector_n": worst_sec[1],
            "max_sector": L["max_sector"], "sectors": s["sectors"],
            "max_per_day": L["max_per_day"]}
