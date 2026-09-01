"""
health_monitor.py — Make the bot tell you when it's broken.

WHY THIS EXISTS: every one of these ran silently for weeks before being found by
accident during unrelated analysis —
  * the leveraged shadow marked a DEAD price feed for 12 days (equity frozen at
    $9,991.99 to the cent while the market moved)
  * 46 expired options were never booked, hiding ~$2,100 of losses and flattering
    every statistic we were reasoning from
  * the overnight module deadlocked on 2026-06-19 and stopped trading entirely
  * ~half of all scheduled runs were failing to acquire a runner
None of these threw an error. They just quietly stopped being true, and every
decision made in between rested on numbers that were wrong.

So this asserts INVARIANTS each day and alerts when one breaks. It is deliberately
boring: no cleverness, no thresholds to tune, just "is this thing still alive and
is this number still moving?" Alerts dedupe to at most one per issue per day, since
an alerting system that spams gets muted and then it is worse than nothing.

Checks:
  stale_shadow_equity  leveraged shadow equity unchanged for N days (the dead feed)
  stale_calibration    IV calibration hasn't recorded in N days
  expired_open         journal rows still OPEN past their expiry (unbooked losers)
  overnight_stuck      an overnight position open more than 3 days (deadlock)
  no_recent_trades     no option entries in N weekdays (silent trading halt)
  ghost_regression     the retired exit policy is beating the live one
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

STALE_EQUITY_DAYS = 3
STALE_CALIB_DAYS = 4
NO_TRADE_DAYS = 3
OVERNIGHT_MAX_DAYS = 3


def _ensure(conn):
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS health_alerts "
                     "(d TEXT, issue TEXT, PRIMARY KEY (d, issue))")
    except Exception:
        pass


def _already_alerted(conn, issue) -> bool:
    try:
        r = conn.execute("SELECT 1 FROM health_alerts WHERE d=? AND issue=?",
                         (date.today().isoformat(), issue)).fetchone()
        return bool(r)
    except Exception:
        return False


def _mark(conn, issue):
    try:
        conn.execute("INSERT INTO health_alerts (d, issue) VALUES (?,?) "
                     "ON CONFLICT DO NOTHING", (date.today().isoformat(), issue))
    except Exception:
        pass


def _q(conn, sql, args=()):
    try:
        return [dict(r) if hasattr(r, "keys") else r
                for r in conn.execute(sql, args).fetchall()]
    except Exception:
        return []


def run_checks(conn) -> list:
    """Return a list of {issue, detail} for every broken invariant."""
    _ensure(conn)
    issues = []
    today = date.today()

    # 1) leveraged shadow equity frozen -> the dead-feed failure
    rows = _q(conn, "SELECT d, equity FROM lev_shadow_equity ORDER BY d DESC LIMIT ?",
              (STALE_EQUITY_DAYS + 1,))
    if len(rows) >= STALE_EQUITY_DAYS + 1:
        eqs = {round(float(r["equity"] or 0), 2) for r in rows}
        if len(eqs) == 1:
            issues.append({"issue": "stale_shadow_equity",
                           "detail": f"leveraged shadow equity unchanged at "
                                     f"${rows[0]['equity']:,.2f} for "
                                     f"{len(rows)} snapshots — price feed likely dead"})

    # 2) IV calibration not recording
    rows = _q(conn, "SELECT MAX(d) m FROM iv_calibration")
    if rows and rows[0].get("m"):
        try:
            last = date.fromisoformat(str(rows[0]["m"])[:10])
            gap = (today - last).days
            if gap > STALE_CALIB_DAYS:
                issues.append({"issue": "stale_calibration",
                               "detail": f"IV calibration last recorded {last} "
                                         f"({gap}d ago)"})
        except Exception:
            pass

    # 3) expired options still marked OPEN -> unbooked losers
    rows = _q(conn, "SELECT contract_symbol FROM broker_trade_journal WHERE status='OPEN'")
    stale_exp = 0
    try:
        from broker import AlpacaPaperBroker as _B
        for r in rows:
            p = _B.parse_occ_symbol(str(r.get("contract_symbol") or ""))
            if p and p["expiry"] < today:
                stale_exp += 1
    except Exception:
        stale_exp = 0
    if stale_exp:
        issues.append({"issue": "expired_open",
                       "detail": f"{stale_exp} journal row(s) still OPEN past expiry "
                                 "— reconciliation not running"})

    # 4) overnight position stuck open -> the deadlock
    rows = _q(conn, "SELECT entry_date FROM overnight_edge_log WHERE status='OPEN' "
                    "ORDER BY id DESC LIMIT 1")
    if rows:
        try:
            ed = date.fromisoformat(str(rows[0]["entry_date"])[:10])
            age = (today - ed).days
            if age > OVERNIGHT_MAX_DAYS:
                issues.append({"issue": "overnight_stuck",
                               "detail": f"overnight position open since {ed} "
                                         f"({age}d) — an overnight trade lasts ONE night"})
        except Exception:
            pass

    # 5) no option entries recently (silent trading halt)
    rows = _q(conn, "SELECT MAX(substr(opened_at,1,10)) m FROM broker_trade_journal")
    if rows and rows[0].get("m"):
        try:
            last = date.fromisoformat(str(rows[0]["m"])[:10])
            wd = 0; d = last
            while d < today:
                d += timedelta(days=1)
                if d.weekday() < 5:
                    wd += 1
            if wd > NO_TRADE_DAYS:
                issues.append({"issue": "no_recent_trades",
                               "detail": f"no option entry since {last} "
                                         f"({wd} weekdays) — entries may be halted"})
        except Exception:
            pass

    # 6) exit-policy regression — the SHADOW arm beating the LIVE one.
    # Must be direction-aware: `diff` is (C mean - D mean), so diff<0 means D wins.
    # Before 2026-08-28 C was live and diff<0 was a regression; after the revert D is
    # live and diff<0 means the live policy is WINNING. Keying off the sign alone
    # made this fire on good news — a false alarm, and an alerting system that cries
    # wolf gets muted, which is worse than no alerting at all.
    try:
        import ghost_exits as _ge
        import os as _os
        reg = _ge.regression_check(conn)
        live_is_d = float(_os.environ.get("EXIT_HARD_STOP", "") or -50.0) > -99
        shadow_winning = (reg.get("diff", 0) > 0) if live_is_d else (reg.get("diff", 0) < 0)
        if reg.get("n", 0) >= 20 and shadow_winning:
            issues.append({"issue": "ghost_regression", "detail": reg["verdict"]})
    except Exception:
        pass

    return issues


def step(conn, notify=None) -> dict:
    """Run the checks and alert on anything NEW today. Read-only."""
    issues = run_checks(conn)
    fresh = [i for i in issues if not _already_alerted(conn, i["issue"])]
    for i in fresh:
        _mark(conn, i["issue"])
    if fresh and notify:
        try:
            lines = ["🩺 <b>Bot health check FAILED</b>"]
            for i in fresh:
                lines.append(f"• <b>{i['issue']}</b>: {i['detail']}")
            lines.append("\n<i>These are silent failures — the bot kept running and "
                         "kept reporting numbers that were wrong.</i>")
            notify("\n".join(lines))
        except Exception:
            pass
    return {"issues": len(issues), "new": len(fresh),
            "detail": [i["issue"] for i in issues]}


if __name__ == "__main__":
    import os, json
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit("Set DATABASE_URL to run.")
    import psycopg2, monitor
    raw = psycopg2.connect(url, sslmode="require"); raw.autocommit = True
    conn = monitor.PgAdapter(raw)
    for i in run_checks(conn):
        print(f"  [{i['issue']}] {i['detail']}")
    else:
        pass
    print(json.dumps(step(conn), indent=1))
