"""
virtual_portfolio.py — Virtual $1k bankroll layered on the real Alpaca paper account.

The Alpaca paper account holds ~$100k, but the bot sizes every trade as if it had
$1,000 (ACCOUNT_EQUITY_OVERRIDE). This module tracks that $1k as a *virtual
portfolio* and auto-resets it when it's blown:

    virtual_value = start_equity + (real_equity_now - baseline_real_at_start)

Because the bot is the only thing trading the account and bets are tiny, every
dollar of realized/unrealized P&L on the real account IS the virtual portfolio's
P&L. When the virtual value falls to the floor (default $0 — i.e. the account has
actually lost the full $1k since this run began), we DON'T stop:

    1. mark the run BLOWN (record final value),
    2. bump the lifetime reset counter,
    3. open a FRESH $1k run off the current real equity.

With $100k real backing a $1k bankroll that's ~100 "lives". The blow-up counter
is the most honest survival stat you can get: "the strategy blew N x $1k accounts."

Sizing is intentionally NOT changed here — the override stays a fixed $1k throttle,
so each run blows after ~$1k of cumulative loss, which keeps the counter meaningful.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _ensure(conn):
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS broker_virtual_portfolio (
                id            INTEGER PRIMARY KEY,
                start_equity  REAL,
                baseline_real REAL,
                started_at    TEXT,
                status        TEXT DEFAULT 'ACTIVE',   -- ACTIVE | BLOWN
                ended_at      TEXT,
                final_value   REAL,
                final_real    REAL
            )
        """)
    except Exception as e:
        print(f"  [virtual_portfolio] table init failed: {e}")


def _g(row, key, idx, default=None):
    """Read a column from a DB row that may be a dict (psycopg) or tuple (sqlite)."""
    if row is None:
        return default
    if hasattr(row, "get"):
        v = row.get(key)
        return v if v is not None else default
    try:
        return row[idx]
    except Exception:
        return default


def _active(conn):
    _ensure(conn)
    try:
        row = conn.execute(
            "SELECT id, start_equity, baseline_real, started_at "
            "FROM broker_virtual_portfolio WHERE status='ACTIVE' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return {
        "id":            int(_g(row, "id", 0, 0) or 0),
        "start_equity":  float(_g(row, "start_equity", 1, 0) or 0),
        "baseline_real": float(_g(row, "baseline_real", 2, 0) or 0),
        "started_at":    _g(row, "started_at", 3, ""),
    }


def _blow_count(conn) -> int:
    try:
        row = conn.execute("SELECT COUNT(*) FROM broker_virtual_portfolio "
                           "WHERE status='BLOWN'").fetchone()
        return int(_g(row, "count", 0, 0) or 0)
    except Exception:
        return 0


def _next_id(conn) -> int:
    try:
        row = conn.execute("SELECT MAX(id) FROM broker_virtual_portfolio").fetchone()
        mx = _g(row, "max", 0, 0)
        return int(mx) + 1 if mx else 1
    except Exception:
        return 1


def _create(conn, pid, start, baseline_real):
    try:
        conn.execute(
            "INSERT INTO broker_virtual_portfolio "
            "(id, start_equity, baseline_real, started_at, status) "
            "VALUES (?,?,?,?, 'ACTIVE')",
            (int(pid), float(start), float(baseline_real),
             datetime.now(timezone.utc).isoformat())
        )
    except Exception as e:
        print(f"  [virtual_portfolio] create run #{pid} failed: {e}")


def step(conn, real_equity, start: float = 1000.0, floor: float = 0.0) -> dict:
    """
    Advance the virtual portfolio one cycle. Initializes run #1 on first call.
    Auto-resets (and increments the counter) when value <= floor.

    Returns a dict with: reset (bool), active_id, value (current $), blow_count
    (lifetime BLOWN runs), start_equity. On a reset it also includes blown_id,
    blown_value and new_id.
    """
    _ensure(conn)
    real_equity = float(real_equity or 0)
    start = float(start or 1000.0)

    act = _active(conn)
    if act is None:
        pid = _next_id(conn)
        _create(conn, pid, start, real_equity)
        return {"reset": False, "active_id": pid, "value": round(start, 2),
                "start_equity": start, "blow_count": _blow_count(conn)}

    val = start + (real_equity - act["baseline_real"])

    if val <= floor:
        try:
            conn.execute(
                "UPDATE broker_virtual_portfolio SET status='BLOWN', "
                "ended_at=?, final_value=?, final_real=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), round(val, 2),
                 round(real_equity, 2), act["id"])
            )
        except Exception as e:
            print(f"  [virtual_portfolio] mark blown failed: {e}")
        new_id = _next_id(conn)
        _create(conn, new_id, start, real_equity)
        return {"reset": True, "blown_id": act["id"], "blown_value": round(val, 2),
                "new_id": new_id, "active_id": new_id, "value": round(start, 2),
                "start_equity": start, "blow_count": _blow_count(conn)}

    return {"reset": False, "active_id": act["id"], "value": round(val, 2),
            "start_equity": start, "blow_count": _blow_count(conn)}


def summary(conn, real_equity, start: float = 1000.0) -> dict:
    """Read-only snapshot for the dashboard (does NOT reset)."""
    _ensure(conn)
    real_equity = float(real_equity or 0)
    start = float(start or 1000.0)
    act = _active(conn)
    cur_id = cur_val = pnl = None
    if act:
        cur_id = act["id"]
        cur_val = start + (real_equity - act["baseline_real"])
        pnl = cur_val - start

    runs = []
    try:
        rows = conn.execute(
            "SELECT id, start_equity, final_value, status, started_at, ended_at "
            "FROM broker_virtual_portfolio ORDER BY id DESC LIMIT 25"
        ).fetchall()
        for r in rows:
            runs.append({
                "id":           _g(r, "id", 0),
                "start_equity": _g(r, "start_equity", 1),
                "final_value":  _g(r, "final_value", 2),
                "status":       _g(r, "status", 3),
                "started_at":   _g(r, "started_at", 4),
                "ended_at":     _g(r, "ended_at", 5),
            })
    except Exception:
        pass

    return {
        "active_id":     cur_id,
        "current_value": round(cur_val, 2) if cur_val is not None else None,
        "pnl":           round(pnl, 2) if pnl is not None else None,
        "blow_count":    _blow_count(conn),
        "start_equity":  start,
        "real_equity":   real_equity,
        "lives_left":    int(real_equity / start) if start else 0,
        "runs":          runs,
    }


if __name__ == "__main__":
    # Tiny in-memory smoke test of the blow-up + reset + counter logic.
    import sqlite3

    class _C:
        def __init__(self):
            self.c = sqlite3.connect(":memory:")
        def execute(self, q, p=()):
            return self.c.execute(q.replace("?", "?"), p)

    c = _C()
    eq = 100_000.0
    print(step(c, eq, start=1000))                 # init run #1
    eq -= 600;  print(step(c, eq, start=1000))     # value $400
    eq -= 500;  print(step(c, eq, start=1000))     # value -$100 -> BLOWN, new run #2
    eq += 300;  print(step(c, eq, start=1000))     # run #2 value $1300
    print("SUMMARY:", summary(c, eq, start=1000))
