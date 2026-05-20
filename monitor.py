#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor.py — Background position monitor for GitHub Actions
============================================================
Runs on a schedule via GitHub Actions (see .github/workflows/monitor.yml).
Connects to Supabase, checks every open paper position against live 1-minute
intraday data, and closes any position that has hit its stop-loss or target.
Also optionally opens new trades during PRIME windows if high-conviction scan
signals are cached in the database.

Required environment variable:
  DATABASE_URL  — Full PostgreSQL connection URL, e.g.:
    postgresql://postgres.PROJECT:PASSWORD@HOST:6543/postgres

Usage (local test):
  set DATABASE_URL=postgresql://...
  python monitor.py
"""

import os
import re as _re
import sys
import traceback
from datetime import datetime

# ── make sure trading_scanner is importable from the same directory ───────────
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import trading_scanner as ts

# ── PostgreSQL adapter (identical copy from app.py) ───────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False


class _PgRow:
    __slots__ = ("_cols", "_vals")

    def __init__(self, cols, vals):
        self._cols = list(cols)
        self._vals = list(vals)

    def __getitem__(self, k):
        if isinstance(k, int):
            return self._vals[k]
        return self._vals[self._cols.index(k)]

    def keys(self):
        return self._cols

    def get(self, k, default=None):
        try:
            return self._vals[self._cols.index(k)]
        except (ValueError, IndexError):
            return default

    def __iter__(self):
        return iter(self._vals)


class _PgCursor:
    def __init__(self, cur):
        self._cur = cur

    def _cols(self):
        return [d[0] for d in self._cur.description] if self._cur.description else []

    def fetchone(self):
        row = self._cur.fetchone()
        return None if row is None else _PgRow(self._cols(), row)

    def fetchall(self):
        cols = self._cols()
        return [_PgRow(cols, r) for r in self._cur.fetchall()]

    def __iter__(self):
        cols = self._cols()
        for row in self._cur:
            yield _PgRow(cols, row)


class _FakeCursor:
    def __init__(self, value):
        self._v = value

    def fetchone(self):
        return (self._v,)

    def fetchall(self):
        return [(self._v,)]


class PgAdapter:
    """Wraps psycopg2 connection to present a sqlite3-compatible API."""

    _AUTOINCREMENT     = _re.compile(r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b", _re.I)
    _INSERT_OR_REPLACE = _re.compile(r"\bINSERT\s+OR\s+REPLACE\b", _re.I)
    _LAST_ROWID        = _re.compile(r"SELECT\s+last_insert_rowid\(\)", _re.I)
    _INTO_TABLE        = _re.compile(r"\bINTO\s+(calls|portfolio)\b", _re.I)

    def __init__(self, pg_conn):
        self._conn    = pg_conn
        self._last_id = None
        # autocommit=True — each statement is its own transaction; a failed
        # statement never poisons the connection for subsequent ones.
        try:
            self._conn.autocommit = True
        except Exception:
            pass

    def _adapt(self, sql: str) -> str:
        sql = sql.replace("?", "%s")
        sql = self._AUTOINCREMENT.sub("BIGSERIAL PRIMARY KEY", sql)
        sql = sql.replace("DATETIME", "TIMESTAMP")
        return sql

    def execute(self, sql: str, params=()):
        if self._LAST_ROWID.search(sql.strip()):
            return _FakeCursor(self._last_id)

        # psycopg2 blocks EVERY command — including cursor() — when the
        # connection is in a failed-transaction state.  Since we commit()
        # after every successful execute(), rolling back here only ever
        # clears a previously failed statement, never valid in-progress work.
        try:
            self._conn.rollback()
        except Exception:
            pass

        is_replace = bool(self._INSERT_OR_REPLACE.search(sql))
        adapted    = self._adapt(sql)

        if is_replace:
            adapted = self._INSERT_OR_REPLACE.sub("INSERT", adapted)
            adapted = adapted.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"

        is_insert     = adapted.strip().upper().startswith("INSERT")
        add_returning = (
            is_insert
            and not is_replace
            and "RETURNING" not in adapted.upper()
            and bool(self._INTO_TABLE.search(adapted))
        )
        if add_returning:
            adapted = adapted.rstrip().rstrip(";") + " RETURNING id"

        cur = self._conn.cursor()
        try:
            cur.execute(adapted, params or ())
        except Exception:
            # Roll back the failed transaction so the connection stays usable
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

        if add_returning:
            row           = cur.fetchone()
            self._last_id = row[0] if row else None

        self.commit()
        return _PgCursor(cur)

    def executescript(self, sql: str):
        adapted = self._adapt(sql)
        for stmt in _re.split(r";[ \t]*\n?", adapted):
            stmt = stmt.strip()
            if not stmt or stmt.startswith("--"):
                continue
            try:
                self._conn.rollback()
            except Exception:
                pass
            try:
                cur = self._conn.cursor()
                cur.execute(stmt)
                self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass

    def commit(self):
        try:
            if not getattr(self._conn, "autocommit", False):
                self._conn.commit()
        except Exception:
            pass

    def close(self):
        self._conn.close()


# ── Telegram helper ───────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    """Send a Telegram message if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set.

    Returns True on success, False if credentials are missing or the call fails.
    Never raises — Telegram failure must never crash the monitor.
    """
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID",   "").strip()
    if not token or not chat_id:
        return False
    try:
        import requests as _req
        resp = _req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        ok = resp.status_code == 200
        if not ok:
            print(f"  [Telegram] send failed {resp.status_code}: {resp.text[:120]}")
        return ok
    except Exception as e:
        print(f"  [Telegram] error: {e}")
        return False


# ── connection helper ─────────────────────────────────────────────────────────

def get_connection() -> PgAdapter:
    """Build a Supabase connection from DATABASE_URL env var."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise EnvironmentError(
            "DATABASE_URL environment variable is not set.\n"
            "Add it as a GitHub Secret named DATABASE_URL."
        )
    if not _PG_AVAILABLE:
        raise ImportError("psycopg2-binary is not installed.")
    raw = psycopg2.connect(url, sslmode="require")
    return PgAdapter(raw)  # PgAdapter.__init__ sets autocommit=True


# ── main logic ────────────────────────────────────────────────────────────────

def main():
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"  Position Monitor  —  {now_utc}")
    print(f"{'='*60}")

    # ── Telegram connection test (manual / workflow_dispatch triggers only) ───
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        print("  [Telegram] Sending connection test...")
        sent = send_telegram(
            f"✅ <b>Breakout Bot — Monitor Connected</b>\n"
            f"Telegram alerts are working!\n"
            f"Time: {now_utc}\n"
            f"Next: alerts fire automatically when stops/targets are hit."
        )
        print(f"  [Telegram] {'✓ message sent!' if sent else '✗ FAILED — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID secrets'}")

    # ── Market window check ───────────────────────────────────────────────────
    session = ts.MarketClock.get_session()
    quality = session.get("quality", "CLOSED")
    print(f"  Market session : {quality} — {session.get('name','')}")
    print(f"  ET time        : {session.get('time_et','')}")

    if quality == "CLOSED":
        print("  Market is closed — nothing to do.\n")
        return

    # ── Connect to Supabase ───────────────────────────────────────────────────
    print("  Connecting to Supabase...")
    try:
        conn = get_connection()
    except Exception as exc:
        print(f"  ERROR: Could not connect to database: {exc}")
        send_telegram(f"⚠️ <b>Monitor DB Error</b>\n{exc}")
        sys.exit(1)

    # ── Load paper engine ─────────────────────────────────────────────────────
    paper     = ts.PaperTradingEngine(conn)
    positions = paper.open_positions
    summary   = paper.get_summary()

    print(f"\n  Portfolio status:")
    print(f"    Available cash : ${summary['available_cash']:,.2f}")
    print(f"    Invested       : ${summary['invested_value']:,.2f}")
    print(f"    Realized P&L   : ${summary['realized_pnl']:+,.2f}")
    print(f"    Open positions : {len(positions)}/{summary['max_positions']}")

    if not positions:
        print("\n  No open positions to monitor.")
    else:
        print(f"\n  Checking {len(positions)} position(s) against live prices...")
        for p in positions:
            t1_flag = " [T1✓]" if p.get("t1_hit") else ""
            print(f"    {p['ticker']:8s}  entry=${p.get('entry_price',0):.2f}"
                  f"  stop=${p.get('stop_loss',0):.2f}"
                  f"  target=${p.get('target_price',0):.2f}{t1_flag}")

        # ── Check stops and targets ───────────────────────────────────────────
        try:
            closed = paper.check_stops_and_targets()
        except Exception as exc:
            print(f"\n  ERROR in check_stops_and_targets: {exc}")
            traceback.print_exc()
            send_telegram(f"⚠️ <b>Monitor Error</b>\n{exc}")
            closed = []

        if closed:
            print(f"\n  {'─'*50}")
            print(f"  POSITIONS CLOSED ({len(closed)}):")

            # ── Telegram notification for each closed position ────────────────
            summary_after = paper.get_summary()
            for c in closed:
                reason = c.get("exit_reason", "UNKNOWN")
                price  = c.get("exit_price",  0) or 0
                pnl    = c.get("net_pnl",     0) or 0
                sign   = "+" if pnl >= 0 else ""

                if "TARGET" in reason:
                    emoji = "🎯"
                elif "TIME" in reason:
                    emoji = "⏱"
                elif "T1" in reason:
                    emoji = "📍"
                else:
                    emoji = "🛑"

                log_line = (f"    [{reason}] {c['ticker']:8s}"
                            f"  exit=${price:.2f}"
                            f"  net_pnl={sign}${pnl:.2f}")
                print(log_line)

                tg_msg = (
                    f"{emoji} <b>{c['ticker']}</b> closed\n"
                    f"Reason : {reason}\n"
                    f"Exit   : ${price:.2f}\n"
                    f"Net P&L: {sign}${pnl:.2f}\n"
                    f"Cash   : ${summary_after['available_cash']:,.2f}"
                    f" | Realized: ${summary_after['realized_pnl']:+,.2f}\n"
                    f"Session: {quality}  ({session.get('name','')})"
                )
                sent = send_telegram(tg_msg)
                print(f"    [Telegram] {'✓ sent' if sent else '✗ not configured'}")

            print(f"  {'─'*50}")
        else:
            print(f"\n  All positions within bounds — no exits triggered.")

        # ── Trailing stop update notification ─────────────────────────────────
        # Re-read positions after any stop updates to show new stop levels
        updated = paper.open_positions
        for p in updated:
            if p.get("t1_hit") and p.get("trailing_stop"):
                print(f"    {p['ticker']:8s}  trailing stop now: ${p['trailing_stop']:.2f}"
                      f"  (T1 hit ✓, highest: ${p.get('highest_since_t1',0):.2f})")

    # ── Skip new entries during AVOID window ──────────────────────────────────
    if quality == "AVOID":
        print(f"\n  AVOID window — skipping new entry scan.")
        conn.close()
        print("\n  Done.\n")
        return

    print(f"\n  Done (session={quality}).\n")
    conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}")
        traceback.print_exc()
        send_telegram(f"💥 <b>Monitor Fatal Error</b>\n{str(exc)[:400]}")
        sys.exit(1)
