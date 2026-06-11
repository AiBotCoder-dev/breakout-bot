"""
claude_pm.py — Claude as Portfolio Manager over the quant bot.

ARCHITECTURE
------------
The bot is the EXECUTOR (rules, every 10 min, no emotion). Claude is the PM:
a scheduled Claude agent runs every weekday pre-market, reads the full data
pack (everything the bot has collected), reasons about it, and writes
DIRECTIVES to the claude_directives table. monitor.py executes pending
directives next cycle — with hard guardrails — and tags every resulting trade
'claude_pm_*' in the journal, SEPARATE from the bot's own trades, so the
"is Claude actually better than the bot?" question is answered with data.

DIRECTIVE ACTIONS (what the PM may order):
  OPEN_CALL <ticker>     buy a call on ticker (bot picks contract + quality-checks)
  OPEN_PUT  <ticker>     buy a put on ticker  (same)
  CLOSE_OPTION <symbol>  close one OCC option position now
  CLOSE_ALL_OPTIONS      flatten the whole options book
  PAUSE_ENTRIES          no new bot entries this day (Claude sees danger)
  NOTE                   no trade — just send the user Claude's market read

GUARDRAILS (enforced by the executor in monitor.py):
  • paper account only (executor lives inside the broker block)
  • directives expire after 24h — a stale market view must never execute late
  • max 3 PM-directed opens per day; PM opens sized at 60% budget (secondary)
  • every execution Telegrams the user WITH Claude's rationale

CLI (used by the scheduled Claude agent):
  python claude_pm.py pack                 -> full JSON data pack to stdout
  python claude_pm.py decide '<json>'      -> insert directives
       '[{"action":"OPEN_CALL","ticker":"NVDA","rationale":"..."}, ...]'
  python claude_pm.py scorecard            -> Claude-PM vs bot attribution
  python claude_pm.py pending              -> list unexecuted directives
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta

VALID_ACTIONS = {"OPEN_CALL", "OPEN_PUT", "CLOSE_OPTION", "CLOSE_ALL_OPTIONS",
                 "PAUSE_ENTRIES", "NOTE"}
DIRECTIVE_TTL_HOURS = 24
MAX_PM_OPENS_PER_DAY = 3


def _conn():
    from monitor import get_connection
    return get_connection()


def ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claude_directives (
            id          BIGSERIAL PRIMARY KEY,
            created_at  TEXT,
            action      TEXT,
            ticker      TEXT,
            symbol      TEXT,
            rationale   TEXT,
            status      TEXT DEFAULT 'PENDING',   -- PENDING/EXECUTED/FAILED/EXPIRED/SKIPPED
            executed_at TEXT,
            result      TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS broker_state_snapshot (
            snap_key   TEXT PRIMARY KEY,          -- always 'latest'
            taken_at   TEXT,
            payload    TEXT                       -- JSON: account + positions
        )
    """)


# ── data pack: everything Claude needs to act as PM ──────────────────────────
def build_pack(conn) -> dict:
    pack = {"generated_at_utc": datetime.now(timezone.utc).isoformat()}

    def rows(sql, args=(), limit=None):
        try:
            rs = conn.execute(sql, args).fetchall()
            out = [dict(r) if hasattr(r, "keys") else list(r) for r in rs]
            return out[:limit] if limit else out
        except Exception as e:
            return [{"error": str(e)[:120]}]

    # live broker state (snapshotted by monitor each cycle — no Alpaca keys needed)
    snap = rows("SELECT taken_at, payload FROM broker_state_snapshot WHERE snap_key='latest'")
    if snap and "payload" in (snap[0] or {}):
        try:
            pack["broker_state"] = {"taken_at": snap[0]["taken_at"],
                                    **json.loads(snap[0]["payload"])}
        except Exception:
            pack["broker_state"] = {"raw": snap[0]}
    else:
        pack["broker_state"] = {"note": "no snapshot yet"}

    pack["equity_curve"] = rows(
        "SELECT snapshot_date, equity, open_options FROM broker_equity_log "
        "ORDER BY snapshot_date DESC LIMIT 30")
    pack["trade_journal_recent"] = rows(
        "SELECT contract_symbol, underlying, setup, quality_score, entry_premium, "
        "qty, cost, opened_at, exit_reason, pnl_pct, pnl_dollars, closed_at "
        "FROM broker_trade_journal ORDER BY opened_at DESC LIMIT 40")
    pack["best_options_scan"] = rows(
        "SELECT ticker, option_type, strike, expiry, dte, premium, quality_score, "
        "quality_grade, decision, sources, thesis_pct, scanned_at "
        "FROM best_options_trades ORDER BY quality_score DESC LIMIT 15")
    pack["overnight_edge"] = rows(
        "SELECT ticker, status, entry_date, entry_price, notional, exit_date, "
        "exit_price, pnl, pnl_pct FROM overnight_edge_log ORDER BY id DESC LIMIT 15")
    pack["vip_posts_7d"] = rows(
        "SELECT vip_name, text, tickers, sentiment, fetched_at FROM vip_posts "
        "WHERE fetched_at >= ? ORDER BY fetched_at DESC LIMIT 20",
        ((datetime.utcnow() - timedelta(days=7)).isoformat(),))
    pack["news_recent"] = rows(
        "SELECT * FROM paper_news_events ORDER BY 1 DESC LIMIT 25")
    pack["my_past_directives"] = rows(
        "SELECT created_at, action, ticker, symbol, rationale, status, result "
        "FROM claude_directives ORDER BY id DESC LIMIT 20")

    # live-computed context (best effort; each guarded)
    try:
        from macro_engine import upcoming_events, event_risk
        pack["macro"] = {"risk": event_risk(), "upcoming": upcoming_events()[:6]}
    except Exception as e:
        pack["macro"] = {"error": str(e)[:120]}
    try:
        from momentum_strategy import MomentumStrategy
        pack["momentum_top10"] = [
            {k: r.get(k) for k in ("ticker", "price", "mom_6m", "mom_3m")}
            for r in MomentumStrategy(conn).rank(top_n=10, min_mom_6m=0.05)]
    except Exception as e:
        pack["momentum_top10"] = [{"error": str(e)[:120]}]
    try:
        from panic_detector import PanicDetector
        st = PanicDetector(conn).status()
        pack["panic"] = {"snap": st.get("snap"),
                         "firing": [s["signature"] for s in st.get("signatures", [])
                                    if s.get("currently_fired") or s.get("active_in_db")]}
    except Exception as e:
        pack["panic"] = {"error": str(e)[:120]}

    return pack


# ── directive CRUD ────────────────────────────────────────────────────────────
def insert_directives(conn, directives: list) -> dict:
    ensure_tables(conn)
    now = datetime.now(timezone.utc).isoformat()
    ok, rejected = 0, []
    for d in directives:
        action = str(d.get("action", "")).upper().strip()
        if action not in VALID_ACTIONS:
            rejected.append({**d, "why": "invalid action"}); continue
        rat = str(d.get("rationale", "")).strip()
        if not rat:
            rejected.append({**d, "why": "rationale required"}); continue
        conn.execute(
            "INSERT INTO claude_directives "
            "(created_at, action, ticker, symbol, rationale) VALUES (?,?,?,?,?)",
            (now, action,
             str(d.get("ticker", "") or "").upper()[:10],
             str(d.get("symbol", "") or "").upper()[:32], rat[:800]))
        ok += 1
    return {"inserted": ok, "rejected": rejected}


def pending(conn) -> list:
    ensure_tables(conn)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=DIRECTIVE_TTL_HOURS)).isoformat()
    # expire stale first
    try:
        conn.execute("UPDATE claude_directives SET status='EXPIRED' "
                     "WHERE status='PENDING' AND created_at < ?", (cutoff,))
    except Exception:
        pass
    rs = conn.execute("SELECT * FROM claude_directives WHERE status='PENDING' "
                      "ORDER BY id ASC").fetchall()
    return [dict(r) if hasattr(r, "keys") else r for r in rs]


def scorecard(conn) -> dict:
    """Claude-PM trades vs bot trades — the 'is Claude actually better?' table."""
    out = {}
    try:
        rs = conn.execute(
            "SELECT CASE WHEN setup LIKE ? THEN 'claude_pm' "
            "ELSE 'bot' END AS who, COUNT(*) n, "
            "SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) wins, "
            "AVG(pnl_pct) avg_pct, SUM(pnl_dollars) total_pnl "
            "FROM broker_trade_journal WHERE closed_at IS NOT NULL "
            "GROUP BY 1", ("claude_pm%",)).fetchall()
        for r in rs:
            d = dict(r) if hasattr(r, "keys") else {}
            who = d.get("who", "?")
            n = int(d.get("n") or 0)
            out[who] = {
                "closed_trades": n,
                "win_rate": round(100 * float(d.get("wins") or 0) / n, 1) if n else 0,
                "avg_return_pct": round(float(d.get("avg_pct") or 0), 1),
                "total_pnl": round(float(d.get("total_pnl") or 0), 2),
            }
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


def _json_default(o):
    return str(o)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pack"
    conn = _conn()
    ensure_tables(conn)
    if cmd == "pack":
        print(json.dumps(build_pack(conn), indent=1, default=_json_default))
    elif cmd == "decide":
        payload = json.loads(sys.argv[2])
        if isinstance(payload, dict):
            payload = [payload]
        print(json.dumps(insert_directives(conn, payload)))
    elif cmd == "pending":
        print(json.dumps(pending(conn), indent=1, default=_json_default))
    elif cmd == "scorecard":
        print(json.dumps(scorecard(conn), indent=1))
    else:
        print(f"unknown command: {cmd}")
    conn.close()
