"""
trade_journal.py — Full attribution journal for broker (Alpaca paper) option trades.

You can't optimize what you don't measure. This records EVERY broker option trade
with the attributes that matter — setup, quality band, DTE, sector, IV, momentum,
exit reason — so that after a couple weeks of the autonomous run we can answer the
optimization questions with EVIDENCE, not guesses:

  • Do A-grade setups actually beat B/C? (validates / refutes the quality gate)
  • Which DTE bucket survives theta best?
  • Which sectors' options work?
  • Does stronger underlying momentum = better option outcome?
  • How do trades end — take-profit, stop, or theta time-stop?

log_entry() on every buy, log_exit() on every close (matched by contract symbol),
analyze() returns the breakdowns for the dashboard.
"""

from __future__ import annotations

from datetime import datetime, date, timezone


# Extra winner-gate / meta-label feature columns, added by _migrate() so existing
# databases pick them up without a manual schema change. Every entry records these
# (even when the gate is OFF) so the meta-model has training data to learn from.
_EXTRA_COLS = [
    ("rv_at_entry", "REAL"),    # realized vol at entry (for strike reachability)
    ("rng_pos", "REAL"),        # position in 20-day range 0..1 (chase detector)
    ("in_uptrend", "INTEGER"),  # price>sma50>sma200 at entry
    ("reach", "REAL"),          # otm_pct / expected_move (strike reachability)
    ("gate_passed", "INTEGER"), # would the winner gate have taken this trade?
    ("gate_score", "REAL"),     # winner gate 0..100 score (also for sizing later)
]


def _migrate(conn):
    """Additively add the winner-gate feature columns to an existing table.
    Each ALTER runs in its own try/except so 'column already exists' is harmless
    (works on both the sqlite and the Postgres-adapter backends)."""
    for col, typ in _EXTRA_COLS:
        try:
            conn.execute(f"ALTER TABLE broker_trade_journal ADD COLUMN {col} {typ}")
        except Exception:
            pass


def _ensure(conn):
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS broker_trade_journal (
                contract_symbol  TEXT PRIMARY KEY,
                underlying       TEXT,
                sector           TEXT,
                setup            TEXT,
                quality_score    REAL,
                quality_band     TEXT,
                dte_at_entry     INTEGER,
                iv_at_entry      REAL,
                otm_pct          REAL,
                mom_6m           REAL,
                mom_3m           REAL,
                entry_premium    REAL,
                qty              INTEGER,
                cost             REAL,
                opened_at        TEXT,
                status           TEXT DEFAULT 'OPEN',
                exit_reason      TEXT,
                exit_premium     REAL,
                pnl_pct          REAL,
                pnl_dollars      REAL,
                hold_days        INTEGER,
                closed_at        TEXT,
                rv_at_entry      REAL,
                rng_pos          REAL,
                in_uptrend       INTEGER,
                reach            REAL,
                gate_passed      INTEGER,
                gate_score       REAL
            )
        """)
        _migrate(conn)
    except Exception as e:
        print(f"  [journal] table init failed: {e}")


def _band(score) -> str:
    try:
        s = float(score or 0)
    except Exception:
        return "?"
    return ("A" if s >= 75 else "B" if s >= 60 else "C" if s >= 50 else "D")


def log_entry(conn, *, contract_symbol, underlying, setup="momentum_call",
              quality_score=None, sector="", dte=None, iv=None, otm_pct=None,
              mom_6m=None, mom_3m=None, entry_premium=None, qty=1, cost=None,
              rv_at_entry=None, rng_pos=None, in_uptrend=None, reach=None,
              gate_passed=None, gate_score=None):
    """Record a new broker option position with full attribution.

    The winner-gate features (rv_at_entry, rng_pos, in_uptrend, reach,
    gate_passed, gate_score) are recorded on EVERY entry — even when the gate is
    not enforcing — so the meta-model accumulates labelled training data and we
    can measure, in shadow, how the gate WOULD have performed."""
    _ensure(conn)

    def _i(b):
        return None if b is None else (1 if b else 0)

    try:
        conn.execute(
            "INSERT INTO broker_trade_journal "
            "(contract_symbol, underlying, sector, setup, quality_score, "
            " quality_band, dte_at_entry, iv_at_entry, otm_pct, mom_6m, mom_3m, "
            " entry_premium, qty, cost, opened_at, status, "
            " rv_at_entry, rng_pos, in_uptrend, reach, gate_passed, gate_score) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'OPEN', ?,?,?,?,?,?) "
            "ON CONFLICT (contract_symbol) DO NOTHING",
            (contract_symbol, str(underlying or "").upper(), sector or "",
             setup, quality_score, _band(quality_score),
             int(dte) if dte is not None else None,
             float(iv) if iv is not None else None,
             float(otm_pct) if otm_pct is not None else None,
             float(mom_6m) if mom_6m is not None else None,
             float(mom_3m) if mom_3m is not None else None,
             float(entry_premium) if entry_premium is not None else None,
             int(qty), float(cost) if cost is not None else None,
             datetime.now(timezone.utc).isoformat(),
             float(rv_at_entry) if rv_at_entry is not None else None,
             float(rng_pos) if rng_pos is not None else None,
             _i(in_uptrend),
             float(reach) if reach is not None else None,
             _i(gate_passed),
             float(gate_score) if gate_score is not None else None)
        )
    except Exception as e:
        print(f"  [journal] log_entry failed: {e}")


def log_exit(conn, contract_symbol, exit_reason, pnl_pct, pnl_dollars,
             exit_premium=None):
    """Match an open journal row by symbol and record the exit."""
    _ensure(conn)
    try:
        row = conn.execute(
            "SELECT opened_at, entry_premium, cost, qty FROM broker_trade_journal "
            "WHERE contract_symbol=? AND status='OPEN'", (contract_symbol,)
        ).fetchone()
        hold_days = 0
        _cost = None
        if row:
            def _g(k, i):
                return row.get(k) if hasattr(row, "get") else row[i]
            opened = _g("opened_at", 0)
            try:
                od = datetime.fromisoformat(str(opened).replace("Z", "+00:00")).date()
                hold_days = (date.today() - od).days
            except Exception:
                hold_days = 0
            try:
                _cost = float(_g("cost", 2) or 0)
            except Exception:
                _cost = None

        # ── P&L reconciliation ────────────────────────────────────────────────
        # A long option can never lose more than its premium, so clamp pct to
        # >= -100. And DERIVE the dollar P&L from THIS row's own entry cost and
        # the % move, rather than trusting the broker's position-level figure
        # (which can reflect a different contract count than this row logged and
        # produced impossible losses, e.g. -$290 on a $130 ticket on 2026-06-12).
        pct = max(-100.0, float(pnl_pct))
        if _cost and _cost > 0:
            dollars = round(pct / 100.0 * _cost, 2)
        else:
            dollars = round(max(float(pnl_dollars), -abs(_cost or 0))
                            if _cost else float(pnl_dollars), 2)
        conn.execute(
            "UPDATE broker_trade_journal SET status='CLOSED', exit_reason=?, "
            "exit_premium=?, pnl_pct=?, pnl_dollars=?, hold_days=?, closed_at=? "
            "WHERE contract_symbol=? AND status='OPEN'",
            (exit_reason, exit_premium, round(pct, 1), dollars, hold_days,
             datetime.now(timezone.utc).isoformat(), contract_symbol)
        )
    except Exception as e:
        print(f"  [journal] log_exit failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS — the breakdowns that drive optimization
# ══════════════════════════════════════════════════════════════════════════════
def _stats(rows) -> dict:
    """rows = list of (pnl_pct, pnl_dollars). Return win rate / avg / expectancy."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    pcts = [r[0] for r in rows if r[0] is not None]
    dols = [r[1] for r in rows if r[1] is not None]
    wins = [p for p in pcts if p > 0]
    losses = [p for p in pcts if p <= 0]
    wr = len(wins) / len(pcts) * 100 if pcts else 0
    avg_w = sum(wins) / len(wins) if wins else 0
    avg_l = sum(losses) / len(losses) if losses else 0
    exp = sum(pcts) / len(pcts) if pcts else 0
    return {
        "n": n, "win_rate": round(wr, 0),
        "avg_win": round(avg_w, 0), "avg_loss": round(avg_l, 0),
        "expectancy": round(exp, 1), "total_pnl": round(sum(dols), 2),
    }


def analyze(conn) -> dict:
    """Return overall + per-dimension performance breakdowns for closed trades."""
    _ensure(conn)
    try:
        rows = conn.execute(
            "SELECT quality_band, dte_at_entry, sector, mom_6m, exit_reason, "
            "pnl_pct, pnl_dollars FROM broker_trade_journal WHERE status='CLOSED'"
        ).fetchall()
    except Exception:
        rows = []

    def g(r, i):
        return r.get(["quality_band","dte_at_entry","sector","mom_6m",
                      "exit_reason","pnl_pct","pnl_dollars"][i]) if hasattr(r,"get") else r[i]

    closed = [(g(r,0), g(r,1), g(r,2), g(r,3), g(r,4), g(r,5), g(r,6)) for r in rows]
    pnl = [(c[5], c[6]) for c in closed]

    def bucket(keyfn):
        out = {}
        for c in closed:
            k = keyfn(c)
            out.setdefault(k, []).append((c[5], c[6]))
        return {k: _stats(v) for k, v in out.items()}

    def dte_b(c):
        d = c[1]
        if d is None: return "?"
        return "≤7d" if d <= 7 else "8-14d" if d <= 14 else "15-21d" if d <= 21 else "22d+"
    def mom_b(c):
        m = c[3]
        if m is None: return "?"
        return "<30%" if m < 0.30 else "30-60%" if m < 0.60 else "60-100%" if m < 1.0 else "100%+"

    # open positions count
    try:
        n_open = conn.execute("SELECT COUNT(*) FROM broker_trade_journal "
                              "WHERE status='OPEN'").fetchone()
        n_open = int(n_open[0] if not hasattr(n_open,"get") else n_open.get("count",0) or 0)
    except Exception:
        n_open = 0

    return {
        "overall": _stats(pnl),
        "n_open": n_open,
        "by_quality": bucket(lambda c: c[0] or "?"),
        "by_dte": bucket(dte_b),
        "by_sector": bucket(lambda c: c[2] or "?"),
        "by_momentum": bucket(mom_b),
        "by_exit": bucket(lambda c: c[4] or "?"),
    }


def count_opened_today(conn) -> int:
    """How many broker option positions were opened today (UTC date)?

    Used by the data-collection floor: if strict gates kept the bot from
    trading, top up the day to a minimum sample (those extra trades are tagged
    'momentum_call_explore' so they never pollute the real-strategy stats).
    """
    _ensure(conn)
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        row = conn.execute(
            "SELECT COUNT(*) FROM broker_trade_journal WHERE opened_at LIKE ?",
            (today + "%",)
        ).fetchone()
        if row is None:
            return 0
        return int(row[0] if not hasattr(row, "get") else row.get("count", 0) or 0)
    except Exception:
        return 0


def recent(conn, limit: int = 40) -> list:
    _ensure(conn)
    try:
        rows = conn.execute(
            "SELECT contract_symbol, underlying, setup, quality_band, "
            "quality_score, dte_at_entry, entry_premium, qty, cost, status, "
            "exit_reason, pnl_pct, pnl_dollars, hold_days, opened_at "
            "FROM broker_trade_journal ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
    except Exception:
        return []
    out = []
    keys = ["contract_symbol","underlying","setup","quality_band","quality_score",
            "dte_at_entry","entry_premium","qty","cost","status","exit_reason",
            "pnl_pct","pnl_dollars","hold_days","opened_at"]
    for r in rows:
        out.append({k: (r.get(k) if hasattr(r,"get") else r[i])
                    for i, k in enumerate(keys)})
    return out
