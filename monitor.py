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

# ── Auto-entry configuration ──────────────────────────────────────────────────
# Raise these thresholds to be more selective; lower them to trade more often.
STOCK_MIN_SCORE          = 75    # explosive_score threshold for stock auto-entry
STOCK_MIN_PROB           = 70    # breakout_prob (%) threshold for stock auto-entry
OPTIONS_MIN_SCORE        = 70    # explosive_score threshold for options auto-entry
OPTIONS_MIN_PROB         = 65    # breakout_prob (%) threshold for options auto-entry
AUTO_MAX_ENTRIES_PER_RUN = 2     # max NEW entries per monitor run (each category)
STOCK_CHASE_LIMIT_PCT    = 3.0   # skip stock entry if price > scan entry by this %

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
        try:
            return self._vals[self._cols.index(k)]
        except ValueError:
            raise KeyError(k)

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
                self.commit()
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


# ── Live price helper ─────────────────────────────────────────────────────────

def _get_live_price(ticker: str):
    """Return the latest trade price for *ticker* via yfinance, or None on failure."""
    try:
        import yfinance as _yf
        price = getattr(_yf.Ticker(ticker).fast_info, "last_price", None)
        if price and float(price) > 0:
            return float(price)
        import pandas as _pd
        hist = _yf.download(ticker, period="1d", interval="1m",
                            progress=False, auto_adjust=True)
        if hist is not None and not hist.empty:
            if isinstance(hist.columns, _pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


# ── Auto-entry: stocks ────────────────────────────────────────────────────────

def auto_enter_stocks(conn, paper, session, quality):
    """
    Automatically open paper stock positions for tickers in the latest breakout
    scan that meet the score/probability thresholds.

    Rules:
      - Only runs during PRIME / NORMAL sessions (caller's responsibility).
      - Skips tickers already held or that have gapped > STOCK_CHASE_LIMIT_PCT above
        the scan entry price (avoids chasing extended moves).
      - At most AUTO_MAX_ENTRIES_PER_RUN new positions per call.
      - PaperTradingEngine.open_position() handles position-limit + cash checks.
    """
    print(f"\n  {'─'*50}")
    print(f"  AUTO-ENTRY — Stocks")

    summary    = paper.get_summary()
    n_open     = len(paper.open_positions)
    cash_avail = summary.get("available_cash", 0)
    max_pos    = summary.get("max_positions", 5)
    slots      = max_pos - n_open

    if slots <= 0:
        print(f"  Portfolio full ({n_open}/{max_pos}) — no new stock entries.")
        return
    if cash_avail < 100:
        print(f"  Cash too low (${cash_avail:.2f}) — skipping stock auto-entry.")
        return

    # ── Read latest scan results ──────────────────────────────────────────────
    try:
        row = conn.execute(
            "SELECT scan_id FROM calls ORDER BY scan_timestamp DESC LIMIT 1"
        ).fetchone()
        if not row:
            print("  No scan data in DB — skipping stock auto-entry.")
            return
        scan_id = row[0]
        rows = conn.execute(
            "SELECT ticker, explosive_score, breakout_prob, pattern_detected, "
            "entry_price, target_price, stop_loss "
            "FROM calls WHERE scan_id=? "
            "ORDER BY explosive_score DESC LIMIT 20",
            (scan_id,)
        ).fetchall()
    except Exception as e:
        print(f"  ERROR reading scan: {e}")
        return

    candidates = [
        r for r in rows
        if float(r.get("explosive_score") or 0) >= STOCK_MIN_SCORE
        and float(r.get("breakout_prob")   or 0) >= STOCK_MIN_PROB
    ]
    print(f"  Scan: {len(rows)} tickers total | "
          f"{len(candidates)} above thresholds "
          f"(score≥{STOCK_MIN_SCORE}, prob≥{STOCK_MIN_PROB}%)")

    held_tickers = {p["ticker"] for p in paper.open_positions}
    new_entries  = 0

    for r in candidates:
        if new_entries >= AUTO_MAX_ENTRIES_PER_RUN:
            print(f"  Reached max new entries ({AUTO_MAX_ENTRIES_PER_RUN}) — stopping.")
            break

        ticker = str(r.get("ticker") or "").upper()
        if not ticker or ticker in held_tickers:
            continue

        # ── Live price ────────────────────────────────────────────────────────
        live_price = _get_live_price(ticker)
        if not live_price:
            live_price = float(r.get("entry_price") or 0)
        if not live_price:
            print(f"    {ticker:8s} — no price data, skip")
            continue

        # ── Chase guard ───────────────────────────────────────────────────────
        scan_entry = float(r.get("entry_price") or live_price)
        if scan_entry > 0:
            drift_pct = (live_price - scan_entry) / scan_entry * 100
            if drift_pct > STOCK_CHASE_LIMIT_PCT:
                print(f"    {ticker:8s} — price drifted +{drift_pct:.1f}% above scan entry, skip (chasing)")
                continue

        stop    = float(r.get("stop_loss")    or live_price * 0.95)
        tgt     = float(r.get("target_price") or live_price * 1.10)
        score   = float(r.get("explosive_score") or 0)
        prob    = float(r.get("breakout_prob")   or 0)
        pattern = str(r.get("pattern_detected")  or "")

        signal = {
            "price":           live_price,
            "stop_price":      stop,
            "tgt_price":       tgt,
            "explosive_score": score,
            "probability":     prob,
            "pattern":         pattern,
        }
        result = paper.open_position(ticker, signal)

        if result.get("success"):
            new_entries += 1
            held_tickers.add(ticker)
            gross   = result.get("gross", 0)
            shares  = result.get("shares", 0)
            print(f"    ✅ AUTO-BUY {ticker:8s}  "
                  f"price=${live_price:.2f}  stop=${stop:.2f}  tgt=${tgt:.2f}  "
                  f"shares={shares:.1f}  cost=${gross:.2f}")
            summ_now = paper.get_summary()
            send_telegram(
                f"🤖 <b>AUTO-BUY: {ticker}</b>\n"
                f"Score   : {score:.0f}  |  Prob: {prob:.0f}%\n"
                f"Pattern : {pattern or '—'}\n"
                f"Entry   : ${live_price:.2f}\n"
                f"Stop    : ${stop:.2f}  |  Target: ${tgt:.2f}\n"
                f"Shares  : {shares:.1f}  |  Cost: ${gross:.2f}\n"
                f"Cash    : ${summ_now.get('available_cash',0):,.2f}\n"
                f"Session : {quality}  ({session.get('name','')})"
            )
        else:
            print(f"    ✗ {ticker:8s} — {result.get('reason', 'failed')}")

    if new_entries == 0:
        print("  No new stock positions opened this run.")
    else:
        print(f"  {new_entries} new stock position(s) opened.")


# ── Auto-entry: options ───────────────────────────────────────────────────────

def auto_enter_options(conn, vix_data, session, quality):
    """
    Automatically open paper options positions using Weekly ATM strategy (5–7 DTE)
    for tickers in the latest breakout scan that meet the thresholds.

    Rules:
      - Skips entirely when VIX regime is EXTREME (options too expensive).
      - Uses ts.get_position_sizing() to choose contract count.
      - Falls back to 1 contract if sizing recommends 0.
      - Skips tickers already held in open options positions.
      - At most AUTO_MAX_ENTRIES_PER_RUN new entries per call.
    """
    print(f"\n  {'─'*50}")
    print(f"  AUTO-ENTRY — Options")

    # ── VIX guard ─────────────────────────────────────────────────────────────
    if vix_data and vix_data.get("regime") == "EXTREME":
        print(f"  VIX EXTREME ({vix_data['vix']:.1f}) — options too expensive, skip.")
        return

    ops_engine  = ts.OptionsPaperEngine(conn)
    ops_summary = ops_engine.get_summary()
    cash_avail  = ops_summary.get("available_cash", 0)

    if cash_avail < 50:
        print(f"  Options cash too low (${cash_avail:.2f}) — skipping.")
        return

    # ── Load scan plays ───────────────────────────────────────────────────────
    try:
        scan_plays = ts.get_scan_options_plays(conn, top_n=10)
    except Exception as e:
        print(f"  ERROR loading options plays: {e}")
        return

    if not scan_plays:
        print("  No scan plays available — skipping options auto-entry.")
        return

    candidates = [
        p for p in scan_plays
        if float(p.get("explosive_score", 0)) >= OPTIONS_MIN_SCORE
        and float(p.get("breakout_prob",   0)) >= OPTIONS_MIN_PROB
    ]
    print(f"  Options plays: {len(scan_plays)} tickers | "
          f"{len(candidates)} above thresholds "
          f"(score≥{OPTIONS_MIN_SCORE}, prob≥{OPTIONS_MIN_PROB}%)")

    held_opt_tickers = {
        str(p.get("ticker", "")).upper()
        for p in ops_engine.get_positions("OPEN")
    }
    closed_hist = ops_engine.get_positions("CLOSED")
    new_entries = 0

    for play in candidates:
        if new_entries >= AUTO_MAX_ENTRIES_PER_RUN:
            print(f"  Reached max new entries ({AUTO_MAX_ENTRIES_PER_RUN}) — stopping.")
            break

        ticker = str(play.get("ticker") or "").upper()
        if not ticker or ticker in held_opt_tickers:
            continue

        suggestions = play.get("suggestions", [])
        if not suggestions:
            print(f"    {ticker:8s} — no strategy suggestions, skip")
            continue

        # Prefer Weekly ATM (highest-probability play)
        sg = next(
            (s for s in suggestions if s.get("strategy") == "Weekly ATM"),
            suggestions[0]
        )

        mid      = float(sg.get("mid",      0) or 0)
        expiry   = str(sg.get("expiry",    "") or "")
        strike   = float(sg.get("strike",   0) or 0)
        opt_type = str(sg.get("opt_type", "call") or "call")
        strategy = str(sg.get("strategy", "Weekly ATM") or "Weekly ATM")
        dte      = int(sg.get("dte",        0) or 0)

        if mid <= 0 or not expiry or strike <= 0:
            print(f"    {ticker:8s} — incomplete strategy data, skip")
            continue

        # ── Dynamic position sizing ───────────────────────────────────────────
        sizing    = ts.get_position_sizing(
            account_cash  = cash_avail,
            mid_price     = mid,
            vix_data      = vix_data,
            closed_trades = closed_hist,
        )
        contracts = max(1, sizing.get("contracts", 1))

        # Ensure we can afford it; try falling back to 1 contract
        cost_total = round(mid * 100 * contracts, 2)
        if cost_total > cash_avail:
            cost_total = round(mid * 100, 2)
            if cost_total > cash_avail:
                print(f"    {ticker:8s} — 1 contract costs ${cost_total:.2f}, "
                      f"cash=${cash_avail:.2f} — skip")
                continue
            contracts = 1

        result = ops_engine.buy(
            ticker          = ticker,
            contract_symbol = f"{ticker}_{expiry}_{opt_type}_{strike:.0f}",
            option_type     = opt_type,
            strike          = strike,
            expiry          = expiry,
            contracts       = contracts,
            entry_price     = mid,
            strategy        = f"AUTO-{strategy}",
        )

        if result.get("ok"):
            new_entries      += 1
            cash_avail       -= cost_total
            held_opt_tickers.add(ticker)

            score = play.get("explosive_score", 0)
            prob  = play.get("breakout_prob",   0)
            pop   = sg.get("pop")
            be    = sg.get("breakeven_move_pct", 0)
            pop_s = f"{pop:.0f}%" if pop is not None else "—"
            vix_m = sizing.get("vix_mult", 1.0)

            print(f"    ✅ AUTO-OPTIONS {ticker:8s}  {strategy}  "
                  f"${strike:.0f}{opt_type[0].upper()} {expiry} ({dte}DTE)  "
                  f"mid=${mid:.2f}  {contracts}×  cost=${cost_total:.2f}  PoP={pop_s}")

            send_telegram(
                f"🤖 <b>AUTO-OPTIONS: {ticker}</b>\n"
                f"Strategy : {strategy}  ({dte} DTE)\n"
                f"Contract : ${strike:.0f} {opt_type.upper()}  exp {expiry}\n"
                f"Score    : {score:.0f}  |  Prob: {prob:.0f}%\n"
                f"PoP      : {pop_s}  |  B/E move: {be:.1f}%\n"
                f"Premium  : ${mid:.2f} × {contracts}× = ${cost_total:.2f}\n"
                f"VIX mult : {vix_m:.1f}×  ({vix_data['regime'] if vix_data else 'N/A'})\n"
                f"Cash     : ${result.get('cash_remaining', 0):,.2f}\n"
                f"Session  : {quality}  ({session.get('name','')})"
            )
        else:
            print(f"    ✗ {ticker:8s} — {result.get('error', 'failed')}")

    if new_entries == 0:
        print("  No new options positions opened this run.")
    else:
        print(f"  {new_entries} new options position(s) opened.")


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
        print(f"\n  AVOID window — skipping auto-entry.")
        conn.close()
        print("\n  Done.\n")
        return

    # ── Auto-entry: stocks ────────────────────────────────────────────────────
    try:
        auto_enter_stocks(conn, paper, session, quality)
    except Exception as exc:
        print(f"\n  ERROR in auto_enter_stocks: {exc}")
        traceback.print_exc()
        send_telegram(f"⚠️ <b>Auto-Entry Stock Error</b>\n{str(exc)[:300]}")

    # ── Auto-entry: options ───────────────────────────────────────────────────
    vix_for_opts = None
    try:
        vix_for_opts = ts.get_vix_level()
    except Exception:
        pass
    try:
        auto_enter_options(conn, vix_for_opts, session, quality)
    except Exception as exc:
        print(f"\n  ERROR in auto_enter_options: {exc}")
        traceback.print_exc()
        send_telegram(f"⚠️ <b>Auto-Entry Options Error</b>\n{str(exc)[:300]}")

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
