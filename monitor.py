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

# ── Optional AI analyst (silently disabled if no API key) ─────────────────────
try:
    from ai_engine import AIAnalyst
    _AI = AIAnalyst()
except Exception:
    _AI = None


def _ai_rationale(label: str, payload: dict, max_tokens: int = 160) -> str:
    """Return a short AI-written rationale or empty string if AI is off."""
    if _AI is None or not _AI.available:
        return ""
    try:
        prompt = (
            f"Write a 2-sentence rationale for this {label}. "
            "Cite specific signals from the JSON. End with one specific risk.\n\n"
            + str(payload)[:2500]
        )
        return _AI.chat(prompt, max_tokens=max_tokens) or ""
    except Exception:
        return ""


# ── Auto-entry configuration ──────────────────────────────────────────────────
# Raise these thresholds to be more selective; lower them to trade more often.
STOCK_MIN_SCORE          = 60    # explosive_score threshold for stock auto-entry
STOCK_MIN_PROB           = 60    # breakout_prob (%) threshold for stock auto-entry
OPTIONS_MIN_SCORE        = 55    # explosive_score threshold for options auto-entry
OPTIONS_MIN_PROB         = 55    # breakout_prob (%) threshold for options auto-entry
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

def auto_enter_stocks(conn, paper, session, quality,
                      risk_mult=1.0, market_regime=None,
                      learning_adjustments=None):
    """
    Automatically open paper stock positions for tickers in the latest breakout
    scan that PASS the Master Score gate.

    Rules:
      - Only runs during PRIME / NORMAL sessions (caller's responsibility).
      - Skips tickers already held or that have gapped > STOCK_CHASE_LIMIT_PCT
        above the scan entry price (avoids chasing extended moves).
      - Master Score must be ≥ 65 (BUY decision) — single unified gate.
      - Position size scaled by Master Score's size_multiplier × portfolio risk_mult.
      - At most AUTO_MAX_ENTRIES_PER_RUN new positions per call.
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

        # ── NEWS VETO — skip outright if major negative news in last 24h ───
        try:
            from news_agent import NewsAgent as _NA
            _na_local = _NA(conn, ai_analyst=None)
            _ni = _na_local.get_news_impact(ticker, hours=24)
            if _ni.get("should_skip"):
                top_h = ""
                if _ni.get("top_event"):
                    top_h = str(_ni["top_event"].get("headline", ""))[:80]
                print(f"    {ticker:8s} — NEWS VETO: major negative news "
                      f"(\"{top_h}\"), skip")
                continue
        except Exception:
            pass

        # ── MASTER SCORE GATE ─────────────────────────────────────────────
        # Single unified check that replaces individual Wyckoff/volume/MTF gates.
        # Threshold = learning_adjustments.min_master_score (default 65).
        try:
            master = ts.compute_master_score(
                ticker, expiry=None, bias="bullish", conn=conn,
                market_regime=market_regime,
            )
        except Exception as _mse:
            print(f"    {ticker:8s} — master score failed: {_mse}, skip")
            continue

        m_score    = master.get("score", 0)
        m_grade    = master.get("grade", "?")
        m_decision = master.get("decision", "SKIP")
        m_mult     = master.get("size_multiplier", 0)
        wy_label = master["components"].get("wyckoff", {}).get("label", "")
        mtf_label = master["components"].get("multi_tf", {}).get("label", "")
        print(f"    {ticker:8s} — Master {m_score}/100 ({m_grade}) {m_decision} | "
              f"{wy_label} | {mtf_label}")

        if m_decision == "SKIP":
            continue

        # ── Apply learning-adjustment floors ──────────────────────────────
        # The bot's punishment-mode + accumulated lessons raise these thresholds.
        _pattern_now = str(r.get("pattern_detected") or "")
        if learning_adjustments:
            min_score_required = float(learning_adjustments.get("min_master_score", 65))
            if m_score < min_score_required:
                print(f"    {ticker:8s} — Master {m_score} < learning floor {min_score_required:.0f}, skip")
                continue

            # Pattern blacklist (from past lessons)
            _pat_bl = learning_adjustments.get("pattern_blacklist", [])
            if _pattern_now and _pattern_now in _pat_bl:
                print(f"    {ticker:8s} — pattern '{_pattern_now}' is BLACKLISTED by learning engine, skip")
                continue

            # Wyckoff blacklist (default: DISTRIBUTION, MARKDOWN)
            _wy_bl = learning_adjustments.get("wyckoff_blacklist", [])
            try:
                _wy_phase_lbl = master["components"].get("wyckoff", {}).get("label", "").upper()
                _wy_skip = False
                for bad in _wy_bl:
                    if bad and bad.upper() in _wy_phase_lbl:
                        print(f"    {ticker:8s} — Wyckoff phase blacklisted ({bad}), skip")
                        _wy_skip = True
                        break
                if _wy_skip:
                    continue
            except Exception:
                pass

        # ── Sizing: master multiplier × portfolio risk multiplier ─────────
        effective_mult = m_mult * risk_mult
        # Also enforce the learning-derived sizing cap
        if learning_adjustments:
            effective_mult = min(effective_mult,
                                 float(learning_adjustments.get("size_multiplier_cap", 1.0)))
        if effective_mult <= 0.05:
            print(f"    {ticker:8s} — effective sizing too low ({effective_mult:.2f}×), skip")
            continue

        stop    = float(r.get("stop_loss")    or live_price * 0.95)
        tgt     = float(r.get("target_price") or live_price * 1.10)
        score   = float(r.get("explosive_score") or 0)
        prob    = float(r.get("breakout_prob")   or 0)
        pattern = str(r.get("pattern_detected")  or "")

        # ── Resolve sector for portfolio risk + learning engine ────────────
        # Without this, the bot can't enforce sector-concentration limits
        # and the Learning Engine can't blacklist losing sectors.
        sector = ""
        try:
            import yfinance as _yf_sec
            info = _yf_sec.Ticker(ticker).info or {}
            sector = str(info.get("sector", "") or "")
        except Exception:
            pass

        # ── CRITIC AGENT (Trading Memory Agent) ─────────────────────────
        # Final sanity check against episodic memory of past similar trades.
        # Can soft-veto setups where 80%+ of similar past trades lost.
        try:
            from trading_memory import TradingMemoryAgent as _TMA
            _critic = _TMA(conn, ai_analyst=_AI).critic_review(
                candidate={
                    "ticker":         ticker,
                    "master_score":   m_score,
                    "sector":         sector,
                    "pattern":        pattern,
                    "wyckoff_phase":  master["components"].get("wyckoff", {}).get("label", ""),
                    "market_regime":  (market_regime or {}).get("regime", ""),
                    "vix_regime":     "",
                },
                master_result=master,
            )
            _verdict = _critic.get("verdict", "BUY")
            _reason  = _critic.get("reasoning", "")
            print(f"    {ticker:8s} — Critic: {_verdict} ({_critic.get('n_similar', 0)} similar past, "
                  f"WR {(_critic.get('similar_win_rate') or 0)*100:.0f}%) — {_reason[:80]}")
            if _verdict == "SKIP":
                print(f"    {ticker:8s} — 🛑 CRITIC VETO: {_reason}")
                continue
        except Exception:
            pass

        signal = {
            "price":           live_price,
            "stop_price":      stop,
            "tgt_price":       tgt,
            "explosive_score": score,
            "probability":     prob,
            "pattern":         pattern,
            "sector":          sector,
        }
        result = paper.open_position(ticker, signal)

        if result.get("success"):
            new_entries += 1
            held_tickers.add(ticker)
            gross   = result.get("gross", 0)
            shares  = result.get("shares", 0)
            wy_phase = master["components"].get("wyckoff", {}).get("label", "—")
            spring_flag = " 💎SPRING" if "SPRING" in wy_phase.upper() else ""
            squeeze_flag = " 🚀" if m_grade in ("A+", "A") else ""
            print(f"    ✅ AUTO-BUY {ticker:8s}  Master={m_score}/100 ({m_grade})  "
                  f"price=${live_price:.2f}  stop=${stop:.2f}  tgt=${tgt:.2f}  "
                  f"shares={shares:.1f}  cost=${gross:.2f}{spring_flag}")
            summ_now = paper.get_summary()

            # ── AI rationale (free, optional) ─────────────────────────────
            _ai_text = _ai_rationale("stock paper-trade entry", {
                "ticker": ticker, "price": live_price,
                "stop": stop, "target": tgt, "pattern": pattern,
                "master_score": m_score, "grade": m_grade,
                "wyckoff": wy_phase,
                "multi_tf": master['components'].get('multi_tf', {}).get('label',''),
                "volume":   master['components'].get('inst_volume', {}).get('label',''),
            })
            _ai_block = f"\n🤖 <i>{_ai_text}</i>\n" if _ai_text else ""

            send_telegram(
                f"🤖 <b>AUTO-BUY: {ticker}</b>{spring_flag}{squeeze_flag}\n"
                f"Master   : <b>{m_score}/100 ({m_grade})</b>  {m_decision}\n"
                f"Pattern  : {pattern or '—'}\n"
                f"{wy_phase}\n"
                f"{master['components'].get('multi_tf', {}).get('label','')}\n"
                f"{master['components'].get('inst_volume', {}).get('label','')}\n"
                f"Entry    : ${live_price:.2f}\n"
                f"Stop     : ${stop:.2f}  |  Target: ${tgt:.2f}\n"
                f"Shares   : {shares:.1f}  |  Cost: ${gross:.2f}\n"
                f"Sizing   : {effective_mult:.2f}× "
                f"(master {m_mult:.1f} × risk {risk_mult:.2f})\n"
                f"Cash     : ${summ_now.get('available_cash',0):,.2f}\n"
                f"Session  : {quality}  ({session.get('name','')})"
                + _ai_block
            )
        else:
            print(f"    ✗ {ticker:8s} — {result.get('reason', 'failed')}")

    if new_entries == 0:
        print("  No new stock positions opened this run.")
    else:
        print(f"  {new_entries} new stock position(s) opened.")


# ── Auto-entry: options ───────────────────────────────────────────────────────

def auto_enter_options(conn, vix_data, session, quality,
                       risk_mult=1.0, market_regime=None,
                       learning_adjustments=None):
    """
    Automatically open paper options positions using Weekly ATM strategy (5–7 DTE)
    for tickers that PASS the Master Score gate.

    Rules:
      - Skips entirely when VIX regime is EXTREME (options too expensive).
      - Master Score must be ≥ 65 (BUY).  Score < 65 → skip.
      - Position size = ts.get_position_sizing() × master mult × portfolio risk_mult.
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

        # ── NEWS VETO — skip outright if major negative news in last 24h ───
        # For options, the veto is bias-aware: bullish puts on bad news = OK.
        try:
            from news_agent import NewsAgent as _NA
            _na_local = _NA(conn, ai_analyst=None)
            _ni = _na_local.get_news_impact(ticker, hours=24)
            if _ni.get("should_skip") and opt_type == "call":
                top_h = ""
                if _ni.get("top_event"):
                    top_h = str(_ni["top_event"].get("headline", ""))[:80]
                print(f"    {ticker:8s} — NEWS VETO (call): negative news "
                      f"(\"{top_h}\"), skip")
                continue
        except Exception:
            pass

        # ── MASTER SCORE GATE (replaces individual Wyckoff/MTF checks) ────
        bias_for_score = "bullish" if opt_type == "call" else "bearish"
        try:
            master = ts.compute_master_score(
                ticker, expiry=expiry, bias=bias_for_score, conn=conn,
                vix_data=vix_data, market_regime=market_regime,
            )
        except Exception as _mse:
            print(f"    {ticker:8s} — master score failed: {_mse}, skip")
            continue

        m_score    = master.get("score", 0)
        m_grade    = master.get("grade", "?")
        m_decision = master.get("decision", "SKIP")
        m_mult     = master.get("size_multiplier", 0)
        wy_label  = master["components"].get("wyckoff", {}).get("label", "")
        mtf_label = master["components"].get("multi_tf", {}).get("label", "")
        print(f"    {ticker:8s} — Master {m_score}/100 ({m_grade}) {m_decision} | "
              f"{wy_label} | {mtf_label}")

        if m_decision == "SKIP":
            continue

        # ── Apply learning-adjustment floors ──────────────────────────────
        if learning_adjustments:
            min_score_required = float(learning_adjustments.get("min_master_score", 65))
            if m_score < min_score_required:
                print(f"    {ticker:8s} — Master {m_score} < learning floor {min_score_required:.0f}, skip")
                continue
            _wy_bl = learning_adjustments.get("wyckoff_blacklist", [])
            try:
                _wy_phase_lbl = master["components"].get("wyckoff", {}).get("label", "").upper()
                _wy_skip = False
                for bad in _wy_bl:
                    if bad and bad.upper() in _wy_phase_lbl:
                        print(f"    {ticker:8s} — Wyckoff phase blacklisted ({bad}) for options, skip")
                        _wy_skip = True
                        break
                if _wy_skip:
                    continue
            except Exception:
                pass

        # ── Gamma squeeze bonus flag ───────────────────────────────────────
        squeeze_str = ""
        try:
            _gs2 = ts.detect_gamma_squeeze_setup(ticker)
            if _gs2.get("squeeze_potential") == "HIGH":
                squeeze_str = " 🚀SQUEEZE"
        except Exception:
            pass

        # ── Apply master + portfolio multipliers to contracts ──────────────
        effective_mult = m_mult * risk_mult
        if learning_adjustments:
            effective_mult = min(effective_mult,
                                 float(learning_adjustments.get("size_multiplier_cap", 1.0)))
        if effective_mult <= 0.05:
            print(f"    {ticker:8s} — effective sizing too low ({effective_mult:.2f}×), skip")
            continue
        contracts = max(1, int(round(contracts * effective_mult)))
        cost_total = round(mid * 100 * contracts, 2)
        if cost_total > cash_avail:
            cost_total = round(mid * 100, 2)
            if cost_total > cash_avail:
                print(f"    {ticker:8s} — 1 contract costs ${cost_total:.2f}, cash=${cash_avail:.2f} — skip")
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

            print(f"    ✅ AUTO-OPTIONS {ticker:8s}  Master={m_score}/100 ({m_grade}) {strategy}  "
                  f"${strike:.0f}{opt_type[0].upper()} {expiry} ({dte}DTE)  "
                  f"mid=${mid:.2f}  {contracts}×  cost=${cost_total:.2f}  PoP={pop_s}{squeeze_str}")

            # ── AI rationale (free, optional) ─────────────────────────────
            _ai_text = _ai_rationale("options paper-trade entry", {
                "ticker": ticker, "strategy": strategy,
                "strike": strike, "expiry": expiry, "dte": dte,
                "option_type": opt_type, "premium": mid, "contracts": contracts,
                "master_score": m_score, "grade": m_grade,
                "pop_pct": pop_s, "breakeven_move_pct": be,
                "wyckoff": master['components'].get('wyckoff', {}).get('label',''),
                "multi_tf": master['components'].get('multi_tf', {}).get('label',''),
                "gamma_squeeze": bool(squeeze_str),
            })
            _ai_block = f"\n🤖 <i>{_ai_text}</i>\n" if _ai_text else ""

            send_telegram(
                f"🤖 <b>AUTO-OPTIONS: {ticker}</b>"
                + (" 🚀" if squeeze_str else "") + "\n"
                f"Master   : <b>{m_score}/100 ({m_grade})</b>  {m_decision}\n"
                f"Strategy : {strategy}  ({dte} DTE)\n"
                f"Contract : ${strike:.0f} {opt_type.upper()}  exp {expiry}\n"
                f"PoP      : {pop_s}  |  B/E move: {be:.1f}%\n"
                f"{master['components'].get('wyckoff', {}).get('label','')}"
                + (f"  |  🚀 SQUEEZE" if squeeze_str else "") + "\n"
                f"{master['components'].get('multi_tf', {}).get('label','')}\n"
                f"Premium  : ${mid:.2f} × {contracts}× = ${cost_total:.2f}\n"
                f"Sizing   : {effective_mult:.2f}× "
                f"(master {m_mult:.1f} × risk {risk_mult:.2f}) · VIX {vix_m:.1f}×\n"
                f"Cash     : ${result.get('cash_remaining', 0):,.2f}\n"
                f"Session  : {quality}  ({session.get('name','')})"
                + _ai_block
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

    # ══════════════════════════════════════════════════════════════════════════
    # NEWS AGENT — Pull and classify fresh news before any trading decisions
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'─'*50}")
    print(f"  NEWS AGENT — Pulling latest market news")
    try:
        from news_agent import NewsAgent
        _news_agent = NewsAgent(conn, ai_analyst=_AI)
        _news_result = _news_agent.run_cycle(max_per_source=15, max_new=20)
        print(f"  Fetched {_news_result['fetched']} items, "
              f"persisted {_news_result['new']} new, "
              f"{_news_result['high_imp']} high-impact")

        # Telegram alert for any HIGH-impact news that just landed
        for hi in (_news_result.get("high_items") or [])[:3]:
            item = hi["item"]
            cls  = hi["classification"]
            tickers_str = ", ".join(cls.get("affected_tickers", [])) or "—"
            send_telegram(
                f"🚨 <b>HIGH-IMPACT NEWS</b>\n"
                f"<b>{item.get('headline', '')[:200]}</b>\n"
                f"Category : {cls.get('category', '?')}  "
                f"Sentiment: {cls.get('sentiment', '?')}  "
                f"Impact: {cls.get('impact_score', 0)}/10\n"
                f"Affects  : {tickers_str}\n"
                f"Source   : {item.get('source', '')}"
            )

        # Market pulse for the run log
        try:
            pulse = _news_agent.get_market_pulse(hours=24)
            print(f"  Market pulse: {pulse['mood']} "
                  f"(net {pulse['net_sentiment']:+.2f} over {pulse['n_events']} events)")
        except Exception:
            pass
    except Exception as _ne:
        print(f"  WARN news agent failed: {_ne}")
        _news_agent = None

    # ══════════════════════════════════════════════════════════════════════════
    # PORTFOLIO-LEVEL RISK GATES
    #   These run BEFORE any auto-entry to halt trading on drawdown / regime risk
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'─'*50}")
    print(f"  RISK ENGINE — Portfolio & Market Gates")

    # ── Gate 0: SELF-LEARNING ENGINE — check for punishment / reset ───────────
    learning_engine = None
    learning_adjustments = {
        "min_master_score":  65,
        "sector_blacklist":  [],
        "pattern_blacklist": [],
        "size_multiplier_cap": 1.0,
        "learning_iteration": 0,
    }
    portfolio_size_mult = 1.0

    try:
        opts_for_reset  = ts.OptionsPaperEngine(conn)
        learning_engine = ts.LearningEngine(conn)

        # ── Continuous learning: process every trade closed since last cycle ──
        # This is the "adapt fast" mechanism — runs every 5 minutes, updates
        # signal performance stats and nudges thresholds based on outcomes.
        try:
            _lc = learning_engine.process_recent_closes(max_per_cycle=50)
            if _lc.get("n_processed", 0) > 0:
                print(f"  Continuous learning: processed {_lc['n_processed']} "
                      f"closed trades ({_lc['n_wins']}W / {_lc['n_losses']}L)")
        except Exception as _cle:
            print(f"  WARN continuous learning failed: {_cle}")

        # ── Episodic memory: record new opens, reflect on new closes ─────
        # The Trading Memory Agent stores rich context per trade and uses
        # AI reflection to learn from each close.  Idempotent watermarks
        # so re-running the cycle is safe.
        try:
            from trading_memory import TradingMemoryAgent
            _tma = TradingMemoryAgent(conn, ai_analyst=_AI)
            _ml = _tma.process_lifecycle_events(generate_reflections=True,
                                                   max_per_cycle=20)
            if (_ml.get("opens_recorded", 0) > 0
                    or _ml.get("closes_updated", 0) > 0):
                print(f"  Memory agent: +{_ml['opens_recorded']} opens, "
                      f"{_ml['closes_updated']} closes, "
                      f"{_ml['reflections']} reflections generated")
        except Exception as _me:
            print(f"  WARN memory agent failed: {_me}")

        le_state = learning_engine.check_state(paper, opts_for_reset)
        learning_adjustments = le_state.get("adjustments", learning_adjustments)
        action = le_state.get("action", "NORMAL")

        print(f"  Learning Engine: {action}  ·  iter #{learning_adjustments['learning_iteration']}")
        print(f"    Drawdown    : {le_state.get('drawdown_pct', 0):.1f}%")
        print(f"    {le_state.get('message', '')}")

        # ── HARD RESET (-50%) ────────────────────────────────────────────────
        if action == "RESET":
            les = le_state.get("lessons", {})
            print(f"  🚨🚨🚨 HARD RESET TRIGGERED 🚨🚨🚨")
            print(f"    {les.get('lessons_summary', '')}")
            send_telegram(
                f"🚨🚨🚨 <b>CAPITAL RESET TRIGGERED</b> 🚨🚨🚨\n"
                f"Drawdown : {le_state.get('drawdown_pct', 0):.1f}%\n"
                f"Peak     : ${le_state.get('peak_equity', 0):,.2f}\n"
                f"Final    : ${le_state.get('current_equity', 0):,.2f}\n"
                f"\n"
                f"📚 <b>Learning Iteration #{les.get('learning_iteration', 1)}</b>\n"
                f"Closed: {les.get('n_trades', 0)} trades "
                f"({les.get('n_wins', 0)}W / {les.get('n_losses', 0)}L · "
                f"{les.get('win_rate_pct', 0):.0f}%)\n"
                f"Avg win: ${les.get('avg_win_pnl', 0):.2f}  |  "
                f"Avg loss: ${les.get('avg_loss_pnl', 0):.2f}\n"
                f"Worst sector: {les.get('worst_sector', '—')}\n"
                f"Worst pattern: {les.get('worst_pattern', '—')}\n"
                f"\n"
                f"🔧 <b>Adjustments active going forward:</b>\n"
                f"• Min Master Score: {les.get('new_min_master_score', 65):.0f}\n"
                f"• Sector blacklist: {', '.join(les.get('sector_blacklist', []) or ['none'])}\n"
                f"• Pattern blacklist: {', '.join(les.get('pattern_blacklist', []) or ['none'])}\n"
                f"\n"
                f"Capital reset to $1,000.  Bot will use these adjustments to avoid the same mistakes."
            )
            # Reload paper engine since capital was just reset
            paper = ts.PaperTradingEngine(conn)
            print(f"\n  Done (session={quality}, action=RESET).\n")
            conn.close()
            return

        # ── PUNISHMENT MODE entry (-15%) ─────────────────────────────────────
        if action == "ENTER_PUNISHMENT":
            send_telegram(
                f"⚠️ <b>PUNISHMENT MODE TRIGGERED</b>\n"
                f"Drawdown : {le_state.get('drawdown_pct', 0):.1f}%\n"
                f"Peak     : ${le_state.get('peak_equity', 0):,.2f}\n"
                f"Current  : ${le_state.get('current_equity', 0):,.2f}\n"
                f"Target   : ${le_state.get('recovery_target', 0):,.2f}\n"
                f"\n"
                f"🛑 No new trades until equity recovers to within 5% of peak.\n"
                f"📈 When unfrozen: Master Score floor raised to 75 (was 65).\n"
                f"Position sizing capped at 0.5×.\n"
                f"\n"
                f"This is automatic discipline — the bot must prove it can recover before being allowed to trade again."
            )
            print(f"\n  Done (session={quality}, action=PUNISHMENT_ENTERED).\n")
            conn.close()
            return

        # ── PUNISHMENT continuing ─────────────────────────────────────────────
        if action == "CONTINUE_PUNISHMENT":
            print(f"  Still in PUNISHMENT — skipping all new entries this cycle.")
            print(f"\n  Done (session={quality}, action=PUNISHMENT_CONTINUE).\n")
            conn.close()
            return

        # ── PUNISHMENT exit (recovered) ───────────────────────────────────────
        if action == "EXIT_PUNISHMENT":
            send_telegram(
                f"✅ <b>PUNISHMENT MODE ENDED</b>\n"
                f"Equity recovered to ${le_state.get('current_equity', 0):,.2f}\n"
                f"Resuming normal trading with active adjustments:\n"
                f"• Min Master Score: {learning_adjustments['min_master_score']:.0f}\n"
                f"• Iteration #{learning_adjustments['learning_iteration']}"
            )

    except Exception as exc:
        print(f"  WARN learning engine check failed: {exc}")
        traceback.print_exc()

    # ── Gate 1: Portfolio drawdown (already handled in learning engine) ──────
    try:
        risk = ts.get_portfolio_risk_status(paper)
        print(f"  Portfolio status: {risk['risk_level']}")
        print(f"    Drawdown      : {risk['drawdown_pct']:.1f}%  "
              f"(peak ${risk['peak_equity']:,.2f} → now ${risk['current_equity']:,.2f})")
        print(f"    Max sector    : {risk['max_sector_pct']:.0f}%")
        print(f"    {risk['reason']}")
        portfolio_size_mult = risk.get("size_multiplier", 1.0)

        # Also honour the size cap from learning adjustments
        portfolio_size_mult = min(portfolio_size_mult,
                                   learning_adjustments.get("size_multiplier_cap", 1.0))

        if not risk.get("can_open_new", True):
            print(f"\n  🛑 RISK ENGINE HALTED NEW ENTRIES")
            print(f"     Reason: {risk['reason']}")
            send_telegram(
                f"🛑 <b>Risk Engine HALT</b>\n"
                f"Level    : {risk['risk_level']}\n"
                f"Drawdown : {risk['drawdown_pct']:.1f}%\n"
                f"Reason   : {risk['reason']}\n"
                f"Session  : {quality}"
            )
            print(f"\n  Done (session={quality}, gate=HALT).\n")
            conn.close()
            return
    except Exception as exc:
        print(f"  WARN risk status check failed: {exc}")

    # ── Gate 2: Market regime ─────────────────────────────────────────────────
    market_regime = None
    regime_size_mult = 1.0
    try:
        import yfinance as _yf
        spy_hist = _yf.download("SPY", period="1y", interval="1d",
                                progress=False, auto_adjust=True)
        if not spy_hist.empty:
            import pandas as _pd
            if isinstance(spy_hist.columns, _pd.MultiIndex):
                spy_hist.columns = spy_hist.columns.get_level_values(0)
            market_regime = ts.MarketRegimeDetector.detect(spy_hist)
            print(f"  Market regime : {market_regime['label']}  "
                  f"(score {market_regime['score']:+d})")
            print(f"    {market_regime['advice']}")

            # Adjust size based on regime
            if market_regime["regime"] == "BEAR":
                regime_size_mult = 0.4
                print(f"  ⚠️ BEAR regime — reducing new position sizes by 60 %")
            elif market_regime["regime"] in ("NEUTRAL", "RECOVERING"):
                regime_size_mult = 0.7
                print(f"  ⚠️ {market_regime['regime']} regime — reducing new sizes by 30 %")
    except Exception as exc:
        print(f"  WARN market regime check failed: {exc}")

    # ── Combined size multiplier passed downstream ────────────────────────────
    combined_mult = portfolio_size_mult * regime_size_mult
    print(f"  Combined sizing multiplier: {combined_mult:.2f}× "
          f"(portfolio {portfolio_size_mult:.2f} × regime {regime_size_mult:.2f})")

    # ── Auto-entry: stocks ────────────────────────────────────────────────────
    try:
        auto_enter_stocks(conn, paper, session, quality,
                          risk_mult=combined_mult, market_regime=market_regime,
                          learning_adjustments=learning_adjustments)
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
        auto_enter_options(conn, vix_for_opts, session, quality,
                           risk_mult=combined_mult, market_regime=market_regime,
                           learning_adjustments=learning_adjustments)
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
