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
from datetime import datetime, timezone, timedelta

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


# ── Execution mode ────────────────────────────────────────────────────────────
# "internal"     = the bot's built-in paper simulation (default; nothing external)
# "alpaca_paper" = route auto-entries to a real Alpaca PAPER account for an honest,
#                  broker-grade track record. Requires ALPACA_PAPER_KEY/SECRET.
# Defaults to internal so NOTHING trades a broker account until you opt in.
BROKER_MODE = os.environ.get("BROKER_MODE", "internal").strip().lower()

# Trade the broker account as if it had THIS much equity, regardless of the real
# balance. Lets you keep Alpaca's $100k paper default but size every trade to your
# real plan (e.g. 1000). 0 = use the real account equity. Set as a secret.
try:
    ACCOUNT_EQUITY_OVERRIDE = float(os.environ.get("ACCOUNT_EQUITY_OVERRIDE", "0") or 0)
except Exception:
    ACCOUNT_EQUITY_OVERRIDE = 0.0
# A single option ticket may not exceed this fraction of (effective) equity.
MAX_TICKET_FRACTION = 0.20

# ── Auto-entry configuration ──────────────────────────────────────────────────
# Raise these thresholds to be more selective; lower them to trade more often.
STOCK_MIN_SCORE          = 60    # explosive_score threshold for stock auto-entry
STOCK_MIN_PROB           = 60    # breakout_prob (%) threshold for stock auto-entry
OPTIONS_MIN_SCORE        = 55    # explosive_score threshold for options auto-entry
OPTIONS_MIN_PROB         = 55    # breakout_prob (%) threshold for options auto-entry
AUTO_MAX_ENTRIES_PER_RUN = 2     # max NEW entries per monitor run (each category)
STOCK_CHASE_LIMIT_PCT    = 3.0   # skip stock entry if price > scan entry by this %
UNIFIED_MOVER_CHASE_PCT  = 20.0  # skip a unified MOVER candidate already up >this% today
                                 # (avoids buying parabolic tops like ASTC +147%; we want
                                 #  to catch the early +8-20% movers, not chase blow-offs)

# Always-on options watchlist — tickers with reliable weekly options + high
# liquidity. The options auto-entry checks these EVERY cycle regardless of
# whether they're in the breakout scan. They go through the same Master
# Score gate, so weak setups still get skipped — but the bot won't go idle
# just because the scan happens to be sparse.
#
# IMPORTANT: List is ORDERED by typical contract cost (cheap → expensive).
# The diagnostic checks affordability in this order, and auto_enter_options
# also iterates in this order — so if cash is tight, cheap tickers get
# priority and don't get crowded out by mega-caps that won't fit anyway.
OPTIONS_AUTO_WATCHLIST = [
    # ── Cheap-premium tier (≤ $35 / contract typical) ────────────────────
    "SNAP",    # ~$6/contract
    "AMC",     # ~$10/contract
    "SPY",     # ~$12/contract (Weekly OTM — major index)
    "GME",     # ~$16/contract
    "SOFI",    # ~$20/contract
    "RIVN",    # ~$21/contract
    "T",       # ~$21/contract
    "AAL",     # ~$26/contract
    "F",       # ~$29/contract
    "LCID",    # ~$30/contract
    "NIO",     # ~$31/contract
    "MARA",    # ~$34/contract
    "BAC",     # ~$35/contract
    # ── Medium-premium tier ($35-100) ────────────────────────────────────
    "PINS",    # ~$37/contract
    "DKNG",    # ~$44/contract
    "CCL",     # ~$44/contract
    "UBER",    # ~$70/contract
    "IWM",     # ~$70/contract
    "RIOT",    # ~$80/contract
    # ── Mid-cap tier ($100-200) ──────────────────────────────────────────
    "QQQ",     # ~$108/contract
    "AAPL",    # ~$114/contract
    "RBLX",    # ~$120/contract
    "AMZN",    # ~$169/contract
    "HOOD",    # ~$172/contract
    "NVDA",    # ~$182/contract
    # ── Premium tier ($200+) ─────────────────────────────────────────────
    "PLTR",    # ~$227/contract
    "MSFT",    # ~$238/contract
    "INTC",    # ~$335/contract
    "META",    # ~$420/contract
    "GOOG",    # ~$420/contract
    "TSLA",    # ~$500/contract  (often on edge of affordability)
]
OPTIONS_MIN_CASH = 5.0   # minimum free cash to consider an options entry
                          # (lowered from $20 — SNAP/AMC contracts are $6-$10)

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

    # ── Merge in Unified Scanner movers/catalyst candidates ────────────────────
    # Stocks that ALREADY exploded on a fresh catalyst (earnings, drone news, etc.)
    # never appear in the technical breakout `calls` table. The unified scan
    # (movers mode) scores them with the SAME Master-Score brain and persists the
    # BUY names — pull them in so they're eligible for auto-entry too. They still
    # get re-validated by the Master-Score gate + critic below.
    try:
        from unified_scanner import get_latest_unified
        existing = {str(c.get("ticker") or "").upper() for c in candidates}
        merged = 0
        for u in get_latest_unified(conn, decisions=("BUY",), limit=25):
            tk = str(u.get("ticker") or "").upper()
            if not tk or tk in existing or tk in held_tickers:
                continue
            # Chase guard for movers: don't buy a name that already ran parabolic
            # today (no scan-entry exists for movers, so the normal drift guard
            # can't protect us — gate on intraday % move instead).
            _pct = u.get("pct_change")
            if _pct is not None and abs(float(_pct)) > UNIFIED_MOVER_CHASE_PCT:
                print(f"    {tk:8s} — unified mover already {_pct:+.0f}% today "
                      f"(> {UNIFIED_MOVER_CHASE_PCT:.0f}% chase cap), skip")
                continue
            candidates.append({
                "ticker":           tk,
                "explosive_score":  u.get("score") or 0,
                "breakout_prob":    u.get("score") or 0,
                "pattern_detected": "unified:" + ",".join(u.get("sources") or []),
                "entry_price":      None,
                "target_price":     None,
                "stop_loss":        None,
            })
            existing.add(tk)
            merged += 1
        if merged:
            print(f"  + Merged {merged} Unified-Scanner BUY candidate(s) "
                  f"→ {len(candidates)} total")
    except Exception as _ue:
        print(f"  [unified] candidate merge skipped: {_ue}")

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

        # Capture rich context at trade-open time so the Memory Agent and
        # Learning Engine can use it for similarity matching later.
        _wyckoff_label = master["components"].get("wyckoff", {}).get("label", "")
        # Extract the phase name (e.g. "Wyckoff: ACCUMULATION" → "ACCUMULATION")
        _wyckoff_phase = ""
        if ":" in _wyckoff_label:
            _wyckoff_phase = _wyckoff_label.split(":", 1)[1].strip()
        else:
            _wyckoff_phase = _wyckoff_label
        _regime_label  = (market_regime or {}).get("regime", "") if market_regime else ""
        _vix_regime    = ""
        _vix_level     = 0.0
        try:
            _vix_data_local = ts.get_vix_level()
            if _vix_data_local:
                _vix_regime = _vix_data_local.get("regime", "")
                _vix_level  = float(_vix_data_local.get("vix", 0))
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
            # Rich context for Memory Agent + Learning Engine
            "master_score":    m_score,
            "wyckoff_phase":   _wyckoff_phase,
            "market_regime":   _regime_label,
            "vix_regime":      _vix_regime,
            "vix_level":       _vix_level,
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


# ── Liquidity guard ───────────────────────────────────────────────────────────
def _is_liquid_us(ticker: str) -> bool:
    """
    Reject the illiquid names that bled the account: any foreign/exchange suffix
    (.TO/.V/.NE/.CN/.AX...) is a non-US listing with thin volume + unreliable data.
    US tickers have no dot (BRK.B etc. are not in our momentum universe).
    """
    t = str(ticker or "").upper()
    return bool(t) and ("." not in t)


# ── Auto-entry: MOMENTUM (the validated strategy) ─────────────────────────────
def auto_enter_momentum(conn, paper, session, quality,
                        risk_mult=1.0, market_regime=None,
                        learning_adjustments=None):
    """
    PRIMARY stock entry path. Buys the strongest-momentum LIQUID names that are in
    a confirmed uptrend (price > 50- & 200-SMA). This is the only stock strategy
    we MCPT-validated with a real edge (cross-sectional momentum, p=0.004, beats
    pure beta by ~93%). The old chart-pattern path is retired (no edge: p≈0.23-0.81).
    """
    print(f"\n  {'─'*50}")
    print(f"  AUTO-ENTRY — Momentum (liquid trend leaders)")

    summary    = paper.get_summary()
    n_open     = len(paper.open_positions)
    cash_avail = summary.get("available_cash", 0)
    max_pos    = summary.get("max_positions", 10)
    slots      = max_pos - n_open
    if slots <= 0:
        print(f"  Portfolio full ({n_open}/{max_pos}) — no new momentum entries.")
        return
    if cash_avail < 100:
        print(f"  Cash too low (${cash_avail:.2f}) — skipping momentum entry.")
        return

    # In a confirmed BEAR regime, momentum crashes — stand down on new longs.
    if market_regime and market_regime.get("regime") == "BEAR":
        print("  BEAR regime — momentum stands down (no new longs).")
        return

    try:
        from momentum_strategy import MomentumStrategy
        ranked = MomentumStrategy(conn).rank(top_n=12, min_mom_6m=0.05)
    except Exception as exc:
        print(f"  ERROR ranking momentum universe: {exc}")
        return
    print(f"  {len(ranked)} liquid uptrend leaders ranked.")

    # ── Broker mode: route entries to a real Alpaca PAPER account ─────────────
    _broker = None
    _broker_equity = None
    if BROKER_MODE == "alpaca_paper":
        try:
            from broker import AlpacaPaperBroker
            _b = AlpacaPaperBroker()
            if _b.available():
                _acct = _b.get_account()
                if "error" not in _acct:
                    _broker = _b
                    _real_eq = _acct.get("equity", 0)
                    _broker_equity = (ACCOUNT_EQUITY_OVERRIDE
                                      if ACCOUNT_EQUITY_OVERRIDE > 0 else _real_eq)
                    _ov = (f" (sizing as ${_broker_equity:,.0f} override)"
                           if ACCOUNT_EQUITY_OVERRIDE > 0 else "")
                    print(f"  🏦 BROKER MODE: Alpaca paper — real equity "
                          f"${_real_eq:,.2f}{_ov}, {len(_b.get_positions())} positions")
                else:
                    print(f"  ⚠️ Alpaca paper account error: {_acct['error']} — "
                          f"falling back to internal sim.")
            else:
                print("  ⚠️ BROKER_MODE=alpaca_paper but no paper keys set — "
                      "falling back to internal sim.")
        except Exception as _bx:
            print(f"  ⚠️ Broker init failed: {_bx} — internal sim fallback.")

    if _broker:
        held_tickers = _broker.held_tickers()
    else:
        held_tickers = {p["ticker"] for p in paper.open_positions}
    held_sectors = {}
    for p in paper.open_positions:
        s = str(p.get("sector", "") or "Unknown")
        held_sectors[s] = held_sectors.get(s, 0) + 1

    new_entries = 0
    for r in ranked:
        if new_entries >= AUTO_MAX_ENTRIES_PER_RUN:
            print(f"  Reached max new entries ({AUTO_MAX_ENTRIES_PER_RUN}).")
            break
        ticker = r["ticker"]
        if ticker in held_tickers or not _is_liquid_us(ticker):
            continue

        live = _get_live_price(ticker) or r["entry"]
        if not live:
            continue

        # News veto — skip on major fresh negative news.
        try:
            from news_agent import NewsAgent as _NA
            _ni = _NA(conn, ai_analyst=None).get_news_impact(ticker, hours=24)
            if _ni.get("should_skip"):
                print(f"    {ticker:8s} — NEWS VETO, skip")
                continue
        except Exception:
            pass

        # Sector concentration cap (avoid 5 semis at once).
        sector = ""
        try:
            import yfinance as _yf
            sector = str((_yf.Ticker(ticker).info or {}).get("sector", "") or "")
        except Exception:
            pass
        sect_key = sector or "Unknown"
        if held_sectors.get(sect_key, 0) >= 4:
            print(f"    {ticker:8s} — sector '{sect_key}' already at cap, skip")
            continue

        # ── BROKER PATH: submit a real Alpaca paper bracket order ─────────────
        if _broker:
            # size ~ position % of real account equity (10% per name, risk-mult applied)
            pos_dollars = max(0.0, (_broker_equity or 0) * 0.10 * float(risk_mult or 1.0))
            if pos_dollars < (live or 0):
                print(f"    {ticker:8s} — ${pos_dollars:.0f} < 1 share (${live:.2f}), skip")
                continue
            br = _broker.submit_bracket_order(ticker, pos_dollars, r["stop"],
                                              r["target"], price=live)
            if br.get("ok"):
                new_entries += 1
                held_tickers.add(ticker)
                held_sectors[sect_key] = held_sectors.get(sect_key, 0) + 1
                print(f"    ✅ ALPACA BUY {ticker:8s}  {br['qty']}sh @ ~${br['entry_est']:.2f}  "
                      f"TP ${br['tp']:.2f} / SL ${br['sl']:.2f}  cost~${br['cost_est']:.0f}")
                send_telegram(
                    f"🏦 <b>ALPACA PAPER BUY: {ticker}</b>\n"
                    f"Momentum 6mo {r['mom_6m']*100:+.0f}% · RSI {r['rsi']:.0f}\n"
                    f"Qty    : {br['qty']} shares @ ~${br['entry_est']:.2f}\n"
                    f"Bracket: TP ${br['tp']:.2f} / SL ${br['sl']:.2f}\n"
                    f"Cost   : ~${br['cost_est']:.0f}  |  Acct equity ${_broker_equity:,.0f}\n"
                    f"Real broker fills — this is your honest track record."
                )
            else:
                print(f"    ✗ {ticker:8s} — Alpaca order failed: {br.get('error')}")
            continue   # broker path done for this ticker

        signal = {
            "price":         live,
            "stop_price":    r["stop"],
            "tgt_price":     r["target"],
            "pattern":       "Momentum",
            "explosive_score": round(r["score"] * 100, 1),
            "probability":   0,
            "sector":        sector,
            "master_score":  round(min(100, 50 + r["mom_6m"] * 100), 1),
            "market_regime": (market_regime or {}).get("regime", "") if market_regime else "",
        }
        result = paper.open_position(ticker, signal)
        if result.get("success"):
            new_entries += 1
            held_tickers.add(ticker)
            held_sectors[sect_key] = held_sectors.get(sect_key, 0) + 1
            shares = result.get("shares", 0)
            gross  = result.get("gross", 0)
            print(f"    ✅ AUTO-BUY {ticker:8s}  6m={r['mom_6m']*100:+.0f}% "
                  f"3m={r['mom_3m']*100:+.0f}% RSI={r['rsi']:.0f}  "
                  f"px=${live:.2f} stop=${r['stop']:.2f}  shares={shares:.2f} ${gross:.0f}")
            summ_now = paper.get_summary()
            send_telegram(
                f"🚀 <b>AUTO-BUY (Momentum): {ticker}</b>\n"
                f"6-mo momentum : {r['mom_6m']*100:+.0f}%   3-mo: {r['mom_3m']*100:+.0f}%\n"
                f"RSI           : {r['rsi']:.0f}{'  ⚠️extended' if r['extended'] else ''}\n"
                f"Entry  : ${live:.2f}\n"
                f"Stop   : ${r['stop']:.2f} (trend / 2×ATR)\n"
                f"Sector : {sector or '—'}\n"
                f"Shares : {shares:.2f}  |  Cost: ${gross:.0f}\n"
                f"Cash   : ${summ_now.get('available_cash',0):,.2f}\n"
                f"Session: {quality}"
            )
        else:
            print(f"    ✗ {ticker:8s} — {result.get('reason','failed')}")

    if new_entries == 0:
        print("  No new momentum positions opened this run.")
    else:
        print(f"  {new_entries} new momentum position(s) opened.")


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

    if cash_avail < OPTIONS_MIN_CASH:
        print(f"  Options cash too low (${cash_avail:.2f} < ${OPTIONS_MIN_CASH}) — skipping.")
        return

    # ── Build merged candidate list: scan plays + always-on watchlist ───────
    # The watchlist guarantees the bot has tickers to evaluate even when the
    # breakout scan is sparse. They still have to pass the same Master Score
    # gate — bad setups get rejected, but the bot doesn't go idle.
    scan_plays  = []
    try:
        scan_plays = ts.get_scan_options_plays(conn, top_n=10) or []
    except Exception as e:
        print(f"  WARN loading scan plays: {e}")

    held_opt_tickers = {
        str(p.get("ticker", "")).upper()
        for p in ops_engine.get_positions("OPEN")
    }
    closed_hist = ops_engine.get_positions("CLOSED")

    # ── Filter scan plays by pre-thresholds ──────────────────────────────────
    scan_candidates = [
        p for p in scan_plays
        if float(p.get("explosive_score", 0)) >= OPTIONS_MIN_SCORE
        and float(p.get("breakout_prob",   0)) >= OPTIONS_MIN_PROB
    ]
    print(f"  Scan plays: {len(scan_plays)} total | "
          f"{len(scan_candidates)} above thresholds "
          f"(score≥{OPTIONS_MIN_SCORE}, prob≥{OPTIONS_MIN_PROB}%)")

    # ── Build watchlist candidates (always evaluated) ────────────────────────
    # For each watchlist ticker, generate fresh strategy suggestions on the fly.
    # These don't have explosive_score from a scan — we set neutral defaults and
    # let the Master Score gate be the real quality check.
    print(f"  Watchlist: evaluating {len(OPTIONS_AUTO_WATCHLIST)} always-on tickers")
    _strat_engine = ts.OptionsStrategyEngine()
    watchlist_candidates = []
    seen_tickers = {str(p.get("ticker", "")).upper() for p in scan_candidates}
    for tk in OPTIONS_AUTO_WATCHLIST:
        tk = tk.upper()
        if tk in seen_tickers or tk in held_opt_tickers:
            continue
        try:
            sugs = _strat_engine.suggest(tk, bias="bullish")
        except Exception:
            sugs = []
        if not sugs:
            continue   # no weekly options available
        watchlist_candidates.append({
            "ticker":          tk,
            "explosive_score": 0,      # placeholder; Master Score is the real gate
            "breakout_prob":   0,
            "suggestions":     sugs,
            "source":          "watchlist",
        })

    # ── Merge & deduplicate (scan candidates win when overlapping) ──────────
    candidates = list(scan_candidates) + watchlist_candidates
    print(f"  Total candidates: {len(candidates)} "
          f"({len(scan_candidates)} from scan + {len(watchlist_candidates)} watchlist)")

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

        # ── SMART STRATEGY SELECTION with affordability fallback ────────────
        # Try Weekly ATM first (highest probability), then Weekly OTM
        # (cheaper, needs bigger move), then Mid-Week ATM as last resort.
        # If ALL suggestions cost more than available cash, skip this ticker.
        _strategy_order = ["Weekly ATM", "Weekly OTM", "Mid-Week ATM"]
        sg = None
        for _strat_name in _strategy_order:
            _candidate = next(
                (s for s in suggestions if s.get("strategy") == _strat_name),
                None,
            )
            if not _candidate:
                continue
            _mid_test = float(_candidate.get("mid", 0) or 0)
            if _mid_test <= 0:
                continue
            _cost_one = _mid_test * 100
            if _cost_one <= cash_avail:
                sg = _candidate
                if _strat_name != "Weekly ATM":
                    print(f"    {ticker:8s} — Weekly ATM unaffordable, "
                          f"fell back to {_strat_name}")
                break

        # Last resort: try ANY suggestion that fits the cash budget
        if sg is None:
            for s in suggestions:
                _mid_test = float(s.get("mid", 0) or 0)
                if _mid_test > 0 and _mid_test * 100 <= cash_avail:
                    sg = s
                    break

        if sg is None:
            _cheapest = min(
                (float(s.get("mid", 0) or 0) for s in suggestions
                  if float(s.get("mid", 0) or 0) > 0),
                default=0,
            )
            print(f"    {ticker:8s} — all suggestions exceed cash (cheapest "
                  f"${_cheapest*100:.0f}, cash ${cash_avail:.2f}) — skip")
            continue

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

        # Ensure we can afford it; fall back to 1 contract
        cost_total = round(mid * 100 * contracts, 2)
        if cost_total > cash_avail:
            cost_total = round(mid * 100, 2)
            if cost_total > cash_avail:
                # Should be unreachable thanks to the fallback above, but safe
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

    # ── Telegram connection test ──────────────────────────────────────────────
    # NOTE: cron-job.org triggers every cycle via workflow_dispatch, so this can
    # NO LONGER key off the event name (that spammed a "Connected" ping every
    # 10 min). Only send when explicitly asked via PING=1 (set on a manual run
    # you do yourself to verify alerts), and only report the result on failure.
    if os.environ.get("PING") == "1":
        sent = send_telegram(
            f"✅ <b>Breakout Bot — Telegram OK</b>\nTime: {now_utc}")
        print(f"  [Telegram] {'✓ ping sent' if sent else '✗ FAILED — check TELEGRAM secrets'}")

    # ── BROKER SELF-TEST — runs even when market is closed ────────────────────
    # Verifies the full chain (keys -> Alpaca -> options level -> BROKER_MODE).
    # Logs to the run output every cycle, but Telegrams ONLY on FAILURE (a
    # broken connection is worth a ping; a healthy one every 10 min is spam).
    # Force a success ping with PING=1 on a manual run if you want to verify.
    print(f"\n  EXECUTION MODE: {BROKER_MODE}")
    if BROKER_MODE == "alpaca_paper":
        try:
            from broker import AlpacaPaperBroker
            _selftest_b = AlpacaPaperBroker()
            _t = _selftest_b.test_connection()
            if _t.get("ok"):
                _ost = _selftest_b.options_status()
                _line = (f"✅ Alpaca paper connected · equity ${_t.get('equity',0):,.0f} · "
                         f"{_ost.get('msg')}")
            else:
                _line = f"❌ Alpaca paper NOT connected: {_t.get('error')}"
            print(f"  BROKER SELF-TEST: {_line}")
            # Only alert on failure (or an explicit PING) — never on healthy runs.
            if not _t.get("ok"):
                send_telegram(f"❌ <b>Broker connection FAILED</b>\n{_line}\n"
                              f"(BROKER_MODE={BROKER_MODE}) — bot can't trade until fixed.")
            elif os.environ.get("PING") == "1":
                send_telegram(f"🏦 <b>Broker self-test</b>\n{_line}")
        except Exception as _ste:
            print(f"  BROKER SELF-TEST failed: {_ste}")
            send_telegram(f"❌ <b>Broker self-test crashed</b>\n{str(_ste)[:200]}")

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

    # ── Benchmark snapshot — forward momentum-vs-SPY scorecard (once/day) ───────
    try:
        from benchmark_tracker import BenchmarkTracker
        BenchmarkTracker(conn).snapshot(paper)
    except Exception as _be:
        print(f"  WARN benchmark snapshot failed: {_be}")

    # ── INSTITUTIONAL MORNING BRIEFING — once per day, Telegram ───────────────
    # Professional desk-analyst read: multi-timeframe structure + internals +
    # regime + bull/bear/neutral probabilities + risk + recommended strategy.
    try:
        from market_analyst import MarketAnalyst
        conn.execute("CREATE TABLE IF NOT EXISTS briefing_state "
                     "(id INTEGER PRIMARY KEY, last_date TEXT)")
        _br_row = conn.execute("SELECT last_date FROM briefing_state WHERE id=1").fetchone()
        _br_last = (_br_row[0] if not hasattr(_br_row, "get") else _br_row.get("last_date")) if _br_row else None
        _today = datetime.now().strftime("%Y-%m-%d")
        if _br_last != _today:
            print(f"\n  {'─'*50}")
            print(f"  MARKET ANALYST — generating institutional morning briefing")
            _brief = MarketAnalyst(conn).generate_briefing()
            conn.execute("INSERT INTO briefing_state (id, last_date) VALUES (1,?) "
                         "ON CONFLICT(id) DO UPDATE SET last_date=excluded.last_date",
                         (_today,))
            _bias = _brief["bias"]; _emoji = {"Bullish":"🟢","Bearish":"🔴","Neutral":"⚪"}.get(_bias,"⚪")
            _struct = _brief["structure"]
            _msg = (
                f"🏛 <b>MORNING BRIEFING</b>  {_emoji}\n"
                f"<b>Bias: {_bias}</b>  (conf {_brief['confidence']:.0f}%)\n"
                f"Bull {_brief['prob_bull']:.0f}% · Bear {_brief['prob_bear']:.0f}% · "
                f"Neutral {_brief['prob_neutral']:.0f}%\n"
                f"Regime: <b>{_brief['market_regime']['primary']}</b> · "
                f"Risk: <b>{_brief['risk']['level']}</b>\n"
                f"\n<b>Structure:</b> W:{_struct['weekly']['trend']} · "
                f"D:{_struct['daily']['trend']} · I:{_struct['intraday']['trend']}\n"
                f"Breadth: {_brief['internals']['breadth'].get('pct_above_200','?')}% >200SMA · "
                f"VIX {_brief['internals'].get('vix',0):.1f}\n"
                f"\n<b>Strategy:</b> {_brief['recommended_strategy']['summary'][:240]}\n"
                f"\n<i>Invalidation:</i> {(_brief['invalidation'][0] if _brief['invalidation'] else '—')}"
            )
            send_telegram(_msg)
            print(f"  Briefing sent: {_bias} bias, {_brief['market_regime']['primary']} regime")
    except Exception as _mbe:
        print(f"  WARN morning briefing failed: {_mbe}")

    # ── Daily IV snapshot — builds historical IV record for true IV Rank ───────
    # Cheap if we only do it once per calendar day. The bot's iv_snapshots table
    # is what makes IV Rank progressively more accurate over time (vs the HV
    # proxy used until ~30+ days of history exist).
    try:
        from options_analytics import snapshot_iv
        row = conn.execute(
            "SELECT MAX(snapshot_date) FROM iv_snapshots").fetchone()
        last = row[0] if row else None
        today_iso = datetime.now().strftime("%Y-%m-%d")
        if (not last) or (str(last)[:10] != today_iso):
            print(f"\n  IV SNAPSHOT — building historical IV record (one-per-day)")
            n_iv = snapshot_iv(conn, OPTIONS_AUTO_WATCHLIST)
            print(f"  IV snapshots recorded: {n_iv}")
    except Exception as _ive:
        print(f"  WARN IV snapshot failed: {_ive}")

    # ── MACRO EVENT-RISK ALERT — once/day when a major release is imminent ────
    try:
        from macro_engine import event_risk as _erk, upcoming_events as _ue
        _er = _erk()
        if _er.get("level") in ("HIGH", "ELEVATED"):
            conn.execute("CREATE TABLE IF NOT EXISTS macro_alert_state "
                         "(id INTEGER PRIMARY KEY, last_date TEXT, last_event TEXT)")
            _mrow = conn.execute("SELECT last_date, last_event FROM macro_alert_state "
                                 "WHERE id=1").fetchone()
            _mlast = (_mrow[0] if not hasattr(_mrow,'get') else _mrow.get('last_date')) if _mrow else None
            _today = datetime.now().strftime("%Y-%m-%d")
            _ev_key = f"{_er.get('next_event')}|{_er.get('next_date')}"
            _mlast_ev = (_mrow[1] if not hasattr(_mrow,'get') else _mrow.get('last_event')) if _mrow else None
            if _mlast != _today or _mlast_ev != _ev_key:
                conn.execute("INSERT INTO macro_alert_state (id, last_date, last_event) "
                             "VALUES (1,?,?) ON CONFLICT(id) DO UPDATE SET "
                             "last_date=excluded.last_date, last_event=excluded.last_event",
                             (_today, _ev_key))
                _emoji = "🛑" if _er["level"] == "HIGH" else "⚠️"
                send_telegram(
                    f"{_emoji} <b>MACRO EVENT RISK: {_er['level']}</b>\n"
                    f"{_er['advice']}\n"
                    f"<i>Strong data keeps the Fed hawkish → growth/tech sells off "
                    f"(the 'good news is bad news' mechanism). Trade smaller / wait.</i>")
                print(f"  🌐 Macro event-risk alert sent: {_er['level']} "
                      f"({_er.get('next_event')})")
    except Exception as _mae:
        print(f"  WARN macro event-risk check failed: {_mae}")

    # ── Whale-watch outcomes — cheap forward P&L update (every cycle) ──────────
    # The expensive `build()` (4 APIs × ~125 tickers) is dashboard-button only;
    # here we just refresh live prices for already-detected picks so the
    # scorecard accumulates.
    try:
        from whale_watch import WhaleWatchlist
        n_upd = WhaleWatchlist(conn).update_outcomes()
        if n_upd:
            print(f"  [whale-watch] outcomes refreshed for {n_upd} pick(s)")
    except Exception as _we:
        print(f"  WARN whale-watch outcomes failed: {_we}")

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

        # ── Trend-break exit for MOMENTUM positions ───────────────────────────
        # A momentum trade is over when the trend breaks: close below the 50-day
        # SMA. This is the natural exit for the validated strategy (the fixed ATR
        # stop is just a disaster backstop). Only applies to liquid US names.
        try:
            from momentum_strategy import MomentumStrategy as _MS
            _ms = _MS(conn)
            for p in list(paper.open_positions):
                tk = p["ticker"]
                if str(p.get("pattern", "")) != "Momentum" or not _is_liquid_us(tk):
                    continue
                chk = _ms.should_exit(tk)
                if chk and chk.get("exit"):
                    px = chk.get("price") or _get_live_price(tk)
                    if px:
                        res = paper.close_position(tk, px, "TREND_BREAK_50SMA")
                        if res.get("success"):
                            print(f"    📉 TREND-BREAK EXIT {tk:8s} @ ${px:.2f} "
                                  f"(below 50SMA ${chk.get('sma50',0):.2f})  "
                                  f"net ${res.get('net_pnl',0):+.2f}")
                            send_telegram(
                                f"📉 <b>{tk}</b> closed — TREND BREAK\n"
                                f"Below 50-SMA (${chk.get('sma50',0):.2f})\n"
                                f"Exit ${px:.2f}  |  Net ${res.get('net_pnl',0):+.2f}")
        except Exception as _te:
            print(f"  WARN trend-break exit check failed: {_te}")

        # ── Trailing stop update notification ─────────────────────────────────
        # Re-read positions after any stop updates to show new stop levels
        updated = paper.open_positions
        for p in updated:
            if p.get("t1_hit") and p.get("trailing_stop"):
                print(f"    {p['ticker']:8s}  trailing stop now: ${p['trailing_stop']:.2f}"
                      f"  (T1 hit ✓, highest: ${p.get('highest_since_t1',0):.2f})")

    # ── 🌙 OVERNIGHT EDGE — must run BEFORE the AVOID early-return ────────────
    # Its SELL fires at the first cycle after the open (9:30-9:45 ET), which is
    # exactly the AVOID window, so it runs here on every open cycle. Validated:
    # ~all index returns accrue close->open (QQQ +255% vs +104% intraday, 10y).
    # Tracked 100% separately in overnight_edge_log.
    if BROKER_MODE == "alpaca_paper":
        try:
            from broker import AlpacaPaperBroker as _OEB
            from overnight_edge import OvernightEdge
            _oeb = _OEB()
            if _oeb.available():
                _oe_eq = (ACCOUNT_EQUITY_OVERRIDE if ACCOUNT_EQUITY_OVERRIDE > 0
                          else _oeb.get_account().get("equity", 0) or 0)
                _oe_res = OvernightEdge(conn, _oeb).step(_oe_eq,
                                                         telegram=send_telegram)
                if _oe_res.get("action") not in ("wait", "hold",
                                                 "market_closed", "disabled"):
                    print(f"  🌙 Overnight edge: {_oe_res.get('action')}")
        except Exception as _oee:
            print(f"  WARN overnight edge failed: {_oee}")

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
            # ── CATALYST READ — apply the decision tree to the named ticker(s) ──
            _read_txt = ""
            try:
                from catalyst_classifier import classify as _clx
                for _rt in (cls.get("affected_tickers", []) or [])[:1]:
                    _cr = _clx(conn, _rt)
                    if _cr:
                        _emo = {"READABLE": "✅", "COIN_FLIP": "🪙",
                                "AVOID": "🛑"}.get(_cr["readability"], "")
                        _read_txt = (f"\n{_emo} <b>Catalyst read:</b> "
                                     f"{_cr['readability'].replace('_',' ')} "
                                     f"({_cr['classification']}) → {_cr['structure_hint']}")
            except Exception:
                pass
            send_telegram(
                f"🚨 <b>HIGH-IMPACT NEWS</b>\n"
                f"<b>{item.get('headline', '')[:200]}</b>\n"
                f"Category : {cls.get('category', '?')}  "
                f"Sentiment: {cls.get('sentiment', '?')}  "
                f"Impact: {cls.get('impact_score', 0)}/10\n"
                f"Affects  : {tickers_str}\n"
                f"Source   : {item.get('source', '')}"
                f"{_read_txt}"
            )
            # ── Attach the exact affordable WEEKLY option for each named ticker ──
            try:
                from catalyst_options import catalyst_to_weekly_alert
                _sent = str(cls.get("sentiment", "")).lower()
                _dir = "bearish" if _sent in ("negative", "bearish") else "bullish"
                for _tk in (cls.get("affected_tickers", []) or [])[:2]:
                    _opt_msg = catalyst_to_weekly_alert(
                        _tk, _dir, item.get("headline", "")[:160])
                    if _opt_msg:
                        send_telegram(_opt_msg)
            except Exception as _coe:
                print(f"  WARN catalyst option alert failed: {_coe}")

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
    # VIP FEED — instant Telegram on market-moving public-figure posts
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'─'*50}")
    print(f"  VIP FEED — Trump (Truth Social) + Fed press releases")
    try:
        from vip_news_monitor import VipNewsMonitor
        _vip_report = VipNewsMonitor(conn).run_cycle(telegram_sender=send_telegram)
        for h, s in _vip_report.items():
            print(f"    {h:6s}  fetched={s['fetched']:3d}  "
                  f"new={s['new']:2d}  alerted={s['alerted']}")
    except Exception as _ve:
        print(f"  WARN VIP monitor failed: {_ve}")

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

    # ── Unified Scanner — feed catalyst movers into the Master-Score brain ─────
    # Runs the day's REAL top movers (Finviz signal + yfinance) through the same
    # compute_master_score() used everywhere, persisting BUY/WATCH names so the
    # stock auto-entry below can pick up catalyst plays (earnings/news gaps) that
    # the technical breakout scan never sees. Bounded + cloud-safe (movers mode).
    print(f"\n  {'─'*50}")
    print(f"  UNIFIED SCANNER — Catalyst movers → Master Score")
    try:
        from unified_scanner import run_unified_scan
        _uni = run_unified_scan(
            conn, mode="movers", min_score=55,
            market_regime=market_regime, watchlist=OPTIONS_AUTO_WATCHLIST,
        )
        _buys = [u for u in _uni if u.get("decision") == "BUY"]
        print(f"  Scored {len(_uni)} movers ≥55 | {len(_buys)} BUY")
        for u in _uni[:8]:
            print(f"    {u['ticker']:6s} {u['score']:5.1f} {u['grade']:<2} "
                  f"{u['decision']:<5} {u.get('pct_change') or 0:+6.1f}% "
                  f"[{','.join(u['sources'])}]")
    except Exception as exc:
        print(f"  WARN unified scan failed: {exc}")

    # ── Auto-entry: stocks (MOMENTUM is now primary — the validated strategy) ──
    # The old chart-pattern path (auto_enter_stocks) is RETIRED: MCPT showed it has
    # no edge (p≈0.23 small caps / 0.81 large caps) and it was filling illiquid
    # micro-caps. Cross-sectional momentum on liquid names is validated (p=0.004,
    # beats pure beta by ~93%). Set ENABLE_LEGACY_PATTERN_ENTRY=1 to re-enable.
    try:
        auto_enter_momentum(conn, paper, session, quality,
                            risk_mult=combined_mult, market_regime=market_regime,
                            learning_adjustments=learning_adjustments)
    except Exception as exc:
        print(f"\n  ERROR in auto_enter_momentum: {exc}")
        traceback.print_exc()
        send_telegram(f"⚠️ <b>Auto-Entry Momentum Error</b>\n{str(exc)[:300]}")

    if os.environ.get("ENABLE_LEGACY_PATTERN_ENTRY") == "1":
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

    # ── Best Options Trades scanner — meta-scan + Telegram on NEW A+/A/B setups ─
    # Pulls underlying candidates from ALL signal sources (momentum + PEAD +
    # whale + VIP + movers), picks best contract per name, scores via the
    # quality engine. Telegram fires only for NEW high-quality setups (deduped).
    # Throttled to once/hour to bound API load (full scan ~60s).
    try:
        from options_scanner import OptionsScanner
        # In broker mode the autonomous-trader block below runs the SAME
        # comprehensive scan (and fires the alerts) every cycle, so skip this
        # standalone hourly copy to avoid double-scanning the whole market.
        _last = conn.execute(
            "SELECT MAX(scanned_at) FROM best_options_trades").fetchone()
        _last_iso = _last[0] if _last else None
        _run_scan = (BROKER_MODE != "alpaca_paper")
        if _last_iso and _run_scan:
            try:
                _last_dt = datetime.fromisoformat(str(_last_iso).replace("Z", "+00:00"))
                if (datetime.now(_last_dt.tzinfo) - _last_dt).total_seconds() < 3600:
                    _run_scan = False
            except Exception:
                pass
        if _run_scan:
            print(f"\n  {'─'*50}")
            print(f"  BEST OPTIONS — meta-scan across all signal sources")
            _osc = OptionsScanner(conn)
            rep = _osc.scan_and_alert(telegram_sender=send_telegram,
                                      min_quality=50, alert_threshold=70)
            print(f"  Surfaced {rep['scanned']} setup(s), "
                  f"sent {rep['alerts_sent']} new alert(s).")
            for r in rep["results"][:5]:
                print(f"    {r['quality_score']:3d} {r['quality_grade']:<2}  "
                      f"{r['ticker']:6s} ${r['strike']:.0f}{r['option_type'][0].upper()} "
                      f"exp {r['expiry']} prem=${r['premium']:.2f} "
                      f"[{','.join(r['sources'])}]")
    except Exception as _oe:
        print(f"  WARN options scanner failed: {_oe}")

    # ── PANIC DETECTOR — fires high-conviction buy signal on extreme fear ──────
    # Backed by 12y of SPY+VIX data (panic_backtest.py):
    #   VIX>=40 close   → 100% win rate, +19.8% mean over 60d (40 events)
    #   SPY -5% day     → 100% win rate, +24.1% mean over 60d (6 events)
    #   COMBO_HARD      → 88.5% win rate, +12.7% mean over 60d (26 events)
    # Deduped via panic_signals table — won't re-alert until conditions normalize.
    try:
        from panic_detector import PanicDetector
        _pan_alerts = PanicDetector(conn).check(telegram_sender=send_telegram)
        if _pan_alerts:
            print(f"  PANIC SIGNAL fired: {len(_pan_alerts)} new alert(s) sent.")
            for a in _pan_alerts:
                print(f"    -> {a['signature']}")
    except Exception as _pde:
        print(f"  WARN panic detector failed: {_pde}")

    # ── NEWS-REVERSAL PUTS — alert on negative catalyst + price confirming ────
    # The validated put edge (shorting overextension was backtest-rejected).
    # Fires when a fresh bearish catalyst hits a liquid name AND price is breaking
    # down. Sends the affordable weekly PUT. Deduped by ticker+day.
    try:
        from news_reversal_puts import NewsReversalPuts
        _nrp_acct = ACCOUNT_EQUITY_OVERRIDE if ACCOUNT_EQUITY_OVERRIDE > 0 else 200.0
        _nrps = NewsReversalPuts(conn, account=_nrp_acct).find_news_reversals()
        if _nrps:
            conn.execute("CREATE TABLE IF NOT EXISTS nrp_alert_state "
                         "(ticker TEXT, alert_date TEXT, PRIMARY KEY(ticker, alert_date))")
            _today = datetime.now().strftime("%Y-%m-%d")
            for _r in _nrps[:3]:
                _seen = conn.execute("SELECT 1 FROM nrp_alert_state WHERE ticker=? "
                                     "AND alert_date=?", (_r["ticker"], _today)).fetchone()
                if _seen:
                    continue
                conn.execute("INSERT INTO nrp_alert_state (ticker, alert_date) "
                             "VALUES (?,?) ON CONFLICT DO NOTHING",
                             (_r["ticker"], _today))
                _eng = NewsReversalPuts(conn, account=_nrp_acct)
                _putmsg = _eng.put_play(_r["ticker"], _r["catalyst"])
                send_telegram(
                    f"🔻 <b>NEWS-REVERSAL PUT: {_r['ticker']}</b>\n"
                    f"Catalyst: {_r['catalyst'][:160]}\n"
                    f"Price breaking down (below 20-EMA / red). "
                    f"Validated put edge = negative catalyst + confirmation.")
                if _putmsg:
                    send_telegram(_putmsg)
                print(f"  🔻 News-reversal put alert: {_r['ticker']}")
    except Exception as _nre:
        print(f"  WARN news-reversal puts failed: {_nre}")

    # ── PUT ENGINE REGIME GATE — alert when puts become active (bearish flip) ──
    # Puts lose in bull markets (backtested), so they're gated to bear/neutral
    # regime. Alert once when the gate OPENS (SPY drops to/below 200-SMA) — that's
    # the state change worth knowing; detailed put scanning stays dashboard-only.
    try:
        from put_engine import market_allows_puts
        _pg = market_allows_puts()
        _pg_state = "active" if _pg.get("allowed") else "holstered"
        _prev = None
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS put_regime_state "
                         "(id INTEGER PRIMARY KEY, state TEXT, updated TEXT)")
            _r = conn.execute("SELECT state FROM put_regime_state WHERE id=1").fetchone()
            _prev = (_r[0] if not hasattr(_r, "get") else _r.get("state")) if _r else None
        except Exception:
            pass
        if _prev != _pg_state:
            conn.execute("INSERT INTO put_regime_state (id, state, updated) "
                         "VALUES (1,?,?) ON CONFLICT(id) DO UPDATE SET "
                         "state=excluded.state, updated=excluded.updated",
                         (_pg_state, datetime.now().isoformat()))
            if _pg_state == "active" and _prev is not None:
                send_telegram(
                    f"🔻 <b>PUT ENGINE ACTIVE</b>\n"
                    f"Regime flipped to <b>{_pg.get('regime')}</b> — {_pg.get('reason')}\n"
                    f"Puts are now in-play. Check the 🔻 Puts tab for bearish setups. "
                    f"(Reminder: even in a bear regime, individual puts are a fat-tail "
                    f"coin-flip, not a sure thing.)")
                print(f"  🔻 PUT ENGINE now ACTIVE (regime {_pg.get('regime')})")
    except Exception as _pge:
        print(f"  WARN put regime check failed: {_pge}")

    # ── SHORT-TERM REVERSAL OPTIONS — alert on backtested capitulation setups ──
    # Cheap market-wide check (2 downloads). Fires only on the strongest, rarest
    # short-horizon edges (SPY -5% day = 83% win +3.4%/d; VIX>=40 = 90% win).
    # Single-name reversal scan stays dashboard-button only (too slow per cycle).
    try:
        from short_term_options import market_panic_signals, select_short_dte_call
        _st_sigs = market_panic_signals()
        for _sig in _st_sigs:
            _c = select_short_dte_call("SPY", _sig.get("price") or 0)
            _stt = _sig["stats"]
            _line = (f"\nSuggested: SPY ${_c['strike']:.0f}C exp {_c['expiry']} "
                     f"({_c['dte']}d) @ ${_c['premium']:.2f}" if _c else "")
            send_telegram(
                f"⚡ <b>SHORT-TERM SETUP: {_stt['label']}</b>\n"
                f"Historical win rate: <b>{_stt['win']}%</b>  ·  forward {_stt['fwd']}\n"
                f"This is the strongest short-horizon edge we have. Consider a "
                f"1–7 DTE ATM/ITM call on SPY/QQQ. Exit +75% / -40% / 3-day stop."
                f"{_line}"
            )
            print(f"  ⚡ SHORT-TERM SETUP fired: {_sig['setup']}")
    except Exception as _ste:
        print(f"  WARN short-term options check failed: {_ste}")

    # ── EVENT-DRIVEN OPTIONS EXITS — close on external risk BEFORE new entries ─
    # External factors that should close an open options position regardless of
    # P&L: bearish VIP post about the underlying, high-impact negative news,
    # earnings creeping into the window, trend break on the underlying, VIX
    # spike to FEAR regime. Runs BEFORE the auto-entry block so freshly-closed
    # capital is available for new setups.
    try:
        from options_event_exits import EventDrivenExitEngine
        _eo = ts.OptionsPaperEngine(conn)
        _evx = EventDrivenExitEngine(conn).check_all_open(_eo, telegram_sender=send_telegram)
        if _evx:
            print(f"  Event-driven options exits: {len(_evx)}")
    except Exception as _eve:
        print(f"  WARN event-exit engine failed: {_eve}")

    # ── Momentum-aligned call buyer (cheap calls on validated mom leaders) ─────
    # Buys slightly-OTM calls (5-15%, 14-42 DTE, premium ≤ $5) ONLY on names that
    # pass the validated cross-sectional momentum rank. Auto-exits at +100% TP /
    # -50% SL / DTE ≤ 2 (theta cliff). Lottery sizing (~5% per ticket). All paper.
    print(f"\n  {'─'*50}")
    print(f"  MOMENTUM CALLS — Cheap leverage on the validated edge")
    try:
        from momentum_options import MomentumOptionsStrategy
        _mopt = MomentumOptionsStrategy(conn)

        # ── BROKER PATH: place real Alpaca PAPER option orders ────────────────
        if BROKER_MODE == "alpaca_paper":
            from broker import AlpacaPaperBroker
            _ob = AlpacaPaperBroker()
            _ost = _ob.options_status() if _ob.available() else {"can_buy_longs": False,
                   "msg": "no paper keys"}
            if not _ob.available() or not _ost["can_buy_longs"]:
                print(f"  ⚠️ Broker options unavailable: {_ost.get('msg')} — skipping.")
            else:
                # ── CLEAN UP STALE UNFILLED ORDERS first (limit-order hygiene) ──
                # A marketable limit fills in seconds; anything still open from a
                # prior cycle didn't fill — cancel it so it can't surprise-fill
                # later or tie up buying power over the unattended week.
                try:
                    _ncxl = _ob.cancel_stale_orders(older_than_min=5)
                    if _ncxl:
                        print(f"  🧹 Canceled {_ncxl} stale unfilled order(s).")
                except Exception as _cxe:
                    print(f"  WARN stale-order cleanup failed: {_cxe}")

                # ── AUTONOMOUS EXIT MANAGER — runs every cycle, closes on rules ─
                # +100% take profit / -50% stop / DTE<=1 time-stop. This is what
                # makes the bot safe to leave running for days unattended.
                # Rebuy-cooldown table: once a contract/underlying is EXITED, it
                # is locked out from re-entry for COOLDOWN_HOURS. This kills the
                # 2026-06-12 churn loop where the scanner rebought TXN $340 ~19x
                # in one day (stop -> rebuy -> stop), bleeding ~$1,300 the journal
                # never showed. Without this, every other fix is undone by churn.
                try:
                    conn.execute("CREATE TABLE IF NOT EXISTS broker_exit_cooldown "
                                 "(contract_symbol TEXT, underlying TEXT, "
                                 "exited_at TEXT)")
                except Exception:
                    pass
                _cooldown_hours = float(os.environ.get("REBUY_COOLDOWN_HOURS", "8") or 8)

                try:
                    # Trailing-stop exit manager (lets winners run; no profit cap)
                    _exits = _ob.manage_option_exits(conn=conn)
                    for _x in _exits:
                        # lock this contract AND underlying out of re-entry
                        try:
                            conn.execute(
                                "INSERT INTO broker_exit_cooldown "
                                "(contract_symbol, underlying, exited_at) VALUES (?,?,?)",
                                (_x["symbol"], _x.get("underlying", ""),
                                 datetime.now(timezone.utc).isoformat()))
                        except Exception:
                            pass
                        emoji = {"TRAILING_STOP": "🎯", "STOP_LOSS": "🛑",
                                 "TIME_STOP": "⏱"}.get(_x["reason"], "📕")
                        _peak = _x.get("peak_pct")
                        _peaktxt = f" (peak +{_peak:.0f}%)" if _peak is not None else ""
                        print(f"    {emoji} EXIT {_x['symbol']} {_x['reason']} "
                              f"{_x['pct']:+.0f}%{_peaktxt} (${_x['pnl']:+.0f})")
                        try:
                            from trade_journal import log_exit as _jlx
                            _jlx(conn, _x["symbol"], _x["reason"], _x["pct"], _x["pnl"])
                        except Exception:
                            pass
                        _pk = _x.get("peak_pct")
                        _pktxt = (f"\nPeak reached: +{_pk:.0f}% (trailing stop "
                                  f"banked the run)") if _pk is not None and _pk > 0 else ""
                        send_telegram(
                            f"{emoji} <b>OPTION EXIT: {_x['underlying']}</b>\n"
                            f"{_x['symbol']}\n"
                            f"Reason: {_x['reason']}  ·  closed {_x['pct']:+.0f}% "
                            f"(${_x['pnl']:+.0f}){_pktxt}")
                    if _exits:
                        print(f"  Exit manager closed {len(_exits)} option position(s).")
                except Exception as _xe:
                    print(f"  WARN exit manager failed: {_xe}")

                acct = _ob.get_account()
                real_equity = acct.get("equity", 0)
                # Size to the OVERRIDE if set (e.g. trade $100k paper as $1k)
                equity = (ACCOUNT_EQUITY_OVERRIDE if ACCOUNT_EQUITY_OVERRIDE > 0
                          else real_equity)
                budget = max(50.0, equity * 0.05)        # ~5% lottery sizing per ticket
                ticket_cap = equity * MAX_TICKET_FRACTION  # one ticket ≤ this $
                print(f"  Sizing on ${equity:,.0f} effective equity "
                      f"(real ${real_equity:,.0f}) · ~${budget:.0f}/ticket · "
                      f"cap ${ticket_cap:.0f}")

                # ══ CIRCUIT BREAKERS — the kill switches that were missing on ══
                # 2026-06-12 (the churn loop bled -$2,708 all day with NOTHING
                # halting it). Two independent floors; either trips -> no new
                # entries for the rest of the day (exits ALWAYS keep running).
                _halt_entries = False
                _halt_reason = ""
                try:
                    conn.execute("CREATE TABLE IF NOT EXISTS broker_day_start "
                                 "(snapshot_date TEXT PRIMARY KEY, equity REAL)")
                    conn.execute("CREATE TABLE IF NOT EXISTS broker_daily_entries "
                                 "(snapshot_date TEXT PRIMARY KEY, n INTEGER)")
                    _today_d = datetime.now().strftime("%Y-%m-%d")
                    _ds = conn.execute("SELECT equity FROM broker_day_start "
                                       "WHERE snapshot_date=?", (_today_d,)).fetchone()
                    if _ds is None:
                        conn.execute("INSERT INTO broker_day_start "
                                     "(snapshot_date, equity) VALUES (?,?)",
                                     (_today_d, real_equity))
                        _day_open_eq = real_equity
                    else:
                        _day_open_eq = float(_ds[0] if not hasattr(_ds, "get")
                                             else _ds.get("equity") or real_equity)

                    # 1) DAILY LOSS LIMIT — in REAL dollars. The override shrinks
                    #    per-ticket SIZE but losses still hit the real account 1:1,
                    #    so the limit must be real dollars (the 6/12 mistake was
                    #    measuring danger on the $100k scale while sizing on $1k).
                    #    Default -$200 ≈ 20% of the intended $1k bankroll per day.
                    # Default raised to -$1500 (~150% of the $1k bankroll) — this is
                    # BLOWUP INSURANCE, not a daily throttle. It only trips when the
                    # account is being destroyed (a runaway bug), because a blown
                    # account ends data collection entirely. Normal bad days run free.
                    _dd_dollars = real_equity - _day_open_eq
                    _max_loss = float(os.environ.get("MAX_DAILY_LOSS_DOLLARS", "1500") or 1500)
                    if _dd_dollars <= -_max_loss:
                        _halt_entries = True
                        _halt_reason = (f"daily loss ${_dd_dollars:+.0f} "
                                        f"(limit -${_max_loss:.0f})")

                    # 2) MAX ENTRIES / DAY — a churn backstop independent of cause.
                    _et = conn.execute("SELECT n FROM broker_daily_entries "
                                       "WHERE snapshot_date=?", (_today_d,)).fetchone()
                    _entries_today = int((_et[0] if not hasattr(_et, "get")
                                          else _et.get("n")) or 0) if _et else 0
                    # Raised to 50 — only stops a runaway churn explosion, never a
                    # normal high-frequency data-collection day of distinct trades.
                    _max_entries_day = int(os.environ.get("MAX_OPENS_PER_DAY", "50") or 50)
                    if _entries_today >= _max_entries_day:
                        _halt_entries = True
                        _halt_reason = (_halt_reason + " · " if _halt_reason else "") + \
                            f"{_entries_today}/{_max_entries_day} entries used today"

                    if _halt_entries:
                        print(f"  🛑 CIRCUIT BREAKER: {_halt_reason} — NO new entries "
                              f"(exits still active).")
                        # alert once per day
                        _ha = conn.execute("SELECT 1 FROM broker_daily_entries "
                                           "WHERE snapshot_date=? AND n<0", (_today_d,)).fetchone()
                        conn.execute("CREATE TABLE IF NOT EXISTS broker_halt_alert "
                                     "(snapshot_date TEXT PRIMARY KEY)")
                        _already = conn.execute("SELECT 1 FROM broker_halt_alert "
                                                "WHERE snapshot_date=?", (_today_d,)).fetchone()
                        if not _already:
                            send_telegram(
                                f"🛑 <b>CIRCUIT BREAKER TRIPPED</b>\n"
                                f"{_halt_reason}\n"
                                f"Real equity ${real_equity:,.2f} "
                                f"(day {_dd_dollars:+,.2f}).\n"
                                f"No new entries the rest of today. Open positions "
                                f"still managed by the exit rules. This is the "
                                f"safety that was missing on the 6/12 churn day.")
                            conn.execute("INSERT INTO broker_halt_alert "
                                         "(snapshot_date) VALUES (?)", (_today_d,))
                except Exception as _cbe:
                    print(f"  WARN circuit breaker failed: {_cbe}")

                # ── VIRTUAL $1k PORTFOLIO + auto-reset on blow-up ──────────────
                # Track the $1k bankroll carved from the real $100k. If the bot
                # loses it, don't stop — bank the result, bump a lifetime counter,
                # and start a fresh $1k off current real equity (~100 lives @ $100k).
                # Pure accounting: does NOT change trade sizing above.
                if ACCOUNT_EQUITY_OVERRIDE > 0 and real_equity > 0:
                    try:
                        import virtual_portfolio as _vp
                        _vfloor = float(os.environ.get("VIRTUAL_BLOW_FLOOR", "0") or 0)
                        _vs = _vp.step(conn, real_equity,
                                       start=ACCOUNT_EQUITY_OVERRIDE, floor=_vfloor)
                        if _vs.get("reset"):
                            print(f"  💥 VIRTUAL $1k BLOWN — run #{_vs['blown_id']} "
                                  f"wiped at ${_vs['blown_value']:,.0f}. Started "
                                  f"#{_vs['new_id']} (lifetime blow-ups: "
                                  f"{_vs['blow_count']}).")
                            send_telegram(
                                f"💥 <b>VIRTUAL $1k PORTFOLIO BLOWN</b>\n"
                                f"Run #{_vs['blown_id']} wiped out "
                                f"(ended ${_vs['blown_value']:,.0f}).\n"
                                f"Auto-started fresh run "
                                f"#{_vs['new_id']} @ ${ACCOUNT_EQUITY_OVERRIDE:,.0f}.\n"
                                f"Lifetime $1k accounts blown: <b>{_vs['blow_count']}</b>\n"
                                f"Real paper balance ${real_equity:,.0f} "
                                f"(~{int(real_equity/ACCOUNT_EQUITY_OVERRIDE)} lives left).")
                        else:
                            print(f"  💼 Virtual $1k run #{_vs['active_id']} value "
                                  f"${_vs['value']:,.0f} · lifetime blow-ups "
                                  f"{_vs['blow_count']}")
                    except Exception as _vpe:
                        print(f"  WARN virtual portfolio step failed: {_vpe}")
                # ── MACRO EVENT GATE — don't buy into a binary data print ──────
                # MACRO_GATE env modes:
                #   hard — block ALL new entries while event risk is HIGH (the
                #          right setting for real money)
                #   soft — keep trading through CPI/FOMC weeks at HALF size
                #          (default for the paper data-collection account: with
                #          CPI+FOMC in the same week a hard gate starves the
                #          sample for half the month; half-size keeps discipline
                #          AND data flowing — event-week fills are honest data)
                #   off  — no gate (not recommended)
                # Exits always run regardless.
                _macro_block = False
                _macro_size_mult = 1.0
                _gate_mode = os.environ.get("MACRO_GATE", "soft").strip().lower()
                try:
                    from macro_engine import event_risk as _erk
                    if _erk().get("level") == "HIGH":
                        if _gate_mode == "hard":
                            _macro_block = True
                            print("  ⏸ Macro event imminent (HIGH risk) — gate=hard, "
                                  "skipping new option buys (exits still active).")
                        elif _gate_mode == "off":
                            print("  ⚠️ Macro event imminent (HIGH risk) — gate=off, "
                                  "trading full size anyway.")
                        else:
                            _macro_size_mult = 0.5
                            print("  ⚠️ Macro event imminent (HIGH risk) — gate=soft, "
                                  "new entries at HALF size this cycle.")
                except Exception:
                    pass

                # ── OPENING-VOLATILITY ENTRY GATE ──────────────────────────────
                # The 2026-06-12 CPI whipsaw taught this the hard way: entries
                # fired at 9:56 ET into the open got stopped at the 10:05 low,
                # then ran +58% to +177%. Don't buy into the opening chaos. Block
                # new entries for the first NO_ENTRY_MIN minutes after the open
                # (longer on HIGH-macro days). Exits + overnight edge still run.
                _entry_block_open = False
                try:
                    import trading_scanner as _tse
                    _et = _tse.MarketClock.now_et()
                    _mins_since_open = (_et.hour - 9) * 60 + (_et.minute - 30)
                    _no_entry = int(os.environ.get("NO_ENTRY_MIN", "45") or 45)
                    # widen the morning lockout on high-volatility macro days
                    if _macro_size_mult < 1.0 or _macro_block:
                        _no_entry = max(_no_entry, int(
                            os.environ.get("NO_ENTRY_MIN_MACRO", "75") or 75))
                    if 0 <= _mins_since_open < _no_entry:
                        _entry_block_open = True
                        print(f"  ⏳ Opening-volatility gate: {_mins_since_open}m "
                              f"since open < {_no_entry}m — no new entries yet "
                              f"(let the open settle). Exits still active.")
                except Exception:
                    pass

                held_u = _ob.held_option_underlyings()

                # ── Single-entry helper — shared by the real-strategy pass and
                # the data-collection floor pass so the order/journal/Telegram
                # logic lives in exactly one place. `setup_tag` is what lands in
                # the journal: 'momentum_call' (real edge) vs
                # 'momentum_call_explore' (data-floor top-up, kept separate).
                def _attempt_entry(s, setup_tag, budget_override=None):
                    # soft macro gate halves the per-ticket budget during events
                    _bud = (budget_override if budget_override else budget) * _macro_size_mult
                    _otype = s.get("option_type", "call")
                    _oc = _otype[0].upper()                  # 'C' or 'P'
                    if s["ticker"] in held_u:
                        return False
                    # REBUY COOLDOWN — don't re-enter a contract/underlying we exited
                    # within the cooldown window (stops the stop->rebuy->stop churn).
                    # Block ONLY the exact contract we just exited (not the whole
                    # underlying) — re-buying the SAME contract is duplicate/noise
                    # data and was the churn bug; a DIFFERENT strike/expiry on the
                    # same name later is a legitimate distinct data point.
                    try:
                        _cut = (datetime.now(timezone.utc)
                                - timedelta(hours=_cooldown_hours)).isoformat()
                        _cd = conn.execute(
                            "SELECT 1 FROM broker_exit_cooldown WHERE exited_at >= ? "
                            "AND contract_symbol = ? LIMIT 1",
                            (_cut, s.get("contract_symbol", ""))
                        ).fetchone()
                        if _cd:
                            print(f"    {s.get('contract_symbol','?')} — same contract "
                                  f"exited < {_cooldown_hours:.0f}h ago (anti-churn), skip")
                            return False
                    except Exception:
                        pass
                    contract = _ob.find_option_contract(
                        s["ticker"], s["expiry"], _otype, s["strike"])
                    if not contract or not contract.get("tradable"):
                        print(f"    {s['ticker']:6s} — no tradable Alpaca {_otype}, skip")
                        return False

                    # ── LIQUIDITY + REAL-PRICE GATE (the 6/12 slippage fix) ─────
                    # Size and cap on the LIVE ask, not the stale scan estimate
                    # (SOXX was estimated $1.30 but really $4.80). Reject wide
                    # spreads, and buy with a LIMIT capped just above the ask so a
                    # thin book can't fill us far above the quote.
                    # FAIL-OPEN: if the options-quote endpoint isn't available on
                    # this account, DON'T halt all trading — fall back to the scan
                    # estimate with a conservative limit (fills when the estimate is
                    # roughly right; harmlessly no-fills when it's way off, which is
                    # exactly the bad-fill case we want to avoid anyway).
                    _q = _ob.get_option_quote(contract["symbol"])
                    _max_spread = float(os.environ.get("MAX_OPT_SPREAD_PCT", "25") or 25)
                    if _q:
                        if _q["spread_pct"] > _max_spread:
                            print(f"    {s['ticker']:6s} — spread {_q['spread_pct']:.0f}% "
                                  f"> {_max_spread:.0f}% (illiquid), skip")
                            return False
                        prem = _q["ask"]                       # REAL price we'd pay
                        _limit_px = round(_q["ask"] * 1.02, 2)  # cap slippage to +2%
                    else:
                        prem = s.get("premium") or contract.get("close_price") or 0
                        if prem <= 0:
                            print(f"    {s['ticker']:6s} — no quote and no estimate, skip")
                            return False
                        _limit_px = round(prem * 1.10, 2)     # est + 10% buffer
                        print(f"    {s['ticker']:6s} — no live quote; fallback to "
                              f"estimate ${prem:.2f} w/ limit ${_limit_px:.2f}")
                    one_ct_cost = prem * 100 if prem > 0 else 0
                    # Per-ticket cap: skip contracts too expensive for the account
                    if one_ct_cost > ticket_cap:
                        print(f"    {s['ticker']:6s} — 1 contract ${one_ct_cost:.0f} "
                              f"> cap ${ticket_cap:.0f} (live ask), skip")
                        return False
                    qty = max(1, int(_bud // one_ct_cost)) if one_ct_cost > 0 else 1

                    # ── WINNER GATE (meta-label) — separate likely winners from losers ─
                    # Computes the pre-trade separators (trend, momentum, range
                    # position, strike reachability) and ALWAYS records them to the
                    # journal so the meta-model accumulates labelled data. When
                    # WINNER_GATE=on it ALSO vetoes low-probability call entries; off
                    # (the default) it only shadow-logs, changing nothing live. Fails
                    # OPEN: if features can't be computed it never blocks a trade.
                    _wg_feats = {}
                    _wg_passed = None
                    _wg_score = None
                    try:
                        from winner_gate import (compute_entry_features as _wgcf,
                                                 evaluate as _wgeval)
                        _wg_feats = _wgcf(s["ticker"], otm_pct=s.get("otm_pct"),
                                          dte=s.get("dte"), iv=s.get("iv"))
                        if _wg_feats:
                            _wgr = _wgeval(_wg_feats)
                            _wg_passed = bool(_wgr["passed"])
                            _wg_score = _wgr["score"]
                            _wg_on = (os.environ.get("WINNER_GATE", "off").strip()
                                      .lower() in ("on", "1", "true", "enforce"))
                            if _otype == "call" and not _wg_passed:
                                if _wg_on:
                                    print(f"    ⛔ {s['ticker']:6s} — winner gate REJECT "
                                          f"(score {_wg_score:.0f}): "
                                          f"{'; '.join(_wgr['reasons'])}")
                                    return False
                                print(f"    ⚠️ {s['ticker']:6s} — winner gate would skip "
                                      f"(shadow; score {_wg_score:.0f}): "
                                      f"{'; '.join(_wgr['reasons'])}")
                    except Exception:
                        pass

                    # ── OPTION STRUCTURE — optional DEBIT SPREAD (opt-in, CALLS only) ─
                    # OPTION_STRUCTURE=spread turns each CALL entry into a bull-call
                    # debit spread: LONG the near-money call already picked, SHORT
                    # ~10% further OTM at the SAME expiry. Validated NET in
                    # full_bot_backtest.py `compare` (+7pp win rate; typical trade
                    # -32% -> -4%). DEFAULT is 'naked' (unchanged). ANY failure here
                    # falls through to the naked buy below, so it can only improve or
                    # no-op — it never blocks a trade.
                    _is_spread = False
                    _short_sym = ""
                    res = {"ok": False}
                    if (os.environ.get("OPTION_STRUCTURE", "naked").strip().lower()
                            == "spread" and _otype == "call"):
                        try:
                            from momentum_options import SPREAD_SHORT_OTM as _SSO
                            _under = contract["strike"] / (1 + (s.get("otm_pct") or 0.0))
                            _short = _ob.find_option_contract(
                                s["ticker"], s["expiry"], "call", _under * (1 + _SSO))
                            if (_short and _short.get("tradable")
                                    and _short["symbol"] != contract["symbol"]):
                                _sq = _ob.get_option_quote(_short["symbol"])
                                _short_bid = (_sq["bid"] if _sq
                                              else max(0.01, prem * 0.30))
                                _net_debit = max(0.05, prem - _short_bid)
                                _debit_cost = _net_debit * 100
                                qty = (max(1, int(_bud // _debit_cost))
                                       if _debit_cost > 0 else 1)
                                res = _ob.submit_option_spread(
                                    contract["symbol"], _short["symbol"], qty,
                                    net_debit_limit=round(_net_debit * 1.05, 2))
                                if res.get("ok"):
                                    _is_spread = True
                                    _short_sym = _short["symbol"]
                                    prem = _net_debit   # net debit is the cost basis
                                    # persist the leg pairing so the exit manager
                                    # treats the spread as ONE unit (close both legs,
                                    # stop on net value, never strip the short hedge).
                                    try:
                                        conn.execute(
                                            "CREATE TABLE IF NOT EXISTS broker_spread_legs "
                                            "(long_symbol TEXT PRIMARY KEY, short_symbol "
                                            "TEXT, underlying TEXT, entry_debit REAL, "
                                            "opened_at TEXT)")
                                        conn.execute(
                                            "INSERT OR REPLACE INTO broker_spread_legs "
                                            "(long_symbol, short_symbol, underlying, "
                                            "entry_debit, opened_at) VALUES (?,?,?,?,?)",
                                            (contract["symbol"], _short["symbol"],
                                             s["ticker"], _net_debit,
                                             datetime.now(timezone.utc).isoformat()))
                                    except Exception:
                                        pass
                                else:
                                    print(f"    spread order failed "
                                          f"({res.get('error')}); naked fallback")
                            else:
                                print(f"    {s['ticker']:6s} — no tradable short leg "
                                      f"for spread; naked fallback")
                        except Exception as _spe:
                            print(f"    spread setup error ({_spe}); naked fallback")

                    if not _is_spread:
                        res = _ob.submit_option_buy(contract["symbol"], qty,
                                                    limit_price=_limit_px)
                    if not res.get("ok"):
                        print(f"    ✗ {s['ticker']:6s} — Alpaca opt order failed: "
                              f"{res.get('error')}")
                        return False
                    held_u.add(s["ticker"])     # don't double-buy this cycle
                    # bump the per-day entry counter (circuit-breaker backstop)
                    try:
                        conn.execute(
                            "INSERT INTO broker_daily_entries (snapshot_date, n) "
                            "VALUES (?,1) ON CONFLICT (snapshot_date) DO UPDATE "
                            "SET n = broker_daily_entries.n + 1",
                            (datetime.now().strftime("%Y-%m-%d"),))
                    except Exception:
                        pass
                    _is_ex = setup_tag.endswith("explore")
                    print(f"    {'🧪' if _is_ex else '✅'} ALPACA "
                          f"{'OPT SPREAD' if _is_spread else 'OPT BUY'} "
                          f"{contract['symbol']}"
                          f"{(' /short ' + _short_sym) if _is_spread else ''} "
                          f"x{qty} (~${prem:.2f}/ct, ${prem*100*qty:.0f})"
                          f"{' [data-floor]' if _is_ex else ''}")
                    # ── Journal: full attribution for later optimization ──
                    try:
                        from trade_journal import log_entry as _jle
                        _jsec = ""
                        try:
                            import yfinance as _yfj
                            _jsec = str((_yfj.Ticker(s["ticker"]).info or {}
                                        ).get("sector", "") or "")
                        except Exception:
                            pass
                        _jle(conn, contract_symbol=contract["symbol"],
                             underlying=s["ticker"], setup=setup_tag,
                             quality_score=s.get("quality_score"), sector=_jsec,
                             dte=s.get("dte"), iv=s.get("iv"),
                             otm_pct=s.get("otm_pct"), mom_6m=s.get("mom_6m"),
                             mom_3m=s.get("mom_3m"), entry_premium=prem,
                             qty=qty, cost=prem * 100 * qty,
                             rv_at_entry=_wg_feats.get("rv"),
                             rng_pos=_wg_feats.get("rng_pos"),
                             in_uptrend=_wg_feats.get("in_uptrend"),
                             reach=_wg_feats.get("reach"),
                             gate_passed=_wg_passed, gate_score=_wg_score)
                    except Exception as _je:
                        print(f"    (journal log_entry skipped: {_je})")
                    _src = ", ".join(s.get("sources", []) or [])
                    _why = f" · why: {_src}" if _src else ""
                    _dir_emoji = "📉" if _otype == "put" else "📈"
                    send_telegram(
                        f"{_dir_emoji} <b>ALPACA PAPER "
                        f"{'CALL DEBIT SPREAD' if _is_spread else 'OPTION BUY'} — "
                        f"{'PUT' if _otype=='put' else 'CALL'}</b>"
                        f"{' 🧪 (data-floor)' if _is_ex else ''}\n"
                        f"{s['ticker']} ${contract['strike']:.0f}{_oc} exp {s['expiry']}"
                        f"{(' / short ' + _short_sym) if _is_spread else ''}\n"
                        f"Qty: {qty} {'spread(s)' if _is_spread else 'contract(s)'} "
                        f"@ ~${prem:.2f} {'net debit' if _is_spread else ''} "
                        f"(~${prem*100*qty:.0f})\n"
                        f"Quality {s.get('quality_score','?')}/100{_why}\n"
                        f"{'DATA-FLOOR top-up (not the core edge)' if _is_ex else 'Full-market scan · real broker fills'}\n"
                        f"Acct equity ${equity:,.0f}")
                    return True

                # ── 🧠 AUTONOMOUS PM BRAIN — fully cloud, no PC required ──────
                # Once per trading day (first cycle at/after 9:20 ET) the brain
                # reviews the compacted data pack via the free AI provider and
                # writes directives. If the user's local real-Claude routine
                # already decided today, the brain defers (skip guard inside).
                try:
                    import trading_scanner as _tsb
                    _et_now = _tsb.MarketClock.now_et()
                    if (_et_now.hour, _et_now.minute) >= (9, 20):
                        import claude_pm as _cpmb
                        _br = _cpmb.run_brain(conn, telegram=send_telegram)
                        if _br.get("ran"):
                            print(f"  🧠 PM brain decided: {_br.get('actions')} "
                                  f"(inserted {_br.get('inserted')})")
                        elif "already exist" not in str(_br.get("why", "")):
                            print(f"  🧠 PM brain skipped: {_br.get('why')}")
                except Exception as _bre:
                    print(f"  WARN PM brain failed: {_bre}")

                # ── 🤖 CLAUDE PM DIRECTIVES — execute the AI portfolio manager ─
                # A scheduled Claude agent reviews the full data pack each
                # morning and writes directives; the bot executes them here with
                # guardrails (24h TTL, max 3 opens/day, paper only). PM trades
                # are tagged claude_pm_* in the journal — tracked separately so
                # "is Claude better than the bot?" gets answered with data.
                _claude_pause = False
                try:
                    import claude_pm as _cpm
                    _cpm.ensure_tables(conn)
                    # persist live state snapshot for the PM's data pack
                    try:
                        import json as _json
                        _snap = {"account": {k: acct.get(k) for k in
                                             ("equity", "cash", "buying_power",
                                              "day_pnl", "day_pnl_pct")},
                                 "effective_equity": equity,
                                 "option_positions": _ob.get_option_positions(),
                                 "stock_positions": _ob.get_positions()}
                        conn.execute(
                            "INSERT INTO broker_state_snapshot (snap_key, taken_at, payload) "
                            "VALUES ('latest', ?, ?) ON CONFLICT (snap_key) DO UPDATE "
                            "SET taken_at=excluded.taken_at, payload=excluded.payload",
                            (datetime.now(timezone.utc).isoformat(),
                             _json.dumps(_snap, default=str)))
                    except Exception as _spe:
                        print(f"  (state snapshot skipped: {_spe})")

                    _pm_opens_today = 0
                    try:
                        _r = conn.execute(
                            "SELECT COUNT(*) FROM claude_directives WHERE "
                            "status='EXECUTED' AND action IN ('OPEN_CALL','OPEN_PUT') "
                            "AND executed_at >= ?",
                            (datetime.now(timezone.utc).strftime("%Y-%m-%d"),)).fetchone()
                        _pm_opens_today = int(_r[0] if not hasattr(_r, "get")
                                              else _r.get("count", 0) or 0)
                    except Exception:
                        pass

                    for _d in _cpm.pending(conn):
                        _did = _d.get("id"); _act = str(_d.get("action", "")).upper()
                        _dtk = str(_d.get("ticker", "") or "").upper()
                        _dsym = str(_d.get("symbol", "") or "").upper()
                        _rat = str(_d.get("rationale", "") or "")[:400]
                        _status, _result = "EXECUTED", ""
                        try:
                            if _act == "NOTE":
                                send_telegram(f"🤖 <b>CLAUDE PM — MARKET READ</b>\n{_rat}")
                                _result = "note sent"
                            elif _act == "PAUSE_ENTRIES":
                                _claude_pause = True
                                send_telegram(f"🤖 <b>CLAUDE PM — ENTRIES PAUSED TODAY</b>\n{_rat}")
                                _result = "entries paused this cycle-day"
                            elif _act == "CLOSE_ALL_OPTIONS":
                                _n = 0
                                for _p in _ob.get_option_positions():
                                    if _ob.close_option(_p["symbol"]).get("ok"):
                                        _n += 1
                                send_telegram(f"🤖 <b>CLAUDE PM — FLATTENED OPTIONS BOOK</b>\n"
                                              f"Closed {_n} position(s).\n{_rat}")
                                _result = f"closed {_n}"
                            elif _act == "CLOSE_OPTION" and _dsym:
                                _r2 = _ob.close_option(_dsym)
                                if _r2.get("ok"):
                                    send_telegram(f"🤖 <b>CLAUDE PM — CLOSED {_dsym}</b>\n{_rat}")
                                    _result = "closed"
                                else:
                                    _status, _result = "FAILED", str(_r2.get("error"))[:200]
                            elif _act in ("OPEN_CALL", "OPEN_PUT") and _dtk:
                                if _halt_entries:
                                    _status, _result = "SKIPPED", f"circuit breaker: {_halt_reason}"
                                elif _entry_block_open:
                                    _status, _result = "PENDING", "waiting out opening-volatility window"
                                    # leave PENDING so it executes a later cycle
                                    continue
                                elif _pm_opens_today >= 3:
                                    _status, _result = "SKIPPED", "max 3 PM opens/day"
                                else:
                                    _otype = "call" if _act == "OPEN_CALL" else "put"
                                    _px = _ob.get_price(_dtk) or 0
                                    if _otype == "call":
                                        from momentum_options import select_call_contract as _scc
                                        _c = _scc(_dtk, _px) if _px > 0 else None
                                    else:
                                        from put_engine import select_put_contract as _spc
                                        _c = _spc(_dtk, _px) if _px > 0 else None
                                    if not _c:
                                        _status, _result = "FAILED", "no suitable contract"
                                    else:
                                        _c["option_type"] = _otype
                                        _c["sources"] = ["claude_pm"]
                                        _c["quality_score"] = _c.get("quality_score")
                                        if _attempt_entry(_c, f"claude_pm_{_otype}",
                                                          budget_override=budget * 0.6):
                                            _pm_opens_today += 1
                                            _result = f"opened {_c.get('contract_symbol','?')}"
                                            send_telegram(f"🤖 <b>CLAUDE PM RATIONALE</b>\n{_rat}")
                                        else:
                                            _status, _result = "FAILED", "entry blocked (cap/held/order)"
                            else:
                                _status, _result = "FAILED", "bad directive shape"
                        except Exception as _dxe:
                            _status, _result = "FAILED", str(_dxe)[:200]
                        try:
                            conn.execute(
                                "UPDATE claude_directives SET status=?, executed_at=?, "
                                "result=? WHERE id=?",
                                (_status, datetime.now(timezone.utc).isoformat(),
                                 _result, _did))
                        except Exception:
                            pass
                        print(f"  🤖 PM directive {_act} {_dtk or _dsym}: {_status} ({_result})")
                except Exception as _cpe:
                    print(f"  WARN claude_pm executor failed: {_cpe}")

                # ══ FULL MARKET SCAN — the trader's primary engine ═════════════
                # ONE scan across EVERY validated tool on the site (momentum,
                # bottom-fisher, PEAD, whale, VIP, movers, put-engine, news-puts,
                # reversal, panic). Returns ONE ranked list of CALLS and PUTS,
                # each quality-scored by options_analytics. The paper trader buys
                # the top setups from this single source of truth. This same call
                # persists results (Best Options table) and fires NEW-setup
                # Telegram alerts, so it replaces the standalone hourly meta-scan.
                opened = 0
                if (not _macro_block and not _claude_pause
                        and not _entry_block_open and not _halt_entries):
                    try:
                        from options_scanner import OptionsScanner
                        _fms = OptionsScanner(conn)
                        _rep = _fms.scan_and_alert(
                            telegram_sender=send_telegram,
                            min_quality=55, alert_threshold=70,
                            fast=True, max_results=12)
                        _fs = _rep["results"]
                        print(f"  🌐 Full market scan: {len(_fs)} setup(s) "
                              f"(quality≥55), {_rep['alerts_sent']} new alert(s).")
                        # Execute the top setups. Momentum = full size (the primary
                        # p=0.004 edge); every other source is SECONDARY → 60% size
                        # (lower win-rate edges, sized down for discipline). PUTS
                        # only fire when the put-engine regime gate already allowed
                        # them upstream, so reaching here means the regime is OK.
                        for s in _fs:
                            if opened >= 5:
                                break
                            _srcs = s.get("sources", [])
                            _is_primary = ("momentum" in _srcs)
                            _tag = ("fullscan_" + (_srcs[0] if _srcs else "scan")
                                    + "_" + s.get("option_type", "call"))
                            _bo = None if _is_primary else budget * 0.6
                            if _attempt_entry(s, _tag, budget_override=_bo):
                                opened += 1
                    except Exception as _fmse:
                        print(f"  WARN full market scan failed: {_fmse}")
                        traceback.print_exc()

                # ── DATA-COLLECTION FLOOR — guarantee a minimum daily sample ───
                # Strict gates can mean very few trades, so it takes weeks to learn
                # whether the edge is real. MIN_TRADES_PER_DAY (env, default 0 =
                # off) tops up the day with the BEST setups *below* the normal bar,
                # TAGGED 'momentum_call_explore' so they live in a separate journal
                # bucket and never inflate or deflate the real-strategy win rate.
                _min_day = int(os.environ.get("MIN_TRADES_PER_DAY", "0") or 0)
                if (_min_day > 0 and not _macro_block and not _claude_pause
                        and not _entry_block_open and not _halt_entries):
                    try:
                        from trade_journal import count_opened_today as _cot
                        _done = _cot(conn)
                    except Exception:
                        _done = opened
                    _need = _min_day - _done
                    if _need > 0:
                        _ex_floor = float(os.environ.get("EXPLORE_QUALITY_FLOOR", "35") or 35)
                        print(f"  📊 Data floor: {_done}/{_min_day} trades today — "
                              f"topping up up to {_need} exploration trade(s) "
                              f"(quality ≥ {_ex_floor:.0f}, tagged separately).")
                        for s in _mopt.find_setups(top_n_underlyings=25,
                                                   min_quality_score=_ex_floor):
                            if _need <= 0:
                                break
                            if _attempt_entry(s, "momentum_call_explore"):
                                opened += 1
                                _need -= 1
                    else:
                        print(f"  📊 Data floor met: {_done}/{_min_day} trades today.")

                print(f"  Momentum options (Alpaca paper): opened {opened} this cycle")

                # ── Once-per-day broker scorecard (track profitability over days)
                try:
                    conn.execute("CREATE TABLE IF NOT EXISTS broker_equity_log "
                                 "(snapshot_date TEXT PRIMARY KEY, equity REAL, "
                                 "open_options INTEGER, logged_at TEXT)")
                    _today = datetime.now().strftime("%Y-%m-%d")
                    _row = conn.execute("SELECT equity FROM broker_equity_log "
                                        "WHERE snapshot_date=?", (_today,)).fetchone()
                    if _row is None:
                        _nopt = len(_ob.get_option_positions())
                        conn.execute("INSERT INTO broker_equity_log "
                                     "(snapshot_date, equity, open_options, logged_at) "
                                     "VALUES (?,?,?,?)",
                                     (_today, real_equity, _nopt,
                                      datetime.now().isoformat()))
                        # first-seen equity = baseline for the curve
                        _base = conn.execute("SELECT equity FROM broker_equity_log "
                                             "ORDER BY snapshot_date ASC LIMIT 1").fetchone()
                        _b0 = float(_base[0] if not hasattr(_base,'get') else _base.get('equity')) if _base else real_equity
                        _chg = (real_equity / _b0 - 1) * 100 if _b0 else 0
                        # GROUND-TRUTH realized P&L from actual fills (the journal
                        # under-counts churn). Surface churn so it can't hide again.
                        _recon = _ob.realized_pnl_from_fills()
                        _churn_txt = ""
                        if _recon.get("churn"):
                            _ct = ", ".join(f"{s.split()[0][:6]}×{n}"
                                            for s, n in _recon["churn"][:4])
                            _churn_txt = (f"\n⚠️ Churn detected (round-trips): {_ct}"
                                          f"\nTotal round-trips: {_recon['round_trips']}")
                        send_telegram(
                            f"🏦 <b>BROKER DAILY SCORECARD</b>\n"
                            f"Real equity: ${real_equity:,.2f}\n"
                            f"Since start: {_chg:+.2f}%  (baseline ${_b0:,.0f})\n"
                            f"<b>True realized P&L (from fills): ${_recon['realized_pnl']:+,.2f}</b>\n"
                            f"Open option positions: {_nopt}\n"
                            f"Day P&L: ${acct.get('day_pnl',0):+,.2f}"
                            f"{_churn_txt}\n"
                            f"<i>Realized P&L is computed from actual Alpaca fills, "
                            f"not the journal estimate.</i>")
                        print(f"  Broker scorecard: equity ${real_equity:,.2f} "
                              f"({_chg:+.2f}%), true realized ${_recon['realized_pnl']:+,.2f}, "
                              f"{_recon['round_trips']} round-trips")

                    # ── FRIDAY WEEK-IN-REVIEW — the one verdict message ────────
                    # On the first Friday cycle, send a consolidated summary: real
                    # P&L from fills + which GRADES actually won (attribution).
                    if datetime.now().weekday() == 4:
                        conn.execute("CREATE TABLE IF NOT EXISTS broker_week_review "
                                     "(snapshot_date TEXT PRIMARY KEY)")
                        _wr = conn.execute("SELECT 1 FROM broker_week_review "
                                           "WHERE snapshot_date=?", (_today,)).fetchone()
                        if not _wr:
                            # win rate by grade (attribution — reliable even if $ are est.)
                            _grade_lines = []
                            try:
                                _gr = conn.execute(
                                    "SELECT quality_band g, COUNT(*) n, "
                                    "SUM(CASE WHEN pnl_pct>0 THEN 1 ELSE 0 END) w "
                                    "FROM broker_trade_journal WHERE status='CLOSED' "
                                    "AND quality_band IS NOT NULL GROUP BY quality_band "
                                    "ORDER BY quality_band").fetchall()
                                for _r in _gr:
                                    _d = dict(_r); _n = int(_d['n'] or 0)
                                    if _n:
                                        _grade_lines.append(
                                            f"{_d['g']}: {int(_d['w'] or 0)}/{_n} "
                                            f"({100*(_d['w'] or 0)/_n:.0f}%)")
                            except Exception:
                                pass
                            _opos2 = _ob.get_option_positions()
                            _unreal2 = sum(float(p.get("unrealized_pnl", 0) or 0)
                                           for p in _opos2)
                            send_telegram(
                                f"📅 <b>WEEK IN REVIEW</b>\n"
                                f"Real equity: ${real_equity:,.2f} "
                                f"({_chg:+.2f}% since start)\n"
                                f"Realized P&L (fills): ${_recon['realized_pnl']:+,.2f}\n"
                                f"Open unrealized: ${_unreal2:+,.2f} "
                                f"({len(_opos2)} positions)\n"
                                f"Total round-trips: {_recon['round_trips']}\n"
                                f"<b>Win rate by grade:</b> "
                                f"{' · '.join(_grade_lines) if _grade_lines else 'no closed trades yet'}\n"
                                f"<i>Judge the PROCESS + grade edge, not just the P&L. "
                                f"Full detail on the dashboard.</i>")
                            conn.execute("INSERT INTO broker_week_review "
                                         "(snapshot_date) VALUES (?)", (_today,))
                            print("  📅 Week-in-review sent.")
                except Exception as _sce:
                    print(f"  WARN broker scorecard failed: {_sce}")
        else:
            # Internal simulation path (default)
            _opt_engine = ts.OptionsPaperEngine(conn)
            _opt_engine.expire_check()          # housekeeping: close expired worthless
            n_closed = _mopt.auto_exit(_opt_engine)
            n_opened = _mopt.auto_enter(_opt_engine, market_regime=market_regime)
            print(f"  Momentum calls: closed {n_closed}, opened {n_opened}")
    except Exception as exc:
        print(f"  ERROR in momentum options: {exc}")
        traceback.print_exc()
        send_telegram(f"⚠️ <b>Momentum Options Error</b>\n{str(exc)[:300]}")

    # ── Options Callout Tracker — aggregate social call/put ideas + track P&L ──
    # Research-only: pulls options callouts from StockTwits (+ Reddit if creds),
    # snapshots the called option's premium NOW, and tracks forward P&L so we
    # build a leaderboard of which callers are actually profitable.
    print(f"\n  {'─'*50}")
    print(f"  OPTIONS CALLOUTS — Social call/put aggregator")
    try:
        from options_callouts import OptionsCalloutTracker
        _oct = OptionsCalloutTracker(conn)
        # First update outcomes on existing open callouts (cheap, bounded)
        _trk = _oct.track_outcomes(max_per_cycle=40)
        # Then ingest fresh callouts (snapshots premium for new ones)
        _ing = _oct.ingest_callouts(include_reddit=True)
        print(f"  Ingest : {_ing.get('new_stored',0)} new "
              f"(fetched {_ing.get('fetched',0)}, "
              f"dup {_ing.get('skipped_duplicate',0)}, "
              f"no-premium {_ing.get('skipped_no_premium',0)})")
        print(f"  Track  : {_trk.get('updated',0)} updated, "
              f"{_trk.get('expired',0)} closed/expired")
        # Surface any standout winners to Telegram (open callouts up big)
        try:
            for w in (_oct.get_recent_winners(hours=24, limit=2) or []):
                send_telegram(
                    f"📢 <b>Hot Options Callout</b>\n"
                    f"<b>${w.get('ticker','')}</b> "
                    f"{str(w.get('option_type','')).upper()} "
                    f"{('$'+str(w.get('strike'))) if w.get('strike') else 'ATM'}\n"
                    f"P&L: {w.get('pnl_pct',0):+.0f}% since called\n"
                    f"By @{w.get('username','?')} on {w.get('source','')}"
                )
        except Exception:
            pass
    except Exception as exc:
        print(f"  WARN options callout tracker failed: {exc}")

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
