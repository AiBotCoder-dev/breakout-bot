#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
options_callouts.py — Social Options Callout Tracker
======================================================
Aggregates options trade ideas from social platforms and tracks their
actual profitability over time.  Builds a leaderboard of profitable
callers so you know which voices to trust.

SOURCES (all free):
  1. StockTwits     — public stream, no auth needed
  2. Reddit         — r/options + r/wallstreetbets + r/SPACs (requires PRAW
                      credentials set as REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET)

WORKFLOW:
  1. ingest_callouts()       — pull messages, parse for $TICKER + call/put
                                 + strike + expiry, snapshot CURRENT option
                                 price as "entry price" for tracking
  2. track_outcomes()        — for each open callout, fetch current option
                                 price, compute P&L%, mark expired ones final
  3. get_leaderboard()       — rank callers by win-rate + avg return on closed
                                 callouts (excluding callouts <24h old)
  4. get_active_callouts()   — list ongoing callouts with live P&L
  5. get_recent_winners()    — see what's currently working

REALISTIC EXPECTATIONS:
  • Most callouts on social are NOISE — historical research suggests
    5-15% are clearly profitable.  Leaderboard finds the signal.
  • This is RESEARCH not advice — never auto-trade based on callouts.
  • Limited by yfinance's lack of historical option prices — we snapshot
    NOW and track forward, can't look up what an option cost yesterday.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

try:
    import requests
except ImportError:
    requests = None


# ══════════════════════════════════════════════════════════════════════════════
# CALLOUT PARSER — extracts ticker / call-or-put / strike / expiry from text
# ══════════════════════════════════════════════════════════════════════════════

# Common callout formats observed on StockTwits + Reddit:
#   $AAPL 200c            $AAPL $200 call       $AAPL 200 calls 11/15
#   AAPL 200c 11/15       loaded AAPL 200 calls bought NVDA puts
#   $TSLA $250p 12/20     SPY $440c 0DTE        QQQ $400c weeklies

_TICKER_RE = re.compile(r"\$?([A-Z]{1,5})(?:\b|\s|$)")
_OPTIONS_KEYWORDS = re.compile(
    r"\b(call|calls|cc|put|puts|p\b|c\b|option|options|"
    r"weekly|weeklies|0dte|monthly|strike|premium|"
    r"iv|theta|gamma|premium)\b",
    re.IGNORECASE,
)

# Patterns ordered most-specific → least.  First match wins.
# NOTE: the ticker group is wrapped in (?-i:...) to force CASE-SENSITIVE
# uppercase matching even though the rest of the pattern is case-insensitive.
# This prevents lowercase English words ("puts", "my", "call") from being
# mis-captured as tickers while still matching "call"/"put"/"calls" suffixes
# in any case.
_CALL_PATTERNS = [
    # $AAPL $200c, $AAPL 200c, AAPL 200c, AAPL $200c
    re.compile(
        r"\$?(?-i:([A-Z]{1,5}))\s+\$?(\d+(?:\.\d+)?)\s*[cC]\b",
        re.IGNORECASE,
    ),
    # $AAPL 200 call / $AAPL 200 calls
    re.compile(
        r"\$?(?-i:([A-Z]{1,5}))\s+\$?(\d+(?:\.\d+)?)\s+calls?\b",
        re.IGNORECASE,
    ),
    # bought / buying / loaded / scooping $AAPL calls (no strike — store as ATM)
    re.compile(
        r"(?:bought|buying|loaded|scooping|grabbing|picked up|holding)\s+"
        r"\$?(?-i:([A-Z]{1,5}))\s+calls?\b",
        re.IGNORECASE,
    ),
    # $AAPL calls (generic, no strike specified) — requires the $ prefix
    re.compile(
        r"\$(?-i:([A-Z]{1,5}))\s+calls?\b",
        re.IGNORECASE,
    ),
]

_PUT_PATTERNS = [
    re.compile(
        r"\$?(?-i:([A-Z]{1,5}))\s+\$?(\d+(?:\.\d+)?)\s*[pP]\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\$?(?-i:([A-Z]{1,5}))\s+\$?(\d+(?:\.\d+)?)\s+puts?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:bought|buying|loaded|scooping|grabbing|picked up|holding)\s+"
        r"\$?(?-i:([A-Z]{1,5}))\s+puts?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\$(?-i:([A-Z]{1,5}))\s+puts?\b",
        re.IGNORECASE,
    ),
]

# Expiry: matches 11/15, 11/15/24, 11/15/2024
_EXPIRY_RE = re.compile(
    r"(?<![\d])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\d])"
)

# 0DTE / weekly / monthly fallback
_WEEKLY_HINT  = re.compile(r"\b(?:0dte|weekly|weeklies)\b", re.IGNORECASE)
_MONTHLY_HINT = re.compile(r"\b(?:monthly|monthlies|next month)\b", re.IGNORECASE)

# Reasonable ticker validation — exclude common false positives
_BLACKLIST_TICKERS = {
    "USD", "GDP", "CPI", "PPI", "FOMC", "NFP", "ECB", "BOJ", "SEC",
    "IPO", "EPS", "OTM", "ITM", "ATM", "P/E", "P/L", "PT", "TA",
    "AI", "ML", "DD", "YOLO", "BTW", "FOMO", "WSB", "ETF", "RH",
    "API", "URL", "PDF", "HTML", "JSON", "FYI", "TLDR",
    # Common letter combinations that match \$?[A-Z]+ but aren't tickers
    "THE", "AND", "FOR", "BUT", "YOU", "ARE", "WAS", "WILL", "BE", "OF",
}


def parse_callout(text: str) -> Optional[dict]:
    """Extract a structured options callout from free text.

    Returns dict with keys:
        ticker, option_type ('call'/'put'), strike, expiry, raw_text
    or None if no callout pattern matches.
    """
    if not text or len(text) < 5:
        return None
    # Must contain at least one options keyword OR a $TICKER + number + c/p
    if not _OPTIONS_KEYWORDS.search(text):
        # Allow the c/p shorthand format even without keyword
        if not (re.search(r"\$[A-Z]+\s+\$?\d+(?:\.\d+)?\s*[cp]\b", text, re.I)):
            return None

    # Try call patterns first
    for pat in _CALL_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        ticker = m.group(1).upper()
        if ticker in _BLACKLIST_TICKERS or len(ticker) < 1:
            continue
        # Strike may not be present
        strike = None
        try:
            if len(m.groups()) >= 2 and m.group(2):
                strike = float(m.group(2))
        except (ValueError, IndexError):
            pass
        expiry = _extract_expiry(text)
        return {
            "ticker":      ticker,
            "option_type": "call",
            "strike":      strike,
            "expiry":      expiry,
            "raw_text":    text[:300],
        }

    # Then puts
    for pat in _PUT_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        ticker = m.group(1).upper()
        if ticker in _BLACKLIST_TICKERS:
            continue
        strike = None
        try:
            if len(m.groups()) >= 2 and m.group(2):
                strike = float(m.group(2))
        except (ValueError, IndexError):
            pass
        expiry = _extract_expiry(text)
        return {
            "ticker":      ticker,
            "option_type": "put",
            "strike":      strike,
            "expiry":      expiry,
            "raw_text":    text[:300],
        }

    return None


def _extract_expiry(text: str) -> Optional[str]:
    """Parse 11/15 or 11/15/24 or weekly/monthly hints into ISO date string."""
    m = _EXPIRY_RE.search(text)
    if m:
        try:
            mo  = int(m.group(1))
            dy  = int(m.group(2))
            yr  = m.group(3)
            if yr:
                yr = int(yr)
                if yr < 100:
                    yr = 2000 + yr
            else:
                yr = datetime.utcnow().year
                # If the date already passed this year, assume next year
                try:
                    candidate = datetime(yr, mo, dy)
                    if candidate < datetime.utcnow() - timedelta(days=3):
                        yr += 1
                except ValueError:
                    pass
            return f"{yr:04d}-{mo:02d}-{dy:02d}"
        except (ValueError, IndexError):
            pass
    # Weekly hint → Friday this week or next
    if _WEEKLY_HINT.search(text):
        today = datetime.utcnow()
        days_to_friday = (4 - today.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        return (today + timedelta(days=days_to_friday)).strftime("%Y-%m-%d")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CALLOUT TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class OptionsCalloutTracker:
    """Ingests + stores + tracks options callouts from social platforms."""

    _INIT_SQL = """
CREATE TABLE IF NOT EXISTS paper_options_callouts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at     TIMESTAMP,
    source          TEXT,
    username        TEXT,
    ticker          TEXT,
    option_type     TEXT,
    strike          REAL,
    expiry          DATE,
    entry_premium   REAL,
    last_premium    REAL,
    last_checked_at TIMESTAMP,
    pnl_pct         REAL,
    pnl_dollars     REAL,
    status          TEXT DEFAULT 'OPEN',
    outcome         TEXT,
    raw_text        TEXT,
    dedupe_key      TEXT
);
CREATE INDEX IF NOT EXISTS idx_callouts_ticker ON paper_options_callouts(ticker);
CREATE INDEX IF NOT EXISTS idx_callouts_status ON paper_options_callouts(status);
CREATE INDEX IF NOT EXISTS idx_callouts_dedupe ON paper_options_callouts(dedupe_key);
CREATE INDEX IF NOT EXISTS idx_callouts_user   ON paper_options_callouts(username);
"""

    def __init__(self, conn):
        self.conn = conn
        try:
            self.conn.executescript(self._INIT_SQL)
        except Exception:
            pass

    # ── Source A: StockTwits public stream ────────────────────────────────────

    def fetch_stocktwits(self, tickers: list = None,
                          max_per_ticker: int = 30) -> list:
        """Pull recent messages mentioning options for each ticker.

        Default tickers list = top liquid tickers most likely to have callouts.
        """
        if requests is None:
            return []
        tickers = tickers or [
            "SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META",
            "AMZN", "GOOG", "PLTR", "AMC", "GME", "SOFI", "HIMS",
        ]
        collected = []
        for tk in tickers:
            try:
                r = requests.get(
                    f"https://api.stocktwits.com/api/2/streams/symbol/{tk.upper()}.json",
                    headers={"User-Agent": "Mozilla/5.0 BreakoutBot/1.0"},
                    timeout=8,
                )
                if r.status_code != 200:
                    continue
                msgs = (r.json() or {}).get("messages", []) or []
                for m in msgs[:max_per_ticker]:
                    body = str(m.get("body", "") or "")
                    user = (m.get("user") or {}).get("username", "")
                    created = m.get("created_at", "")
                    callout = parse_callout(body)
                    if callout:
                        callout.update({
                            "source":      "stocktwits",
                            "username":    user,
                            "created_at":  str(created),
                            "raw_text":    body[:300],
                        })
                        collected.append(callout)
            except Exception:
                continue
        return collected

    # ── Source B: Reddit (optional, requires PRAW + creds) ───────────────────

    def fetch_reddit(self, max_per_sub: int = 50) -> list:
        """Pull recent posts from options-focused subreddits.

        Requires environment variables (or Streamlit secrets):
            REDDIT_CLIENT_ID
            REDDIT_CLIENT_SECRET
            REDDIT_USER_AGENT  (e.g. 'BreakoutBot/1.0 by yourusername')
        Returns empty list if unconfigured.
        """
        cid    = os.environ.get("REDDIT_CLIENT_ID",     "").strip()
        csec   = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
        ua     = os.environ.get("REDDIT_USER_AGENT",    "BreakoutBot/1.0").strip()
        if not cid or not csec:
            return []
        try:
            import praw
        except ImportError:
            return []

        collected = []
        try:
            reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua,
                                   check_for_async=False)
            for sub_name in ("options", "wallstreetbets", "thetagang", "smallstreetbets"):
                try:
                    sub = reddit.subreddit(sub_name)
                    for post in sub.new(limit=max_per_sub):
                        title = post.title or ""
                        body  = post.selftext or ""
                        text  = f"{title}\n{body}"
                        callout = parse_callout(text)
                        if callout:
                            callout.update({
                                "source":      f"reddit/{sub_name}",
                                "username":    str(post.author) if post.author else "[deleted]",
                                "created_at":  datetime.utcfromtimestamp(
                                                    post.created_utc).isoformat(),
                                "raw_text":    text[:300],
                            })
                            collected.append(callout)
                except Exception:
                    continue
        except Exception:
            return []
        return collected

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _dedupe_key(self, c: dict) -> str:
        """Stable hash so we don't re-store the same callout twice."""
        import hashlib
        s = (f"{c.get('source','')}-{c.get('username','')}-"
             f"{c.get('ticker','')}-{c.get('option_type','')}-"
             f"{c.get('strike','')}-{c.get('expiry','')}-"
             f"{str(c.get('raw_text',''))[:60]}")
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

    def _exists(self, dedupe_key: str) -> bool:
        try:
            row = self.conn.execute(
                "SELECT 1 FROM paper_options_callouts WHERE dedupe_key=? LIMIT 1",
                (dedupe_key,)
            ).fetchone()
            return row is not None
        except Exception:
            return False

    def _snapshot_premium(self, ticker: str, opt_type: str,
                            strike: Optional[float], expiry: Optional[str]) -> Optional[float]:
        """Look up the current mid-premium of the called-out option via yfinance."""
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            exps = list(tk.options or [])
            if not exps:
                return None

            # Find best expiry match
            chosen_expiry = None
            if expiry:
                if expiry in exps:
                    chosen_expiry = expiry
                else:
                    # Closest expiry on or after target
                    try:
                        target = datetime.strptime(expiry, "%Y-%m-%d").date()
                        for e in exps:
                            ed = datetime.strptime(e, "%Y-%m-%d").date()
                            if ed >= target:
                                chosen_expiry = e
                                break
                    except Exception:
                        pass
            if not chosen_expiry:
                chosen_expiry = exps[0]   # nearest

            chain = tk.option_chain(chosen_expiry)
            df = chain.calls if opt_type == "call" else chain.puts
            if df is None or df.empty:
                return None

            if strike:
                # Find nearest strike
                idx = (df["strike"] - strike).abs().idxmin()
                row = df.loc[idx]
            else:
                # ATM (use current price)
                spot = float(tk.fast_info.last_price or 0)
                if spot <= 0:
                    return None
                idx = (df["strike"] - spot).abs().idxmin()
                row = df.loc[idx]

            bid = float(row.get("bid") or 0)
            ask = float(row.get("ask") or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(row.get("lastPrice") or 0)
            return round(mid, 4) if mid > 0 else None
        except Exception:
            return None

    # ── Ingestion (idempotent) ───────────────────────────────────────────────

    def ingest_callouts(self, tickers: list = None,
                         include_reddit: bool = True) -> dict:
        """Pull from all sources, parse, snapshot premium, store new callouts."""
        new_callouts = []
        st_raw = self.fetch_stocktwits(tickers=tickers)
        for c in st_raw:
            new_callouts.append(c)
        if include_reddit:
            rd_raw = self.fetch_reddit()
            for c in rd_raw:
                new_callouts.append(c)

        stored = 0
        skipped_existing = 0
        skipped_no_premium = 0
        for c in new_callouts:
            c["dedupe_key"] = self._dedupe_key(c)
            if self._exists(c["dedupe_key"]):
                skipped_existing += 1
                continue
            # Snapshot current premium
            prem = self._snapshot_premium(
                c["ticker"], c["option_type"],
                c.get("strike"), c.get("expiry"),
            )
            if prem is None or prem <= 0:
                skipped_no_premium += 1
                continue
            try:
                self.conn.execute(
                    "INSERT INTO paper_options_callouts "
                    "(detected_at, source, username, ticker, option_type, "
                    "strike, expiry, entry_premium, last_premium, "
                    "last_checked_at, pnl_pct, pnl_dollars, status, "
                    "raw_text, dedupe_key) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (datetime.utcnow().isoformat(),
                     c.get("source", ""), c.get("username", ""),
                     c["ticker"], c["option_type"],
                     c.get("strike"), c.get("expiry"),
                     prem, prem,
                     datetime.utcnow().isoformat(),
                     0.0, 0.0,
                     "OPEN",
                     c.get("raw_text", ""),
                     c["dedupe_key"]),
                )
                stored += 1
            except Exception:
                pass
        return {
            "fetched":            len(new_callouts),
            "new_stored":         stored,
            "skipped_duplicate":  skipped_existing,
            "skipped_no_premium": skipped_no_premium,
        }

    # ── Outcome tracking ─────────────────────────────────────────────────────

    def track_outcomes(self, max_per_cycle: int = 50) -> dict:
        """Update last_premium + pnl on open callouts.  Mark expired ones."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_options_callouts "
                "WHERE status='OPEN' "
                "ORDER BY last_checked_at ASC LIMIT ?",
                (int(max_per_cycle),)
            ).fetchall()
        except Exception:
            return {"updated": 0, "expired": 0}

        updated = 0
        expired = 0
        today = datetime.utcnow().date()
        for r in rows or []:
            try:
                d = dict(r)
            except Exception:
                continue
            ticker = d.get("ticker", "")
            opt_t  = d.get("option_type", "call")
            strike = d.get("strike")
            expiry = d.get("expiry")
            entry  = float(d.get("entry_premium") or 0)
            cid    = int(d.get("id") or 0)
            if entry <= 0:
                continue

            # Expired?
            if expiry:
                try:
                    ed = datetime.strptime(str(expiry)[:10], "%Y-%m-%d").date()
                    if ed < today:
                        # Final outcome = the LAST known premium snapshot
                        last = float(d.get("last_premium", entry) or entry)
                        pnl_pct = (last - entry) / entry * 100 if entry else 0
                        outcome = ("WIN" if pnl_pct > 5 else
                                    "LOSS" if pnl_pct < -5 else "BREAKEVEN")
                        try:
                            self.conn.execute(
                                "UPDATE paper_options_callouts "
                                "SET status='CLOSED', outcome=?, pnl_pct=?, "
                                "last_checked_at=? WHERE id=?",
                                (outcome, pnl_pct, datetime.utcnow().isoformat(), cid)
                            )
                            expired += 1
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass

            # Still open — fetch live premium
            last = self._snapshot_premium(ticker, opt_t, strike, expiry)
            if last is None:
                continue
            pnl_pct = (last - entry) / entry * 100
            pnl_dollars = (last - entry) * 100   # 1 contract
            try:
                self.conn.execute(
                    "UPDATE paper_options_callouts "
                    "SET last_premium=?, last_checked_at=?, pnl_pct=?, "
                    "pnl_dollars=? WHERE id=?",
                    (last, datetime.utcnow().isoformat(),
                     pnl_pct, pnl_dollars, cid),
                )
                updated += 1
            except Exception:
                pass

        return {"updated": updated, "expired": expired}

    # ── Read APIs ────────────────────────────────────────────────────────────

    def get_active_callouts(self, limit: int = 50, min_age_minutes: int = 0) -> list:
        """Currently OPEN callouts, sorted by P&L descending."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_options_callouts "
                "WHERE status='OPEN' "
                "ORDER BY pnl_pct DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
            return [dict(r) for r in rows or []]
        except Exception:
            return []

    def get_recent_winners(self, hours: int = 48, limit: int = 20) -> list:
        """Best-performing open callouts in last N hours."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_options_callouts "
                "WHERE detected_at >= ? AND status='OPEN' AND pnl_pct > 10 "
                "ORDER BY pnl_pct DESC LIMIT ?",
                (cutoff, int(limit))
            ).fetchall()
            return [dict(r) for r in rows or []]
        except Exception:
            return []

    def get_leaderboard(self, min_callouts: int = 3) -> list:
        """Rank callers by win rate + average return on CLOSED callouts."""
        try:
            rows = self.conn.execute(
                "SELECT username, source, "
                "COUNT(*) as n_total, "
                "SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as n_wins, "
                "SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) as n_losses, "
                "AVG(pnl_pct) as avg_pnl_pct, "
                "MAX(pnl_pct) as best_pnl_pct, "
                "MIN(pnl_pct) as worst_pnl_pct "
                "FROM paper_options_callouts "
                "WHERE status='CLOSED' "
                "GROUP BY username, source "
                "HAVING n_total >= ? "
                "ORDER BY (n_wins*100.0/n_total) DESC, avg_pnl_pct DESC",
                (int(min_callouts),)
            ).fetchall()
            out = []
            for r in rows or []:
                d = dict(r)
                d["win_rate_pct"] = round(d["n_wins"] / d["n_total"] * 100, 1) if d["n_total"] else 0
                out.append(d)
            return out
        except Exception:
            return []

    def get_stats(self) -> dict:
        try:
            n_total = int(self.conn.execute(
                "SELECT COUNT(*) FROM paper_options_callouts").fetchone()[0] or 0)
            n_open  = int(self.conn.execute(
                "SELECT COUNT(*) FROM paper_options_callouts WHERE status='OPEN'").fetchone()[0] or 0)
            n_wins  = int(self.conn.execute(
                "SELECT COUNT(*) FROM paper_options_callouts WHERE outcome='WIN'").fetchone()[0] or 0)
            n_loss  = int(self.conn.execute(
                "SELECT COUNT(*) FROM paper_options_callouts WHERE outcome='LOSS'").fetchone()[0] or 0)
            n_close = n_wins + n_loss
            return {
                "n_total":          n_total,
                "n_open":           n_open,
                "n_closed":         n_close,
                "n_wins":           n_wins,
                "n_losses":         n_loss,
                "overall_win_rate": round(n_wins / n_close * 100, 1) if n_close else 0.0,
            }
        except Exception:
            return {"n_total": 0}
