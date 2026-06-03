"""
whale_watch.py — "Follow the money" watchlist with concrete entry points.

WHAT THIS IS
------------
The legal analog to "insider info": stocks where smart money has *publicly* shown
their hand — Congressional disclosures (STOCK Act), SEC 13D/13G activist filings,
Form 4 insider open-market buying, federal contract awards. The information is
public, but retail rarely aggregates it, so there's a real (if modest) edge.

This module:
  1. Scans the liquid universe through WhaleIntelligence (Congressional + SEC +
     Gov contracts + StockTwits), keeps only names with a meaningful whale score.
  2. Computes a concrete ENTRY / STOP / TARGET for each (current price, ATR-based
     stop, target at recent high or +R:R).
  3. Persists the latest snapshot for the dashboard and records first-detection
     entry prices alongside SPY so we can build a forward "whale picks vs SPY"
     scorecard.

HONEST CAVEAT
-------------
Post-STOCK-Act academic studies are MIXED on whether following Pelosi-style
trades still beats the market. 13D activist filings and Form 4 insider buying
have firmer documented edges. The scorecard tells us the truth in real money.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None
try:
    import pandas as pd
    import numpy as np
except Exception:                       # pragma: no cover
    pd = None
    np = None

from whale_intelligence import WhaleIntelligence
from momentum_strategy import LIQUID_UNIVERSE


# ── Knobs ─────────────────────────────────────────────────────────────────────
MIN_WHALE_SCORE = 20      # ≥ first multiplier tier — anything real
MAX_WATCHLIST   = 30      # cap the list so the dashboard stays readable
ATR_STOP_MULT   = 1.5     # stop = entry - 1.5×ATR
ATR_TARGET_MULT = 3.0     # target = entry + 3×ATR  (R:R ≈ 2:1)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY-POINT CALCULATION
# ══════════════════════════════════════════════════════════════════════════════
def _ohlc_metrics(ticker: str) -> dict | None:
    """Pull recent OHLC and compute the bits we need for entry/stop/target."""
    if yf is None:
        return None
    try:
        raw = yf.download(ticker, period="6mo", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 30:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=["Close"])
    except Exception:
        return None

    try:
        live = float(yf.Ticker(ticker).fast_info["last_price"])
    except Exception:
        live = float(raw["Close"].iloc[-1])
    if not live or live <= 0:
        return None

    high, low, close = raw["High"], raw["Low"], raw["Close"]
    tr = pd.concat([(high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1] or (live * 0.02))

    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    sma50 = float(close.iloc[-50:].mean()) if len(close) >= 50 else ema20
    recent_high = float(close.iloc[-90:].max()) if len(close) >= 90 else float(close.max())

    return {
        "price": round(live, 2),
        "atr": round(atr, 3),
        "ema20": round(ema20, 2),
        "sma50": round(sma50, 2),
        "recent_high_90d": round(recent_high, 2),
    }


def _entry_plan(m: dict) -> dict:
    """Compute concrete entry / pullback / stop / target / R:R from price metrics."""
    price = m["price"]; atr = m["atr"]
    entry_now      = round(price, 2)
    entry_pullback = round(max(m["ema20"], m["sma50"], price - 1.5 * atr), 2)
    stop           = round(entry_now - ATR_STOP_MULT * atr, 2)
    # Target = whichever is larger: recent high (if above current), or ATR-target
    atr_target = entry_now + ATR_TARGET_MULT * atr
    target     = round(max(m["recent_high_90d"], atr_target), 2)
    risk   = max(entry_now - stop, 0.01)
    reward = max(target - entry_now, 0)
    return {
        "entry_now":      entry_now,
        "entry_pullback": entry_pullback,
        "stop":           stop,
        "target":         target,
        "risk_pct":       round(risk / entry_now * 100, 2),
        "reward_pct":     round(reward / entry_now * 100, 2),
        "rr":             round(reward / risk, 2) if risk > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════
class WhaleWatchlist:
    def __init__(self, conn, universe: list | None = None):
        self.conn = conn
        self.wi = WhaleIntelligence()
        self.universe = [t.upper() for t in (universe or LIQUID_UNIVERSE) if "." not in t]
        self._ensure_tables()

    def _ensure_tables(self):
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS whale_watchlist (
                    snapshot_at      TEXT,
                    ticker           TEXT,
                    whale_score      INTEGER,
                    raw_score        INTEGER,
                    key_signal       TEXT,
                    flags            TEXT,
                    price_now        REAL,
                    entry_now        REAL,
                    entry_pullback   REAL,
                    stop             REAL,
                    target           REAL,
                    risk_pct         REAL,
                    reward_pct       REAL,
                    rr               REAL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS whale_watch_outcomes (
                    ticker          TEXT PRIMARY KEY,
                    detected_at     TEXT,
                    entry_price     REAL,
                    spy_at_entry    REAL,
                    last_price      REAL,
                    spy_last        REAL,
                    ticker_ret_pct  REAL,
                    spy_ret_pct     REAL,
                    alpha_pct       REAL,
                    days_held       INTEGER,
                    last_checked    TEXT
                )
            """)
        except Exception as e:
            print(f"  [whale-watch] table init failed: {e}")

    # ── build the latest watchlist (slow: 4 API calls × N tickers) ─────────────
    def build(self, min_score: int = MIN_WHALE_SCORE,
              max_results: int = MAX_WATCHLIST, progress=None) -> list:
        rows = []
        total = len(self.universe)
        for i, t in enumerate(self.universe):
            if progress:
                try:
                    progress(i + 1, total, t)
                except Exception:
                    pass
            try:
                rep = self.wi.full_report(t)
            except Exception:
                continue
            score = int(rep.get("whale_score", 0) or 0)
            if score < min_score:
                continue

            m = _ohlc_metrics(t)
            if not m:
                continue
            plan = _entry_plan(m)

            # Single "key signal" line for the dashboard (first non-empty flag).
            flags = [f for f in (rep.get("flags") or []) if f]
            key_signal = flags[0] if flags else f"Whale score {score}/100"

            rows.append({
                "ticker":         t,
                "whale_score":    score,
                "raw_score":      int(rep.get("raw_score", 0) or 0),
                "key_signal":     key_signal,
                "flags":          flags,
                "price_now":      m["price"],
                **plan,
            })

        rows.sort(key=lambda r: r["whale_score"], reverse=True)
        return rows[:max_results]

    # ── persist latest snapshot + first-detection entry prices ─────────────────
    def persist(self, rows: list) -> int:
        if not rows:
            return 0
        ts = datetime.now(timezone.utc).isoformat()
        try:
            self.conn.execute("DELETE FROM whale_watchlist")
            for r in rows:
                self.conn.execute(
                    "INSERT INTO whale_watchlist "
                    "(snapshot_at, ticker, whale_score, raw_score, key_signal, "
                    " flags, price_now, entry_now, entry_pullback, stop, target, "
                    " risk_pct, reward_pct, rr) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ts, r["ticker"], r["whale_score"], r["raw_score"],
                     r["key_signal"], " | ".join(r.get("flags") or []),
                     r["price_now"], r["entry_now"], r["entry_pullback"],
                     r["stop"], r["target"], r["risk_pct"], r["reward_pct"], r["rr"])
                )
        except Exception as e:
            print(f"  [whale-watch] persist failed: {e}")
            return 0

        # Record first-detection entry alongside SPY for the forward scorecard.
        spy_now = self._live("SPY")
        for r in rows:
            try:
                existing = self.conn.execute(
                    "SELECT ticker FROM whale_watch_outcomes WHERE ticker=?",
                    (r["ticker"],)).fetchone()
                if not existing and spy_now:
                    self.conn.execute(
                        "INSERT INTO whale_watch_outcomes "
                        "(ticker, detected_at, entry_price, spy_at_entry, "
                        " last_price, spy_last, ticker_ret_pct, spy_ret_pct, "
                        " alpha_pct, days_held, last_checked) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (r["ticker"], ts, r["price_now"], spy_now,
                         r["price_now"], spy_now, 0.0, 0.0, 0.0, 0, ts))
            except Exception:
                continue
        return len(rows)

    # ── refresh outcome P&L for every previously-detected whale pick ───────────
    def update_outcomes(self) -> int:
        try:
            rows = self.conn.execute(
                "SELECT ticker, detected_at, entry_price, spy_at_entry "
                "FROM whale_watch_outcomes").fetchall()
        except Exception:
            return 0
        spy_now = self._live("SPY")
        if not spy_now:
            return 0
        ts = datetime.now(timezone.utc).isoformat()
        n = 0
        for r in rows:
            def g(k, idx): return r.get(k) if hasattr(r, "get") else r[idx]
            tk = str(g("ticker", 0) or "").upper()
            if not tk:
                continue
            entry  = float(g("entry_price", 2) or 0)
            spy_e  = float(g("spy_at_entry", 3) or 0)
            if entry <= 0 or spy_e <= 0:
                continue
            last = self._live(tk)
            if not last:
                continue
            tret = (last / entry - 1) * 100
            sret = (spy_now / spy_e - 1) * 100
            try:
                det_ts = str(g("detected_at", 1) or "")[:10]
                days = (datetime.utcnow().date() -
                        datetime.strptime(det_ts, "%Y-%m-%d").date()).days
            except Exception:
                days = 0
            try:
                self.conn.execute(
                    "UPDATE whale_watch_outcomes SET last_price=?, spy_last=?, "
                    "ticker_ret_pct=?, spy_ret_pct=?, alpha_pct=?, days_held=?, "
                    "last_checked=? WHERE ticker=?",
                    (last, spy_now, round(tret, 2), round(sret, 2),
                     round(tret - sret, 2), days, ts, tk))
                n += 1
            except Exception:
                continue
        return n

    # ── read helpers for the dashboard ─────────────────────────────────────────
    def get_latest(self) -> list:
        try:
            rows = self.conn.execute(
                "SELECT * FROM whale_watchlist ORDER BY whale_score DESC"
            ).fetchall()
        except Exception:
            return []
        out = []
        for r in rows:
            def g(k):
                return r.get(k) if hasattr(r, "get") else None
            out.append({
                "ticker":         g("ticker"),
                "whale_score":    int(g("whale_score") or 0),
                "key_signal":     g("key_signal") or "",
                "flags":          (g("flags") or "").split(" | "),
                "price_now":      float(g("price_now") or 0),
                "entry_now":      float(g("entry_now") or 0),
                "entry_pullback": float(g("entry_pullback") or 0),
                "stop":           float(g("stop") or 0),
                "target":         float(g("target") or 0),
                "risk_pct":       float(g("risk_pct") or 0),
                "reward_pct":     float(g("reward_pct") or 0),
                "rr":             float(g("rr") or 0),
            })
        return out

    def get_outcomes(self) -> dict:
        try:
            rows = self.conn.execute(
                "SELECT ticker, detected_at, entry_price, last_price, "
                "ticker_ret_pct, spy_ret_pct, alpha_pct, days_held "
                "FROM whale_watch_outcomes ORDER BY alpha_pct DESC").fetchall()
        except Exception:
            return {"picks": [], "summary": {}}
        picks = []
        for r in rows:
            def g(k, idx): return r.get(k) if hasattr(r, "get") else r[idx]
            picks.append({
                "ticker":      str(g("ticker", 0) or ""),
                "detected_at": str(g("detected_at", 1) or "")[:10],
                "entry":       float(g("entry_price", 2) or 0),
                "last":        float(g("last_price", 3) or 0),
                "ticker_ret":  float(g("ticker_ret_pct", 4) or 0),
                "spy_ret":     float(g("spy_ret_pct", 5) or 0),
                "alpha":       float(g("alpha_pct", 6) or 0),
                "days":        int(g("days_held", 7) or 0),
            })
        if not picks:
            return {"picks": [], "summary": {}}
        n = len(picks)
        wins = sum(1 for p in picks if p["alpha"] > 0)
        avg_alpha = sum(p["alpha"] for p in picks) / n
        avg_ticker = sum(p["ticker_ret"] for p in picks) / n
        avg_spy = sum(p["spy_ret"] for p in picks) / n
        return {
            "picks": picks,
            "summary": {
                "n_picks": n, "n_winners": wins,
                "win_rate": round(wins / n * 100, 1),
                "avg_pick_return": round(avg_ticker, 2),
                "avg_spy_return": round(avg_spy, 2),
                "avg_alpha": round(avg_alpha, 2),
            },
        }

    @staticmethod
    def _live(ticker: str) -> float | None:
        if yf is None:
            return None
        try:
            return float(yf.Ticker(ticker).fast_info["last_price"])
        except Exception:
            return None
