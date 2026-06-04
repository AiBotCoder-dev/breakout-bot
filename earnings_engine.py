"""
earnings_engine.py — PEAD scanner + earnings calendar + badges.

THE EDGE WE'RE ACTUALLY CHASING
-------------------------------
Post-Earnings Announcement Drift (PEAD) is one of the longest-documented anomalies
in academic finance: stocks that **beat earnings AND gap up >3%** tend to keep
drifting up for 2-12 weeks afterwards because the market under-reacts to good
news. Unlike trading INTO earnings (where IV crush destroys long-premium plays),
PEAD is a clean directional edge that compounds with our momentum strategy.

WHAT THIS MODULE DOES
---------------------
1. `EarningsCalendar` — caches per-ticker next earnings date + last-quarter beat
   data (estimate, reported, surprise %). Powers the "📅 earnings in 5d" badges.
2. `PEADScanner` — scans the liquid universe, scores each name on the PEAD
   criteria (beat magnitude × post-earnings gap × trend intact × time since),
   returns ranked candidates with entry/stop/target.
3. `earnings_badge(ticker)` — one-call helper used by every panel that lists
   tickers (Momentum Leaders, Momentum Calls, Whale Watch) so you SEE the
   earnings-risk context before you click — never get IV-crushed again.

PEAD CRITERIA (cumulative score, threshold 50/100)
--------------------------------------------------
  +30  Last quarter beat estimate by >5%
  +20  Last quarter beat estimate by >0%
  +30  Post-earnings day gap up >= 3%
  +20  Post-earnings day gap up >= 1%
  +20  Trend still intact (close > earnings-day close AND > 50-SMA)
  +X   Recency bonus (sweet spot: 7-21 days post-earnings)

We do NOT recommend trading INTO earnings. The badge surfaces the risk so you
can SKIP names with earnings in the next 7 days — same rule as momentum_options.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except Exception:                       # pragma: no cover
    yf = pd = np = None


# ══════════════════════════════════════════════════════════════════════════════
# CALENDAR — cached next earnings dates + last-quarter beat data
# ══════════════════════════════════════════════════════════════════════════════
class EarningsCalendar:
    def __init__(self, conn):
        self.conn = conn
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS earnings_calendar (
                    ticker             TEXT PRIMARY KEY,
                    next_earnings      TEXT,
                    last_earnings      TEXT,
                    last_eps_estimate  REAL,
                    last_eps_reported  REAL,
                    last_surprise_pct  REAL,
                    refreshed_at       TEXT
                )
            """)
        except Exception as e:
            print(f"  [earn] table init failed: {e}")

    # ── one-ticker fetch ───────────────────────────────────────────────────────
    @staticmethod
    def _fetch(ticker: str) -> dict | None:
        if yf is None:
            return None
        try:
            ed = yf.Ticker(ticker).earnings_dates
            if ed is None or ed.empty:
                return None
        except Exception:
            return None

        now_utc = pd.Timestamp.now(tz=ed.index.tz) if ed.index.tz else pd.Timestamp.utcnow()
        upcoming = ed[ed.index > now_utc]
        historic = ed[ed.index <= now_utc]

        next_e = None
        if not upcoming.empty:
            # Earliest upcoming
            next_e = str(upcoming.sort_index().index[0].date())

        last_e = last_est = last_rep = last_surp = None
        if not historic.empty:
            row = historic.sort_index().iloc[-1]   # most recent past
            last_e = str(historic.sort_index().index[-1].date())
            try: last_est  = float(row.get("EPS Estimate")) if pd.notna(row.get("EPS Estimate")) else None
            except Exception: last_est = None
            try: last_rep  = float(row.get("Reported EPS")) if pd.notna(row.get("Reported EPS")) else None
            except Exception: last_rep = None
            try: last_surp = float(row.get("Surprise(%)"))  if pd.notna(row.get("Surprise(%)"))  else None
            except Exception: last_surp = None

        return {
            "ticker": ticker.upper(),
            "next_earnings":     next_e,
            "last_earnings":     last_e,
            "last_eps_estimate": last_est,
            "last_eps_reported": last_rep,
            "last_surprise_pct": last_surp,
        }

    # ── refresh many (threaded) ────────────────────────────────────────────────
    def refresh(self, tickers: list, progress=None, max_workers: int = 6) -> int:
        if not tickers:
            return 0
        now_iso = datetime.now(timezone.utc).isoformat()
        n_ok = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self._fetch, t.upper()): t.upper() for t in tickers}
            for i, fut in enumerate(as_completed(futs), 1):
                t = futs[fut]
                if progress:
                    try:
                        progress(i, len(futs), t)
                    except Exception:
                        pass
                r = fut.result()
                if not r:
                    continue
                try:
                    self.conn.execute(
                        "INSERT INTO earnings_calendar "
                        "(ticker, next_earnings, last_earnings, last_eps_estimate, "
                        " last_eps_reported, last_surprise_pct, refreshed_at) "
                        "VALUES (?,?,?,?,?,?,?) "
                        "ON CONFLICT (ticker) DO UPDATE SET "
                        "next_earnings=excluded.next_earnings, "
                        "last_earnings=excluded.last_earnings, "
                        "last_eps_estimate=excluded.last_eps_estimate, "
                        "last_eps_reported=excluded.last_eps_reported, "
                        "last_surprise_pct=excluded.last_surprise_pct, "
                        "refreshed_at=excluded.refreshed_at",
                        (r["ticker"], r["next_earnings"], r["last_earnings"],
                         r["last_eps_estimate"], r["last_eps_reported"],
                         r["last_surprise_pct"], now_iso)
                    )
                    n_ok += 1
                except Exception:
                    continue
        return n_ok

    # ── reads ──────────────────────────────────────────────────────────────────
    def get(self, ticker: str) -> dict | None:
        try:
            r = self.conn.execute(
                "SELECT * FROM earnings_calendar WHERE ticker=?",
                (ticker.upper(),)).fetchone()
        except Exception:
            return None
        if not r:
            return None
        def g(k):
            return r.get(k) if hasattr(r, "get") else None
        return {
            "ticker":            g("ticker") or ticker.upper(),
            "next_earnings":     g("next_earnings"),
            "last_earnings":     g("last_earnings"),
            "last_eps_estimate": g("last_eps_estimate"),
            "last_eps_reported": g("last_eps_reported"),
            "last_surprise_pct": g("last_surprise_pct"),
            "refreshed_at":      g("refreshed_at"),
        }


# ══════════════════════════════════════════════════════════════════════════════
# BADGE — used by every panel to surface earnings-risk context
# ══════════════════════════════════════════════════════════════════════════════
def earnings_badge(conn, ticker: str) -> dict:
    """
    Returns a small dict that's safe to render on any panel.

    Keys:
      days_to_earnings:  int|None     (None when no upcoming date known)
      next_earnings:     str|None     ISO date
      risk_window:       "imminent" (<7d) | "near" (<21d) | "far" | None
      surprise_pct:      last quarter's beat % (None if unknown)
      iv_pct:            ATM call IV %  (None if no chain)
      iv_level:          "low" / "normal" / "high" / "extreme"
    """
    cal = EarningsCalendar(conn).get(ticker)
    out = {"days_to_earnings": None, "next_earnings": None,
           "risk_window": None, "surprise_pct": None,
           "iv_pct": None, "iv_level": None}

    if cal:
        ne = cal.get("next_earnings")
        out["next_earnings"] = ne
        out["surprise_pct"]  = cal.get("last_surprise_pct")
        if ne:
            try:
                d = date.fromisoformat(ne[:10])
                dt = (d - date.today()).days
                out["days_to_earnings"] = dt
                out["risk_window"] = ("imminent" if dt is not None and 0 <= dt <= 7 else
                                      "near"     if dt is not None and 0 <= dt <= 21 else
                                      "far"      if dt is not None and dt > 21 else None)
            except Exception:
                pass

    # IV approximation — ATM call IV of the nearest expiry
    if yf is not None:
        try:
            tk = yf.Ticker(ticker)
            exps = list(tk.options or [])
            if exps:
                chain = tk.option_chain(exps[0]).calls
                if chain is not None and not chain.empty:
                    px = float(tk.fast_info["last_price"])
                    # Strike closest to underlying
                    chain = chain.assign(_d=(chain["strike"] - px).abs())
                    row = chain.sort_values("_d").iloc[0]
                    iv = float(row.get("impliedVolatility", 0) or 0)
                    if iv > 0:
                        out["iv_pct"] = round(iv * 100, 1)
                        out["iv_level"] = ("low"     if iv < 0.30 else
                                           "normal"  if iv < 0.60 else
                                           "high"    if iv < 1.00 else "extreme")
        except Exception:
            pass

    return out


# ══════════════════════════════════════════════════════════════════════════════
# PEAD SCANNER — the actual edge
# ══════════════════════════════════════════════════════════════════════════════
class PEADScanner:
    """
    Scans the liquid universe for Post-Earnings Announcement Drift setups:
      • Beat estimate (preferably > 5%)
      • Post-earnings day gap up >= 1% (preferably >= 3%)
      • Trend still intact (close > earnings-day close AND > 50-SMA)
      • Days since earnings in the drift sweet spot (7-21 days bonus)
    """

    def __init__(self, conn, calendar: EarningsCalendar | None = None):
        self.conn = conn
        self.cal = calendar or EarningsCalendar(conn)
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pead_candidates (
                    ticker         TEXT PRIMARY KEY,
                    scored_at      TEXT,
                    earnings_date  TEXT,
                    days_since     INTEGER,
                    surprise_pct   REAL,
                    gap_pct        REAL,
                    trend_intact   INTEGER,
                    score          INTEGER,
                    entry_now      REAL,
                    stop           REAL,
                    target         REAL,
                    summary        TEXT
                )
            """)
        except Exception as e:
            print(f"  [pead] table init failed: {e}")

    @staticmethod
    def _score_ticker(ticker: str, cal_row: dict) -> dict | None:
        """Evaluate one ticker. Returns candidate dict or None if it doesn't qualify."""
        if yf is None or not cal_row:
            return None
        last_e = cal_row.get("last_earnings")
        surp   = cal_row.get("last_surprise_pct")
        if not last_e or surp is None:
            return None

        try:
            ed = date.fromisoformat(last_e[:10])
        except Exception:
            return None
        days_since = (date.today() - ed).days
        # PEAD window: 1-60 days post-earnings; sweet spot 7-21
        if days_since < 1 or days_since > 60:
            return None
        # Only positive surprises qualify
        if surp <= 0:
            return None

        # Pull recent OHLC bracketing the earnings date
        try:
            start = ed - timedelta(days=10)
            end   = date.today() + timedelta(days=1)
            raw = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna(subset=["Close"])
            if raw.empty:
                return None
        except Exception:
            return None

        # Earnings-day close (or nearest trading day after)
        dates = [d.date() for d in raw.index]
        try:
            i_e = next(i for i, d in enumerate(dates) if d >= ed)
        except StopIteration:
            return None
        if i_e == 0:
            return None
        close_e   = float(raw["Close"].iloc[i_e])
        close_pre = float(raw["Close"].iloc[i_e - 1])
        gap_pct   = (close_e / close_pre - 1) * 100 if close_pre else 0.0

        # Trend intact?
        close_now = float(raw["Close"].iloc[-1])
        # 50-SMA approx (use whatever we have, fall back gracefully)
        try:
            extra = yf.download(ticker, period="6mo", interval="1d",
                                progress=False, auto_adjust=True)
            if isinstance(extra.columns, pd.MultiIndex):
                extra.columns = extra.columns.get_level_values(0)
            extra = extra.dropna(subset=["Close"])
            sma50 = float(extra["Close"].iloc[-50:].mean())
        except Exception:
            sma50 = close_now * 0.95
        trend_intact = (close_now >= close_e * 0.97) and (close_now > sma50)

        # ── PEAD score ────────────────────────────────────────────────────────
        score = 0
        if surp >= 5:     score += 30
        elif surp > 0:    score += 20
        if gap_pct >= 3:  score += 30
        elif gap_pct >= 1: score += 20
        if trend_intact:  score += 20
        # Recency bonus: peak in the 7-21 day window
        if 7 <= days_since <= 21: score += 10
        elif 22 <= days_since <= 40: score += 5

        if score < 50:
            return None

        # Entry plan
        atr_proxy = max(close_now * 0.02, abs(close_e - close_pre))
        stop   = round(max(sma50 * 0.99, close_now - 2 * atr_proxy, close_now * 0.92), 2)
        stop   = min(stop, close_now * 0.97)
        target = round(close_now * 1.20, 2)

        summary = (f"Beat {surp:+.1f}% · gapped {gap_pct:+.1f}% · "
                   f"{days_since}d post-earn · "
                   f"{'trend intact' if trend_intact else 'trend wobbly'}")

        return {
            "ticker":        ticker.upper(),
            "earnings_date": ed.isoformat(),
            "days_since":    days_since,
            "surprise_pct":  round(surp, 2),
            "gap_pct":       round(gap_pct, 2),
            "trend_intact":  int(trend_intact),
            "score":         int(score),
            "entry_now":     round(close_now, 2),
            "stop":          stop,
            "target":        target,
            "summary":       summary,
        }

    def scan(self, universe: list | None = None, progress=None,
             max_workers: int = 6) -> list:
        from momentum_strategy import LIQUID_UNIVERSE
        uni = [t.upper() for t in (universe or LIQUID_UNIVERSE) if "." not in t]
        # Refresh calendar for every name first (so we have last_earnings / surprise)
        self.cal.refresh(uni, progress=progress)

        # Then score in parallel using cached calendar rows
        rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {}
            for t in uni:
                cal_row = self.cal.get(t)
                if cal_row and cal_row.get("last_earnings"):
                    futs[ex.submit(self._score_ticker, t, cal_row)] = t
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    rows.append(r)

        rows.sort(key=lambda r: r["score"], reverse=True)
        self._persist(rows)
        return rows

    def _persist(self, rows: list):
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            self.conn.execute("DELETE FROM pead_candidates")
            for r in rows:
                self.conn.execute(
                    "INSERT INTO pead_candidates "
                    "(ticker, scored_at, earnings_date, days_since, surprise_pct, "
                    " gap_pct, trend_intact, score, entry_now, stop, target, summary) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["ticker"], now_iso, r["earnings_date"], r["days_since"],
                     r["surprise_pct"], r["gap_pct"], r["trend_intact"],
                     r["score"], r["entry_now"], r["stop"], r["target"], r["summary"])
                )
        except Exception as e:
            print(f"  [pead] persist failed: {e}")

    def get_latest(self) -> list:
        try:
            rows = self.conn.execute(
                "SELECT * FROM pead_candidates ORDER BY score DESC"
            ).fetchall()
        except Exception:
            return []
        out = []
        for r in rows:
            def g(k): return r.get(k) if hasattr(r, "get") else None
            out.append({
                "ticker":        g("ticker") or "",
                "earnings_date": g("earnings_date") or "",
                "days_since":    int(g("days_since") or 0),
                "surprise_pct":  float(g("surprise_pct") or 0),
                "gap_pct":       float(g("gap_pct") or 0),
                "trend_intact":  bool(g("trend_intact")),
                "score":         int(g("score") or 0),
                "entry_now":     float(g("entry_now") or 0),
                "stop":          float(g("stop") or 0),
                "target":        float(g("target") or 0),
                "summary":       g("summary") or "",
            })
        return out
