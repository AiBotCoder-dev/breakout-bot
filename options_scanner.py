"""
options_scanner.py — "Today's best options trades" meta-scanner.

WHAT THIS IS
------------
The single screen that answers: "given everything the bot knows right now,
what are the best options trades to take and how?" Pulls underlying candidates
from EVERY signal source we've built, picks the best contract per name, scores
each via the quality engine, and returns one ranked list.

Sources of underlying candidates (each tagged on the result so you see WHY):
  • momentum    — top 15 from MomentumStrategy.rank() (validated p=0.004 edge)
  • pead        — top 10 Post-Earnings Drift candidates (the safe earnings edge)
  • whale       — top 10 from whale_watchlist (Congressional / 13D / Form 4 / gov)
  • vip         — tickers mentioned by tracked VIPs (Trump / Fed) in last 7d
  • mover       — recent unified-scanner movers (catalyst gaps)

Contract pick + quality scoring uses the EXISTING engines (no duplication):
  • momentum_options.select_call_contract   for strike/expiry/premium
  • options_analytics.options_trade_score   for IVR + IV-RV + EM + Greeks + UOA

ALERTING
--------
`scan_and_alert(send_fn)` is the cycle-callable entrypoint:
  • Runs the full scan, persists every result (best_options_trades).
  • Sends a Telegram alert for NEW high-quality setups (score >= alert_threshold,
    deduped by contract_symbol so the same trade isn't re-alerted on every run).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, date, timedelta


# ── Defaults ──────────────────────────────────────────────────────────────────
MIN_QUALITY_SHOW   = 50      # what shows in the dashboard
MIN_QUALITY_ALERT  = 70      # what fires a Telegram alert
MAX_RESULTS        = 12
PER_SOURCE_LIMIT   = {       # per-source candidate caps (controls API load)
    "momentum":      15,
    "pead":          10,
    "whale":         10,
    "vip":           10,
    "mover":         10,
    "bottom_fisher":  6,     # validated oversold-at-support (p=0.005)  CALLS
    "reversal":       6,     # downtrend→base→uptrend TRIGGERED         CALLS
    "put_engine":     8,     # validated bearish edge                   PUTS
    "news_put":       6,     # news-reversal puts (validated)           PUTS
    "exhaustion_put": 4,     # exhausted-high reversal (p=0.013, rare)  PUTS
    "panic":          2,     # capitulation → index calls (100% 60d)    CALLS
}

# Per-source default direction + the validated-edge note shown to the user.
SOURCE_META = {
    "momentum":      ("call", "Cross-sectional momentum (MCPT p=0.004)"),
    "pead":          ("call", "Post-earnings drift (the safe earnings edge)"),
    "whale":         ("call", "Smart money — Congress / 13D / Form 4 / gov"),
    "vip":           ("call", "Tracked VIP (Trump/Fed) mention < 7d"),
    "mover":         ("call", "Unified-scanner catalyst mover"),
    "bottom_fisher": ("call", "Oversold-at-support bottom (MCPT p=0.005)"),
    "reversal":      ("call", "Downtrend→base→uptrend reclaim (62% @60d)"),
    "put_engine":    ("put",  "Validated bearish breakdown edge"),
    "news_put":      ("put",  "News-reversal put (validated)"),
    "exhaustion_put":("put",  "Exhausted-high reversal (60% win, p=0.013, rare)"),
    "panic":         ("call", "Capitulation rebound (100% win @60d, 12y)"),
}


class OptionsScanner:
    def __init__(self, conn):
        self.conn = conn
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS best_options_trades (
                    scanned_at        TEXT,
                    contract_symbol   TEXT PRIMARY KEY,
                    ticker            TEXT,
                    option_type       TEXT,
                    strike            REAL,
                    expiry            TEXT,
                    dte               INTEGER,
                    premium           REAL,
                    iv_pct            REAL,
                    underlying_price  REAL,
                    quality_score     INTEGER,
                    quality_grade     TEXT,
                    decision          TEXT,
                    sources           TEXT,
                    thesis_pct        REAL,
                    alerted           INTEGER DEFAULT 0
                )
            """)
        except Exception as e:
            print(f"  [opt-scan] table init failed: {e}")

    # ── candidate gathering — pull from EVERY validated tool on the site ───────
    def gather_candidates(self, fast: bool = False) -> dict[str, dict]:
        """
        The full market scan. Pulls option candidates (CALLS and PUTS) from every
        validated engine, keyed by "TICKER|DIRECTION" so a name can appear as both
        a call and a put idea from different tools.

        fast=True bounds the heaviest network sources (skips the 2y reversal
        download) so the live paper-trader cycle stays inside its time budget.
        The user-triggered dashboard scan uses fast=False for full coverage.

        Returns { "TICKER|dir": { ticker, direction, sources:set, price_hint,
                                  mom_6m, mom_3m, thesis_hint } }
        """
        uni: dict[str, dict] = {}

        def add(tk, source, direction=None, price=None, mom_6m=None,
                mom_3m=None, thesis=None):
            tk = str(tk or "").upper().strip()
            if not tk or "." in tk:
                return
            d = direction or SOURCE_META.get(source, ("call", ""))[0]
            key = f"{tk}|{d}"
            e = uni.setdefault(key, {"ticker": tk, "direction": d, "sources": set(),
                                     "price_hint": None, "mom_6m": None,
                                     "mom_3m": None, "thesis_hint": None})
            e["sources"].add(source)
            if price and (e["price_hint"] is None):
                e["price_hint"] = float(price)
            if mom_6m is not None and e["mom_6m"] is None:
                e["mom_6m"] = float(mom_6m)
            if mom_3m is not None and e["mom_3m"] is None:
                e["mom_3m"] = float(mom_3m)
            if thesis is not None and (e["thesis_hint"] is None or thesis > e["thesis_hint"]):
                e["thesis_hint"] = float(thesis)

        # 1. Momentum leaders (PRIMARY validated edge) — CALLS
        try:
            from momentum_strategy import MomentumStrategy
            for r in MomentumStrategy(self.conn).rank(
                    top_n=PER_SOURCE_LIMIT["momentum"], min_mom_6m=0.05):
                add(r["ticker"], "momentum", "call", price=r.get("price"),
                    mom_6m=r.get("mom_6m"), mom_3m=r.get("mom_3m"))
        except Exception as e:
            print(f"  [opt-scan] momentum source failed: {e}")

        # 2. PEAD candidates — CALLS — DISABLED by default (2026-07-22 review):
        #    live record 1/21 win (5%), -$811 — the worst setup in the book.
        #    Buying calls right after earnings walks into IV crush. Re-enable
        #    with PEAD_CALLS=on ONLY after it proves a forward edge.
        if os.environ.get("PEAD_CALLS", "off").strip().lower() == "on":
            try:
                from earnings_engine import PEADScanner
                for r in PEADScanner(self.conn).get_latest()[:PER_SOURCE_LIMIT["pead"]]:
                    add(r["ticker"], "pead", "call", price=r.get("entry_now"))
            except Exception as e:
                print(f"  [opt-scan] pead source failed: {e}")
        else:
            print("  [opt-scan] pead calls disabled (live 1/21, -$811 — IV crush)")

        # 3. Whale-watch picks — CALLS
        try:
            from whale_watch import WhaleWatchlist
            for r in WhaleWatchlist(self.conn).get_latest()[:PER_SOURCE_LIMIT["whale"]]:
                add(r["ticker"], "whale", "call", price=r.get("price_now"))
        except Exception as e:
            print(f"  [opt-scan] whale source failed: {e}")

        # 4. VIP-mentioned tickers (last 7d) — CALLS
        try:
            cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
            rows = self.conn.execute(
                "SELECT tickers FROM vip_posts WHERE fetched_at >= ? "
                "AND tickers <> '' AND tickers IS NOT NULL",
                (cutoff,)).fetchall()
            _vc = 0
            for r in rows[:50]:
                t_csv = r.get("tickers") if hasattr(r, "get") else r[0]
                for tk in str(t_csv or "").split(","):
                    if tk:
                        add(tk, "vip", "call"); _vc += 1
                if _vc >= PER_SOURCE_LIMIT["vip"]:
                    break
        except Exception as e:
            print(f"  [opt-scan] vip source failed: {e}")

        # 5. Recent unified-scanner movers — CALLS
        try:
            from unified_scanner import get_latest_unified
            for r in get_latest_unified(self.conn, decisions=("BUY","WATCH"),
                                        limit=PER_SOURCE_LIMIT["mover"]):
                add(r["ticker"], "mover", "call")
        except Exception as e:
            print(f"  [opt-scan] mover source failed: {e}")

        # 6. Bottom Fisher — oversold-at-support (validated p=0.005) — CALLS
        try:
            from bottom_fisher import BottomFisher
            _bf = [r for r in BottomFisher(self.conn).scan()
                   if r.get("stage") == "TRIGGERED"][:PER_SOURCE_LIMIT["bottom_fisher"]]
            for r in _bf:
                add(r["ticker"], "bottom_fisher", "call",
                    price=r.get("price"), thesis=9.0)
        except Exception as e:
            print(f"  [opt-scan] bottom_fisher source failed: {e}")

        # 7. Reversal finder — downtrend→base→uptrend TRIGGERED — CALLS
        #    (heavy 2y download; skipped in fast/live-trader mode)
        if not fast:
            try:
                from reversal_finder import ReversalFinder
                _rv = [r for r in ReversalFinder(self.conn).scan()
                       if r.get("stage") == "TRIGGERED"][:PER_SOURCE_LIMIT["reversal"]]
                for r in _rv:
                    add(r["ticker"], "reversal", "call",
                        price=r.get("price"), thesis=8.0)
            except Exception as e:
                print(f"  [opt-scan] reversal source failed: {e}")

        # 8. Put engine — validated bearish breakdown edge — PUTS
        #    Regime-gated: only when the market actually allows puts.
        try:
            from put_engine import bearish_rank, market_allows_puts
            if market_allows_puts().get("allowed", False):
                for r in bearish_rank(self.conn,
                                      top_n=PER_SOURCE_LIMIT["put_engine"],
                                      priority_only=True):
                    add(r["ticker"], "put_engine", "put",
                        price=r.get("price"), thesis=10.0)
            else:
                print("  [opt-scan] puts gated off (regime not bearish-friendly)")
        except Exception as e:
            print(f"  [opt-scan] put_engine source failed: {e}")

        # 9. News-reversal puts — validated, news-driven breakdowns — PUTS
        try:
            from news_reversal_puts import NewsReversalPuts
            for r in (NewsReversalPuts(self.conn).scan() or [])[:PER_SOURCE_LIMIT["news_put"]]:
                add(r.get("ticker"), "news_put", "put",
                    price=r.get("price"), thesis=10.0)
        except Exception as e:
            print(f"  [opt-scan] news_put source failed: {e}")

        # 9b. Exhaustion-reversal puts — exhausted-high breakdown (p=0.013, RARE).
        #     Heavy (2y download x40 names) but the setup persists for days, so we
        #     THROTTLE the underlying scan to once / ~90 min and cache the tickers
        #     — the live trader (fast) still sees it without paying the cost each
        #     cycle. Regime-gated like the other puts.
        try:
            from put_engine import market_allows_puts as _map
            if _map().get("allowed", False):
                self.conn.execute("CREATE TABLE IF NOT EXISTS exhaustion_cache "
                                  "(ticker TEXT, price REAL, scanned_at TEXT)")
                _row = self.conn.execute(
                    "SELECT MAX(scanned_at) FROM exhaustion_cache").fetchone()
                _last = _row[0] if _row else None
                _fresh = False
                if _last:
                    try:
                        _dt = datetime.fromisoformat(str(_last).replace("Z", "+00:00"))
                        _age = (datetime.now(_dt.tzinfo) - _dt).total_seconds()
                        _fresh = _age < 5400          # 90 min
                    except Exception:
                        pass
                if not _fresh:
                    from exhaustion_reversal import ExhaustionReversal
                    _ex = ExhaustionReversal(self.conn).scan()[:PER_SOURCE_LIMIT["exhaustion_put"]]
                    _now = datetime.now(timezone.utc).isoformat()
                    self.conn.execute("DELETE FROM exhaustion_cache")
                    for r in _ex:
                        self.conn.execute("INSERT INTO exhaustion_cache "
                                          "(ticker, price, scanned_at) VALUES (?,?,?)",
                                          (r["ticker"], r.get("price"), _now))
                # read from cache (fresh or just-written)
                for r in self.conn.execute(
                        "SELECT ticker, price FROM exhaustion_cache").fetchall():
                    _d = dict(r) if hasattr(r, "keys") else {"ticker": r[0], "price": r[1]}
                    add(_d["ticker"], "exhaustion_put", "put",
                        price=_d.get("price"), thesis=12.0)
        except Exception as e:
            print(f"  [opt-scan] exhaustion_put source failed: {e}")

        # 10. Panic detector — capitulation rebound → index CALLS (100% @60d)
        #     status() = {snap, signatures[]}; panic is "on" if any signature is
        #     currently firing or still active in the DB.
        try:
            from panic_detector import PanicDetector
            _ps = PanicDetector(self.conn).status()
            _sigs = (_ps or {}).get("signatures", []) or []
            _panic_on = any(s.get("currently_fired") or s.get("active_in_db")
                            for s in _sigs)
            if _panic_on:
                for tk in ("SPY", "QQQ")[:PER_SOURCE_LIMIT["panic"]]:
                    add(tk, "panic", "call", thesis=12.0)
        except Exception as e:
            print(f"  [opt-scan] panic source failed: {e}")

        return uni

    # ── score a single candidate end-to-end (direction-aware) ──────────────────
    def _score_candidate(self, meta: dict) -> dict | None:
        """Pick the best CALL or PUT contract for the candidate and score it."""
        try:
            from momentum_options import (select_call_contract, _earnings_within,
                                          EARNINGS_AVOID_DAYS, MIN_THESIS_PCT)
            from options_analytics import options_trade_score
        except Exception:
            return None

        tk        = meta.get("ticker")
        direction = meta.get("direction", "call")
        if not tk:
            return None

        # Skip if earnings imminent (IV crush risk on long premium, both sides)
        if _earnings_within(tk, EARNINGS_AVOID_DAYS):
            return None

        # Get underlying price
        price = meta.get("price_hint")
        if not price:
            try:
                import yfinance as yf
                price = float(yf.Ticker(tk).fast_info["last_price"])
            except Exception:
                return None
        if not price or price <= 0:
            return None

        # Direction-aware contract pick (reuse existing engines, no duplication)
        if direction == "put":
            try:
                from put_engine import select_put_contract
                c = select_put_contract(tk, price)
            except Exception:
                c = None
        else:
            c = select_call_contract(tk, price)
        if not c:
            return None
        c.update({"underlying_price": price, "option_type": direction})

        # Thesis = expected underlying move magnitude over the option's life.
        # Prefer an explicit per-source thesis hint; else derive from momentum;
        # floored at MIN_THESIS_PCT so thin moves (theta can't pay) are skipped.
        mom6   = meta.get("mom_6m") or 0.0
        thint  = meta.get("thesis_hint")
        thesis = (thint if thint else
                  (abs(mom6) / 3.0 * 100 if mom6 else MIN_THESIS_PCT))
        thesis = max(MIN_THESIS_PCT, thesis)

        try:
            qs = options_trade_score(self.conn, tk,
                                     {**c, "iv": c.get("iv", 0)},
                                     thesis_move_pct=thesis)
        except Exception:
            return None

        # ── RS/RVOL BONUS — momentum-type sources only ─────────────────────────
        # Backtested: RVOL>=1.3x + RS>0 lifts momentum-long win rate ~+2pp. It
        # HURTS dip setups, so applies_to() excludes bottom_fisher/reversal/puts.
        _score = qs["score"]
        _rsrv = None
        _srcs = meta.get("sources", [])
        try:
            import rs_rvol
            if rs_rvol.applies_to(_srcs):
                _rsrv = rs_rvol.compute(tk)
                _score = max(0, min(100, _score + _rsrv["bonus"]))
        except Exception:
            pass

        return {
            "ticker":           tk,
            "contract_symbol":  c.get("contract_symbol", ""),
            "option_type":      direction,
            "strike":           c.get("strike"),
            "expiry":           c.get("expiry"),
            "dte":              c.get("dte"),
            "premium":          c.get("premium"),
            "iv_pct":           round(float(c.get("iv", 0) or 0) * 100, 1),
            "underlying_price": round(price, 2),
            "quality_score":    _score,
            "base_score":       qs["score"],
            "rs_rvol":          (_rsrv["label"] if _rsrv else None),
            "rs_rvol_bonus":    (_rsrv["bonus"] if _rsrv else 0),
            "quality_grade":    qs["grade"],
            "decision":         qs["decision"],
            "components":       qs["components"],
            "sources":          sorted(_srcs),
            "thesis_pct":       round(thesis, 2),
        }

    # Source priority — validated edges first. Controls which candidates get the
    # expensive option-chain fetch when we cap scoring for the live cycle.
    _SOURCE_PRIORITY = {
        "momentum": 0, "bottom_fisher": 1, "exhaustion_put": 1, "put_engine": 2,
        "panic": 2, "pead": 3, "news_put": 3, "whale": 4, "reversal": 5,
        "vip": 6, "mover": 7,
    }

    # ── full scan ──────────────────────────────────────────────────────────────
    def scan(self, min_quality: int = MIN_QUALITY_SHOW,
             max_results: int = MAX_RESULTS, progress=None,
             fast: bool = False, max_candidates: int | None = None) -> list:
        cands = self.gather_candidates(fast=fast)
        # Each scored candidate triggers an option-chain fetch (~1-2s). Cap how
        # many we score so the live trader cycle stays inside its time budget,
        # prioritizing the validated-edge sources. Default: tight in fast mode.
        if max_candidates is None:
            max_candidates = 18 if fast else 45
        items = sorted(
            cands.items(),
            key=lambda kv: min((self._SOURCE_PRIORITY.get(s, 9)
                                for s in kv[1].get("sources", [])), default=9),
        )[:max_candidates]
        total = len(items)
        results = []
        for i, (key, meta) in enumerate(items):
            if progress:
                try:
                    progress(i + 1, total, meta.get("ticker", key))
                except Exception:
                    pass
            r = self._score_candidate(meta)
            if r and r["quality_score"] >= min_quality:
                results.append(r)
        results.sort(key=lambda r: r["quality_score"], reverse=True)
        return results[:max_results]

    # ── tradeable setups for the paper trader (broker-ready shape) ─────────────
    def tradeable_setups(self, min_quality: int = 55, max_results: int = 10,
                         fast: bool = True) -> list:
        """
        The list the autonomous paper-trader executes. Same as scan() but defaults
        to fast mode (bounded for the live cycle) and a stricter quality floor.
        Each item already carries ticker/option_type/strike/expiry/premium/
        contract_symbol so monitor's _attempt_entry can buy it directly.
        """
        return self.scan(min_quality=min_quality, max_results=max_results,
                         fast=fast)

    # ── persist (and remember what we've already alerted on) ───────────────────
    def persist(self, results: list):
        if not results:
            return
        ts = datetime.now(timezone.utc).isoformat()
        try:
            self.conn.execute("DELETE FROM best_options_trades")
            for r in results:
                self.conn.execute(
                    "INSERT INTO best_options_trades "
                    "(scanned_at, contract_symbol, ticker, option_type, strike, "
                    " expiry, dte, premium, iv_pct, underlying_price, "
                    " quality_score, quality_grade, decision, sources, "
                    " thesis_pct, alerted) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0) "
                    "ON CONFLICT (contract_symbol) DO UPDATE SET "
                    "scanned_at=excluded.scanned_at, "
                    "quality_score=excluded.quality_score, "
                    "decision=excluded.decision, "
                    "sources=excluded.sources",
                    (ts, r["contract_symbol"], r["ticker"], r["option_type"],
                     r["strike"], r["expiry"], r["dte"], r["premium"],
                     r["iv_pct"], r["underlying_price"], r["quality_score"],
                     r["quality_grade"], r["decision"],
                     ",".join(r["sources"]), r["thesis_pct"])
                )
        except Exception as e:
            print(f"  [opt-scan] persist failed: {e}")

    # ── read latest snapshot for the dashboard ────────────────────────────────
    def get_latest(self) -> list:
        try:
            rows = self.conn.execute(
                "SELECT * FROM best_options_trades ORDER BY quality_score DESC"
            ).fetchall()
        except Exception:
            return []
        out = []
        for r in rows:
            def g(k):
                return r.get(k) if hasattr(r, "get") else None
            out.append({
                "ticker":           g("ticker"),
                "contract_symbol":  g("contract_symbol"),
                "option_type":      g("option_type"),
                "strike":           float(g("strike") or 0),
                "expiry":           g("expiry") or "",
                "dte":              int(g("dte") or 0),
                "premium":          float(g("premium") or 0),
                "iv_pct":           float(g("iv_pct") or 0),
                "underlying_price": float(g("underlying_price") or 0),
                "quality_score":    int(g("quality_score") or 0),
                "quality_grade":    g("quality_grade") or "?",
                "decision":         g("decision") or "?",
                "sources":          [s for s in str(g("sources") or "").split(",") if s],
                "thesis_pct":       float(g("thesis_pct") or 0),
                "alerted":          bool(g("alerted")),
                "scanned_at":       g("scanned_at") or "",
            })
        return out

    # ── alerts ─────────────────────────────────────────────────────────────────
    def scan_and_alert(self, telegram_sender=None,
                       min_quality: int = MIN_QUALITY_SHOW,
                       alert_threshold: int = MIN_QUALITY_ALERT,
                       progress=None, fast: bool = False,
                       max_results: int = MAX_RESULTS) -> dict:
        """
        One-call cycle entrypoint: scan, persist, send Telegram for NEW setups
        meeting `alert_threshold`. Returns {scanned, results, alerts_sent}.
        The autonomous trader calls this with fast=True and reads `results`.
        """
        # Read prior contracts so we only alert on truly new ones
        try:
            prior_rows = self.conn.execute(
                "SELECT contract_symbol, alerted FROM best_options_trades"
            ).fetchall()
            prior_alerted = {
                (r.get("contract_symbol") if hasattr(r, "get") else r[0])
                for r in prior_rows
                if (r.get("alerted") if hasattr(r, "get") else r[1])
            }
        except Exception:
            prior_alerted = set()

        results = self.scan(min_quality=min_quality, max_results=max_results,
                             progress=progress, fast=fast)
        self.persist(results)

        alerts_sent = 0
        if telegram_sender:
            for r in results:
                if r["quality_score"] < alert_threshold:
                    continue
                if r["contract_symbol"] in prior_alerted:
                    continue
                msg = self._format_alert(r)
                try:
                    if telegram_sender(msg):
                        alerts_sent += 1
                        try:
                            self.conn.execute(
                                "UPDATE best_options_trades SET alerted=1 "
                                "WHERE contract_symbol=?", (r["contract_symbol"],))
                        except Exception:
                            pass
                except Exception:
                    continue
        return {"scanned": len(results), "results": results,
                "alerts_sent": alerts_sent}

    @staticmethod
    def _format_alert(r: dict) -> str:
        grade_emoji = {"A+": "🔥", "A": "⭐", "B": "✅"}.get(r["quality_grade"], "📊")
        is_put = r["option_type"] == "put"
        dir_tag = "📉 PUT" if is_put else "📈 CALL"
        # Translate source tags into the validated-edge note for the user
        edges = [SOURCE_META.get(s, ("", s))[1] or s for s in r["sources"]]
        why = " · ".join(dict.fromkeys(edges))     # dedupe, keep order
        return (
            f"{grade_emoji} <b>FULL-SCAN OPTIONS — {r['quality_grade']} {dir_tag}</b>\n"
            f"<b>{r['ticker']}</b>  ${r['strike']:.0f}{r['option_type'][0].upper()} "
            f"exp {r['expiry']} ({r['dte']}d)\n"
            f"Premium : <b>${r['premium']:.2f}</b>  "
            f"(1 contract = ${r['premium']*100:.0f})\n"
            f"Quality : <b>{r['quality_score']}/100</b>  →  {r['decision']}\n"
            f"IV : {r['iv_pct']:.0f}%  ·  Underlying: ${r['underlying_price']:.2f}\n"
            f"Edge : <i>{why}</i>"
        )
