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

from datetime import datetime, timezone, date, timedelta


# ── Defaults ──────────────────────────────────────────────────────────────────
MIN_QUALITY_SHOW   = 50      # what shows in the dashboard
MIN_QUALITY_ALERT  = 70      # what fires a Telegram alert
MAX_RESULTS        = 12
PER_SOURCE_LIMIT   = {       # per-source candidate caps (controls API load)
    "momentum": 15,
    "pead":     10,
    "whale":    10,
    "vip":      10,
    "mover":    10,
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

    # ── candidate gathering — pull from every signal source ────────────────────
    def gather_candidates(self) -> dict[str, dict]:
        """
        Returns { ticker: { "sources": set, "price_hint": float|None,
                            "mom_6m": float|None, "mom_3m": float|None } }
        """
        uni: dict[str, dict] = {}

        def add(tk, source, price=None, mom_6m=None, mom_3m=None):
            tk = str(tk or "").upper().strip()
            if not tk or "." in tk:
                return
            e = uni.setdefault(tk, {"sources": set(), "price_hint": None,
                                    "mom_6m": None, "mom_3m": None})
            e["sources"].add(source)
            if price and (e["price_hint"] is None):
                e["price_hint"] = float(price)
            if mom_6m is not None and e["mom_6m"] is None:
                e["mom_6m"] = float(mom_6m)
            if mom_3m is not None and e["mom_3m"] is None:
                e["mom_3m"] = float(mom_3m)

        # 1. Momentum leaders (PRIMARY — validated edge)
        try:
            from momentum_strategy import MomentumStrategy
            for r in MomentumStrategy(self.conn).rank(
                    top_n=PER_SOURCE_LIMIT["momentum"], min_mom_6m=0.05):
                add(r["ticker"], "momentum",
                    price=r.get("price"),
                    mom_6m=r.get("mom_6m"),
                    mom_3m=r.get("mom_3m"))
        except Exception as e:
            print(f"  [opt-scan] momentum source failed: {e}")

        # 2. PEAD candidates
        try:
            from earnings_engine import PEADScanner
            for r in PEADScanner(self.conn).get_latest()[:PER_SOURCE_LIMIT["pead"]]:
                add(r["ticker"], "pead", price=r.get("entry_now"))
        except Exception as e:
            print(f"  [opt-scan] pead source failed: {e}")

        # 3. Whale-watch picks
        try:
            from whale_watch import WhaleWatchlist
            for r in WhaleWatchlist(self.conn).get_latest()[:PER_SOURCE_LIMIT["whale"]]:
                add(r["ticker"], "whale", price=r.get("price_now"))
        except Exception as e:
            print(f"  [opt-scan] whale source failed: {e}")

        # 4. VIP-mentioned tickers (last 7d)
        try:
            cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
            rows = self.conn.execute(
                "SELECT tickers FROM vip_posts WHERE fetched_at >= ? "
                "AND tickers <> '' AND tickers IS NOT NULL",
                (cutoff,)).fetchall()
            for r in rows[:50]:
                t_csv = r.get("tickers") if hasattr(r, "get") else r[0]
                for tk in str(t_csv or "").split(","):
                    if tk:
                        add(tk, "vip")
                if sum(1 for v in uni.values() if "vip" in v["sources"]) >= PER_SOURCE_LIMIT["vip"]:
                    break
        except Exception as e:
            print(f"  [opt-scan] vip source failed: {e}")

        # 5. Recent unified-scanner movers
        try:
            from unified_scanner import get_latest_unified
            for r in get_latest_unified(self.conn, decisions=("BUY","WATCH"),
                                        limit=PER_SOURCE_LIMIT["mover"]):
                add(r["ticker"], "mover")
        except Exception as e:
            print(f"  [opt-scan] mover source failed: {e}")

        return uni

    # ── score a single candidate end-to-end ───────────────────────────────────
    def _score_candidate(self, tk: str, meta: dict) -> dict | None:
        """Pick the best call contract for `tk` and quality-score it."""
        try:
            from momentum_options import select_call_contract, _earnings_within, EARNINGS_AVOID_DAYS
            from options_analytics import options_trade_score
        except Exception:
            return None

        # Skip if earnings imminent (IV crush risk on long premium)
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

        c = select_call_contract(tk, price)
        if not c:
            return None

        c.update({
            "underlying_price": price,
            "option_type":      "call",
        })

        # Thesis = expected underlying move over the option's life. Uses the
        # 3-month-equivalent (mom_6m/3) of the strategy's measured momentum,
        # floored at 8% so we never surface options on theses too weak to pay
        # for the strike OTM + theta. A momentum stock putting up +60% over 6m
        # is genuinely expected to keep ~10%/mo, not 5%/mo (1/6 of cumulative).
        from momentum_options import MIN_THESIS_PCT
        mom6 = meta.get("mom_6m") or 0.0
        thesis = max(MIN_THESIS_PCT, abs(mom6) / 3.0 * 100) if mom6 else MIN_THESIS_PCT
        if thesis < MIN_THESIS_PCT:
            return None

        try:
            qs = options_trade_score(self.conn, tk, {**c, "iv": c.get("iv", 0)},
                                     thesis_move_pct=thesis)
        except Exception:
            return None

        return {
            "ticker":           tk,
            "contract_symbol":  c.get("contract_symbol", ""),
            "option_type":      "call",
            "strike":           c.get("strike"),
            "expiry":           c.get("expiry"),
            "dte":              c.get("dte"),
            "premium":          c.get("premium"),
            "iv_pct":           round(float(c.get("iv", 0) or 0) * 100, 1),
            "underlying_price": round(price, 2),
            "quality_score":    qs["score"],
            "quality_grade":    qs["grade"],
            "decision":         qs["decision"],
            "components":       qs["components"],
            "sources":          sorted(meta.get("sources", [])),
            "thesis_pct":       round(thesis, 2),
        }

    # ── full scan ──────────────────────────────────────────────────────────────
    def scan(self, min_quality: int = MIN_QUALITY_SHOW,
             max_results: int = MAX_RESULTS, progress=None) -> list:
        cands = self.gather_candidates()
        total = len(cands)
        results = []
        for i, (tk, meta) in enumerate(cands.items()):
            if progress:
                try:
                    progress(i + 1, total, tk)
                except Exception:
                    pass
            r = self._score_candidate(tk, meta)
            if r and r["quality_score"] >= min_quality:
                results.append(r)
        results.sort(key=lambda r: r["quality_score"], reverse=True)
        return results[:max_results]

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
                       progress=None) -> dict:
        """
        One-call cycle entrypoint: scan, persist, send Telegram for NEW setups
        meeting `alert_threshold`. Returns {scanned, results, alerts_sent}.
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

        results = self.scan(min_quality=min_quality, progress=progress)
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
        src = " · ".join(r["sources"])
        return (
            f"{grade_emoji} <b>OPTIONS SCAN — {r['quality_grade']} setup</b>\n"
            f"<b>{r['ticker']}</b>  ${r['strike']:.0f}{r['option_type'][0].upper()} "
            f"exp {r['expiry']} ({r['dte']}d)\n"
            f"Premium : <b>${r['premium']:.2f}</b>  "
            f"(1 contract = ${r['premium']*100:.0f})\n"
            f"Quality : <b>{r['quality_score']}/100</b>  →  {r['decision']}\n"
            f"IV : {r['iv_pct']:.0f}%  ·  Underlying: ${r['underlying_price']:.2f}\n"
            f"Why : <i>{src}</i>"
        )
