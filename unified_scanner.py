"""
unified_scanner.py — ONE orchestrator that feeds every universe into the Master Score.

WHY THIS EXISTS
---------------
The bot accumulated ~12 fragmented signal engines (BreakoutScanner, SqueezeScanner,
EarlyMomentumScanner, NewsCatalystEngine, WhaleIntelligence, OptionsFlowScorer, ...).
But the only universe that ever reached a *trading decision* was the technical
breakout scan (the `calls` table) plus a fixed watchlist. So stocks that had ALREADY
exploded on a fresh catalyst — SNOW on earnings, RCAT/UMAC on drone-funding news —
were never even evaluated, because they were never in the universe.

KEY INSIGHT
-----------
`trading_scanner.compute_master_score()` already FUSES every engine into one 0-100
number (TV technicals × news × duration × squeeze × memory × whale × early-momentum)
and it works on ANY ticker — not just ones in the breakout scan. So we don't need to
merge scanner *code*. We need one front-end that gathers candidates from several
universes and runs them all through that single scoring brain, then persists one
ranked list that both monitor.py (auto-entry) and the dashboard share.

UNIVERSES (modes)
-----------------
  movers     — today's REAL top % gainers / unusual-volume / losers
               (Finviz signal screen, price>$1 & avgvol>500K, confirmed via yfinance)
  watchlist  — the always-on options/stock watchlist
  breakout   — latest technical breakout scan (calls table)
  smart      — movers + watchlist + breakout, deduped   (default; fast, cloud-safe)
  universal  — the FULL exchange universe (UniverseBuilder.build('all')); slow, manual

The junk floor is intentional: penny (<$1) and 100-share-volume pump-and-dumps are
filtered server-side by Finviz so the bot never wastes a Master-Score evaluation on
something it could never trade.
"""

from __future__ import annotations

import time
import uuid as _uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except Exception:                       # pragma: no cover
    requests = None
try:
    from bs4 import BeautifulSoup
except Exception:                       # pragma: no cover
    BeautifulSoup = None
try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None


# ══════════════════════════════════════════════════════════════════════════════
# TOP MOVERS / CATALYST FEED
# ══════════════════════════════════════════════════════════════════════════════
class TopMoversScanner:
    """
    Pulls the day's *actual* biggest movers — the names that show up on a
    TradingView '%change' screen — but pre-filtered to TRADEABLE liquidity.

    Source: Finviz signal screens (free, no auth). Filters applied server-side:
        sh_price_o1   → price > $1   (kills sub-dollar pennies)
        sh_avgvol_o500→ avg vol > 500K (kills 100-share pump-and-dumps)
    Each candidate is then confirmed via yfinance for live price / % change / volume.
    """

    _HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    # signal -> (url, default bias)
    _SIGNALS = {
        "gainers": ("https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
                    "&f=sh_price_o1,sh_avgvol_o500&ft=4", "bullish"),
        "unusual": ("https://finviz.com/screener.ashx?v=111&s=ta_unusualvolume"
                    "&f=sh_price_o1,sh_avgvol_o500&ft=4", "bullish"),
        "losers":  ("https://finviz.com/screener.ashx?v=111&s=ta_toplosers"
                    "&f=sh_price_o1,sh_avgvol_o500&ft=4", "bearish"),
    }

    def _finviz_tickers(self, url: str, max_pages: int = 2) -> list:
        if requests is None or BeautifulSoup is None:
            return []
        out, offset, pages = [], 1, 0
        try:
            while pages < max_pages:
                r = requests.get(url + f"&r={offset}", headers=self._HDRS, timeout=10)
                soup = BeautifulSoup(r.text, "lxml")
                tbl = soup.select_one("table.screener_table")
                # Finviz markup: ticker links are <a class="tab-link" href="stock?t=...">
                cells = tbl.select("a.tab-link") if tbl else []
                if not cells:
                    cells = soup.select("a.screener-link-primary")   # legacy fallback
                if not cells:
                    break
                batch = [c.text.strip().upper() for c in cells
                         if c.text.strip().isalpha() and 1 <= len(c.text.strip()) <= 5]
                out.extend(batch)
                if len(batch) < 20:
                    break
                offset += 20
                pages += 1
                time.sleep(0.3)
        except Exception:
            pass
        return out

    def _enrich(self, ticker: str) -> dict | None:
        """Confirm live price / % change / volume via yfinance fast_info."""
        if yf is None:
            return None
        price = prev = vol = None
        try:
            fi = yf.Ticker(ticker).fast_info
            price = float(fi["last_price"])
            prev  = float(fi["previous_close"])
            vol   = float(fi.get("last_volume", 0) or 0)
        except Exception:
            try:                                          # fallback: 2-day history
                raw = yf.download(ticker, period="2d", progress=False, auto_adjust=True)
                if raw is None or raw.empty:
                    return None
                closes = raw["Close"].dropna()
                price  = float(closes.iloc[-1])
                prev   = float(closes.iloc[-2]) if len(closes) > 1 else price
                vol    = float(raw["Volume"].iloc[-1]) if "Volume" in raw else 0
            except Exception:
                return None
        if not price or not prev or prev <= 0:
            return None
        pct = (price - prev) / prev * 100.0
        return {"price": round(price, 4), "pct_change": round(pct, 2), "volume": int(vol or 0)}

    def fetch(self, min_pct: float = 8.0, min_price: float = 1.0,
              max_results: int = 60, include_losers: bool = True,
              signals: list | None = None) -> list:
        """
        Returns a list of dicts sorted by absolute % move:
            {ticker, price, pct_change, volume, bias}
        """
        sigs = signals or (["gainers", "unusual"] + (["losers"] if include_losers else []))
        raw: dict[str, str] = {}
        for sig in sigs:
            url, bias = self._SIGNALS.get(sig, (None, "bullish"))
            if not url:
                continue
            for t in self._finviz_tickers(url):
                if t and t not in raw:
                    raw[t] = bias

        results = []
        if not raw:
            return results
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(self._enrich, t): (t, b) for t, b in raw.items()}
            for fut in as_completed(futs):
                t, base_bias = futs[fut]
                info = fut.result()
                if not info:
                    continue
                if info["price"] < min_price:
                    continue
                if abs(info["pct_change"]) < min_pct:
                    continue
                info["ticker"] = t
                info["bias"] = "bearish" if info["pct_change"] < 0 else base_bias
                results.append(info)

        results.sort(key=lambda x: abs(x["pct_change"]), reverse=True)
        return results[:max_results]


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
# Default cap on how many tickers get a (relatively expensive) Master-Score eval,
# per mode. Keeps the monitor cycle bounded and the universal scan finite.
_DEFAULT_LIMITS = {
    "movers":    40,
    "watchlist": 40,
    "breakout":  40,
    "smart":     120,
    "universal": 400,
}


class UnifiedScanner:
    """
    Gathers candidates from one or more universes, scores each with the single
    Master-Score brain, ranks them, and (optionally) persists the latest snapshot
    to `unified_scan_results` for the dashboard + auto-entry to consume.
    """

    def __init__(self, conn, market_regime=None, watchlist: list | None = None):
        self.conn = conn
        self.market_regime = market_regime
        self.watchlist = [str(t).upper() for t in (watchlist or [])]
        self._ensure_table()

    # ── schema ────────────────────────────────────────────────────────────────
    def _ensure_table(self):
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS unified_scan_results (
                    scan_id          TEXT,
                    scanned_at       TEXT,
                    mode             TEXT,
                    ticker           TEXT,
                    score            REAL,
                    grade            TEXT,
                    decision         TEXT,
                    size_multiplier  REAL,
                    bias             TEXT,
                    sources          TEXT,
                    pct_change       REAL,
                    summary          TEXT
                )
            """)
        except Exception as e:
            print(f"  [unified] table init failed: {e}")

    # ── universe gathering ──────────────────────────────────────────────────────
    def gather(self, mode: str = "smart", mover_min_pct: float = 8.0) -> dict:
        """Return {ticker: {"sources": set, "bias": str, "pct_change": float|None}}."""
        uni: dict[str, dict] = {}

        def add(t, source, bias="bullish", pct=None):
            t = str(t or "").upper().strip()
            if not t:
                return
            e = uni.setdefault(t, {"sources": set(), "bias": "bullish", "pct_change": None})
            e["sources"].add(source)
            if pct is not None:
                e["pct_change"] = pct
            if bias == "bearish":          # bearish wins (it's a warning flag)
                e["bias"] = "bearish"

        # --- movers ---
        if mode in ("movers", "smart"):
            try:
                for m in TopMoversScanner().fetch(min_pct=mover_min_pct):
                    add(m["ticker"], "mover", m["bias"], m["pct_change"])
            except Exception as e:
                print(f"  [unified] movers feed failed: {e}")

        # --- watchlist ---
        if mode in ("watchlist", "smart"):
            for t in self.watchlist:
                add(t, "watchlist")

        # --- latest breakout scan ---
        if mode in ("breakout", "smart"):
            try:
                row = self.conn.execute(
                    "SELECT scan_id FROM calls ORDER BY scan_timestamp DESC LIMIT 1"
                ).fetchone()
                if row:
                    sid = row[0] if not hasattr(row, "get") else row.get("scan_id", row[0])
                    rows = self.conn.execute(
                        "SELECT ticker FROM calls WHERE scan_id=? "
                        "ORDER BY explosive_score DESC LIMIT 40", (sid,)
                    ).fetchall()
                    for r in rows:
                        tk = r.get("ticker") if hasattr(r, "get") else r[0]
                        add(tk, "breakout")
            except Exception as e:
                print(f"  [unified] breakout pull failed: {e}")

        # --- full exchange universe (heavy) ---
        if mode == "universal":
            try:
                from trading_scanner import UniverseBuilder
                raw = UniverseBuilder().build("all")
                for t in raw:
                    add(t, "universal")
            except Exception as e:
                print(f"  [unified] universal build failed: {e}")

        return uni

    # ── scoring ────────────────────────────────────────────────────────────────
    def scan(self, mode: str = "smart", min_score: int = 55,
             limit: int | None = None, progress=None) -> list:
        from trading_scanner import compute_master_score

        if limit is None:
            limit = _DEFAULT_LIMITS.get(mode, 120)

        uni = self.gather(mode)
        tickers = list(uni.keys())

        # Universal: prefilter the (potentially thousands of) tickers down to the
        # most tradeable `limit` before paying for full Master-Score evals.
        if mode == "universal" and len(tickers) > limit:
            try:
                from trading_scanner import UniverseBuilder

                class _A:  # minimal args shim for quick_filter
                    max_cap = None
                    max_float = None

                passed = UniverseBuilder().quick_filter(tickers, _A())
                tickers = [p["ticker"] for p in passed][:limit]
            except Exception as e:
                print(f"  [unified] universal prefilter failed: {e}")
                tickers = tickers[:limit]
        else:
            tickers = tickers[:limit]

        results, total = [], len(tickers)
        for i, t in enumerate(tickers):
            if progress:
                try:
                    progress(i + 1, total, t)
                except Exception:
                    pass
            meta = uni.get(t, {})
            bias = meta.get("bias", "bullish")
            try:
                m = compute_master_score(
                    t, expiry=None, bias=bias, conn=self.conn,
                    market_regime=self.market_regime,
                )
            except Exception:
                continue
            score = float(m.get("score", 0) or 0)
            if score < min_score:
                continue
            results.append({
                "ticker": t,
                "score": round(score, 1),
                "grade": m.get("grade", "?"),
                "decision": m.get("decision", "SKIP"),
                "size_multiplier": float(m.get("size_multiplier", 0) or 0),
                "bias": bias,
                "sources": sorted(meta.get("sources", [])),
                "pct_change": meta.get("pct_change"),
                "summary": str(m.get("summary", ""))[:240],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ── persistence (single latest snapshot) ───────────────────────────────────
    def persist(self, results: list, mode: str) -> str:
        scan_id = str(_uuid.uuid4())[:8]
        ts = datetime.now(timezone.utc).isoformat()
        try:
            self.conn.execute("DELETE FROM unified_scan_results")
            for r in results:
                self.conn.execute(
                    "INSERT INTO unified_scan_results "
                    "(scan_id, scanned_at, mode, ticker, score, grade, decision, "
                    " size_multiplier, bias, sources, pct_change, summary) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (scan_id, ts, mode, r["ticker"], r["score"], r["grade"],
                     r["decision"], r["size_multiplier"], r["bias"],
                     ",".join(r["sources"]), r.get("pct_change"), r["summary"])
                )
        except Exception as e:
            print(f"  [unified] persist failed: {e}")
        return scan_id


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS (used by monitor.py + app.py)
# ══════════════════════════════════════════════════════════════════════════════
def run_unified_scan(conn, mode: str = "smart", min_score: int = 55,
                     limit: int | None = None, market_regime=None,
                     watchlist: list | None = None, persist: bool = True,
                     progress=None) -> list:
    """Run a unified scan and (optionally) persist the latest snapshot."""
    us = UnifiedScanner(conn, market_regime=market_regime, watchlist=watchlist)
    results = us.scan(mode=mode, min_score=min_score, limit=limit, progress=progress)
    if persist:
        us.persist(results, mode)
    return results


def get_latest_unified(conn, decisions: tuple | None = None, limit: int = 60) -> list:
    """Read the latest persisted unified snapshot (for UI / auto-entry)."""
    try:
        rows = conn.execute(
            "SELECT scan_id, scanned_at, mode, ticker, score, grade, decision, "
            "size_multiplier, bias, sources, pct_change, summary "
            "FROM unified_scan_results ORDER BY score DESC LIMIT ?", (limit,)
        ).fetchall()
    except Exception:
        return []
    out = []
    for r in rows:
        def g(key, idx):
            return r.get(key) if hasattr(r, "get") else r[idx]
        d = {
            "scan_id": g("scan_id", 0), "scanned_at": g("scanned_at", 1),
            "mode": g("mode", 2), "ticker": g("ticker", 3),
            "score": g("score", 4), "grade": g("grade", 5),
            "decision": g("decision", 6), "size_multiplier": g("size_multiplier", 7),
            "bias": g("bias", 8), "sources": str(g("sources", 9) or "").split(","),
            "pct_change": g("pct_change", 10), "summary": g("summary", 11),
        }
        if decisions and d["decision"] not in decisions:
            continue
        out.append(d)
    return out


if __name__ == "__main__":
    # Standalone smoke test: just the movers feed (no DB needed).
    print("Fetching today's top movers (Finviz + yfinance)...")
    movers = TopMoversScanner().fetch(min_pct=5.0, max_results=25)
    print(f"\n{len(movers)} tradeable movers:\n")
    for m in movers:
        print(f"  {m['ticker']:6s}  {m['pct_change']:+7.2f}%  "
              f"${m['price']:<9.2f} vol={m['volume']:>12,}  [{m['bias']}]")
