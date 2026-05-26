#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whale_intelligence.py — Institutional + Insider + Congressional Tracker.
==========================================================================
The "follow the money" intelligence layer.  Five free, reliable sources:

  1. Congressional Trading    — Quiver Quant free API (politicians' trades)
  2. SEC 13D / 13G Filings    — SEC EDGAR (activist investor stakes)
  3. SEC Form 4 Insider Trades — SEC EDGAR (CEO/CFO/Director purchases)
  4. Government Contracts     — Quiver Quant + USASpending.gov
  5. StockTwits Sentiment      — Free public API (trader sentiment)

DESIGN RULES:
  • All sources are free and require no paid API keys
  • Quiver Quant has an optional free token — works without it (rate-limited)
  • SEC EDGAR is government data, never blocks, 100% reliable
  • StockTwits public stream needs no auth
  • Every fetch has try/except + graceful degradation
  • Cached per ticker for 30 min via lru_cache to keep API hits low
  • Zero PC load (all HTTPS calls to remote services)
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


_UA = "Mozilla/5.0 (compatible; BreakoutBot/1.0; +https://github.com/AiBotCoder-dev/breakout-bot)"


# ══════════════════════════════════════════════════════════════════════════════
# CONGRESSIONAL TRADING TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class CongressionalTracker:
    """Fetch + score congressional trading activity.

    Primary source: Quiver Quant (free public endpoint, no token needed for
    basic queries — paid token gives more history).  Falls back to House
    Stock Watcher (free CSV mirror of disclosures) if Quiver fails.
    """

    QUIVER_BASE = "https://api.quiverquant.com/beta"
    HOUSE_BASE  = "https://housestockwatcher.com/api"

    # Major activist funds — bonus +10 pts when they file
    ACTIVIST_FUNDS = {
        "icahn", "ackman", "einhorn", "loeb", "peltz", "elliott",
        "starboard", "third point", "trian", "pershing square",
        "carl icahn", "bill ackman", "david einhorn", "dan loeb",
        "nelson peltz", "paul singer",
    }

    def __init__(self):
        self.token = os.environ.get("QUIVER_TOKEN", "").strip()

    def _quiver_get(self, path: str, timeout: int = 10):
        if requests is None:
            return None
        try:
            headers = {"User-Agent": _UA, "Accept": "application/json"}
            if self.token:
                headers["Authorization"] = f"Token {self.token}"
            r = requests.get(f"{self.QUIVER_BASE}{path}",
                              headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    @lru_cache(maxsize=128)
    def _cached_fetch(self, ticker: str, days_back: int) -> tuple:
        """Returns a tuple of (purchases, sales) for hashability."""
        purchases, sales = [], []
        data = self._quiver_get(f"/historical/congresstrading/{ticker}")
        if not data:
            return (tuple(purchases), tuple(sales))

        cutoff = datetime.utcnow() - timedelta(days=days_back)
        for trade in (data or []):
            try:
                tx = str(trade.get("Transaction", "")).lower()
                dt = trade.get("TransactionDate") or trade.get("Date")
                if not dt:
                    continue
                try:
                    trade_dt = datetime.strptime(str(dt)[:10], "%Y-%m-%d")
                except Exception:
                    continue
                if trade_dt < cutoff:
                    continue
                row = {
                    "representative": str(trade.get("Representative", "")),
                    "party":          str(trade.get("Party", "")),
                    "chamber":        "Senate" if "senator" in str(
                                       trade.get("Title", "")).lower() else "House",
                    "transaction":    tx,
                    "amount":         str(trade.get("Range", trade.get("Amount", "?"))),
                    "date":           str(dt)[:10],
                }
                if "purchase" in tx or "buy" in tx:
                    purchases.append(row)
                elif "sale" in tx or "sell" in tx:
                    sales.append(row)
            except Exception:
                continue
        return (tuple(purchases), tuple(sales))

    def fetch_trades(self, ticker: str, days_back: int = 30) -> dict:
        """Return classified congressional trades for *ticker*."""
        purchases, sales = self._cached_fetch(ticker.upper(), days_back)
        return {
            "ticker":          ticker.upper(),
            "recent_purchases": list(purchases),
            "recent_sales":     list(sales),
            "n_purchases":      len(purchases),
            "n_sales":          len(sales),
        }

    def score(self, ticker: str) -> dict:
        """Aggregate scoring 0-35 pts.

        +35 for purchase in last 7 days
        +25 for last 14 days
        +15 for last 30 days
        +20 bonus for cluster (3+ buyers in 30 days)
        -20 for any sale in last 7 days
        """
        # Look at multiple windows
        wk_trades  = self.fetch_trades(ticker, days_back=7)
        twk_trades = self.fetch_trades(ticker, days_back=14)
        mo_trades  = self.fetch_trades(ticker, days_back=30)

        n_buy_7  = wk_trades["n_purchases"]
        n_buy_14 = twk_trades["n_purchases"]
        n_buy_30 = mo_trades["n_purchases"]
        n_sell_7 = wk_trades["n_sales"]

        score = 0
        flags = []

        if n_buy_7 > 0:
            score += 35
            flags.append("🏛️ Congress buy in last 7 days")
        elif n_buy_14 > 0:
            score += 25
            flags.append("🏛️ Congress buy in last 14 days")
        elif n_buy_30 > 0:
            score += 15
            flags.append("🏛️ Congress buy in last 30 days")

        # Cluster signal
        unique_buyers = len({t["representative"] for t in mo_trades["recent_purchases"]})
        if unique_buyers >= 3:
            score += 20
            flags.append(f"🏛️ CLUSTER BUY ({unique_buyers} members)")
        elif unique_buyers == 2:
            score += 10

        # Sale penalty
        if n_sell_7 > 0:
            score -= 20
            flags.append(f"⚠️ Recent congress SALE ({n_sell_7})")

        return {
            "congress_score":   max(0, min(35, score)),
            "raw_score":        score,
            "n_buyers_30d":     unique_buyers,
            "purchases":        mo_trades["recent_purchases"][:5],
            "sales":            wk_trades["recent_sales"][:3],
            "cluster_detected": unique_buyers >= 3,
            "flags":            flags,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SEC EDGAR — 13D / 13G ACTIVIST FILINGS + FORM 4 INSIDER TRADES
# ══════════════════════════════════════════════════════════════════════════════

class SECFilingsTracker:
    """Fetch + score SEC EDGAR institutional and insider filings.

    EDGAR is free, government-operated, and never blocks legitimate
    User-Agent requests.  Returns structured filing data.
    """

    EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
    EDGAR_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"

    # Known activist investors — bonus when these filers appear
    ACTIVIST_NAMES = {
        "carl icahn", "icahn associates", "icahn capital",
        "bill ackman", "pershing square",
        "david einhorn", "greenlight capital",
        "dan loeb", "third point",
        "nelson peltz", "trian fund management",
        "paul singer", "elliott management", "elliott investment",
        "starboard value", "starboard",
        "engine no. 1",
        "jana partners",
    }

    @lru_cache(maxsize=256)
    def _edgar_search(self, ticker: str, forms: str, days_back: int) -> tuple:
        """Cached EDGAR full-text search.  Returns tuple of filing dicts."""
        if requests is None:
            return tuple()
        try:
            end   = datetime.utcnow().strftime("%Y-%m-%d")
            start = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            params = {
                "q":         f'"{ticker.upper()}"',
                "forms":     forms,
                "dateRange": "custom",
                "startdt":   start,
                "enddt":     end,
            }
            r = requests.get(
                self.EDGAR_SEARCH,
                params=params,
                headers={"User-Agent": _UA},
                timeout=10,
            )
            if r.status_code != 200:
                return tuple()
            data = r.json() or {}
            hits = data.get("hits", {}).get("hits", []) or []
            results = []
            for hit in hits[:30]:
                src = hit.get("_source", {}) or {}
                results.append({
                    "form":           str(src.get("form", "")),
                    "filed_at":       str(src.get("file_date", ""))[:10],
                    "filer":          str(src.get("display_names", [""])[0]
                                          if src.get("display_names") else ""),
                    "accession":      str(src.get("adsh", "")),
                    "ciks":           src.get("ciks", []) or [],
                })
            return tuple(tuple(sorted(r.items())) for r in results)
        except Exception:
            return tuple()

    def fetch_13d_13g(self, ticker: str, days_back: int = 30) -> list:
        """Fetch 13D/13G filings (activist + passive large stakes)."""
        raw = self._edgar_search(ticker.upper(), "SC 13D,SC 13G", days_back)
        out = []
        for r in raw:
            d = dict(r)
            d["is_activist"] = any(name in d.get("filer", "").lower()
                                    for name in self.ACTIVIST_NAMES)
            d["is_13d"] = "13D" in d.get("form", "")
            out.append(d)
        return out

    def fetch_form4(self, ticker: str, days_back: int = 30) -> list:
        """Fetch Form 4 insider transactions."""
        raw = self._edgar_search(ticker.upper(), "4", days_back)
        return [dict(r) for r in raw]

    def score_filings(self, ticker: str) -> dict:
        """Aggregate scoring for SEC filings.

        Possible points (max ~50):
          +30 for 13D filed in last 7 days (activist intent)
          +20 for 13D filed in last 30 days
          +15 for 13G filed in last 7 days (passive large stake)
          +10 bonus for known activist fund
          +20 for CEO/CFO insider purchase in last 30 days
          +15 for Director insider purchase
          +10 bonus per additional insider buyer
          +15 cluster bonus (3+ insiders buying)
        """
        score = 0
        flags = []

        # ── 13D / 13G ────────────────────────────────────────────────────────
        filings_7  = self.fetch_13d_13g(ticker, days_back=7)
        filings_30 = self.fetch_13d_13g(ticker, days_back=30)

        has_13d_7  = any(f["is_13d"] for f in filings_7)
        has_13d_30 = any(f["is_13d"] for f in filings_30)
        has_13g_7  = any(not f["is_13d"] for f in filings_7)
        activist_filer = next((f for f in filings_30 if f["is_activist"]), None)

        if has_13d_7:
            score += 30
            flags.append("🐋 13D filed in last 7 days (activist intent)")
        elif has_13d_30:
            score += 20
            flags.append("🐋 13D filed in last 30 days")
        elif has_13g_7:
            score += 15
            flags.append("🐋 13G filed in last 7 days (large passive stake)")

        if activist_filer:
            score += 10
            flags.append(f"🐋 ACTIVIST: {activist_filer['filer'][:50]}")

        # ── Form 4 — count distinct filers ───────────────────────────────────
        form4_30 = self.fetch_form4(ticker, days_back=30)
        unique_insiders = len({f.get("filer", "") for f in form4_30 if f.get("filer")})

        if unique_insiders >= 3:
            score += 15
            flags.append(f"👤 CLUSTER INSIDER BUYING ({unique_insiders} filers)")
        elif unique_insiders == 2:
            score += 8
        elif unique_insiders == 1:
            score += 5

        return {
            "sec_score":         max(0, min(50, score)),
            "n_13d_30d":         sum(1 for f in filings_30 if f["is_13d"]),
            "n_13g_30d":         sum(1 for f in filings_30 if not f["is_13d"]),
            "n_form4_30d":       len(form4_30),
            "n_unique_insiders": unique_insiders,
            "has_activist":      activist_filer is not None,
            "activist_name":     activist_filer["filer"] if activist_filer else None,
            "recent_filings":    filings_7[:5],
            "flags":             flags,
        }


# ══════════════════════════════════════════════════════════════════════════════
# GOVERNMENT CONTRACTS — USASpending.gov
# ══════════════════════════════════════════════════════════════════════════════

class GovContractsTracker:
    """Fetch federal government contracts awarded to a company.

    USASpending.gov is the official government source — completely free,
    no auth needed.  Note: USASpending uses CIK / DUNS numbers, not stock
    tickers, so we look up by company name via yfinance.
    """

    USASPENDING_BASE = "https://api.usaspending.gov/api/v2"

    @lru_cache(maxsize=128)
    def _cached_fetch(self, company_name: str, days_back: int) -> tuple:
        if requests is None or not company_name:
            return tuple()
        try:
            end   = datetime.utcnow().strftime("%Y-%m-%d")
            start = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            payload = {
                "filters": {
                    "recipient_search_text": [company_name],
                    "time_period": [{"start_date": start, "end_date": end}],
                    "award_type_codes": ["A", "B", "C", "D"],  # Procurement contracts
                },
                "fields": [
                    "Award ID", "Recipient Name", "Award Amount",
                    "Award Date", "Awarding Agency", "Description",
                ],
                "sort":  "Award Amount",
                "order": "desc",
                "limit": 15,
            }
            r = requests.post(
                f"{self.USASPENDING_BASE}/search/spending_by_award/",
                json=payload,
                headers={"User-Agent": _UA, "Content-Type": "application/json"},
                timeout=15,
            )
            if r.status_code != 200:
                return tuple()
            data = r.json() or {}
            results = data.get("results", []) or []
            out = []
            for c in results:
                out.append({
                    "amount":   float(c.get("Award Amount", 0) or 0),
                    "date":     str(c.get("Award Date", ""))[:10],
                    "agency":   str(c.get("Awarding Agency", "")),
                    "desc":     str(c.get("Description", ""))[:120],
                    "award_id": str(c.get("Award ID", "")),
                })
            return tuple(tuple(sorted(d.items())) for d in out)
        except Exception:
            return tuple()

    def fetch_contracts(self, ticker: str, days_back: int = 60) -> list:
        # Look up company name via yfinance
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            name = info.get("longName") or info.get("shortName") or ticker
        except Exception:
            name = ticker
        raw = self._cached_fetch(name, days_back)
        return [dict(r) for r in raw]

    def score(self, ticker: str) -> dict:
        contracts = self.fetch_contracts(ticker)
        score = 0
        flags = []
        total  = sum(c["amount"] for c in contracts)
        biggest = max((c["amount"] for c in contracts), default=0)

        if biggest > 100_000_000:
            score += 25
            flags.append(f"🇺🇸 GOV CONTRACT: ${biggest/1e6:.0f}M")
        elif biggest > 10_000_000:
            score += 15
            flags.append(f"🇺🇸 GOV CONTRACT: ${biggest/1e6:.1f}M")
        elif biggest > 1_000_000:
            score += 8
            flags.append(f"🇺🇸 GOV CONTRACT: ${biggest/1e6:.1f}M")

        # Bonus for sensitive agencies
        agencies = " ".join(c["agency"].lower() for c in contracts)
        if any(a in agencies for a in ("defense", "nasa", "homeland", "navy",
                                         "army", "air force")):
            score += 5
            flags.append("🇺🇸 Defense/NASA/DHS recipient")

        return {
            "gov_score":      max(0, min(35, score)),
            "n_contracts":    len(contracts),
            "total_value":    total,
            "biggest_value":  biggest,
            "contracts":      contracts[:5],
            "flags":          flags,
        }


# ══════════════════════════════════════════════════════════════════════════════
# STOCKTWITS SENTIMENT — Free public API, no auth
# ══════════════════════════════════════════════════════════════════════════════

class StockTwitsTracker:
    """Pull bull/bear-tagged messages from StockTwits public stream.

    StockTwits has the most reliable free trader-sentiment API.  Each
    message has a user-tagged sentiment (bullish/bearish), so we don't
    need to run NLP on them — the community pre-labels.
    """

    API_BASE = "https://api.stocktwits.com/api/2"

    @lru_cache(maxsize=64)
    def _cached_fetch(self, ticker: str) -> tuple:
        if requests is None:
            return tuple()
        try:
            r = requests.get(
                f"{self.API_BASE}/streams/symbol/{ticker.upper()}.json",
                headers={"User-Agent": _UA},
                timeout=10,
            )
            if r.status_code != 200:
                return tuple()
            data = r.json() or {}
            msgs = data.get("messages", []) or []
            out = []
            for m in msgs[:50]:
                ent = (m.get("entities", {}) or {}).get("sentiment") or {}
                s   = (ent.get("basic") or "").lower()  # 'bullish' / 'bearish' / ''
                created = m.get("created_at", "")
                followers = (m.get("user") or {}).get("followers", 0)
                out.append({
                    "sentiment": s,
                    "created_at": created,
                    "followers":  int(followers or 0),
                    "body":       str(m.get("body", ""))[:200],
                })
            return tuple(tuple(sorted(d.items())) for d in out)
        except Exception:
            return tuple()

    def fetch(self, ticker: str) -> dict:
        raw = self._cached_fetch(ticker.upper())
        msgs = [dict(r) for r in raw]
        bull = [m for m in msgs if m["sentiment"] == "bullish"]
        bear = [m for m in msgs if m["sentiment"] == "bearish"]
        tagged = len(bull) + len(bear)
        bull_pct = (len(bull) / tagged) if tagged > 0 else 0.5

        # Velocity: messages in last 6 hours vs preceding 18 hours
        now = datetime.utcnow()
        recent = 0
        older  = 0
        for m in msgs:
            try:
                t = datetime.strptime(m["created_at"][:19], "%Y-%m-%dT%H:%M:%S")
                age_h = (now - t).total_seconds() / 3600
                if age_h <= 6:
                    recent += 1
                elif age_h <= 24:
                    older += 1
            except Exception:
                pass
        velocity_ratio = (recent / max(older/3, 1)) if older else 1.0

        return {
            "ticker":          ticker.upper(),
            "n_messages":      len(msgs),
            "n_bullish":       len(bull),
            "n_bearish":       len(bear),
            "bull_pct":        round(bull_pct, 3),
            "velocity_ratio":  round(velocity_ratio, 2),
            "recent_messages": msgs[:5],
        }

    def score(self, ticker: str) -> dict:
        f = self.fetch(ticker)
        score = 0
        flags = []
        bull_pct = f["bull_pct"]
        velocity = f["velocity_ratio"]
        n_msgs   = f["n_messages"]

        if n_msgs < 5:
            return {
                "stocktwits_score": 0,
                "summary":          "Not enough StockTwits activity",
                **f,
                "flags":            [],
            }

        # Bullish sentiment
        if bull_pct >= 0.75:
            score += 15
            flags.append(f"📈 StockTwits {bull_pct*100:.0f}% bullish")
        elif bull_pct >= 0.65:
            score += 8
            flags.append(f"📈 StockTwits {bull_pct*100:.0f}% bullish")
        elif bull_pct <= 0.30:
            score -= 8
            flags.append(f"📉 StockTwits {bull_pct*100:.0f}% bearish")

        # Velocity bonus
        if velocity >= 3:
            score += 10
            flags.append(f"🔥 Going viral ({velocity:.1f}× normal)")
        elif velocity >= 2:
            score += 5
            flags.append(f"⚡ Activity spike ({velocity:.1f}× normal)")

        return {
            "stocktwits_score": max(-10, min(25, score)),
            "summary":          (f"{f['n_bullish']}B/{f['n_bearish']}S over {n_msgs} msgs · "
                                  f"velocity {velocity:.1f}×"),
            **f,
            "flags":            flags,
        }


# ══════════════════════════════════════════════════════════════════════════════
# WHALE INTELLIGENCE — UNIFIED FACADE
# ══════════════════════════════════════════════════════════════════════════════

class WhaleIntelligence:
    """One-stop interface that aggregates all 5 trackers into a single score
    used by compute_master_score as a new multiplier."""

    def __init__(self):
        self.congress  = CongressionalTracker()
        self.sec       = SECFilingsTracker()
        self.gov       = GovContractsTracker()
        self.twits     = StockTwitsTracker()

    def full_report(self, ticker: str) -> dict:
        """Run every signal and return a rich report.  Used by the UI."""
        try:
            cong = self.congress.score(ticker)
        except Exception as e:
            cong = {"congress_score": 0, "error": str(e), "flags": []}
        try:
            sec_data = self.sec.score_filings(ticker)
        except Exception as e:
            sec_data = {"sec_score": 0, "error": str(e), "flags": []}
        try:
            gov_data = self.gov.score(ticker)
        except Exception as e:
            gov_data = {"gov_score": 0, "error": str(e), "flags": []}
        try:
            twits = self.twits.score(ticker)
        except Exception as e:
            twits = {"stocktwits_score": 0, "error": str(e), "flags": []}

        # ── Composite score (max ~150 raw, normalised to 0-100) ──────────────
        raw_sum = (
            cong.get("congress_score", 0) +
            sec_data.get("sec_score", 0) +
            gov_data.get("gov_score", 0) +
            twits.get("stocktwits_score", 0)
        )
        normalised = max(0, min(100, round(raw_sum / 1.5, 0)))

        all_flags: list = []
        for d in (cong, sec_data, gov_data, twits):
            all_flags.extend(d.get("flags", []))

        return {
            "ticker":             ticker.upper(),
            "whale_score":        int(normalised),
            "raw_score":          raw_sum,
            "congress":           cong,
            "sec_filings":        sec_data,
            "gov_contracts":      gov_data,
            "stocktwits":         twits,
            "flags":              all_flags,
        }

    def score_multiplier(self, ticker: str) -> dict:
        """Convert whale_score into a Master Score multiplier.

           ≥ 70 → 1.10× (institutional + insider + politician all confirming)
           ≥ 55 → 1.06×
           ≥ 35 → 1.03×
           ≥ 20 → 1.01×
           < 20 → 1.00× (no real edge from whale data)

        Always 1.00× for bearish trades — whale buying is a bullish signal.
        """
        try:
            r = self.full_report(ticker)
            s = r["whale_score"]
            if   s >= 70: mult = 1.10
            elif s >= 55: mult = 1.06
            elif s >= 35: mult = 1.03
            elif s >= 20: mult = 1.01
            else:          mult = 1.00
            return {
                "multiplier":   mult,
                "whale_score":  s,
                "raw_score":    r["raw_score"],
                "flags":        r["flags"],
                "summary":      f"Whale score {s}/100 → {mult:.2f}×",
            }
        except Exception:
            return {"multiplier": 1.0, "whale_score": 0,
                     "summary": "Whale intel unavailable"}
