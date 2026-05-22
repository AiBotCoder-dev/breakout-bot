#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_agent.py — News-driven market intelligence engine.
==========================================================
Pulls live headlines from multiple free RSS sources, classifies each via
the AI engine for category + sentiment + impact, identifies affected
tickers / sectors, and exposes a clean API so the rest of the bot can:

  • Skip auto-entries when major negative news just landed on a ticker
  • Boost sizing when major positive news is breaking
  • Surface news context in Telegram alerts + the Streamlit UI
  • Add a news component to the Master Score

CATEGORIES (what the agent is looking for):
  HEALTH       — pandemics, FDA decisions, drug approvals, recalls
  WAR          — geopolitical conflicts, military events, sanctions
  TARIFFS      — trade policy, import duties, embargoes
  NEGOTIATIONS — M&A, trade deals, peace talks, settlements
  EARNINGS     — earnings releases, guidance, pre-announcements
  REGULATORY   — SEC actions, antitrust, new laws, fines
  MACRO        — Fed decisions, inflation data, GDP, jobs reports
  PRODUCT      — major launches, partnerships, breakthroughs
  LEGAL        — lawsuits, criminal indictments, large settlements
  OTHER        — anything else market-relevant

FREE SOURCES (all RSS, no API keys required):
  - MarketWatch Top Stories
  - CNBC Markets
  - Federal Reserve press releases
  - Investing.com latest news
  - Yahoo Finance headlines

ON CLOUD: fully free, runs inside the existing monitor.py workflow.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    import requests
except ImportError:
    requests = None

try:
    import feedparser
except ImportError:
    feedparser = None


# ── RSS sources (verified working as of 2026) ─────────────────────────────────

_RSS_SOURCES = [
    ("MarketWatch",  "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("CNBC Markets", "https://www.cnbc.com/id/15839069/device/rss/rss.html"),
    ("Investing",    "https://www.investing.com/rss/news.rss"),
    ("Fed",          "https://www.federalreserve.gov/feeds/press_all.xml"),
    ("Yahoo",        "https://finance.yahoo.com/news/rssindex"),
]

_UA = "Mozilla/5.0 (compatible; BreakoutBot/1.0)"


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: row access compatible with sqlite3.Row + PgAdapter._PgRow
# ══════════════════════════════════════════════════════════════════════════════

def _rg(row, key, default=None):
    if row is None:
        return default
    try:
        if hasattr(row, "get") and callable(row.get):
            return row.get(key, default)
    except Exception:
        pass
    try:
        return dict(row).get(key, default)
    except Exception:
        try:
            return row[key]
        except Exception:
            return default


# ══════════════════════════════════════════════════════════════════════════════
# NEWS AGENT
# ══════════════════════════════════════════════════════════════════════════════

class NewsAgent:
    """Pulls news, classifies via AI, persists, and exposes decision helpers."""

    _INIT_SQL = """
CREATE TABLE IF NOT EXISTS paper_news_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at         TIMESTAMP,
    published_at       TIMESTAMP,
    source             TEXT,
    headline           TEXT,
    summary            TEXT,
    url                TEXT,
    category           TEXT,
    sentiment          TEXT,
    impact_score       INTEGER,
    affected_tickers   TEXT,
    affected_sectors   TEXT,
    urgency            TEXT,
    classified         INTEGER DEFAULT 0,
    dedupe_key         TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_dedupe   ON paper_news_events(dedupe_key);
CREATE INDEX IF NOT EXISTS idx_news_fetched  ON paper_news_events(fetched_at);
CREATE INDEX IF NOT EXISTS idx_news_ticker   ON paper_news_events(affected_tickers);
"""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def __init__(self, conn, ai_analyst=None):
        self.conn = conn
        self.ai   = ai_analyst
        try:
            self.conn.executescript(self._INIT_SQL)
        except Exception:
            pass

    # ── Fetching: pull raw headlines from all RSS sources ───────────────────

    def fetch_all_sources(self, max_per_source: int = 15) -> list:
        """Pull latest headlines from every configured RSS source.
        Returns list of raw dicts (not yet classified or persisted)."""
        if feedparser is None:
            return []
        out = []
        for src_name, url in _RSS_SOURCES:
            try:
                feed = feedparser.parse(url, request_headers={"User-Agent": _UA})
                for entry in (feed.entries or [])[:max_per_source]:
                    headline = (getattr(entry, "title", "") or "").strip()
                    summary  = (getattr(entry, "summary", "") or
                                getattr(entry, "description", "") or "").strip()[:600]
                    link     = (getattr(entry, "link", "") or "").strip()
                    pub      = (getattr(entry, "published", "") or
                                getattr(entry, "updated", "") or "")
                    if not headline:
                        continue
                    out.append({
                        "source":       src_name,
                        "headline":     headline,
                        "summary":      summary,
                        "url":          link,
                        "published_at": str(pub)[:64],
                        "dedupe_key":   self._dedupe_key(headline),
                    })
            except Exception:
                continue
        return out

    @staticmethod
    def _dedupe_key(headline: str) -> str:
        """Stable hash of a normalized headline for dedup."""
        import hashlib
        h = (headline or "").strip().lower()
        h = "".join(c for c in h if c.isalnum() or c.isspace())
        h = " ".join(h.split())
        return hashlib.md5(h.encode("utf-8")).hexdigest()[:16]

    # ── Classification: send each headline through AI ───────────────────────

    _CATEGORIES = (
        "HEALTH WAR TARIFFS NEGOTIATIONS EARNINGS REGULATORY "
        "MACRO PRODUCT LEGAL OTHER"
    ).split()

    def classify(self, item: dict) -> dict:
        """Use AI to classify a single news item.  Falls back to keyword
        heuristics when AI isn't available so the agent still functions."""
        headline = item.get("headline", "")
        summary  = item.get("summary", "")
        if not headline:
            return {}

        # ── AI path ────────────────────────────────────────────────────────
        if self.ai and getattr(self.ai, "available", False):
            prompt = (
                "Classify this market news headline.  Return STRICT JSON with "
                "keys: category, sentiment, impact_score, affected_tickers, "
                "affected_sectors, urgency.\n\n"
                f"Headline: {headline}\n"
                f"Summary: {summary}\n\n"
                "Constraints:\n"
                "  category: one of " + " | ".join(self._CATEGORIES) + "\n"
                "  sentiment: POSITIVE | NEGATIVE | NEUTRAL  "
                "(for US equities broadly)\n"
                "  impact_score: integer 1-10 (1=trivial, 10=market-moving)\n"
                "  affected_tickers: array of up to 5 US ticker symbols. "
                "Empty array if no specific company is implicated.\n"
                "  affected_sectors: array of sectors (e.g. ['Technology', "
                "'Energy']).\n"
                "  urgency: LOW | MEDIUM | HIGH\n\n"
                "Output ONLY the JSON object — no markdown, no explanation."
            )
            try:
                resp = self.ai.chat(prompt, max_tokens=350)
                # Strip code fences if AI wrapped them
                if "```" in resp:
                    resp = resp.split("```")[1]
                    if resp.startswith("json"):
                        resp = resp[4:]
                resp = resp.strip()
                # Find the JSON object boundary
                if "{" in resp and "}" in resp:
                    resp = resp[resp.index("{"): resp.rindex("}") + 1]
                data = json.loads(resp)
                # Normalize
                return {
                    "category":         str(data.get("category", "OTHER")).upper(),
                    "sentiment":        str(data.get("sentiment", "NEUTRAL")).upper(),
                    "impact_score":     int(data.get("impact_score", 5)),
                    "affected_tickers": [str(t).upper() for t in
                                         (data.get("affected_tickers") or [])][:5],
                    "affected_sectors": [str(s) for s in
                                         (data.get("affected_sectors") or [])][:5],
                    "urgency":          str(data.get("urgency", "LOW")).upper(),
                }
            except Exception:
                pass

        # ── Keyword fallback (works without AI) ─────────────────────────────
        return self._keyword_classify(headline + " " + summary)

    @staticmethod
    def _keyword_classify(text: str) -> dict:
        """Deterministic fallback when AI is unavailable."""
        t = text.lower()

        # Category — first match wins
        if any(w in t for w in (
            "fda ", "pandemic", "outbreak", "vaccine", "drug recall",
            "clinical trial", "approval", "health emergency",
        )):
            category = "HEALTH"
        elif any(w in t for w in (
            "war", "missile", "invasion", "ceasefire", "military strike",
            "armed conflict", "nuclear", "sanctions",
        )):
            category = "WAR"
        elif any(w in t for w in (
            "tariff", "import duty", "trade barrier", "embargo",
        )):
            category = "TARIFFS"
        elif any(w in t for w in (
            "merger", "acquisition", "deal", "negotiation", "settle",
            "agreement",
        )):
            category = "NEGOTIATIONS"
        elif any(w in t for w in (
            "earnings", "revenue", "eps", "guidance", "forecast",
            "pre-announce",
        )):
            category = "EARNINGS"
        elif any(w in t for w in (
            "sec ", "antitrust", "regulator", "fine", "investigation",
            "subpoena", "lawsuit", "indict",
        )):
            category = "REGULATORY"
        elif any(w in t for w in (
            "fed ", "fomc", "powell", "inflation", "cpi", "ppi",
            "gdp", "jobs report", "unemployment", "interest rate",
        )):
            category = "MACRO"
        elif any(w in t for w in (
            "launch", "release", "unveil", "breakthrough", "patent",
            "partnership",
        )):
            category = "PRODUCT"
        else:
            category = "OTHER"

        # Sentiment
        neg_words = ("decline", "crash", "fall", "drop", "warn", "miss",
                     "downgrade", "loss", "sue", "ban", "halt", "concern",
                     "fail", "fraud", "recall", "fire", "outage", "delay")
        pos_words = ("rise", "surge", "gain", "beat", "exceed", "upgrade",
                     "approval", "win", "boost", "rally", "record",
                     "breakthrough", "outperform", "raise", "expansion")
        neg = sum(1 for w in neg_words if w in t)
        pos = sum(1 for w in pos_words if w in t)
        if pos > neg:
            sentiment = "POSITIVE"
        elif neg > pos:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        # Impact: based on category and word strength
        weight = {"WAR": 8, "HEALTH": 7, "MACRO": 7, "TARIFFS": 7,
                  "REGULATORY": 6, "EARNINGS": 6, "NEGOTIATIONS": 5,
                  "LEGAL": 5, "PRODUCT": 4, "OTHER": 3}
        impact = weight.get(category, 4) + (1 if abs(pos - neg) >= 3 else 0)

        return {
            "category":         category,
            "sentiment":        sentiment,
            "impact_score":     min(10, impact),
            "affected_tickers": [],
            "affected_sectors": [],
            "urgency":          "HIGH" if impact >= 8 else ("MEDIUM" if impact >= 5 else "LOW"),
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def _exists(self, dedupe_key: str) -> bool:
        try:
            row = self.conn.execute(
                "SELECT 1 FROM paper_news_events WHERE dedupe_key=? LIMIT 1",
                (dedupe_key,)
            ).fetchone()
            return row is not None
        except Exception:
            return False

    def _persist(self, item: dict, classification: dict) -> Optional[int]:
        try:
            self.conn.execute(
                "INSERT INTO paper_news_events "
                "(fetched_at, published_at, source, headline, summary, url, "
                "category, sentiment, impact_score, affected_tickers, "
                "affected_sectors, urgency, classified, dedupe_key) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(),
                 item.get("published_at", ""),
                 item.get("source", ""),
                 item.get("headline", ""),
                 item.get("summary", "")[:1000],
                 item.get("url", ""),
                 classification.get("category", "OTHER"),
                 classification.get("sentiment", "NEUTRAL"),
                 int(classification.get("impact_score", 3)),
                 ",".join(classification.get("affected_tickers", [])),
                 ",".join(classification.get("affected_sectors", [])),
                 classification.get("urgency", "LOW"),
                 1,
                 item.get("dedupe_key", ""))
            )
            return True
        except Exception:
            return False

    # ── Orchestration: fetch + classify + persist in one call ────────────────

    def run_cycle(self, max_per_source: int = 15, max_new: int = 30) -> dict:
        """Pull, classify, and store new headlines.  Idempotent — already-seen
        headlines are skipped via dedupe_key.

        Returns dict:
          fetched   — total raw items fetched
          new       — number of newly persisted items
          high_imp  — number of HIGH-urgency items in this batch
        """
        raw = self.fetch_all_sources(max_per_source=max_per_source)
        new_count   = 0
        high_count  = 0
        new_items: list = []
        for item in raw:
            if not item.get("dedupe_key"):
                continue
            if self._exists(item["dedupe_key"]):
                continue
            if new_count >= max_new:
                break
            cls = self.classify(item)
            if not cls:
                continue
            ok = self._persist(item, cls)
            if ok:
                new_count += 1
                if cls.get("urgency") == "HIGH" or cls.get("impact_score", 0) >= 8:
                    high_count += 1
                    new_items.append({"item": item, "classification": cls})
        return {
            "fetched":  len(raw),
            "new":      new_count,
            "high_imp": high_count,
            "high_items": new_items,
        }

    # ── Decision helpers (what monitor.py + compute_master_score consume) ────

    def get_recent_news_for_ticker(self, ticker: str, hours: int = 24,
                                    limit: int = 10) -> list:
        """Return news events affecting *ticker* in the last N hours."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_news_events "
                "WHERE fetched_at >= ? "
                "AND (affected_tickers LIKE ? OR affected_tickers LIKE ? "
                "     OR affected_tickers LIKE ? OR affected_tickers = ?) "
                "ORDER BY fetched_at DESC LIMIT ?",
                (cutoff, f"%{ticker.upper()}%", f"{ticker.upper()},%",
                 f"%,{ticker.upper()}", ticker.upper(), int(limit))
            ).fetchall()
            return [dict(r) for r in rows or []]
        except Exception:
            return []

    def get_recent_market_news(self, hours: int = 24,
                                 min_impact: int = 5,
                                 limit: int = 20) -> list:
        """Return high-impact market-wide news in the last N hours."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_news_events "
                "WHERE fetched_at >= ? AND impact_score >= ? "
                "ORDER BY impact_score DESC, fetched_at DESC LIMIT ?",
                (cutoff, int(min_impact), int(limit))
            ).fetchall()
            return [dict(r) for r in rows or []]
        except Exception:
            return []

    def get_news_impact(self, ticker: str, hours: int = 24) -> dict:
        """Compute aggregate news impact for *ticker* over recent hours.

        Returns dict:
          n_events         — count of news items
          net_sentiment    — float -1.0 to +1.0 (impact-weighted)
          max_impact       — highest single-event impact score
          dominant_category — most frequent category
          top_event        — highest-impact headline
          score_modifier   — multiplier for Master Score (0.7 to 1.15)
          should_skip      — bool — true if major negative news → don't enter
          alert            — plain-English summary for Telegram
        """
        events = self.get_recent_news_for_ticker(ticker, hours=hours, limit=30)
        if not events:
            return {
                "n_events":          0,
                "net_sentiment":     0.0,
                "max_impact":        0,
                "dominant_category": "",
                "top_event":         None,
                "score_modifier":    1.0,
                "should_skip":       False,
                "alert":             "",
            }

        # Weight each event: sign * impact_score
        sign = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
        total_signed = sum(
            sign.get(_rg(e, "sentiment", "NEUTRAL"), 0) *
            int(_rg(e, "impact_score", 0) or 0)
            for e in events
        )
        total_weight = sum(int(_rg(e, "impact_score", 0) or 0) for e in events) or 1
        net_sent = max(-1.0, min(1.0, total_signed / total_weight))

        # Aggregate stats
        max_impact = max(int(_rg(e, "impact_score", 0) or 0) for e in events)
        cats = {}
        for e in events:
            c = _rg(e, "category", "OTHER")
            cats[c] = cats.get(c, 0) + 1
        dominant_cat = max(cats, key=lambda k: cats[k]) if cats else "OTHER"

        top_event = max(events, key=lambda e: int(_rg(e, "impact_score", 0) or 0))

        # ── Score modifier mapping ──────────────────────────────────────────
        # net_sent of +1 → 1.15x   (15 % bonus)
        # net_sent of  0 →  1.00x
        # net_sent of -1 → 0.70x   (30 % penalty)
        if net_sent >= 0:
            modifier = 1.0 + (net_sent * 0.15)
        else:
            modifier = 1.0 + (net_sent * 0.30)
        modifier = max(0.5, min(1.2, modifier))

        # Skip rule: any HIGH-urgency negative event with impact ≥ 7
        skip_events = [
            e for e in events
            if _rg(e, "sentiment") == "NEGATIVE"
            and int(_rg(e, "impact_score", 0) or 0) >= 7
            and _rg(e, "urgency") in ("HIGH", "MEDIUM")
            and _rg(e, "category") in ("HEALTH", "WAR", "REGULATORY",
                                         "LEGAL", "EARNINGS")
        ]
        should_skip = len(skip_events) > 0

        # Alert text (used in Telegram)
        top_headline = str(_rg(top_event, "headline", "") or "")[:120]
        alert = (
            f"📰 {ticker}: {len(events)} news event(s) in last {hours}h.  "
            f"Net sentiment {net_sent:+.2f} ({dominant_cat}).  "
            f"Top: \"{top_headline}\""
        )
        if should_skip:
            alert = "🛑 NEGATIVE NEWS — " + alert

        return {
            "n_events":           len(events),
            "net_sentiment":      round(net_sent, 3),
            "max_impact":         max_impact,
            "dominant_category":  dominant_cat,
            "top_event":          dict(top_event) if top_event else None,
            "score_modifier":     round(modifier, 3),
            "should_skip":        should_skip,
            "skip_events":        [dict(e) for e in skip_events],
            "alert":              alert,
        }

    def get_market_pulse(self, hours: int = 24) -> dict:
        """Snapshot of the broad market mood from news alone — used by the
        daily briefing and Streamlit dashboard."""
        events = self.get_recent_market_news(hours=hours, min_impact=4, limit=50)
        if not events:
            return {"mood": "NEUTRAL", "n_events": 0, "headlines": []}

        sign = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
        signed = sum(
            sign.get(_rg(e, "sentiment", "NEUTRAL"), 0) *
            int(_rg(e, "impact_score", 0) or 0)
            for e in events
        )
        weight = sum(int(_rg(e, "impact_score", 0) or 0) for e in events) or 1
        net = signed / weight
        if net >= 0.3:
            mood = "BULLISH"
        elif net >= 0.1:
            mood = "MILDLY_BULLISH"
        elif net <= -0.3:
            mood = "BEARISH"
        elif net <= -0.1:
            mood = "MILDLY_BEARISH"
        else:
            mood = "NEUTRAL"

        cats = {}
        for e in events:
            c = _rg(e, "category", "OTHER")
            cats[c] = cats.get(c, 0) + 1
        return {
            "mood":            mood,
            "net_sentiment":   round(net, 3),
            "n_events":        len(events),
            "category_counts": cats,
            "headlines":       [
                {
                    "headline":     _rg(e, "headline", ""),
                    "source":       _rg(e, "source", ""),
                    "category":     _rg(e, "category", ""),
                    "sentiment":    _rg(e, "sentiment", ""),
                    "impact_score": int(_rg(e, "impact_score", 0) or 0),
                    "url":          _rg(e, "url", ""),
                    "fetched_at":   str(_rg(e, "fetched_at", "")),
                }
                for e in events[:10]
            ],
        }
