"""
vip_news_monitor.py — Instant Telegram alerts when market-moving public figures post.

WHAT THIS DOES
--------------
Polls public RSS feeds for the highest-impact public figures and sends a Telegram
alert THE MOMENT a new post mentions a stock (or comes from the Fed). Designed for
the user's actual frustration: Trump posting "I love $NVDA" can move the stock 5%
in minutes, and you want to know NOW, not from a news article 30 minutes later.

CURRENT FEEDS (verified working)
--------------------------------
  • Donald Trump — Truth Social (via trumpstruth.org public RSS, 100 entries deep)
  • Federal Reserve — press releases (federalreserve.gov RSS, 20 entries deep)

Add more by appending to VIP_FEEDS. The runner is generic — any well-formed
RSS feed works (it dedupes via the entry id/link).

ALERT LOGIC
-----------
  • A post is ALERTED on Telegram when:
        - it mentions at least one ticker (cashtag $TICKER or known company name), OR
        - it's from an "official" feed (the Fed — every press release matters)
  • Posts WITHOUT market relevance still get logged to `vip_posts` for the
    dashboard feed, but don't fire a Telegram alert (so you're not spammed by
    every Trump rant).
  • Idempotent: posts are deduped by UNIQUE(vip_handle, post_id), so a second
    poll won't re-alert. Safe to call every monitor cycle.
"""

from __future__ import annotations

import re
import html
from datetime import datetime, timezone

try:
    import feedparser
except Exception:                       # pragma: no cover
    feedparser = None
try:
    import requests
except Exception:                       # pragma: no cover
    requests = None


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
VIP_FEEDS = [
    {
        "name":             "Donald Trump",
        "handle":           "DJT",
        "rss":              "https://trumpstruth.org/feed",
        "kind":             "social",       # alert ONLY if market-relevant
        "telegram_prefix":  "🚨 TRUMP POST",
    },
    {
        "name":             "Federal Reserve",
        "handle":           "FED",
        "rss":              "https://www.federalreserve.gov/feeds/press_all.xml",
        "kind":             "official",     # alert on EVERY release
        "telegram_prefix":  "🏦 FED RELEASE",
    },
]

# Company-name → ticker. Top liquid US names + names Trump references most. The
# key is *lowercase* and matched case-insensitively with word boundaries.
COMPANY_TICKERS = {
    # Trump-orbit / frequently mentioned
    # NB: bare "DJT" intentionally NOT mapped — Trump signs posts "President DJT",
    # which would false-alert on every signed post. We catch $DJT cashtags via the
    # regex, and his company by its name aliases.
    "trump media": "DJT", "trump media & technology": "DJT", "tmtg": "DJT",
    "truth social": "DJT",
    # Mega caps
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META", "instagram": "META",
    "tesla": "TSLA", "elon musk": "TSLA",  # proxy
    "netflix": "NFLX", "broadcom": "AVGO", "oracle": "ORCL",
    "amd": "AMD", "advanced micro devices": "AMD",
    "intel": "INTC", "qualcomm": "QCOM", "cisco": "CSCO",
    # Defense / aerospace (Trump-relevant)
    "boeing": "BA", "lockheed": "LMT", "lockheed martin": "LMT",
    "raytheon": "RTX", "rtx": "RTX", "northrop": "NOC",
    "general dynamics": "GD",
    # Energy
    "exxon": "XOM", "exxon mobil": "XOM", "chevron": "CVX",
    "conocophillips": "COP",
    # Financial
    "jpmorgan": "JPM", "jp morgan": "JPM", "bank of america": "BAC",
    "wells fargo": "WFC", "goldman sachs": "GS", "morgan stanley": "MS",
    "visa": "V", "mastercard": "MA",
    # Other Trump-tracked
    "palantir": "PLTR", "coinbase": "COIN", "robinhood": "HOOD",
    "shopify": "SHOP", "berkshire": "BRK-B", "warren buffett": "BRK-B",
    "spacex": "TSLA",   # not public, proxy via Musk
    "twitter": "X",     # mention proxy (X Corp private; keep aware)
    "uber": "UBER", "airbnb": "ABNB", "snowflake": "SNOW",
    # Crypto / digital
    "bitcoin": "MSTR", "btc": "MSTR", "microstrategy": "MSTR",
    "strategy inc": "MSTR",
    "ethereum": "ETH-USD",
    # Indices / ETFs Trump mentions
    "s&p 500": "SPY", "s&p": "SPY", "nasdaq": "QQQ", "russell 2000": "IWM",
    "dow jones": "DIA", "dow": "DIA",
    # Recent comebacks
    "intel foundry": "INTC", "tsmc": "TSM", "taiwan semi": "TSM",
    "asml": "ASML", "samsung": "SSNLF",
}

# Cashtags: $TSLA, $NVDA — 1-5 uppercase letters.
CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")

# Catch-all bullish/bearish keywords for crude sentiment.
_BULL_RE = re.compile(
    r"\b(great|amazing|tremendous|huge win|love|buy|surge|rally|booming|massive)\b",
    re.IGNORECASE)
_BEAR_RE = re.compile(
    r"\b(disaster|terrible|crash|sell|tank|collapse|failing|disgrace|stupid|sue)\b",
    re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════
def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)               # strip tags
    s = html.unescape(s)                         # decode &amp; etc.
    return re.sub(r"\s+", " ", s).strip()


def extract_tickers(text: str) -> list[str]:
    """Cashtags + company-name matches; deduped, sorted, capped at 8."""
    if not text:
        return []
    out: set[str] = set()
    for m in CASHTAG_RE.findall(text):
        if m.isalpha() and 1 <= len(m) <= 5:
            out.add(m.upper())
    low = text.lower()
    # Match longest names first so "trump media & technology" wins over "trump"
    for name in sorted(COMPANY_TICKERS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", low):
            out.add(COMPANY_TICKERS[name])
    # Filter out obviously bogus cashtags that aren't real tickers
    _BLOCK = {"USA", "USD", "US", "CEO", "AI", "BIG", "GDP", "OK", "ALL", "ONE",
              "I", "A", "S", "THE", "OF", "AND", "OR", "TO", "IN"}
    return sorted(t for t in out if t not in _BLOCK)


def classify_sentiment(text: str) -> str:
    """Naive but useful: 'bullish' / 'bearish' / 'neutral'."""
    if not text:
        return "neutral"
    bull = len(_BULL_RE.findall(text))
    bear = len(_BEAR_RE.findall(text))
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"


# ══════════════════════════════════════════════════════════════════════════════
# MONITOR
# ══════════════════════════════════════════════════════════════════════════════
class VipNewsMonitor:
    def __init__(self, conn):
        self.conn = conn
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS vip_posts (
                    id            SERIAL PRIMARY KEY,
                    vip_handle    TEXT,
                    vip_name      TEXT,
                    post_id       TEXT,
                    posted_at     TEXT,
                    fetched_at    TEXT,
                    url           TEXT,
                    title         TEXT,
                    text          TEXT,
                    tickers       TEXT,
                    sentiment     TEXT,
                    alerted       INTEGER DEFAULT 0,
                    UNIQUE (vip_handle, post_id)
                )
            """)
        except Exception as e:
            print(f"  [vip] table init failed: {e}")

    # ── feed fetch ─────────────────────────────────────────────────────────────
    @staticmethod
    def _fetch(url: str) -> list:
        if requests is None or feedparser is None:
            return []
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code != 200:
                return []
            f = feedparser.parse(r.content)
            return list(f.entries or [])
        except Exception:
            return []

    # ── one cycle: poll all feeds, dedupe, alert new market-relevant posts ─────
    def run_cycle(self, telegram_sender=None, max_per_feed: int = 25) -> dict:
        """
        telegram_sender: callable(str) -> bool, e.g. monitor.send_telegram.
        Returns dict { feed_handle: { fetched, new, alerted } }.
        """
        report: dict[str, dict] = {}
        for feed in VIP_FEEDS:
            handle = feed["handle"]
            stats = {"fetched": 0, "new": 0, "alerted": 0}
            try:
                entries = self._fetch(feed["rss"])[:max_per_feed]
                stats["fetched"] = len(entries)
                for e in entries:
                    pid = (e.get("id") or e.get("link") or e.get("guid") or "").strip()
                    if not pid:
                        continue
                    title = _strip_html(e.get("title", ""))
                    body  = _strip_html(e.get("summary", "") or e.get("description", ""))
                    url   = (e.get("link") or "").strip()
                    posted_at = (e.get("published") or e.get("updated") or "").strip()
                    full = f"{title}\n{body}".strip()
                    tickers = extract_tickers(full)
                    sentiment = classify_sentiment(full)

                    # Idempotent insert; if it already exists we silently skip.
                    inserted = self._insert(
                        feed, pid, posted_at, url, title, body, tickers, sentiment,
                    )
                    if not inserted:
                        continue
                    stats["new"] += 1

                    # Alert rules
                    is_official = feed.get("kind") == "official"
                    relevant    = bool(tickers) or is_official
                    if relevant and telegram_sender:
                        msg = self._format_telegram(feed, title, body, url,
                                                    tickers, sentiment, posted_at)
                        try:
                            if telegram_sender(msg):
                                stats["alerted"] += 1
                                self._mark_alerted(feed["handle"], pid)
                        except Exception:
                            pass
                        # ── Attach the exact affordable WEEKLY option to trade ──
                        # For each named ticker, find a weekly contract that fits
                        # the account ($200 default) and send a precise ENTER-NOW
                        # alert. This is the "news -> exact option" path.
                        for _tk in tickers[:2]:
                            try:
                                from catalyst_options import catalyst_to_weekly_alert
                                _cat = (title or body)[:160]
                                _opt_msg = catalyst_to_weekly_alert(
                                    _tk, sentiment, f"{feed['name']}: {_cat}")
                                if _opt_msg:
                                    telegram_sender(_opt_msg)
                            except Exception:
                                pass
            except Exception as ex:
                print(f"  [vip] {handle} feed failed: {ex}")
            report[handle] = stats
        return report

    # ── persistence ────────────────────────────────────────────────────────────
    def _insert(self, feed, post_id, posted_at, url, title, body, tickers, sentiment) -> bool:
        """
        Returns True only when a NEW row was actually inserted (conflict → False).
        Uses a pre-check SELECT to avoid relying on RETURNING behavior across the
        PgAdapter wrapper (and to keep the "is new" semantics race-free under the
        single-writer monitor pattern).
        """
        try:
            existing = self.conn.execute(
                "SELECT 1 FROM vip_posts WHERE vip_handle=? AND post_id=?",
                (feed["handle"], post_id),
            ).fetchone()
            if existing is not None:
                return False
            self.conn.execute(
                "INSERT INTO vip_posts "
                "(vip_handle, vip_name, post_id, posted_at, fetched_at, url, "
                " title, text, tickers, sentiment, alerted) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,0) "
                "ON CONFLICT (vip_handle, post_id) DO NOTHING",
                (feed["handle"], feed["name"], post_id, posted_at,
                 datetime.now(timezone.utc).isoformat(), url,
                 title[:500], body[:4000], ",".join(tickers), sentiment)
            )
            return True
        except Exception as e:
            print(f"  [vip] insert failed: {e}")
            return False

    def _mark_alerted(self, handle: str, post_id: str):
        try:
            self.conn.execute(
                "UPDATE vip_posts SET alerted=1 WHERE vip_handle=? AND post_id=?",
                (handle, post_id),
            )
        except Exception:
            pass

    @staticmethod
    def _format_telegram(feed, title, body, url, tickers, sentiment, posted_at):
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(sentiment, "⚪")
        snippet = title or body
        if len(snippet) > 380:
            snippet = snippet[:380] + "…"
        ticker_line = (f"\n<b>Tickers:</b> {' '.join('$'+t for t in tickers)}"
                       if tickers else "")
        when = f"\n<b>Posted:</b> {posted_at}" if posted_at else ""
        url_line = f"\n<a href=\"{url}\">{url}</a>" if url else ""
        return (
            f"{feed['telegram_prefix']}  {emoji}\n"
            f"<b>{feed['name']}</b>\n"
            f"<i>{snippet}</i>"
            f"{ticker_line}{when}{url_line}"
        )

    # ── read helpers for dashboard ─────────────────────────────────────────────
    def get_recent(self, limit: int = 50, ticker_only: bool = False) -> list:
        try:
            sql = ("SELECT vip_handle, vip_name, posted_at, fetched_at, url, "
                   "title, text, tickers, sentiment, alerted "
                   "FROM vip_posts ")
            if ticker_only:
                sql += "WHERE tickers <> '' AND tickers IS NOT NULL "
            sql += "ORDER BY fetched_at DESC LIMIT ?"
            rows = self.conn.execute(sql, (int(limit),)).fetchall()
        except Exception:
            return []
        out = []
        for r in rows:
            def g(k):
                return r.get(k) if hasattr(r, "get") else None
            out.append({
                "vip_handle":  g("vip_handle") or "",
                "vip_name":    g("vip_name") or "",
                "posted_at":   g("posted_at") or "",
                "fetched_at":  g("fetched_at") or "",
                "url":         g("url") or "",
                "title":       g("title") or "",
                "text":        g("text") or "",
                "tickers":     [t for t in (g("tickers") or "").split(",") if t],
                "sentiment":   g("sentiment") or "neutral",
                "alerted":     bool(g("alerted")),
            })
        return out


if __name__ == "__main__":
    # Standalone smoke test (no DB writes — just shows what would be detected).
    print("Probing VIP feeds + extracting tickers (no alerts sent)…\n")
    for f in VIP_FEEDS:
        entries = VipNewsMonitor._fetch(f["rss"])
        print(f"── {f['name']} ({f['handle']}) — {len(entries)} entries")
        for e in entries[:4]:
            title = _strip_html(e.get("title", ""))[:120]
            body  = _strip_html(e.get("summary", ""))[:120]
            tk    = extract_tickers(f"{title}\n{body}")
            sent  = classify_sentiment(f"{title}\n{body}")
            tag   = f" [{','.join(tk)}]" if tk else ""
            print(f"  {sent:>7}  {title!r}{tag}")
        print()
