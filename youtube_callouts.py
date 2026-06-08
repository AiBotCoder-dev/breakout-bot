"""
youtube_callouts.py — Pull option callouts from finance YouTube videos (transcripts).

The honest version of "watch YouTube": we can't read on-screen charts/text like an
image, but most finance videos have CAPTIONS. We pull the transcript (spoken
words, as text) for free — no video download, no API key — and parse it for
ticker/option callouts with the same engine used for social callouts.

PIPELINE (all free, no key):
  1. Channel RSS  https://www.youtube.com/feeds/videos.xml?channel_id=UC...  -> recent video IDs
  2. youtube-transcript-api -> the transcript text
  3. parse_callout + spoken-ticker detection -> callouts tagged source=youtube/<channel>
  4. routed into OptionsCalloutTracker -> the SAME win-rate gate (observation-only
     until a channel proves >=55% win over >=20 closed callouts)

HONEST LIMITS:
  • SPOKEN words only — NOT what's shown on screen (charts, tickers in graphics).
  • Auto-caption NOISE — "Nvidia" not "$NVDA", numbers garbled -> lower parse
    accuracy than text posts. We map company NAMES to tickers to compensate.
  • CLOUD IP BLOCKS — youtube-transcript-api is often rate-limited/blocked from
    datacenter IPs (Streamlit Cloud / GitHub Actions). Works locally; may be
    flaky on cloud. Treat as best-effort.
  • LIVE streams have no transcript; skipped automatically.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

try:
    import requests
    import feedparser
except Exception:                       # pragma: no cover
    requests = feedparser = None

# Reuse the callout parser + spoken-ticker map already in the project
from options_callouts import parse_callout
from vip_news_monitor import COMPANY_TICKERS, CASHTAG_RE, classify_sentiment


# Default finance channels (channel IDs). Add your own via YOUTUBE_CHANNELS env
# (comma-separated channel IDs and/or @handles). Channels you TRUST — the
# win-rate gate will filter the rest anyway.
DEFAULT_CHANNELS = [
    "UCEAZeUIeJs0IjQiqTCdVSIg",   # Yahoo Finance
    "UCrp_UI8XtuYfpiqluWLD7Lw",   # CNBC Television
    "UCIALMKvObZNtJ6AmdCLP7Lg",   # Bloomberg Television
]

_BLACKLIST = {"USD","USA","US","CEO","AI","GDP","ETF","IPO","SEC","FED","NYSE",
              "NASDAQ","DOW","SPY","CPI","PCE","Q1","Q2","Q3","Q4"}


def _channels() -> list:
    env = os.environ.get("YOUTUBE_CHANNELS", "").strip()
    if env:
        return [c.strip() for c in env.split(",") if c.strip()]
    return DEFAULT_CHANNELS


def resolve_channel_id(handle_or_id: str) -> str | None:
    """A 'UC...' id is used directly; an @handle is resolved by scraping the page."""
    s = handle_or_id.strip()
    if s.startswith("UC") and len(s) >= 20:
        return s
    if requests is None:
        return None
    try:
        url = f"https://www.youtube.com/{s if s.startswith('@') else '@'+s}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        m = re.search(r'"channelId":"(UC[\w-]{20,})"', r.text)
        return m.group(1) if m else None
    except Exception:
        return None


def recent_videos(channel_id: str, max_videos: int = 5) -> list:
    """Recent uploads via the free channel RSS feed. Returns [{id, title, published}]."""
    if requests is None or feedparser is None:
        return []
    try:
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code != 200:
            return []
        feed = feedparser.parse(r.content)
        out = []
        chan = (feed.feed.get("title") if hasattr(feed, "feed") else "") or channel_id
        for e in feed.entries[:max_videos]:
            vid = e.get("yt_videoid") or (e.get("id", "").split(":")[-1])
            if vid:
                out.append({"id": vid, "title": e.get("title", ""),
                            "published": e.get("published", ""), "channel": chan})
        return out
    except Exception:
        return []


def fetch_transcript(video_id: str) -> str | None:
    """Transcript text for a video, or None (live/no-captions/blocked)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        return None
    try:
        api = YouTubeTranscriptApi()
        tr = api.fetch(video_id)
        segs = tr.snippets if hasattr(tr, "snippets") else tr
        return " ".join((s.text if hasattr(s, "text") else s.get("text", ""))
                        for s in segs)
    except Exception:
        return None


def _spoken_tickers(text: str) -> list:
    """Detect tickers in spoken transcript: cashtags + company-name mentions."""
    out = set()
    for m in CASHTAG_RE.findall(text or ""):
        if m.isalpha() and 1 <= len(m) <= 5 and m not in _BLACKLIST:
            out.add(m.upper())
    low = (text or "").lower()
    for name in sorted(COMPANY_TICKERS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", low):
            t = COMPANY_TICKERS[name]
            if t not in _BLACKLIST and t.isalpha():
                out.add(t)
    return sorted(out)


def _extract_callouts(text: str, source: str, channel: str, title: str) -> list:
    """Parse a transcript into callout dicts (option-pattern + spoken-ticker)."""
    callouts = []
    # Sentence-ish chunks so a parsed strike/expiry stays near its ticker
    chunks = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    for ch in chunks:
        if len(ch) < 8:
            continue
        # explicit option pattern (e.g. "buying the Nvidia 200 calls")
        c = parse_callout(ch)
        if c and c.get("ticker"):
            key = (c["ticker"], c.get("option_type"), c.get("strike"), c.get("expiry"))
            if key in seen:
                continue
            seen.add(key)
            c.update({"source": source, "username": channel,
                      "created_at": datetime.utcnow().isoformat(),
                      "raw_text": (title + " :: " + ch)[:300]})
            callouts.append(c)
    # Lighter signal: spoken ticker + bullish/bearish context (no strike) — only
    # if no explicit option callout already covered it.
    if not callouts:
        sent = classify_sentiment(text[:4000])
        if sent in ("bullish", "bearish"):
            for tk in _spoken_tickers(text)[:3]:
                callouts.append({
                    "ticker": tk,
                    "option_type": "call" if sent == "bullish" else "put",
                    "strike": None, "expiry": None,
                    "source": source, "username": channel,
                    "created_at": datetime.utcnow().isoformat(),
                    "raw_text": (title + " :: spoken " + sent)[:300],
                })
    return callouts


def fetch_youtube_callouts(max_videos_per_channel: int = 4) -> list:
    """Scan configured channels' recent videos and return parsed callout dicts."""
    all_callouts = []
    for ch in _channels():
        cid = resolve_channel_id(ch)
        if not cid:
            continue
        for vid in recent_videos(cid, max_videos_per_channel):
            title = (vid.get("title") or "").lower()
            if "live" in title and ("begins" in title or "stream" in title):
                continue   # skip obvious live streams
            text = fetch_transcript(vid["id"])
            if not text or len(text) < 200:
                continue
            chan = vid.get("channel") or cid
            src = "youtube/" + re.sub(r"[^A-Za-z0-9]+", "_", chan)[:24].strip("_").lower()
            all_callouts.extend(
                _extract_callouts(text, src, chan, vid.get("title", "")))
    return all_callouts


if __name__ == "__main__":
    print("Scanning finance YouTube channels for spoken option callouts...\n")
    print("NOTE: spoken transcript only (not on-screen charts); auto-caption noise;")
    print("routed through the win-rate gate (observation-only until proven).\n")
    cs = fetch_youtube_callouts(max_videos_per_channel=3)
    print(f"Parsed {len(cs)} callouts from YouTube transcripts:")
    for c in cs[:15]:
        st = f" ${c['strike']:.0f}{(c.get('option_type') or '?')[0].upper()}" if c.get("strike") else f" {c.get('option_type')}"
        print(f"  {c['ticker']:6s}{st}  [{c['source']}]  {c['raw_text'][:60]}")
