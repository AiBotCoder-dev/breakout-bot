#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_providers.py — Unified live data layer with provider fallback chain.
==========================================================================
Wraps multiple market data providers behind a single API.  Tries the best-
quality provider first and falls back automatically if unavailable.

PRIMARY PROVIDERS:
  1. Alpaca Markets    — live IEX feed, official API, 200 req/min free
                         Set ALPACA_API_KEY + ALPACA_API_SECRET env vars
                         Sign up: https://alpaca.markets/
  2. yfinance          — final fallback (rate-limited on cloud, often slow)

SIGNAL PROVIDERS:
  TradingView TA       — pulls TradingView's BUY/SELL recommendation
                         No API key needed (uses public scraping)
                         pip install tradingview-ta

All functions return empty dicts/None on failure — callers should handle
missing data gracefully rather than treat absence as zero.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

try:
    import requests
except ImportError:                                                     # pragma: no cover
    requests = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# ALPACA MARKETS — official live US equity data
# ══════════════════════════════════════════════════════════════════════════════

_ALPACA_BASE = "https://data.alpaca.markets"


def _alpaca_creds() -> tuple[str, str]:
    """Read Alpaca credentials from environment (or Streamlit secrets in the app)."""
    key    = os.environ.get("ALPACA_API_KEY",    "").strip()
    secret = os.environ.get("ALPACA_API_SECRET", "").strip()
    return key, secret


def alpaca_available() -> bool:
    """True iff Alpaca credentials are set and requests is importable."""
    key, secret = _alpaca_creds()
    return bool(key and secret and requests is not None)


def _alpaca_headers() -> dict:
    key, secret = _alpaca_creds()
    return {
        "APCA-API-KEY-ID":     key,
        "APCA-API-SECRET-KEY": secret,
    }


def get_live_prices_alpaca(tickers: tuple) -> dict:
    """Bulk fetch latest trade prices for *tickers*.  Returns {ticker: price}.

    Uses Alpaca's snapshot endpoint which returns latest trade, quote, and
    daily bar in a single call.  Falls back through (trade → quote mid →
    dailyBar close) so a missing trade doesn't fail the whole symbol.
    """
    if not alpaca_available() or not tickers:
        return {}
    try:
        url = f"{_ALPACA_BASE}/v2/stocks/snapshots"
        resp = requests.get(
            url,
            headers=_alpaca_headers(),
            params={"symbols": ",".join(t.upper() for t in tickers)},
            timeout=15,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json() or {}
        prices: dict = {}
        for tk, snap in data.items():
            if not isinstance(snap, dict):
                continue
            try:
                # Priority: latestTrade.p → midpoint of latestQuote → dailyBar.c
                p = None
                lt = snap.get("latestTrade") or {}
                if lt.get("p"):
                    p = float(lt["p"])
                else:
                    lq = snap.get("latestQuote") or {}
                    if lq.get("bp") and lq.get("ap"):
                        p = (float(lq["bp"]) + float(lq["ap"])) / 2
                    else:
                        db = snap.get("dailyBar") or {}
                        if db.get("c"):
                            p = float(db["c"])
                if p and p > 0:
                    prices[tk.upper()] = p
            except Exception:
                pass
        return prices
    except Exception:
        return {}


def get_alpaca_bars(ticker: str, timeframe: str = "1Day",
                    days: int = 30) -> list:
    """Return list of OHLCV bars from Alpaca for a single ticker.

    timeframe accepts: 1Min, 5Min, 15Min, 1Hour, 1Day.
    Returns [] on failure.
    """
    if not alpaca_available() or not ticker:
        return []
    try:
        from datetime import datetime, timedelta, timezone
        end   = datetime.now(timezone.utc) - timedelta(minutes=16)   # IEX free has 15-min lag
        start = end - timedelta(days=days)
        resp = requests.get(
            f"{_ALPACA_BASE}/v2/stocks/{ticker.upper()}/bars",
            headers=_alpaca_headers(),
            params={
                "timeframe": timeframe,
                "start":     start.isoformat().replace("+00:00", "Z"),
                "end":       end.isoformat().replace("+00:00", "Z"),
                "limit":     1000,
                "adjustment": "raw",
                "feed":      "iex",
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return []
        return (resp.json() or {}).get("bars", []) or []
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED LIVE-PRICE FETCH — Alpaca first, yfinance fallback
# ══════════════════════════════════════════════════════════════════════════════

def get_live_prices(tickers: tuple, allow_yfinance_fallback: bool = True) -> dict:
    """Best-effort fetch of latest prices for *tickers*.

    Tries Alpaca first (if configured), then falls back to yfinance for any
    missing tickers.  Returns {ticker: price} — missing tickers are simply
    absent from the result so callers can flag them honestly.

    Performance: a typical call with both providers takes ~0.5s for 10
    tickers when Alpaca succeeds; ~3s when only yfinance is available.
    """
    if not tickers:
        return {}

    tickers = tuple(t.upper() for t in tickers if t)
    prices: dict = {}

    # ── Strategy 1: Alpaca (preferred) ────────────────────────────────────────
    if alpaca_available():
        prices.update(get_live_prices_alpaca(tickers))

    # ── Strategy 2: yfinance for whatever is missing ──────────────────────────
    missing = tuple(t for t in tickers if t not in prices)
    if missing and allow_yfinance_fallback:
        prices.update(_get_live_prices_yfinance(missing))

    return prices


def _get_live_prices_yfinance(tickers: tuple) -> dict:
    """yfinance fallback with the same 4-strategy chain used in app.py."""
    if not tickers:
        return {}
    prices: dict = {}
    remaining = set(tickers)

    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return prices

    def _extract(raw, target_tickers):
        out: dict = {}
        if raw is None or getattr(raw, "empty", True):
            return out
        if isinstance(raw.columns, pd.MultiIndex):
            _price_labels = {"Open", "High", "Low", "Close", "Volume"}
            lvl = 0 if _price_labels & set(raw.columns.get_level_values(0)) else 1
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(lvl)
            if isinstance(raw.columns, pd.MultiIndex):
                try:
                    raw = raw.xs("Close", axis=1, level=0)
                except Exception:
                    return out
        if isinstance(raw, pd.DataFrame):
            if "Close" in raw.columns:
                s = raw["Close"].dropna()
                if not s.empty and len(target_tickers) == 1:
                    out[next(iter(target_tickers))] = float(s.iloc[-1])
            else:
                for tk in target_tickers:
                    if tk in raw.columns:
                        s = raw[tk].dropna()
                        if not s.empty:
                            out[tk] = float(s.iloc[-1])
        elif isinstance(raw, pd.Series):
            s = raw.dropna()
            if not s.empty and target_tickers:
                out[next(iter(target_tickers))] = float(s.iloc[-1])
        return out

    # 5-min bars
    try:
        raw = yf.download(list(remaining), period="2d", interval="5m",
                          progress=False, auto_adjust=True, threads=True)
        prices.update(_extract(raw, remaining))
        remaining -= set(prices.keys())
    except Exception:
        pass

    # 1-day bars
    if remaining:
        try:
            raw = yf.download(list(remaining), period="5d", interval="1d",
                              progress=False, auto_adjust=True, threads=True)
            prices.update(_extract(raw, remaining))
            remaining -= set(prices.keys())
        except Exception:
            pass

    # Per-ticker fast_info
    for tk in list(remaining):
        try:
            v = getattr(yf.Ticker(tk).fast_info, "last_price", None)
            if v and float(v) > 0:
                prices[tk] = float(v)
        except Exception:
            pass

    return prices


# ══════════════════════════════════════════════════════════════════════════════
# TRADINGVIEW TA — pulls TradingView's BUY/SELL recommendation
# ══════════════════════════════════════════════════════════════════════════════

_TV_AVAILABLE: Optional[bool] = None
# Cache TV signals per session — they only change once a day
_TV_CACHE: Dict[str, Tuple[float, dict]] = {}
_TV_CACHE_TTL = 600   # 10 minutes


def tradingview_ta_available() -> bool:
    """True iff the tradingview-ta package is importable."""
    global _TV_AVAILABLE
    if _TV_AVAILABLE is not None:
        return _TV_AVAILABLE
    try:
        import tradingview_ta  # noqa
        _TV_AVAILABLE = True
    except ImportError:
        _TV_AVAILABLE = False
    return _TV_AVAILABLE


def get_tradingview_signal(ticker: str,
                            interval: str = "1d") -> dict:
    """Pull TradingView's technical analysis for *ticker*.

    Tries NASDAQ then NYSE then AMEX automatically — most US tickers are
    on one of these three.  Caches results for 10 min per ticker.

    interval: '1m', '5m', '15m', '1h', '4h', '1d', '1W', '1M'

    Returns dict:
      recommendation       — STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
      buy_count, sell_count, neutral_count
      ma_recommendation    — MA-only verdict
      osc_recommendation   — oscillator-only verdict
      exchange             — where the symbol was found
      score                — normalized -10 (STRONG_SELL) … +10 (STRONG_BUY)
    Returns {} if unavailable or all exchanges failed.
    """
    if not tradingview_ta_available() or not ticker:
        return {}

    ticker = ticker.upper().strip()
    cache_key = f"{ticker}:{interval}"

    # Cache hit
    if cache_key in _TV_CACHE:
        ts, val = _TV_CACHE[cache_key]
        if (time.time() - ts) < _TV_CACHE_TTL:
            return val

    try:
        from tradingview_ta import TA_Handler, Interval as TVI
    except ImportError:
        return {}

    _interval_map = {
        "1m":  TVI.INTERVAL_1_MINUTE,
        "5m":  TVI.INTERVAL_5_MINUTES,
        "15m": TVI.INTERVAL_15_MINUTES,
        "1h":  TVI.INTERVAL_1_HOUR,
        "4h":  TVI.INTERVAL_4_HOURS,
        "1d":  TVI.INTERVAL_1_DAY,
        "1W":  TVI.INTERVAL_1_WEEK,
        "1M":  TVI.INTERVAL_1_MONTH,
    }
    tv_interval = _interval_map.get(interval, TVI.INTERVAL_1_DAY)

    _SCORE_MAP = {
        "STRONG_BUY":  10, "BUY":   5,
        "NEUTRAL":      0,
        "SELL":        -5, "STRONG_SELL": -10,
    }

    for exch in ("NASDAQ", "NYSE", "AMEX"):
        try:
            handler = TA_Handler(
                symbol   = ticker,
                screener = "america",
                exchange = exch,
                interval = tv_interval,
            )
            a = handler.get_analysis()
            if not a:
                continue
            rec = a.summary.get("RECOMMENDATION", "NEUTRAL")
            out = {
                "recommendation":      rec,
                "buy_count":           int(a.summary.get("BUY",     0)),
                "sell_count":          int(a.summary.get("SELL",    0)),
                "neutral_count":       int(a.summary.get("NEUTRAL", 0)),
                "ma_recommendation":   (a.moving_averages or {}).get("RECOMMENDATION", "?"),
                "osc_recommendation":  (a.oscillators or {}).get("RECOMMENDATION", "?"),
                "exchange":            exch,
                "score":               int(_SCORE_MAP.get(rec, 0)),
                "interval":            interval,
            }
            _TV_CACHE[cache_key] = (time.time(), out)
            return out
        except Exception:
            continue

    # All exchanges failed — cache the negative result briefly to avoid retry storms
    _TV_CACHE[cache_key] = (time.time(), {})
    return {}


def get_tradingview_multiplier(ticker: str, interval: str = "1d") -> float:
    """Convert TradingView signal into a Master Score multiplier.

    Maps:
      STRONG_BUY   → 1.10x   (bonus 10 %)
      BUY          → 1.05x
      NEUTRAL      → 1.00x
      SELL         → 0.92x
      STRONG_SELL  → 0.85x
      No data      → 1.00x   (no effect)
    """
    sig = get_tradingview_signal(ticker, interval=interval)
    if not sig:
        return 1.0
    return {
        "STRONG_BUY":   1.10,
        "BUY":          1.05,
        "NEUTRAL":      1.00,
        "SELL":         0.92,
        "STRONG_SELL":  0.85,
    }.get(sig.get("recommendation", "NEUTRAL"), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: provider availability summary
# ══════════════════════════════════════════════════════════════════════════════

def provider_status() -> dict:
    """Return a snapshot of which providers are configured + their last test result."""
    return {
        "alpaca":        alpaca_available(),
        "tradingview":   tradingview_ta_available(),
        "yfinance":      True,   # always available as fallback
    }
