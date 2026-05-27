#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
early_momentum.py — Pre-Explosion Detector
============================================
Catches stocks in the 1-3 day window BEFORE they explode, not after.

The Whale Intel + Squeeze Scanner detect setups when signals are at PEAK
(StockTwits 90%+, BB width in bottom 15%, etc.) — by then, the move is
already in progress.  This scanner detects the BUILDUP phase:

  • StockTwits velocity going 1× → 2× → 3× (not yet 10×)
  • BB width declining for 3+ days (not yet bottom 15%)
  • Volume base building (rising volume on flat price)
  • Multi-TF starting to align (1H crossed, daily approaching)
  • Short interest velocity (covering starting)
  • Fresh catalyst (news <48h with impact ≥ 7)

Each signal individually is noisy.  The edge comes from CLUSTERS — when
3+ of these fire on the same ticker, the probability of an imminent move
is high.  The composite Early Momentum Score gates entries via a
multiplier in compute_master_score, AND surfaces a "Pre-Explosion
Watchlist" in the Management dashboard.

DESIGN:
  • All signals are LEADING indicators (predictive, not coincident)
  • Reuses the shared history cache so no extra yfinance hits
  • Scores 0-100 with 4 tiers: WATCH / BUILDING / IMMINENT / ACTIVE
  • Zero PC load — runs in monitor.py cycle
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# EARLY MOMENTUM SCANNER
# ══════════════════════════════════════════════════════════════════════════════

class EarlyMomentumScanner:
    """6-signal pre-explosion detector with categorized output."""

    # Tier thresholds
    _TIER_IMMINENT = 70    # 70-100 → IMMINENT (move within 1-2 days likely)
    _TIER_BUILDING = 50    # 50-69  → BUILDING (3-5 day setup forming)
    _TIER_WATCH    = 30    # 30-49  → WATCH (early warning, monitor)

    def __init__(self, conn=None, ai_analyst=None):
        self.conn = conn
        self.ai   = ai_analyst

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 1 — Coiled Spring (BB width declining trend)
    # ─────────────────────────────────────────────────────────────────────────
    def detect_coiled_spring(self, ticker: str) -> dict:
        """BB width DECREASING for 3+ days AND in bottom 30% (but not yet 15%).

        Catches the compression-building phase before extreme squeeze fires.
        Max 18 pts.
        """
        try:
            import trading_scanner as ts
            hist = ts.get_cached_history(ticker, period="1y", interval="1d")
            hist = hist.dropna(subset=["Close"]) if not hist.empty else hist
            if len(hist) < 50:
                return {"score": 0, "label": "Not enough data"}

            close = hist["Close"]
            sma   = close.rolling(20).mean()
            std   = close.rolling(20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            bb_width = ((upper - lower) / sma * 100).dropna().tail(180)

            current  = float(bb_width.iloc[-1])
            pct_rank = float(bb_width.rank(pct=True).iloc[-1]) * 100

            # Count consecutive days of declining BB width
            days_declining = 0
            recent = bb_width.tail(10).values
            for i in range(len(recent) - 1, 0, -1):
                if recent[i] < recent[i-1]:
                    days_declining += 1
                else:
                    break

            score = 0
            label = ""
            if 15 < pct_rank <= 30 and days_declining >= 3:
                score = 18
                label = f"🌀 Coiling: {days_declining}d declining, rank {pct_rank:.0f}"
            elif 15 < pct_rank <= 40 and days_declining >= 2:
                score = 12
                label = f"Coiling: {days_declining}d declining, rank {pct_rank:.0f}"
            elif pct_rank <= 30:
                score = 8
                label = f"Pre-compressed (rank {pct_rank:.0f})"
            else:
                label = f"BB width rank {pct_rank:.0f} — normal"

            return {
                "score":          score,
                "label":          label,
                "pct_rank":       round(pct_rank, 1),
                "days_declining": days_declining,
                "bb_width":       round(current, 2),
            }
        except Exception:
            return {"score": 0, "label": "BB analysis failed"}

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 2 — Volume Base Building
    # ─────────────────────────────────────────────────────────────────────────
    def detect_volume_base(self, ticker: str) -> dict:
        """Rising volume on flat price = institutional accumulation.

        Sign: 5-day avg vol > 20-day avg by 30%+, AND price std/avg <= 3%.
        Max 18 pts.
        """
        try:
            import trading_scanner as ts
            hist = ts.get_cached_history(ticker, period="2mo", interval="1d")
            hist = hist.dropna(subset=["Close", "Volume"]) if not hist.empty else hist
            if len(hist) < 25:
                return {"score": 0, "label": "Not enough data"}

            vol = hist["Volume"]
            close = hist["Close"]
            vol_5  = float(vol.tail(5).mean())
            vol_20 = float(vol.tail(20).mean())
            vol_ratio = vol_5 / vol_20 if vol_20 else 1.0

            price_5  = close.tail(5)
            price_std_pct = (float(price_5.std()) / float(price_5.mean()) * 100
                              if float(price_5.mean()) else 100)

            score = 0
            label = ""
            if vol_ratio >= 1.5 and price_std_pct <= 3.0:
                score = 18
                label = f"💼 Strong accumulation: vol {vol_ratio:.2f}× on flat price"
            elif vol_ratio >= 1.3 and price_std_pct <= 4.0:
                score = 12
                label = f"Accumulation: vol {vol_ratio:.2f}× on stable price"
            elif vol_ratio >= 1.2:
                score = 6
                label = f"Volume building: vol {vol_ratio:.2f}× avg"
            else:
                label = f"Vol {vol_ratio:.2f}× avg — no clear base"

            return {
                "score":         score,
                "label":         label,
                "vol_ratio":     round(vol_ratio, 2),
                "price_stable":  round(price_std_pct, 2),
            }
        except Exception:
            return {"score": 0, "label": "Volume analysis failed"}

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 3 — StockTwits Velocity Acceleration
    # ─────────────────────────────────────────────────────────────────────────
    def detect_velocity_acceleration(self, ticker: str) -> dict:
        """Message rate accelerating but not yet at peak.

        Catches viral builds — going from 1× to 3× normal (BEFORE 10× peak).
        Max 18 pts.
        """
        try:
            from whale_intelligence import StockTwitsTracker
            st = StockTwitsTracker().fetch(ticker)
            n_msgs = st.get("n_messages", 0)
            vel    = float(st.get("velocity_ratio", 1) or 1)
            bull   = float(st.get("bull_pct", 0.5) or 0.5)

            score = 0
            label = ""
            # Sweet spot — velocity climbing but not yet peaked
            if 2.5 <= vel <= 6 and bull >= 0.65:
                score = 18
                label = f"🚀 Velocity surging: {vel:.1f}× normal, {bull*100:.0f}% bull"
            elif 1.7 <= vel < 2.5 and bull >= 0.6:
                score = 12
                label = f"Velocity building: {vel:.1f}× normal, {bull*100:.0f}% bull"
            elif 1.4 <= vel < 1.7:
                score = 6
                label = f"Early chatter: {vel:.1f}× normal"
            elif vel > 6:
                # Already exploded — caught too late
                score = 4
                label = f"⚠️ Already viral ({vel:.1f}× — likely late)"
            else:
                label = f"Velocity {vel:.1f}× — quiet"

            return {
                "score":     score,
                "label":     label,
                "velocity":  vel,
                "bull_pct":  bull,
                "n_msgs":    n_msgs,
            }
        except Exception:
            return {"score": 0, "label": "Velocity check failed"}

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 4 — Multi-Timeframe Transitioning
    # ─────────────────────────────────────────────────────────────────────────
    def detect_mtf_transition(self, ticker: str) -> dict:
        """Captures the 1H first → Daily approaching → Weekly to follow flow.

        Best entry is when 1H and Daily are aligned but Weekly hasn't caught
        up yet (the trend is just starting at intermediate timeframe).
        Max 15 pts.
        """
        try:
            import trading_scanner as ts
            mtf = ts.confirm_multi_timeframe(ticker, bias="bullish")
            if not mtf:
                return {"score": 0, "label": "MTF unavailable"}

            tfs = mtf.get("timeframes", {})
            h1 = tfs.get("1H",     {}).get("aligned", False)
            d1 = tfs.get("Daily",  {}).get("aligned", False)
            w1 = tfs.get("Weekly", {}).get("aligned", False)

            score = 0
            label = ""
            if h1 and d1 and not w1:
                score = 15
                label = "🔄 1H+Daily aligned, Weekly catching up — early trend"
            elif h1 and not d1 and not w1:
                score = 10
                label = "🔄 1H just crossed — earliest signal"
            elif h1 and d1 and w1:
                score = 8
                label = "Full alignment (3/3) — trend established"
            elif not h1:
                label = "1H not aligned yet"
            else:
                score = 4
                label = "Partial alignment"

            return {
                "score":       score,
                "label":       label,
                "aligned":     mtf.get("aligned", 0),
            }
        except Exception:
            return {"score": 0, "label": "MTF check failed"}

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 5 — Short Covering Started
    # ─────────────────────────────────────────────────────────────────────────
    def detect_short_covering(self, ticker: str) -> dict:
        """Days-to-cover dropping while shares short remains high = squeeze
        starting.  Max 15 pts."""
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            short_pct = float(info.get("shortPercentOfFloat") or 0) * 100
            days_to_cover = float(info.get("shortRatio") or 0)
            shares_short = int(info.get("sharesShort") or 0)
            shares_short_prior = int(info.get("sharesShortPriorMonth") or 0)

            score = 0
            label = ""

            # Decrease in shares_short month-over-month = covering
            if shares_short_prior > 0 and shares_short > 0:
                change_pct = (shares_short - shares_short_prior) / shares_short_prior * 100
            else:
                change_pct = 0

            # High short% + recent covering = ideal squeeze fuel
            if short_pct >= 20 and change_pct < -5:
                score = 15
                label = f"🔥 Squeeze loading: {short_pct:.0f}% short, covering {change_pct:.1f}% MoM"
            elif short_pct >= 15 and days_to_cover >= 3:
                score = 12
                label = f"Squeeze fuel: {short_pct:.0f}% short, {days_to_cover:.1f}d cover"
            elif short_pct >= 10:
                score = 6
                label = f"Moderate short: {short_pct:.0f}%"
            else:
                label = f"Low short interest ({short_pct:.0f}%)"

            return {
                "score":          score,
                "label":          label,
                "short_pct":      round(short_pct, 1),
                "days_to_cover":  round(days_to_cover, 1),
                "covering_pct":   round(change_pct, 1),
            }
        except Exception:
            return {"score": 0, "label": "Short data unavailable"}

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 6 — Fresh Catalyst
    # ─────────────────────────────────────────────────────────────────────────
    def detect_fresh_catalyst(self, ticker: str) -> dict:
        """News in last 48h with impact >= 7 = recent catalyst → momentum
        ahead.  Max 16 pts."""
        if self.conn is None:
            return {"score": 0, "label": "No DB connection"}
        try:
            from news_agent import NewsAgent
            na = NewsAgent(self.conn, ai_analyst=None)
            news = na.get_recent_news_for_ticker(ticker, hours=48, limit=10)

            if not news:
                return {"score": 0, "label": "No recent news"}

            # Find the highest-impact positive event
            best_impact = 0
            best_headline = ""
            best_sentiment = ""
            for n in news:
                try:
                    imp = int(n.get("impact_score", 0) or 0)
                    if imp > best_impact:
                        best_impact = imp
                        best_headline = str(n.get("headline", ""))[:80]
                        best_sentiment = str(n.get("sentiment", ""))
                except Exception:
                    continue

            score = 0
            label = ""
            if best_impact >= 8 and best_sentiment == "POSITIVE":
                score = 16
                label = f"📰 HIGH catalyst (impact {best_impact}): \"{best_headline[:60]}\""
            elif best_impact >= 6 and best_sentiment == "POSITIVE":
                score = 10
                label = f"📰 Catalyst (impact {best_impact}): \"{best_headline[:60]}\""
            elif best_impact >= 5:
                score = 5
                label = f"Recent news (impact {best_impact})"
            else:
                label = f"News present, low impact"

            return {
                "score":      score,
                "label":      label,
                "n_news":     len(news),
                "max_impact": best_impact,
            }
        except Exception:
            return {"score": 0, "label": "News check failed"}

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN SCANNER
    # ─────────────────────────────────────────────────────────────────────────
    def scan(self, ticker: str) -> dict:
        """Run all 6 detectors and return composite Early Momentum Score."""
        signals = {
            "coiled_spring":    self.detect_coiled_spring(ticker),
            "volume_base":      self.detect_volume_base(ticker),
            "velocity":         self.detect_velocity_acceleration(ticker),
            "mtf_transition":   self.detect_mtf_transition(ticker),
            "short_covering":   self.detect_short_covering(ticker),
            "fresh_catalyst":   self.detect_fresh_catalyst(ticker),
        }
        total = sum(s.get("score", 0) for s in signals.values())
        n_firing = sum(1 for s in signals.values() if s.get("score", 0) >= 6)

        # Tier
        if total >= self._TIER_IMMINENT:
            tier = "IMMINENT"
            tier_emoji = "🚨"
            outlook = "Move within 1-2 days highly likely"
        elif total >= self._TIER_BUILDING:
            tier = "BUILDING"
            tier_emoji = "🔥"
            outlook = "Setup forming — 3-5 day window"
        elif total >= self._TIER_WATCH:
            tier = "WATCH"
            tier_emoji = "👀"
            outlook = "Early signals — add to watchlist"
        else:
            tier = "QUIET"
            tier_emoji = "⚪"
            outlook = "No clear pre-explosion setup"

        return {
            "ticker":        ticker.upper(),
            "score":         total,
            "tier":          tier,
            "tier_emoji":    tier_emoji,
            "n_signals":     n_firing,
            "outlook":       outlook,
            "signals":       signals,
            "max_possible":  100,
        }

    def scan_universe(self, tickers: list, min_score: int = 30) -> list:
        """Batch scan a list of tickers, return sorted by score descending."""
        results = []
        for tk in tickers:
            try:
                r = self.scan(tk)
                if r["score"] >= min_score:
                    results.append(r)
            except Exception:
                continue
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def early_momentum_multiplier(self, ticker: str) -> dict:
        """Score → Master Score multiplier mapping.

        IMMINENT (70+) → 1.10×    BUILDING (50-69) → 1.06×
        WATCH (30-49)  → 1.02×    QUIET (<30)      → 1.00×
        """
        r = self.scan(ticker)
        s = r["score"]
        if s >= self._TIER_IMMINENT:
            mult = 1.10
        elif s >= self._TIER_BUILDING:
            mult = 1.06
        elif s >= self._TIER_WATCH:
            mult = 1.02
        else:
            mult = 1.00
        return {
            "multiplier":     mult,
            "score":          s,
            "tier":           r["tier"],
            "n_signals":      r["n_signals"],
            "summary":        f"Early Momentum {s}/100 ({r['tier']}) → {mult:.2f}×",
            "raw":            r,
        }
