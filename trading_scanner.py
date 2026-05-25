#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trading_scanner.py — Professional Stock Breakout Scanner
Identifies S&P 500 stocks with strong upside breakout potential using
10-pattern detection engines and a Bayesian probability scoring system.
"""

# ── UTF-8 stdout so emoji and box chars work on Windows ──────────────────────
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import io
import json
import os
import sqlite3
import uuid as _uuid
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Auto-install any missing packages ─────────────────────────────────────────
def _ensure(pkg, import_name=None):
    import importlib, subprocess
    try:
        importlib.import_module(import_name or pkg)
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"], check=True)


import time

for _pkg, _mod in [("yfinance", "yfinance"),
                   ("tqdm", "tqdm"), ("tabulate", "tabulate"),
                   ("scipy", "scipy"), ("numpy", "numpy"),
                   ("requests", "requests"), ("beautifulsoup4", "bs4"),
                   ("feedparser", "feedparser"), ("lxml", "lxml")]:
    _ensure(_pkg, _mod)

import yfinance as yf
try:
    import pandas_ta as ta
except Exception:
    ta = None
from scipy.signal import argrelextrema
from tabulate import tabulate
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False


# ══════════════════════════════════════════════════════════════════════════════
# MARKET CLOCK — real-time ET session tracker with optimal trading windows
# ══════════════════════════════════════════════════════════════════════════════
class MarketClock:
    """
    Tracks US market sessions and returns trading-window quality.

    Windows (all times Eastern):
      AVOID     09:30–09:45   Opening volatility — unpredictable, wide spreads
      PRIME     09:45–11:30   Best breakout resolution; institutions active
      CAUTION   11:30–13:00   Midday chop — low follow-through
      SECONDARY 13:00–14:30   Afternoon directional drift; still tradeable
      PRIME     14:30–15:45   Power hour — volume surge, clean momentum entries
      AVOID     15:45–16:00   MOC orders distort price — avoid new entries
      PREMARKET 04:00–09:30   Scan & plan only; no live paper-trade opens
      CLOSED    all other     Review and prepare for next session
    """

    # Each row: (start_h, start_m, end_h, end_m, quality, name, hex_color)
    _WINDOWS = [
        ( 0,  0,  4,  0, "CLOSED",     "Overnight",            "#30363d"),
        ( 4,  0,  9, 30, "PREMARKET",  "Pre-Market Setup",     "#6e40c9"),
        ( 9, 30,  9, 45, "AVOID",      "Opening Volatility",   "#f85149"),
        ( 9, 45, 11, 30, "PRIME",      "Prime Entry Window",   "#3fb950"),
        (11, 30, 13,  0, "CAUTION",    "Midday Chop",          "#e3b341"),
        (13,  0, 14, 30, "SECONDARY",  "Early Afternoon",      "#58a6ff"),
        (14, 30, 15, 45, "PRIME",      "Power Hour",           "#3fb950"),
        (15, 45, 16,  0, "AVOID",      "Closing Volatility",   "#f85149"),
        (16,  0, 20,  0, "CLOSED",     "After Hours",          "#30363d"),
        (20,  0, 24,  0, "CLOSED",     "Overnight",            "#30363d"),
    ]

    @classmethod
    def now_et(cls) -> "datetime":
        """Current time in US/Eastern, handling DST automatically."""
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo("America/New_York"))
        except ImportError:
            try:
                import pytz
                return datetime.now(pytz.timezone("America/New_York"))
            except ImportError:
                # rough fallback — UTC−5 (will be off by 1h during EDT)
                return datetime.utcnow() - timedelta(hours=5)

    @classmethod
    def get_session(cls) -> dict:
        """Return current session metadata including countdown and quality."""
        now = cls.now_et()

        # Weekend — find next Monday 9:45 AM ET
        if now.weekday() >= 5:
            days = (7 - now.weekday()) % 7 or 7
            nxt  = (now + timedelta(days=days)).replace(
                hour=9, minute=45, second=0, microsecond=0)
            secs = max(0, int((nxt - now).total_seconds()))
            return {
                "quality": "CLOSED", "name": "Weekend — Market Closed",
                "color": "#30363d", "remaining_seconds": secs,
                "next_prime_seconds": secs, "is_open": False, "time_et": now,
            }

        h, m  = now.hour, now.minute
        total = h * 60 + m

        for sh, sm, eh, em, quality, name, color in cls._WINDOWS:
            if sh * 60 + sm <= total < eh * 60 + em:
                # Handle the 24:00 sentinel ("end of day" — midnight rollover).
                # datetime.replace() only accepts hour 0–23, so we add 1 day
                # and set hour=0 instead.
                if eh >= 24:
                    end_dt = (now + timedelta(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0)
                else:
                    end_dt = now.replace(hour=eh, minute=em,
                                         second=0, microsecond=0)
                remaining = max(0, int((end_dt - now).total_seconds()))
                return {
                    "quality":            quality,
                    "name":               name,
                    "color":              color,
                    "remaining_seconds":  remaining,
                    "next_prime_seconds": cls._secs_to_next_prime(now),
                    "is_open":            quality in ("PRIME", "SECONDARY"),
                    "time_et":            now,
                }

        return {"quality": "CLOSED", "name": "Closed", "color": "#30363d",
                "remaining_seconds": 0, "next_prime_seconds": 0,
                "is_open": False, "time_et": now}

    @classmethod
    def _secs_to_next_prime(cls, now) -> int:
        h, m  = now.hour, now.minute
        total = h * 60 + m
        for sh, sm, eh, em, quality, *_ in cls._WINDOWS:
            if quality == "PRIME" and sh * 60 + sm > total:
                t = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
                return max(0, int((t - now).total_seconds()))
        # Roll to next trading day at 9:45
        for i in range(1, 8):
            nxt = now + timedelta(days=i)
            if nxt.weekday() < 5:
                t = nxt.replace(hour=9, minute=45, second=0, microsecond=0)
                return max(0, int((t - now).total_seconds()))
        return 0

    @classmethod
    def should_trade(cls) -> bool:
        return cls.get_session()["quality"] in ("PRIME", "SECONDARY")

    @classmethod
    def get_advice(cls) -> str:
        s   = cls.get_session()
        q   = s["quality"]
        rem = s.get("remaining_seconds", 0)
        h_, r  = divmod(rem, 3600)
        m_, sc = divmod(r,   60)
        timer  = (f"{h_}h {m_}m" if h_ else f"{m_}m {sc}s") if rem else ""
        if q == "PRIME":
            return f"✅ PRIME WINDOW — Best time to scan & trade. {timer} remaining."
        if q == "SECONDARY":
            return f"🟡 SECONDARY — Good for momentum plays. {timer} remaining."
        if q == "CAUTION":
            return f"⚠️ MIDDAY CHOP — Low follow-through. Avoid new entries. {timer} left."
        if q == "AVOID":
            return f"🔴 {s['name']} — High noise. Skip new entries. Clears in {timer}."
        if q == "PREMARKET":
            return f"🔵 PRE-MARKET — Good time to scan setups. Trade after 9:45 ET. {timer} to open."
        nxt = s.get("next_prime_seconds", 0)
        h2, r2 = divmod(nxt, 3600); m2, _ = divmod(r2, 60)
        nxt_str = f"{h2}h {m2}m" if h2 else f"{m2}m"
        return f"⚫ CLOSED — Review setups. Next prime window in ~{nxt_str}."


# ══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME DETECTOR — classifies SPY trend for adaptive signal thresholds
# ══════════════════════════════════════════════════════════════════════════════
class MarketRegimeDetector:
    """
    Classifies broad market environment using SPY daily price action.

    Regimes:
      STRONG BULL  — above 20/50/200d SMA, positive 20d return, low vol
      BULL         — above 50d + 200d SMA
      NEUTRAL      — mixed (above one, below other)
      RECOVERING   — above 50d but below 200d (bear-market rally)
      BEAR         — below both 50d and 200d SMA
    """

    @staticmethod
    def detect(spy_df: "pd.DataFrame") -> dict:
        if spy_df is None or len(spy_df) < 50:
            return {
                "regime": "UNKNOWN", "label": "Unknown", "color": "#8b949e",
                "score": 0, "advice": "Insufficient SPY data — using standard filters.",
                "above_50": True, "above_200": True,
                "dist_200_pct": 0.0, "ret_5d": 0.0, "ret_20d": 0.0, "hist_vol": 15.0,
            }

        C   = spy_df["Close"].values
        cur = float(C[-1])

        sma20  = float(C[-20:].mean()) if len(C) >= 20  else cur
        sma50  = float(C[-50:].mean()) if len(C) >= 50  else cur
        sma200 = float(C[-200:].mean()) if len(C) >= 200 else float(C.mean())

        above_20  = cur > sma20
        above_50  = cur > sma50
        above_200 = cur > sma200

        ret_5d  = float((cur - C[-6])  / C[-6])  if len(C) > 5  else 0.0
        ret_20d = float((cur - C[-21]) / C[-21]) if len(C) > 20 else 0.0
        dist200 = (cur - sma200) / sma200 * 100

        # 20-day annualised historical volatility
        if len(C) >= 21:
            lrets    = np.diff(np.log(np.clip(C[-21:], 1e-9, None)))
            hist_vol = float(np.std(lrets) * np.sqrt(252) * 100)
        else:
            hist_vol = 15.0

        # Bull/bear score: each confirmed factor +1, each failed factor -1
        score = sum([
            1 if above_200 else -1,
            1 if above_50  else -1,
            1 if above_20  else -1,
            1 if ret_20d > 0 else -1,
            1 if ret_5d  > 0 else -1,
        ])

        if score >= 4 and ret_20d > 0.02:
            regime, color, label = "STRONG BULL", "#3fb950", "Strong Bull"
            advice = "Strong uptrend. Standard scan parameters. Full position sizing."
        elif score >= 3:
            regime, color, label = "BULL", "#58a6ff", "Bull"
            advice = "Uptrend intact. Standard filters work well."
        elif score >= 1:
            regime, color, label = "NEUTRAL", "#e3b341", "Neutral"
            advice = "Mixed signals. Recommend prob ≥ 65% filter."
        elif score >= -1:
            regime, color, label = "RECOVERING", "#e3b341", "Recovering"
            advice = "Bear-market rally. High-quality setups only (prob ≥ 70%)."
        else:
            regime, color, label = "BEAR", "#f85149", "Bear"
            advice = "Bear market. Only RS leaders pass. Reduce position size."

        if hist_vol > 35:
            advice = f"⚡ HIGH VOL ({hist_vol:.0f}% ann.) — " + advice

        return {
            "regime": regime, "label": label, "color": color,
            "score": score, "advice": advice,
            "above_50": above_50, "above_200": above_200,
            "dist_200_pct": round(dist200, 1),
            "ret_5d": round(ret_5d * 100, 2),
            "ret_20d": round(ret_20d * 100, 2),
            "hist_vol": round(hist_vol, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTOR ROTATION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
class SectorRotationDetector:
    """
    Ranks all 11 S&P 500 sectors by 1-month momentum using SPDR ETFs.
    Results are cached for 30 minutes so a full scan only downloads once.

    Usage:
        tier = SectorRotationDetector.classify("Technology")
        # → "LEADING" | "NEUTRAL" | "LAGGING"

        rankings = SectorRotationDetector.get()
        # → {"Technology": {"rank":1, "momentum":+3.2, "tier":"LEADING"}, ...}
    """

    SECTOR_ETFS: dict = {
        "Technology":             "XLK",
        "Consumer Discretionary": "XLY",
        "Energy":                 "XLE",
        "Financials":             "XLF",
        "Health Care":            "XLV",
        "Industrials":            "XLI",
        "Materials":              "XLB",
        "Real Estate":            "XLRE",
        "Consumer Staples":       "XLP",
        "Utilities":              "XLU",
        "Communication Services": "XLC",
    }

    _cache:    dict  = {}
    _cache_ts: float = 0.0
    _TTL:      int   = 1800   # 30-minute cache

    @classmethod
    def get(cls) -> dict:
        """Return sector rankings, refreshing the cache if stale."""
        if time.time() - cls._cache_ts < cls._TTL and cls._cache:
            return cls._cache

        etf_list = list(cls.SECTOR_ETFS.values())
        rankings: dict = {}
        try:
            df = yf.download(etf_list, period="3mo", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                return cls._cache

            close = df["Close"] if "Close" in df.columns else df.xs("Close", axis=1, level=0)
            if not isinstance(close, pd.DataFrame):
                return cls._cache

            moms: dict = {}
            for sector, etf in cls.SECTOR_ETFS.items():
                if etf in close.columns:
                    prices = close[etf].dropna()
                    if len(prices) >= 20:
                        # 1-month (≈21 trading days) momentum
                        moms[sector] = float(prices.iloc[-1] / prices.iloc[-21] - 1)

            if not moms:
                return cls._cache

            sorted_sectors = sorted(moms, key=lambda s: moms[s], reverse=True)
            n = len(sorted_sectors)
            top_cut    = max(1, n // 3)       # top third  = LEADING
            bottom_cut = n - max(1, n // 3)  # bottom third = LAGGING

            for rank, sector in enumerate(sorted_sectors, 1):
                tier = ("LEADING" if rank <= top_cut else
                        "LAGGING" if rank > bottom_cut else
                        "NEUTRAL")
                rankings[sector] = {
                    "rank":     rank,
                    "of":       n,
                    "momentum": round(moms[sector] * 100, 2),
                    "tier":     tier,
                    "etf":      cls.SECTOR_ETFS[sector],
                }

            cls._cache    = rankings
            cls._cache_ts = time.time()
        except Exception:
            pass

        return cls._cache or {}

    @classmethod
    def classify(cls, sector: str) -> str:
        """Return 'LEADING', 'NEUTRAL', or 'LAGGING' for the given sector name."""
        if not sector:
            return "NEUTRAL"
        rankings = cls.get()
        if not rankings:
            return "NEUTRAL"
        # Exact match first, then partial
        data = rankings.get(sector)
        if not data:
            sl = sector.lower()
            for k, v in rankings.items():
                if sl in k.lower() or k.lower() in sl:
                    data = v
                    break
        return (data or {}).get("tier", "NEUTRAL")

    @classmethod
    def top_sectors(cls, n: int = 4) -> list:
        """Return the names of the top-N LEADING sectors by momentum."""
        rankings = cls.get()
        if not rankings:
            return []
        return [s for s, d in sorted(rankings.items(),
                                     key=lambda x: x[1]["rank"])[:n]]


# ══════════════════════════════════════════════════════════════════════════════
# FEE ENGINE — 1.5% per transaction (broker + spread + slippage)
# ══════════════════════════════════════════════════════════════════════════════
class FeeEngine:
    FEE_RATE = 0.015  # 1.5% per transaction

    def calculate_buy(self, gross_amount: float, price: float) -> dict:
        fee    = gross_amount * self.FEE_RATE
        net    = gross_amount - fee
        shares = net / price if price > 0 else 0
        return {"gross": gross_amount, "fee": fee, "net_investment": net,
                "shares": shares,
                "effective_price": gross_amount / shares if shares > 0 else price}

    def calculate_sell(self, shares: float, sell_price: float) -> dict:
        gross = shares * sell_price
        fee   = gross * self.FEE_RATE
        net   = gross - fee
        return {"gross": gross, "fee": fee, "net": net,
                "effective_price": net / shares if shares > 0 else sell_price}

    def calculate_round_trip(self, entry: float, exit_: float, shares: float) -> dict:
        buy_fee  = shares * entry  * self.FEE_RATE
        sell_fee = shares * exit_  * self.FEE_RATE
        total    = buy_fee + sell_fee
        invested = shares * entry
        drag_pct = total / invested * 100 if invested > 0 else 0
        return {"buy_fee": buy_fee, "sell_fee": sell_fee,
                "total_fees": total, "fee_drag_pct": drag_pct}

    def break_even_move(self, entry: float) -> dict:
        r        = self.FEE_RATE
        be_price = entry * (1 + r) / (1 - r)
        be_pct   = (be_price - entry) / entry * 100
        return {"break_even_price": be_price, "break_even_pct": round(be_pct, 2)}

    def fee_adjusted_rr(self, entry: float, target: float, stop: float) -> dict:
        r = self.FEE_RATE
        eff_entry  = entry  * (1 + r)
        eff_target = target * (1 - r)
        eff_stop   = stop   * (1 - r)

        raw_reward = target - entry
        raw_risk   = entry  - stop
        raw_rr     = raw_reward / raw_risk if raw_risk > 0 else 0
        raw_reward_pct = raw_reward / entry * 100 if entry > 0 else 0

        adj_reward = eff_target - eff_entry
        adj_risk   = eff_entry  - eff_stop
        adj_rr     = adj_reward / adj_risk if adj_risk > 0 else 0
        adj_reward_pct = adj_reward / eff_entry * 100 if eff_entry > 0 else 0

        return {"raw_rr": round(raw_rr, 2), "raw_reward_pct": round(raw_reward_pct, 1),
                "fee_adj_rr": round(adj_rr, 2),
                "fee_adj_reward_pct": round(adj_reward_pct, 1)}


_FEE_ENGINE = FeeEngine()


# ══════════════════════════════════════════════════════════════════════════════
# TRADE FILTER — hard R/R ≥ 2.0 AND reward ≥ 20% (fee-adjusted)
# ══════════════════════════════════════════════════════════════════════════════
class TradeFilter:
    MIN_RR      = 2.0
    MIN_REWARD  = 20.0  # percent

    def __init__(self, fee: FeeEngine = None):
        self.fee = fee or _FEE_ENGINE

    def check(self, result: dict) -> dict:
        entry  = result.get("price")       or 0
        target = result.get("tgt_price")   or 0
        stop   = result.get("stop_price")  or 0

        if not entry or not target or not stop or stop >= entry:
            calc = {"raw_rr": 0, "raw_reward_pct": 0,
                    "fee_adj_rr": 0, "fee_adj_reward_pct": 0}
            return {"pass": False, "rr_pass": False, "reward_pass": False,
                    "reason": "Rejected: insufficient price data", **calc}

        calc       = self.fee.fee_adjusted_rr(entry, target, stop)
        adj_rr     = calc["fee_adj_rr"]
        adj_reward = calc["fee_adj_reward_pct"]

        rr_pass     = adj_rr     >= self.MIN_RR
        reward_pass = adj_reward >= self.MIN_REWARD

        reasons = []
        if not rr_pass:
            reasons.append(f"R/R {adj_rr:.1f} (min 2.0)")
        if not reward_pass:
            reasons.append(f"Reward +{adj_reward:.1f}% (min +20%)")

        return {"pass": rr_pass and reward_pass,
                "rr_pass": rr_pass, "reward_pass": reward_pass,
                "reason": ("✅ PASS" if not reasons
                           else "Rejected: " + " AND ".join(reasons)),
                **calc}


_TRADE_FILTER = TradeFilter()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — all numeric thresholds here, no logic changes needed below
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "lookback_days":        252,
    "max_workers":          10,
    # VCP
    "vcp_swing_order":      5,
    "vcp_pivot_pct":        0.03,
    "vcp_atr_63d":          63,
    "vcp_atr_126d":         126,
    # Cup & Handle
    "cup_min_days":         30,
    "cup_max_days":         130,
    "cup_min_depth":        0.12,
    "cup_max_depth":        0.50,
    "cup_handle_min":       0.08,
    "cup_handle_max":       0.15,
    "cup_handle_days":      25,
    # Wyckoff
    "wk_range_days_max":    80,
    "wk_range_pct_min":     0.10,
    "wk_range_pct_max":     0.35,
    "wk_spring_pct":        0.05,
    "wk_spring_window":     15,
    "wk_sos_window":        8,
    # HTF
    "htf_min_gain":         0.50,
    "htf_max_pullback":     0.20,
    "htf_flag_min":         5,
    "htf_flag_max":         15,
    # Bull Flag
    "bf_min_gain":          0.15,
    "bf_pole_max_days":     10,
    "bf_retrace_min":       0.25,
    "bf_retrace_max":       0.45,
    "bf_flag_max_days":     20,
    # Pennant
    "pn_min_gain":          0.15,
    "pn_pole_max_days":     10,
    "pn_duration_max":      15,
    # Ascending Triangle
    "at_min_days":          20,
    "at_max_days":          60,
    "at_touches":           3,
    "at_res_tol":           0.02,
    # Flat Base
    "fb_min_days":          25,
    "fb_max_days":          50,
    "fb_range_max":         0.15,
    "fb_max_week_loss":     0.05,
    # Double Bottom
    "db_price_tol":         0.05,
    "db_min_sep":           20,
    # Symmetrical Triangle
    "sym_min_days":         15,
    "sym_max_days":         40,
    "sym_apex_min":         0.60,
    "sym_apex_max":         0.80,
    # BB Squeeze
    "bb_squeeze_6m":        126,
    "bb_squeeze_3m":        63,
    # Volume
    "vol_strong":           1.5,
    "vol_very_strong":      2.0,
    # ADX
    "adx_strong":           30,
    "adx_medium":           25,
    # Scoring
    "tier1_pts":            40,
    "tier2_pts":            30,
    "tier3_pts":            20,
}

BASE_RATES = {
    "VCP_3plus":             72,
    "VCP_2":                 58,
    "Cup and Handle":        68,
    "Wyckoff Spring":        67,
    "High and Tight Flag":   75,
    "Bull Flag":             67,
    "Pennant":               65,
    "Ascending Triangle":    63,
    "Flat Base":             61,
    "Double Bottom":         58,
    "Symmetrical Triangle":  54,
    "No Pattern":            42,
}

# ══════════════════════════════════════════════════════════════════════════════
# LEARNED WEIGHTS  — persisted in weights.json, updated by WeightOptimizer
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS_FILE = "weights.json"


def _load_weights() -> dict:
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"adjustments": {}, "base_rate_adjustments": {}, "samples": 0, "last_updated": None}


def _save_weights(data: dict):
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


LEARNED_WEIGHTS = _load_weights()


# Trade thesis templates — one per pattern
THESES = {
    "VCP": (
        "{ticker} has formed {n_cont} successive volatility contractions "
        "({depths}), a textbook sign of institutional supply absorption. "
        "The tight coil near ${pivot:.2f} on shrinking ATR signals an "
        "imminent expansion. Enter on a volume-backed close above the pivot; "
        "invalidate on a close below the last contraction low."
    ),
    "Cup and Handle": (
        "{ticker} carved a {depth:.0%} cup base over {weeks:.0f} weeks "
        "and formed a {handle:.0%} handle — a classic accumulation pattern. "
        "Low-volume handle drift sets up a high-probability breakout above "
        "${pivot:.2f}. Invalidate on a close below the handle low."
    ),
    "Wyckoff Spring": (
        "{ticker} executed a Wyckoff Spring, flushing weak holders below "
        "range support before snapping back — a sign of smart-money absorption. "
        "The Sign of Strength confirms demand control. Enter near current "
        "levels targeting a full re-rating; stop below the spring low."
    ),
    "High and Tight Flag": (
        "{ticker} surged {gain:.0%} in the flagpole — a rare HTF setup. "
        "The tight {pullback:.0%} consolidation on declining volume signals "
        "minimal overhead supply. This pattern resolves higher the majority "
        "of the time. Risk is well-defined below the flag low."
    ),
    "Bull Flag": (
        "{ticker} built a {gain:.0%} flagpole and is consolidating in an "
        "orderly {retrace:.0%} pullback — a healthy pause within the uptrend. "
        "A breakout above the upper channel triggers the measured move. "
        "Risk: a close below the channel or >50% retrace negates the setup."
    ),
    "Pennant": (
        "{ticker} rallied {gain:.0%} then formed a symmetrical pennant "
        "as volume dried to multi-week lows near the apex — balanced supply "
        "and demand before a resolution. At {apex:.0%} to apex, breakout "
        "timing is ideal. Risk: a breakdown below the lower trendline."
    ),
    "Ascending Triangle": (
        "{ticker} has tested resistance at ${resist:.2f} {touches} times "
        "while making higher lows — demand accumulation at a key level. "
        "Price is coiled in the upper range; a volume surge above resistance "
        "completes the breakout. Stop below the most recent higher low."
    ),
    "Flat Base": (
        "{ticker} spent {weeks:.0f} weeks in a tight {range:.0%} range near "
        "its 52-week high — institutional holders not selling concentrates "
        "supply at the pivot. A volume-backed close above the base high "
        "triggers the measured move. Base failure = close below midpoint."
    ),
    "Double Bottom": (
        "{ticker} formed two troughs near ${low:.2f} with {div}bullish "
        "divergence, signaling seller exhaustion. A breakout above the "
        "neckline at ${neck:.2f} confirms the reversal. Invalidate on a "
        "close below the second bottom."
    ),
    "Symmetrical Triangle": (
        "{ticker} is compressing into a symmetrical triangle at {apex:.0%} "
        "progress to apex — volume dried up and resolution is near. Given "
        "the broader uptrend, the bullish resolution is higher probability. "
        "Risk: a breakdown below the lower trendline flips the bias bearish."
    ),
    "No Pattern": (
        "{ticker} shows momentum characteristics without a classic pattern. "
        "Monitor for a volume-backed push to new highs as the entry trigger. "
        "Risk/reward is less defined than a structured setup — size accordingly."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _fcol(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Return first column whose name starts with prefix, else empty Series."""
    for c in df.columns:
        if c.startswith(prefix):
            return df[c].dropna()
    return pd.Series(dtype=float)


def _slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def _r2(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    resid = arr - np.polyval(np.polyfit(x, arr, 1), x)
    ss_tot = np.sum((arr - arr.mean()) ** 2)
    return float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else 0.0


def _swings(arr: np.ndarray, order: int = 5):
    """Return (highs_idx, lows_idx) arrays using scipy argrelextrema."""
    h = argrelextrema(arr, np.greater, order=order)[0]
    l = argrelextrema(arr, np.less,    order=order)[0]
    return h, l


def _norm(raw) -> pd.DataFrame:
    """Flatten yfinance MultiIndex to single-level OHLCV — handles both column orderings."""
    _OHLCV = {"Open", "High", "Low", "Close", "Volume"}
    if isinstance(raw.columns, pd.MultiIndex):
        # Detect which level contains the price-type names
        lvl = 0
        if not _OHLCV.issubset(set(raw.columns.get_level_values(0))):
            lvl = 1
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(lvl)
        # Drop duplicate column names (yfinance sometimes emits duplicates)
        raw = raw.loc[:, ~raw.columns.duplicated()]
    return raw[["Open", "High", "Low", "Close", "Volume"]].copy()


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
class PatternDetector:
    """
    One method per pattern.
    Every method returns:
      { "detected": bool, "confidence": float 0-1, "summary": str, "name": str, ...extras }
    """

    def __init__(self, df: pd.DataFrame, spy_df: pd.DataFrame = None, cfg: dict = None):
        self.df  = df
        self.spy = spy_df
        self.cfg = cfg or CONFIG
        self.C   = df["Close"].values
        self.H   = df["High"].values
        self.L   = df["Low"].values
        self.V   = df["Volume"].values.astype(float)
        self.n   = len(df)

    def _r(self, det, conf, summ, name="", **kw):
        return {"detected": det, "confidence": float(np.clip(conf, 0, 1)),
                "summary": summ, "name": name, **kw}

    # ── TIER 1 ───────────────────────────────────────────────────────────────

    def detect_vcp(self):
        C, H, L, V, n, cfg = self.C, self.H, self.L, self.V, self.n, self.cfg
        if n < 60:
            return self._r(False, 0, "Insufficient data", "VCP")

        sma50  = _fcol(self.df, "SMA_50")
        sma200 = _fcol(self.df, "SMA_200")
        if sma50.empty or sma200.empty:
            return self._r(False, 0, "Missing SMAs", "VCP")
        cur = float(C[-1])
        if cur < float(sma50.iloc[-1]) or cur < float(sma200.iloc[-1]):
            return self._r(False, 0, "Not in Stage 2 uptrend", "VCP")

        lb   = min(n, 120)
        subH = H[-lb:]; subL = L[-lb:]; subV = V[-lb:]
        h_idx, l_idx = _swings(subH, order=cfg["vcp_swing_order"])

        if len(h_idx) < 2 or len(l_idx) < 2:
            return self._r(False, 0, "Too few swing points", "VCP")

        used = set()
        contractions = []
        for shi in h_idx:
            nxt = [i for i in l_idx if i > shi and i not in used]
            if not nxt:
                continue
            sli = nxt[0]; used.add(sli)
            sh_v = subH[shi]; sl_v = subL[sli]
            contractions.append({
                "depth":    (sh_v - sl_v) / sh_v,
                "duration": sli - shi,
                "high":     sh_v, "low": sl_v,
                "hi":       shi,  "lo":  sli,
                "avg_vol":  float(subV[shi:sli + 1].mean()) if sli > shi else float(subV[shi]),
            })

        if len(contractions) < 2:
            return self._r(False, 0, "< 2 contractions", "VCP")

        valid = [contractions[0]]
        for i in range(1, len(contractions)):
            pr, cu = contractions[i - 1], contractions[i]
            if cu["depth"] < pr["depth"] and cu["duration"] <= pr["duration"]:
                valid.append(cu)
            else:
                break
        n_cont = len(valid)
        if n_cont < 2:
            return self._r(False, 0, f"Only {n_cont} valid contraction", "VCP")

        vol_final_lowest = valid[-1]["avg_vol"] == min(v["avg_vol"] for v in valid)
        last   = valid[-1]
        pivot  = last["high"]
        fdepth = last["depth"]
        prox   = (pivot - cur) / pivot
        if prox > cfg["vcp_pivot_pct"] or prox < -0.01:
            return self._r(False, 0.3, f"Price {prox:.1%} from pivot ${pivot:.2f}", "VCP")

        atr = _fcol(self.df, "ATRr")
        at63  = len(atr) >= 63  and float(atr.iloc[-1]) <= float(atr.iloc[-63:].min())  * 1.02
        at126 = len(atr) >= 126 and float(atr.iloc[-1]) <= float(atr.iloc[-126:].min()) * 1.02

        conf = 0.60
        if n_cont >= 3:       conf += 0.10
        if fdepth < 0.06:     conf += 0.10
        if vol_final_lowest:  conf += 0.10
        if at126:             conf += 0.10
        elif at63:            conf += 0.05

        depths_str = " → ".join(f"{c['depth']:.0%}" for c in valid)
        pkey  = "VCP_3plus" if n_cont >= 3 else "VCP_2"
        summ  = f"VCP {n_cont}x ({depths_str}), pivot ${pivot:.2f}, {prox:.1%} below"
        return self._r(True, conf, summ, "VCP",
                       n_contractions=n_cont, pivot=pivot, final_depth=fdepth,
                       pattern_key=pkey, depths=depths_str)

    def detect_cup_and_handle(self):
        C, H, V, n, cfg = self.C, self.H, self.V, self.n, self.cfg
        if n < 40:
            return self._r(False, 0, "Insufficient data", "Cup and Handle")

        best_conf, best_kw = 0.0, None
        for cup_days in range(min(cfg["cup_max_days"], n - 10), cfg["cup_min_days"] - 1, -10):
            sub = C[-cup_days:]; slen = len(sub); w = max(5, slen // 6)

            left_high  = float(sub[:w].max())
            cup_base   = float(sub[slen // 4: 3 * slen // 4].min())
            right_high = float(sub[-w:].max())

            depth = (left_high - cup_base) / left_high if left_high > 0 else 1
            if not (cfg["cup_min_depth"] <= depth <= cfg["cup_max_depth"]):
                continue
            if (left_high - right_high) / left_high > 0.05:
                continue

            # V-shape check: recovery speed
            lh = sub[:slen // 2]; rh = sub[slen // 2:]
            v_shaped = (float(rh.max()) - float(rh.min())) < (float(lh.max()) - float(lh.min())) * 0.80

            hd = min(cfg["cup_handle_days"], n - 1)
            hC = C[-hd:]; hh = float(hC.max()); hl = float(hC.min())
            hret = (hh - hl) / hh if hh > 0 else 1
            if not (cfg["cup_handle_min"] <= hret <= cfg["cup_handle_max"]):
                continue
            if hl < cup_base + (left_high - cup_base) / 2:
                continue

            hV = V[-hd:]; pV = V[-hd * 2:-hd] if n > hd * 2 else V[:hd]
            vol_ok = float(hV.mean()) < float(pV.mean())

            # VCP sub-structure in handle
            hH_sw, hL_sw = _swings(hC, order=2)
            handle_vcp   = len(hH_sw) >= 2 and len(hL_sw) >= 2

            conf = 0.60
            if 0.25 <= depth <= 0.35:                             conf += 0.10
            if handle_vcp:                                        conf += 0.10
            if vol_ok:                                            conf += 0.05
            if (left_high - right_high) / left_high <= 0.02:     conf += 0.10
            if v_shaped:                                          conf -= 0.10

            if conf > best_conf:
                best_conf = conf
                best_kw = {"cup_depth": depth, "handle_retrace": hret,
                           "pivot": hh, "v_shaped": v_shaped,
                           "weeks": cup_days / 5, "handle_vcp": handle_vcp}

        if best_kw and best_conf >= 0.50:
            s = (f"Cup & Handle: {best_kw['cup_depth']:.0%} cup ({best_kw['weeks']:.0f}w), "
                 f"{best_kw['handle_retrace']:.0%} handle, pivot ${best_kw['pivot']:.2f}"
                 f"{' [V-shaped]' if best_kw['v_shaped'] else ''}")
            return self._r(True, best_conf, s, "Cup and Handle", **best_kw)
        return self._r(False, 0, "No Cup and Handle", "Cup and Handle")

    def detect_wyckoff_spring(self):
        C, H, L, V, n, cfg = self.C, self.H, self.L, self.V, self.n, self.cfg
        if n < 30:
            return self._r(False, 0, "Insufficient data", "Wyckoff Spring")

        lb   = min(cfg["wk_range_days_max"], n)
        subC = C[-lb:]; subH = H[-lb:]; subL = L[-lb:]; subV = V[-lb:]
        rh   = float(np.percentile(subH, 85)); rl = float(np.percentile(subL, 15))
        rng  = (rh - rl) / rl if rl > 0 else 1

        if not (cfg["wk_range_pct_min"] <= rng <= cfg["wk_range_pct_max"]):
            return self._r(False, 0, f"Range {rng:.1%} outside 10–35%", "Wyckoff Spring")
        if float(np.mean((subC >= rl * 0.98) & (subC <= rh * 1.02))) < 0.80:
            return self._r(False, 0, "< 80% candles in range", "Wyckoff Spring")

        avg_v = float(V[-20:].mean())
        spring_idx = -1; spring_vr = 1.0; same_sess = False
        win = min(cfg["wk_spring_window"], lb - 1)

        for i in range(lb - win, lb):
            if subL[i] < rl * (1 - cfg["wk_spring_pct"]) and subC[i] > rl * 0.99:
                spring_idx = i
                spring_vr  = subV[i] / avg_v if avg_v > 0 else 1
                same_sess  = bool(subC[i] > rl)
                break

        if spring_idx < 0:
            return self._r(False, 0, "No Spring below support", "Wyckoff Spring")
        if spring_vr > 1.0:
            return self._r(False, 0.2, "Spring on above-avg volume", "Wyckoff Spring")

        sos_found = False; sos_vr = 0.0
        for i in range(spring_idx, min(spring_idx + cfg["wk_sos_window"], lb)):
            if subC[i] > rh and subV[i] > avg_v * 1.5:
                sos_found = True; sos_vr = subV[i] / avg_v; break

        if not sos_found:
            return self._r(False, 0.4, "Spring found, SOS pending", "Wyckoff Spring")

        lps      = bool(float(subV[-5:].mean()) < avg_v * 0.80)
        long_rng = lb >= 40

        conf = 0.60
        if same_sess:   conf += 0.10
        if sos_vr >= 2: conf += 0.10
        if lps:         conf += 0.10
        if long_rng:    conf += 0.10

        s = (f"Wyckoff Spring ${subL[spring_idx]:.2f} ({spring_vr:.1f}x vol), "
             f"SOS {sos_vr:.1f}x{', LPS' if lps else ''}")
        return self._r(True, conf, s, "Wyckoff Spring",
                       spring_vr=spring_vr, sos_vr=sos_vr, lps=lps, range_days=lb)

    # ── TIER 2 ───────────────────────────────────────────────────────────────

    def detect_high_tight_flag(self):
        C, H, L, V, n, cfg = self.C, self.H, self.L, self.V, self.n, self.cfg
        if n < 20:
            return self._r(False, 0, "Insufficient data", "High and Tight Flag")

        best_conf, best_kw = 0.0, None
        for window in [10, 15, 20, 25]:
            for off in range(5, 20, 5):
                if window + off + 2 >= n:
                    continue
                seg = C[-(window + off):-off]
                if len(seg) < 2:
                    continue
                lo   = float(seg.min()); hi = float(seg.max())
                gain = (hi - lo) / lo if lo > 0 else 0
                if gain < cfg["htf_min_gain"]:
                    continue

                flag_C = C[-off:]
                if len(flag_C) < cfg["htf_flag_min"] or len(flag_C) > cfg["htf_flag_max"]:
                    continue
                pullback = (hi - float(flag_C.min())) / hi if hi > 0 else 1
                if pullback > cfg["htf_max_pullback"]:
                    continue

                poleV = V[-(window + off):-off]
                flagV = V[-off:]
                vol_ok = float(flagV.mean()) <= float(poleV.mean()) * 0.70 if len(poleV) > 0 else False

                conf = 0.65
                if pullback < 0.10:       conf += 0.15
                if gain >= 0.75:          conf += 0.10
                if len(flag_C) <= 10:     conf += 0.10
                if conf > best_conf:
                    best_conf = conf
                    best_kw   = {"pole_gain": gain, "pullback": pullback,
                                 "flag_days": len(flag_C), "vol_ok": vol_ok}

        if best_kw and best_conf >= 0.60:
            s = (f"HTF: {best_kw['pole_gain']:.0%} pole, "
                 f"{best_kw['pullback']:.0%} pullback, {best_kw['flag_days']}d flag")
            return self._r(True, best_conf, s, "High and Tight Flag", **best_kw)
        return self._r(False, 0, "No HTF", "High and Tight Flag")

    def detect_bull_flag(self):
        C, H, L, V, n, cfg = self.C, self.H, self.L, self.V, self.n, self.cfg
        if n < 20:
            return self._r(False, 0, "Insufficient data", "Bull Flag")
        sma20 = _fcol(self.df, "SMA_20")
        avg20 = float(V[-20:].mean())
        best_conf, best_kw = 0.0, None

        for fd in range(5, min(cfg["bf_flag_max_days"] + 1, n // 2)):
            for pd_ in range(3, cfg["bf_pole_max_days"] + 1):
                off = fd + pd_
                if off + 3 >= n:
                    continue
                pole_C = C[-(off + 3):-fd]
                if len(pole_C) < 2:
                    continue
                ps = float(pole_C[0]); ph = float(pole_C.max())
                gain = (ph - ps) / ps if ps > 0 else 0
                if gain < cfg["bf_min_gain"]:
                    continue
                pV = V[-(off + 3):-fd]
                if float(pV.mean()) < avg20 * 1.5:
                    continue  # pole needs high volume

                flag_C = C[-fd:]; flagL = L[-fd:]; flagV = V[-fd:]
                fh = float(C[-fd:].max()); fl = float(flagL.min())
                denom   = ph - ps
                retrace = (ph - fl) / denom if denom > 0 else 1
                if not (cfg["bf_retrace_min"] <= retrace <= cfg["bf_retrace_max"]):
                    continue
                if _slope(flag_C) >= 0:
                    continue
                if float(flagV.mean()) >= float(pV.mean()) * 0.60:
                    continue

                above20 = not sma20.empty and float(flagL.min()) > float(sma20.iloc[-1])
                conf = 0.60
                if retrace < 0.35: conf += 0.10
                if float(flagV.min()) <= avg20: conf += 0.10
                if above20:        conf += 0.10
                if gain >= 0.20:   conf += 0.10
                if conf > best_conf:
                    best_conf = conf
                    best_kw   = {"pole_gain": gain, "retrace_pct": retrace,
                                 "flag_days": fd, "above20": above20, "pivot": fh}

        if best_kw and best_conf >= 0.55:
            s = (f"Bull Flag: {best_kw['pole_gain']:.0%} pole, "
                 f"{best_kw['retrace_pct']:.0%} retrace, {best_kw['flag_days']}d")
            return self._r(True, best_conf, s, "Bull Flag", **best_kw)
        return self._r(False, 0, "No Bull Flag", "Bull Flag")

    def detect_pennant(self):
        C, H, L, V, n, cfg = self.C, self.H, self.L, self.V, self.n, self.cfg
        if n < 18:
            return self._r(False, 0, "Insufficient data", "Pennant")
        avg20 = float(V[-20:].mean()); avg30 = float(V[-30:].min())
        best_conf, best_kw = 0.0, None

        for nd in range(5, cfg["pn_duration_max"] + 1):
            for pd_ in range(3, cfg["pn_pole_max_days"] + 1):
                off = nd + pd_
                if off + 3 >= n:
                    continue
                pole_C = C[-(off + 3):-nd]
                if len(pole_C) < 2:
                    continue
                ps = float(pole_C[0]); ph = float(pole_C.max())
                gain = (ph - ps) / ps if ps > 0 else 0
                if gain < cfg["pn_min_gain"]:
                    continue

                pnH = H[-nd:]; pnL = L[-nd:]; pnV = V[-nd:]
                if len(pnH) < 4:
                    continue
                if _slope(pnH) >= 0 or _slope(pnL) <= 0:
                    continue

                x     = np.arange(nd, dtype=float)
                hline = np.polyval(np.polyfit(x, pnH, 1), x)
                lline = np.polyval(np.polyfit(x, pnL, 1), x)
                diff  = hline - lline
                if diff[-1] <= 0 or diff[0] <= 0:
                    continue
                apex_pct = 1 - diff[-1] / diff[0]
                if not (0.50 <= apex_pct <= 0.80):
                    continue

                vol30_low  = float(pnV.min()) < avg30 * 1.02
                steep_pole = gain >= 0.15 and pd_ <= 5

                conf = 0.58
                if vol30_low:                conf += 0.10
                if 0.60 <= apex_pct <= 0.75: conf += 0.10
                if steep_pole:               conf += 0.10
                if conf > best_conf:
                    best_conf = conf
                    best_kw   = {"pole_gain": gain, "pennant_days": nd, "apex_pct": apex_pct}

        if best_kw and best_conf >= 0.55:
            s = (f"Pennant: {best_kw['pole_gain']:.0%} pole, "
                 f"{best_kw['pennant_days']}d, {best_kw['apex_pct']:.0%} to apex")
            return self._r(True, best_conf, s, "Pennant", **best_kw)
        return self._r(False, 0, "No Pennant", "Pennant")

    # ── TIER 3 ───────────────────────────────────────────────────────────────

    def detect_ascending_triangle(self):
        H, L, C, V, n, cfg = self.H, self.L, self.C, self.V, self.n, self.cfg
        lb = min(cfg["at_max_days"], n)
        if lb < cfg["at_min_days"]:
            return self._r(False, 0, "Insufficient data", "Ascending Triangle")

        sH = H[-lb:]; sL = L[-lb:]; sC = C[-lb:]; sV = V[-lb:]
        h_idx, _ = _swings(sH, order=4)
        if len(h_idx) < 2:
            return self._r(False, 0, "Too few swing highs", "Ascending Triangle")

        resistance = float(np.median([sH[i] for i in h_idx]))
        touches    = sum(1 for i in h_idx if abs(sH[i] - resistance) / resistance <= cfg["at_res_tol"])
        if touches < cfg["at_touches"]:
            return self._r(False, 0, f"{touches} resistance touches (need {cfg['at_touches']}+)", "Ascending Triangle")

        _, l_idx = _swings(sL, order=4)
        if len(l_idx) < 2:
            return self._r(False, 0, "Too few swing lows", "Ascending Triangle")
        sup_lows = np.array([sL[i] for i in l_idx])
        if _slope(sup_lows) <= 0:
            return self._r(False, 0, "Support not rising", "Ascending Triangle")

        tri_l  = float(sL[l_idx].min())
        coiled = float(sC[-1]) >= tri_l + (resistance - tri_l) * 0.75
        age_ok = 15 <= lb <= 25
        vol_ok = _slope(sV) < 0
        r2     = _r2(sup_lows)

        conf = 0.55
        if touches >= 4: conf += 0.10
        if age_ok:       conf += 0.10
        if vol_ok:       conf += 0.10
        if r2 > 0.85:    conf += 0.10

        s = (f"Ascending Triangle: {touches} touches at ${resistance:.2f}, "
             f"{'coiled near top' if coiled else 'building'}")
        return self._r(True, conf, s, "Ascending Triangle",
                       resistance=resistance, touches=touches, coiled=coiled)

    def detect_flat_base(self):
        C, H, L, V, n, cfg = self.C, self.H, self.L, self.V, self.n, self.cfg
        if n < cfg["fb_min_days"]:
            return self._r(False, 0, "Insufficient data", "Flat Base")

        sma50  = _fcol(self.df, "SMA_50")
        sma200 = _fcol(self.df, "SMA_200")
        atr    = _fcol(self.df, "ATRr")
        h252   = float(H[-252:].max()) if n >= 252 else float(H.max())
        cur    = float(C[-1])
        best_conf, best_kw = 0.0, None

        for lb in range(cfg["fb_min_days"], cfg["fb_max_days"] + 1, 5):
            if lb >= n:
                continue
            sH = H[-lb:]; sL = L[-lb:]; sC = C[-lb:]
            rng = (float(sH.max()) - float(sL.min())) / float(sL.min()) if float(sL.min()) > 0 else 1
            if rng > cfg["fb_range_max"]:
                continue

            wl_ok = True
            for w in range(0, lb - 4, 5):
                wo = float(sC[w]); wc = float(sC[min(w + 4, lb - 1)])
                if wo > 0 and (wo - wc) / wo > cfg["fb_max_week_loss"]:
                    wl_ok = False; break
            if not wl_ok:
                continue

            above = True
            if not sma50.empty  and cur < float(sma50.iloc[-1]):  above = False
            if not sma200.empty and cur < float(sma200.iloc[-1]): above = False

            atr_low = (len(atr) >= 126 and
                       float(atr.iloc[-1]) <= float(atr.iloc[-126:].min()) * 1.02)
            p52  = (h252 - cur) / h252
            near = p52 <= 0.05

            rs30 = None
            if self.spy is not None and n >= 31:
                sp = self.spy["Close"].values
                if len(sp) >= 31:
                    rs30 = ((C[-1] - C[-31]) / C[-31]) - ((sp[-1] - sp[-31]) / sp[-31])

            ideal_wks = 30 <= lb <= 40
            conf = 0.55
            if rng <= 0.10:                    conf += 0.10
            if rs30 is not None and rs30 > 0:  conf += 0.10
            if atr_low:                        conf += 0.10
            if ideal_wks:                      conf += 0.10
            if conf > best_conf:
                best_conf = conf
                best_kw   = {"range_pct": rng, "near_high": near, "pct52": p52,
                             "above_smas": above, "weeks": lb / 5}

        if best_kw and best_conf >= 0.50:
            s = (f"Flat Base: {best_kw['range_pct']:.0%} range "
                 f"{best_kw['weeks']:.0f}w, {'near 52W hi' if best_kw['near_high'] else 'consolidating'}")
            return self._r(True, best_conf, s, "Flat Base", **best_kw)
        return self._r(False, 0, "No Flat Base", "Flat Base")

    def detect_double_bottom(self):
        C, H, L, V, n, cfg = self.C, self.H, self.L, self.V, self.n, self.cfg
        if n < 50:
            return self._r(False, 0, "Insufficient data", "Double Bottom")

        lb  = min(120, n)
        sL  = L[-lb:]; sH = H[-lb:]; sC = C[-lb:]; sV = V[-lb:]
        rsi = _fcol(self.df, "RSI_")
        cur = float(C[-1])
        _, l_idx = _swings(sL, order=6)
        if len(l_idx) < 2:
            return self._r(False, 0, "Too few swing lows", "Double Bottom")

        best_conf, best_kw = 0.0, None
        for i in range(len(l_idx)):
            for j in range(i + 1, len(l_idx)):
                i1, i2 = l_idx[i], l_idx[j]
                v1, v2 = float(sL[i1]), float(sL[i2])
                if i2 - i1 < cfg["db_min_sep"]:
                    continue
                if abs(v1 - v2) / v1 > cfg["db_price_tol"]:
                    continue

                neckline = float(sH[i1:i2 + 1].max())
                pct_neck = (neckline - cur) / neckline
                if pct_neck > 0.05 or pct_neck < -0.02:
                    continue

                undercut    = bool(v2 < v1)
                vol_exhaust = bool(sV[i2] < sV[i1])
                rsi_div     = False
                if len(rsi) >= lb:
                    ri1 = float(rsi.iloc[-lb + i1]) if lb - i1 <= len(rsi) else None
                    ri2 = float(rsi.iloc[-lb + i2]) if lb - i2 <= len(rsi) else None
                    if ri1 and ri2 and v2 <= v1 and ri2 > ri1:
                        rsi_div = True

                conf = 0.52
                if undercut:                      conf += 0.10
                if rsi_div:                       conf += 0.10
                if sV[i2] <= sV[i1] * 0.70:      conf += 0.10
                if cur >= neckline * 0.99:        conf += 0.10
                if conf > best_conf:
                    best_conf = conf
                    best_kw   = {"low1": v1, "low2": v2, "neckline": neckline,
                                 "undercut": undercut, "rsi_div": rsi_div,
                                 "vol_exhaust": vol_exhaust}

        if best_kw and best_conf >= 0.50:
            s = (f"Double Bottom: ${best_kw['low1']:.2f}/{best_kw['low2']:.2f} "
                 f"neckline ${best_kw['neckline']:.2f}"
                 f"{', shakeout' if best_kw['undercut'] else ''}"
                 f"{', RSI div' if best_kw['rsi_div'] else ''}")
            return self._r(True, best_conf, s, "Double Bottom", **best_kw)
        return self._r(False, 0, "No Double Bottom", "Double Bottom")

    def detect_symmetrical_triangle(self):
        H, L, C, V, n, cfg = self.H, self.L, self.C, self.V, self.n, self.cfg
        sma50  = _fcol(self.df, "SMA_50")
        sma200 = _fcol(self.df, "SMA_200")
        adx    = _fcol(self.df, "ADX_")
        cur    = float(C[-1])

        if not sma50.empty and cur < float(sma50.iloc[-1]):
            return self._r(False, 0, "Below SMA50 — not bullish", "Symmetrical Triangle")

        strong_trend = (not sma200.empty and cur > float(sma200.iloc[-1]) and
                        not adx.empty and float(adx.iloc[-1]) > 25)
        avg30 = float(V[-30:].min())
        best_conf, best_kw = 0.0, None

        for lb in range(cfg["sym_min_days"], min(cfg["sym_max_days"] + 1, n)):
            sH = H[-lb:]; sL = L[-lb:]; sV = V[-lb:]
            if _slope(sH) >= 0 or _slope(sL) <= 0:
                continue

            x     = np.arange(lb, dtype=float)
            hline = np.polyval(np.polyfit(x, sH, 1), x)
            lline = np.polyval(np.polyfit(x, sL, 1), x)
            diff  = hline - lline
            if diff[-1] <= 0 or diff[0] <= 0:
                continue

            apex_pct = 1 - diff[-1] / diff[0]
            if not (cfg["sym_apex_min"] <= apex_pct <= cfg["sym_apex_max"]):
                continue

            vol30 = float(sV.min()) < avg30 * 1.02
            conf  = 0.50
            if 0.65 <= apex_pct <= 0.75: conf += 0.10
            if vol30:                    conf += 0.10
            if strong_trend:             conf += 0.10
            if conf > best_conf:
                best_conf = conf
                best_kw   = {"apex_pct": apex_pct, "days": lb}

        if best_kw and best_conf >= 0.50:
            s = f"Sym Triangle: {best_kw['days']}d, {best_kw['apex_pct']:.0%} to apex"
            return self._r(True, best_conf, s, "Symmetrical Triangle", **best_kw)
        return self._r(False, 0, "No Sym Triangle", "Symmetrical Triangle")

    # ── TIER 4 — Universal Signals ────────────────────────────────────────────

    def detect_volume_surge(self):
        V, n, cfg = self.V, self.n, self.cfg
        if n < 21:
            return {"ratio": 0.0, "strong": False, "very_strong": False}
        avg20 = float(V[-21:-1].mean())
        vr    = float(V[-1]) / avg20 if avg20 > 0 else 0
        return {"ratio": vr, "strong": vr >= cfg["vol_strong"],
                "very_strong": vr >= cfg["vol_very_strong"]}

    def detect_bb_squeeze(self):
        cfg = self.cfg
        upper = _fcol(self.df, "BBU_")
        lower = _fcol(self.df, "BBL_")
        mid   = _fcol(self.df, "BBM_")
        null  = {"squeeze_6m": False, "squeeze_3m": False, "bb_width": None}
        if upper.empty or lower.empty or mid.empty:
            return null
        idx   = upper.index.intersection(lower.index).intersection(mid.index)
        if len(idx) < 20:
            return null
        width = ((upper.loc[idx] - lower.loc[idx]) / mid.loc[idx]).dropna()
        cur   = float(width.iloc[-1])
        s6m   = (len(width) >= cfg["bb_squeeze_6m"] and
                 cur <= float(width.iloc[-cfg["bb_squeeze_6m"]:].min()) * 1.02)
        s3m   = (len(width) >= cfg["bb_squeeze_3m"] and
                 cur <= float(width.iloc[-cfg["bb_squeeze_3m"]:].min()) * 1.02)
        return {"squeeze_6m": bool(s6m), "squeeze_3m": bool(s3m), "bb_width": cur}

    def detect_retest(self):
        C, H, L, n = self.C, self.H, self.L, self.n
        if n < 10:
            return {"detected": False, "level": None}
        for days_ago in range(3, 8):
            if days_ago + 8 >= n:
                continue
            level = float(np.percentile(H[-(days_ago + 8):-days_ago], 90))
            if float(C[-days_ago]) <= level:
                continue
            if (abs(float(L[-days_ago:].min()) - level) / level <= 0.02 and
                    float(C[-1]) > level * 0.99):
                return {"detected": True, "level": level}
        return {"detected": False, "level": None}

    def validate_four_layers(self, pattern_result: dict):
        C, H, V, n = self.C, self.H, self.V, self.n
        if n < 3:
            return {"layers_passed": 0, "all_passed": False}
        passed = 0
        if pattern_result.get("confidence", 0) >= 0.60:
            passed += 1
        if float(C[-1]) > float(H[-2]):
            passed += 1
        avg20 = float(V[-21:-1].mean()) if n > 21 else float(V[:-1].mean())
        if float(V[-1]) >= avg20 * self.cfg["vol_strong"]:
            passed += 1
        if n >= 3:
            res = float(np.percentile(H[-10:-1], 90))
            if float(C[-2]) > res and float(C[-1]) > res * 0.99:
                passed += 1
        return {"layers_passed": passed, "all_passed": passed == 4}

    def run_all(self) -> dict:
        results = {
            "vcp":                  self.detect_vcp(),
            "cup_and_handle":       self.detect_cup_and_handle(),
            "wyckoff":              self.detect_wyckoff_spring(),
            "htf":                  self.detect_high_tight_flag(),
            "bull_flag":            self.detect_bull_flag(),
            "pennant":              self.detect_pennant(),
            "ascending_triangle":   self.detect_ascending_triangle(),
            "flat_base":            self.detect_flat_base(),
            "double_bottom":        self.detect_double_bottom(),
            "symmetrical_triangle": self.detect_symmetrical_triangle(),
            "volume_surge":         self.detect_volume_surge(),
            "bb_squeeze":           self.detect_bb_squeeze(),
            "retest":               self.detect_retest(),
        }
        best_conf, best_pat = 0.0, {}
        for k in ["vcp", "cup_and_handle", "wyckoff", "htf", "bull_flag", "pennant",
                  "ascending_triangle", "flat_base", "double_bottom", "symmetrical_triangle"]:
            p = results[k]
            if p.get("detected") and p.get("confidence", 0) > best_conf:
                best_conf = p["confidence"]; best_pat = p
        results["four_layer"] = self.validate_four_layers(best_pat)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# BREAKOUT PROBABILITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class BreakoutProbabilityEngine:
    """
    Three-layer Bayesian-style model:
      Layer 1 — base win rate from pattern history
      Layer 2 — additive modifiers (25+ factors)
      Layer 3 — confidence band from signal count
    Returns:
      { probability, band, band_label, base_rate, positive_mods,
        negative_mods, signals_count, display }
    """

    def __init__(self, pattern_results: dict, df: pd.DataFrame,
                 spy_df: pd.DataFrame = None, cfg: dict = None,
                 pm_gap: float = 0.0):
        self.pr     = pattern_results
        self.df     = df
        self.spy    = spy_df
        self.cfg    = cfg or CONFIG
        self._pm_gap = float(pm_gap or 0.0)
        self._mods: list = []
        self._n:    int  = 0

    def _base_rate(self):
        detected = []
        mapping  = {
            "vcp":                 ("VCP",                "VCP_2"),
            "cup_and_handle":      ("Cup and Handle",     "Cup and Handle"),
            "wyckoff":             ("Wyckoff Spring",     "Wyckoff Spring"),
            "htf":                 ("High and Tight Flag","High and Tight Flag"),
            "bull_flag":           ("Bull Flag",          "Bull Flag"),
            "pennant":             ("Pennant",            "Pennant"),
            "ascending_triangle":  ("Ascending Triangle", "Ascending Triangle"),
            "flat_base":           ("Flat Base",          "Flat Base"),
            "double_bottom":       ("Double Bottom",      "Double Bottom"),
            "symmetrical_triangle":("Symmetrical Triangle","Symmetrical Triangle"),
        }
        br_adj = LEARNED_WEIGHTS.get("base_rate_adjustments", {})
        for key, (label, _) in mapping.items():
            p = self.pr.get(key, {})
            if not p.get("detected"):
                continue
            wr  = BASE_RATES.get(p.get("pattern_key", label) if key == "vcp" else label, 42)
            wr  = min(90, max(30, wr + br_adj.get(label, 0)))
            lbl = "VCP" if key == "vcp" else label
            detected.append((lbl, wr, p.get("confidence", 0.5)))

        if not detected:
            return 42, "No Pattern", []
        detected.sort(key=lambda x: x[1], reverse=True)
        bonus = min(len(detected) - 1, 3) * 5          # +5% per extra pattern, cap +15%
        prob  = min(detected[0][1] + bonus, 90)
        return prob, detected[0][0], detected

    def _apply_mods(self, base: float):
        df   = self.df
        pr   = self.pr
        C    = df["Close"].values
        H    = df["High"].values
        cur  = float(C[-1])
        adj  = []
        _wt_adj = LEARNED_WEIGHTS.get("adjustments", {})

        def add(label, val):
            adj.append((label, val + _wt_adj.get(label, 0)))

        # ── Volume ────────────────────────────────────────────────────────────
        vs = pr.get("volume_surge", {})
        vr = vs.get("ratio", 0)
        if vr >= 2.0:          add(f"Volume {vr:.1f}x average",        +8)
        elif vr >= 1.5:        add(f"Volume {vr:.1f}x average",        +5)
        elif vr >= 1.2:        add(f"Volume {vr:.1f}x average",        +2)
        elif 0 < vr < 1.0:    add("Breakout on below-avg volume",     -10)
        elif 1.0 <= vr < 1.2: add(f"Weak volume ({vr:.1f}x avg)",     -5)

        # ── BB Squeeze ────────────────────────────────────────────────────────
        bb = pr.get("bb_squeeze", {})
        if bb.get("squeeze_6m"):   add("BB Squeeze (6-month low)",     +7)
        elif bb.get("squeeze_3m"): add("BB Squeeze (3-month low)",     +4)

        # ── SMA Stack ────────────────────────────────────────────────────────
        sma20  = _fcol(df, "SMA_20");  sma50 = _fcol(df, "SMA_50")
        sma200 = _fcol(df, "SMA_200")
        a20  = not sma20.empty  and cur > float(sma20.iloc[-1])
        a50  = not sma50.empty  and cur > float(sma50.iloc[-1])
        a200 = not sma200.empty and cur > float(sma200.iloc[-1])
        if a20 and a50 and a200: add("Full SMA stack aligned (20/50/200)", +6)
        elif a50 and a200:       add("Above SMA50 and SMA200",             +3)
        if not a200:  add("Price below SMA200",     -8)
        elif not a50: add("Price below SMA50",      -5)

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi = _fcol(df, "RSI_")
        rv  = float(rsi.iloc[-1]) if not rsi.empty else None
        if rv is not None:
            if 55 <= rv <= 65:                        add(f"RSI ideal zone ({rv:.0f})",  +5)
            elif (50 <= rv < 55) or (65 < rv <= 70): add(f"RSI good zone ({rv:.0f})",   +2)
            if rv > 80:   add(f"Severely overbought RSI ({rv:.0f})", -8)
            elif rv > 75: add(f"Overbought RSI ({rv:.0f})",          -4)

        # ── ADX ───────────────────────────────────────────────────────────────
        adx = _fcol(df, "ADX_")
        av  = float(adx.iloc[-1]) if not adx.empty else None
        if av is not None:
            if av > 30:          add(f"ADX strong ({av:.0f})",    +5)
            elif 25 <= av <= 30: add(f"ADX moderate ({av:.0f})",  +3)

        # ── Retest ────────────────────────────────────────────────────────────
        if pr.get("retest", {}).get("detected"):
            add("Retest of breakout level confirmed", +8)

        # ── Relative Strength ─────────────────────────────────────────────────
        if self.spy is not None:
            sp = self.spy["Close"].values
            if len(sp) > 30 and len(C) > 30:
                sr30  = (C[-1] - C[-31]) / C[-31]   if C[-31] != 0  else 0
                spr30 = (sp[-1] - sp[-31]) / sp[-31] if sp[-31] != 0 else 0
                if sr30 > spr30:
                    add("Outperforming SPY (30d)", +5)
                elif len(C) > 10 and len(sp) > 10:
                    sr10  = (C[-1] - C[-11]) / C[-11]   if C[-11] != 0  else 0
                    spr10 = (sp[-1] - sp[-11]) / sp[-11] if sp[-11] != 0 else 0
                    if sr10 > spr10:
                        add("Outperforming SPY (10d)", +3)

        # ── Pattern Quality ───────────────────────────────────────────────────
        vcp = pr.get("vcp", {})
        if vcp.get("detected"):
            fd = vcp.get("final_depth", 1)
            if fd < 0.04: add("VCP final contraction < 4% (extremely tight)", +6)
            if fd > 0.12: add("VCP final contraction widening (invalid)",     -5)

        htf = pr.get("htf", {})
        if htf.get("detected") and htf.get("pullback", 1) < 0.10:
            add("HTF pullback < 10% (extremely tight)", +4)

        cup = pr.get("cup_and_handle", {})
        if cup.get("detected"):
            if cup.get("handle_retrace", 1) < 0.10: add("Cup handle < 10% deep", +4)
            if cup.get("v_shaped"):                 add("Cup is V-shaped",        -5)

        db = pr.get("double_bottom", {})
        if db.get("detected") and db.get("undercut"):
            add("Double Bottom shakeout confirmed",  +3)

        wy = pr.get("wyckoff", {})
        if wy.get("detected") and wy.get("spring_vr", 1) < 0.5 and wy.get("lps"):
            add("Wyckoff Spring very low vol + LPS", +5)

        bf = pr.get("bull_flag", {})
        if bf.get("detected") and bf.get("retrace_pct", 0) > 0.50:
            add("Bull Flag retraced > 50% of flagpole", -7)

        at = pr.get("ascending_triangle", {})
        if at.get("detected") and at.get("touches", 3) < 3:
            add("Ascending Triangle < 3 resistance touches", -4)

        # ── 52-Week High Proximity ────────────────────────────────────────────
        h252 = float(H[-252:].max()) if len(H) >= 252 else float(H.max())
        p52  = (h252 - cur) / h252 if h252 > 0 else 1
        if p52 <= 0.01:   add("Within 1% of 52-week high", +5)
        elif p52 <= 0.03: add(f"Within {p52:.0%} of 52-week high", +3)
        elif p52 <= 0.05: add(f"Within {p52:.0%} of 52-week high", +1)

        # ── Four-Layer Validation ─────────────────────────────────────────────
        if pr.get("four_layer", {}).get("all_passed"):
            add("All four validation layers passed", +6)

        # ── Weekly Chart (approximate) ────────────────────────────────────────
        try:
            weekly = df["Close"].resample("W").last().dropna()
            if len(weekly) >= 10:
                wsma10 = float(weekly.rolling(10).mean().iloc[-1])
                if float(weekly.iloc[-1]) > wsma10:
                    add("Weekly chart bullish (W close > 10-week SMA)", +4)
        except Exception:
            pass

        # ── Broad Market ─────────────────────────────────────────────────────
        if self.spy is not None:
            sp = self.spy["Close"].values
            if len(sp) >= 50:
                sma50_val = float(sp[-50:].mean())
                if sp[-1] < sma50_val:
                    add("SPY below its 50-day SMA (headwind)", -8)
                elif len(sp) >= 4 and all(sp[-i] < sp[-i - 1] for i in range(1, 4)):
                    add("SPY down 3+ consecutive days",         -4)

        # ── ATR Expansion ─────────────────────────────────────────────────────
        atr = _fcol(df, "ATRr")
        if len(atr) >= 14:
            if float(atr.iloc[-1]) > float(atr.iloc[-14:-7].mean()) * 1.10:
                add("ATR expanding (volatility rising, not coiling)", -5)

        # ── Earnings Risk ─────────────────────────────────────────────────────
        ed = pr.get("earnings_days", None)
        if ed is not None and 0 <= ed <= 5:
            add(f"Earnings in {ed} days (binary event)", -10)

        # ── Late-Stage Base ───────────────────────────────────────────────────
        try:
            H252 = H[-252:] if len(H) >= 252 else H
            h_top, _ = _swings(H252, order=10)
            sma50_v  = _fcol(df, "SMA_50").values
            n_bases  = sum(1 for hi in h_top[:-1]
                           if hi < len(sma50_v) and H252[hi] > sma50_v[-(len(H252) - hi)])
            if n_bases >= 3:
                add(f"Late-stage base (approx #{n_bases + 1})", -6)
        except Exception:
            pass

        # ── MACD ─────────────────────────────────────────────────────────────
        try:
            if "MACD_line" in df.columns and "MACD_signal" in df.columns:
                ml  = df["MACD_line"].dropna();    ms = df["MACD_signal"].dropna()
                mh  = df["MACD_hist"].dropna() if "MACD_hist" in df.columns else pd.Series(dtype=float)
                if len(ml) >= 2 and len(ms) >= 2:
                    mv = float(ml.iloc[-1]); sv = float(ms.iloc[-1])
                    hv = float(mh.iloc[-1]) if not mh.empty else mv - sv
                    hp = float(mh.iloc[-2]) if len(mh) >= 2 else hv
                    if mv > sv and mv > 0:
                        add("MACD above signal & zero line (strong momentum)", +6)
                    elif mv > sv and mv < 0:
                        add("MACD bullish crossover (momentum turning up)", +4)
                    elif mv < sv:
                        add("MACD below signal line (bearish momentum)",      -5)
                    if hv > 0 and hv > hp:
                        add("MACD histogram expanding (acceleration)",         +4)
                    elif hv < 0 and hv < hp:
                        add("MACD histogram contracting negative (weakening)", -3)
        except Exception:
            pass

        # ── OBV ──────────────────────────────────────────────────────────────
        try:
            if "OBV" in df.columns and "OBV_SMA20" in df.columns:
                ov  = float(df["OBV"].iloc[-1])
                osv = df["OBV_SMA20"].dropna()
                if not osv.empty:
                    if ov > float(osv.iloc[-1]):
                        add("OBV above 20-SMA — institutional buying confirmed", +5)
                    else:
                        add("OBV below 20-SMA — distribution underway",          -4)
        except Exception:
            pass

        # ── MFI ──────────────────────────────────────────────────────────────
        try:
            if "MFI_14" in df.columns:
                mfi_s = df["MFI_14"].dropna()
                if not mfi_s.empty:
                    mv = float(mfi_s.iloc[-1])
                    if 40 <= mv <= 60:
                        add(f"MFI neutral-accumulation zone ({mv:.0f})", +3)
                    elif 60 < mv <= 80:
                        add(f"MFI bullish zone ({mv:.0f})",               +4)
                    elif mv > 80:
                        add(f"MFI overbought ({mv:.0f})",                 -4)
                    elif mv < 20:
                        add(f"MFI oversold ({mv:.0f}) — reversal setup",  +2)
        except Exception:
            pass

        # ── Stochastic RSI ────────────────────────────────────────────────────
        try:
            if "SRSI_K" in df.columns and "SRSI_D" in df.columns:
                k_s = df["SRSI_K"].dropna(); d_s = df["SRSI_D"].dropna()
                if len(k_s) >= 2 and len(d_s) >= 2:
                    kv = float(k_s.iloc[-1]); dv = float(d_s.iloc[-1])
                    kp = float(k_s.iloc[-2]); dp = float(d_s.iloc[-2])
                    if kv > dv and kp <= dp and kv < 80:
                        add("StochRSI golden cross K > D (fresh bullish signal)", +5)
                    elif kv > dv and 20 <= kv < 80:
                        add("StochRSI bullish (K > D, room to run)",               +3)
                    elif kv > 80:
                        add(f"StochRSI overbought ({kv:.0f})",                    -4)
                    elif kv < 20 and kv > dv:
                        add("StochRSI oversold turning up",                        +2)
        except Exception:
            pass

        # ── Weekly Chart Alignment ────────────────────────────────────────────
        try:
            if "W_SMA10" in df.columns:
                w10 = df["W_SMA10"].dropna()
                if not w10.empty:
                    wv10 = float(w10.iloc[-1])
                    if wv10 > 0 and cur > wv10:
                        if "W_SMA20" in df.columns:
                            w20 = df["W_SMA20"].dropna()
                            if not w20.empty:
                                wv20 = float(w20.iloc[-1])
                                if wv20 > 0 and cur > wv20 and wv10 > wv20:
                                    add("Weekly chart bullish — above W-SMA10 & W-SMA20 (stack)", +6)
                                else:
                                    add("Weekly chart bullish — above W-SMA10",                    +3)
                            else:
                                add("Weekly chart bullish — above W-SMA10",                        +3)
                        else:
                            add("Weekly chart bullish — above W-SMA10",                            +3)
                    elif wv10 > 0:
                        add("Weekly chart bearish — below W-SMA10 (weekly downtrend)",             -5)
        except Exception:
            pass

        # ── RS Rating — 12-month relative strength vs SPY ────────────────────
        try:
            if self.spy is not None:
                sp_v = self.spy["Close"].values
                if len(C) >= 252 and len(sp_v) >= 252:
                    rs_12m  = float(C[-1]    / C[-252]    - 1)
                    spy_12m = float(sp_v[-1] / sp_v[-252] - 1)
                    diff    = rs_12m - spy_12m
                    if diff > 0.30:
                        add(f"RS elite: +{diff:.0%} vs SPY (12m)",      +8)
                    elif diff > 0.15:
                        add(f"RS strong: +{diff:.0%} vs SPY (12m)",     +5)
                    elif diff > 0.02:
                        add(f"RS outperforming SPY +{diff:.0%} (12m)",  +3)
                    elif diff < -0.25:
                        add(f"RS laggard: {diff:.0%} vs SPY (12m)",     -7)
                    elif diff < -0.10:
                        add(f"RS lagging SPY {diff:.0%} (12m)",         -4)
        except Exception:
            pass

        # ── Pre-market gap ────────────────────────────────────────────────────
        try:
            pg = self._pm_gap
            if pg >= 0.05:
                add(f"Pre-market gap up {pg:.1%} — strong catalyst",    +8)
            elif pg >= 0.03:
                add(f"Pre-market gap up {pg:.1%}",                      +5)
            elif pg >= 0.01:
                add(f"Pre-market gap up {pg:.1%} (mild)",               +2)
            elif pg <= -0.05:
                add(f"Pre-market gap DOWN {pg:.1%} — red flag",         -9)
            elif pg <= -0.02:
                add(f"Pre-market gap down {pg:.1%}",                    -5)
        except Exception:
            pass

        total = sum(v for _, v in adj)
        final = max(5, min(95, base + total))
        self._mods = adj
        self._n    = len(adj)
        return final, adj

    def _band(self):
        n = self._n
        if n >= 8: return 5,  "HIGH"
        if n >= 5: return 10, "MEDIUM"
        if n >= 3: return 15, "LOW"
        return 20, "SPECULATIVE"

    def calculate_probability(self, earnings_days=None) -> dict:
        if earnings_days is not None:
            self.pr["earnings_days"] = earnings_days
        base, primary, all_pats = self._base_rate()
        final, adj = self._apply_mods(base)
        band, label = self._band()
        pos = [(l, v) for l, v in adj if v > 0]
        neg = [(l, v) for l, v in adj if v < 0]
        return {
            "probability":     final,
            "base_rate":       base,
            "primary_pattern": primary,
            "all_patterns":    all_pats,
            "positive_mods":   pos,
            "negative_mods":   neg,
            "band":            band,
            "band_label":      label,
            "signals_count":   self._n,
            "display":         f"{final}% ± {band}% [{label}]",
        }


# ══════════════════════════════════════════════════════════════════════════════
# BREAKOUT SCANNER
# ══════════════════════════════════════════════════════════════════════════════
class BreakoutScanner:
    """Orchestrates: fetch → indicators → detect → score → probability → output."""

    def __init__(self, cfg: dict = None):
        self.cfg     = cfg or CONFIG
        self.spy     = None
        self.results = []
        self.args    = None

    # ── Data ──────────────────────────────────────────────────────────────────

    def _fetch_sp500(self) -> list:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            return pd.read_html(url)[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception as e:
            print(f"  [warn] Could not fetch S&P 500: {e} — using 30-ticker fallback")
            return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","UNH",
                    "XOM","PG","MA","HD","CVX","MRK","LLY","ABBV","KO","PEP",
                    "COST","AVGO","CSCO","TMO","ACN","MCD","ABT","CRM","NKE","ADBE"]

    def _fetch_ticker(self, ticker: str):
        try:
            end   = datetime.now()
            start = end - timedelta(days=self.cfg["lookback_days"] + 60)
            raw   = yf.download(ticker, start=start, end=end,
                                progress=False, auto_adjust=True)
            if raw.empty or len(raw) < 60:
                return None, None
            df = _norm(raw)

            ed = None
            try:
                cal = yf.Ticker(ticker).calendar
                if cal is not None:
                    if hasattr(cal, "empty") and not cal.empty:
                        d = pd.to_datetime(cal.iloc[0, 0])
                        days = (d - pd.Timestamp.now()).days
                        if 0 <= days <= 30:
                            ed = int(days)
                    elif isinstance(cal, dict):
                        for k in ("Earnings Date", "earningsDate"):
                            if k in cal:
                                d = pd.to_datetime(cal[k])
                                if hasattr(d, "__iter__"):
                                    d = list(d)[0]
                                days = (pd.Timestamp(d) - pd.Timestamp.now()).days
                                if 0 <= days <= 30:
                                    ed = int(days)
                                break
            except Exception:
                pass
            return df, ed
        except Exception:
            return None, None

    def _fetch_spy(self):
        try:
            end   = datetime.now()
            start = end - timedelta(days=self.cfg["lookback_days"] + 60)
            raw   = yf.download("SPY", start=start, end=end,
                                progress=False, auto_adjust=True)
            return _norm(raw) if not raw.empty else None
        except Exception:
            return None

    # ── Indicators ────────────────────────────────────────────────────────────

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for p in [20, 50, 200]:
            df[f"SMA_{p}"] = df["Close"].rolling(p).mean()
        for p in [10, 21]:
            df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
        try:
            df.ta.rsi(length=14, append=True)
        except Exception:
            pass
        try:
            df.ta.bbands(length=20, std=2, append=True)
        except Exception:
            r = df["Close"].rolling(20)
            df["BBM_20_2.0"] = r.mean()
            df["BBU_20_2.0"] = r.mean() + 2 * r.std()
            df["BBL_20_2.0"] = r.mean() - 2 * r.std()
        try:
            df.ta.atr(length=14, append=True)
        except Exception:
            hl = df["High"] - df["Low"]
            hc = (df["High"] - df["Close"].shift()).abs()
            lc = (df["Low"]  - df["Close"].shift()).abs()
            df["ATRr_14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=14).mean()
        try:
            df.ta.adx(length=14, append=True)
        except Exception:
            pass
        # Anchored VWAP — anchored 63 bars back (≈ start of current base)
        try:
            n_anc = max(0, len(df) - 63)
            tp    = (df["High"] + df["Low"] + df["Close"]) / 3
            tp_s  = tp.iloc[n_anc:]
            v_s   = df["Volume"].iloc[n_anc:]
            avwap = (tp_s * v_s).cumsum() / v_s.cumsum().replace(0, np.nan)
            df["AVWAP_63"] = np.nan
            df.loc[avwap.index, "AVWAP_63"] = avwap.values
        except Exception:
            pass

        # ── MACD (12, 26, 9) ─────────────────────────────────────────────────
        try:
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD_line"]   = ema12 - ema26
            df["MACD_signal"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
            df["MACD_hist"]   = df["MACD_line"] - df["MACD_signal"]
        except Exception:
            pass

        # ── OBV (On-Balance Volume) ───────────────────────────────────────────
        try:
            closes = df["Close"].values
            vols   = df["Volume"].values.astype(float)
            obv    = np.zeros(len(closes))
            for i in range(1, len(closes)):
                if closes[i] > closes[i - 1]:
                    obv[i] = obv[i - 1] + vols[i]
                elif closes[i] < closes[i - 1]:
                    obv[i] = obv[i - 1] - vols[i]
                else:
                    obv[i] = obv[i - 1]
            df["OBV"]      = obv
            df["OBV_SMA20"] = pd.Series(obv, index=df.index).rolling(20).mean()
        except Exception:
            pass

        # ── MFI — Money Flow Index (14) ───────────────────────────────────────
        try:
            tp_mfi  = (df["High"] + df["Low"] + df["Close"]) / 3
            mf_raw  = tp_mfi * df["Volume"]
            pos_mf  = mf_raw.where(tp_mfi > tp_mfi.shift(1), 0.0)
            neg_mf  = mf_raw.where(tp_mfi < tp_mfi.shift(1), 0.0)
            pmf14   = pos_mf.rolling(14).sum()
            nmf14   = neg_mf.rolling(14).sum()
            mfr     = pmf14 / nmf14.replace(0, np.nan)
            df["MFI_14"] = 100 - (100 / (1 + mfr))
        except Exception:
            pass

        # ── Stochastic RSI (14, 3, 3) ─────────────────────────────────────────
        try:
            rsi_s = _fcol(df, "RSI_")
            if not rsi_s.empty and len(rsi_s) >= 17:
                rsi_a = rsi_s.reindex(df.index).ffill()
                rmin  = rsi_a.rolling(14).min()
                rmax  = rsi_a.rolling(14).max()
                stoch = (rsi_a - rmin) / (rmax - rmin + 1e-8) * 100
                df["SRSI_K"] = stoch.rolling(3).mean()
                df["SRSI_D"] = df["SRSI_K"].rolling(3).mean()
        except Exception:
            pass

        # ── Weekly SMA 10 + 20 (mapped back to daily) ────────────────────────
        try:
            wkly = df["Close"].resample("W").last().dropna()
            if len(wkly) >= 10:
                df["W_SMA10"] = wkly.rolling(10).mean().reindex(
                    df.index, method="ffill")
            if len(wkly) >= 20:
                df["W_SMA20"] = wkly.rolling(20).mean().reindex(
                    df.index, method="ffill")
        except Exception:
            pass

        return df

    # ── Composite Score ───────────────────────────────────────────────────────

    def _score(self, df: pd.DataFrame, pr: dict) -> float:
        C   = df["Close"].values; H = df["High"].values; cur = float(C[-1])
        sc  = 0.0
        tier_map = [
            ("vcp", 40), ("cup_and_handle", 40), ("wyckoff", 40),
            ("htf", 30), ("bull_flag", 30), ("pennant", 30),
            ("ascending_triangle", 20), ("flat_base", 20),
            ("double_bottom", 20), ("symmetrical_triangle", 20),
        ]
        pts = [wt * p.get("confidence", 0.5)
               for key, wt in tier_map if (p := pr.get(key, {})).get("detected")]
        if pts:
            pts.sort(reverse=True)
            sc += min(40, pts[0] + (pts[1] * 0.5 if len(pts) > 1 else 0))

        rsi = _fcol(df, "RSI_")
        if not rsi.empty and 50 <= float(rsi.iloc[-1]) <= 70: sc += 10
        sma20 = _fcol(df, "SMA_20"); sma50 = _fcol(df, "SMA_50")
        if not sma20.empty and cur > float(sma20.iloc[-1]): sc += 5
        if not sma50.empty and cur > float(sma50.iloc[-1]): sc += 5

        vr = pr.get("volume_surge", {}).get("ratio", 0)
        sc += min(20, (vr / 2.0) * 20)
        if pr.get("bb_squeeze", {}).get("squeeze_6m") or pr.get("bb_squeeze", {}).get("squeeze_3m"):
            sc += 5

        atr = _fcol(df, "ATRr")
        h52 = float(H[-252:].max()) if len(H) >= 252 else float(H.max())
        if not atr.empty:
            av = float(atr.iloc[-1])
            td = h52 - cur
            if av > 0 and td / (av * 1.5) >= 2: sc += 10
        if pr.get("retest", {}).get("detected"): sc += 10

        adx = _fcol(df, "ADX_")
        if not adx.empty and float(adx.iloc[-1]) > 25: sc += 5
        if self.spy is not None and len(self.spy) > 30 and len(C) > 30:
            sp = self.spy["Close"].values
            if len(sp) > 30:
                sr  = (C[-1] - C[-31]) / C[-31]   if C[-31] != 0  else 0
                spr = (sp[-1] - sp[-31]) / sp[-31] if sp[-31] != 0 else 0
                if sr > spr: sc += 5
        return min(100.0, sc)

    # ── Trade Thesis ──────────────────────────────────────────────────────────

    def _thesis(self, r: dict) -> str:
        pat = r["pattern"]; pr = r.get("patterns", {}); cur = r["price"]
        key = ("VCP"               if "VCP"          in pat else
               "Cup and Handle"    if "Cup"           in pat else
               "Wyckoff Spring"    if "Wyckoff"       in pat else
               "High and Tight Flag" if "High"        in pat else
               "Bull Flag"         if "Bull Flag"     in pat else
               "Pennant"           if "Pennant"       in pat else
               "Ascending Triangle" if "Ascending"    in pat else
               "Flat Base"         if "Flat"          in pat else
               "Double Bottom"     if "Double"        in pat else
               "Symmetrical Triangle" if "Symmetrical" in pat else "No Pattern")
        tmpl = THESES.get(key, THESES["No Pattern"])

        vcp_p = pr.get("vcp", {}); cup_p = pr.get("cup_and_handle", {})
        htf_p = pr.get("htf", {}); bf_p  = pr.get("bull_flag", {})
        pn_p  = pr.get("pennant", {}); at_p = pr.get("ascending_triangle", {})
        fb_p  = pr.get("flat_base", {}); db_p = pr.get("double_bottom", {})
        sym_p = pr.get("symmetrical_triangle", {})

        gain_val = htf_p.get("pole_gain", bf_p.get("pole_gain", pn_p.get("pole_gain", 0.20)))
        fmt = {
            "ticker":   r["ticker"],
            "n_cont":   vcp_p.get("n_contractions", 2),
            "depths":   vcp_p.get("depths", "N/A"),
            "pivot":    vcp_p.get("pivot", cup_p.get("pivot", cur * 1.02)),
            "depth":    cup_p.get("cup_depth", 0.30),
            "weeks":    cup_p.get("weeks", fb_p.get("weeks", 10)),
            "handle":   cup_p.get("handle_retrace", 0.10),
            "gain":     gain_val,
            "pullback": htf_p.get("pullback", 0.15),
            "retrace":  bf_p.get("retrace_pct", 0.35),
            "apex":     pn_p.get("apex_pct", sym_p.get("apex_pct", 0.70)),
            "resist":   at_p.get("resistance", cur * 1.02),
            "touches":  at_p.get("touches", 3),
            "range":    fb_p.get("range_pct", 0.10),
            "low":      db_p.get("low2", db_p.get("low1", cur * 0.95)),
            "neck":     db_p.get("neckline", cur * 1.02),
            "div":      "bullish RSI divergence and " if db_p.get("rsi_div") else "",
        }
        try:
            return tmpl.format(**fmt)
        except Exception:
            return (f"{r['ticker']} shows a {pat} setup with {r['probability']}% "
                    "breakout probability. Enter on volume confirmation above the "
                    "pivot and stop below the nearest structure low.")

    # ── Per-ticker pipeline ────────────────────────────────────────────────────

    def _process_with_info(self, item: dict, catalyst: dict = None):
        """Full pipeline when we already have .info from quick_filter."""
        import traceback as _tb
        ticker = item["ticker"]
        info   = item.get("info", {})
        raw, ed = self._fetch_ticker(ticker)
        if raw is None:
            if getattr(self, "_debug", False):
                print(f"  [DEBUG] {ticker}: _fetch_ticker returned None")
            return None
        try:
            df   = self._indicators(raw)
            det  = PatternDetector(df, self.spy, self.cfg)
            pats = det.run_all()
            eng  = BreakoutProbabilityEngine(pats, df, self.spy, self.cfg)
            prob = eng.calculate_probability(earnings_days=ed)
            sc   = self._score(df, pats)
            news_sc = (catalyst or {}).get("catalyst_score")
            expl = ExplosiveMoveDetector(df, info, self.spy, pats, prob).calculate(news_sc)
            # ── New signal layers ──────────────────────────────────────────
            eq   = EarningsQualityEngine().score(ticker)
            flow = OptionsFlowScorer().score(ticker)
            # Pre-market gap from info dict (already fetched in quick_filter)
            pre   = float(info.get("preMarketPrice", 0) or 0)
            prev  = float(info.get("previousClose",  0) or 0)
            pm_gap = (pre - prev) / prev if pre > 0 and prev > 0 else 0.0
            # 12-month return for RS Rating (calculated post-scan)
            C   = df["Close"].values
            rs_12m = float(C[-1] / C[-252] - 1) if len(C) >= 252 else float(C[-1] / C[0] - 1)
        except Exception as e:
            if getattr(self, "_debug", False):
                print(f"  [DEBUG] {ticker}: {e}")
                _tb.print_exc()
            return None
        return self._build_result(ticker, df, pats, prob, sc, ed, expl,
                                  catalyst, info, eq, flow, pm_gap, rs_12m)

    def _init_sector_rotation(self):
        """Warm up the sector rotation cache before the parallel scan starts."""
        try:
            print("  Fetching sector rotation data (SPDR ETFs)...")
            rankings = SectorRotationDetector.get()
            if rankings:
                top = SectorRotationDetector.top_sectors(4)
                print(f"  Leading sectors: {', '.join(top)}")
            else:
                print("  Sector data unavailable — rotation filter disabled.")
        except Exception:
            pass

    def _process(self, ticker: str):
        raw, ed = self._fetch_ticker(ticker)
        if raw is None:
            return None
        try:
            df   = self._indicators(raw)
            det  = PatternDetector(df, self.spy, self.cfg)
            pats = det.run_all()
            eng  = BreakoutProbabilityEngine(pats, df, self.spy, self.cfg)
            prob = eng.calculate_probability(earnings_days=ed)
            sc   = self._score(df, pats)
            expl = ExplosiveMoveDetector(df, {}, self.spy, pats, prob).calculate()
        except Exception:
            return None
        return self._build_result(ticker, df, pats, prob, sc, ed, expl, None, {})

    def _build_result(self, ticker, df, pats, prob, sc, ed, expl,
                      catalyst, info, eq=None, flow=None, pm_gap=0.0, rs_12m=0.0):
        C   = df["Close"].values; H = df["High"].values
        cur = float(C[-1])
        h52 = float(H[-252:].max()) if len(H) >= 252 else float(H.max())
        p52 = (h52 - cur) / h52 if h52 > 0 else 0

        rsi = _fcol(df, "RSI_")
        rv  = float(rsi.iloc[-1]) if not rsi.empty else None

        atr = _fcol(df, "ATRr")
        av  = float(atr.iloc[-1]) if not atr.empty else cur * 0.02
        # Dynamic ATR stop: 2x ATR below entry (standard swing-trade buffer)
        sd  = av * 2.0
        # Primary target: 52-week high; secondary (3:1) target for tight setups
        td_52w = h52 - cur
        td_3x  = sd * 3.0                          # guaranteed 3:1 R/R
        td     = max(td_52w, td_3x)                # use whichever is larger
        rr     = td / sd if sd > 0 else 0
        # T1 partial target: 1.5× risk (book 50% at T1, let rest run to target)
        t1_price = cur + sd * 1.5

        primary = prob["primary_pattern"]
        sig_sum = primary
        for k in ["vcp","cup_and_handle","wyckoff","htf","bull_flag","pennant",
                  "ascending_triangle","flat_base","double_bottom","symmetrical_triangle"]:
            pp = pats.get(k, {})
            if pp.get("detected") and pp.get("name","") == primary:
                sig_sum = pp.get("summary", primary)[:65]; break

        erisk = ("Yes"  if ed is not None and ed <= 5  else
                 "Soon" if ed is not None and ed <= 10 else "No")

        mkt_cap = float(info.get("marketCap", 0) or 0)

        result = {
            "ticker":         ticker,   "price":        cur,   "score":       sc,
            "probability":    prob["probability"],
            "prob_display":   prob["display"],
            "conf_band":      prob["band"],
            "conf_label":     prob["band_label"],
            "pattern":        primary,  "rsi":          rv,
            "vol_ratio":      pats.get("volume_surge",{}).get("ratio",0),
            "bb_squeeze":     (pats.get("bb_squeeze",{}).get("squeeze_6m") or
                               pats.get("bb_squeeze",{}).get("squeeze_3m")),
            "pct_52w":        p52,      "rr":           rr,
            "stop_price":     cur - sd, "tgt_price":    cur + td,
            "stop_pct":       sd / cur if cur > 0 else 0,
            "earnings_days":  ed,       "earnings_risk": erisk,
            "signal_summary": sig_sum,
            "prob_detail":    prob,     "patterns":     pats,
            "market_cap":     mkt_cap,
            # Explosive
            "explosive_score":  expl["score"],
            "explosive_grade":  expl["grade"],
            "move_low":         expl["move_low"],
            "move_high":        expl["move_high"],
            "expl_short_pct":   expl["short_pct"],
            "expl_float":       expl["float_sh"],
            "expl_short_ratio": expl["short_ratio"],
            "expl_comp_pts":    expl["compression_pts"],
            "expl_squeeze_pts": expl["squeeze_pts"],
            "expl_cat_pts":     expl["catalyst_pts"],
            "expl_comp_det":    expl["compression_det"],
            "expl_squeeze_det": expl["squeeze_det"],
            "expl_cat_det":     expl["catalyst_det"],
            # Catalyst
            "catalyst_score":   (catalyst or {}).get("catalyst_score", 0),
            "catalyst_flags":   (catalyst or {}).get("catalyst_flags", []),
            "top_flag":         (catalyst or {}).get("top_flag", ""),
            "top_headline":     (catalyst or {}).get("top_headline", ""),
            # Earnings Quality
            "eq_score":         (eq or {}).get("eq_score",       50),
            "eq_grade":         (eq or {}).get("eq_grade",       "N/A"),
            "eq_acceleration":  (eq or {}).get("eq_acceleration",False),
            "eq_detail":        (eq or {}).get("eq_detail",      ""),
            # Options Flow
            "flow_score":       (flow or {}).get("flow_score",   0),
            "flow_signal":      (flow or {}).get("flow_signal",  ""),
            "cp_ratio":         (flow or {}).get("cp_ratio",     0.0),
            "unusual_calls":    (flow or {}).get("unusual_calls",False),
            "flow_detail":      (flow or {}).get("flow_detail",  ""),
            # Pre-market gap & VWAP
            "pm_gap":           pm_gap,
            "rs_12m":           rs_12m,
            "rs_rating":        50,   # placeholder — set post-scan in run()
            # Anchored VWAP vs price
            "above_avwap":      self._above_avwap(df),
            # ATR-based sizing helpers
            "atr_value":        av,
            "t1_price":         t1_price,   # partial exit at 1.5× risk
        }
        result["thesis"] = self._thesis(result)
        return result

    def _above_avwap(self, df: pd.DataFrame) -> bool:
        """True if current close is above the 63-bar anchored VWAP."""
        try:
            avwap = df["AVWAP_63"].dropna()
            if avwap.empty:
                return False
            return float(df["Close"].iloc[-1]) > float(avwap.iloc[-1])
        except Exception:
            return False

    # ── Market Context ────────────────────────────────────────────────────────

    def _market_context(self):
        if self.spy is None:
            return 0, "N/A", "N/A"
        sp = self.spy["Close"].values
        if len(sp) < 50:
            return 0, "Insufficient data", "N/A"
        sma50 = float(sp[-50:].mean()); cur = float(sp[-1])
        pct   = (cur - sma50) / sma50

        if cur < sma50:
            spy_str = f"BELOW -{abs(pct):.1%}"; mod = -8
        else:
            spy_str = f"ABOVE +{pct:.1%}"; mod = 0

        if len(sp) >= 4 and all(sp[-i] < sp[-i - 1] for i in range(1, 4)):
            three_d = "DOWN"
            if mod == 0: mod = -4
        elif len(sp) >= 2 and sp[-1] > sp[-2]:
            three_d = "UP"
        else:
            three_d = "FLAT"

        return mod, spy_str, three_d

    # ── Output ────────────────────────────────────────────────────────────────

    def _print_market_box(self, n_scanned, n_pats):
        mod, spy_vs, three_d = self._market_context()
        top_p = self.results[0]["probability"] if self.results else 0

        spy_ok = "OK" if "ABOVE" in spy_vs else "!!"
        td_ok  = "OK" if three_d == "UP" else ("!!" if three_d == "DOWN" else "--")

        raw      = getattr(self, "_universe_raw",      n_scanned)
        filtered = getattr(self, "_universe_filtered",  n_scanned)
        advanced = getattr(self, "_universe_advanced",  n_scanned)

        lines = [
            f"  SPY vs SMA50      :  {spy_vs} [{spy_ok}]",
            f"  SPY 3-day trend   :  {three_d} [{td_ok}]",
            f"  Market modifier   :  {mod:+d}%",
            f"  Universe (raw)    :  {raw} tickers fetched",
            f"  Quick filter      :  {filtered} passed  →  {advanced} fully analyzed",
            f"  Qualified results :  {n_scanned}   Patterns found: {n_pats}",
            f"  Highest prob      :  {top_p}%",
        ]
        w = max(len(l) for l in lines) + 1
        sep = "+" + "-" * (w + 2) + "+"
        print(sep)
        print("|" + "  MARKET CONTEXT".ljust(w + 2) + "|")
        print(sep)
        for ln in lines:
            print("|  " + ln.ljust(w) + "|")
        print(sep)
        print()

    def _print_table(self, top_n: int):
        headers = ["#", "Ticker", "Price", "RS", "EQ", "Flow",
                   "Breakout Prob", "Pattern", "AVWAP", "PM Gap",
                   "%<52WH", "R:R", "Earn", "Signal Summary"]
        rows = []
        for i, r in enumerate(self.results[:top_n], 1):
            pm = r.get("pm_gap", 0)
            pm_str = f"+{pm:.1%}" if pm > 0.005 else ("" if pm == 0 else f"{pm:.1%}")
            rows.append([
                i, r["ticker"], f"${r['price']:.2f}",
                r.get("rs_rating", "--"),
                r.get("eq_grade", "N/A"),
                f"{r.get('flow_score',0)}" if r.get("flow_score",0) > 0 else "--",
                r["prob_display"], r["pattern"][:16],
                "▲" if r.get("above_avwap") else "▼",
                pm_str if pm_str else "--",
                f"{r['pct_52w']:.1%}", f"{r['rr']:.1f}:1",
                r["earnings_risk"],
                r["signal_summary"][:36],
            ])
        try:
            print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()

    def _print_breakdown(self, r: dict, rank: int):
        d   = r["prob_detail"]; cur = r["price"]
        pct = (r["tgt_price"] - cur) / cur if cur > 0 else 0
        sep = "=" * 58

        print(f"\n{sep}")
        print(f"  #{rank}  {r['ticker']}   ({r['pattern']})")
        print(sep)
        print(f"  Base Win Rate    : {d['base_rate']}%")
        if len(d["all_patterns"]) > 1:
            others = ", ".join(f"{p[0]} ({p[1]}%)" for p in d["all_patterns"][1:])
            print(f"  Co-Patterns      : {others}")
        print()

        print("  [+] Positive Modifiers:")
        for lbl, val in d["positive_mods"]:
            print(f"      {lbl:<48} -> +{val}%")
        if not d["positive_mods"]:
            print("      (none)")

        print()
        print("  [-] Negative Modifiers:")
        for lbl, val in d["negative_mods"]:
            print(f"      {lbl:<48} -> {val}%")
        if not d["negative_mods"]:
            print("      (none)")

        print()
        print(f"  {'─'*54}")
        print(f"  BREAKOUT PROBABILITY : {d['display']}")
        print(f"  {'─'*54}")
        print()
        print(f"  Entry Trigger  : ${cur:.2f}  (current / near pivot)")
        print(f"  Stop Loss      : ${r['stop_price']:.2f}  (-{r['stop_pct']:.1%})")
        print(f"  Target Price   : ${r['tgt_price']:.2f}  (+{pct:.1%})")
        print(f"  Reward:Risk    : {r['rr']:.1f} : 1")
        print()

        # ── New signal layers ─────────────────────────────────────────────────
        print(f"  RS Rating      : {r.get('rs_rating', '--')}/99"
              + ("  [TOP DECILE]" if r.get("rs_rating", 0) >= 90 else
                 "  [STRONG]"    if r.get("rs_rating", 0) >= 75 else ""))
        eq_acc = "  ★ ACCELERATING" if r.get("eq_acceleration") else ""
        print(f"  Earnings Qual  : {r.get('eq_grade','N/A')} (score {r.get('eq_score',50)}){eq_acc}")
        if r.get("eq_detail"):
            print(f"    {r['eq_detail']}")
        flow_s = r.get("flow_score", 0)
        if flow_s > 0:
            print(f"  Options Flow   : {flow_s}/100  {r.get('flow_signal','')}")
            if r.get("flow_detail"):
                print(f"    {r['flow_detail']}")
        else:
            print(f"  Options Flow   : No unusual activity")
        pm = r.get("pm_gap", 0)
        if abs(pm) > 0.005:
            print(f"  Pre-market Gap : {pm:+.1%}  {'⚡ CATALYST GAP' if pm > 0.05 else ''}")
        avwap_str = "Above (bullish)" if r.get("above_avwap") else "Below (caution)"
        print(f"  Anchored VWAP  : {avwap_str}  (63-bar base anchor)")
        print()

        thesis = r.get("thesis", "")
        if thesis:
            words  = thesis.split()
            lines  = []
            line   = "  Trade Thesis   : "
            for w in words:
                if len(line) + len(w) + 1 > 72:
                    lines.append(line); line = "                   " + w + " "
                else:
                    line += w + " "
            if line.strip():
                lines.append(line)
            for ln in lines:
                print(ln.rstrip())
        print()

    def _export(self, top_n: int):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_p = f"breakout_scan_{ts}.csv"
        txt_p = f"top15_{ts}.txt"

        if self.results:
            rows = [{
                "Ticker":       r["ticker"],
                "Price":        round(r["price"], 2),
                "Score":        round(r["score"], 1),
                "Breakout_Prob": r["probability"],
                "Confidence":   r["conf_label"],
                "Pattern":      r["pattern"],
                "RSI":          round(r["rsi"], 1) if r["rsi"] else None,
                "Vol_Ratio":    round(r["vol_ratio"], 2),
                "BB_Squeeze":   r["bb_squeeze"],
                "Pct_52W":      round(r["pct_52w"], 4),
                "RR":           round(r["rr"], 2),
                "Stop":         round(r["stop_price"], 2),
                "Target":       round(r["tgt_price"], 2),
                "Earnings":     r["earnings_risk"],
                "Signal":       r["signal_summary"],
            } for r in self.results]
            pd.DataFrame(rows).to_csv(csv_p, index=False)

        old, sys.stdout = sys.stdout, io.StringIO()
        try:
            n_pats = sum(1 for r in self.results if r["pattern"] != "No Pattern")
            self._print_market_box(len(self.results), n_pats)
            self._print_table(top_n)
            for i, r in enumerate(self.results[:5], 1):
                self._print_breakdown(r, i)
            txt_content = sys.stdout.getvalue()
        finally:
            sys.stdout = old

        with open(txt_p, "w", encoding="utf-8") as f:
            f.write(txt_content)

        print(f"  Exported: {csv_p}")
        print(f"  Exported: {txt_p}")
        return csv_p, txt_p

    # ── Explosive output ──────────────────────────────────────────────────────

    def _print_explosive_table(self, top_n: int = 15):
        rows = [r for r in self.results if r["explosive_score"] >= 40]
        rows.sort(key=lambda x: x["explosive_score"], reverse=True)
        if not rows:
            print("  No explosive candidates found (score ≥ 40).\n"); return
        hdrs = ["#","Ticker","Price","MktCap","Float","Short%",
                "Exp Score","Grade","Catalyst","Est Move%","Pattern","Prob","Headline"]
        tbl = []
        for i, r in enumerate(rows[:top_n], 1):
            mc  = r["market_cap"]; fs = r["expl_float"]
            mc_s = f"${mc/1e9:.1f}B" if mc >= 1e9 else (f"${mc/1e6:.0f}M" if mc >= 1e6 else "—")
            fs_s = f"{fs/1e6:.1f}M" if fs >= 1e6 else (f"{fs/1e3:.0f}K" if fs > 0 else "—")
            tbl.append([i, r["ticker"], f"${r['price']:.2f}", mc_s, fs_s,
                        f"{r['expl_short_pct']:.0%}" if r["expl_short_pct"] else "—",
                        f"{r['explosive_score']}/100",
                        r["explosive_grade"],
                        r["top_flag"][:18] if r["top_flag"] else "—",
                        f"+{r['move_low']:.0f}%–{r['move_high']:.0f}%",
                        r["pattern"][:14],
                        r["prob_display"][:12],
                        r["top_headline"][:30] if r["top_headline"] else "—"])
        try:
            print(tabulate(tbl, headers=hdrs, tablefmt="rounded_outline"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            print(tabulate(tbl, headers=hdrs, tablefmt="grid"))
        print()

    def _print_explosive_breakdown(self, r: dict, rank: int):
        sep  = "═" * 58
        sep2 = "─" * 56
        cur  = r["price"]
        pct  = (r["tgt_price"] - cur) / cur if cur > 0 else 0
        sq_t = cur * (1 + r["move_high"] / 100)

        print(f"\n⚡ #{rank}  {r['ticker']} (${cur:.2f})  —  {r['explosive_grade']}")
        print(sep)
        print(f"  Explosive Score  : {r['explosive_score']}/100  {r['explosive_grade']}")
        print(f"  Estimated Move   : +{r['move_low']:.0f}% to +{r['move_high']:.0f}%")
        print(f"  Breakout Prob    : {r['prob_display']}")
        if r["market_cap"]:
            mc = r["market_cap"]
            print(f"  Market Cap       : ${mc/1e9:.2f}B" if mc >= 1e9 else f"  Market Cap       : ${mc/1e6:.0f}M")
        print()

        # Catalyst flags
        if r["catalyst_flags"]:
            print(f"  🚀 CATALYST (score {r['catalyst_score']}/100):")
            for fl in r["catalyst_flags"][:4]:
                sign = "+" if fl["score"] > 0 else ""
                hl   = f'  "{fl["headline"]}"' if fl["headline"] else ""
                print(f"     {fl['flag']}  [{sign}{fl['score']}]{hl}")
            print()

        # Compression
        print(f"  🗜️  COMPRESSION ({r['expl_comp_pts']}/35):")
        for desc, pts in (r["expl_comp_det"] or [("—", 0)]):
            print(f"     {desc:<52} → {pts} pts")

        # Squeeze
        print(f"\n  🔥 SQUEEZE FUEL ({r['expl_squeeze_pts']}/30):")
        for desc, pts in (r["expl_squeeze_det"] or [("—", 0)]):
            print(f"     {desc:<52} → {pts} pts")

        # Technical catalyst
        print(f"\n  📡 TECH CATALYST ({r['expl_cat_pts']}/25):")
        for desc, pts in (r["expl_cat_det"] or [("—", 0)]):
            print(f"     {desc:<52} → {pts} pts")

        print(f"\n  {sep2}")
        print(f"  Entry Trigger    : ${cur:.2f}")
        print(f"  Stop Loss        : ${r['stop_price']:.2f}  (-{r['stop_pct']:.1%})")
        print(f"  Base Target      : ${r['tgt_price']:.2f}  (+{pct:.1%})")
        print(f"  Squeeze Target   : ${sq_t:.2f}  (+{r['move_high']:.0f}%)")
        print(f"  Reward:Risk      : {r['rr']:.1f} : 1")

        thesis = r.get("thesis", "")
        if thesis:
            words, lines, line = thesis.split(), [], "  Thesis: "
            for w in words:
                if len(line) + len(w) + 1 > 72:
                    lines.append(line); line = "          " + w + " "
                else:
                    line += w + " "
            if line.strip(): lines.append(line)
            print()
            for ln in lines: print(ln.rstrip())

        sp = r["expl_short_pct"]; fs = r["expl_float"]
        risks = []
        if sp > 0.20: risks.append(f"High short interest {sp:.0%} — binary squeeze risk")
        if fs > 0 and fs < 5e6: risks.append("Micro float — low liquidity, wide spreads")
        if r["earnings_risk"] in ("Yes","Soon"): risks.append("Earnings event upcoming — binary risk")
        if any(f["score"] < 0 for f in r["catalyst_flags"]):
            neg = [f["flag"] for f in r["catalyst_flags"] if f["score"] < 0]
            risks.append(", ".join(neg))
        if risks:
            print(f"\n  ⚠️  Key Risks: {'; '.join(risks)}")
        print(sep)

    # ── Main entry ────────────────────────────────────────────────────────────

    def _emit_progress(self, phase: int, current: int, total: int, message: str = ""):
        """Invoke an optional progress callback for UI updates.

        Set self.progress_callback = fn(phase, current, total, message) before
        calling run() to receive live progress updates.  Phases are 1-4 mapping
        to Universe / Filter / Catalyst / Analysis.
        """
        cb = getattr(self, "progress_callback", None)
        if cb is None:
            return
        try:
            cb(phase, current, total, message)
        except Exception:
            pass   # never let UI errors break the scan

    def run(self):
        args = self.args or argparse.Namespace(
            pattern=None, watchlist=None, min_prob=0, no_earnings=False,
            top=15, export=True, universe="all", min_explosive=0,
            catalyst_only=False, biotech=False, squeeze=False,
            max_float=None, max_cap=None, no_otc=False)

        t0 = datetime.now()
        self.results = []
        self._universe_raw      = 0
        self._universe_filtered = 0
        self._universe_advanced = 0
        self._emit_progress(1, 0, 1, "Starting scan...")

        # ── Phase 1: Universe ──────────────────────────────────────────────
        self._emit_progress(1, 0, 1, "Building universe...")
        if getattr(args, "watchlist", None):
            tickers_info = [{"ticker": t.strip().upper(), "info": {},
                             "market_cap": 0, "price": 0, "avg_vol": 0,
                             "float_shares": 0, "short_pct": 0,
                             "short_ratio": 0, "sector": ""}
                            for t in args.watchlist.split(",")]
            n = len(tickers_info)
            self._universe_raw = self._universe_filtered = n
            print(f"\n[Phase 1/4] Custom watchlist: {n} tickers")
            self._emit_progress(1, 1, 1, f"Custom watchlist: {n} tickers")
        else:
            print("\n[Phase 1/4] Building universe...")
            ub      = UniverseBuilder()
            raw_t   = ub.build(
                universe_type = getattr(args, "universe", "all"),
                file_path     = getattr(args, "file",     None),
            )
            self._universe_raw = len(raw_t)
            print(f"  Raw universe: {len(raw_t)} tickers")
            self._emit_progress(1, 1, 1, f"Raw universe: {len(raw_t)} tickers")

            # ── Phase 2: Quick filter ──────────────────────────────────────
            self._emit_progress(2, 0, 1, f"Filtering {len(raw_t)} tickers...")
            print(f"[Phase 2/4] Quick filtering {len(raw_t)} tickers...")
            tickers_info = ub.quick_filter(raw_t, args)
            self._universe_filtered = len(tickers_info)
            print(f"  {len(tickers_info)} stocks passed quick filter\n")
            self._emit_progress(2, 1, 1,
                                 f"{len(tickers_info)}/{len(raw_t)} passed filter")

        # ── Phase 3: News catalyst ─────────────────────────────────────────
        cat_threshold = 20 if getattr(args, "catalyst_only", False) else 0
        print(f"[Phase 3/4] News catalyst scan ({len(tickers_info)} stocks)...")
        self._emit_progress(3, 0, len(tickers_info),
                             f"News catalyst scan on {len(tickers_info)} tickers")
        nce      = NewsCatalystEngine()
        catalyst_map: dict = {}
        _cat_done = 0
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(nce.scan, item["ticker"]): item["ticker"]
                    for item in tickers_info}
            with tqdm(total=len(tickers_info), desc="  Catalyst scan",
                      unit="stock", ncols=80) as bar:
                for fut in as_completed(futs):
                    r = fut.result()
                    catalyst_map[r["ticker"]] = r
                    bar.update(1)
                    _cat_done += 1
                    # Throttle UI updates to every 5 tickers to avoid spam
                    if _cat_done % 5 == 0 or _cat_done == len(tickers_info):
                        self._emit_progress(3, _cat_done, len(tickers_info),
                                             f"Catalyst: {r['ticker']} ({_cat_done}/{len(tickers_info)})")

        if getattr(args, "biotech", False):
            BIOTECH_KWS = ["biotech","pharma","therapeutics","biosciences","oncology","health"]
            tickers_info = [x for x in tickers_info
                            if any(k in x.get("sector","").lower() for k in BIOTECH_KWS)
                            or catalyst_map.get(x["ticker"],{}).get("top_flag","").startswith("🧬")]

        advanced = [x for x in tickers_info
                    if catalyst_map.get(x["ticker"],{}).get("catalyst_score",0) >= cat_threshold]
        self._universe_advanced = len(advanced)
        print(f"  {len(advanced)} stocks advancing to full scan\n")

        # ── Phase 4: Full pattern detection ───────────────────────────────
        print(f"[Phase 4/4] Pattern detection + scoring ({len(advanced)} stocks)...")
        self._emit_progress(4, 0, len(advanced),
                             f"Fetching SPY market data...")
        print("Fetching SPY market data...")
        self.spy    = self._fetch_spy()
        self.regime = MarketRegimeDetector.detect(self.spy)
        print(f"  Market regime: {self.regime['regime']}  |  {self.regime['advice']}")
        self._init_sector_rotation()
        self._emit_progress(4, 0, len(advanced),
                             f"Analyzing {len(advanced)} candidates "
                             f"(regime: {self.regime['regime']})")

        _analyzed = 0
        with ThreadPoolExecutor(max_workers=self.cfg["max_workers"]) as ex:
            futs = {ex.submit(self._process_with_info, item,
                              catalyst_map.get(item["ticker"])): item["ticker"]
                    for item in advanced}
            with tqdm(total=len(advanced), desc="  Analyzing",
                      unit="stock", ncols=80) as bar:
                for fut in as_completed(futs):
                    _analyzed += 1
                    # Surface progress every ticker so the user sees live activity.
                    _tk_done = futs.get(fut, "?")
                    self._emit_progress(4, _analyzed, len(advanced),
                                         f"Analyzed {_tk_done} ({_analyzed}/{len(advanced)})")
                    try:
                        r = fut.result()
                        if r is None:
                            continue
                        min_prob  = getattr(args, "min_prob", 0)
                        min_expl  = getattr(args, "min_explosive", 0)
                        min_rs    = getattr(args, "min_rs", 0)
                        no_earn   = getattr(args, "no_earnings", False)
                        eq_only   = getattr(args, "earnings_quality", False)
                        pm_only   = getattr(args, "premarket", False)
                        pat_filt  = getattr(args, "pattern", None)
                        sq_only   = getattr(args, "squeeze", False)
                        if r["probability"] < min_prob: continue
                        if r["explosive_score"] < min_expl: continue
                        if no_earn and r["earnings_risk"] == "Yes": continue
                        if pat_filt and pat_filt.lower() not in r["pattern"].lower(): continue
                        if sq_only and r["expl_short_pct"] < 0.10: continue
                        if eq_only and r["eq_score"] < 65: continue
                        if pm_only and r["pm_gap"] < 0.03: continue
                        self.results.append(r)
                    except Exception as _e:
                        if getattr(self, "_debug", False):
                            import traceback as _tb2
                            print(f"  [DEBUG phase4] {_e}")
                            _tb2.print_exc()
                    finally:
                        bar.update(1)

        elapsed = (datetime.now() - t0).total_seconds()
        self._emit_progress(4, len(advanced), len(advanced),
                             f"Complete: {len(self.results)} qualified in {elapsed:.1f}s")

        # ── RS Rating: rank 1-99 by 12-month return vs all results ───────────
        if self.results:
            all_rs = [r["rs_12m"] for r in self.results]
            all_rs_sorted = sorted(all_rs)
            n = len(all_rs_sorted)
            for r in self.results:
                pct = all_rs_sorted.index(r["rs_12m"]) / max(n - 1, 1)
                r["rs_rating"] = max(1, min(99, round(pct * 98 + 1)))
            # Apply min-rs filter now that ratings are assigned
            min_rs = getattr(args, "min_rs", 0)
            if min_rs > 0:
                self.results = [r for r in self.results if r["rs_rating"] >= min_rs]

        # ── Apply TradeFilter (R/R ≥ 2.0 AND reward ≥ 20%, fee-adjusted) ────────
        for r in self.results:
            tf = _TRADE_FILTER.check(r)
            r["trade_filter_pass"]    = tf["pass"]
            r["rejection_reason"]     = tf["reason"]
            r["fee_adj_rr"]           = tf["fee_adj_rr"]
            r["fee_adj_reward_pct"]   = tf["fee_adj_reward_pct"]
            r["raw_rr"]               = tf["raw_rr"]
            r["raw_reward_pct"]       = tf["raw_reward_pct"]
            r["rr_pass"]              = tf["rr_pass"]
            r["reward_pass"]          = tf["reward_pass"]
            r["break_even_pct"]       = _FEE_ENGINE.break_even_move(
                                            r.get("price") or 1)["break_even_pct"]

        self.rejected_results = [r for r in self.results if not r["trade_filter_pass"]]
        self.results          = [r for r in self.results if  r["trade_filter_pass"]]

        # Sort by explosive score first, then breakout prob
        expl_results = sorted(self.results,
                              key=lambda x: (x["explosive_score"], x["probability"]),
                              reverse=True)
        self.results.sort(key=lambda x: (x["probability"], x["score"]), reverse=True)

        print(f"\nScan complete: {elapsed:.0f}s  |  {len(self.results)} qualified"
              f"  |  {len(self.rejected_results)} rejected by R/R filter\n")

        n_pats = sum(1 for r in self.results if r["pattern"] != "No Pattern")
        self._print_market_box(len(self.results), n_pats)

        # TABLE 1 — Explosive movers
        top_n = getattr(args, "top", 15)
        print("=" * 62)
        print("  ⚡ EXPLOSIVE MOVE CANDIDATES  (sorted by Explosive Score)")
        print("=" * 62)
        self.results_by_explosive = expl_results
        self._print_explosive_table(top_n)

        # Top 3 explosive breakdowns
        for i, r in enumerate(expl_results[:3], 1):
            if r["explosive_score"] >= 40:
                self._print_explosive_breakdown(r, i)

        # TABLE 2 — Breakout probability (original)
        print("\n" + "=" * 62)
        print("  📊 BREAKOUT PROBABILITY RANKINGS  (sorted by Prob %)")
        print("=" * 62)
        self._print_table(top_n)

        print("=" * 58)
        print("  PROBABILITY BREAKDOWNS — TOP 5")
        print("=" * 58)
        for i, r in enumerate(self.results[:5], 1):
            self._print_breakdown(r, i)

        if getattr(args, "export", True):
            print("\nExporting results...")
            self._export(top_n)

        # REJECTION LOG
        if self.rejected_results:
            print("\n" + "═" * 70)
            print("  ❌ REJECTED TRADES  (did not meet R/R ≥ 2.0 AND reward ≥ 20%)")
            print("═" * 70)
            rej_rows = []
            for r in sorted(self.rejected_results,
                            key=lambda x: x.get("probability", 0), reverse=True):
                rej_rows.append([
                    r["ticker"],
                    f"{r.get('probability', 0)}%",
                    f"{r.get('fee_adj_rr', 0):.1f}",
                    f"+{r.get('fee_adj_reward_pct', 0):.1f}%",
                    r.get("rejection_reason", "—"),
                ])
            print(tabulate(rej_rows,
                           headers=["Ticker", "Score", "Fee-Adj R/R",
                                    "Fee-Adj Reward", "Rejection Reason"],
                           tablefmt="simple"))
            print("═" * 70)

        print(f"\nTotal scan time: {elapsed:.0f}s")


# ══════════════════════════════════════════════════════════════════════════════
# EXPLOSIVE MOVE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
class ExplosiveMoveDetector:
    """
    Scores a stock 0-100 for large-fast-move potential.
    Three pillars:
      Compression  (0-35 pts) — how coiled the spring is
      Squeeze      (0-30 pts) — what forces the move
      Catalyst     (0-25 pts) — technical trigger proximity
    News catalyst score (0-100) from NewsCatalystEngine can replace
    the Catalyst pillar with 50% weight in the final formula.
    """

    def __init__(self, df: pd.DataFrame, info: dict = None,
                 spy_df: pd.DataFrame = None, pattern_results: dict = None,
                 prob_result: dict = None):
        self.df   = df
        self.info = info or {}
        self.spy  = spy_df
        self.pr   = pattern_results or {}
        self.prob = prob_result or {}
        self.C = df["Close"].values
        self.H = df["High"].values
        self.L = df["Low"].values
        self.V = df["Volume"].values.astype(float)
        self.n = len(df)

    # ── Compression ───────────────────────────────────────────────────────────
    def compression_score(self):
        C, H, L, n = self.C, self.H, self.L, self.n
        pts, det = 0, []

        # ATR compression ratio
        atr = _fcol(self.df, "ATRr")
        if len(atr) >= 63:
            atr_now = float(atr.iloc[-1])
            atr_63  = float(atr.iloc[-63])
            if atr_63 > 0:
                ratio = atr_now / atr_63
                p = (10 if ratio < 0.40 else 8 if ratio < 0.50 else
                     6  if ratio < 0.60 else 4 if ratio < 0.70 else
                     2  if ratio < 0.80 else 0)
                if p:
                    pts += p
                    det.append((f"ATR compressed to {ratio:.0%} of 63-day level", p))

        # BB width percentile
        upper = _fcol(self.df, "BBU_")
        lower = _fcol(self.df, "BBL_")
        mid   = _fcol(self.df, "BBM_")
        if not upper.empty and not lower.empty and not mid.empty:
            idx = upper.index.intersection(lower.index).intersection(mid.index)
            if len(idx) >= 60:
                w = ((upper.loc[idx] - lower.loc[idx]) / mid.loc[idx]).dropna()
                if len(w) >= 60:
                    cur_w  = float(w.iloc[-1])
                    hist   = w.iloc[-min(252, len(w)):]
                    pctile = float((hist < cur_w).mean() * 100)
                    p = (10 if pctile <= 5  else 8 if pctile <= 10 else
                         6  if pctile <= 15 else 4 if pctile <= 20 else
                         2  if pctile <= 30 else 0)
                    if p:
                        pts += p
                        det.append((f"BB squeeze — bottom {pctile:.0f}th percentile", p))

        # Historical volatility collapse
        if n >= 90:
            lr = np.log(C[1:] / np.where(C[:-1] > 0, C[:-1], 1))
            hv10 = float(np.std(lr[-10:]) * np.sqrt(252)) if len(lr) >= 10 else 1
            hv90 = float(np.std(lr[-90:]) * np.sqrt(252)) if len(lr) >= 90 else 1
            if hv90 > 0:
                rv = hv10 / hv90
                p = (8 if rv < 0.30 else 6 if rv < 0.40 else
                     4 if rv < 0.50 else 2 if rv < 0.60 else 0)
                if p:
                    pts += p
                    det.append((f"10-day HV at {rv:.0%} of 90-day norm", p))

        # Inside bar accumulation
        if n >= 3:
            count = 0
            for i in range(n - 1, max(n - 16, 0), -1):
                if H[i] <= H[i-1] and L[i] >= L[i-1]:
                    count += 1
                else:
                    break
            p = (7 if count >= 5 else 5 if count >= 4 else
                 3 if count >= 3 else 1 if count >= 2 else 0)
            if p:
                h52 = float(H[-252:].max()) if n >= 252 else float(H.max())
                near = (h52 - float(C[-1])) / h52 <= 0.03 if h52 > 0 else False
                bonus = 2 if near else 0
                pts += p + bonus
                det.append((f"{count} consecutive inside bars — max compression"
                             + (" at resistance" if near else ""), p + bonus))

        return min(35, pts), det

    # ── Squeeze ───────────────────────────────────────────────────────────────
    def squeeze_score(self):
        V, n = self.V, self.n
        pts, det = 0, []
        info = self.info

        # Short interest
        sp  = float(info.get("shortPercentOfFloat", 0) or 0)
        sr  = float(info.get("shortRatio", 0) or 0)
        p = (15 if sp > 0.20 and sr > 5 else 12 if sp > 0.15 and sr > 4 else
             8  if sp > 0.10 and sr > 3 else  4 if sp > 0.07 and sr > 2 else 0)
        if p:
            flag = " — SQUEEZE CANDIDATE" if sp > 0.15 else ""
            pts += p
            det.append((f"Short float {sp:.0%}, {sr:.1f}d to cover{flag}", p))

        # Float size
        fs = int(info.get("floatShares", 0) or 0)
        if fs > 0:
            fm = fs / 1e6
            if   fs < 10e6:  p, lbl = 10, "micro"
            elif fs < 25e6:  p, lbl =  8, "small"
            elif fs < 50e6:  p, lbl =  6, "small"
            elif fs < 100e6: p, lbl =  4, "medium"
            elif fs < 200e6: p, lbl =  2, "medium"
            else:            p, lbl =  0, "large"
            if p:
                pts += p
                det.append((f"Float {fm:.1f}M shares ({lbl})", p))

        # Volume dry-up
        if n >= 20:
            avg20 = float(V[-21:-1].mean())
            avg5  = float(V[-5:].mean())
            ratio = avg5 / avg20 if avg20 > 0 else 1.0
            p = (5 if ratio < 0.40 else 4 if ratio < 0.50 else
                 3 if ratio < 0.60 else 2 if ratio < 0.70 else
                 1 if ratio < 0.80 else 0)
            if p:
                pts += p
                det.append((f"Volume dry-up: {ratio:.0%} of 20-day avg — calm before storm", p))

        return min(30, pts), det

    # ── Technical catalyst ────────────────────────────────────────────────────
    def catalyst_score(self):
        C, H, V, n = self.C, self.H, self.V, self.n
        pts, det = 0, []
        cur = float(C[-1])

        # 52W high proximity
        h52 = float(H[-252:].max()) if n >= 252 else float(H.max())
        if h52 > 0:
            pct = (cur - h52) / h52
            if   cur >= h52:    p, d = 10, "Fresh 52-week high breakout"
            elif pct >= -0.005: p, d =  9, f"Within 0.5% of 52-week high"
            elif pct >= -0.01:  p, d =  7, f"Within 1% of 52-week high"
            elif pct >= -0.02:  p, d =  5, f"Within 2% of 52-week high"
            elif pct >= -0.03:  p, d =  3, f"Within 3% of 52-week high"
            elif pct >= -0.05:  p, d =  1, f"Within 5% of 52-week high"
            else:               p, d =  0, ""
            if p:
                pts += p
                det.append((d, p))

        # Relative strength vs SPY
        if self.spy is not None:
            sp = self.spy["Close"].values
            rs_pts, rs_parts = 0, []
            for days, thresh, award in [(5, 0.03, 3), (10, 0.05, 3), (20, 0.10, 2)]:
                if len(sp) > days and len(C) > days and C[-days] > 0 and sp[-days] > 0:
                    rs = (C[-1]/C[-days] - 1) - (sp[-1]/sp[-days] - 1)
                    if rs > thresh:
                        rs_pts += award
                        rs_parts.append(f"+{rs:.1%} {days}d")
            if rs_pts:
                pts += rs_pts
                det.append((f"Outperforming SPY: {', '.join(rs_parts)}", rs_pts))

        # Momentum ignition candle (last 3 days)
        O = self.df["Open"].values if "Open" in self.df.columns else C
        if n >= 22:
            avg_body = float(np.mean(np.abs(C[-21:-1] - O[-21:-1])))
            avg_vol  = float(V[-21:-1].mean())
            for days_ago in range(1, 4):
                i = -days_ago
                body  = abs(float(C[i]) - float(O[i]))
                vol   = float(V[i])
                rng   = float(H[i]) - float(self.L[i])
                top25 = rng > 0 and (float(C[i]) - float(self.L[i])) / rng >= 0.75
                big_b = avg_body > 0 and body >= avg_body * 2
                big_v = avg_vol  > 0 and vol  >= avg_vol  * 2
                if big_b and top25 and big_v:
                    pts += 7; det.append((f"Momentum ignition candle {days_ago}d ago", 7)); break
                elif big_b and big_v:
                    pts += 4; det.append((f"High-conviction candle {days_ago}d ago", 4)); break
                elif avg_vol > 0 and vol >= avg_vol * 3:
                    pts += 2; det.append((f"3x+ volume spike {days_ago}d ago", 2)); break

        return min(25, pts), det

    # ── Move estimate ─────────────────────────────────────────────────────────
    def estimate_move(self):
        C, H, n = self.C, self.H, self.n
        cur = float(C[-1])
        atr_s   = _fcol(self.df, "ATRr")
        atr_val = float(atr_s.iloc[-1]) if not atr_s.empty else cur * 0.02

        pat = self.prob.get("primary_pattern", "No Pattern")
        vcp = self.pr.get("vcp", {}); cup = self.pr.get("cup_and_handle", {})
        htf = self.pr.get("htf", {}); bf  = self.pr.get("bull_flag", {})
        at  = self.pr.get("ascending_triangle", {}); db = self.pr.get("double_bottom", {})
        wy  = self.pr.get("wyckoff", {})
        h52 = float(H[-252:].max()) if n >= 252 else float(H.max())

        if   "VCP"        in pat: base = vcp.get("final_depth", 0.10) * 1.5
        elif "Cup"        in pat: base = cup.get("cup_depth",   0.30)
        elif "High"       in pat: base = htf.get("pole_gain",   0.50)
        elif "Wyckoff"    in pat:
            lo = float(self.L[-252:].min()) if n >= 252 else float(self.L.min())
            base = ((h52 - lo) / cur) * 1.5 if cur > 0 else 0.30
        elif "Bull Flag"  in pat: base = bf.get("pole_gain",    0.20)
        elif "Ascending"  in pat:
            resist = at.get("resistance", cur * 1.02)
            lo     = float(self.L[-60:].min()) if n >= 60 else float(self.L.min())
            base   = (resist - lo) / cur if cur > 0 else 0.15
        elif "Double"     in pat:
            neck = db.get("neckline", cur * 1.02); lo = db.get("low1", cur * 0.90)
            base = (neck - lo) / cur * 2 if cur > 0 else 0.20
        elif "Flat"       in pat: base = 0.20
        else:                     base = atr_val / cur * 5 if cur > 0 else 0.10

        base = max(0.05, min(1.50, base))

        sp  = float(self.info.get("shortPercentOfFloat", 0) or 0)
        fs  = int(self.info.get("floatShares", 0) or 0)
        sqm = 2.0 if sp > 0.20 else 1.5 if sp > 0.10 else 1.2 if sp > 0.05 else 1.0
        fm  = 1.5 if fs < 10e6 else 1.3 if fs < 25e6 else 1.1 if fs < 50e6 else 1.0

        lo_est  = base * sqm * fm
        hi_est  = min(3.0, lo_est + (atr_val / cur * 3 if cur > 0 else 0.10) + (sqm - 1) * 0.30)
        return round(lo_est * 100, 1), round(hi_est * 100, 1)

    # ── Main calculate ────────────────────────────────────────────────────────
    def calculate(self, news_score: float = None) -> dict:
        c_pts, c_det = self.compression_score()
        s_pts, s_det = self.squeeze_score()
        t_pts, t_det = self.catalyst_score()

        if news_score is not None:
            raw = (c_pts / 35) * 25 + (s_pts / 30) * 25 + (news_score / 100) * 50
        else:
            raw = (c_pts + s_pts + t_pts) / 90 * 100

        score = min(100, max(0, round(raw)))
        grade = ("⚡ EXPLOSIVE" if score >= 85 else "🔥 HIGH"     if score >= 70 else
                 "📈 MODERATE"  if score >= 55 else "👀 WATCH"    if score >= 40 else "❌ PASS")

        move_lo, move_hi = self.estimate_move()
        return {
            "score":           score,  "grade":         grade,
            "compression_pts": c_pts,  "squeeze_pts":   s_pts,
            "catalyst_pts":    t_pts,  "compression_det": c_det,
            "squeeze_det":     s_det,  "catalyst_det":  t_det,
            "move_low":        move_lo, "move_high":    move_hi,
            "short_pct":  float(self.info.get("shortPercentOfFloat", 0) or 0),
            "float_sh":   int(self.info.get("floatShares",          0) or 0),
            "short_ratio":float(self.info.get("shortRatio",         0) or 0),
        }


# ══════════════════════════════════════════════════════════════════════════════
# NEWS CATALYST ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# EARNINGS QUALITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class EarningsQualityEngine:
    """
    Scores EPS quality 0-100 based on:
      - Profitability (positive EPS)
      - Growth rate (YoY or QoQ)
      - Acceleration (each quarter better than last)
      - Revenue growth alignment
    """
    def score(self, ticker: str) -> dict:
        empty = {"eq_score": 50, "eq_grade": "N/A", "eq_acceleration": False,
                 "eq_eps_vals": [], "eq_detail": "No data"}
        try:
            t  = yf.Ticker(ticker)
            qe = None
            for attr in ("quarterly_earnings", "quarterly_income_stmt"):
                try:
                    qe = getattr(t, attr)
                    if qe is not None and not qe.empty:
                        break
                except Exception:
                    pass
            if qe is None or qe.empty:
                return empty

            # quarterly_earnings → index=date, cols=[Earnings, Revenue]
            # quarterly_income_stmt → cols=dates, rows=line items
            if hasattr(qe, "columns") and "Earnings" in qe.columns:
                eps = qe["Earnings"].dropna().values[:4]
            else:
                # income_stmt format: rows are items, cols are dates
                for row_key in ("Net Income", "Net Income Common Stockholders",
                                "Diluted EPS", "Basic EPS"):
                    if row_key in qe.index:
                        eps = qe.loc[row_key].dropna().values[:4]
                        break
                else:
                    return empty

            if len(eps) < 2:
                return empty

            score = 50
            accel = False
            detail_parts = []

            # Profitability
            if eps[0] > 0:
                score += 15
            elif eps[0] > eps[1]:
                score += 5   # improving loss

            # Growth rate (most recent vs same quarter LY if 4 quarters available)
            if len(eps) >= 4 and eps[3] != 0:
                yoy = (eps[0] - eps[3]) / abs(eps[3])
                if yoy > 0.50: score += 20
                elif yoy > 0.25: score += 15
                elif yoy > 0.10: score += 10
                elif yoy > 0:    score += 5
                elif yoy < -0.20: score -= 10
                detail_parts.append(f"YoY {yoy:+.0%}")

            # Acceleration: recent quarter better than previous 2
            if len(eps) >= 3 and eps[1] != 0 and eps[2] != 0:
                g1 = (eps[0] - eps[1]) / abs(eps[1])
                g2 = (eps[1] - eps[2]) / abs(eps[2])
                if g1 > g2 > 0 or (eps[0] > eps[1] > eps[2] and all(e > 0 for e in eps[:3])):
                    accel = True
                    score += 15

            score = max(0, min(100, score))
            if score >= 85:   grade = "A+"
            elif score >= 75: grade = "A"
            elif score >= 65: grade = "B+"
            elif score >= 55: grade = "B"
            elif score >= 45: grade = "C"
            else:             grade = "D"

            detail_parts.append(f"EPS:{'/'.join(f'{e:.2f}' for e in eps[:3])}")
            return {"eq_score": score, "eq_grade": grade, "eq_acceleration": accel,
                    "eq_eps_vals": list(eps[:4]), "eq_detail": " | ".join(detail_parts)}
        except Exception:
            return empty


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS FLOW SCORER
# ══════════════════════════════════════════════════════════════════════════════
class OptionsFlowScorer:
    """
    Scores unusual options activity 0-100 using yfinance options chain.
    Signals: call/put volume ratio, vol > OI (new positioning), large OI buildup.
    """
    def score(self, ticker: str) -> dict:
        empty = {"flow_score": 0, "flow_signal": "", "cp_ratio": 0.0,
                 "unusual_calls": False, "flow_detail": ""}
        try:
            t = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                return empty

            total_call_vol = 0; total_put_vol = 0; total_call_oi = 0
            unusual = False

            for d in dates[:3]:   # sample nearest 3 expiries
                try:
                    chain = t.option_chain(d)
                    cv = float(chain.calls["volume"].fillna(0).sum())
                    pv = float(chain.puts ["volume"].fillna(0).sum())
                    co = float(chain.calls["openInterest"].fillna(0).sum())
                    total_call_vol += cv; total_put_vol += pv; total_call_oi += co
                    # Vol > OI means fresh positioning (not just delta hedge)
                    mask = (chain.calls["volume"].fillna(0) >
                            chain.calls["openInterest"].fillna(1) * 0.8)
                    if mask.any():
                        unusual = True
                except Exception:
                    pass

            if total_put_vol == 0 and total_call_vol == 0:
                return empty

            cp = total_call_vol / max(total_put_vol, 1)
            sc = 0
            if   cp > 4.0: sc += 40
            elif cp > 3.0: sc += 30
            elif cp > 2.0: sc += 20
            elif cp > 1.5: sc += 12
            elif cp > 1.2: sc +=  6
            if unusual: sc += 30
            if total_call_oi > 0 and total_call_vol / total_call_oi > 0.40:
                sc += 15   # high vol/OI turnover

            sc = min(100, sc)
            if sc >= 60:   sig = f"UNUSUAL CALLS  C/P {cp:.1f}x"
            elif sc >= 35: sig = f"Elevated calls C/P {cp:.1f}x"
            elif sc >= 15: sig = f"Call bias C/P {cp:.1f}x"
            else:          sig = ""

            return {"flow_score": sc, "flow_signal": sig, "cp_ratio": round(cp, 2),
                    "unusual_calls": unusual,
                    "flow_detail": f"Call vol {total_call_vol:,.0f} | Put vol {total_put_vol:,.0f}"}
        except Exception:
            return empty


class NewsCatalystEngine:
    _HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    TIER1 = [
        ("fda",      90, "🧬 FDA CATALYST",    ["fda approval","fda approved","phase 3","phase 2 results","pdufa","nda ","bla ","breakthrough therapy","primary endpoint met"]),
        ("ma",       95, "🤝 M&A CATALYST",    ["acquisition","merger","buyout","takeover","acquired by","definitive agreement","all-cash offer"]),
        ("squeeze",  75, "🔥 SQUEEZE CATALYST",["short squeeze","gamma squeeze","heavily shorted","most shorted"]),
        ("contract", 70, "📋 CONTRACT",        ["contract awarded","government contract","dod contract","billion dollar","multi-year agreement","exclusive agreement"]),
        ("activist", 65, "🏦 ACTIVIST",        ["13d filing","activist investor","strategic review","exploring strategic","board seat"]),
    ]
    TIER2 = [
        ("earnings", 45, "📊 EARNINGS BEAT",   ["beats estimates","earnings beat","raises guidance","record revenue","exceeded expectations"]),
        ("product",  40, "🚀 PRODUCT LAUNCH",  ["launches","unveiled","breakthrough","patent granted","world's first","first-ever"]),
        ("uplist",   50, "📈 UPLISTING",        ["nyse uplisting","nasdaq uplisting","uplisting","approved for listing"]),
        ("insider",  35, "👤 INSIDER BUYING",  ["insider purchase","ceo buys","director purchases","open market purchase"]),
        ("upgrade",  30, "⭐ UPGRADE",         ["upgraded to buy","price target raised","outperform","initiation of coverage"]),
    ]
    NEGATIVE = [
        ("dilution", -30, "⚠️ DILUTION RISK",  ["public offering","share offering","atm offering","registered direct","private placement","dilutive"]),
        ("legal",    -50, "🚨 LEGAL RISK",      ["sec investigation","securities fraud","class action","delisting notice"]),
        ("miss",     -20, "📉 MISS",            ["misses estimates","earnings miss","cuts guidance","disappointing results"]),
    ]

    def fetch_news(self, ticker: str) -> list:
        news = []
        cutoff = datetime.now() - timedelta(days=7)

        # yfinance news
        try:
            for item in (yf.Ticker(ticker).news or [])[:15]:
                ts = item.get("providerPublishTime", 0)
                pub = datetime.fromtimestamp(ts) if ts else datetime.now()
                if pub >= cutoff:
                    news.append({"title": item.get("title",""), "date": pub,
                                 "source": item.get("publisher","yf")})
        except Exception:
            pass

        # Yahoo RSS
        try:
            if _HAS_FEEDPARSER:
                feed = feedparser.parse(
                    f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                    f"?s={ticker}&region=US&lang=en-US")
                import email.utils
                for e in feed.entries[:10]:
                    try:
                        pub = datetime(*email.utils.parsedate(e.get("published",""))[:6])
                    except Exception:
                        pub = datetime.now()
                    if pub >= cutoff:
                        news.append({"title": e.get("title",""), "date": pub, "source": "RSS"})
        except Exception:
            pass

        # Reddit WSB
        try:
            url  = (f"https://www.reddit.com/r/wallstreetbets/search.json"
                    f"?q={ticker}&sort=new&limit=25&restrict_sr=on")
            resp = requests.get(url, headers=self._HDRS, timeout=5)
            if resp.status_code == 200:
                posts   = resp.json().get("data",{}).get("children",[])
                recent  = sum(1 for p in posts
                              if datetime.fromtimestamp(p["data"].get("created_utc",0)) >= cutoff)
                if recent >= 5:
                    news.append({"title": f"WSB: {recent} mentions in last 7 days",
                                 "date": datetime.now(), "source": "Reddit",
                                 "_wsb": recent})
        except Exception:
            pass

        return news

    def classify(self, news_list: list):
        score, flags = 0, []
        text = " ".join(n["title"].lower() for n in news_list)

        for groups in [self.TIER1, self.TIER2]:
            for _, base, flag, kws in groups:
                if not any(k in text for k in kws):
                    continue
                score += base
                hl, ds = "", ""
                for n in news_list:
                    if any(k in n["title"].lower() for k in kws):
                        hl = n["title"][:80]
                        ds = n["date"].strftime("%m-%d %H:%M") if hasattr(n["date"],"strftime") else ""
                        break
                flags.append({"flag": flag, "score": base, "headline": hl, "date": ds})

        for _, pen, flag, kws in self.NEGATIVE:
            if any(k in text for k in kws):
                score += pen
                flags.append({"flag": flag, "score": pen, "headline": "", "date": ""})

        wsb = [n for n in news_list if "_wsb" in n]
        if wsb:
            cnt = wsb[0]["_wsb"]
            p   = 20 if cnt >= 10 else 10
            score += p
            flags.append({"flag": "📱 SOCIAL", "score": p,
                          "headline": f"{cnt} WSB mentions", "date": ""})

        return max(0, min(100, score)), flags

    def scan(self, ticker: str) -> dict:
        try:
            news  = self.fetch_news(ticker)
            sc, fl = self.classify(news)
            top_flag = fl[0]["flag"] if fl else ""
            top_hl   = fl[0]["headline"] if fl else ""
        except Exception:
            sc, fl, top_flag, top_hl = 0, [], "", ""
        return {"ticker": ticker, "catalyst_score": sc, "catalyst_flags": fl,
                "top_flag": top_flag, "top_headline": top_hl[:60]}


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
class UniverseBuilder:
    _HDRS    = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    _SCREENS = [
        "https://finviz.com/screener.ashx?v=111&f=cap_small,cap_micro,ta_highlow52w_b0to10h,sh_float_u50,sh_avgvol_o300&ft=4",
        "https://finviz.com/screener.ashx?v=111&f=cap_small,cap_micro,sh_short_o15&ft=4",
        "https://finviz.com/screener.ashx?v=111&f=cap_micro,cap_nano,ta_highlow52w_b0to5h&ft=4",
        "https://finviz.com/screener.ashx?v=111&f=cap_small,cap_micro,sh_relvol_o2&ft=4",
    ]
    _FALLBACK = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","UNH",
                 "XOM","PG","MA","HD","CVX","MRK","LLY","ABBV","KO","PEP",
                 "COST","AVGO","CSCO","TMO","ACN","MCD","ABT","CRM","NKE","ADBE"]

    # ── 1000+ curated active small-cap tickers ────────────────────────────────
    _SMALLCAP = [
        # Biotech / Pharma
        "ACAD","ACRS","ADAP","ADAG","ADMA","ADUS","AGEN","AGRX","AGIO","AKRO",
        "ALBO","ALDX","ALEC","ALKS","ALPN","ALRN","ALXO","AMPH","AMRX","AMTI",
        "ANAB","ANIP","APLS","APLT","APVO","ARDX","ARIA","ARCT","AROA","ARQT",
        "ARVN","ARWR","ASND","ASRT","ATAI","ATNX","ATRC","ATXI","AUTL","AUPH",
        "AVDL","AVEO","AVXL","AXGN","AXSM","AZRX","BCRX","BDTX","BHVN","BIOL",
        "BLCM","BLFS","BLUE","BNGO","BPMC","BTAI","BYSI","CAPR","CARA","CASI",
        "CCRN","CDTX","CELC","CEMI","CERS","CERT","CGEM","CLDX","CLLS","CLPT",
        "CLRB","CLSD","CLVS","CMRX","CNCE","CODX","CORT","CPRX","CRBU","CRDF",
        "CRIS","CRVS","CTMX","CYTK","DCPH","DERM","DNLI","DVAX","DYAI","EDIT",
        "EIDX","EPZM","ERAS","ETNB","EVFM","EVLO","EXEL","FATE","FDMT","FGEN",
        "FIXX","FOLD","FMTX","FSTX","FULC","GALT","GBIO","GLSI","GNPX","GPCR",
        "GRTS","GTHX","HALO","HOOK","HRMY","HUMA","ICVX","IDYA","IGMS","IMAB",
        "IMMP","IMTX","IMUX","IMVT","INSM","IONS","IRTC","ITOS","JNCE","KARX",
        "KDNY","KPTI","KRTX","KYMR","KZIA","LBPH","LENZ","LGND","LNTH","LQDA",
        "MACK","MCRB","MDGL","MGTA","MGTX","MIRM","MNKD","MNMD","MRUS","MYOV",
        "NKTR","NNOX","NRIX","NTLA","NVAX","NVCR","OCGN","OCSL","OMER","ORGO",
        "PAHC","PCVX","PCYG","PDCO","PLRX","PLSE","PMVP","PRAX","PRLD","PRME",
        "PTCT","PTGX","RAPT","RCKT","RETA","RGEN","RIGL","RLAY","RNAC","RPRX",
        "SAGE","SAVA","SEER","SGMO","SGTX","SNDX","SRNE","STRO","SURF","SVRA",
        "SYRS","TBPH","TGTX","TMDX","TNXP","TPTX","TRVN","TTOO","TVTX","TWST",
        "TXMD","URGN","VBIV","VERA","VERV","VKTX","VNDA","VRNA","VSTM","VYGR",
        "XENE","XNCR","ZYME",
        # Technology / Software
        "ACLS","ACMR","AEHR","AEIS","AIXI","ALKT","ALLT","AMBA","AMPL","ANGI",
        "AOSL","APPS","ARQQ","ASLE","ASPU","ATEN","ATNI","AUDC","AVNW","BAND",
        "BBAI","BIGC","BLKB","BLNK","BLZE","BMBL","CALX","CASA","CDLX","CHPT",
        "CLBT","COHU","CRAI","CRSR","CSGS","DCBO","DFIN","DGII","DMRC","DOMO",
        "DSGX","DTIL","DUOS","EGHT","EGAN","EGIO","EMKR","ENFN","EVBG","EVLV",
        "EXLS","EXPI","EXTR","FARO","FIVN","FORM","FOUR","FRGE","GCTS","GDYN",
        "GENI","GLBE","GRPN","GTLB","HOLI","IBEX","ICHR","IDCC","IESC","IIIV",
        "IMXI","INFU","INPX","INSG","IONQ","IRIX","IRMD","JAMF","KLIC","LASR",
        "LPSN","MAPS","MARK","MCRI","MGNI","MNTV","MRAM","MRCY","MRVI","MYPS",
        "NCNO","NEOG","NEON","NOVA","NOVT","NRDS","NTNX","NVTS","ONDS","OUST",
        "PAYA","PLAB","PLUG","POWI","PRFT","PRKR","PRTS","PSIX","PXLW","PYCR",
        "QNST","QUBT","QUIK","RBBN","RCII","RELY","RMNI","RNET","ROLL","RPAY",
        "RSKD","RXRX","SMSI","SOFI","SOUN","SQSP","SSYS","STEM","STNE","STRM",
        "SWIR","TASK","TDUP","TTEC","TTGT","TWKS","VLTA","WKHS",
        # Energy / Oil & Gas / Shipping
        "BATL","BCEI","BSM","BRY","CDEV","CHRD","CIVI","CRK","CRGY","DHT",
        "DKL","DNN","DSX","DUNE","EGLE","EGY","EPM","ESTE","FTCI","GFI",
        "GHM","GLNG","GNLN","GNE","GORO","GPOR","GPRE","HEP","HESM","HY",
        "IMPP","INDO","INSW","KILM","KOS","LBRT","LTHM","MGY","MNRL","MOD",
        "MRC","MVO","NGD","NTIC","NOG","OII","PARR","PTEN","REI","ROAN",
        "SBR","SBOW","SD","SFL","SMR","SOI","SPND","STNG","TALO","TGA",
        "TRMD","TUSK","UAN","VGR","VOC","WTI","MXC","MARPS",
        "ASC","CMRE","DAC","DLNG","EGLE","ESEA","GNK","GOGL","ICON","NMM",
        "PANL","SALT","SBLK","TK","FREE","SB","TOPS","PSHG","SHIP",
        # Financials / Community Banks / Insurance
        "ACNB","ACEL","ATLC","BANF","BANR","BHLB","BKSC","BMTC","BPFH",
        "BRKL","BSRR","BYFC","CATY","CBMB","CCBG","CFFI","CFFN","CHCO",
        "CIVB","CLBK","CNOB","COFS","CRWS","CSFL","CTBI","CVBF","CWBC",
        "DCOM","ESSA","EVBN","FBMS","FBNC","FBP","FCBP","FCCO","FCF",
        "FFIN","FFNW","FGBI","FISI","FLIC","FLMN","FMAO","FNLC","FRAF",
        "FRBA","FRST","FSBC","FSBW","FSFG","FSTR","GFED","GNTY","HAFC",
        "HBCP","HBIA","HBMD","HFBL","HFWA","HMST","HTBI","HTBK","HTLF",
        "HWBK","IBCP","IBOC","IROQ","ISBA","ISTR","JMSB","LBAI","LCNB",
        "LKFN","MBWM","MCBC","MCBS","METC","MFNC","MNSB","MPB","NBTB",
        "NKSH","NRIM","NWIN","OBNK","OFG","OLBK","OPBK","ORRF","OSBC",
        "OVBC","PBHC","PBIP","PFIS","PLBC","PLMR","PNFP","PPBI",
        # Consumer / Retail / Restaurants
        "ARKO","ARLO","BLMN","BOOT","BRC","CAKE","CHUY","CNXN","COKE",
        "CONN","COUR","CTRN","CULP","DINE","DNUT","DOOR","DRVN","EFC",
        "ENVA","EPAC","ERII","ESNT","EXPR","EVA","EVRI","FCFS","FIZZ",
        "FWONK","GES","GIL","GMS","GOED","GPC","GRWG","HAYN","HCI","HIMS",
        "HLLY","HTHT","HURC","HVT","HWKN","IPAR","JACK","JBSS","KFRC",
        "KRUS","KW","LCUT","LILA","LINC","LOVE","MATW","MBUU","MCFT",
        "MFAC","MGPI","MMSI","MODV","MRTN","NATH","NDLS","NTGR","OLPX",
        # Industrials / Defense / Manufacturing
        "AVAV","AXON","BKTI","BOOM","BORR","CACI","CNXN","EPAC","FLIR",
        "GHM","HII","HURC","HWKN","KTOS","LDOS","LMT","MANT","MRCY",
        "MTRX","OSIS","PAE","PLTR","POWL","PRFT","RGR","ROLL","RRBI",
        "SASR","SCSC","SPAR","SPSC","STRL","SWBI","TAIT","TECL","TITN",
        "TRMK","TRMR","TRNO","TRUP","TTEC","TTGT","TURA","TWKS","TXRH",
        "TYRA","UAVS","UFPI","UHAL","ULTA","UMBF","UMPQ","UNFI","UNIT",
        "UPLD","UPWK","URBN","USCR","USPH","UTMD","UUUU","VCEL","VCNX",
        # Materials / Mining / Metals
        "AUMN","AXU","BTG","DNN","EGO","ERO","FSM","FURY","GFI","GORO",
        "GPL","GSS","HMY","HL","IAG","IAUX","KORE","LODE","MAG","MTA",
        "MTAL","MUX","NGD","NMG","OR","ORLA","PAAS","RBY","SBSW","SILV",
        "SSRM","TMQ","USAS","WDO","WRN","GATO","ZNOG",
        # Healthcare Services / Devices
        "ACCD","ACHC","AORT","APAM","ARAY","ATRC","ATRI","AXNX","BCOR",
        "BFAM","CANO","CCXI","CHNG","CRVL","CSII","DXCM","EHTH","ELMD",
        "ENSG","EVFM","FLGT","FWRD","GDRX","GKOS","GPRO","HAAC","HCAT",
        "HCSG","HIMS","HMHC","HOLX","HUMA","HWBK","IART","ICAD","ICUI",
        "IDXX","INVA","IRTC","ITGR","JNCE","KDNY","LHCG","LIVN","MDRX",
        "MDSO","MEDNAX","MMSI","MNMD","MODN","NARI","NEOG","NKTR","NVCR",
        "NVST","OBDC","OFIX","OMCL","OPCH","OSIS","PDCO","PNTG","PRSC",
        "PRVA","PSTG","QTWO","RGEN","RMTI","RNST","ROIC","RXRX","SCPH",
        # Real Estate / REITs
        "AIV","APTS","BRT","CLDT","CPLG","ELME","EFC","FCPT","GMRE",
        "HIW","HPP","INN","JBGS","KRG","LTC","MDV","MDRR","NREF","NXRT",
        "OFC","PKST","PLYM","PSTL","RC","RLJ","ROIC","RPT","SAFE",
        "SHO","SILA","SKT","SLG","STAR","STRS","STWD","UHT","VRE","ESRT",
        # Cannabis
        "ACB","APHA","IIPR","KERN","SNDL","TLRY","YOLO","ZYNE",
    ]

    # ── US Exchange fetchers ──────────────────────────────────────────────────

    # Exchange name normalisation for the SEC tickers file
    _SEC_EXCH_MAP = {
        "nasdaq":    "NASDAQ",
        "nyse":      "NYSE",
        "nyse mkt":  "NYSE",   # NYSE American
        "nyse arca": "NYSE",
        "cboe":      "CBOE",
        "otc":       "OTC",
    }

    def _sec_tickers(self) -> list:
        """All US-listed equities from SEC EDGAR company_tickers_exchange.json.
        Returns list of (ticker, exchange_normalised) tuples."""
        try:
            r = requests.get(
                "https://www.sec.gov/files/company_tickers_exchange.json",
                headers={**self._HDRS,
                         "Accept": "application/json",
                         "Host": "www.sec.gov"},
                timeout=30)
            if r.status_code != 200:
                return []
            data  = r.json()
            fields = [f.lower() for f in data.get("fields", [])]
            ti = next((i for i, f in enumerate(fields) if "ticker" in f), None)
            ei = next((i for i, f in enumerate(fields) if "exchange" in f), None)
            if ti is None:
                return []
            out = []
            for row in data.get("data", []):
                sym  = str(row[ti]).strip().upper() if ti < len(row) else ""
                exch = str(row[ei]).strip().lower() if (ei is not None and ei < len(row)) else ""
                if not sym or len(sym) > 5 or any(c in sym for c in ("$","^","."," ")):
                    continue
                exch_norm = self._SEC_EXCH_MAP.get(exch, exch.upper())
                out.append((sym, exch_norm))
            return out
        except Exception as e:
            print(f"  [warn] SEC tickers: {e}")
            return []

    def _us_nasdaq(self) -> list:
        """NASDAQ-listed equities."""
        # Primary: SEC EDGAR
        rows = self._sec_tickers()
        if rows:
            return [sym for sym, ex in rows if ex == "NASDAQ"]
        # Fallback: NASDAQ Trader FTP
        tickers = []
        for proto in ("http", "https"):
            try:
                r = requests.get(
                    f"{proto}://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
                    headers=self._HDRS, timeout=20)
                if r.status_code == 200 and "|" in r.text:
                    for line in r.text.splitlines()[1:]:
                        parts = line.split("|")
                        if len(parts) < 7: continue
                        sym = parts[0].strip()
                        if not sym or sym == "File Creation Time": continue
                        if len(sym) > 5 or any(c in sym for c in ("$","^",".")): continue
                        if parts[6].strip() == "Y": continue
                        tickers.append(sym)
                    break
            except Exception:
                continue
        return tickers

    def _us_other(self, exchanges: set = None) -> list:
        """NYSE / NYSE Arca / CBOE / etc.
        exchanges: set of normalised names like {'NYSE','CBOE'}; None = all non-NASDAQ."""
        rows = self._sec_tickers()
        if rows:
            return [sym for sym, ex in rows
                    if ex != "NASDAQ" and ex != "OTC"
                    and (exchanges is None or ex in exchanges)]
        # Fallback: NASDAQ Trader FTP otherlisted.txt
        # Exchange codes: N=NYSE, A=NYSE American, P=NYSE Arca, Z=CBOE BZX
        ftp_map = {"NYSE": {"N","A","P"}, "CBOE": {"Z"}}
        ftp_codes: set = set()
        if exchanges:
            for ex in exchanges:
                ftp_codes |= ftp_map.get(ex, set())
        tickers = []
        for proto in ("http", "https"):
            try:
                r = requests.get(
                    f"{proto}://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt",
                    headers=self._HDRS, timeout=20)
                if r.status_code == 200 and "|" in r.text:
                    for line in r.text.splitlines()[1:]:
                        parts = line.split("|")
                        if len(parts) < 5: continue
                        sym  = parts[0].strip()
                        exch = parts[2].strip()
                        if not sym or sym == "File Creation Time": continue
                        if len(sym) > 5 or any(c in sym for c in ("$","^",".")): continue
                        if ftp_codes and exch not in ftp_codes: continue
                        if parts[4].strip() == "Y": continue
                        tickers.append(sym)
                    break
            except Exception:
                continue
        return tickers

    def _otc_select(self) -> list:
        """OTCQX (Expert) and OTCQB (Premium) tier stocks — Select OTC."""
        tickers = []
        for market in ("Expert", "Premium"):
            for proto in ("https", "http"):
                try:
                    url = (f"{proto}://backend.otcmarkets.com/otcapi/companies/"
                           f"securities/all/csv?market={market}")
                    r = requests.get(url, headers=self._HDRS, timeout=20)
                    if r.status_code == 200 and len(r.text) > 200:
                        import csv as _csv, io as _io
                        reader = _csv.DictReader(_io.StringIO(r.text))
                        for row in reader:
                            sym = (row.get("Symbol") or row.get("symbol","")).strip().upper()
                            if sym and 1 <= len(sym) <= 5 and sym.isalpha():
                                tickers.append(sym)
                        break
                except Exception:
                    continue
        if not tickers:
            for furl in [
                "https://finviz.com/screener.ashx?v=111&f=exch_otcqx,cap_small,cap_micro&ft=4",
                "https://finviz.com/screener.ashx?v=111&f=exch_otcqb,cap_small,cap_micro&ft=4",
            ]:
                tickers.extend(self._finviz_page(furl))
        return list(set(tickers))

    # ── Canadian Exchange fetchers ────────────────────────────────────────────

    # Curated fallback lists for when live fetches fail
    _TSX_BASE = [
        # Banks & Financials
        "RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","NA.TO","MFC.TO","SLF.TO",
        "POW.TO","FFH.TO","GWO.TO","IAG.TO","EQB.TO","IFC.TO","TRI.TO",
        # Energy
        "CNQ.TO","SU.TO","CVE.TO","IMO.TO","TRP.TO","ENB.TO","PPL.TO",
        "ARC.TO","PEY.TO","TVE.TO","CPG.TO","MEG.TO","VET.TO","ERF.TO",
        "BTE.TO","WCP.TO","BIR.TO","PXT.TO","KEL.TO","GTE.TO","CJ.TO",
        # Mining & Materials
        "ABX.TO","AGI.TO","AEM.TO","FM.TO","K.TO","LUN.TO","WPM.TO",
        "TECK.TO","CS.TO","OR.TO","EDV.TO","SSL.TO","FNV.TO","PAAS.TO",
        "HBM.TO","CG.TO","IMG.TO","AR.TO","NGT.TO","MAG.TO","ELD.TO",
        "PVG.TO","SVM.TO","MUX.TO","KNT.TO","IAU.TO","DPM.TO",
        # Tech & Telecom
        "SHOP.TO","CSU.TO","LSPD.TO","BCE.TO","T.TO","QBR-B.TO","RCI-B.TO",
        "OTEX.TO","DSG.TO","DND.TO","ENGH.TO","TVA-B.TO","KXS.TO",
        "HUT.TO","BITF.TO","MOGO.TO","TOI.TO","CIGI.TO","GIB-A.TO",
        # Utilities & Infrastructure
        "FTS.TO","H.TO","AQN.TO","EMA.TO","CU.TO","NPI.TO","BEP-UN.TO",
        "INE.TO","ACO-X.TO","CPX.TO","TA.TO","RNW.TO","BEPC.TO",
        # Consumer & Retail
        "ATD.TO","L.TO","MRU.TO","EMP-A.TO","DOL.TO","CTC-A.TO",
        "CCL-B.TO","QSP-UN.TO","RECP.TO","MTY.TO","PZA.TO","FOOD.TO",
        # Healthcare
        "SIA.TO","WELL.TO","DRT.TO","HLS.TO","DTOL.TO","PMZ-UN.TO",
        # Real Estate
        "H.TO","CAR-UN.TO","DIR-UN.TO","AP-UN.TO","HR-UN.TO","SRU-UN.TO",
        "CRR-UN.TO","GRT-UN.TO","AAR-UN.TO","IIP-UN.TO","CHP-UN.TO",
        # Other
        "CNR.TO","CP.TO","WN.TO","MX.TO","PAR.TO","PLC.TO","ACB.TO",
        "TLRY.TO","HEXO.TO","CRON.TO","WEED.TO","OGI.TO","NRGI.TO",
    ]

    _TSXV_BASE = [
        # Junior Mining (gold, silver, copper, lithium)
        "AAB.V","ABN.V","ACN.V","AFM.V","AGM.V","AHR.V","AKG.V","ALO.V",
        "AMK.V","ANX.V","AOT.V","APM.V","ARG.V","ARL.V","ARM.V","ASM.V",
        "ATX.V","AUN.V","AZM.V","BAR.V","BBB.V","BCM.V","BHS.V","BRD.V",
        "BSX.V","BTT.V","BYN.V","CAD.V","CBK.V","CCW.V","CDG.V","CGC.V",
        "CLM.V","CLZ.V","CMA.V","CMC.V","CNL.V","COG.V","COR.V","CPP.V",
        "CRE.V","CRJ.V","CRS.V","CSD.V","CSR.V","CTM.V","CUU.V","CWM.V",
        "DAU.V","DCC.V","DEF.V","DGO.V","DIO.V","DLR.V","DMX.V","DNG.V",
        "EAU.V","EFM.V","EGM.V","ELC.V","EMM.V","EPO.V","ERO.V","ESM.V",
        "ETG.V","EXG.V","FEL.V","FGD.V","FIL.V","FMG.V","FRG.V","FRK.V",
        "FSX.V","GBR.V","GCX.V","GGO.V","GLD.V","GLG.V","GLO.V","GMX.V",
        "GNX.V","GOG.V","GPM.V","GSP.V","GTT.V","GUA.V","GUN.V","GWG.V",
        # Cannabis (TSXV)
        "APHA.V","CRON.V","ACB.V","VFF.V","LABS.V","TRST.V","PLTH.V",
        # Tech & Other
        "HOM.V","HWY.V","IAU.V","ICG.V","ILC.V","ILS.V","IMC.V","INX.V",
        "IPA.V","IPT.V","ISC.V","ISV.V","IVN.V","JOY.V","KBG.V","KCC.V",
        "KDX.V","KGC.V","KNT.V","KOR.V","KRR.V","KTN.V","LAB.V","LGO.V",
        "LIO.V","LIS.V","LIT.V","LME.V","LOT.V","LRS.V","LSG.V","LTO.V",
        "MAI.V","MAO.V","MAY.V","MBX.V","MCB.V","MCG.V","MCM.V","MDA.V",
        "MDI.V","MDO.V","MEI.V","MER.V","MES.V","MGI.V","MIL.V","MIS.V",
        "MLA.V","MLX.V","MMG.V","MNB.V","MND.V","MOD.V","MON.V","MPH.V",
        "MRC.V","MRD.V","MRG.V","MSG.V","MTB.V","MTO.V","MTS.V","MVG.V",
    ]

    _CSE_BASE = [
        # Cannabis
        "APHA.CN","ACB.CN","CRON.CN","OGI.CN","HEXO.CN","FIRE.CN","TGOD.CN",
        "TLRY.CN","NRGI.CN","CBDT.CN","XTRX.CN","GTII.CN","CURA.CN",
        # Crypto / Blockchain
        "BTCX.CN","EBIT.CN","GBTC.CN","BLOC.CN","DMGI.CN","HASH.CN",
        # Mining
        "CSE.CN","GGD.CN","GIGA.CN","GWX.CN","HAV.CN","HMX.CN","HPQ.CN",
        "HUB.CN","IBG.CN","IDM.CN","ILC.CN","IMG.CN","IMV.CN","INX.CN",
        "ION.CN","IPA.CN","IRM.CN","ISS.CN","IZZ.CN","JAG.CN","JNC.CN",
        # Tech
        "MAGT.CN","MARA.CN","MAPS.CN","MEDI.CN","MEDS.CN","META.CN",
        "MFON.CN","MGB.CN","MGMT.CN","MGR.CN","MHM.CN","MIA.CN",
        "NILI.CN","NIO.CN","NKL.CN","NLC.CN","NMI.CN","NNRG.CN",
        # Other
        "OGEN.CN","OGN.CN","OIII.CN","OMG.CN","ONE.CN","ONEX.CN",
        "ONOV.CN","OPC.CN","OPCT.CN","OPEN.CN","OR.CN","OREA.CN",
    ]

    _NEO_BASE = [
        # Major NEO-listed issuers
        "EGLX.NE","QBTC.NE","QETH.NE","BTCQ.NE","ETHQ.NE","BFIN.NE",
        "BANK.NE","PAYS.NE","PAYO.NE","HLTH.NE","WELL.NE","MEDI.NE",
        "HIVE.NE","BITF.NE","HUT.NE","GIGA.NE","BNXG.NE","DEFI.NE",
        "CBDC.NE","NBIT.NE","BTCC.NE","ETHH.NE","AVAX.NE","DOGE.NE",
        "PRBL.NE","SMCC.NE","MONA.NE","REAL.NE","BTRE.NE","BLDS.NE",
        "LIXT.NE","LIFE.NE","LQRT.NE","MSFT.NE","AMZN.NE","AAPL.NE",
        "NVDA.NE","GOOG.NE","TSLA.NE","META.NE","NFLX.NE","SHOP.NE",
    ]

    def _tsx(self) -> list:
        """TSX-listed companies with .TO suffix for yfinance."""
        tickers = []
        # Method 1: Wikipedia S&P/TSX Composite Index
        for wiki_url in [
            "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index",
            "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        ]:
            try:
                for t in pd.read_html(wiki_url):
                    for col in ("Ticker symbol","Symbol","Ticker","Ticker Symbol"):
                        if col in t.columns:
                            for sym in t[col].dropna():
                                s = str(sym).strip().replace(".","-")
                                if s and 1 <= len(s) <= 8 and s not in ("Symbol","Ticker"):
                                    tickers.append(s + ".TO")
                            break
                if tickers:
                    break
            except Exception:
                continue
        # Always add base list to ensure solid coverage
        tickers.extend(self._TSX_BASE)
        return list(set(tickers))

    def _tsxv(self) -> list:
        """TSXV-listed companies with .V suffix for yfinance."""
        return list(set(self._TSXV_BASE))

    def _cse(self) -> list:
        """CSE-listed companies with .CN suffix for yfinance."""
        tickers = list(self._CSE_BASE)
        # Try live CSE API
        for api_url in [
            "https://thecse.com/api/v1/securities?format=json&limit=2000",
            "https://thecse.com/en/listings/listed-securities.json",
        ]:
            try:
                r = requests.get(api_url, headers=self._HDRS, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    items = data if isinstance(data, list) else data.get("results", data.get("data", []))
                    for item in items:
                        sym = (item.get("symbol") or item.get("ticker","")).strip().upper()
                        if sym and 1 <= len(sym) <= 6 and " " not in sym:
                            tickers.append(sym if sym.endswith(".CN") else sym + ".CN")
                    break
            except Exception:
                continue
        return list(set(tickers))

    def _neo(self) -> list:
        """NEO / CBOE Canada listed companies with .NE suffix for yfinance."""
        tickers = list(self._NEO_BASE)
        # Try live fetch
        for api_url in [
            "https://api.cboe.ca/api/v1/equities/listed-issuers",
            "https://www.neo.inc/en/neo-exchange/listed-issuers",
        ]:
            try:
                r = requests.get(api_url, headers=self._HDRS, timeout=15)
                if r.status_code == 200 and r.headers.get("content-type","").startswith("application/json"):
                    data = r.json()
                    items = data if isinstance(data, list) else data.get("data", data.get("results", []))
                    for item in items:
                        sym = (item.get("symbol") or item.get("ticker","")).strip().upper()
                        if sym and 1 <= len(sym) <= 6:
                            tickers.append(sym if sym.endswith(".NE") else sym + ".NE")
                    break
            except Exception:
                continue
        return list(set(tickers))

    # ── Legacy method (kept for --universe nasdaq backward compat) ────────────

    def _nasdaq(self) -> list:
        """All US equities from NASDAQ Trader FTP (both files). Legacy."""
        return list(set(self._us_nasdaq() + self._us_other()))

    def _load_file(self, path: str) -> list:
        """Load tickers from a text or CSV file (one per line, or comma-separated)."""
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            # support comma, newline, semicolon, tab delimiters
            import re
            tokens = re.split(r"[,\n;\t]+", raw)
            return [t.strip().upper() for t in tokens if t.strip()]
        except Exception as e:
            print(f"  [warn] Could not load file '{path}': {e}")
            return []

    def _finviz_page(self, url: str) -> list:
        tickers = []
        try:
            offset = 1
            while True:
                r    = requests.get(url + f"&r={offset}", headers=self._HDRS, timeout=10)
                soup = BeautifulSoup(r.text, "lxml")
                cells = soup.select("a.screener-link-primary")
                if not cells:
                    break
                batch = [c.text.strip() for c in cells]
                tickers.extend(batch)
                if len(batch) < 20:
                    break
                offset += 20
                time.sleep(0.4)
        except Exception:
            pass
        return tickers

    def _russell2000(self) -> list:
        try:
            url = "https://en.wikipedia.org/wiki/Russell_2000_Index"
            for t in pd.read_html(url):
                for col in ("Symbol", "Ticker"):
                    if col in t.columns:
                        return t[col].str.replace(".", "-", regex=False).dropna().tolist()
        except Exception:
            pass
        return []

    def _sp500(self) -> list:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            return pd.read_html(url)[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception:
            return []

    def build(self, universe_type: str = "all", file_path: str = None) -> list:
        tickers: set = set()

        # ── File override ──────────────────────────────────────────────────────
        if file_path:
            batch = self._load_file(file_path)
            tickers.update(batch)
            print(f"  File '{file_path}': {len(batch)} tickers")
            return list(tickers)

        # ── Single-exchange shortcuts ──────────────────────────────────────────
        if universe_type == "nasdaq":
            batch = self._us_nasdaq()
            tickers.update(batch)
            print(f"  NASDAQ: {len(batch)} tickers")
            return list(tickers)

        if universe_type == "nyse":
            batch = self._us_other({"NYSE"})
            tickers.update(batch)
            print(f"  NYSE/NYSE Arca/NYSE American: {len(batch)} tickers")
            return list(tickers)

        if universe_type == "cboe":
            batch = self._us_other({"CBOE"})
            tickers.update(batch)
            print(f"  CBOE BZX: {len(batch)} tickers")
            return list(tickers)

        if universe_type == "otc":
            batch = self._otc_select()
            tickers.update(batch)
            print(f"  OTC Select (OTCQX/OTCQB): {len(batch)} tickers")
            return list(tickers)

        if universe_type == "tsx":
            batch = self._tsx()
            tickers.update(batch)
            print(f"  TSX (.TO): {len(batch)} tickers")
            return list(tickers)

        if universe_type == "tsxv":
            batch = self._tsxv()
            tickers.update(batch)
            print(f"  TSXV (.V): {len(batch)} tickers")
            return list(tickers)

        if universe_type == "cse":
            batch = self._cse()
            tickers.update(batch)
            print(f"  CSE (.CN): {len(batch)} tickers")
            return list(tickers)

        if universe_type == "neo":
            batch = self._neo()
            tickers.update(batch)
            print(f"  NEO/CBOE Canada (.NE): {len(batch)} tickers")
            return list(tickers)

        # ── Canadian-only preset ───────────────────────────────────────────────
        if universe_type == "canadian":
            for name, fn, suffix in [
                ("TSX",            self._tsx,  ".TO"),
                ("TSXV",           self._tsxv, ".V"),
                ("CSE",            self._cse,  ".CN"),
                ("NEO/CBOE Canada",self._neo,  ".NE"),
            ]:
                batch = fn()
                tickers.update(batch)
                print(f"  {name:<18}: {len(batch)} tickers ({suffix})")
            return list(tickers)

        # ── US-only preset ─────────────────────────────────────────────────────
        if universe_type == "us":
            batch = self._us_nasdaq()
            tickers.update(batch)
            print(f"  NASDAQ           : {len(batch)} tickers")
            batch = self._us_other({"NYSE"})
            tickers.update(batch)
            print(f"  NYSE             : {len(batch)} tickers")
            batch = self._us_other({"CBOE"})
            tickers.update(batch)
            print(f"  CBOE BZX         : {len(batch)} tickers")
            batch = self._otc_select()
            tickers.update(batch)
            print(f"  OTC Select       : {len(batch)} tickers")
            return list(tickers)

        # ── Default "all" — all specified exchanges ────────────────────────────
        if universe_type == "all":
            # Canadian exchanges
            for name, fn, suffix in [
                ("TSX",            self._tsx,  ".TO"),
                ("TSXV",           self._tsxv, ".V"),
                ("CSE",            self._cse,  ".CN"),
                ("NEO/CBOE Canada",self._neo,  ".NE"),
            ]:
                batch = fn()
                tickers.update(batch)
                print(f"  {name:<18}: {len(batch)} tickers ({suffix})")
            # US exchanges
            batch = self._us_nasdaq()
            tickers.update(batch)
            print(f"  NASDAQ           : {len(batch)} tickers")
            batch = self._us_other({"NYSE"})
            tickers.update(batch)
            print(f"  NYSE             : {len(batch)} tickers")
            batch = self._us_other({"CBOE"})
            tickers.update(batch)
            print(f"  CBOE BZX         : {len(batch)} tickers")
            batch = self._otc_select()
            tickers.update(batch)
            print(f"  OTC Select       : {len(batch)} tickers")

        # ── Legacy presets (still accessible via --universe) ──────────────────
        if universe_type == "smallcap":
            tickers.update(self._SMALLCAP)
            print(f"  Small-cap list   : {len(self._SMALLCAP)} tickers")

        if universe_type == "russell2000":
            batch = self._russell2000()
            tickers.update(batch)
            print(f"  Russell 2000     : {len(batch)} tickers")

        if universe_type in ("finviz", "microcap"):
            for i, url in enumerate(self._SCREENS, 1):
                batch = self._finviz_page(url)
                tickers.update(batch)
                print(f"  Finviz screen {i} : {len(batch)} tickers")

        # ── Fallbacks ──────────────────────────────────────────────────────────
        if not tickers:
            batch = self._sp500()
            tickers.update(batch)
            print(f"  S&P 500 fallback : {len(batch)} tickers")

        if not tickers:
            tickers = set(self._FALLBACK)
            print(f"  Hardcoded fallback: {len(tickers)} tickers")

        return list(tickers)

    def quick_filter(self, tickers: list, args) -> list:
        max_cap   = getattr(args, "max_cap",   None)
        max_float = getattr(args, "max_float", None)

        def _check(ticker):
            info = {}
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                pass

            # If info came back empty (401/rate-limit), do a lightweight price check
            price = float(info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0)
            if price == 0 and not info:
                try:
                    end   = datetime.now()
                    start = end - timedelta(days=10)
                    raw   = yf.download(ticker, start=start, end=end,
                                        progress=False, auto_adjust=True)
                    if raw.empty:
                        return None
                    df_  = _norm(raw)
                    price = float(df_["Close"].iloc[-1])
                    avg_v = float(df_["Volume"].mean())
                    if price < 0.50 or avg_v < 50_000:
                        return None
                    return {"ticker": ticker, "market_cap": 0, "price": price,
                            "avg_vol": avg_v, "float_shares": 0,
                            "short_pct": 0, "short_ratio": 0,
                            "sector": "", "qs": 0, "info": {}}
                except Exception:
                    return None

            qt = info.get("quoteType", "")
            if qt and qt != "EQUITY":
                return None
            mkt_cap = float(info.get("marketCap", 0) or 0)
            avg_vol = float(info.get("averageVolume", 0) or 0)
            if price < 0.50 or avg_vol < 50_000:
                return None
            if max_cap   and mkt_cap > max_cap   * 1e6: return None
            if max_float:
                fs = float(info.get("floatShares", 0) or 0)
                if fs > max_float * 1e6: return None
            h52  = float(info.get("fiftyTwoWeekHigh", price) or price)
            near = h52 > 0 and (h52 - price) / h52 <= 0.10
            sp   = float(info.get("shortPercentOfFloat", 0) or 0)
            qs   = int(near) + int(sp > 0.10)
            return {"ticker": ticker, "market_cap": mkt_cap, "price": price,
                    "avg_vol": avg_vol,
                    "float_shares": float(info.get("floatShares", 0) or 0),
                    "short_pct":    sp,
                    "short_ratio":  float(info.get("shortRatio", 0) or 0),
                    "sector":       info.get("sector", ""),
                    "qs":           qs,
                    "info":         info}

        passed = []
        with ThreadPoolExecutor(max_workers=8) as ex:   # 8 workers avoids 401 rate-limits
            futs = {ex.submit(_check, t): t for t in tickers}
            with tqdm(total=len(tickers), desc="  Quick filter", unit="stock", ncols=80) as bar:
                for fut in as_completed(futs):
                    r = fut.result()
                    if r:
                        passed.append(r)
                    bar.update(1)

        # sort: near-52W-high + high short interest first, cap at 400
        passed.sort(key=lambda x: (x.get("qs", 0), x["short_pct"]), reverse=True)
        return passed[:400]


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class BacktestEngine:
    """
    Replays pattern detection at historical dates and measures forward outcomes.
    For each ticker:
      - Fetches 400 days of data
      - Steps backward in 10-day increments over the backtest window
      - At each point, runs detection on data up to that day (no look-ahead)
      - Records outcome: did price hit the target or stop in the next `hold` days?
    """

    def __init__(self, scanner: "BreakoutScanner", days: int = 60, hold: int = 20):
        self.scanner = scanner
        self.days    = days   # how far back to generate signals
        self.hold    = hold   # forward window to check outcome
        self.signals = []

    def run(self, tickers: list) -> list:
        print(f"\nBacktest: last {self.days} trading days  |  {self.hold}-day hold  |  {len(tickers)} tickers\n")
        with ThreadPoolExecutor(max_workers=self.scanner.cfg["max_workers"]) as ex:
            futs = {ex.submit(self._replay_ticker, t): t for t in tickers}
            with tqdm(total=len(tickers), desc="Backtesting", unit="stock", ncols=80) as bar:
                for fut in as_completed(futs):
                    try:
                        sigs = fut.result()
                        if sigs:
                            self.signals.extend(sigs)
                    except Exception:
                        pass
                    finally:
                        bar.update(1)
        print(f"\n  {len(self.signals)} historical signals found across {len(tickers)} tickers\n")
        return self.signals

    def _replay_ticker(self, ticker: str) -> list:
        try:
            end   = datetime.now()
            start = end - timedelta(days=400)
            raw   = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty or len(raw) < 80:
                return []
            df_full = _norm(raw)
        except Exception:
            return []

        n       = len(df_full)
        signals = []
        # Earliest point where we have enough history + a full forward window
        end_back   = min(self.days + self.hold + 20, n - self.hold)
        start_back = self.hold + 20

        for back in range(end_back, start_back - 1, -10):
            cutoff = n - back
            if cutoff < 60:
                continue
            df_slice = df_full.iloc[:cutoff].copy()
            df_fwd   = df_full.iloc[cutoff:cutoff + self.hold]
            if len(df_fwd) < 5:
                continue

            try:
                df_ind = self.scanner._indicators(df_slice)
                det    = PatternDetector(df_ind, None, self.scanner.cfg)
                pats   = det.run_all()
                eng    = BreakoutProbabilityEngine(pats, df_ind, None, self.scanner.cfg)
                prob   = eng.calculate_probability()
            except Exception:
                continue

            if prob["probability"] < 50:
                continue

            entry   = float(df_slice["Close"].iloc[-1])
            atr_s   = _fcol(df_ind, "ATRr")
            atr_val = float(atr_s.iloc[-1]) if not atr_s.empty else entry * 0.02
            stop    = entry - atr_val * 1.5
            h_hist  = (float(df_slice["High"].values[-252:].max())
                       if cutoff >= 252 else float(df_slice["High"].max()))
            target  = h_hist

            # Walk forward day-by-day: first hit wins
            outcome = "neutral"
            for _, row in df_fwd.iterrows():
                if target > entry and float(row["High"]) >= target:
                    outcome = "win";  break
                if float(row["Low"]) <= stop:
                    outcome = "loss"; break
            if outcome == "neutral":
                ret     = (float(df_fwd["Close"].iloc[-1]) - entry) / entry if entry > 0 else 0
                outcome = "win" if ret > 0.02 else ("loss" if ret < -0.02 else "neutral")

            fwd_return = (float(df_fwd["Close"].iloc[-1]) - entry) / entry if entry > 0 else 0

            signals.append({
                "ticker":      ticker,
                "signal_date": str(df_slice.index[-1].date()),
                "pattern":     prob["primary_pattern"],
                "probability": prob["probability"],
                "outcome":     outcome,
                "fwd_return":  round(fwd_return, 4),
                "entry":       round(entry, 2),
                "stop":        round(stop, 2),
                "target":      round(target, 2),
                "modifiers":   prob["positive_mods"] + prob["negative_mods"],
            })

        return signals

    def print_report(self):
        if not self.signals:
            print("No signals generated in the backtest window.")
            return

        df    = pd.DataFrame(self.signals)
        total = len(df)
        wins  = (df["outcome"] == "win").sum()
        losses= (df["outcome"] == "loss").sum()
        neut  = (df["outcome"] == "neutral").sum()
        wr    = wins / total * 100 if total > 0 else 0
        avg_r = df["fwd_return"].mean() * 100

        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  BACKTEST RESULTS  |  {total} signals  |  {self.hold}-day hold period")
        print(f"{sep}")
        print(f"  Overall Win Rate  : {wr:.1f}%   ({wins}W / {losses}L / {neut} neutral)")
        print(f"  Avg Forward Return: {avg_r:+.2f}%")
        print()

        # Per-pattern breakdown
        print(f"  {'Pattern':<26} {'N':>4}  {'Win%':>6}  {'AvgRet':>8}")
        print(f"  {'-'*26} {'-'*4}  {'-'*6}  {'-'*8}")
        for pat, grp in df.groupby("pattern"):
            pw  = (grp["outcome"] == "win").sum()
            pr  = grp["fwd_return"].mean() * 100
            pwr = pw / len(grp) * 100
            print(f"  {pat:<26} {len(grp):>4}  {pwr:>5.1f}%  {pr:>+7.2f}%")

        # Modifier correlation analysis
        win_mods  = {}
        loss_mods = {}
        for _, row in df.iterrows():
            bucket = win_mods if row["outcome"] == "win" else loss_mods
            for lbl, _ in row["modifiers"]:
                bucket[lbl] = bucket.get(lbl, 0) + 1

        all_mods = set(win_mods) | set(loss_mods)
        mod_data = []
        for m in all_mods:
            w = win_mods.get(m, 0); l = loss_mods.get(m, 0)
            if w + l >= 3:
                mod_data.append((m, w, l, w / (w + l) * 100))

        if mod_data:
            print(f"\n  Best modifiers (highest win rate, min 3 occurrences):")
            print(f"  {'Modifier':<50} {'N':>4}  {'Win%':>6}")
            print(f"  {'-'*50} {'-'*4}  {'-'*6}")
            for m, w, l, wr_ in sorted(mod_data, key=lambda x: -x[3])[:8]:
                print(f"  {m[:50]:<50} {w+l:>4}  {wr_:>5.1f}%")

            print(f"\n  Worst modifiers (lowest win rate):")
            for m, w, l, wr_ in sorted(mod_data, key=lambda x: x[3])[:5]:
                print(f"  {m[:50]:<50} {w+l:>4}  {wr_:>5.1f}%")
        print()

    def to_csv(self) -> str | None:
        if not self.signals:
            return None
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"backtest_{ts}.csv"
        rows = [{k: v for k, v in s.items() if k != "modifiers"} for s in self.signals]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Backtest exported: {path}")
        return path


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
class WeightOptimizer:
    """
    Analyzes BacktestEngine results and updates weights.json.

    Logic:
      - Compute baseline win rate across all backtest signals
      - For each modifier, compute its empirical win rate
      - If a modifier's win rate deviates >10% from baseline, adjust its weight
        (+1 point per 10% deviation, capped at ±5 points)
      - Blend empirical base rates into BASE_RATES adjustments (30% weight)
    Results accumulate across runs — each --backtest --learn pass refines further.
    """

    def update(self, signals: list, verbose: bool = True):
        if len(signals) < 10:
            print(f"  Only {len(signals)} signals — need 10+ to learn. "
                  "Use a larger watchlist or wider --backtest window.")
            return

        df           = pd.DataFrame(signals)
        baseline_wr  = (df["outcome"] == "win").sum() / len(df)

        # Collect per-modifier win/loss counts
        mod_stats: dict = {}
        for _, row in df.iterrows():
            is_win = row["outcome"] == "win"
            for lbl, val in row["modifiers"]:
                if lbl not in mod_stats:
                    mod_stats[lbl] = {"wins": 0, "total": 0, "default": val}
                mod_stats[lbl]["total"] += 1
                if is_win:
                    mod_stats[lbl]["wins"] += 1

        data        = _load_weights()
        adjustments = data.get("adjustments", {})
        changed     = []

        for lbl, stats in mod_stats.items():
            if stats["total"] < 3:
                continue
            mod_wr  = stats["wins"] / stats["total"]
            delta   = mod_wr - baseline_wr
            new_adj = int(round(delta * 10))
            new_adj = max(-5, min(5, new_adj))
            old_adj = adjustments.get(lbl, 0)
            if abs(new_adj - old_adj) >= 1:
                adjustments[lbl] = new_adj
                changed.append((lbl, old_adj, new_adj, mod_wr * 100, stats["total"]))

        # Blend empirical base rates
        br_adj = data.get("base_rate_adjustments", {})
        for pat, grp in df.groupby("pattern"):
            if len(grp) >= 5:
                emp_wr  = (grp["outcome"] == "win").sum() / len(grp) * 100
                current = BASE_RATES.get(pat, 50)
                delta   = int(round((emp_wr - current) * 0.3))
                br_adj[pat] = max(-10, min(10, delta))

        data.update({
            "adjustments":            adjustments,
            "base_rate_adjustments":  br_adj,
            "samples":                data.get("samples", 0) + len(signals),
            "last_updated":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        _save_weights(data)

        # Refresh in-memory weights immediately
        global LEARNED_WEIGHTS
        LEARNED_WEIGHTS = data

        if verbose:
            print(f"\n  Learning complete — {len(signals)} signals analyzed")
            print(f"  Baseline win rate: {baseline_wr*100:.1f}%")
            if changed:
                print(f"  {len(changed)} weight adjustments applied:")
                for lbl, old, new, wr, n in sorted(changed, key=lambda x: abs(x[2] - x[1]), reverse=True):
                    print(f"    {lbl[:50]:<50}  {old:+d} -> {new:+d}  "
                          f"(win rate {wr:.0f}%, n={n})")
            else:
                print("  No significant adjustments needed with current sample.")
            print(f"  Accumulated samples: {data['samples']}")
            print(f"  Saved to {WEIGHTS_FILE}")

    def reset(self):
        global LEARNED_WEIGHTS
        if os.path.exists(WEIGHTS_FILE):
            os.remove(WEIGHTS_FILE)
            print(f"  Weights reset — {WEIGHTS_FILE} deleted.")
        else:
            print("  No weights file found (already at defaults).")
        LEARNED_WEIGHTS = {"adjustments": {}, "base_rate_adjustments": {}, "samples": 0}

    @staticmethod
    def status():
        data = _load_weights()
        n    = data.get("samples", 0)
        ts   = data.get("last_updated", "never")
        adjs = data.get("adjustments", {})
        print(f"\n  Weights file : {WEIGHTS_FILE}")
        print(f"  Last updated : {ts}")
        print(f"  Samples seen : {n}")
        print(f"  Adjustments  : {len(adjs)} modifiers tuned")
        if adjs:
            for lbl, val in sorted(adjs.items(), key=lambda x: -abs(x[1])):
                print(f"    {lbl[:52]:<52}  {val:+d}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO TRACKER & HISTORICAL CALL LOGGER
# ══════════════════════════════════════════════════════════════════════════════

_DB_FILE  = "trading_scanner_history.db"
_POS_SIZE = 10_000.0   # virtual $ per tracked position


# ── Intraday volume pace ───────────────────────────────────────────────────────
def _intraday_volume_pace(ticker: str) -> float:
    """Compare today's accumulated volume to what's historically normal at this
    time of day.

    Returns a ratio:
        1.0 = exactly on pace
        2.0 = running at double the expected pace  (strong confirmation)
        0.5 = running at half the expected pace    (weak, skip entry)

    Returns 1.0 (neutral) if data is unavailable so we never falsely block trades.

    Volume schedule model (approximates typical intraday distribution):
        First 30 min  (9:30–10:00): ~20% of daily volume  (opening rush)
        Remaining 360 min (10:00–16:00): ~80% spread roughly uniformly
    """
    try:
        # 5 days of 1-min bars gives today + up to 4 prior complete days
        df5 = yf.download(ticker, period="5d", interval="1m",
                          progress=False, auto_adjust=True)
        if df5 is None or df5.empty:
            return 1.0
        if isinstance(df5.columns, pd.MultiIndex):
            df5.columns = df5.columns.get_level_values(0)

        # Normalise the index to ET dates (handles DST automatically)
        idx = df5.index
        try:
            if hasattr(idx, "tz") and idx.tz is not None:
                idx_et = idx.tz_convert("America/New_York")
            else:
                idx_et = idx.tz_localize("America/New_York", ambiguous="infer",
                                         nonexistent="shift_forward")
            dates = pd.DatetimeIndex([t.date() for t in idx_et])
        except Exception:
            # Fallback: assume index dates are already local
            dates = pd.DatetimeIndex([t.date() for t in idx])

        now_et   = MarketClock.now_et()
        today    = now_et.date()

        today_mask = (dates == today)
        today_vol  = float(df5["Volume"].values[today_mask].sum())
        if today_vol <= 0:
            return 1.0

        # Average daily volume from the prior days in the window
        prev_mask  = (dates < today)
        prev_dates = sorted(set(dates[prev_mask]))
        if not prev_dates:
            return 1.0
        day_vols = [float(df5["Volume"].values[dates == d].sum())
                    for d in prev_dates if (dates == d).any()]
        adv = float(np.mean(day_vols)) if day_vols else 0.0
        if adv <= 0:
            return 1.0

        # Minutes elapsed since 9:30 ET
        elapsed = (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30)
        elapsed = max(0, min(int(elapsed), 390))          # clamp to [0, 390]
        if elapsed <= 0:
            return 1.0

        # Piecewise expected-fraction model
        if elapsed <= 30:
            expected_frac = 0.20 * (elapsed / 30.0)
        else:
            expected_frac = 0.20 + 0.80 * ((elapsed - 30.0) / 360.0)

        expected_vol = adv * expected_frac
        return (today_vol / expected_vol) if expected_vol > 0 else 1.0

    except Exception:
        return 1.0


_PAPER_BUDGET        = 1_000.0
_PAPER_MAX_POSITIONS = 10        # ← raised from 5 to give the bot more room
_PAPER_MAX_PCT       = 0.15      # ← lowered from 0.20 to suit smaller slot sizes
_PAPER_MIN_CASH      = 50.0      # ← lowered from 100 so the 10th slot can fill


class PaperTradingEngine:
    """$1,000 paper portfolio that auto-trades top signals with fee simulation."""

    _migrated_conns: set = set()   # tracks which connections have been migrated

    def __init__(self, conn):
        self.conn = conn
        self.fee  = _FEE_ENGINE
        self._init_schema()
        self._load_state()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker            TEXT,
                entry_price       REAL,
                entry_date        DATE,
                shares            REAL,
                gross_invested    REAL,
                buy_fee           REAL,
                stop_loss         REAL,
                target_price      REAL,
                fee_adj_rr        REAL,
                pattern           TEXT,
                explosive_score   REAL,
                breakout_prob     REAL,
                status            TEXT DEFAULT 'OPEN',
                exit_price        REAL,
                exit_date         DATE,
                exit_reason       TEXT,
                sell_fee          REAL,
                gross_pnl         REAL,
                net_pnl           REAL,
                net_pnl_pct       REAL,
                t1_price          REAL,
                t1_hit            INTEGER DEFAULT 0,
                trailing_stop     REAL,
                highest_since_t1  REAL,
                atr_value         REAL,
                sector            TEXT
            );
            CREATE TABLE IF NOT EXISTS paper_state (
                id               INTEGER PRIMARY KEY,
                total_capital    REAL,
                available_cash   REAL,
                total_fees_paid  REAL,
                realized_pnl     REAL,
                trades_made      INTEGER,
                starting_capital REAL
            );
            CREATE TABLE IF NOT EXISTS fees (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id          INTEGER,
                ticker           TEXT,
                transaction_type TEXT,
                transaction_date DATE,
                gross_amount     REAL,
                fee_amount       REAL,
                net_amount       REAL,
                running_total    REAL,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            );
            CREATE TABLE IF NOT EXISTS paper_trading (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date        DATE,
                total_capital        REAL,
                available_cash       REAL,
                invested_value       REAL,
                unrealized_pnl       REAL,
                realized_pnl         REAL,
                total_fees_paid      REAL,
                open_positions       INTEGER,
                portfolio_return_pct REAL,
                spy_return_pct       REAL,
                alpha                REAL
            );
        """)
        # ── Schema migration: add new columns to existing databases ───────────
        # Use the connection id as key so migrations only run once per
        # connection object — avoids repeated ALTER TABLE failures on every
        # PaperTradingEngine instantiation when using a cached connection.
        _conn_key = id(self.conn)
        if _conn_key not in PaperTradingEngine._migrated_conns:
            PaperTradingEngine._migrated_conns.add(_conn_key)
            _new_cols = [
                ("paper_portfolio", "t1_price",         "REAL"),
                ("paper_portfolio", "t1_hit",            "INTEGER DEFAULT 0"),
                ("paper_portfolio", "trailing_stop",     "REAL"),
                ("paper_portfolio", "highest_since_t1",  "REAL"),
                ("paper_portfolio", "atr_value",         "REAL"),
                ("paper_portfolio", "sector",            "TEXT"),
            ]
            for _tbl, _col, _typ in _new_cols:
                try:
                    self.conn.execute(
                        f"ALTER TABLE {_tbl} ADD COLUMN {_col} {_typ}")
                except Exception:
                    pass   # column already exists — safe to ignore

    def _load_state(self):
        try:
            row = self.conn.execute(
                "SELECT * FROM paper_state WHERE id=1").fetchone()
        except Exception:
            row = None
        if row:
            try:
                self._cash         = float(row.get("available_cash",  0) or 0)
                self._fees_paid    = float(row.get("total_fees_paid",  0) or 0)
                self._realized_pnl = float(row.get("realized_pnl",    0) or 0)
                self._trades       = int(  row.get("trades_made",      0) or 0)
                self._starting     = float(row.get("starting_capital", _PAPER_BUDGET) or _PAPER_BUDGET)
            except Exception:
                row = None   # fall through to default init below
        if not row:
            self._cash         = _PAPER_BUDGET
            self._fees_paid    = 0.0
            self._realized_pnl = 0.0
            self._trades       = 0
            self._starting     = _PAPER_BUDGET
            self._save_state()

        # ── Self-healing: fix stale state left by the old ON CONFLICT DO NOTHING bug ──
        # If cash looks untouched (≈ starting capital) but open positions exist,
        # the paper_state row was never properly updated. Recalculate from the
        # actual paper_portfolio records and overwrite the stale row.
        try:
            positions = self.open_positions
            invested  = sum((p.get("gross_invested") or 0) for p in positions)
            if invested > 10 and self._cash >= self._starting * 0.98:
                self._recalculate_from_portfolio()
        except Exception:
            pass

    def _recalculate_from_portfolio(self):
        """Derive accurate cash / fees / realized P&L from actual DB records."""
        try:
            # Open positions
            positions = self.open_positions
            invested  = sum((p.get("gross_invested") or 0) for p in positions)
            open_fees = sum((p.get("buy_fee")         or 0) for p in positions)

            # Closed positions
            cr = self.conn.execute("""
                SELECT COUNT(*),
                       SUM(COALESCE(net_pnl,0)),
                       SUM(COALESCE(buy_fee,0) + COALESCE(sell_fee,0))
                FROM paper_portfolio WHERE status='CLOSED'
            """).fetchone()
            n_closed    = int(  cr[0] or 0) if cr else 0
            closed_pnl  = float(cr[1] or 0) if cr else 0.0
            closed_fees = float(cr[2] or 0) if cr else 0.0

            self._realized_pnl = closed_pnl
            self._fees_paid    = open_fees + closed_fees
            self._trades       = len(positions) + n_closed
            # cash = what's left after all open investment, plus any realised gains
            self._cash         = max(0.0, self._starting + closed_pnl - invested)
            self._save_state()
        except Exception:
            pass

    def _save_state(self):
        # DELETE + INSERT avoids the PostgreSQL "ON CONFLICT DO NOTHING" translation
        # that PgAdapter applies to INSERT OR REPLACE — which silently ignores updates
        # once row id=1 already exists.  This pattern works for both SQLite and PG.
        try:
            self.conn.execute("DELETE FROM paper_state WHERE id=1")
        except Exception:
            pass
        self.conn.execute("""
            INSERT INTO paper_state
            (id, total_capital, available_cash, total_fees_paid,
             realized_pnl, trades_made, starting_capital)
            VALUES (?,?,?,?,?,?,?)
        """, (1, self.total_capital, self._cash, self._fees_paid,
              self._realized_pnl, self._trades, self._starting))
        self.conn.commit()

    @property
    def open_positions(self) -> list:
        rows = self.conn.execute(
            "SELECT * FROM paper_portfolio WHERE status='OPEN'").fetchall()
        return [dict(r) for r in rows]

    @property
    def total_capital(self) -> float:
        invested = sum((p.get("gross_invested") or 0) - (p.get("buy_fee") or 0)
                       for p in self.open_positions)
        return self._cash + invested

    def open_position(self, ticker: str, signal: dict) -> dict:
        positions = self.open_positions
        if len(positions) >= _PAPER_MAX_POSITIONS:
            return {"success": False,
                    "reason": f"Max positions reached ({_PAPER_MAX_POSITIONS})"}
        if any(p["ticker"] == ticker for p in positions):
            return {"success": False, "reason": f"Already holding {ticker}"}
        if self._cash < _PAPER_MIN_CASH:
            return {"success": False,
                    "reason": f"Insufficient cash (${self._cash:.2f})"}

        price = signal.get("price") or 0
        if not price:
            return {"success": False, "reason": "No price data"}

        # ── ATR-based position sizing ──────────────────────────────────────────
        # Risk exactly 1.5% of current total capital per trade
        stop_loss  = signal.get("stop_price") or (price * 0.95)
        stop_dist  = max(abs(price - stop_loss), price * 0.02)  # floor at 2%
        risk_amt   = self.total_capital * 0.015          # 1.5% risk per trade
        shares_rsk = risk_amt / stop_dist                # shares implied by risk
        gross_risk = shares_rsk * price                  # gross $ from risk rule

        # Cap at 20% of portfolio AND even slot split
        slots      = max(1, _PAPER_MAX_POSITIONS - len(positions))
        gross_slot = self._cash / slots
        gross      = min(gross_risk, self._cash * _PAPER_MAX_PCT, gross_slot)
        gross      = max(gross, _PAPER_MIN_CASH)
        if gross > self._cash:
            gross = self._cash

        buy_calc = self.fee.calculate_buy(gross, price)

        self._cash      -= gross
        self._fees_paid += buy_calc["fee"]
        self._trades    += 1
        today = datetime.now().strftime("%Y-%m-%d")

        # Derive T1 price and ATR value from signal or compute fallback
        _stop   = signal.get("stop_price") or (price * 0.95)
        _risk   = abs(price - _stop)
        _t1     = signal.get("t1_price") or (price + _risk * 1.5)
        _atr    = signal.get("atr_value") or _risk / 2.0
        _sector = signal.get("sector", "") or ""

        self.conn.execute("""
            INSERT INTO paper_portfolio
            (ticker, entry_price, entry_date, shares, gross_invested, buy_fee,
             stop_loss, target_price, fee_adj_rr, pattern, explosive_score,
             breakout_prob, t1_price, trailing_stop, atr_value, sector)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ticker, price, today, buy_calc["shares"], gross, buy_calc["fee"],
              _stop, signal.get("tgt_price", 0),
              signal.get("fee_adj_rr", 0), signal.get("pattern", ""),
              signal.get("explosive_score", 0), signal.get("probability", 0),
              _t1, _stop, _atr, _sector))

        self.conn.execute("""
            INSERT INTO fees
            (ticker, transaction_type, transaction_date, gross_amount,
             fee_amount, net_amount, running_total)
            VALUES (?,?,?,?,?,?,?)
        """, (ticker, "BUY", today, gross,
              buy_calc["fee"], buy_calc["net_investment"], self._fees_paid))
        self._save_state()
        return {"success": True, "shares": round(buy_calc["shares"], 4),
                "gross": gross, "fee": round(buy_calc["fee"], 2),
                "effective_price": round(buy_calc["effective_price"], 4)}

    def close_position(self, ticker: str, exit_price: float,
                       reason: str = "MANUAL") -> dict:
        row = self.conn.execute(
            "SELECT * FROM paper_portfolio WHERE ticker=? AND status='OPEN'",
            (ticker,)).fetchone()
        if not row:
            return {"success": False, "reason": f"No open position for {ticker}"}
        pos      = dict(row)
        sell     = self.fee.calculate_sell(pos["shares"], exit_price)
        gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        net_pnl   = gross_pnl - (pos.get("buy_fee") or 0) - sell["fee"]
        net_pct   = net_pnl / (pos.get("gross_invested") or 1) * 100
        today     = datetime.now().strftime("%Y-%m-%d")

        self._cash         += sell["net"]
        self._fees_paid    += sell["fee"]
        self._realized_pnl += net_pnl

        self.conn.execute("""
            UPDATE paper_portfolio
            SET status='CLOSED', exit_price=?, exit_date=?, exit_reason=?,
                sell_fee=?, gross_pnl=?, net_pnl=?, net_pnl_pct=?
            WHERE ticker=? AND status='OPEN'
        """, (exit_price, today, reason, sell["fee"],
              gross_pnl, net_pnl, net_pct, ticker))

        self.conn.execute("""
            INSERT INTO fees
            (ticker, transaction_type, transaction_date, gross_amount,
             fee_amount, net_amount, running_total)
            VALUES (?,?,?,?,?,?,?)
        """, (ticker, "SELL", today, sell["gross"],
              sell["fee"], sell["net"], self._fees_paid))
        self._save_state()
        return {"success": True, "exit_price": exit_price,
                "sell_fee": round(sell["fee"], 2),
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2), "net_pnl_pct": round(net_pct, 1)}

    def check_stops_and_targets(self) -> list:
        """Check all open positions against live intraday prices.

        Uses 1-minute candles and checks High/Low of each candle for precise
        stop/target execution.  Also:
          • T1 hit → moves stop to break-even (entry price)
          • Trailing stop → trails at (highest_since_t1 × 0.92) after T1
          • Time-based exit → closes stagnant positions after 10 trading days
            if they haven't reached T1 yet

        Returns a list of close-result dicts for every position that was exited.
        """
        closed = []
        for pos in self.open_positions:
            try:
                ticker = pos["ticker"]
                stop   = float(pos.get("stop_loss")   or 0)
                target = float(pos.get("target_price") or 0)
                t1     = float(pos.get("t1_price")     or 0)
                entry  = float(pos.get("entry_price")  or 0)
                atr_v  = float(pos.get("atr_value")    or (entry * 0.02))

                # ── 1-min data (fallback chain: 1m → 5m → 1d) ─────────────
                raw = yf.download(ticker, period="2d", interval="1m",
                                  progress=False, auto_adjust=True)
                if raw is None or raw.empty:
                    raw = yf.download(ticker, period="5d", interval="5m",
                                      progress=False, auto_adjust=True)
                if raw is None or raw.empty:
                    raw = yf.download(ticker, period="5d", interval="1d",
                                      progress=False, auto_adjust=True)
                if raw is None or raw.empty:
                    continue

                # Flatten MultiIndex columns (yfinance ≥ 0.2.x)
                if isinstance(raw.columns, pd.MultiIndex):
                    _price_labels = {"Open", "High", "Low", "Close", "Volume"}
                    _lvl = 0 if _price_labels & set(raw.columns.get_level_values(0)) else 1
                    raw.columns = raw.columns.get_level_values(_lvl)
                # Drop rows where OHLC data is completely missing
                raw = raw.dropna(subset=["Close"])
                if raw.empty:
                    continue

                # Filter to candles on or after entry date
                entry_date = pos.get("entry_date")
                if entry_date:
                    try:
                        ed   = pd.Timestamp(entry_date).tz_localize(None)
                        ridx = raw.index.tz_localize(None) if raw.index.tzinfo else raw.index
                        raw  = raw[ridx >= ed]
                    except Exception:
                        pass

                if raw.empty:
                    continue

                reason     = None
                exit_price = None
                last_close = float(raw["Close"].dropna().iloc[-1])

                # ── T1 / trailing stop updates (before exit check) ─────────
                t1_hit     = bool(pos.get("t1_hit"))
                highest    = float(pos.get("highest_since_t1") or 0)

                if t1 and not t1_hit:
                    # Check if any candle's High has reached T1
                    if float(raw["High"].max()) >= t1:
                        # T1 hit → move stop to break-even
                        new_stop = max(entry, stop)   # never lower the stop
                        self.conn.execute("""
                            UPDATE paper_portfolio
                            SET t1_hit=1, stop_loss=?, trailing_stop=?,
                                highest_since_t1=?
                            WHERE ticker=? AND status='OPEN'
                        """, (new_stop, new_stop, last_close, ticker))
                        self.conn.commit()
                        stop    = new_stop
                        t1_hit  = True
                        highest = last_close

                if t1_hit:
                    # Trailing stop: trail at 8% below the highest close since T1
                    if last_close > highest:
                        highest = last_close
                        self.conn.execute("""
                            UPDATE paper_portfolio SET highest_since_t1=?
                            WHERE ticker=? AND status='OPEN'
                        """, (highest, ticker))
                        self.conn.commit()
                    trail = highest * 0.92          # 8% trailing cushion
                    if trail > stop:
                        self.conn.execute("""
                            UPDATE paper_portfolio SET stop_loss=?, trailing_stop=?
                            WHERE ticker=? AND status='OPEN'
                        """, (trail, trail, ticker))
                        self.conn.commit()
                        stop = trail

                # ── Scan candles for exact stop / target hits ──────────────
                for _, row in raw.iterrows():
                    low   = float(row.get("Low",   row.get("Close", 0)))
                    high  = float(row.get("High",  row.get("Close", 0)))

                    if target and high >= target:
                        reason     = "TARGET_HIT"
                        exit_price = target
                        break
                    if stop and low <= stop:
                        reason     = "STOP_HIT"
                        exit_price = stop
                        break

                # Gap / EOD backstop
                if not reason:
                    if stop   and last_close <= stop:
                        reason = "STOP_HIT";   exit_price = last_close
                    elif target and last_close >= target:
                        reason = "TARGET_HIT"; exit_price = last_close

                # ── Time-based exit (10 trading days without T1) ───────────
                if not reason and entry_date and not t1_hit:
                    try:
                        bdays = len(pd.bdate_range(str(entry_date),
                                                   str(datetime.now().date()))) - 1
                        if bdays >= 10:
                            reason     = "TIME_STOP"
                            exit_price = last_close
                    except Exception:
                        pass

                if reason:
                    res = self.close_position(ticker, exit_price, reason)
                    res["ticker"] = ticker
                    closed.append(res)

            except Exception:
                pass
        return closed

    def auto_trade(self, scan_results: list) -> list:
        # Always check stops/targets on open positions first
        self.check_stops_and_targets()

        # ── Trading-window gate ────────────────────────────────────────────────
        session = MarketClock.get_session()
        self._last_session = session  # expose for UI messaging
        if session["quality"] == "AVOID":
            # Opening / closing volatility — skip new entries, protect existing ones
            self._save_snapshot()
            return []

        # In CAUTION or PREMARKET, require higher-conviction signals
        min_prob  = 65
        min_score = 70
        if session["quality"] in ("CAUTION", "PREMARKET"):
            min_prob  = 72
            min_score = 75

        candidates = [r for r in scan_results
                      if r.get("trade_filter_pass")
                      and (r.get("explosive_score", 0) >= min_score
                           or r.get("probability", 0) >= min_prob)]

        # Sort by combined conviction: explosive score + probability
        candidates.sort(
            key=lambda x: (x.get("explosive_score", 0) * 0.6 +
                           x.get("probability",     0) * 0.4),
            reverse=True,
        )
        opened = []
        for sig in candidates:
            # ── Sector rotation gate ───────────────────────────────────────────
            # Skip entries in lagging sectors — even great setups fail when
            # institutional money is rotating OUT of that sector.
            sector = sig.get("sector", "") or ""
            if sector:
                sector_tier = SectorRotationDetector.classify(sector)
                if sector_tier == "LAGGING":
                    sig["_sector_skip"] = True
                    sig["_sector_tier"] = sector_tier
                    continue
                sig["_sector_tier"] = sector_tier

            # ── Volume-at-time-of-day gate ─────────────────────────────────────
            # Skip entry if volume is tracking below 50% of expected pace —
            # a breakout on anemic volume rarely follows through.
            if session.get("is_open"):
                pace = _intraday_volume_pace(sig["ticker"])
                if pace < 0.50:
                    sig["_vol_pace_skip"] = True
                    sig["_vol_pace"]      = round(pace, 2)
                    continue
                sig["_vol_pace"] = round(pace, 2)

            res = self.open_position(sig["ticker"], sig)
            if res.get("success"):
                opened.append({
                    "ticker":      sig["ticker"],
                    "session":     session["quality"],
                    "vol_pace":    sig.get("_vol_pace", None),
                    "sector_tier": sig.get("_sector_tier", "NEUTRAL"),
                    **res,
                })
        self._save_snapshot()
        return opened

    def _save_snapshot(self):
        try:
            positions = self.open_positions
            invested  = sum((p.get("gross_invested") or 0) - (p.get("buy_fee") or 0)
                            for p in positions)
            today     = datetime.now().strftime("%Y-%m-%d")
            ret_pct   = (self.total_capital - self._starting) / self._starting * 100
            self.conn.execute("""
                INSERT INTO paper_trading
                (snapshot_date, total_capital, available_cash, invested_value,
                 unrealized_pnl, realized_pnl, total_fees_paid, open_positions,
                 portfolio_return_pct, spy_return_pct, alpha)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (today, self.total_capital, self._cash, invested,
                  0.0, self._realized_pnl, self._fees_paid, len(positions),
                  ret_pct, 0.0, 0.0))
            self.conn.commit()
        except Exception:
            pass

    def reset(self, budget: float = _PAPER_BUDGET):
        self.conn.executescript("""
            DELETE FROM paper_portfolio;
            DELETE FROM paper_state;
            DELETE FROM fees;
            DELETE FROM paper_trading;
        """)
        self._cash = budget; self._fees_paid = 0.0
        self._realized_pnl = 0.0; self._trades = 0; self._starting = budget
        self._save_state()

    def get_summary(self) -> dict:
        positions  = self.open_positions
        invested   = sum((p.get("gross_invested") or 0) - (p.get("buy_fee") or 0)
                         for p in positions)
        total      = self._cash + invested
        total_ret  = total - self._starting
        ret_pct    = total_ret / self._starting * 100 if self._starting else 0
        try:
            row = self.conn.execute(
                "SELECT COUNT(*), SUM(net_pnl) FROM paper_portfolio WHERE status='CLOSED'"
            ).fetchone()
            n_closed = int(row[0] or 0) if row else 0
        except Exception:
            n_closed = 0
        return {
            "starting_capital": self._starting,
            "total_capital":    total,
            "available_cash":   self._cash,
            "invested_value":   invested,
            "total_return":     total_ret,
            "total_return_pct": round(ret_pct, 2),
            "total_fees_paid":  self._fees_paid,
            "fee_drag_pct":     round(self._fees_paid / self._starting * 100, 2)
                                if self._starting else 0,
            "realized_pnl":     self._realized_pnl,
            "unrealized_pnl":   invested - sum((p.get("gross_invested") or 0)
                                               for p in positions),
            "open_positions":   len(positions),
            "max_positions":    _PAPER_MAX_POSITIONS,
            "trades_made":      self._trades,
            "n_closed":         n_closed,
            "break_even_pct":   _FEE_ENGINE.break_even_move(1.0)["break_even_pct"],
        }

    def get_unrealized_pnl(self) -> dict:
        """Fetch live prices via yfinance and compute unrealized P&L for all open positions.

        Returns a dict with keys:
          total               – total unrealized P&L dollars
          pct                 – total unrealized P&L as % of cost
          positions           – list of per-position dicts
          total_current_value – sum of (shares × live_price) for all positions
          total_cost          – sum of gross_invested for all positions
          fetched             – True if at least one live price was retrieved
        """
        positions = self.open_positions
        if not positions:
            return {"total": 0.0, "pct": 0.0, "positions": [],
                    "total_current_value": 0.0, "total_cost": 0.0, "fetched": False}

        tickers = [p["ticker"] for p in positions]
        prices  = {}
        fetched = False

        try:
            if len(tickers) == 1:
                df = yf.download(tickers[0], period="2d", interval="5m",
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    prices[tickers[0]] = float(df["Close"].dropna().iloc[-1])
            else:
                df = yf.download(tickers, period="2d", interval="5m",
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    close_data = df.get("Close", df)
                    if isinstance(close_data, pd.DataFrame):
                        for tk in tickers:
                            if tk in close_data.columns:
                                s = close_data[tk].dropna()
                                if not s.empty:
                                    prices[tk] = float(s.iloc[-1])
                    elif isinstance(close_data, pd.Series):
                        s = close_data.dropna()
                        if not s.empty and tickers:
                            prices[tickers[0]] = float(s.iloc[-1])
            fetched = bool(prices)
        except Exception:
            pass

        result_positions = []
        total_unrealized = 0.0
        total_cost       = 0.0
        total_current    = 0.0

        for p in positions:
            tk     = p["ticker"]
            ep     = float(p.get("entry_price") or 0)
            shares = float(p.get("shares") or 0)
            gross  = float(p.get("gross_invested") or 0)   # cash deployed (incl buy fee)

            cur      = prices.get(tk)
            has_live = cur is not None
            if not has_live:
                cur = ep  # fall back to entry price so maths are neutral

            cur_value = shares * cur
            unr_pnl   = cur_value - gross
            unr_pct   = (unr_pnl / gross * 100) if gross else 0.0

            total_unrealized += unr_pnl
            total_cost       += gross
            total_current    += cur_value

            result_positions.append({
                "ticker":         tk,
                "entry_price":    ep,
                "current_price":  cur,
                "shares":         shares,
                "cost_basis":     gross,
                "current_value":  cur_value,
                "unrealized_pnl": unr_pnl,
                "unrealized_pct": unr_pct,
                "has_live_price": has_live,
            })

        total_pct = (total_unrealized / total_cost * 100) if total_cost else 0.0
        return {
            "total":               total_unrealized,
            "pct":                 total_pct,
            "positions":           result_positions,
            "total_current_value": total_current,
            "total_cost":          total_cost,
            "fetched":             fetched,
        }


class CallLogger:
    """Records every scanner call and tracks outcomes persistently via SQLite."""

    MIN_SCORE = 40
    MIN_PROB  = 55

    def __init__(self, db: str = _DB_FILE):
        self.db   = db
        self.conn = sqlite3.connect(db, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self):
        self.conn.close()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS calls (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id           TEXT,
                scan_timestamp    DATETIME,
                ticker            TEXT,
                entry_price       REAL,
                entry_date        DATE,
                explosive_score   REAL,
                explosive_grade   TEXT,
                breakout_prob     REAL,
                breakout_prob_band REAL,
                pattern_detected  TEXT,
                catalyst_flag     TEXT,
                catalyst_score    REAL,
                short_pct         REAL,
                float_shares      REAL,
                market_cap        REAL,
                target_price      REAL,
                stop_loss         REAL,
                est_move_pct_low  REAL,
                est_move_pct_high REAL,
                est_duration_min  INTEGER,
                est_duration_max  INTEGER,
                peak_day          INTEGER,
                urgency           TEXT,
                trade_type        TEXT,
                reward_risk       REAL,
                outcome_price     REAL,
                outcome_date      DATE,
                outcome_pct       REAL,
                peak_price        REAL,
                peak_pct          REAL,
                trough_price      REAL,
                trough_pct        REAL,
                hit_target        INTEGER,
                hit_stop          INTEGER,
                result            TEXT DEFAULT 'PENDING',
                result_reason     TEXT,
                actual_duration   INTEGER,
                duration_accurate INTEGER,
                notes             TEXT
            );
            CREATE TABLE IF NOT EXISTS portfolio (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id           INTEGER,
                ticker            TEXT,
                position_type     TEXT DEFAULT 'VIRTUAL',
                shares            REAL,
                entry_price       REAL,
                entry_date        DATE,
                status            TEXT DEFAULT 'OPEN',
                current_price     REAL,
                current_pct       REAL,
                current_value     REAL,
                exit_price        REAL,
                exit_date         DATE,
                exit_reason       TEXT,
                realized_pct      REAL,
                realized_pnl      REAL,
                position_size_pct REAL,
                buy_fee           REAL,
                sell_fee          REAL,
                total_fees        REAL,
                fee_drag_pct      REAL,
                gross_pnl         REAL,
                net_pnl           REAL,
                net_pnl_pct       REAL,
                fee_adjusted_rr   REAL,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            );
            CREATE TABLE IF NOT EXISTS scan_runs (
                scan_id          TEXT PRIMARY KEY,
                timestamp        DATETIME,
                universe_size    INTEGER,
                patterns_found   INTEGER,
                calls_made       INTEGER,
                spy_price        REAL,
                spy_vs_sma50     REAL,
                market_condition TEXT,
                notes            TEXT
            );
            CREATE TABLE IF NOT EXISTS performance_stats (
                stat_date     DATE PRIMARY KEY,
                total_calls   INTEGER,
                wins          INTEGER,
                losses        INTEGER,
                breakeven     INTEGER,
                pending       INTEGER,
                win_rate      REAL,
                avg_win_pct   REAL,
                avg_loss_pct  REAL,
                avg_hold_days REAL,
                profit_factor REAL,
                expectancy    REAL,
                best_call     TEXT,
                worst_call    TEXT,
                best_pattern  TEXT,
                best_catalyst TEXT
            );
        """)
        self.conn.commit()

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_scan_results(self, results: list, meta: dict) -> int:
        scan_id = meta.get("scan_id", str(_uuid.uuid4())[:8])
        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today   = datetime.now().strftime("%Y-%m-%d")

        qualifying = [r for r in results
                      if r.get("explosive_score", 0) >= self.MIN_SCORE
                      or r.get("probability", 0) >= self.MIN_PROB]
        added = 0
        for r in qualifying:
            tk = r["ticker"]
            if self.conn.execute(
                    "SELECT id FROM calls WHERE ticker=? AND entry_date=?",
                    (tk, today)).fetchone():
                continue
            t = r.get("timing", {}) or {}
            self.conn.execute("""
                INSERT INTO calls (
                    scan_id, scan_timestamp, ticker, entry_price, entry_date,
                    explosive_score, explosive_grade, breakout_prob, breakout_prob_band,
                    pattern_detected, catalyst_flag, catalyst_score,
                    short_pct, float_shares, market_cap,
                    target_price, stop_loss, est_move_pct_low, est_move_pct_high,
                    est_duration_min, est_duration_max, peak_day, urgency, trade_type,
                    reward_risk, result
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                scan_id, ts, tk, r.get("price"), today,
                r.get("explosive_score", 0), r.get("explosive_grade", ""),
                r.get("probability", 0),     r.get("conf_band", 0),
                r.get("pattern", "No Pattern"),
                r.get("top_flag", ""),       r.get("catalyst_score", 0),
                r.get("expl_short_pct", 0),  r.get("expl_float", 0),
                r.get("market_cap", 0),
                r.get("tgt_price"),          r.get("stop_price"),
                r.get("move_low"),           r.get("move_high"),
                t.get("min_days"),           t.get("max_days"),
                t.get("peak_day"),           t.get("urgency_key", ""),
                t.get("trade_type", ""),     r.get("rr", 0),
                "PENDING",
            ))
            cid = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            ep  = r.get("price") or 1
            self.conn.execute("""
                INSERT INTO portfolio
                (call_id, ticker, position_type, shares, entry_price, entry_date,
                 status, current_price, current_pct, current_value)
                VALUES (?,?,'VIRTUAL',?,?,?,'OPEN',?,0.0,?)
            """, (cid, tk, _POS_SIZE / ep, ep, today, ep, _POS_SIZE))
            added += 1

        # Log scan run metadata
        spy_df = meta.get("spy")
        spy_px = spy_vs = 0.0
        mkt    = "NEUTRAL"
        if spy_df is not None and not spy_df.empty:
            sp     = spy_df["Close"].values
            spy_px = float(sp[-1])
            if len(sp) >= 50:
                sma50  = float(sp[-50:].mean())
                spy_vs = (spy_px - sma50) / sma50
                mkt    = "BULL" if spy_px > sma50 else "BEAR"
        n_pats = sum(1 for r in results
                     if r.get("pattern", "No Pattern") != "No Pattern")
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO scan_runs
                (scan_id, timestamp, universe_size, patterns_found, calls_made,
                 spy_price, spy_vs_sma50, market_condition)
                VALUES (?,?,?,?,?,?,?,?)
            """, (scan_id, ts, meta.get("universe_size", len(results)),
                  n_pats, added, spy_px, spy_vs, mkt))
        except Exception:
            pass
        self.conn.commit()
        return added

    # ── Outcome updater ───────────────────────────────────────────────────────

    def update_calls(self) -> int:
        pending = self.conn.execute("""
            SELECT id, ticker, entry_price, entry_date,
                   target_price, stop_loss, est_duration_min, est_duration_max,
                   pattern_detected
            FROM calls WHERE result = 'PENDING'
        """).fetchall()
        if not pending:
            return 0

        today   = datetime.now().date()
        updated = 0

        for row in pending:
            cid     = row["id"]
            tk      = row["ticker"]
            ep      = row["entry_price"] or 0
            entry_s = row["entry_date"]
            target  = row["target_price"]
            stop    = row["stop_loss"]
            dur_min = row["est_duration_min"] or 0
            dur_max = row["est_duration_max"] or 0
            pattern = (row["pattern_detected"] if "pattern_detected" in row.keys() else "") or ""

            # ── Predictor fallback when est_duration_max is unset ────────────
            # Many older calls were saved without a duration estimate, so the
            # bot couldn't time-stop them.  Use the predictor as a fallback.
            if dur_max <= 0 and ep > 0 and target:
                try:
                    _pred = predict_duration_to_target(tk, ep, target, pattern=pattern)
                    if _pred and _pred.get("max_days_est"):
                        # 2× grace period — only time-stop if we're well past expected
                        dur_max = int(_pred["max_days_est"]) * 2
                except Exception:
                    pass
            if dur_max <= 0:
                dur_max = 20    # absolute fallback

            try:
                if isinstance(entry_s, str):
                    edate = datetime.strptime(entry_s[:10], "%Y-%m-%d").date()
                elif hasattr(entry_s, "hour"):
                    edate = entry_s.date()
                else:
                    edate = entry_s
            except Exception:
                continue
            held = (today - edate).days
            if held < 0:
                continue

            try:
                hist = yf.download(
                    tk, start=entry_s,
                    end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=True)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                if hist.empty:
                    continue
            except Exception:
                continue

            C = hist["Close"].values if "Close" in hist.columns else None
            H = hist["High"].values  if "High"  in hist.columns else None
            L = hist["Low"].values   if "Low"   in hist.columns else None
            if C is None or len(C) == 0:
                continue

            cur_px  = float(C[-1])
            peak_px = float(H.max()) if H is not None else cur_px
            trgh_px = float(L.min()) if L is not None else cur_px
            out_pct = (cur_px  - ep) / ep if ep else 0
            pk_pct  = (peak_px - ep) / ep if ep else 0
            tr_pct  = (trgh_px - ep) / ep if ep else 0

            hit_tgt = hit_stp = 0
            stp_first = False
            if H is not None and L is not None:
                for i in range(len(C)):
                    if stop   and float(L[i]) <= stop   and not stp_first:
                        hit_stp = 1; stp_first = True; break
                    if target and float(H[i]) >= target:
                        hit_tgt = 1; break
            else:
                if target and peak_px >= target: hit_tgt = 1
                if stop   and trgh_px <= stop:   hit_stp = 1

            expired = held > dur_max
            if stp_first:
                result = "LOSS";      reason = f"Stop hit on Day {held}"
            elif hit_tgt:
                result = "WIN";       reason = f"Target hit on Day {held}"
            elif expired:
                if   out_pct >  0.02: result = "WIN";       reason = f"Expired +{out_pct:.1%}"
                elif out_pct < -0.02: result = "LOSS";      reason = f"Expired {out_pct:.1%}"
                else:                 result = "BREAKEVEN";  reason = f"Expired flat ({out_pct:.1%})"
            else:
                result = "PENDING";   reason = None

            dur_acc = 1 if (dur_min and dur_min <= held <= dur_max) else 0

            self.conn.execute("""
                UPDATE calls SET
                    outcome_price=?, outcome_date=?, outcome_pct=?,
                    peak_price=?, peak_pct=?, trough_price=?, trough_pct=?,
                    hit_target=?, hit_stop=?, result=?, result_reason=?,
                    actual_duration=?, duration_accurate=?
                WHERE id=?
            """, (cur_px, today.strftime("%Y-%m-%d"), out_pct,
                  peak_px, pk_pct, trgh_px, tr_pct,
                  hit_tgt, hit_stp, result, reason, held, dur_acc, cid))

            if result != "PENDING":
                xr   = ("TARGET_HIT" if hit_tgt and not stp_first
                         else "STOP_HIT" if stp_first else "EXPIRED")
                rpnl = out_pct * _POS_SIZE
                self.conn.execute("""
                    UPDATE portfolio SET
                        status='CLOSED', current_price=?, current_pct=?,
                        current_value=?, exit_price=?, exit_date=?,
                        exit_reason=?, realized_pct=?, realized_pnl=?
                    WHERE call_id=? AND status='OPEN'
                """, (cur_px, out_pct * 100, _POS_SIZE * (1 + out_pct),
                      cur_px, today.strftime("%Y-%m-%d"), xr,
                      out_pct * 100, rpnl, cid))
            else:
                self.conn.execute("""
                    UPDATE portfolio SET current_price=?, current_pct=?, current_value=?
                    WHERE call_id=? AND status='OPEN'
                """, (cur_px, out_pct * 100, _POS_SIZE * (1 + out_pct), cid))
            updated += 1

        self.conn.commit()
        self._refresh_stats()
        return updated

    def _refresh_stats(self):
        rows = self.conn.execute("""
            SELECT ticker, result, outcome_pct, pattern_detected,
                   catalyst_flag, actual_duration
            FROM calls WHERE result != 'PENDING'
        """).fetchall()
        if not rows:
            return

        wins   = [r for r in rows if r["result"] == "WIN"]
        losses = [r for r in rows if r["result"] == "LOSS"]
        be     = [r for r in rows if r["result"] == "BREAKEVEN"]
        pend   = self.conn.execute(
            "SELECT COUNT(*) FROM calls WHERE result='PENDING'").fetchone()[0]

        n_wins = len(wins); n_loss = len(losses)
        res    = n_wins + n_loss
        wr     = n_wins / res if res > 0 else 0
        aw     = float(np.mean([r["outcome_pct"] for r in wins]))   * 100 if wins   else 0
        al     = float(np.mean([r["outcome_pct"] for r in losses])) * 100 if losses else 0
        ws     = sum(r["outcome_pct"] or 0 for r in wins)
        ls     = abs(sum(r["outcome_pct"] or 0 for r in losses))
        pf     = ws / ls if ls > 0 else (999.0 if ws > 0 else 0.0)
        holds  = [r["actual_duration"] for r in rows if r["actual_duration"]]
        ah     = float(np.mean(holds)) if holds else 0

        all_s     = sorted(rows, key=lambda r: r["outcome_pct"] or 0, reverse=True)
        best_call  = all_s[0]["ticker"]  if all_s else ""
        worst_call = all_s[-1]["ticker"] if all_s else ""

        pd_  = defaultdict(lambda: [0, 0])
        cd_  = defaultdict(lambda: [0, 0])
        for r in rows:
            p = r["pattern_detected"] or "Unknown"
            c = (r["catalyst_flag"] or "None")[:20]
            pd_[p][1] += 1; cd_[c][1] += 1
            if r["result"] == "WIN":
                pd_[p][0] += 1; cd_[c][0] += 1

        def _best(d):
            return max(d.keys(),
                       key=lambda k: d[k][0] / d[k][1] if d[k][1] >= 3 else 0,
                       default="")

        today = datetime.now().strftime("%Y-%m-%d")
        self.conn.execute("""
            INSERT OR REPLACE INTO performance_stats
            (stat_date, total_calls, wins, losses, breakeven, pending,
             win_rate, avg_win_pct, avg_loss_pct, avg_hold_days,
             profit_factor, expectancy, best_call, worst_call,
             best_pattern, best_catalyst)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (today, len(rows), n_wins, n_loss, len(be), pend,
              wr * 100, aw, al, ah, pf,
              wr * aw + (1 - wr) * al,
              best_call, worst_call, _best(pd_), _best(cd_)))
        self.conn.commit()

    # ── Manual operations ─────────────────────────────────────────────────────

    def add_manual_call(self, ticker: str):
        print(f"\nAdding manual call for {ticker.upper()}")
        try:
            ep  = float(input("  Entry price   : $").strip())
            tgt = float(input("  Target price  : $").strip())
            stp = float(input("  Stop loss     : $").strip())
            nts = input("  Notes (optional): ").strip()
        except (ValueError, KeyboardInterrupt):
            print("  Cancelled."); return
        today = datetime.now().strftime("%Y-%m-%d")
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute("""
            INSERT INTO calls
            (scan_id, scan_timestamp, ticker, entry_price, entry_date,
             target_price, stop_loss, result, notes)
            VALUES ('MANUAL',?,?,?,?,?,?,'PENDING',?)
        """, (ts, ticker.upper(), ep, today, tgt, stp, nts))
        cid = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute("""
            INSERT INTO portfolio
            (call_id, ticker, position_type, shares, entry_price, entry_date,
             status, current_price, current_pct, current_value)
            VALUES (?,?,'REAL',?,?,?,'OPEN',?,0.0,?)
        """, (cid, ticker.upper(), _POS_SIZE / ep if ep > 0 else 0,
              ep, today, ep, _POS_SIZE))
        self.conn.commit()
        print(f"  ✅ Call #{cid}: {ticker.upper()} @ ${ep:.2f}  "
              f"Target ${tgt:.2f}  Stop ${stp:.2f}")

    def close_call(self, call_id: int):
        row = self.conn.execute(
            "SELECT ticker, entry_price FROM calls WHERE id=?",
            (call_id,)).fetchone()
        if not row:
            print(f"  Call #{call_id} not found."); return
        tk, ep = row["ticker"], row["entry_price"] or 0
        try:
            xp  = float(input(f"  Exit price for {tk} (entry ${ep:.2f}): $").strip())
            why = input("  Reason (TARGET_HIT/STOP_HIT/MANUAL): ").strip().upper() or "MANUAL"
        except (ValueError, KeyboardInterrupt):
            print("  Cancelled."); return
        pct   = (xp - ep) / ep if ep else 0
        rpnl  = pct * _POS_SIZE
        today = datetime.now().strftime("%Y-%m-%d")
        res   = "WIN" if pct > 0.02 else ("LOSS" if pct < -0.02 else "BREAKEVEN")
        self.conn.execute("""
            UPDATE calls SET result=?, result_reason=?,
                outcome_price=?, outcome_date=?, outcome_pct=? WHERE id=?
        """, (res, why, xp, today, pct, call_id))
        self.conn.execute("""
            UPDATE portfolio SET
                status='CLOSED', exit_price=?, exit_date=?, exit_reason=?,
                realized_pct=?, realized_pnl=?, current_price=?
            WHERE call_id=? AND status='OPEN'
        """, (xp, today, why, pct * 100, rpnl, xp, call_id))
        self.conn.commit()
        self._refresh_stats()
        sign = "+" if rpnl >= 0 else "-"
        print(f"  ✅ #{call_id} {tk}: ${xp:.2f}  {pct:+.1%}  ({sign}${abs(rpnl):.0f})")

    def export_history_csv(self):
        df = pd.read_sql_query(
            "SELECT * FROM calls ORDER BY scan_timestamp DESC", self.conn)
        fn = f"call_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(fn, index=False)
        print(f"  ✅ Exported {len(df)} calls → {fn}")

    def reset_history(self):
        resp = input(
            "  ⚠️  Delete ALL call history? Type YES to confirm: ").strip()
        if resp != "YES":
            print("  Cancelled."); return
        self.conn.executescript("""
            DELETE FROM calls; DELETE FROM portfolio;
            DELETE FROM scan_runs; DELETE FROM performance_stats;
        """)
        self.conn.commit()
        print("  ✅ All history deleted.")


class PortfolioTracker:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_open_positions(self) -> list:
        rows = self.conn.execute("""
            SELECT p.id, p.call_id, p.ticker, p.position_type, p.shares,
                   p.entry_price, p.entry_date, p.current_price, p.current_pct,
                   p.current_value, c.target_price, c.stop_loss,
                   c.est_duration_max, c.pattern_detected, c.catalyst_flag,
                   c.explosive_grade
            FROM portfolio p
            JOIN calls c ON p.call_id = c.id
            WHERE p.status = 'OPEN'
            ORDER BY p.entry_date ASC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_closed_positions(self, days_back: int = 90) -> list:
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        rows   = self.conn.execute("""
            SELECT p.*, c.pattern_detected, c.catalyst_flag,
                   c.explosive_grade, c.explosive_score, c.breakout_prob
            FROM portfolio p
            JOIN calls c ON p.call_id = c.id
            WHERE p.status = 'CLOSED' AND p.exit_date >= ?
            ORDER BY p.realized_pct DESC
        """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]

    def get_portfolio_summary(self) -> dict:
        open_p   = self.get_open_positions()
        unreal   = sum((p.get("current_pct") or 0) / 100 * _POS_SIZE for p in open_p)
        real_pnl = self.conn.execute(
            "SELECT COALESCE(SUM(realized_pnl),0) FROM portfolio WHERE status='CLOSED'"
        ).fetchone()[0]
        top_w  = max(open_p, key=lambda p: p.get("current_pct") or 0, default=None)
        top_l  = min(open_p, key=lambda p: p.get("current_pct") or 0, default=None)
        first  = self.conn.execute(
            "SELECT MIN(entry_date) FROM calls").fetchone()[0] or "—"
        return {
            "n_open": len(open_p), "capital": len(open_p) * _POS_SIZE,
            "unrealized_pnl": unreal, "realized_pnl": float(real_pnl or 0),
            "top_winner": top_w, "top_loser": top_l, "first_call_date": first,
        }


class PerformanceAnalyzer:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _resolved(self, where_extra: str = "", params: list = None) -> list:
        q = ("SELECT * FROM calls WHERE result IN ('WIN','LOSS','BREAKEVEN')"
             + (" AND " + where_extra if where_extra else ""))
        return [dict(r) for r in self.conn.execute(q, params or []).fetchall()]

    def summary(self) -> dict:
        rows   = self._resolved()
        wins   = [r for r in rows if r["result"] == "WIN"]
        losses = [r for r in rows if r["result"] == "LOSS"]
        res    = len(wins) + len(losses)
        wr     = len(wins) / res if res > 0 else 0
        aw     = float(np.mean([r["outcome_pct"] for r in wins]))   * 100 if wins   else 0
        al     = float(np.mean([r["outcome_pct"] for r in losses])) * 100 if losses else 0
        pend   = self.conn.execute(
            "SELECT COUNT(*) FROM calls WHERE result='PENDING'").fetchone()[0]
        ws = sum(r["outcome_pct"] or 0 for r in wins)
        ls = abs(sum(r["outcome_pct"] or 0 for r in losses))
        pf = ws / ls if ls > 0 else (999.0 if ws > 0 else 0.0)
        exp = wr * aw + (1 - wr) * al
        return {"total": len(rows), "wins": len(wins), "losses": len(losses),
                "breakeven": len(rows) - len(wins) - len(losses), "pending": pend,
                "win_rate": wr * 100, "avg_win": aw, "avg_loss": al,
                "profit_factor": pf, "expectancy": exp}

    def pattern_ranking(self) -> list:
        rows = self._resolved()
        data = defaultdict(lambda: {"w": 0, "n": 0, "wps": [], "lps": []})
        for r in rows:
            p = r["pattern_detected"] or "Unknown"
            data[p]["n"] += 1
            if r["result"] == "WIN":
                data[p]["w"] += 1; data[p]["wps"].append(r["outcome_pct"] or 0)
            elif r["result"] == "LOSS":
                data[p]["lps"].append(r["outcome_pct"] or 0)
        out = []
        for pat, d in data.items():
            wr = d["w"] / d["n"] if d["n"] > 0 else 0
            aw = float(np.mean(d["wps"])) * 100 if d["wps"] else 0
            al = float(np.mean(d["lps"])) * 100 if d["lps"] else 0
            ws = sum(d["wps"]); ls = abs(sum(d["lps"]))
            out.append({"pattern": pat, "total": d["n"], "wins": d["w"],
                        "win_rate": wr * 100, "avg_win": aw, "avg_loss": al,
                        "pf": ws / ls if ls > 0 else 999.0})
        return sorted(out, key=lambda x: (x["win_rate"], x["total"]), reverse=True)

    def catalyst_ranking(self) -> list:
        rows = self._resolved()
        data = defaultdict(lambda: {"w": 0, "n": 0, "wps": [], "lps": []})
        for r in rows:
            c = (r["catalyst_flag"] or "None")[:25]
            data[c]["n"] += 1
            if r["result"] == "WIN":
                data[c]["w"] += 1; data[c]["wps"].append(r["outcome_pct"] or 0)
            elif r["result"] == "LOSS":
                data[c]["lps"].append(r["outcome_pct"] or 0)
        out = []
        for cat, d in data.items():
            wr = d["w"] / d["n"] if d["n"] > 0 else 0
            aw = float(np.mean(d["wps"])) * 100 if d["wps"] else 0
            al = float(np.mean(d["lps"])) * 100 if d["lps"] else 0
            out.append({"catalyst": cat, "total": d["n"], "wins": d["w"],
                        "win_rate": wr * 100, "avg_win": aw, "avg_loss": al})
        return sorted(out, key=lambda x: (x["win_rate"], x["total"]), reverse=True)

    def score_calibration(self) -> list:
        out = []
        for lo, hi in [(60, 70), (70, 80), (80, 90), (90, 101)]:
            rows = self.conn.execute("""
                SELECT result FROM calls
                WHERE breakout_prob >= ? AND breakout_prob < ?
                AND result IN ('WIN','LOSS','BREAKEVEN')
            """, (lo, hi)).fetchall()
            if not rows:
                continue
            wins = sum(1 for r in rows if r["result"] == "WIN")
            n    = len(rows)
            awr  = wins / n * 100
            exp  = (lo + hi) / 2
            out.append({"range": f"{lo}–{hi}%", "n": n,
                        "expected": exp, "actual": awr, "delta": awr - exp})
        return out

    def get_streak(self) -> dict:
        rows = self.conn.execute("""
            SELECT result FROM calls WHERE result IN ('WIN','LOSS')
            ORDER BY scan_timestamp ASC
        """).fetchall()
        if not rows:
            return {"current": 0, "direction": "—", "longest_win": 0, "longest_loss": 0}
        res = [r["result"] for r in rows]
        cur = 1; cur_dir = res[-1]
        for i in range(len(res) - 2, -1, -1):
            if res[i] == cur_dir: cur += 1
            else: break
        lw = ll = cs = 0; cr = None
        for r in res:
            cs = cs + 1 if r == cr else 1; cr = r
            if r == "WIN": lw = max(lw, cs)
            else:          ll = max(ll, cs)
        return {"current": cur, "direction": cur_dir,
                "longest_win": lw, "longest_loss": ll}

    def benchmark_vs_spy(self) -> dict:
        rows = self.conn.execute("""
            SELECT entry_date, outcome_date, outcome_pct FROM calls
            WHERE result IN ('WIN','LOSS','BREAKEVEN')
            AND outcome_date IS NOT NULL AND outcome_pct IS NOT NULL
        """).fetchall()
        if not rows:
            return {"avg_alpha": 0, "n": 0}
        min_date = min(r["entry_date"] for r in rows)
        try:
            spy = yf.download("SPY", start=min_date, progress=False, auto_adjust=True)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
        except Exception:
            return {"avg_alpha": 0, "n": 0}
        alphas = []
        for r in rows:
            try:
                s = spy.loc[r["entry_date"]:r["outcome_date"]]["Close"]
                if len(s) < 2: continue
                sr = (float(s.iloc[-1]) - float(s.iloc[0])) / float(s.iloc[0])
                alphas.append((r["outcome_pct"] or 0) - sr)
            except Exception:
                continue
        if not alphas:
            return {"avg_alpha": 0, "n": 0}
        return {"avg_alpha": float(np.mean(alphas)) * 100, "n": len(alphas)}

    def duration_accuracy(self) -> dict:
        rows = self.conn.execute("""
            SELECT est_duration_min, est_duration_max, actual_duration, duration_accurate
            FROM calls WHERE result != 'PENDING' AND actual_duration IS NOT NULL
        """).fetchall()
        if not rows:
            return {"pct_accurate": 0, "avg_est": 0, "avg_actual": 0, "n": 0}
        n_acc   = sum(1 for r in rows if r["duration_accurate"])
        est_avs = [((r["est_duration_min"] or 0) + (r["est_duration_max"] or 0)) / 2
                   for r in rows]
        acts    = [r["actual_duration"] for r in rows]
        return {"pct_accurate": n_acc / len(rows) * 100,
                "avg_est": float(np.mean(est_avs)) if est_avs else 0,
                "avg_actual": float(np.mean(acts)) if acts else 0, "n": len(rows)}

    def suggest_improvements(self, min_n: int = 5) -> list:
        tips = []
        for p in self.pattern_ranking():
            if p["total"] >= min_n and p["win_rate"] < 50:
                tips.append(
                    f"Pattern '{p['pattern']}' win rate {p['win_rate']:.0f}% "
                    f"({p['total']} calls) — below 50%. Recommend raising "
                    f"min explosive score to 80+ for this pattern only.")
        rows  = self._resolved()
        lo_sc = [r for r in rows if 50 <= (r.get("explosive_score") or 0) < 65]
        hi_sc = [r for r in rows if (r.get("explosive_score") or 0) >= 70]
        if len(lo_sc) >= min_n and len(hi_sc) >= min_n:
            lwr = sum(1 for r in lo_sc if r["result"] == "WIN") / len(lo_sc) * 100
            hwr = sum(1 for r in hi_sc if r["result"] == "WIN") / len(hi_sc) * 100
            if hwr > lwr + 15:
                tips.append(
                    f"Explosive score 50–65 wins {lwr:.0f}%, score 70+ wins {hwr:.0f}%. "
                    f"Recommend raising --min-explosive to ~68.")
        dur = self.duration_accuracy()
        if dur["n"] >= min_n and dur["pct_accurate"] < 60:
            tips.append(
                f"Duration accuracy {dur['pct_accurate']:.0f}%. "
                f"Avg actual hold {dur['avg_actual']:.1f}d vs est {dur['avg_est']:.1f}d. "
                f"Consider hard exit at Day {int(dur['avg_actual']) + 2}.")
        for c in self.catalyst_ranking():
            if c["total"] >= min_n and c["win_rate"] < 45:
                tips.append(
                    f"Catalyst '{c['catalyst']}' wins {c['win_rate']:.0f}% "
                    f"({c['total']} calls) — recommend reducing catalyst score weight.")
        return tips


# ── Dashboard & display helpers ───────────────────────────────────────────────

def _print_dashboard(tracker: PortfolioTracker,
                     analyzer: PerformanceAnalyzer,
                     logger: CallLogger):
    s    = analyzer.summary()
    port = tracker.get_portfolio_summary()
    stk  = analyzer.get_streak()
    spy  = analyzer.benchmark_vs_spy()
    dur  = analyzer.duration_accuracy()
    pats = analyzer.pattern_ranking()
    cats = analyzer.catalyst_ranking()
    W    = 66

    if s["total"] == 0:
        print("\n  📊 No call history yet — scanner will begin tracking from this run.\n")
        return

    def _ln(txt=""):
        inner = ("  " + txt).ljust(W - 2)
        try:    print("║" + inner + "║")
        except: print("|" + inner.encode("ascii", "replace").decode() + "|")

    def _sep(c="═"):
        try:    print("╠" + c * (W - 2) + "╣")
        except: print("+" + "-" * (W - 2) + "+")

    def _top():
        try:    print("╔" + "═" * (W - 2) + "╗")
        except: print("+" + "=" * (W - 2) + "+")

    def _bot():
        try:    print("╚" + "═" * (W - 2) + "╝")
        except: print("+" + "=" * (W - 2) + "+")

    _top()
    _ln(f"{'📊 SCANNER PERFORMANCE DASHBOARD':^{W - 4}}")
    _sep()

    since = port.get("first_call_date", "—")
    try:    since = datetime.strptime(since, "%Y-%m-%d").strftime("%b %d %Y")
    except: pass
    _ln(f"Total Calls Made    : {s['total']:<8}   Since : {since}")
    _ln(f"Win / Loss / BE     : {s['wins']} / {s['losses']} / {s['breakeven']}"
        f"   Pending: {s['pending']}")
    _ln(f"Overall Win Rate    : {s['win_rate']:.1f}%     "
        f"Profit Factor : {s['profit_factor']:.1f}")
    _ln(f"Avg Win             : +{s['avg_win']:.1f}%     "
        f"Avg Loss      : {s['avg_loss']:.1f}%")
    exp_str = f"{s['expectancy']:+.1f}%"
    alp_str = f"{spy['avg_alpha']:+.1f}%" if spy.get("n") else "N/A"
    _ln(f"Expectancy/Trade    : {exp_str:<12}  vs SPY Alpha  : {alp_str}")
    icon = "🟢" if stk["direction"] == "WIN" else "🔴"
    _ln(f"Current Streak      : {icon} {stk['current']} "
        f"{'win' if stk['direction'] == 'WIN' else 'loss'}s in a row")

    _sep()
    _ln(f"💼 VIRTUAL PORTFOLIO (${_POS_SIZE:,.0f} per position)")
    cap = port["capital"]
    ur  = port["unrealized_pnl"]
    rp  = port["realized_pnl"]
    _ln(f"Open Positions      : {port['n_open']:<8}   Capital Deployed: ${cap:,.0f}")
    if cap:
        ur_s = "+" if ur >= 0 else ""
        _ln(f"Unrealized P&L      : {ur_s}${abs(ur):,.0f}  ({ur / cap * 100:+.1f}%)")
    base   = max((s["total"] + s["pending"]) * _POS_SIZE, 1)
    rp_pct = rp / base * 100
    rp_s   = "+" if rp >= 0 else "-"
    _ln(f"Realized P&L (all)  : {rp_s}${abs(rp):,.0f} ({rp_pct:+.1f}% on base capital)")

    _sep()
    bp = pats[0]  if pats           else None
    wp = pats[-1] if len(pats) > 1  else None
    bc = cats[0]  if cats           else None
    wc = cats[-1] if len(cats) > 1  else None
    if bp: _ln(f"🏆 BEST PATTERN     : {bp['pattern'][:14]:<14}  "
               f"Win Rate: {bp['win_rate']:.0f}%  ({bp['total']} calls)")
    if bc: _ln(f"🏆 BEST CATALYST    : {bc['catalyst'][:14]:<14}  "
               f"Win Rate: {bc['win_rate']:.0f}%  ({bc['total']} calls)")
    if wp: _ln(f"⚠  WORST PATTERN   : {wp['pattern'][:14]:<14}  "
               f"Win Rate: {wp['win_rate']:.0f}%  ({wp['total']} calls)")
    if wc: _ln(f"⚠  WORST CATALYST  : {wc['catalyst'][:14]:<14}  "
               f"Win Rate: {wc['win_rate']:.0f}%  ({wc['total']} calls)")

    if dur.get("n"):
        _sep()
        _ln(f"📏 DURATION ACCURACY: {dur['pct_accurate']:.0f}% of moves hit within est. window")
        _ln(f"Avg Est Duration    : {dur['avg_est']:.1f} days   "
            f"Avg Actual: {dur['avg_actual']:.1f} days")

    _bot()
    print()


def _print_open_positions(tracker: PortfolioTracker):
    pos   = tracker.get_open_positions()
    today = datetime.now().date()
    if not pos:
        print("  📂 No open positions.\n"); return
    hdrs  = ["#", "Ticker", "Type", "Entry", "Date", "Current", "P&L%", "Day#", "Status"]
    rows  = []
    warns = []
    for i, p in enumerate(pos, 1):
        ep   = p.get("entry_price") or 0
        cur  = p.get("current_price") or ep
        pct  = p.get("current_pct") or 0
        tgt  = p.get("target_price")
        stp  = p.get("stop_loss")
        try:
            edate = datetime.strptime(p["entry_date"], "%Y-%m-%d").date()
            day_n = (today - edate).days + 1
        except Exception:
            day_n = 0
        if tgt and cur >= tgt * 0.95:
            status = "✅ NEAR TGT"
        elif stp and cur <= stp * 1.05:
            status = "🔴 NEAR STP"
            warns.append(f"  ⚠️  {p['ticker']} approaching stop loss (${stp:.2f}).")
        elif pct > 20:
            status = "🟢 WIN"
            warns.append(f"  ✅ {p['ticker']} +{pct:.1f}% — consider partial profits.")
        elif pct < -8:  status = "🔴 LOSS"
        elif pct > 0:   status = "🟡 UP"
        else:           status = "🔴 DOWN"
        rows.append([i, p["ticker"], (p.get("position_type") or "VIR")[:3],
                     f"${ep:.2f}", p["entry_date"][5:],
                     f"${cur:.2f}", f"{pct:+.1f}%", f"D{day_n}", status])
    print(f"\n  📂 OPEN POSITIONS ({len(pos)} active)")
    print("  " + "═" * 72)
    try:
        print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(tabulate(rows, headers=hdrs, tablefmt="grid"))
    for w in warns:
        print(w)
    print()


def _print_recent_history(logger: CallLogger, n: int = 20):
    rows = logger.conn.execute(f"""
        SELECT id, ticker, entry_price, outcome_price, outcome_pct,
               actual_duration, pattern_detected, catalyst_flag, result, entry_date
        FROM calls WHERE result IN ('WIN','LOSS','BREAKEVEN')
        ORDER BY scan_timestamp DESC LIMIT {int(n)}
    """).fetchall()
    if not rows:
        print("  📋 No closed calls yet.\n"); return
    hdrs = ["#", "ID", "Ticker", "Entry", "Exit", "P&L%", "Days", "Pattern", "Result"]
    tbl  = []
    for i, r in enumerate(rows, 1):
        icon = "✅" if r["result"] == "WIN" else ("❌" if r["result"] == "LOSS" else "➖")
        pct  = (r["outcome_pct"] or 0) * 100
        pat  = (r["pattern_detected"] or "—")[:12]
        cat  = (r["catalyst_flag"] or "")[:8]
        tbl.append([i, r["id"], r["ticker"],
                    f"${r['entry_price']:.2f}" if r["entry_price"] else "—",
                    f"${r['outcome_price']:.2f}" if r["outcome_price"] else "—",
                    f"{pct:+.1f}%", f"{r['actual_duration'] or '?'}d",
                    f"{pat}{'+'+ cat if cat else ''}"[:18],
                    f"{icon} {r['result']}"])
    print(f"\n  📋 RECENT CALL HISTORY (last {n} closed)")
    print("  " + "═" * 72)
    try:
        print(tabulate(tbl, headers=hdrs, tablefmt="rounded_outline"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(tabulate(tbl, headers=hdrs, tablefmt="grid"))
    print()


def _print_pattern_performance(analyzer: PerformanceAnalyzer):
    pats = analyzer.pattern_ranking()
    if not pats:
        print("  No pattern data yet.\n"); return
    hdrs = ["Pattern", "Calls", "Wins", "Win%", "Avg Win", "Avg Loss", "P.Factor"]
    rows = [[p["pattern"][:22], p["total"], p["wins"],
             f"{p['win_rate']:.0f}%", f"+{p['avg_win']:.1f}%",
             f"{p['avg_loss']:.1f}%", f"{min(p['pf'], 99.9):.1f}"]
            for p in pats]
    print("\n  📊 WIN RATE BY PATTERN (resolved calls only)")
    print("  " + "═" * 62)
    try:
        print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(tabulate(rows, headers=hdrs, tablefmt="grid"))
    print()


def _print_calibration(analyzer: PerformanceAnalyzer):
    cals = analyzer.score_calibration()
    for c in cals:
        if abs(c["delta"]) >= 10 and c["n"] >= 5:
            direction = "overconfident" if c["delta"] < 0 else "underconfident"
            print(f"\n  ⚠️  CALIBRATION ALERT ({c['range']} band, n={c['n']}):")
            print(f"     {c['range']} calls are winning {c['actual']:.0f}% "
                  f"(expected ~{c['expected']:.0f}%) — bot is {direction}.")
            if c["delta"] < -10:
                print(f"     Recommend: Raise prob threshold by ~{abs(c['delta']):.0f}% "
                      f"or reduce position size by 30% in this range.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Explosive Move & Breakout Scanner — trading_scanner.py",
        formatter_class=argparse.RawTextHelpFormatter)
    # Universe
    ap.add_argument("--universe",      type=str, default="all",
                    choices=["all","canadian","us",
                             "tsx","tsxv","cse","neo",
                             "nasdaq","nyse","cboe","otc",
                             "smallcap","russell2000","microcap","finviz"],
                    help="Universe to scan:\n"
                         "  all       — TSX + TSXV + CSE + NEO + NYSE + NASDAQ + CBOE + OTC (default)\n"
                         "  canadian  — TSX + TSXV + CSE + NEO/CBOE Canada only\n"
                         "  us        — NYSE + NASDAQ + CBOE BZX + OTC Select only\n"
                         "  tsx       — Toronto Stock Exchange (.TO)\n"
                         "  tsxv      — TSX Venture Exchange (.V)\n"
                         "  cse       — Canadian Securities Exchange (.CN)\n"
                         "  neo       — NEO / CBOE Canada (.NE)\n"
                         "  nasdaq    — NASDAQ-listed only\n"
                         "  nyse      — NYSE / NYSE Arca / NYSE American only\n"
                         "  cboe      — CBOE BZX only\n"
                         "  otc       — OTCQX + OTCQB (Select OTC) only\n"
                         "  smallcap  — legacy curated small-cap list\n"
                         "  russell2000 — Russell 2000 from Wikipedia\n"
                         "  finviz    — Finviz screener results\n"
                         "  microcap  — micro-cap Finviz screens only")
    ap.add_argument("--file",          type=str, default=None, metavar="PATH",
                    help="Load tickers from a .txt or .csv file (one per line or comma-sep)")
    ap.add_argument("--watchlist",     type=str, default=None,
                    help="Comma-separated tickers, e.g. AAPL,NVDA,MSFT")
    # Filters
    ap.add_argument("--pattern",       type=str, default=None,
                    help="Filter by pattern name (vcp, cup, wyckoff, htf, ...)")
    ap.add_argument("--min-prob",      type=int, default=0,
                    help="Minimum breakout probability %% (default: 0)")
    ap.add_argument("--min-explosive", type=int, default=0,
                    help="Minimum explosive score (default: 0)")
    ap.add_argument("--no-earnings",   action="store_true",
                    help="Exclude stocks with earnings within 7 days")
    ap.add_argument("--catalyst-only", action="store_true",
                    help="Only show stocks with a detected news catalyst")
    ap.add_argument("--biotech",       action="store_true",
                    help="Only biotech/pharma stocks")
    ap.add_argument("--squeeze",       action="store_true",
                    help="Only short squeeze candidates (short%% > 10%%)")
    ap.add_argument("--max-float",     type=int, default=None, metavar="M",
                    help="Max float in millions (e.g. --max-float 50)")
    ap.add_argument("--max-cap",       type=int, default=None, metavar="M",
                    help="Max market cap in millions (e.g. --max-cap 500)")
    ap.add_argument("--no-otc",        action="store_true",
                    help="Exclude OTC stocks")
    ap.add_argument("--min-rs",        type=int, default=0, metavar="N",
                    help="Min RS Rating 1-99 (e.g. --min-rs 80 = top 20%%)")
    ap.add_argument("--earnings-quality", action="store_true",
                    help="Only show stocks with accelerating or strong EPS (EQ score >= 65)")
    ap.add_argument("--premarket",     action="store_true",
                    help="Only show stocks with pre-market gap > 3%%")
    # Output
    ap.add_argument("--top",           type=int, default=15,
                    help="Rows per table (default: 15)")
    ap.add_argument("--export",        action="store_true", default=True,
                    help="Export CSV + TXT (default: True)")
    # Backtest / learning
    ap.add_argument("--backtest",      type=int, default=None, metavar="DAYS",
                    help="Backtest signals from last N trading days")
    ap.add_argument("--hold",          type=int, default=20, metavar="DAYS",
                    help="Hold period for backtest (default: 20)")
    ap.add_argument("--learn",         action="store_true",
                    help="Update weights after --backtest")
    ap.add_argument("--reset-weights", action="store_true",
                    help="Reset learned weights to defaults and exit")
    ap.add_argument("--weight-status", action="store_true",
                    help="Show current weight adjustments and exit")
    ap.add_argument("--debug",          action="store_true",
                    help="Print detailed errors when a stock fails processing")
    # Portfolio tracker flags
    ap.add_argument("--dashboard",      action="store_true",
                    help="Show performance dashboard and exit")
    ap.add_argument("--history",        type=int, default=None, metavar="N",
                    help="Show last N closed calls and exit")
    ap.add_argument("--open",           action="store_true",
                    help="Show open virtual positions and exit")
    ap.add_argument("--add-call",       type=str, default=None, metavar="TICKER",
                    help="Manually add a call for TICKER (interactive)")
    ap.add_argument("--close-call",     type=int, default=None, metavar="ID",
                    help="Manually close call by ID (interactive)")
    ap.add_argument("--update-call",    type=int, default=None, metavar="ID",
                    help="Force refresh outcome for a specific call ID")
    ap.add_argument("--stats",          action="store_true",
                    help="Show full performance stats summary and exit")
    ap.add_argument("--stats-pattern",  action="store_true",
                    help="Show win rate breakdown by pattern and exit")
    ap.add_argument("--stats-catalyst", action="store_true",
                    help="Show win rate breakdown by catalyst and exit")
    ap.add_argument("--export-history", action="store_true",
                    help="Export full call history to CSV and exit")
    ap.add_argument("--reset-history",  action="store_true",
                    help="Delete all call history (prompts for confirmation)")
    ap.add_argument("--no-dashboard",   action="store_true",
                    help="Skip dashboard display before scanning")
    ap.add_argument("--paper-only",     action="store_true",
                    help="Run scan but do NOT log results to history")
    ap.add_argument("--calibrate",      action="store_true",
                    help="Show probability calibration warnings and exit")
    ap.add_argument("--paper",          action="store_true",
                    help="Show paper trading portfolio summary and exit")
    ap.add_argument("--paper-reset",    action="store_true",
                    help="Reset paper portfolio back to $1,000 (asks confirmation)")
    ap.add_argument("--paper-budget",   type=float, default=None, metavar="N",
                    help="Set paper portfolio starting budget (default $1,000)")
    ap.add_argument("--fees",           action="store_true",
                    help="Show fees tracker summary and exit")
    ap.add_argument("--no-auto-trade",  action="store_true",
                    help="Scan without auto-opening paper trades")
    ap.add_argument("--rejected",       action="store_true",
                    help="Show full rejection log from last scan and exit")
    args = ap.parse_args()

    # ── Weight management shortcuts ───────────────────────────────────────────
    if args.reset_weights:
        WeightOptimizer().reset()
        return

    if args.weight_status:
        WeightOptimizer.status()
        return

    # ── Portfolio tracker setup ───────────────────────────────────────────────
    logger   = CallLogger()
    tracker  = PortfolioTracker(logger.conn)
    analyzer = PerformanceAnalyzer(logger.conn)

    # Standalone portfolio commands (no scan needed)
    if args.dashboard:
        _print_dashboard(tracker, analyzer, logger)
        return

    if args.history is not None:
        _print_recent_history(logger, n=args.history)
        return

    if args.open:
        _print_open_positions(tracker)
        return

    if args.stats:
        _print_dashboard(tracker, analyzer, logger)
        _print_pattern_performance(analyzer)
        cats = analyzer.catalyst_ranking()
        if cats:
            hdrs = ["Catalyst", "Calls", "Wins", "Win%", "Avg Win", "Avg Loss"]
            rows = [[c["catalyst"][:22], c["total"], c["wins"],
                     f"{c['win_rate']:.0f}%", f"+{c['avg_win']:.1f}%",
                     f"{c['avg_loss']:.1f}%"] for c in cats]
            print("\n  📊 WIN RATE BY CATALYST (resolved calls only)")
            print("  " + "═" * 62)
            try:
                print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                print(tabulate(rows, headers=hdrs, tablefmt="grid"))
            print()
        tips = analyzer.suggest_improvements()
        if tips:
            print("  💡 SELF-IMPROVEMENT SUGGESTIONS")
            print("  " + "─" * 62)
            for t in tips:
                print(f"    • {t}")
            print()
        return

    if args.stats_pattern:
        _print_pattern_performance(analyzer)
        return

    if args.stats_catalyst:
        cats = analyzer.catalyst_ranking()
        if cats:
            hdrs = ["Catalyst", "Calls", "Wins", "Win%", "Avg Win", "Avg Loss"]
            rows = [[c["catalyst"][:22], c["total"], c["wins"],
                     f"{c['win_rate']:.0f}%", f"+{c['avg_win']:.1f}%",
                     f"{c['avg_loss']:.1f}%"] for c in cats]
            print("\n  📊 WIN RATE BY CATALYST")
            print("  " + "═" * 62)
            try:
                print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                print(tabulate(rows, headers=hdrs, tablefmt="grid"))
            print()
        return

    if args.export_history:
        logger.export_history_csv()
        return

    if args.reset_history:
        logger.reset_history()
        return

    if args.calibrate:
        _print_calibration(analyzer)
        return

    if args.add_call:
        logger.add_manual_call(args.add_call)
        return

    if args.close_call is not None:
        logger.close_call(args.close_call)
        return

    if args.update_call is not None:
        cid = args.update_call
        row = logger.conn.execute(
            "SELECT result FROM calls WHERE id=?", (cid,)).fetchone()
        if not row:
            print(f"  Call #{cid} not found.")
        else:
            logger.conn.execute(
                "UPDATE calls SET result='PENDING' WHERE id=?", (cid,))
            logger.conn.commit()
            n = logger.update_calls()
            print(f"  ✅ Updated {n} call(s).")
        return

    scanner        = BreakoutScanner()
    scanner.args   = args
    scanner._debug = getattr(args, "debug", False)

    # ── Backtest mode ─────────────────────────────────────────────────────────
    if args.backtest:
        if args.watchlist:
            tickers = [t.strip().upper() for t in args.watchlist.split(",")]
            print(f"\nCustom watchlist: {tickers}")
        else:
            print("\nFetching S&P 500 ticker list from Wikipedia...")
            tickers = scanner._fetch_sp500()
            print(f"Loaded {len(tickers)} tickers")

        print("Fetching SPY market data...")
        scanner.spy = scanner._fetch_spy()

        bt = BacktestEngine(scanner, days=args.backtest, hold=args.hold)
        bt.run(tickers)
        bt.print_report()
        bt.to_csv()

        if args.learn:
            print("Updating weights from backtest results...")
            WeightOptimizer().update(bt.signals)
        else:
            print("  Tip: add --learn to update modifier weights from these results.\n")
        return

    # ── Normal scan mode ──────────────────────────────────────────────────────

    # Pre-scan: refresh pending outcomes, then show dashboard
    print("  [Portfolio tracker: refreshing pending outcomes...]")
    n_updated = logger.update_calls()
    if n_updated:
        print(f"  [Updated {n_updated} call outcome(s)]")

    if not args.no_dashboard:
        _print_dashboard(tracker, analyzer, logger)
        _print_open_positions(tracker)
        _print_calibration(analyzer)

    if LEARNED_WEIGHTS.get("samples", 0) > 0:
        ts = LEARNED_WEIGHTS.get("last_updated", "unknown")
        print(f"  [Using learned weights — {LEARNED_WEIGHTS['samples']} backtest samples, updated {ts}]")

    scanner.run()

    # Post-scan: log qualifying results
    if not args.paper_only and hasattr(scanner, "results") and scanner.results:
        import uuid as _uuid2
        scan_id = str(_uuid2.uuid4())[:8]
        spy_df  = getattr(scanner, "spy", None)
        meta    = {
            "scan_id":       scan_id,
            "spy":           spy_df,
            "universe_size": len(scanner.results),
        }
        n_logged = logger.log_scan_results(scanner.results, meta)
        if n_logged:
            print(f"\n  📝 Portfolio tracker: logged {n_logged} new call(s) to history.")
    elif args.paper_only:
        print("\n  [--paper-only: results not logged to history]")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS ENGINE — Black-Scholes · Chain Analyzer · Scanner · Strategy · Paper
# ══════════════════════════════════════════════════════════════════════════════

try:
    # IMPORTANT: do NOT import this as `_norm` — that name is already used at
    # module level (line 729) for the DataFrame OHLCV normalizer.  Shadowing it
    # would silently break the entire scanner.  Use a distinct name.
    from scipy.stats import norm as _norm_dist
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

_RISK_FREE      = 0.05        # assumed risk-free rate (5 %)
_OPTIONS_BUDGET = 500.0       # virtual paper capital


def _bs_greeks(S, K, T, r, sigma, opt_type="call"):
    """Black-Scholes price + Greeks. Returns {} on bad inputs or missing scipy."""
    if not _SCIPY_OK:
        return {}
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {}
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = _norm_dist.pdf(d1)
        if opt_type == "call":
            price = S * _norm_dist.cdf(d1) - K * np.exp(-r * T) * _norm_dist.cdf(d2)
            delta = _norm_dist.cdf(d1)
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * _norm_dist.cdf(d2)) / 365
        else:
            price = K * np.exp(-r * T) * _norm_dist.cdf(-d2) - S * _norm_dist.cdf(-d1)
            delta = _norm_dist.cdf(d1) - 1
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * _norm_dist.cdf(-d2)) / 365
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega  = S * pdf_d1 * np.sqrt(T) / 100
        return {
            "bs_price": round(float(price), 4),
            "delta":    round(float(delta), 3),
            "gamma":    round(float(gamma), 5),
            "theta":    round(float(theta), 4),
            "vega":     round(float(vega),  4),
        }
    except Exception:
        return {}


def _pop_from_bs(S, K_be, T, sigma, opt_type="call"):
    """
    Probability of Profit at expiry using Black-Scholes.
    K_be — break-even price (strike + premium for calls, strike − premium for puts).
    Returns float 0–100, or None on failure.
    """
    if not _SCIPY_OK:
        return None
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K_be <= 0:
            return None
        d2 = (np.log(S / K_be) + (_RISK_FREE - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if opt_type == "call":
            return round(float(_norm_dist.cdf(d2)) * 100, 1)
        else:
            return round(float(_norm_dist.cdf(-d2)) * 100, 1)
    except Exception:
        return None


# ── Options Chain Analyzer ────────────────────────────────────────────────────

class OptionsChainAnalyzer:
    """Fetch + enrich a full options chain for a single ticker."""

    def __init__(self, ticker):
        self.ticker = str(ticker).upper()
        self._tk = yf.Ticker(self.ticker)

    # ── public ───────────────────────────────────────────────────────────────

    def expirations(self):
        """Return list of available expiry date strings."""
        try:
            return list(self._tk.options)
        except Exception:
            return []

    def get_chain(self, expiry=None):
        """
        Return dict:
          calls, puts  — enriched DataFrames
          spot         — current price
          expiry       — date string used
          dte          — calendar days to expiry
        Returns {} on failure.
        """
        exps = self.expirations()
        if not exps:
            return {}
        if expiry is None or expiry not in exps:
            expiry = exps[0]
        try:
            chain = self._tk.option_chain(expiry)
        except Exception:
            return {}

        spot = self._get_spot()
        dte  = max(0, (datetime.strptime(expiry, "%Y-%m-%d").date()
                       - datetime.utcnow().date()).days)
        T = dte / 365.0

        calls = self._enrich(chain.calls.copy(), spot, T, "call")
        puts  = self._enrich(chain.puts.copy(),  spot, T, "put")

        return {"calls": calls, "puts": puts, "spot": spot,
                "expiry": expiry, "dte": dte}

    def iv_rank(self):
        """
        IV rank 0-100 using 30-day realised HV as IV proxy.
        Returns dict with iv_rank, current_hv, hv_52_hi, hv_52_lo.
        """
        try:
            hist = yf.download(self.ticker, period="1y", interval="1d",
                               progress=False, auto_adjust=True)
            if hist is None or hist.empty:
                return {}
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            closes    = hist["Close"].dropna()
            log_ret   = np.log(closes / closes.shift(1)).dropna()
            hv_series = log_ret.rolling(30).std() * np.sqrt(252) * 100
            hv_series = hv_series.dropna()
            if len(hv_series) < 30:
                return {}
            current  = float(hv_series.iloc[-1])
            hi       = float(hv_series.max())
            lo       = float(hv_series.min())
            rng      = hi - lo
            rank     = round((current - lo) / rng * 100, 1) if rng > 0 else 50.0
            return {"iv_rank": rank, "current_hv": round(current, 1),
                    "hv_52_hi": round(hi, 1), "hv_52_lo": round(lo, 1)}
        except Exception:
            return {}

    def max_pain(self, expiry=None):
        """Compute max pain strike. Returns float or None."""
        try:
            data = self.get_chain(expiry)
            if not data:
                return None
            calls   = data["calls"]
            puts    = data["puts"]
            strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            if not strikes:
                return None
            pain = []
            c_oi = dict(zip(calls["strike"], calls.get("openInterest", pd.Series([], dtype=float)).fillna(0)))
            p_oi = dict(zip(puts["strike"],  puts.get("openInterest",  pd.Series([], dtype=float)).fillna(0)))
            for S_t in strikes:
                c_loss = sum(max(0, S_t - k) * oi * 100 for k, oi in c_oi.items())
                p_loss = sum(max(0, k - S_t) * oi * 100 for k, oi in p_oi.items())
                pain.append(c_loss + p_loss)
            return float(strikes[pain.index(min(pain))])
        except Exception:
            return None

    def cp_ratio(self, expiry=None):
        """Call-to-Put volume ratio. Returns float or None."""
        try:
            data = self.get_chain(expiry)
            if not data:
                return None
            cv = data["calls"]["volume"].fillna(0).sum()
            pv = data["puts"]["volume"].fillna(0).sum()
            return round(float(cv) / float(pv), 2) if pv > 0 else None
        except Exception:
            return None

    # ── private ──────────────────────────────────────────────────────────────

    def _get_spot(self):
        try:
            info = self._tk.fast_info
            for attr in ("last_price", "regularMarketPrice"):
                p = getattr(info, attr, None)
                if p:
                    return float(p)
        except Exception:
            pass
        try:
            h = yf.download(self.ticker, period="1d", interval="1m",
                            progress=False, auto_adjust=True)
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.get_level_values(0)
            return float(h["Close"].dropna().iloc[-1])
        except Exception:
            return 0.0

    def _enrich(self, df, spot, T, opt_type):
        """Add mid, iv_pct, moneyness, ITM flag, BS Greeks."""
        df = df.copy()
        bid = df.get("bid", pd.Series([0.0] * len(df))).fillna(0)
        ask = df.get("ask", pd.Series([0.0] * len(df))).fillna(0)
        df["mid"] = ((bid + ask) / 2).round(2)

        if "impliedVolatility" in df.columns:
            df["iv_pct"] = (df["impliedVolatility"].fillna(0) * 100).round(1)
        else:
            df["iv_pct"] = 0.0

        if "strike" in df.columns and spot > 0:
            df["itm"] = (df["strike"] < spot) if opt_type == "call" else (df["strike"] > spot)
            df["moneyness"] = (df["strike"] / spot).round(3)

        greeks_rows = []
        for _, row in df.iterrows():
            K     = float(row.get("strike", 0) or 0)
            sigma = float(row.get("impliedVolatility", 0) or 0)
            g = _bs_greeks(spot, K, T, _RISK_FREE, sigma, opt_type) if K > 0 and sigma > 0 and T > 0 else {}
            greeks_rows.append(g)

        gdf = pd.DataFrame(greeks_rows, index=df.index)
        for col in ("delta", "gamma", "theta", "vega"):
            if col in gdf.columns:
                df[col] = gdf[col]
        return df

    @staticmethod
    def _live_mid_for_strike(ticker, expiry, opt_type, strike):
        """Fetch current bid/ask mid for a specific contract. Returns float or None."""
        try:
            chain = yf.Ticker(ticker).option_chain(expiry)
            df    = chain.calls if opt_type == "call" else chain.puts
            row   = df[df["strike"] == strike]
            if row.empty:
                return None
            b = float(row["bid"].iloc[0] or 0)
            a = float(row["ask"].iloc[0] or 0)
            return round((b + a) / 2, 2) if b + a > 0 else None
        except Exception:
            return None


# ── Aggressive Options Scanner ────────────────────────────────────────────────

class AggressiveOptionsScanner:
    """Detect sweeps, blocks, and gamma plays — 0DTE to 7DTE only."""

    MAX_DTE           = 7      # hard cap — nothing longer than 1 week
    SWEEP_VOL_OI_MULT = 5.0    # vol > 5× OI  → fresh institutional
    SWEEP_MIN_VOL     = 50     # lower bar for short-dated (0DTE has tiny OI)
    BLOCK_MIN_VOL     = 1_000  # block threshold (smaller for weeklies)
    GAMMA_MIN_VOL     = 50     # gamma rip: OTM call with any real volume

    def scan_ticker(self, ticker):
        """
        Scan 0DTE–7DTE expiries only for unusual activity.
        Returns list of alert dicts.
        """
        alerts = []
        try:
            analyzer = OptionsChainAnalyzer(ticker)
            exps     = analyzer.expirations()
            spot     = analyzer._get_spot()
            if not exps or spot <= 0:
                return alerts

            today = datetime.utcnow().date()
            for expiry in exps:
                try:
                    dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
                except ValueError:
                    continue
                if dte > self.MAX_DTE:       # skip anything beyond 7 days
                    break                    # expirations are sorted ascending
                if dte < 0:
                    continue

                try:
                    chain = analyzer.get_chain(expiry)
                except Exception:
                    continue
                if not chain:
                    continue

                for opt_type, df in (("call", chain["calls"]), ("put", chain["puts"])):
                    for _, row in df.iterrows():
                        strike = float(row.get("strike", 0) or 0)
                        vol    = float(row.get("volume",       0) or 0)
                        oi     = float(row.get("openInterest", 0) or 0)
                        mid    = float(row.get("mid",          0) or 0)
                        iv_pct = float(row.get("iv_pct",       0) or 0)
                        symbol = str(row.get("contractSymbol", ""))

                        if strike <= 0 or vol < 5:
                            continue

                        tags = []
                        # 0DTE tag
                        if dte == 0:
                            tags.append("0DTE")
                        # Sweep: high vol vs OI = new money, not hedge rolls
                        if vol >= self.SWEEP_MIN_VOL and (oi == 0 or vol >= oi * self.SWEEP_VOL_OI_MULT):
                            tags.append("SWEEP")
                        # Block trade
                        if vol >= self.BLOCK_MIN_VOL:
                            tags.append("BLOCK")
                        # Gamma rip: OTM call, short-dated, real volume
                        if opt_type == "call" and strike > spot and vol >= self.GAMMA_MIN_VOL:
                            tags.append("GAMMA_RIP")
                        # Put sweep on breakout ticker = smart hedge / counter signal
                        if opt_type == "put" and vol >= self.SWEEP_MIN_VOL and (oi == 0 or vol >= oi * self.SWEEP_VOL_OI_MULT):
                            tags.append("PUT_SWEEP")

                        if tags:
                            alerts.append({
                                "ticker":          ticker.upper(),
                                "expiry":          expiry,
                                "dte":             dte,
                                "type":            opt_type.upper(),
                                "strike":          strike,
                                "spot":            spot,
                                "volume":          int(vol),
                                "open_interest":   int(oi),
                                "vol_oi_ratio":    round(vol / oi, 1) if oi > 0 else 999,
                                "mid_price":       mid,
                                "iv_pct":          iv_pct,
                                "tags":            " · ".join(tags),
                                "contract_symbol": symbol,
                                "bias":            "BULLISH" if opt_type == "call" else "BEARISH",
                            })
        except Exception:
            pass
        return alerts

    def scan_many(self, tickers):
        """Scan multiple tickers. Returns alerts sorted by volume descending."""
        all_alerts = []
        for tk in tickers:
            all_alerts.extend(self.scan_ticker(tk))
        all_alerts.sort(key=lambda x: x.get("volume", 0), reverse=True)
        return all_alerts


# ── Options Strategy Engine ───────────────────────────────────────────────────

class OptionsStrategyEngine:
    """Auto-suggest weekly options plays (primary: 5–7 DTE, fallback: 2–4 DTE)."""

    def suggest(self, ticker, bias="bullish"):
        """
        Focuses on the current week's Friday expiry (~5–7 DTE sweet spot).
        Strategies (bullish):
          1. Weekly ATM    — ATM call,    5–7 DTE  (highest probability, full delta)
          2. Weekly OTM    — 2% OTM call, 5–7 DTE  (cheaper, higher gamma lever)
          3. Mid-Week ATM  — ATM call,    2–4 DTE  (faster, if no 5-7 DTE found)
        Strategies (bearish): same with puts.
        """
        suggestions = []
        try:
            analyzer = OptionsChainAnalyzer(ticker)
            exps     = analyzer.expirations()
            spot     = analyzer._get_spot()
            if not exps or spot <= 0:
                return suggestions

            bias     = str(bias).lower()
            is_call  = (bias != "bearish")
            opt_type = "call" if is_call else "put"
            label    = "C" if is_call else "P"

            # Fetch the 5-7 DTE chain once; reuse for both Weekly plays
            exp_week  = self._nearest_expiry(exps, 5, 7)
            week_data = analyzer.get_chain(exp_week) if exp_week else None

            # ── 1. Weekly ATM (5–7 DTE, at-the-money) ────────────────────────
            if week_data:
                df  = week_data["calls"] if is_call else week_data["puts"]
                atm = self._atm_strike(df["strike"].tolist(), spot)
                r   = df[df["strike"] == atm]
                if not r.empty:
                    mid = float(r["mid"].iloc[0])
                    iv  = float(r.get("iv_pct", pd.Series([0])).iloc[0]) if "iv_pct" in r.columns else 0.0
                    if mid > 0:
                        suggestions.append({
                            "strategy":    "Weekly ATM",
                            "description": (
                                f"ATM {ticker} ${atm:.0f}{label}  "
                                f"{week_data['dte']} DTE — sweet spot, full delta exposure"
                            ),
                            "legs":       [
                                f"BUY 1× {ticker} {exp_week} "
                                f"${atm:.0f} {opt_type.upper()} @ ${mid:.2f}"
                            ],
                            "entry_cost": round(mid * 100, 2),
                            "max_loss":   round(mid * 100, 2),
                            "max_profit": "Unlimited" if is_call else f"${atm * 100:.2f}",
                            "dte":        week_data["dte"],
                            "expiry":     exp_week,
                            "strike":     atm,
                            "opt_type":   opt_type,
                            "mid":        mid,
                            "iv_pct":     iv,
                        })
                        suggestions[-1].update(
                            OptionsStrategyEngine._compute_pop_metrics(
                                spot, atm, mid, week_data["dte"], iv, opt_type
                            )
                        )

            # ── 2. Weekly OTM (5–7 DTE, 2% out-of-the-money) ─────────────────
            if week_data:
                df = week_data["calls"] if is_call else week_data["puts"]
                if is_call:
                    otm_strike = self._otm_strike(df["strike"].tolist(), spot, 0.02)
                else:
                    otm_strike = self._itm_put_strike(df["strike"].tolist(), spot, 0.02)
                r_otm = df[df["strike"] == otm_strike]
                # Only add if it's a different strike from ATM
                if not r_otm.empty and otm_strike != (suggestions[0]["strike"] if suggestions else None):
                    mid = float(r_otm["mid"].iloc[0])
                    iv  = float(r_otm["iv_pct"].iloc[0]) if "iv_pct" in r_otm.columns else 0.0
                    if mid > 0:
                        suggestions.append({
                            "strategy":    "Weekly OTM",
                            "description": (
                                f"2% OTM {ticker} ${otm_strike:.0f}{label}  "
                                f"{week_data['dte']} DTE — cheaper entry, needs bigger move"
                            ),
                            "legs":       [
                                f"BUY 1× {ticker} {exp_week} "
                                f"${otm_strike:.0f} {opt_type.upper()} @ ${mid:.2f}"
                            ],
                            "entry_cost": round(mid * 100, 2),
                            "max_loss":   round(mid * 100, 2),
                            "max_profit": "Unlimited" if is_call else f"${otm_strike * 100:.2f}",
                            "dte":        week_data["dte"],
                            "expiry":     exp_week,
                            "strike":     otm_strike,
                            "opt_type":   opt_type,
                            "mid":        mid,
                            "iv_pct":     iv,
                        })
                        suggestions[-1].update(
                            OptionsStrategyEngine._compute_pop_metrics(
                                spot, otm_strike, mid, week_data["dte"], iv, opt_type
                            )
                        )

            # ── 3. Mid-Week ATM (2–4 DTE fallback) ───────────────────────────
            exp_mid  = self._nearest_expiry(exps, 2, 4)
            mid_data = analyzer.get_chain(exp_mid) if exp_mid else None
            if mid_data:
                df  = mid_data["calls"] if is_call else mid_data["puts"]
                atm = self._atm_strike(df["strike"].tolist(), spot)
                r   = df[df["strike"] == atm]
                if not r.empty:
                    mid = float(r["mid"].iloc[0])
                    iv  = float(r["iv_pct"].iloc[0]) if "iv_pct" in r.columns else 0.0
                    if mid > 0:
                        suggestions.append({
                            "strategy":    "Mid-Week ATM",
                            "description": (
                                f"ATM {ticker} ${atm:.0f}{label}  "
                                f"{mid_data['dte']} DTE — faster play, more theta risk"
                            ),
                            "legs":       [
                                f"BUY 1× {ticker} {exp_mid} "
                                f"${atm:.0f} {opt_type.upper()} @ ${mid:.2f}"
                            ],
                            "entry_cost": round(mid * 100, 2),
                            "max_loss":   round(mid * 100, 2),
                            "max_profit": "Unlimited" if is_call else f"${atm * 100:.2f}",
                            "dte":        mid_data["dte"],
                            "expiry":     exp_mid,
                            "strike":     atm,
                            "opt_type":   opt_type,
                            "mid":        mid,
                            "iv_pct":     iv,
                        })
                        suggestions[-1].update(
                            OptionsStrategyEngine._compute_pop_metrics(
                                spot, atm, mid, mid_data["dte"], iv, opt_type
                            )
                        )

        except Exception:
            pass
        return suggestions

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _nearest_expiry(exps, min_dte, max_dte):
        today = datetime.utcnow().date()
        for exp in exps:
            try:
                dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
                if min_dte <= dte <= max_dte:
                    return exp
            except ValueError:
                pass
        return None

    @staticmethod
    def _atm_strike(strikes, spot):
        return min(strikes, key=lambda k: abs(k - spot)) if strikes else spot

    @staticmethod
    def _otm_strike(strikes, spot, offset_pct=0.01):
        target = spot * (1 + offset_pct)
        otm = [s for s in strikes if s > spot]
        if not otm:
            return max(strikes) if strikes else spot
        return min(otm, key=lambda k: abs(k - target))

    @staticmethod
    def _itm_put_strike(strikes, spot, offset_pct=0.01):
        """Slightly below spot — 1% OTM for puts."""
        target = spot * (1 - offset_pct)
        below = [s for s in strikes if s < spot]
        if not below:
            return min(strikes) if strikes else spot
        return min(below, key=lambda k: abs(k - target))

    @staticmethod
    def _compute_pop_metrics(spot, strike, mid, dte, iv, opt_type):
        """
        Compute PoP, break-even move %, prob-ITM, and EV signal for a single
        long option leg.  Returns a dict of extra keys to merge into a suggestion.
        """
        T    = max(int(dte), 1) / 365.0
        sig  = (float(iv) / 100.0) if (iv and float(iv) > 0) else 0.30
        is_c = (opt_type == "call")
        be   = float(strike) + float(mid) if is_c else float(strike) - float(mid)
        pop  = _pop_from_bs(float(spot), be, T, sig, opt_type)
        gs   = _bs_greeks(float(spot), float(strike), T, _RISK_FREE, sig, opt_type)
        iv_f = float(iv) if iv else 50.0
        return {
            "breakeven_move_pct": round(float(mid) / float(spot) * 100, 2) if spot > 0 else 0.0,
            "pop":                pop,
            "prob_itm":           round(abs(gs.get("delta", 0.5)) * 100, 1) if gs else 50.0,
            "ev_signal":          ("Favorable" if iv_f < 30
                                   else "Unfavorable" if iv_f > 60 else "Neutral"),
        }


# ── Scan-to-Options Bridge ────────────────────────────────────────────────────

def get_scan_options_plays(conn, top_n=8, bias_override=None):
    """
    Query the most recent breakout scan results from the *calls* table and
    generate 0DTE–7DTE options plays for each ticker.

    Parameters
    ----------
    conn          : PgAdapter / sqlite3 connection
    top_n         : how many top-scoring tickers to process (default 8)
    bias_override : force 'bullish' or 'bearish' for all tickers (None = auto)

    Returns
    -------
    List of dicts, one per ticker:
      ticker, explosive_score, breakout_prob, pattern_detected,
      entry_price, target_price, stop_loss, scan_timestamp,
      bias, suggestions (list of strategy dicts)
    """
    results = []
    try:
        # Get the most recent scan_id so we only look at the latest batch
        row = conn.execute(
            "SELECT scan_id FROM calls ORDER BY scan_timestamp DESC LIMIT 1"
        ).fetchone()
        if not row:
            return results
        latest_scan_id = row[0] if row else None

        if latest_scan_id:
            rows = conn.execute(
                "SELECT ticker, explosive_score, breakout_prob, pattern_detected, "
                "entry_price, target_price, stop_loss, scan_timestamp "
                "FROM calls WHERE scan_id = ? "
                "ORDER BY explosive_score DESC LIMIT ?",
                (latest_scan_id, top_n)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ticker, explosive_score, breakout_prob, pattern_detected, "
                "entry_price, target_price, stop_loss, scan_timestamp "
                "FROM calls ORDER BY scan_timestamp DESC, explosive_score DESC LIMIT ?",
                (top_n,)
            ).fetchall()
    except Exception:
        return results

    engine = OptionsStrategyEngine()
    for row in rows:
        try:
            ticker = str(row.get("ticker") or row[0] or "")
            if not ticker:
                continue
            # Breakout scanner picks are bullish by nature; allow override
            bias = bias_override or "bullish"
            plays = engine.suggest(ticker, bias=bias)
            results.append({
                "ticker":           ticker,
                "explosive_score":  float(row.get("explosive_score") or 0),
                "breakout_prob":    float(row.get("breakout_prob")   or 0),
                "pattern_detected": str(row.get("pattern_detected")  or ""),
                "entry_price":      float(row.get("entry_price")     or 0),
                "target_price":     float(row.get("target_price")    or 0),
                "stop_loss":        float(row.get("stop_loss")       or 0),
                "scan_timestamp":   str(row.get("scan_timestamp")    or ""),
                "bias":             bias,
                "suggestions":      plays,
            })
        except Exception:
            continue
    return results


# ── Options Paper Trading Engine ──────────────────────────────────────────────

class OptionsPaperEngine:
    """Virtual options trading — $5,000 capital, 1 contract = 100 shares."""

    _INIT_SQL = """
CREATE TABLE IF NOT EXISTS options_positions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT,
    contract_symbol  TEXT,
    option_type      TEXT,
    strike           REAL,
    expiry           DATE,
    contracts        INTEGER,
    entry_price      REAL,
    entry_date       DATE,
    gross_invested   REAL,
    status           TEXT DEFAULT 'OPEN',
    exit_price       REAL,
    exit_date        DATE,
    exit_reason      TEXT,
    gross_pnl        REAL,
    net_pnl          REAL,
    strategy         TEXT
);
CREATE TABLE IF NOT EXISTS options_state (
    id               INTEGER PRIMARY KEY,
    available_cash   REAL,
    realized_pnl     REAL,
    trades_made      INTEGER,
    starting_capital REAL
);
"""

    def __init__(self, conn):
        self.conn = conn
        try:
            self.conn.executescript(self._INIT_SQL)
        except Exception:
            pass
        self._load_state()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_state(self):
        try:
            row = self.conn.execute(
                "SELECT * FROM options_state WHERE id=1"
            ).fetchone()
        except Exception:
            row = None
        if row:
            try:
                self._cash     = float(row.get("available_cash",   0) or 0)
                self._realized = float(row.get("realized_pnl",     0) or 0)
                self._trades   = int(  row.get("trades_made",      0) or 0)
                self._starting = float(row.get("starting_capital", _OPTIONS_BUDGET) or _OPTIONS_BUDGET)
                return
            except Exception:
                pass
        self._cash     = _OPTIONS_BUDGET
        self._realized = 0.0
        self._trades   = 0
        self._starting = _OPTIONS_BUDGET
        self._save_state()

    def _save_state(self):
        try:
            self.conn.execute(
                "INSERT INTO options_state "
                "(id, available_cash, realized_pnl, trades_made, starting_capital) "
                "VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "available_cash=excluded.available_cash, "
                "realized_pnl=excluded.realized_pnl, "
                "trades_made=excluded.trades_made, "
                "starting_capital=excluded.starting_capital",
                (self._cash, self._realized, self._trades, self._starting)
            )
        except Exception:
            pass

    # ── trading ──────────────────────────────────────────────────────────────

    def buy(self, ticker, contract_symbol, option_type,
            strike, expiry, contracts, entry_price, strategy="manual"):
        """Buy options. entry_price = premium per share; cost = price × 100 × contracts."""
        contracts  = max(1, int(contracts))
        cost       = round(float(entry_price) * 100 * contracts, 2)
        if cost <= 0:
            return {"ok": False, "error": "Invalid entry price"}
        if cost > self._cash:
            return {"ok": False,
                    "error": f"Insufficient cash (need ${cost:.2f}, have ${self._cash:.2f})"}
        today = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            self.conn.execute(
                "INSERT INTO options_positions "
                "(ticker, contract_symbol, option_type, strike, expiry, contracts, "
                "entry_price, entry_date, gross_invested, status, strategy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)",
                (str(ticker).upper(), str(contract_symbol),
                 str(option_type).lower(), float(strike), str(expiry),
                 contracts, float(entry_price), today, cost, str(strategy))
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        self._cash   -= cost
        self._trades += 1
        self._save_state()
        return {"ok": True, "cost": cost, "cash_remaining": round(self._cash, 2)}

    def close(self, position_id, exit_price, exit_reason="MANUAL"):
        """Close open position at exit_price (per share premium)."""
        try:
            row = self.conn.execute(
                "SELECT * FROM options_positions WHERE id=? AND status='OPEN'",
                (int(position_id),)
            ).fetchone()
        except Exception as e:
            return {"ok": False, "error": str(e)}
        if not row:
            return {"ok": False, "error": "Position not found or already closed"}

        contracts  = int(row.get("contracts", 1) or 1)
        invested   = float(row.get("gross_invested", 0) or 0)
        gross_exit = round(float(exit_price) * 100 * contracts, 2)
        gross_pnl  = round(gross_exit - invested, 2)
        today      = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            self.conn.execute(
                "UPDATE options_positions "
                "SET status='CLOSED', exit_price=?, exit_date=?, "
                "exit_reason=?, gross_pnl=?, net_pnl=? WHERE id=?",
                (float(exit_price), today, str(exit_reason),
                 gross_pnl, gross_pnl, int(position_id))
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        self._cash     += gross_exit
        self._realized += gross_pnl
        self._save_state()
        return {"ok": True, "net_pnl": gross_pnl, "cash": round(self._cash, 2)}

    def expire_check(self):
        """Auto-expire worthless options past their expiry date."""
        closed = []
        today  = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            expired = self.conn.execute(
                "SELECT * FROM options_positions WHERE status='OPEN' AND expiry < ?",
                (today,)
            ).fetchall()
        except Exception:
            return []
        for row in expired:
            result = self.close(int(row.get("id")), 0.0, "EXPIRED_WORTHLESS")
            if result.get("ok"):
                d = {k: row.get(k) for k in row.keys()}
                d["net_pnl"] = result["net_pnl"]
                closed.append(d)
        return closed

    def check_exit_alerts(self):
        """
        Scan open positions for auto-exit trigger conditions.

        Alert types (non-exclusive per position):
          TAKE_PROFIT — unrealized P&L >= +50 %
          STOP_LOSS   — unrealized P&L <= -50 %
          TIME_DECAY  — ≤ 2 calendar days to expiry

        Returns list of alert dicts:
          id, ticker, strike, option_type, expiry, dte_remaining,
          unrealized_pct, alert_type, message
        """
        alerts = []
        today  = datetime.utcnow().date()
        positions = self.get_positions("OPEN")
        for p in positions:
            pid    = p.get("id")
            tkr    = str(p.get("ticker",      "") or "")
            stk    = float(p.get("strike",    0)  or 0)
            otype  = str(p.get("option_type", "call") or "call")
            exp    = str(p.get("expiry",      "") or "")
            letter = otype[0].upper() if otype else "C"

            # DTE remaining
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte_rem  = max(0, (exp_date - today).days)
            except Exception:
                dte_rem  = 99

            unreal_pct = p.get("unrealized_pct")   # None if no live price

            # ── +50 % take profit ─────────────────────────────────────────
            if unreal_pct is not None and unreal_pct >= 50:
                alerts.append({
                    "id":             pid,
                    "ticker":         tkr,
                    "strike":         stk,
                    "option_type":    otype,
                    "expiry":         exp,
                    "dte_remaining":  dte_rem,
                    "unrealized_pct": unreal_pct,
                    "alert_type":     "TAKE_PROFIT",
                    "message": (
                        f"🎯 TAKE PROFIT — {tkr} ${stk:.0f}{letter} "
                        f"is up {unreal_pct:+.1f}%.  "
                        f"Rule: close at 50 % profit to lock gains."
                    ),
                })

            # ── -50 % stop loss ───────────────────────────────────────────
            elif unreal_pct is not None and unreal_pct <= -50:
                alerts.append({
                    "id":             pid,
                    "ticker":         tkr,
                    "strike":         stk,
                    "option_type":    otype,
                    "expiry":         exp,
                    "dte_remaining":  dte_rem,
                    "unrealized_pct": unreal_pct,
                    "alert_type":     "STOP_LOSS",
                    "message": (
                        f"🛑 STOP LOSS — {tkr} ${stk:.0f}{letter} "
                        f"is down {unreal_pct:.1f}%.  "
                        f"Rule: cut losses at -50 % to preserve capital."
                    ),
                })

            # ── ≤ 2 DTE time-decay warning ────────────────────────────────
            if 0 <= dte_rem <= 2:
                alerts.append({
                    "id":             pid,
                    "ticker":         tkr,
                    "strike":         stk,
                    "option_type":    otype,
                    "expiry":         exp,
                    "dte_remaining":  dte_rem,
                    "unrealized_pct": unreal_pct,
                    "alert_type":     "TIME_DECAY",
                    "message": (
                        f"⏱ TIME DECAY — {tkr} ${stk:.0f}{letter} "
                        f"expires in {dte_rem} day(s) ({exp}).  "
                        f"Theta decay is maximal — exit or accept full loss."
                    ),
                })

        return alerts

    def get_positions(self, status="OPEN"):
        """Return positions list. OPEN positions get live unrealized P&L."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM options_positions WHERE status=? ORDER BY entry_date DESC",
                (status,)
            ).fetchall()
        except Exception:
            return []
        result = []
        for row in rows:
            d = {k: row.get(k) for k in row.keys()}
            if status == "OPEN":
                contracts = int(d.get("contracts", 1) or 1)
                invested  = float(d.get("gross_invested", 0) or 0)
                live      = OptionsChainAnalyzer._live_mid_for_strike(
                    str(d.get("ticker", "")),
                    str(d.get("expiry", "")),
                    str(d.get("option_type", "call")),
                    float(d.get("strike", 0) or 0),
                )
                if live and live > 0:
                    cur_val = live * 100 * contracts
                    d["current_price"]  = live
                    d["current_value"]  = round(cur_val, 2)
                    d["unrealized_pnl"] = round(cur_val - invested, 2)
                    d["unrealized_pct"] = round(
                        (cur_val - invested) / invested * 100, 1
                    ) if invested > 0 else 0.0
                else:
                    d["current_price"]  = None
                    d["current_value"]  = None
                    d["unrealized_pnl"] = None
                    d["unrealized_pct"] = None
            result.append(d)
        return result

    def get_summary(self):
        try:
            n_open = int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM options_positions WHERE status='OPEN'"
                ).fetchone()[0] or 0
            )
        except Exception:
            n_open = 0
        try:
            row = self.conn.execute(
                "SELECT COUNT(*), SUM(net_pnl) "
                "FROM options_positions WHERE status='CLOSED'"
            ).fetchone()
            n_closed   = int(row[0] or 0)
            total_real = float(row[1] or 0)
        except Exception:
            n_closed   = 0
            total_real = 0.0
        try:
            n_wins = int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM options_positions "
                    "WHERE status='CLOSED' AND net_pnl > 0"
                ).fetchone()[0] or 0
            )
        except Exception:
            n_wins = 0
        win_rate = round(n_wins / n_closed * 100, 1) if n_closed > 0 else 0.0
        return {
            "available_cash":   round(self._cash,     2),
            "realized_pnl":     round(self._realized, 2),
            "starting_capital": self._starting,
            "total_return_pct": round(self._realized / self._starting * 100, 2),
            "n_open":           n_open,
            "n_closed":         n_closed,
            "n_wins":           n_wins,
            "win_rate":         win_rate,
            "trades_made":      self._trades,
        }

    def reset(self):
        """Wipe all options trades and reset cash to $500."""
        try:
            self.conn.execute("DELETE FROM options_positions")
        except Exception:
            pass
        try:
            self.conn.execute("DELETE FROM options_state")
        except Exception:
            pass
        self._cash     = _OPTIONS_BUDGET
        self._realized = 0.0
        self._trades   = 0
        self._starting = _OPTIONS_BUDGET
        self._save_state()


# ══════════════════════════════════════════════════════════════════════════════
# MARKET CONTEXT UTILITIES — VIX · Fear & Greed · Economic Calendar · Earnings
# ══════════════════════════════════════════════════════════════════════════════

def get_vix_level():
    """
    Fetch current VIX from yfinance.
    Returns dict {vix, regime, advice} or None on failure.
    """
    try:
        price = None
        try:
            price = getattr(yf.Ticker("^VIX").fast_info, "last_price", None)
        except Exception:
            pass
        if not price:
            hist = yf.download("^VIX", period="1d", interval="5m",
                               progress=False, auto_adjust=True)
            if not hist.empty:
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                price = float(hist["Close"].dropna().iloc[-1])
        if not price:
            return None
        level = round(float(price), 2)
        if level < 15:
            regime = "LOW"
            advice = "Vol is cheap — option buyers have edge; premiums affordable"
        elif level < 20:
            regime = "NORMAL"
            advice = "Typical vol environment — balanced risk/reward"
        elif level < 30:
            regime = "ELEVATED"
            advice = "Vol expensive — be selective; weekly spreads better value"
        else:
            regime = "EXTREME"
            advice = "Fear spike — options very expensive; only high-conviction plays"
        return {"vix": level, "regime": regime, "advice": advice}
    except Exception:
        return None


def get_fear_greed():
    """
    Fetch CNN Fear & Greed index from their public data endpoint.
    Returns dict {score, rating, emoji, color} or None on failure.
    """
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"},
        )
        if resp.status_code != 200:
            return None
        fg     = resp.json().get("fear_and_greed", {})
        score  = round(float(fg.get("score", 50)), 1)
        rating = str(fg.get("rating", "Neutral"))
        if score <= 25:
            emoji, color = "😱", "#f85149"
        elif score <= 45:
            emoji, color = "😨", "#e3b341"
        elif score <= 55:
            emoji, color = "😐", "#8b949e"
        elif score <= 75:
            emoji, color = "😏", "#3fb950"
        else:
            emoji, color = "🤑", "#58a6ff"
        return {"score": score, "rating": rating, "emoji": emoji, "color": color}
    except Exception:
        return None


def get_economic_events(days_ahead=7):
    """
    Fetch high-impact USD economic events for the next N days.
    Source: ForexFactory public JSON feed.
    Returns list of {title, date, time, impact} dicts, sorted by date.
    Returns [] on failure.
    """
    events = []
    today  = datetime.utcnow().date()
    cutoff = today + timedelta(days=days_ahead)
    for url in [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
    ]:
        try:
            resp = requests.get(
                url, timeout=8,
                headers={"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"},
            )
            if resp.status_code != 200:
                continue
            for ev in resp.json():
                if ev.get("country") != "USD":
                    continue
                if str(ev.get("impact", "")).lower() != "high":
                    continue
                try:
                    ev_date = datetime.strptime(
                        str(ev.get("date", ""))[:10], "%Y-%m-%d"
                    ).date()
                except ValueError:
                    continue
                if today <= ev_date <= cutoff:
                    events.append({
                        "title":  str(ev.get("title", "")),
                        "date":   str(ev_date),
                        "time":   str(ev.get("time", "")),
                        "impact": "High",
                    })
        except Exception:
            continue
    seen, unique = set(), []
    for e in events:
        key = f"{e['date']}_{e['title']}"
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return sorted(unique, key=lambda x: x["date"])


def check_earnings_in_window(ticker, expiry_date_str):
    """
    Check if *ticker* reports earnings between today and *expiry_date_str*.
    Returns dict:
      has_earnings  — bool
      earnings_date — str  (only if has_earnings)
      days_away     — int  (only if has_earnings)
      warning       — str  (only if has_earnings)
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return {"has_earnings": False}

        earnings_date = None

        # yfinance ≥ 0.2.x returns a plain dict
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed is not None:
                candidates = ed if isinstance(ed, (list, tuple)) else [ed]
                for c in candidates:
                    try:
                        earnings_date = pd.Timestamp(c).date()
                        break
                    except Exception:
                        pass

        # Older yfinance returns a DataFrame
        elif hasattr(cal, "columns"):
            try:
                col = next(
                    (c for c in cal.columns if "Earnings" in str(c)), None
                )
                if col is not None:
                    val = cal[col].dropna()
                    if not val.empty:
                        earnings_date = pd.Timestamp(val.iloc[0]).date()
            except Exception:
                pass

        if earnings_date is None:
            return {"has_earnings": False}

        today  = datetime.utcnow().date()
        expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()

        if today <= earnings_date <= expiry:
            days_away = (earnings_date - today).days
            return {
                "has_earnings":  True,
                "earnings_date": str(earnings_date),
                "days_away":     days_away,
                "warning": (
                    f"⚠️ EARNINGS in {days_away}d ({earnings_date}) — "
                    "IV will spike before then crush after the report. "
                    "Exit before earnings or size down significantly."
                ),
            }
    except Exception:
        pass
    return {"has_earnings": False}


# ══════════════════════════════════════════════════════════════════════════════
# RISK & PROFITABILITY UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_entry_quality(ticker, bias="bullish"):
    """
    Entry Quality Score 0–100 built from five independent signals:

      1. Time of day ET  (30 pts) — prime windows score higher
      2. VWAP position   (25 pts) — price above/below VWAP matches bias
      3. Extension %     (20 pts) — proximity to VWAP; overextension penalised
      4. Volume ratio    (15 pts) — today vs 20-day average, projected to full day
      5. IV rank         (10 pts) — lower rank → cheaper options → better entry

    Returns dict:
      score      — int 0-100
      grade      — "A" / "B" / "C" / "D"
      advice     — plain-English summary
      components — per-signal breakdown {pts, label, ...}
    """
    # ── 1. Time of day ────────────────────────────────────────────────────────
    try:
        import pytz
        now_et   = datetime.now(pytz.timezone("US/Eastern"))
        time_dec = now_et.hour + now_et.minute / 60.0
    except Exception:
        time_dec = 10.5   # assume mid-morning if timezone unavailable

    components = {}
    score      = 0

    if 9.75 <= time_dec <= 10.5:
        t_pts, t_lbl = 30, "Prime open 9:45–10:30 ✓"
    elif 10.5 < time_dec <= 11.5:
        t_pts, t_lbl = 22, "Good window 10:30–11:30"
    elif 14.0 <= time_dec <= 15.0:
        t_pts, t_lbl = 25, "Power hour 2–3 PM ✓"
    elif 15.0 < time_dec <= 15.5:
        t_pts, t_lbl = 18, "Late momentum 3–3:30 PM"
    elif 11.5 < time_dec < 14.0:
        t_pts, t_lbl = 8,  "Midday dead zone — avoid"
    else:
        t_pts, t_lbl = 5,  "Off-hours"
    score += t_pts
    components["time_of_day"] = {"pts": t_pts, "label": t_lbl}

    # ── 2–4. Intraday data ────────────────────────────────────────────────────
    is_bullish = str(bias).lower() != "bearish"
    try:
        hist1m = yf.download(ticker, period="1d", interval="1m",
                             progress=False, auto_adjust=True)
        if isinstance(hist1m.columns, pd.MultiIndex):
            hist1m.columns = hist1m.columns.get_level_values(0)
        hist1m = hist1m.dropna(subset=["Close"])

        if not hist1m.empty:
            close    = hist1m["Close"]
            volume   = hist1m["Volume"].clip(lower=0)
            spot     = float(close.iloc[-1])
            cum_vol  = volume.cumsum()
            vwap     = (close * volume).cumsum() / cum_vol.replace(0, np.nan)
            vwap_now = float(vwap.iloc[-1]) if not vwap.empty else spot

            # 2. VWAP (25 pts)
            above = spot > vwap_now
            if (is_bullish and above) or (not is_bullish and not above):
                v_pts = 25
                v_lbl = f"Price {'above' if above else 'below'} VWAP ✓  (VWAP ${vwap_now:.2f})"
            else:
                v_pts = 0
                v_lbl = f"Price on wrong side of VWAP ${vwap_now:.2f} ✗"
            score += v_pts
            components["vwap"] = {"pts": v_pts, "label": v_lbl, "vwap": round(vwap_now, 2)}

            # 3. Extension (20 pts)
            ext_pct = abs(spot - vwap_now) / vwap_now * 100 if vwap_now > 0 else 0.0
            if ext_pct <= 0.5:
                e_pts, e_lbl = 20, f"Tight ({ext_pct:.2f}% from VWAP) — ideal entry ✓"
            elif ext_pct <= 1.5:
                e_pts, e_lbl = 14, f"Moderate extension ({ext_pct:.2f}% from VWAP)"
            elif ext_pct <= 3.0:
                e_pts, e_lbl = 6,  f"Extended {ext_pct:.2f}% — risky entry"
            else:
                e_pts, e_lbl = 0,  f"Overextended {ext_pct:.2f}% — wait for pullback"
            score += e_pts
            components["extension"] = {"pts": e_pts, "label": e_lbl,
                                        "ext_pct": round(ext_pct, 2)}

            # 4. Volume ratio (15 pts)
            try:
                hist_d  = yf.download(ticker, period="1mo", interval="1d",
                                      progress=False, auto_adjust=True)
                if isinstance(hist_d.columns, pd.MultiIndex):
                    hist_d.columns = hist_d.columns.get_level_values(0)
                avg_vol  = float(hist_d["Volume"].tail(20).mean()) if not hist_d.empty else 0
                elapsed  = max(1, len(hist1m))
                proj_vol = float(volume.sum()) * 390 / elapsed
                ratio    = proj_vol / avg_vol if avg_vol > 0 else 1.0
                if ratio >= 1.5:
                    vol_pts, vol_lbl = 15, f"High volume ({ratio:.1f}× avg) — conviction ✓"
                elif ratio >= 1.0:
                    vol_pts, vol_lbl = 10, f"Normal volume ({ratio:.1f}× avg)"
                else:
                    vol_pts, vol_lbl = 3,  f"Low volume ({ratio:.1f}× avg) — weak move"
            except Exception:
                ratio = 1.0
                vol_pts, vol_lbl = 7, "Volume data unavailable (neutral)"
            score += vol_pts
            components["volume"] = {"pts": vol_pts, "label": vol_lbl,
                                     "ratio": round(ratio, 2)}
        else:
            score += 12
            components["vwap"]      = {"pts": 0,  "label": "No intraday data"}
            components["extension"] = {"pts": 0,  "label": "No intraday data"}
            components["volume"]    = {"pts": 12, "label": "No intraday data (partial)"}
    except Exception:
        score += 12
        components["vwap"]      = {"pts": 0,  "label": "Data unavailable"}
        components["extension"] = {"pts": 0,  "label": "Data unavailable"}
        components["volume"]    = {"pts": 12, "label": "Data unavailable"}

    # ── 5. IV rank (10 pts) ───────────────────────────────────────────────────
    try:
        ivr     = OptionsChainAnalyzer(ticker).iv_rank()
        iv_rank = float(ivr.get("iv_rank", 50)) if ivr else 50.0
        if iv_rank < 30:
            iv_pts, iv_lbl = 10, f"IV Rank {iv_rank:.0f} — cheap options ✓"
        elif iv_rank < 60:
            iv_pts, iv_lbl = 5,  f"IV Rank {iv_rank:.0f} — normal premium"
        else:
            iv_pts, iv_lbl = 0,  f"IV Rank {iv_rank:.0f} — expensive options ✗"
    except Exception:
        iv_rank = 50.0
        iv_pts, iv_lbl = 5, "IV data unavailable (neutral)"
    score += iv_pts
    components["iv_rank"] = {"pts": iv_pts, "label": iv_lbl,
                              "rank": round(iv_rank, 1)}

    # ── Grade ─────────────────────────────────────────────────────────────────
    score = min(100, max(0, score))
    if score >= 75:
        grade, advice = "A", "Excellent setup — high-conviction entry"
    elif score >= 60:
        grade, advice = "B", "Good setup — proceed with standard sizing"
    elif score >= 45:
        grade, advice = "C", "Marginal — reduce size or wait for better entry"
    else:
        grade, advice = "D", "Poor setup — avoid or paper trade only"

    return {"score": score, "grade": grade, "advice": advice, "components": components}


def get_conviction_score(ticker, expiry, bias, conn,
                         vix_data=None, entry_quality=None):
    """
    5-signal Conviction Score 0–100.  Each signal contributes 20 pts.

      1. Breakout scan membership — ticker in latest scan with strong explosive_score
      2. Unusual options flow     — unusual activity aligns with bias direction
      3. VIX regime               — LOW/NORMAL favoured; ELEVATED/EXTREME penalised
      4. Earnings safety          — no earnings inside the trade window
      5. Entry quality            — entry_quality score >= 60 for full credit

    Returns dict:
      score   — int 0-100
      grade   — "A" / "B" / "C" / "D"
      advice  — recommended action
      signals — per-signal breakdown {pts, label}
    """
    signals    = {}
    score      = 0
    is_bullish = str(bias).lower() != "bearish"

    # ── 1. Breakout scan (20 pts) ─────────────────────────────────────────────
    try:
        row = conn.execute(
            "SELECT explosive_score FROM calls "
            "WHERE ticker=? ORDER BY scan_timestamp DESC LIMIT 1",
            (str(ticker).upper(),)
        ).fetchone()
        if row:
            exp_score = float(row.get("explosive_score") or row[0] or 0)
            if exp_score >= 70:
                s1_pts, s1_lbl = 20, f"Strong breakout score {exp_score:.0f} ✓"
            elif exp_score >= 40:
                s1_pts, s1_lbl = 12, f"Moderate score {exp_score:.0f}"
            else:
                s1_pts, s1_lbl = 5,  f"Weak score {exp_score:.0f}"
        else:
            s1_pts, s1_lbl = 0, "Not in breakout scan ✗"
    except Exception:
        s1_pts, s1_lbl = 10, "Scan data unavailable (neutral)"
    score += s1_pts
    signals["breakout_scan"] = {"pts": s1_pts, "label": s1_lbl}

    # ── 2. Unusual flow (20 pts) ──────────────────────────────────────────────
    try:
        flow       = AggressiveOptionsScanner().scan_ticker(ticker)
        bull_flow  = sum(1 for f in flow if f.get("bias") == "BULLISH")
        bear_flow  = sum(1 for f in flow if f.get("bias") == "BEARISH")
        total_flow = bull_flow + bear_flow
        if total_flow == 0:
            s2_pts, s2_lbl = 10, "No unusual flow detected (neutral)"
        else:
            aligned   = bull_flow if is_bullish else bear_flow
            align_pct = aligned / total_flow * 100
            if align_pct >= 70:
                s2_pts = 20
                s2_lbl = f"{'Bullish' if is_bullish else 'Bearish'} flow {align_pct:.0f}% aligned ✓"
            elif align_pct >= 40:
                s2_pts, s2_lbl = 10, f"Mixed flow — {align_pct:.0f}% aligned"
            else:
                s2_pts, s2_lbl = 0,  f"Flow opposing bias ✗"
    except Exception:
        s2_pts, s2_lbl = 10, "Flow data unavailable (neutral)"
    score += s2_pts
    signals["unusual_flow"] = {"pts": s2_pts, "label": s2_lbl}

    # ── 3. VIX regime (20 pts) ────────────────────────────────────────────────
    vix = vix_data or get_vix_level()
    if vix:
        regime = vix.get("regime", "NORMAL")
        if regime == "LOW":
            s3_pts, s3_lbl = 20, f"VIX {vix['vix']:.1f} — LOW vol ✓"
        elif regime == "NORMAL":
            s3_pts, s3_lbl = 16, f"VIX {vix['vix']:.1f} — NORMAL"
        elif regime == "ELEVATED":
            s3_pts, s3_lbl = 8,  f"VIX {vix['vix']:.1f} — ELEVATED"
        else:
            s3_pts, s3_lbl = 2,  f"VIX {vix['vix']:.1f} — EXTREME ✗"
    else:
        s3_pts, s3_lbl = 10, "VIX unavailable (neutral)"
    score += s3_pts
    signals["vix_regime"] = {"pts": s3_pts, "label": s3_lbl}

    # ── 4. Earnings safety (20 pts) ───────────────────────────────────────────
    try:
        ew = check_earnings_in_window(ticker, expiry)
        if not ew.get("has_earnings"):
            s4_pts, s4_lbl = 20, "No earnings in window ✓"
        else:
            days_away = ew.get("days_away", 0)
            if days_away >= 5:
                s4_pts, s4_lbl = 8,  f"Earnings in {days_away}d — risk elevated"
            else:
                s4_pts, s4_lbl = 0,  f"Earnings in {days_away}d — very high risk ✗"
    except Exception:
        s4_pts, s4_lbl = 15, "Earnings data unavailable (partial)"
    score += s4_pts
    signals["earnings_safety"] = {"pts": s4_pts, "label": s4_lbl}

    # ── 5. Entry quality (20 pts) ─────────────────────────────────────────────
    eq = entry_quality
    if eq is None:
        try:
            eq = get_entry_quality(ticker, bias)
        except Exception:
            eq = None
    if eq:
        eq_score = eq.get("score", 50)
        if eq_score >= 75:
            s5_pts, s5_lbl = 20, f"Entry quality {eq_score}/100 — A ✓"
        elif eq_score >= 60:
            s5_pts, s5_lbl = 15, f"Entry quality {eq_score}/100 — B"
        elif eq_score >= 45:
            s5_pts, s5_lbl = 8,  f"Entry quality {eq_score}/100 — C"
        else:
            s5_pts, s5_lbl = 2,  f"Entry quality {eq_score}/100 — poor ✗"
    else:
        s5_pts, s5_lbl = 10, "Entry quality unavailable (neutral)"
    score += s5_pts
    signals["entry_quality"] = {"pts": s5_pts, "label": s5_lbl}

    # ── Grade ─────────────────────────────────────────────────────────────────
    score = min(100, max(0, score))
    if score >= 80:
        grade, advice = "A", "HIGH CONVICTION — full standard size"
    elif score >= 65:
        grade, advice = "B", "Good conviction — standard position size"
    elif score >= 50:
        grade, advice = "C", "Moderate conviction — reduce to half size"
    else:
        grade, advice = "D", "Low conviction — skip or paper trade only"

    return {"score": score, "grade": grade, "advice": advice, "signals": signals}


def get_position_sizing(account_cash, mid_price, vix_data=None,
                        closed_trades=None, risk_pct=5.0):
    """
    Dynamic position sizing: VIX-adjusted fixed fraction blended with half-Kelly
    derived from live trade history.

    Parameters
    ----------
    account_cash  : float — available cash
    mid_price     : float — option premium per share (1 contract = 100 shares)
    vix_data      : dict from get_vix_level() or None
    closed_trades : list of position dicts with 'net_pnl' and 'gross_invested'
    risk_pct      : float — base % of capital to risk per trade (default 5 %)

    Returns dict:
      contracts     — recommended number of contracts (≥ 1 or 0 if too expensive)
      total_cost    — dollar cost of recommended size
      max_risk_pct  — total_cost as % of account
      kelly_f       — half-Kelly fraction (0 if history < 10 trades)
      vix_mult      — VIX regime multiplier applied
      sizing_method — 'half_kelly' or 'fixed_fraction'
      advice        — human-readable explanation
    """
    if account_cash <= 0 or mid_price <= 0:
        return {"contracts": 0,
                "advice": "Insufficient cash or invalid premium price.",
                "total_cost": 0, "max_risk_pct": 0,
                "kelly_f": 0, "vix_mult": 1.0, "sizing_method": "n/a"}

    # ── VIX multiplier ────────────────────────────────────────────────────────
    vix = vix_data or get_vix_level()
    if vix:
        regime   = vix.get("regime", "NORMAL")
        vix_mult = {"LOW": 1.2, "NORMAL": 1.0, "ELEVATED": 0.6, "EXTREME": 0.3}.get(regime, 1.0)
    else:
        vix_mult = 1.0

    # ── Half-Kelly (needs ≥ 10 closed trades) ─────────────────────────────────
    kelly_f       = 0.0
    sizing_method = "fixed_fraction"
    if closed_trades and len(closed_trades) >= 10:
        try:
            pairs   = [(float(t.get("net_pnl", 0) or 0),
                        float(t.get("gross_invested", 1) or 1))
                       for t in closed_trades]
            returns = [p / i for p, i in pairs if i > 0]
            wins    = [r for r in returns if r > 0]
            losses  = [abs(r) for r in returns if r < 0]
            if wins and losses:
                p_win      = len(wins) / len(returns)
                b_avg      = sum(wins)   / len(wins)
                a_avg      = sum(losses) / len(losses)
                full_kelly = (p_win * b_avg - (1 - p_win) * a_avg) / b_avg if b_avg > 0 else 0.0
                kelly_f    = max(0.0, min(0.25, full_kelly / 2))
                sizing_method = "half_kelly"
        except Exception:
            kelly_f = 0.0

    # ── Effective fraction ────────────────────────────────────────────────────
    base_frac = risk_pct / 100.0
    if sizing_method == "half_kelly" and kelly_f > 0:
        effective_frac = (base_frac + kelly_f) / 2
    else:
        effective_frac = base_frac
    effective_frac *= vix_mult

    risk_dollars      = round(account_cash * effective_frac, 2)
    cost_per_contract = round(mid_price * 100, 2)
    contracts         = max(1, int(risk_dollars / cost_per_contract))

    # Safety cap: never > 25 % of cash in one trade
    max_by_cash = max(1, int(account_cash * 0.25 / cost_per_contract))
    contracts   = min(contracts, max_by_cash)

    total_cost   = round(cost_per_contract * contracts, 2)
    max_risk_pct = round(total_cost / account_cash * 100, 1) if account_cash > 0 else 0.0

    if total_cost > account_cash:
        return {"contracts": 0,
                "advice": f"Insufficient cash — 1 contract costs ${cost_per_contract:.2f}.",
                "total_cost": 0, "max_risk_pct": 0,
                "kelly_f": round(kelly_f, 4), "vix_mult": vix_mult,
                "sizing_method": sizing_method}

    advice = (
        f"{'Half-Kelly' if sizing_method == 'half_kelly' else 'Fixed-fraction'} sizing: "
        f"{contracts} contract(s) @ ${mid_price:.2f} = ${total_cost:.2f} "
        f"({max_risk_pct:.1f}% of capital)  ·  VIX mult {vix_mult:.1f}×"
        + (f"  ·  Kelly f={kelly_f:.3f}" if kelly_f > 0 else "")
    )
    return {
        "contracts":    contracts,
        "risk_dollars": round(risk_dollars, 2),
        "total_cost":   total_cost,
        "max_risk_pct": max_risk_pct,
        "kelly_f":      round(kelly_f, 4),
        "vix_mult":     vix_mult,
        "sizing_method": sizing_method,
        "advice":        advice,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SMART MONEY ANALYSIS — Wyckoff · Institutional Volume · Gamma Walls · Squeeze
# ══════════════════════════════════════════════════════════════════════════════

def detect_wyckoff_phase(ticker, lookback_days=60):
    """
    Identify which Wyckoff market phase the stock is currently in.

    Uses daily OHLCV data to score six structural conditions:
      ACCUMULATION   — price near lows, volume building, tight range
      SPRING         — shakeout below prior range low + recovery (best long entry)
      MARKUP         — breakout above range on strong volume (trend started)
      REACCUMULATION — mid-range pause with decreasing volume (continuation)
      DISTRIBUTION   — price at highs, volume expanding, closes weak
      MARKDOWN       — falling prices, no institutional buying support

    Returns dict:
      phase, confidence (0-100), description, spring_detected,
      price_position_pct, volume_trend, period_high, period_low,
      is_tradeable (True for ACCUMULATION / SPRING / MARKUP / REACCUMULATION)
      action  — plain-English trading implication
    """
    try:
        # Use shared cache — same data fetched by master_score, squeeze scanner,
        # duration predictor, multi-TF check.  Eliminates 5× duplicate downloads.
        hist = get_cached_history(ticker, period="3mo", interval="1d")
        hist = hist.dropna(subset=["Close"]) if not hist.empty else hist
        if len(hist) < 20:
            return {"phase": "INSUFFICIENT_DATA", "confidence": 0,
                    "description": "Not enough history", "is_tradeable": False,
                    "action": "Skip — insufficient data"}

        recent = hist.tail(lookback_days)

        period_high  = float(recent["High"].max())
        period_low   = float(recent["Low"].min())
        period_range = max(period_high - period_low, 0.01)
        current      = float(recent["Close"].iloc[-1])
        price_pos    = (current - period_low) / period_range   # 0 = bottom, 1 = top

        # Volume trend: recent 5-bar avg vs lookback avg
        avg_vol    = float(recent["Volume"].mean())
        recent_vol = float(recent["Volume"].tail(5).mean())
        vol_trend  = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # Range contraction: last 10-bar range vs full period range
        r10          = float(recent["High"].tail(10).max() - recent["Low"].tail(10).min())
        range_ratio  = r10 / period_range

        # Close position within each bar (strength): 0 = close at low, 1 = close at high
        bar_range    = (recent["High"] - recent["Low"]).replace(0, np.nan)
        close_pct    = ((recent["Close"] - recent["Low"]) / bar_range).tail(5).mean()
        if pd.isna(close_pct):
            close_pct = 0.5

        # Spring: recent 5-bar low dips below the period low, then recovers above it
        recent_5_low    = float(recent["Low"].tail(5).min())
        spring_detected = (recent_5_low < period_low * 1.002) and (current > period_low)

        # ── Phase logic ───────────────────────────────────────────────────────
        if spring_detected and price_pos < 0.40:
            phase      = "SPRING"
            confidence = 85
            desc       = ("Shakeout below prior range low followed by recovery — "
                          "institutions engineered a final flush. Classic Wyckoff Spring: "
                          "the best long entry in the entire cycle.")
            action     = "STRONG BUY — enter on recovery above spring low with tight stop"
            tradeable  = True

        elif price_pos < 0.30 and range_ratio < 0.35 and vol_trend >= 1.0:
            phase      = "ACCUMULATION"
            confidence = 75
            desc       = ("Price consolidating near lows with building volume and tight range. "
                          "Institutions are quietly absorbing supply — retail doesn't notice yet.")
            action     = "BUY — accumulate near range lows, stop below range"
            tradeable  = True

        elif price_pos > 0.65 and vol_trend > 1.2 and close_pct > 0.55:
            phase      = "MARKUP"
            confidence = 70
            desc       = ("Price breaking above the accumulation range on strong volume "
                          "and closing near highs — institutional demand is driving the trend.")
            action     = "BUY — momentum entry, trail stop at VWAP"
            tradeable  = True

        elif 0.30 <= price_pos <= 0.65 and range_ratio < 0.30 and vol_trend < 0.9:
            phase      = "REACCUMULATION"
            confidence = 60
            desc       = ("Mid-range pause with declining volume and tight price action — "
                          "a healthy rest before the next leg up.")
            action     = "BUY — wait for range break with volume confirmation"
            tradeable  = True

        elif price_pos > 0.70 and close_pct < 0.40 and vol_trend > 1.1:
            phase      = "DISTRIBUTION"
            confidence = 65
            desc       = ("Price near highs but closing in the lower half of bars on "
                          "high volume — institutions are selling into retail strength.")
            action     = "AVOID / SHORT — distribution in progress; do not buy"
            tradeable  = False

        elif price_pos < 0.35 and vol_trend < 0.8 and close_pct < 0.45:
            phase      = "MARKDOWN"
            confidence = 60
            desc       = ("Price falling on declining volume with weak closes — "
                          "no institutional demand is stepping in.")
            action     = "AVOID — no institutional support; wait for selling climax"
            tradeable  = False

        else:
            phase      = "UNDEFINED"
            confidence = 30
            desc       = "No clear Wyckoff phase. Price is in a mixed structure."
            action     = "NEUTRAL — wait for clearer setup"
            tradeable  = False

        return {
            "phase":              phase,
            "confidence":         confidence,
            "description":        desc,
            "action":             action,
            "spring_detected":    spring_detected,
            "price_position_pct": round(price_pos * 100, 1),
            "volume_trend":       round(vol_trend, 2),
            "range_ratio":        round(range_ratio, 2),
            "period_high":        round(period_high, 2),
            "period_low":         round(period_low, 2),
            "current_price":      round(current, 2),
            "is_tradeable":       tradeable,
        }
    except Exception:
        return {"phase": "ERROR", "confidence": 0,
                "description": "Analysis failed", "is_tradeable": False,
                "action": "Skip — data error"}


def analyze_institutional_volume(ticker, lookback_days=20):
    """
    Detect institutional footprints in the daily volume/price relationship.

    Patterns detected:
      ABSORPTION        — high volume + narrow range (supply being soaked up)
      SELLING_CLIMAX    — extreme down-volume bar that closes upper half (buyers win)
      VOLUME_DRY_UP     — declining volume on a pullback (healthy, institutions not selling)
      EFFORT_NO_RESULT  — huge volume, tiny price move (war between buyers/sellers)
      DISTRIBUTION_SIGN — high volume at highs closing in lower half (institutions selling)
      BREAKOUT_CONFIRM  — high volume bar closing at/near session high above prior range
      NORMAL            — no unusual institutional signal detected

    Returns dict:
      primary_pattern, all_patterns, bullish_score (0-100),
      vol_ratio, close_pct, description, implication
    """
    try:
        # Shared cache — re-uses the 3mo daily fetch from Wyckoff/duration
        hist = get_cached_history(ticker, period="3mo", interval="1d")
        hist = hist.dropna(subset=["Close"]) if not hist.empty else hist
        if len(hist) < 10:
            return {"primary_pattern": "INSUFFICIENT_DATA", "bullish_score": 50,
                    "description": "Not enough data", "vol_ratio": 1.0}

        avg_vol  = float(hist["Volume"].tail(lookback_days).mean())
        last     = hist.iloc[-1]
        bar_rng  = float(last["High"] - last["Low"])
        avg_rng  = float((hist["High"] - hist["Low"]).tail(lookback_days).mean())

        vol_ratio   = float(last["Volume"]) / avg_vol if avg_vol > 0 else 1.0
        range_ratio = bar_rng / avg_rng if avg_rng > 0 else 1.0
        close_pct   = ((float(last["Close"]) - float(last["Low"])) / bar_rng
                       if bar_rng > 0 else 0.5)
        is_up_bar   = float(last["Close"]) >= float(last["Open"])

        patterns     = []
        bullish_score = 50  # neutral

        # ── Breakout confirm: high vol, close near high, above 20-bar high ──
        prior_high = float(hist["High"].tail(lookback_days + 1).iloc[:-1].max())
        if vol_ratio > 1.8 and close_pct > 0.70 and float(last["Close"]) > prior_high:
            patterns.append("BREAKOUT_CONFIRM")
            bullish_score += 30

        # ── Absorption: high vol, narrow range, close mid-bar ───────────────
        if vol_ratio > 2.0 and range_ratio < 0.60 and 0.35 < close_pct < 0.65:
            patterns.append("ABSORPTION")
            bullish_score += 15

        # ── Selling climax: huge down-vol bar but closes upper half ──────────
        if vol_ratio > 2.5 and not is_up_bar and close_pct > 0.60:
            patterns.append("SELLING_CLIMAX")
            bullish_score += 25

        # ── Effort no result: huge vol, tiny range ────────────────────────────
        if vol_ratio > 3.0 and range_ratio < 0.40:
            patterns.append("EFFORT_NO_RESULT")
            # Neutral — need direction confirmation

        # ── Distribution sign: high vol at multi-day high, weak close ────────
        at_highs = float(last["High"]) >= float(hist["High"].tail(10).max()) * 0.99
        if vol_ratio > 1.8 and at_highs and close_pct < 0.40:
            patterns.append("DISTRIBUTION_SIGN")
            bullish_score -= 25

        # ── Volume dry-up on pullback: last 3 bars declining vol + price ─────
        if len(hist) >= 4:
            last3_vol   = hist["Volume"].tail(3).tolist()
            last3_close = hist["Close"].tail(3).tolist()
            vol_decline   = all(last3_vol[i] > last3_vol[i+1]  for i in range(2))
            price_decline = last3_close[-1] < last3_close[0]
            if vol_decline and price_decline and float(last["Volume"]) < avg_vol:
                patterns.append("VOLUME_DRY_UP")
                bullish_score += 12

        if not patterns:
            patterns.append("NORMAL")

        bullish_score = min(100, max(0, bullish_score))
        primary       = patterns[0]

        _descs = {
            "BREAKOUT_CONFIRM":  "High-volume breakout above prior range — institutional participation confirmed",
            "ABSORPTION":        "High volume + narrow range — institutions absorbing overhead supply",
            "SELLING_CLIMAX":    "Huge selling volume but price closed strong — buyers overwhelmed sellers",
            "EFFORT_NO_RESULT":  "Enormous volume, tiny move — supply and demand in balance; watch next bar",
            "DISTRIBUTION_SIGN": "High volume at highs with weak close — institutions may be distributing",
            "VOLUME_DRY_UP":     "Volume drying up on pullback — institutions not selling; healthy correction",
            "NORMAL":            "No unusual institutional volume pattern detected",
        }
        _impl = {
            "BREAKOUT_CONFIRM":  "✅ BULLISH — strong institutional buy signal; momentum entry valid",
            "ABSORPTION":        "✅ BULLISH — supply being removed; move up likely after consolidation",
            "SELLING_CLIMAX":    "✅ BULLISH — exhaustion of sellers; potential reversal point",
            "EFFORT_NO_RESULT":  "⚠️ NEUTRAL — wait for next bar direction to confirm bias",
            "DISTRIBUTION_SIGN": "🛑 BEARISH — smart money selling; avoid new longs",
            "VOLUME_DRY_UP":     "✅ BULLISH — pullback is healthy; continuation likely",
            "NORMAL":            "⚪ NEUTRAL — no edge from volume analysis",
        }

        return {
            "primary_pattern": primary,
            "all_patterns":    patterns,
            "bullish_score":   bullish_score,
            "vol_ratio":       round(vol_ratio, 2),
            "range_ratio":     round(range_ratio, 2),
            "close_pct":       round(close_pct, 2),
            "description":     _descs.get(primary, ""),
            "implication":     _impl.get(primary, ""),
        }
    except Exception:
        return {"primary_pattern": "ERROR", "bullish_score": 50,
                "description": "Analysis failed", "vol_ratio": 1.0}


def get_gamma_walls(ticker, expiry=None):
    """
    Identify gamma wall levels from the options chain open interest.

    Call walls (high OI above spot) = resistance — market makers sell stock as
    price approaches, creating a ceiling.

    Put walls (high OI below spot) = support — market makers buy stock as price
    falls toward the strike, creating a floor.

    Also returns Max Pain (the strike where the most options expire worthless) and
    the gravitational direction it implies for price into expiry.

    Returns dict:
      spot, expiry, call_walls, put_walls,
      nearest_resistance, nearest_support,
      max_pain, max_pain_distance_pct, gravity ('UP'/'DOWN'/'NEUTRAL'),
      is_opex_week, days_to_expiry,
      squeeze_zone (True if dominant OTM call wall is within 5% of spot)
    """
    try:
        analyzer = OptionsChainAnalyzer(ticker)
        exps     = analyzer.expirations()
        spot     = analyzer._get_spot()
        if not exps or spot <= 0:
            return None

        # Use nearest weekly expiry if none specified
        if not expiry:
            today = datetime.utcnow().date()
            for exp in exps:
                try:
                    dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
                    if 0 < dte <= 7:
                        expiry = exp
                        break
                except ValueError:
                    pass
            if not expiry:
                expiry = exps[0]

        chain = analyzer.get_chain(expiry)
        if not chain:
            return None

        calls_df = chain["calls"]
        puts_df  = chain["puts"]

        # Average OI across all strikes
        all_oi  = pd.concat([calls_df["openInterest"].fillna(0),
                              puts_df["openInterest"].fillna(0)])
        avg_oi  = float(all_oi.mean()) if not all_oi.empty else 1.0

        # ── Call walls (above spot — resistance) ──────────────────────────────
        call_walls = []
        for _, row in calls_df.iterrows():
            oi = float(row.get("openInterest") or 0)
            sk = float(row.get("strike")       or 0)
            if oi > avg_oi * 1.8 and sk > spot:
                call_walls.append({
                    "strike":       sk,
                    "oi":           int(oi),
                    "strength":     round(oi / avg_oi, 1),
                    "distance_pct": round((sk - spot) / spot * 100, 2),
                    "type":         "RESISTANCE",
                    "note":         "MMs sell stock as price approaches — ceiling",
                })
        call_walls.sort(key=lambda x: x["strike"])

        # ── Put walls (below spot — support) ──────────────────────────────────
        put_walls = []
        for _, row in puts_df.iterrows():
            oi = float(row.get("openInterest") or 0)
            sk = float(row.get("strike")       or 0)
            if oi > avg_oi * 1.8 and sk < spot:
                put_walls.append({
                    "strike":       sk,
                    "oi":           int(oi),
                    "strength":     round(oi / avg_oi, 1),
                    "distance_pct": round((spot - sk) / spot * 100, 2),
                    "type":         "SUPPORT",
                    "note":         "MMs buy stock as price falls here — floor",
                })
        put_walls.sort(key=lambda x: x["strike"], reverse=True)

        # ── Max pain ──────────────────────────────────────────────────────────
        mp_data        = analyzer.max_pain(expiry)
        max_pain_price = float(mp_data.get("max_pain", spot)) if mp_data else spot
        max_pain_dist  = (spot - max_pain_price) / spot * 100 if spot > 0 else 0.0

        if abs(max_pain_dist) <= 1.0:
            gravity = "NEUTRAL"
        elif max_pain_dist > 0:
            gravity = "DOWN"   # price above max pain → MMs profit from dip
        else:
            gravity = "UP"     # price below max pain → MMs profit from rally

        # ── OpEx status ───────────────────────────────────────────────────────
        today          = datetime.utcnow().date()
        exp_date       = datetime.strptime(expiry, "%Y-%m-%d").date()
        days_to_expiry = max(0, (exp_date - today).days)
        is_opex_week   = days_to_expiry <= 5

        # ── Squeeze zone: nearest call wall within 3% above spot ─────────────
        squeeze_zone = (
            bool(call_walls)
            and call_walls[0]["distance_pct"] <= 3.0
        )

        return {
            "spot":                  round(spot, 2),
            "expiry":                expiry,
            "call_walls":            call_walls[:6],
            "put_walls":             put_walls[:6],
            "nearest_resistance":    call_walls[0]["strike"] if call_walls else None,
            "nearest_support":       put_walls[0]["strike"]  if put_walls else None,
            "max_pain":              round(max_pain_price, 2),
            "max_pain_distance_pct": round(max_pain_dist, 2),
            "gravity":               gravity,
            "is_opex_week":          is_opex_week,
            "days_to_expiry":        days_to_expiry,
            "squeeze_zone":          squeeze_zone,
        }
    except Exception:
        return None


def detect_gamma_squeeze_setup(ticker):
    """
    Detect whether the options market is set up for a gamma squeeze.

    A gamma squeeze occurs when:
    1. Large OTM call open interest builds at strikes just above current price
    2. Market makers must buy shares to delta-hedge as price rises
    3. Buying begets more buying in a self-reinforcing loop

    Checks the nearest 3 expiries.

    Returns dict:
      squeeze_potential ('HIGH' / 'MODERATE' / 'LOW'),
      otm_call_dominance_pct, dominant_strike,
      total_call_oi, total_put_oi, cp_ratio,
      nearest_otm_wall_pct (distance from spot to dominant strike %),
      description, what_happens_if_price_rises
    """
    try:
        analyzer = OptionsChainAnalyzer(ticker)
        exps     = analyzer.expirations()
        spot     = analyzer._get_spot()
        if not exps or spot <= 0:
            return {"squeeze_potential": "LOW",
                    "description": "No options data available"}

        total_call_oi = 0
        total_put_oi  = 0
        otm_call_oi   = 0
        dominant_strike   = None
        max_otm_strike_oi = 0

        for exp in exps[:3]:
            chain = analyzer.get_chain(exp)
            if not chain:
                continue
            calls_df = chain["calls"]
            puts_df  = chain["puts"]

            c_oi = int(calls_df["openInterest"].fillna(0).sum())
            p_oi = int(puts_df["openInterest"].fillna(0).sum())
            total_call_oi += c_oi
            total_put_oi  += p_oi

            # OTM calls: strikes > 1% above spot
            otm_calls = calls_df[calls_df["strike"] > spot * 1.01]
            for _, row in otm_calls.iterrows():
                oi = int(row.get("openInterest") or 0)
                otm_call_oi += oi
                if oi > max_otm_strike_oi:
                    max_otm_strike_oi = oi
                    dominant_strike   = float(row["strike"])

        if total_call_oi == 0 and total_put_oi == 0:
            return {"squeeze_potential": "LOW", "description": "No open interest data"}

        otm_dominance = (otm_call_oi / total_call_oi * 100
                         if total_call_oi > 0 else 0.0)
        cp_ratio      = (total_call_oi / total_put_oi
                         if total_put_oi > 0 else 99.0)
        wall_pct      = ((dominant_strike - spot) / spot * 100
                         if dominant_strike else 99.0)

        if otm_dominance > 60 and cp_ratio > 1.5:
            potential = "HIGH"
            desc      = (f"Heavy OTM call build — {otm_dominance:.0f}% of all calls are OTM. "
                         f"C/P ratio {cp_ratio:.1f}. Dominant strike ${dominant_strike:.0f} "
                         f"({wall_pct:.1f}% above spot). "
                         "MMs must buy more shares if price rises — squeeze risk is real.")
            what_if   = (f"If {ticker} pushes toward ${dominant_strike:.0f}, "
                         "MMs are forced to buy ~100 shares per contract to stay delta-neutral. "
                         "This buying accelerates the move — feedback loop.")
        elif otm_dominance > 40 and cp_ratio > 1.0:
            potential = "MODERATE"
            desc      = (f"Moderate OTM call positioning ({otm_dominance:.0f}%). "
                         f"C/P {cp_ratio:.1f}. Watch for squeeze if catalyst appears.")
            what_if   = ("A strong catalyst could trigger forced MM hedging. "
                         "Not a full squeeze setup but worth monitoring.")
        else:
            potential = "LOW"
            desc      = (f"Balanced options positioning — OTM calls only "
                         f"{otm_dominance:.0f}% of total calls. C/P {cp_ratio:.1f}. "
                         "No significant gamma squeeze pressure.")
            what_if   = "No meaningful forced buying expected from current positioning."

        return {
            "squeeze_potential":       potential,
            "otm_call_dominance_pct":  round(otm_dominance, 1),
            "dominant_strike":         dominant_strike,
            "total_call_oi":           total_call_oi,
            "total_put_oi":            total_put_oi,
            "cp_ratio":                round(cp_ratio, 2),
            "nearest_otm_wall_pct":    round(wall_pct, 2),
            "description":             desc,
            "what_happens_if_price_rises": what_if,
        }
    except Exception:
        return {"squeeze_potential": "LOW",
                "description": "Analysis failed"}


# ══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL RISK ENGINE
#   The five things that separate top 0.0001% trading desks from everyone else:
#     1. Multi-timeframe confirmation (1h × daily × weekly alignment)
#     2. Portfolio-level drawdown circuit breaker (auto-pause on losses)
#     3. Correlation-aware sizing (avoid stacking into same trade)
#     4. Sector concentration limits (no over-exposure)
#     5. Master Score — ONE unified 0-100 number that gates every trade
# ══════════════════════════════════════════════════════════════════════════════

def confirm_multi_timeframe(ticker, bias="bullish"):
    """
    Check trend alignment across three timeframes — institutional gold standard.
    A signal is high-conviction only when 1H, Daily, and Weekly all agree.

      1-Hour  trend  — price vs 50 EMA on 60-min bars (intraday momentum)
      Daily   trend  — price vs 50 SMA on daily bars (swing direction)
      Weekly  trend  — price vs 20 SMA on weekly bars (primary trend)

    Returns dict:
      aligned          — int 0–3 (how many timeframes agree with bias)
      score            — int 0-100 (33 per aligned TF + bonus for full stack)
      grade            — "A+" / "A" / "B" / "C" / "F"
      htf_trend        — categorical 'STRONG'/'NEUTRAL'/'WEAK'
      timeframes       — per-TF dict {tf: {trend, aligned, price, sma}}
      description      — plain-English summary
    """
    is_bullish = str(bias).lower() != "bearish"
    timeframes = {}
    aligned    = 0

    # ── 1-Hour: 50 EMA on 60-min bars over 60 days ────────────────────────────
    try:
        h1 = yf.download(ticker, period="60d", interval="60m",
                         progress=False, auto_adjust=True)
        if isinstance(h1.columns, pd.MultiIndex):
            h1.columns = h1.columns.get_level_values(0)
        h1 = h1.dropna(subset=["Close"])
        if len(h1) >= 50:
            ema50 = h1["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
            spot  = float(h1["Close"].iloc[-1])
            is_up = spot > float(ema50)
            ok    = (is_up and is_bullish) or (not is_up and not is_bullish)
            timeframes["1H"] = {
                "trend":   "UP" if is_up else "DOWN",
                "aligned": ok, "price": round(spot, 2),
                "sma":     round(float(ema50), 2),
                "label":   "1H: " + ("✓ aligned" if ok else "✗ against"),
            }
            if ok:
                aligned += 1
        else:
            timeframes["1H"] = {"trend": "?", "aligned": False, "label": "1H: no data"}
    except Exception:
        timeframes["1H"] = {"trend": "?", "aligned": False, "label": "1H: error"}

    # ── Daily: 50 SMA on daily bars over 6 months ─────────────────────────────
    try:
        hd = yf.download(ticker, period="6mo", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(hd.columns, pd.MultiIndex):
            hd.columns = hd.columns.get_level_values(0)
        hd = hd.dropna(subset=["Close"])
        if len(hd) >= 50:
            sma50d = float(hd["Close"].tail(50).mean())
            spotd  = float(hd["Close"].iloc[-1])
            is_up  = spotd > sma50d
            ok     = (is_up and is_bullish) or (not is_up and not is_bullish)
            timeframes["Daily"] = {
                "trend": "UP" if is_up else "DOWN",
                "aligned": ok, "price": round(spotd, 2),
                "sma":     round(sma50d, 2),
                "label":   "D: " + ("✓ aligned" if ok else "✗ against"),
            }
            if ok:
                aligned += 1
        else:
            timeframes["Daily"] = {"trend": "?", "aligned": False, "label": "D: no data"}
    except Exception:
        timeframes["Daily"] = {"trend": "?", "aligned": False, "label": "D: error"}

    # ── Weekly: 20 SMA on weekly bars over 2 years ────────────────────────────
    try:
        hw = yf.download(ticker, period="2y", interval="1wk",
                         progress=False, auto_adjust=True)
        if isinstance(hw.columns, pd.MultiIndex):
            hw.columns = hw.columns.get_level_values(0)
        hw = hw.dropna(subset=["Close"])
        if len(hw) >= 20:
            sma20w = float(hw["Close"].tail(20).mean())
            spotw  = float(hw["Close"].iloc[-1])
            is_up  = spotw > sma20w
            ok     = (is_up and is_bullish) or (not is_up and not is_bullish)
            timeframes["Weekly"] = {
                "trend": "UP" if is_up else "DOWN",
                "aligned": ok, "price": round(spotw, 2),
                "sma":     round(sma20w, 2),
                "label":   "W: " + ("✓ aligned" if ok else "✗ against"),
            }
            if ok:
                aligned += 1
        else:
            timeframes["Weekly"] = {"trend": "?", "aligned": False, "label": "W: no data"}
    except Exception:
        timeframes["Weekly"] = {"trend": "?", "aligned": False, "label": "W: error"}

    # ── Score & grade ─────────────────────────────────────────────────────────
    base_score = aligned * 33   # 0/33/66/99
    bonus      = 1 if aligned == 3 else 0
    score      = min(100, base_score + bonus)

    if aligned == 3:
        grade, htf = "A+", "STRONG"
        desc       = "All three timeframes aligned — institutional-grade trend signal"
    elif aligned == 2:
        grade, htf = "B",  "NEUTRAL"
        desc       = "Two of three timeframes aligned — proceed with reduced size"
    elif aligned == 1:
        grade, htf = "C",  "WEAK"
        desc       = "Only one timeframe agrees — counter-trend trade, high risk"
    else:
        grade, htf = "F",  "WEAK"
        desc       = "No timeframe alignment — do not trade"

    return {
        "aligned":     aligned,
        "score":       score,
        "grade":       grade,
        "htf_trend":   htf,
        "timeframes":  timeframes,
        "description": desc,
        "bias":        "bullish" if is_bullish else "bearish",
    }


def correlation_with_holdings(ticker, held_tickers, lookback_days=30):
    """
    Compute the maximum pairwise return correlation between *ticker* and the
    list of currently-held tickers.  Used to avoid stacking into the same trade.

    Two stocks with > 0.85 correlation are basically the same position — buying
    both is concentrating risk, not diversifying.

    Returns dict:
      max_corr        — highest correlation found (−1 to +1)
      avg_corr        — average correlation across all holdings
      most_correlated — ticker of the highest-correlated holding
      risk_level      — 'LOW' (<0.5) / 'MEDIUM' (0.5-0.75) / 'HIGH' (>0.75)
      size_multiplier — 1.0 / 0.7 / 0.3 (downsize based on overlap)
      advice          — plain-English recommendation
    """
    if not held_tickers:
        return {"max_corr": 0.0, "avg_corr": 0.0, "most_correlated": None,
                "risk_level": "LOW", "size_multiplier": 1.0,
                "advice": "No existing holdings — no correlation risk."}

    held_unique = sorted({str(t).upper() for t in held_tickers
                          if str(t).upper() != str(ticker).upper()})
    if not held_unique:
        return {"max_corr": 0.0, "avg_corr": 0.0, "most_correlated": None,
                "risk_level": "LOW", "size_multiplier": 1.0,
                "advice": "Only holding the same ticker — no correlation calc needed."}

    try:
        tickers_all = [str(ticker).upper()] + held_unique
        df = yf.download(tickers_all, period=f"{lookback_days * 2}d",
                         interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return {"max_corr": 0.0, "avg_corr": 0.0, "most_correlated": None,
                    "risk_level": "LOW", "size_multiplier": 1.0,
                    "advice": "No price data — correlation skipped."}

        if isinstance(df.columns, pd.MultiIndex):
            close = df.xs("Close", axis=1, level=0)
        else:
            close = df[["Close"]].rename(columns={"Close": tickers_all[0]})

        rets = close.pct_change().dropna().tail(lookback_days)
        if len(rets) < 10:
            return {"max_corr": 0.0, "avg_corr": 0.0, "most_correlated": None,
                    "risk_level": "LOW", "size_multiplier": 1.0,
                    "advice": "Not enough history — correlation skipped."}

        tk_up = str(ticker).upper()
        if tk_up not in rets.columns:
            return {"max_corr": 0.0, "avg_corr": 0.0, "most_correlated": None,
                    "risk_level": "LOW", "size_multiplier": 1.0,
                    "advice": "Ticker not in returns — correlation skipped."}

        corrs = {}
        for t in held_unique:
            if t in rets.columns:
                c = rets[tk_up].corr(rets[t])
                if not pd.isna(c):
                    corrs[t] = float(c)

        if not corrs:
            return {"max_corr": 0.0, "avg_corr": 0.0, "most_correlated": None,
                    "risk_level": "LOW", "size_multiplier": 1.0,
                    "advice": "No valid pairs — correlation skipped."}

        max_t       = max(corrs, key=lambda k: corrs[k])
        max_c       = corrs[max_t]
        avg_c       = sum(corrs.values()) / len(corrs)

        if max_c < 0.5:
            risk, mult = "LOW", 1.0
            advice = f"Low correlation (max {max_c:.2f} with {max_t}) — full size OK"
        elif max_c < 0.75:
            risk, mult = "MEDIUM", 0.7
            advice = f"Moderate overlap with {max_t} (corr {max_c:.2f}) — reduce size 30 %"
        else:
            risk, mult = "HIGH", 0.3
            advice = f"High overlap with {max_t} (corr {max_c:.2f}) — same trade, cut size 70 %"

        return {
            "max_corr":         round(max_c, 3),
            "avg_corr":         round(avg_c, 3),
            "most_correlated":  max_t,
            "risk_level":       risk,
            "size_multiplier":  mult,
            "advice":           advice,
            "all_correlations": {k: round(v, 3) for k, v in corrs.items()},
        }
    except Exception:
        return {"max_corr": 0.0, "avg_corr": 0.0, "most_correlated": None,
                "risk_level": "LOW", "size_multiplier": 1.0,
                "advice": "Correlation calc failed (data error) — assuming low."}


def get_portfolio_risk_status(paper_engine):
    """
    Portfolio-level circuit breaker — measures drawdown and concentration.
    The reason top funds outlast everyone else is they STOP TRADING when down.

    Inputs:
      paper_engine — a PaperTradingEngine instance

    Returns dict:
      drawdown_pct      — current drawdown from peak equity (positive number)
      peak_equity       — highest equity recorded
      current_equity    — cash + invested value
      sector_exposure   — {sector: %_of_capital}
      max_sector_pct    — % in most-concentrated sector
      risk_level        — 'NORMAL' / 'CAUTION' / 'HALT' / 'EMERGENCY_HALT'
      can_open_new      — bool — whether the bot should be opening new trades
      size_multiplier   — 1.0 / 0.5 / 0.0 (apply to new entries)
      reason            — why the gate is open or closed
    """
    try:
        summary = paper_engine.get_summary()
        cash    = float(summary.get("available_cash", 0))
        real    = float(summary.get("realized_pnl",   0))
        start   = float(summary.get("starting_capital", 1000))
    except Exception:
        return {"risk_level": "NORMAL", "can_open_new": True,
                "size_multiplier": 1.0, "drawdown_pct": 0.0,
                "reason": "Could not load portfolio state."}

    # ── Current equity = cash + invested (mark to market via gross_invested) ──
    try:
        positions     = paper_engine.open_positions
        invested      = sum(float(p.get("gross_invested") or 0) for p in positions)
    except Exception:
        positions     = []
        invested      = 0.0

    current_equity = cash + invested
    # Peak equity = starting capital + cumulative realized P&L if positive;
    # otherwise just the starting capital — drawdown is measured from there.
    peak_equity    = max(start, start + max(0, real))

    drawdown_pct   = (peak_equity - current_equity) / peak_equity * 100 \
                     if peak_equity > 0 else 0.0
    drawdown_pct   = max(0.0, round(drawdown_pct, 2))

    # ── Sector exposure ───────────────────────────────────────────────────────
    sector_exp = {}
    if invested > 0 and current_equity > 0:
        for p in positions:
            sec = str(p.get("sector") or "Unknown") or "Unknown"
            sector_exp[sec] = sector_exp.get(sec, 0.0) + float(p.get("gross_invested") or 0)
        sector_exp = {k: round(v / current_equity * 100, 1)
                      for k, v in sector_exp.items()}
    max_sector_pct = max(sector_exp.values()) if sector_exp else 0.0

    # ── Decision gates ────────────────────────────────────────────────────────
    if drawdown_pct >= 25:
        risk_level     = "EMERGENCY_HALT"
        can_open       = False
        size_mult      = 0.0
        reason         = (f"🚨 EMERGENCY: drawdown {drawdown_pct:.1f}% ≥ 25 %. "
                          "All new entries halted. Close losing positions, review strategy.")
    elif drawdown_pct >= 15:
        risk_level     = "HALT"
        can_open       = False
        size_mult      = 0.0
        reason         = (f"🛑 HALT: drawdown {drawdown_pct:.1f}% ≥ 15 %. "
                          "No new positions until equity recovers above peak − 10 %.")
    elif drawdown_pct >= 8:
        risk_level     = "CAUTION"
        can_open       = True
        size_mult      = 0.5
        reason         = (f"⚠️ CAUTION: drawdown {drawdown_pct:.1f}% — "
                          "reduce new position sizes by 50 %.")
    elif max_sector_pct > 40:
        risk_level     = "CAUTION"
        can_open       = True
        size_mult      = 0.7
        reason         = (f"⚠️ Sector concentration {max_sector_pct:.0f}% — "
                          "diversify; reduce new same-sector entries.")
    else:
        risk_level     = "NORMAL"
        can_open       = True
        size_mult      = 1.0
        reason         = (f"✓ Normal operating range. Drawdown {drawdown_pct:.1f} %, "
                          f"max sector exposure {max_sector_pct:.0f} %.")

    return {
        "drawdown_pct":    drawdown_pct,
        "peak_equity":     round(peak_equity, 2),
        "current_equity":  round(current_equity, 2),
        "sector_exposure": sector_exp,
        "max_sector_pct":  round(max_sector_pct, 1),
        "risk_level":      risk_level,
        "can_open_new":    can_open,
        "size_multiplier": size_mult,
        "reason":          reason,
    }


def compute_master_score(ticker, expiry, bias, conn,
                         vix_data=None, market_regime=None,
                         entry_quality=None, wyckoff=None,
                         inst_volume=None, multi_tf=None,
                         skip_slow_checks=False):
    """
    THE one number that decides every trade.

    Weighted composite of every smart-money signal the bot computes — designed
    so that *one* clean threshold replaces the scattered checks in monitor.py
    and gives the user a clear "should I take this trade" answer.

    Component weights (sum = 100):
      Breakout scan score ........ 15
      Entry quality .............. 15
      Wyckoff phase .............. 15
      Institutional volume ....... 10
      Multi-timeframe alignment .. 15
      Market regime (SPY) ........ 15
      VIX environment ............ 10
      Earnings safety ............ 5

    Returns dict:
      score           — 0-100 final composite
      grade           — A+ / A / B / C / D / F
      decision        — 'BUY' / 'WATCH' / 'SKIP'
      size_multiplier — 1.2 / 1.0 / 0.6 / 0.3 / 0.0
      components      — per-signal breakdown (max points, earned points, label)
      summary         — plain-English headline
    """
    components = {}
    total      = 0

    # ── 1. Breakout scan (15 pts) ─────────────────────────────────────────────
    try:
        row = conn.execute(
            "SELECT explosive_score, breakout_prob FROM calls "
            "WHERE ticker=? ORDER BY scan_timestamp DESC LIMIT 1",
            (str(ticker).upper(),)
        ).fetchone()
        if row:
            exp_score = float(row.get("explosive_score") or row[0] or 0)
            prob_     = float(row.get("breakout_prob")   or row[1] or 0)
            blended   = (exp_score + prob_) / 2
            pts       = round(blended / 100 * 15, 1)
            label     = f"Breakout scan {exp_score:.0f}/100 × prob {prob_:.0f}%"
        else:
            pts, label = 5, "Not in latest scan — neutral 5/15"
    except Exception:
        pts, label = 5, "Scan unavailable — neutral 5/15"
    total += pts
    components["breakout_scan"] = {"max": 15, "pts": pts, "label": label}

    # ── 2. Entry quality (15 pts) ─────────────────────────────────────────────
    if entry_quality is None and not skip_slow_checks:
        try:
            entry_quality = get_entry_quality(ticker, bias)
        except Exception:
            entry_quality = None
    if entry_quality:
        eq_s = entry_quality.get("score", 50)
        pts  = round(eq_s / 100 * 15, 1)
        label = f"Entry quality {eq_s}/100 ({entry_quality.get('grade','?')})"
    else:
        pts, label = 7.5, "Entry quality unknown — neutral"
    total += pts
    components["entry_quality"] = {"max": 15, "pts": pts, "label": label}

    # ── 3. Wyckoff phase (15 pts) ─────────────────────────────────────────────
    if wyckoff is None and not skip_slow_checks:
        try:
            wyckoff = detect_wyckoff_phase(ticker)
        except Exception:
            wyckoff = None
    _wy_map = {
        "SPRING":         15, "ACCUMULATION":   12, "MARKUP":         11,
        "REACCUMULATION": 9,  "UNDEFINED":      6,  "INSUFFICIENT_DATA": 6,
        "DISTRIBUTION":   2,  "MARKDOWN":       0,  "ERROR":          6,
    }
    if wyckoff:
        wp  = wyckoff.get("phase", "UNDEFINED")
        pts = _wy_map.get(wp, 6)
        label = f"Wyckoff: {wp.replace('_',' ')}"
    else:
        pts, label = 6, "Wyckoff unknown — neutral"
    total += pts
    components["wyckoff"] = {"max": 15, "pts": pts, "label": label}

    # ── 4. Institutional volume (10 pts) ──────────────────────────────────────
    if inst_volume is None and not skip_slow_checks:
        try:
            inst_volume = analyze_institutional_volume(ticker)
        except Exception:
            inst_volume = None
    if inst_volume:
        bs = inst_volume.get("bullish_score", 50)
        # Map for bearish bias: invert score
        if str(bias).lower() == "bearish":
            bs = 100 - bs
        pts = round(bs / 100 * 10, 1)
        label = f"Vol pattern: {inst_volume.get('primary_pattern','NORMAL').replace('_',' ')}"
    else:
        pts, label = 5, "Volume pattern unknown — neutral"
    total += pts
    components["inst_volume"] = {"max": 10, "pts": pts, "label": label}

    # ── 5. Multi-timeframe alignment (15 pts) ─────────────────────────────────
    if multi_tf is None and not skip_slow_checks:
        try:
            multi_tf = confirm_multi_timeframe(ticker, bias)
        except Exception:
            multi_tf = None
    if multi_tf:
        a   = multi_tf.get("aligned", 0)
        pts = a * 5    # 0/5/10/15
        label = f"Multi-TF: {a}/3 timeframes aligned ({multi_tf.get('grade','?')})"
    else:
        pts, label = 7, "Multi-TF unknown — neutral"
    total += pts
    components["multi_tf"] = {"max": 15, "pts": pts, "label": label}

    # ── 6. Market regime (15 pts) ─────────────────────────────────────────────
    _reg_map = {
        "STRONG BULL": 15, "BULL": 12, "NEUTRAL": 8,
        "RECOVERING": 5,   "BEAR": 0,  "UNKNOWN": 7,
    }
    if market_regime:
        reg = market_regime.get("regime", "UNKNOWN")
        pts = _reg_map.get(reg, 7)
        label = f"SPY regime: {market_regime.get('label', reg)}"
    else:
        pts, label = 7, "Market regime unknown — neutral"
    total += pts
    components["market_regime"] = {"max": 15, "pts": pts, "label": label}

    # ── 7. VIX environment (10 pts) ───────────────────────────────────────────
    vix = vix_data or get_vix_level()
    if vix:
        regime = vix.get("regime", "NORMAL")
        pts    = {"LOW": 10, "NORMAL": 8, "ELEVATED": 4, "EXTREME": 0}.get(regime, 5)
        label  = f"VIX {vix.get('vix',0):.1f} — {regime}"
    else:
        pts, label = 5, "VIX unknown — neutral"
    total += pts
    components["vix"] = {"max": 10, "pts": pts, "label": label}

    # ── 8. Earnings safety (5 pts) ────────────────────────────────────────────
    try:
        if expiry:
            ew = check_earnings_in_window(ticker, expiry)
            if not ew.get("has_earnings"):
                pts, label = 5, "No earnings in window"
            else:
                days = ew.get("days_away", 0)
                pts   = 2 if days >= 5 else 0
                label = f"Earnings in {days}d — risky"
        else:
            pts, label = 4, "No expiry checked"
    except Exception:
        pts, label = 3, "Earnings check failed"
    total += pts
    components["earnings_safety"] = {"max": 5, "pts": pts, "label": label}

    # ── 9. TradingView TA signal (multiplier, optional) ──────────────────────
    # Not a fixed-point component — instead it modulates the final score by
    # up to ±15 % based on TradingView's own technical analysis verdict.
    # If tradingview-ta isn't installed or returns no data, multiplier = 1.0.
    tv_mult = 1.0
    tv_sig  = {}
    if not skip_slow_checks:
        try:
            from data_providers import get_tradingview_signal, get_tradingview_multiplier
            tv_sig  = get_tradingview_signal(ticker, interval="1d") or {}
            tv_mult = get_tradingview_multiplier(ticker, interval="1d")
            # Invert multiplier if the trade is bearish — TV "STRONG_BUY" is
            # a NEGATIVE for a short / put trade.
            if str(bias).lower() == "bearish" and tv_mult != 1.0:
                tv_mult = 2.0 - tv_mult   # 1.10 → 0.90, 0.85 → 1.15, etc.
        except Exception:
            tv_mult = 1.0

    # ── 10. News impact (multiplier, optional) ───────────────────────────────
    # Aggregates news events affecting *ticker* in the last 24 hours via
    # NewsAgent.get_news_impact().  Negative news = score penalty;
    # positive news = score boost.  Invert for bearish trades.
    news_mult = 1.0
    news_imp  = {}
    if not skip_slow_checks:
        try:
            from news_agent import NewsAgent
            _na = NewsAgent(conn, ai_analyst=None)   # AI not needed for read-only impact
            news_imp = _na.get_news_impact(ticker, hours=24) or {}
            news_mult = float(news_imp.get("score_modifier", 1.0) or 1.0)
            if str(bias).lower() == "bearish" and news_mult != 1.0:
                news_mult = 2.0 - news_mult
        except Exception:
            news_mult = 1.0

    # ── 11. Predicted duration (multiplier, optional) ────────────────────────
    # Favours fast-resolving setups (capital cycles quicker) and penalises
    # slow ones (capital tied up for weeks).  Uses the latest call entry +
    # target from the calls table as the prediction inputs.
    duration_mult = 1.0
    duration_info = {}
    if not skip_slow_checks:
        try:
            _trow = conn.execute(
                "SELECT entry_price, target_price, pattern_detected "
                "FROM calls WHERE ticker=? AND result='PENDING' "
                "ORDER BY scan_timestamp DESC LIMIT 1",
                (str(ticker).upper(),)
            ).fetchone()
            if _trow:
                _t_ep  = float(_row_get(_trow, "entry_price",     0) or 0)
                _t_tp  = float(_row_get(_trow, "target_price",    0) or 0)
                _t_pat = str(_row_get(_trow, "pattern_detected", "") or "")
                if _t_ep > 0 and _t_tp > 0:
                    _pred = predict_duration_to_target(
                        ticker, _t_ep, _t_tp,
                        pattern=_t_pat, market_regime=market_regime,
                    )
                    if _pred:
                        _pd_val = int(_pred.get("predicted_days", 0) or 0)
                        # Reward fast setups (faster capital turnover);
                        # penalise slow ones (capital tied up = opportunity cost)
                        if   _pd_val <= 0:  duration_mult = 1.00
                        elif _pd_val <= 5:  duration_mult = 1.05    # fast 🚀
                        elif _pd_val <= 10: duration_mult = 1.00    # standard
                        elif _pd_val <= 15: duration_mult = 0.97    # slow
                        elif _pd_val <= 20: duration_mult = 0.93
                        else:                duration_mult = 0.88   # very slow
                        duration_info = _pred
        except Exception:
            duration_mult = 1.0
    if duration_info:
        components["duration"] = {
            "max":         "×",
            "pts":         f"{duration_mult:.2f}",
            "label":       (f"Predicted {duration_info.get('predicted_days', '?')}d "
                            f"(range {duration_info.get('min_days_est', '?')}-"
                            f"{duration_info.get('max_days_est', '?')}d, "
                            f"{duration_info.get('confidence', '?')}) → "
                            f"{duration_mult:.2f}×"),
            "multiplier":  duration_mult,
            "raw":         duration_info,
        }
    else:
        components["duration"] = {
            "max":   "×",
            "pts":   "1.00",
            "label": "Duration prediction unavailable (1.00×)",
            "multiplier": 1.0,
        }
    if news_imp.get("n_events"):
        components["news_impact"] = {
            "max":        "×",
            "pts":        f"{news_mult:.2f}",
            "label":      (f"News: {news_imp['n_events']} event(s), "
                            f"net {news_imp['net_sentiment']:+.2f}, "
                            f"{news_imp['dominant_category']} → {news_mult:.2f}×"),
            "multiplier": news_mult,
            "raw":        news_imp,
        }
    else:
        components["news_impact"] = {
            "max":        "×",
            "pts":        "1.00",
            "label":      "No recent news for this ticker (1.00×)",
            "multiplier": 1.0,
        }
    if tv_sig:
        components["tradingview_ta"] = {
            "max":   "×",
            "pts":   f"{tv_mult:.2f}",
            "label": (
                f"TradingView {tv_sig.get('recommendation','?')} "
                f"({tv_sig.get('buy_count',0)}B/{tv_sig.get('sell_count',0)}S/"
                f"{tv_sig.get('neutral_count',0)}N) → {tv_mult:.2f}×"
            ),
            "multiplier": tv_mult,
            "raw":        tv_sig,
        }
    else:
        components["tradingview_ta"] = {
            "max":   "×",
            "pts":   "1.00",
            "label": "TradingView TA unavailable — no effect (1.00×)",
            "multiplier": 1.0,
        }

    # ── 12. Squeeze multiplier (optional bonus for explosive setups) ────────
    # Compresses long-form research into one multiplier:
    #   EXPLOSIVE → 1.15× (rare 2-10x setups get a major boost)
    #   HIGH      → 1.08×
    #   MODERATE  → 1.03×
    #   LOW       → 1.00×
    # Always 1.00× for bearish trades — squeezes only work on the upside.
    squeeze_mult = 1.0
    squeeze_data = {}
    if not skip_slow_checks and str(bias).lower() != "bearish":
        try:
            squeeze_data = SqueezeScanner(ticker, conn=conn).scan() or {}
            sq_cat = squeeze_data.get("category", "LOW")
            squeeze_mult = {
                "EXPLOSIVE": 1.15,
                "HIGH":      1.08,
                "MODERATE":  1.03,
                "LOW":       1.00,
                "NONE":      1.00,
            }.get(sq_cat, 1.0)
        except Exception:
            squeeze_mult = 1.0
    if squeeze_data and squeeze_data.get("squeeze_score", 0) > 0:
        components["squeeze"] = {
            "max":         "×",
            "pts":         f"{squeeze_mult:.2f}",
            "label":       (f"Squeeze: {squeeze_data.get('category', 'LOW')} "
                            f"({squeeze_data.get('squeeze_score', 0)}/100) → {squeeze_mult:.2f}×"),
            "multiplier":  squeeze_mult,
            "raw":         squeeze_data,
        }
    else:
        components["squeeze"] = {
            "max":   "×",
            "pts":   "1.00",
            "label": "Squeeze data unavailable (1.00×)",
            "multiplier": 1.0,
        }

    # ── Final score: base × TV × news × duration × squeeze, clamped 0-100 ───
    raw_score = max(0, total)
    score = round(min(100, raw_score * tv_mult * news_mult *
                       duration_mult * squeeze_mult), 1)

    # ── Grade + decision ──────────────────────────────────────────────────────
    if score >= 85:
        grade, decision, mult = "A+", "BUY", 1.2
        summary = "EXCEPTIONAL setup — institutional-grade alignment. Size up."
    elif score >= 75:
        grade, decision, mult = "A",  "BUY", 1.0
        summary = "Strong setup — multiple signals confirm. Standard size."
    elif score >= 65:
        grade, decision, mult = "B",  "BUY", 0.6
        summary = "Decent setup — proceed with reduced size."
    elif score >= 50:
        grade, decision, mult = "C",  "WATCH", 0.3
        summary = "Marginal setup — watch only; paper trade size."
    elif score >= 35:
        grade, decision, mult = "D",  "SKIP", 0.0
        summary = "Weak setup — skip this trade."
    else:
        grade, decision, mult = "F",  "SKIP", 0.0
        summary = "Failed setup — do not trade."

    return {
        "score":               score,
        "grade":               grade,
        "decision":            decision,
        "size_multiplier":     mult,
        "components":          components,
        "summary":             summary,
        "tv_signal":           tv_sig,
        "tv_multiplier":       tv_mult,
        "news_impact":         news_imp,
        "news_multiplier":     news_mult,
        "duration_prediction": duration_info,
        "duration_multiplier": duration_mult,
        "squeeze_data":        squeeze_data,
        "squeeze_multiplier":  squeeze_mult,
        "base_score":          round(raw_score, 1),
    }


def get_market_context_full():
    """
    Single call that returns everything needed for the risk dashboard.
    Bundles VIX, Fear & Greed, market regime, sector rotation, economic events.
    Returns dict (any sub-key may be None if its call failed).
    """
    ctx = {}
    try:
        ctx["vix"] = get_vix_level()
    except Exception:
        ctx["vix"] = None
    try:
        ctx["fg"] = get_fear_greed()
    except Exception:
        ctx["fg"] = None
    try:
        spy = yf.download("SPY", period="1y", interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        ctx["regime"] = MarketRegimeDetector.detect(spy)
    except Exception:
        ctx["regime"] = None
    try:
        ctx["sectors"] = SectorRotationDetector.get()
    except Exception:
        ctx["sectors"] = None
    try:
        ctx["events"] = get_economic_events(7)
    except Exception:
        ctx["events"] = []
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# DURATION PREDICTION ENGINE
#   Predicts how many trading days a setup should take to reach its target,
#   based on ATR-implied daily move speed × pattern type × market regime.
#   Used by the Dashboard "Day #" column to show actual vs expected progress.
# ══════════════════════════════════════════════════════════════════════════════

# Pattern → speed multiplier.  Lower = faster setups, higher = slower.
# Derived from historical observation of how each pattern typically resolves.
_PATTERN_DURATION_MULT = {
    "high and tight flag":  0.5,
    "high tight flag":      0.5,
    "flag":                 0.7,
    "pennant":              0.7,
    "breakout":             0.6,
    "gap and go":           0.4,
    "vcp":                  0.8,
    "ascending triangle":   0.9,
    "triangle":             1.0,
    "symmetrical triangle": 1.0,
    "wedge":                1.2,
    "falling wedge":        1.2,
    "rising wedge":         1.2,
    "double bottom":        1.3,
    "double top":           1.3,
    "cup and handle":       1.5,
    "cup":                  1.3,
    "inverse head shoulders": 1.6,
    "head and shoulders":   1.6,
    "rectangle":            1.4,
    "channel":              1.2,
    "no pattern":           1.0,
}

# Market regime → duration scaling.  Bull markets compress, bear stretches.
_REGIME_DURATION_MULT = {
    "STRONG BULL":  0.70,
    "BULL":         0.85,
    "NEUTRAL":      1.00,
    "RECOVERING":   1.20,
    "BEAR":         1.50,
    "UNKNOWN":      1.00,
}


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY CACHE — share yfinance fetches across all engines within a cycle
#   Critical optimization: Wyckoff, multi-TF, institutional volume, squeeze
#   scanner, duration predictor all need daily bars.  Without this cache they
#   each fetch independently — 5+ identical yfinance calls per ticker per cycle.
# ══════════════════════════════════════════════════════════════════════════════

import time as _time_cache

_HIST_CACHE: dict = {}     # key (ticker, period, interval) → (timestamp, dataframe)
_HIST_CACHE_TTL = 300       # 5 min — fresh enough for intra-cycle reuse


def get_cached_history(ticker: str, period: str = "3mo",
                        interval: str = "1d"):
    """Return cached yfinance OHLCV or fetch + cache.  Drop-in replacement for
    yf.download(ticker, period=..., interval=...) inside any engine.

    Eliminates the ~5x duplicate fetches per ticker per monitor cycle that
    were happening before.  TTL is 5 minutes — short enough for live trading,
    long enough to amortize across all engines.
    """
    key = (str(ticker).upper(), str(period), str(interval))
    now = _time_cache.time()
    cached = _HIST_CACHE.get(key)
    if cached and (now - cached[0]) < _HIST_CACHE_TTL:
        return cached[1]
    try:
        df = yf.download(key[0], period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        _HIST_CACHE[key] = (now, df)
        return df
    except Exception:
        return pd.DataFrame()


def _hist_cache_stats() -> dict:
    """Diagnostic stats for the history cache."""
    return {"entries": len(_HIST_CACHE), "ttl_seconds": _HIST_CACHE_TTL}


def _hist_cache_clear():
    """Purge the cache — for tests or explicit refresh."""
    global _HIST_CACHE
    _HIST_CACHE = {}


# ══════════════════════════════════════════════════════════════════════════════
# BOLLINGER SQUEEZE DETECTOR
#   When BB width compresses to historical bottom 15%, a violent move usually
#   follows.  Used by both SqueezeScanner and duration prediction.
# ══════════════════════════════════════════════════════════════════════════════

def detect_bollinger_squeeze(ticker: str, period: int = 20,
                              lookback_days: int = 180) -> dict:
    """Detect Bollinger Band squeeze.  Returns:
       bb_width_pct       — current BB width as % of price
       pct_rank           — current width's percentile in lookback window (0 = tightest)
       is_squeezed        — True when pct_rank ≤ 15
       squeeze_strength   — 'EXTREME' / 'STRONG' / 'MODERATE' / 'NONE'
       days_in_squeeze    — consecutive days at pct_rank ≤ 25
    Returns {} on insufficient data.
    """
    try:
        hist = get_cached_history(ticker, period="1y", interval="1d")
        hist = hist.dropna(subset=["Close"])
        if len(hist) < period + 30:
            return {}

        close = hist["Close"]
        sma   = close.rolling(period).mean()
        std   = close.rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        bb_width = (upper - lower) / sma * 100
        bb_width = bb_width.dropna().tail(lookback_days)

        current   = float(bb_width.iloc[-1])
        pct_rank  = float(bb_width.rank(pct=True).iloc[-1]) * 100
        is_squeezed = pct_rank <= 15

        # Consecutive days in squeeze (pct_rank ≤ 25 from end)
        rank_series = bb_width.rank(pct=True) * 100
        days_in_squeeze = 0
        for v in rank_series.iloc[::-1]:
            if v <= 25:
                days_in_squeeze += 1
            else:
                break

        if pct_rank <= 5:
            strength = "EXTREME"
        elif pct_rank <= 15:
            strength = "STRONG"
        elif pct_rank <= 25:
            strength = "MODERATE"
        else:
            strength = "NONE"

        return {
            "bb_width_pct":     round(current, 2),
            "pct_rank":         round(pct_rank, 1),
            "is_squeezed":      is_squeezed,
            "squeeze_strength": strength,
            "days_in_squeeze":  int(days_in_squeeze),
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SQUEEZE SCANNER — finds 2-10x potential setups
#   Combines: short interest, float size, days-to-cover, BB squeeze,
#   volume velocity, gamma exposure proxy.  Single 0-100 score with
#   actionable category.
# ══════════════════════════════════════════════════════════════════════════════

class SqueezeScanner:
    """Hunts for explosive 2-10x squeeze candidates using free yfinance data.

    Component scoring (sums to 100):
      Short Interest Score   30   short% of float, very explosive >20%
      Float Tightness        20   smaller float = bigger moves
      Days to Cover          15   high days-to-cover = harder to unwind
      BB Squeeze             15   compressed volatility ready to release
      Volume Velocity        10   accelerating volume = early entry
      Catalyst Proximity     10   recent news = trigger
    """

    # Tier thresholds for category labelling
    _TIER_EXPLOSIVE  = 80   # 2-10x potential, RARE
    _TIER_HIGH       = 65   # 50-100% upside reasonable
    _TIER_MODERATE   = 50   # 20-50% upside

    def __init__(self, ticker: str, conn=None):
        self.ticker = str(ticker).upper()
        self.conn   = conn

    def scan(self) -> dict:
        """Run full squeeze analysis.  Returns rich dict with score + components."""
        out = {
            "ticker":          self.ticker,
            "squeeze_score":   0,
            "category":        "NONE",
            "components":      {},
            "shortable_float_pct":  None,
            "days_to_cover":   None,
            "float_shares":    None,
            "avg_volume":      None,
            "bb_squeeze":      {},
            "recommendation":  "",
            "upside_estimate": "",
        }

        # ── Pull yfinance fundamentals (.info has short interest data) ──────
        try:
            tk = yf.Ticker(self.ticker)
            info = tk.info or {}
        except Exception:
            info = {}

        short_pct_float = float(info.get("shortPercentOfFloat") or 0) * 100
        shares_short    = int(info.get("sharesShort") or 0)
        days_to_cover   = float(info.get("shortRatio") or 0)
        float_shares    = int(info.get("floatShares") or 0)
        avg_volume      = int(info.get("averageVolume") or 0)

        out["shortable_float_pct"] = round(short_pct_float, 2)
        out["days_to_cover"]       = days_to_cover
        out["float_shares"]        = float_shares
        out["avg_volume"]          = avg_volume

        # If short data is genuinely missing, fall back to days_to_cover only
        # (compute it ourselves if shares_short and avg_volume both exist)
        if days_to_cover <= 0 and shares_short and avg_volume:
            days_to_cover = shares_short / max(avg_volume, 1)
            out["days_to_cover"] = round(days_to_cover, 1)

        # ── 1. Short Interest score (30 pts) ────────────────────────────────
        if short_pct_float >= 30:
            si_pts, si_lbl = 30, f"EXTREME short interest {short_pct_float:.0f}% of float 🔥"
        elif short_pct_float >= 20:
            si_pts, si_lbl = 24, f"HIGH short interest {short_pct_float:.0f}% of float"
        elif short_pct_float >= 12:
            si_pts, si_lbl = 16, f"Moderate short interest {short_pct_float:.0f}%"
        elif short_pct_float >= 5:
            si_pts, si_lbl = 8,  f"Low short interest {short_pct_float:.0f}%"
        else:
            si_pts, si_lbl = 0,  f"No meaningful short interest ({short_pct_float:.1f}%)"
        out["components"]["short_interest"] = {"pts": si_pts, "max": 30, "label": si_lbl}

        # ── 2. Float Tightness (20 pts) ─────────────────────────────────────
        if 0 < float_shares <= 10_000_000:
            f_pts, f_lbl = 20, f"NANO float {float_shares:,} — explosive moves possible"
        elif float_shares <= 30_000_000:
            f_pts, f_lbl = 16, f"Micro float {float_shares:,}"
        elif float_shares <= 75_000_000:
            f_pts, f_lbl = 11, f"Small float {float_shares:,}"
        elif float_shares <= 200_000_000:
            f_pts, f_lbl = 6,  f"Medium float {float_shares:,}"
        elif float_shares > 0:
            f_pts, f_lbl = 2,  f"Large float {float_shares:,} — moves dampened"
        else:
            f_pts, f_lbl = 0,  "Float data unavailable"
        out["components"]["float"] = {"pts": f_pts, "max": 20, "label": f_lbl}

        # ── 3. Days to Cover (15 pts) ───────────────────────────────────────
        if days_to_cover >= 10:
            d_pts, d_lbl = 15, f"{days_to_cover:.1f} days to cover — shorts trapped 🔒"
        elif days_to_cover >= 5:
            d_pts, d_lbl = 11, f"{days_to_cover:.1f} days to cover — high pressure"
        elif days_to_cover >= 3:
            d_pts, d_lbl = 7,  f"{days_to_cover:.1f} days to cover — moderate"
        elif days_to_cover > 0:
            d_pts, d_lbl = 3,  f"{days_to_cover:.1f} days to cover — low"
        else:
            d_pts, d_lbl = 0,  "Days-to-cover unavailable"
        out["components"]["days_to_cover"] = {"pts": d_pts, "max": 15, "label": d_lbl}

        # ── 4. Bollinger Squeeze (15 pts) ───────────────────────────────────
        bb = detect_bollinger_squeeze(self.ticker)
        out["bb_squeeze"] = bb
        if bb:
            strength = bb.get("squeeze_strength", "NONE")
            days_sq  = bb.get("days_in_squeeze", 0)
            if strength == "EXTREME":
                bb_pts, bb_lbl = 15, f"EXTREME BB squeeze ({days_sq}d) — release imminent ⚡"
            elif strength == "STRONG":
                bb_pts, bb_lbl = 11, f"STRONG BB squeeze ({days_sq}d)"
            elif strength == "MODERATE":
                bb_pts, bb_lbl = 5,  f"Moderate BB compression ({days_sq}d)"
            else:
                bb_pts, bb_lbl = 0,  "No BB squeeze present"
        else:
            bb_pts, bb_lbl = 0, "BB analysis unavailable"
        out["components"]["bb_squeeze"] = {"pts": bb_pts, "max": 15, "label": bb_lbl}

        # ── 5. Volume Velocity (10 pts) ─────────────────────────────────────
        try:
            hist = get_cached_history(self.ticker, period="2mo", interval="1d")
            hist = hist.dropna(subset=["Volume"])
            if len(hist) >= 21:
                recent_avg = float(hist["Volume"].tail(5).mean())
                base_avg   = float(hist["Volume"].tail(20).mean())
                ratio = recent_avg / base_avg if base_avg else 1.0
                if ratio >= 2.5:
                    v_pts, v_lbl = 10, f"Volume {ratio:.1f}× normal — institutional accumulation 🚀"
                elif ratio >= 1.5:
                    v_pts, v_lbl = 7,  f"Volume {ratio:.1f}× normal — building interest"
                elif ratio >= 1.2:
                    v_pts, v_lbl = 4,  f"Volume {ratio:.1f}× normal — mild uptick"
                else:
                    v_pts, v_lbl = 0,  f"Volume {ratio:.1f}× normal — quiet"
            else:
                v_pts, v_lbl = 0, "Volume data insufficient"
        except Exception:
            v_pts, v_lbl = 0, "Volume analysis failed"
        out["components"]["volume_velocity"] = {"pts": v_pts, "max": 10, "label": v_lbl}

        # ── 6. Catalyst Proximity (10 pts) ──────────────────────────────────
        # Uses the NewsAgent if available; otherwise heuristic via yfinance news
        c_pts, c_lbl = 0, "No recent catalyst detected"
        if self.conn:
            try:
                from news_agent import NewsAgent
                ni = NewsAgent(self.conn, ai_analyst=None).get_news_impact(
                    self.ticker, hours=72
                )
                if ni and ni.get("n_events", 0) > 0:
                    impact = ni.get("max_impact", 0)
                    sent   = (ni.get("top_event") or {}).get("sentiment", "")
                    if impact >= 7 and sent == "POSITIVE":
                        c_pts, c_lbl = 10, f"Strong positive catalyst (impact {impact}/10) 📰"
                    elif impact >= 5:
                        c_pts, c_lbl = 6,  f"Catalyst present (impact {impact}/10)"
                    elif ni.get("n_events", 0) >= 3:
                        c_pts, c_lbl = 3,  f"Multiple news events ({ni['n_events']})"
            except Exception:
                pass
        out["components"]["catalyst"] = {"pts": c_pts, "max": 10, "label": c_lbl}

        # ── Sum + category ──────────────────────────────────────────────────
        total = si_pts + f_pts + d_pts + bb_pts + v_pts + c_pts
        out["squeeze_score"] = total

        if total >= self._TIER_EXPLOSIVE:
            out["category"]        = "EXPLOSIVE"
            out["upside_estimate"] = "2-10x potential"
            out["recommendation"]  = ("🔥 EXPLOSIVE SETUP — short squeeze ALL signals firing. "
                                       "High risk, high reward. Small position size + tight stop.")
        elif total >= self._TIER_HIGH:
            out["category"]        = "HIGH"
            out["upside_estimate"] = "50-100% upside"
            out["recommendation"]  = ("⚡ HIGH SQUEEZE potential — multiple compression signals "
                                       "aligned.  Reasonable position size with wider stop.")
        elif total >= self._TIER_MODERATE:
            out["category"]        = "MODERATE"
            out["upside_estimate"] = "20-50% upside"
            out["recommendation"]  = ("📊 MODERATE squeeze setup — some compression signals.  "
                                       "Trade with normal sizing.")
        else:
            out["category"]        = "LOW"
            out["upside_estimate"] = "Standard upside"
            out["recommendation"]  = ("Standard setup — no significant squeeze advantage.")

        return out

    @staticmethod
    def scan_many(tickers: list, conn=None,
                   min_score: int = 50) -> list:
        """Scan a batch of tickers and return only those scoring ≥ min_score,
        sorted by squeeze_score descending."""
        results = []
        for tk in tickers:
            try:
                r = SqueezeScanner(tk, conn=conn).scan()
                if r["squeeze_score"] >= min_score:
                    results.append(r)
            except Exception:
                continue
        results.sort(key=lambda x: x["squeeze_score"], reverse=True)
        return results


def predict_duration_to_target(ticker, entry_price, target_price,
                                pattern="", market_regime=None,
                                min_days=1, max_days=30):
    """
    Predict trading days until *ticker* reaches *target_price* from *entry_price*.

    Model:
        base_days     = move_required_pct / avg_daily_move_pct
        predicted     = base_days × pattern_multiplier × regime_multiplier
        clipped to [min_days, max_days]

    Where avg_daily_move_pct = 14-day ATR / entry_price × 100.

    Returns dict:
        predicted_days  — int, the central estimate
        min_days_est    — int, lower bound (60 %)
        max_days_est    — int, upper bound (150 %)
        confidence      — 'HIGH' / 'MEDIUM' / 'LOW'
        atr_pct         — average daily move as % of entry
        move_required_pct — % move needed to hit target
        pattern_mult    — pattern speed multiplier applied
        regime_mult     — regime scaling applied
        method          — description of model used
    """
    try:
        entry_price  = float(entry_price)
        target_price = float(target_price)
        if entry_price <= 0 or target_price <= 0:
            return None

        # ── Move required to hit target ───────────────────────────────────────
        move_required_pct = (target_price - entry_price) / entry_price * 100
        if move_required_pct <= 0:
            # Target is at or below entry — already there
            return {
                "predicted_days":    0,
                "min_days_est":      0,
                "max_days_est":      0,
                "confidence":        "HIGH",
                "atr_pct":           0.0,
                "move_required_pct": round(move_required_pct, 2),
                "pattern_mult":      1.0,
                "regime_mult":       1.0,
                "method":            "Target already reached at entry price",
            }

        # ── ATR-based daily move estimate (uses shared history cache) ────────
        hist = get_cached_history(ticker, period="3mo", interval="1d")
        hist = hist.dropna(subset=["Close"]) if not hist.empty else hist
        if len(hist) < 15:
            return None

        # 14-day ATR
        hl = (hist["High"] - hist["Low"]).abs()
        hc = (hist["High"] - hist["Close"].shift(1)).abs()
        lc = (hist["Low"]  - hist["Close"].shift(1)).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().dropna().iloc[-1])
        atr_pct = atr / entry_price * 100 if entry_price > 0 else 0

        if atr_pct <= 0.1:   # essentially zero daily move
            return None

        base_days = move_required_pct / atr_pct

        # ── Pattern multiplier ────────────────────────────────────────────────
        pat_key = str(pattern or "").strip().lower()
        pattern_mult = _PATTERN_DURATION_MULT.get(pat_key, 1.0)

        # ── Regime multiplier ─────────────────────────────────────────────────
        regime_mult = 1.0
        if market_regime:
            regime_mult = _REGIME_DURATION_MULT.get(
                str(market_regime.get("regime", "NEUTRAL")), 1.0
            )

        # ── Final prediction with clamps ──────────────────────────────────────
        predicted = base_days * pattern_mult * regime_mult
        predicted = max(min_days, min(max_days, int(round(predicted))))
        lo        = max(min_days, int(round(predicted * 0.6)))
        hi        = min(max_days, int(round(predicted * 1.5)))

        # ── Confidence: high for moderate moves, low for outliers ─────────────
        if 3 <= move_required_pct <= 15 and atr_pct >= 1.0:
            confidence = "HIGH"
        elif 1.5 <= move_required_pct <= 25 and atr_pct >= 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "predicted_days":     predicted,
            "min_days_est":       lo,
            "max_days_est":       hi,
            "confidence":         confidence,
            "atr_pct":            round(atr_pct, 2),
            "move_required_pct":  round(move_required_pct, 2),
            "pattern_mult":       pattern_mult,
            "regime_mult":        regime_mult,
            "method":             f"ATR ({atr_pct:.2f}%/d) × pattern ({pattern_mult:.2f}×) × regime ({regime_mult:.2f}×)",
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SELF-LEARNING RISK ENGINE
#   Two-layer drawdown protection with persistent learning across resets.
#     • -15 % drawdown  → enters PUNISHMENT mode (halt + raise score floor)
#                         until equity recovers to within 5 % of peak.
#     • -50 % drawdown  → triggers HARD RESET: closes all positions, analyses
#                         losing trades, saves lessons, resets capital, and
#                         tightens entry thresholds for the next cycle.
#   Adjustments persist in the DB so the bot truly adapts over time.
# ══════════════════════════════════════════════════════════════════════════════

def _row_get(row, key, default=None):
    """Read a key from any row-like (sqlite3.Row or PgAdapter _PgRow).

    sqlite3.Row doesn't have .get() so the standard helper is to convert
    to a dict first.  This wrapper handles both transparently."""
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


class LearningEngine:
    """Self-improvement engine — punishes drawdowns, learns from failed cycles."""

    PUNISHMENT_THRESHOLD = 15.0   # % drawdown to enter PUNISHMENT mode
    RESET_THRESHOLD      = 50.0   # % drawdown to trigger HARD RESET
    RECOVERY_BAND_PCT    = 5.0    # must recover to within 5 % of peak to exit punishment

    _INIT_SQL = """
CREATE TABLE IF NOT EXISTS paper_lessons (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    learning_iteration  INTEGER,
    reset_timestamp     TIMESTAMP,
    reason              TEXT,
    peak_equity         REAL,
    final_equity        REAL,
    drawdown_pct        REAL,
    n_trades            INTEGER,
    n_wins              INTEGER,
    n_losses            INTEGER,
    avg_win_pnl         REAL,
    avg_loss_pnl        REAL,
    win_rate_pct        REAL,
    worst_sector        TEXT,
    worst_pattern       TEXT,
    worst_wyckoff       TEXT,
    avg_losing_score    REAL,
    lessons_summary     TEXT
);
CREATE TABLE IF NOT EXISTS paper_adjustments (
    id                  INTEGER PRIMARY KEY,
    learning_iteration  INTEGER DEFAULT 0,
    min_master_score    REAL    DEFAULT 65,
    sector_blacklist    TEXT    DEFAULT '[]',
    pattern_blacklist   TEXT    DEFAULT '[]',
    wyckoff_blacklist   TEXT    DEFAULT '["DISTRIBUTION","MARKDOWN"]',
    size_multiplier_cap REAL    DEFAULT 1.0,
    updated_at          TIMESTAMP
);
CREATE TABLE IF NOT EXISTS paper_punishment (
    id                      INTEGER PRIMARY KEY,
    active                  INTEGER DEFAULT 0,
    triggered_at            TIMESTAMP,
    peak_equity_at_trigger  REAL,
    recovery_target         REAL,
    master_score_floor      REAL DEFAULT 75,
    size_cap                REAL DEFAULT 0.5
);
CREATE TABLE IF NOT EXISTS paper_suggestions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_date         DATE,
    learning_iteration      INTEGER,
    category                TEXT,
    suggestion              TEXT,
    rationale               TEXT,
    action                  TEXT,
    applied                 INTEGER DEFAULT 0,
    diagnostics_json        TEXT,
    created_at              TIMESTAMP
);
CREATE TABLE IF NOT EXISTS paper_signal_performance (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type             TEXT,
    signal_value            TEXT,
    n_trades                INTEGER DEFAULT 0,
    n_wins                  INTEGER DEFAULT 0,
    n_losses                INTEGER DEFAULT 0,
    sum_win_pnl             REAL    DEFAULT 0,
    sum_loss_pnl            REAL    DEFAULT 0,
    last_updated            TIMESTAMP,
    UNIQUE(signal_type, signal_value)
);
CREATE TABLE IF NOT EXISTS paper_learn_state (
    id                      INTEGER PRIMARY KEY,
    last_processed_trade_id INTEGER DEFAULT 0,
    recent_wins             INTEGER DEFAULT 0,
    recent_losses           INTEGER DEFAULT 0,
    nudges_made             INTEGER DEFAULT 0,
    last_nudge_at           TIMESTAMP,
    notes                   TEXT
);
CREATE TABLE IF NOT EXISTS paper_learning_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_at                TIMESTAMP,
    event_type              TEXT,
    detail                  TEXT,
    before_value            TEXT,
    after_value             TEXT
);
"""

    def __init__(self, conn):
        self.conn = conn
        try:
            self.conn.executescript(self._INIT_SQL)
        except Exception:
            pass
        # Seed the adjustments row if missing
        try:
            row = self.conn.execute(
                "SELECT id FROM paper_adjustments WHERE id=1"
            ).fetchone()
            if not row:
                self.conn.execute(
                    "INSERT INTO paper_adjustments (id, updated_at) VALUES (1, ?)",
                    (datetime.utcnow().isoformat(),)
                )
        except Exception:
            pass

    # ── State check (called every monitor cycle) ──────────────────────────────

    def check_state(self, paper_engine, options_engine=None) -> dict:
        """Inspect portfolio, decide whether to enter punishment, exit, or reset.

        Returns dict:
          action       — 'NORMAL' | 'ENTER_PUNISHMENT' | 'CONTINUE_PUNISHMENT'
                         | 'EXIT_PUNISHMENT' | 'RESET'
          drawdown_pct — current drawdown from peak
          peak_equity  — peak equity tracked
          current_equity — current cash + invested
          message      — human-readable summary
          adjustments  — current active adjustments dict (post-action)
        """
        try:
            risk = get_portfolio_risk_status(paper_engine)
        except Exception:
            return {"action": "NORMAL", "drawdown_pct": 0,
                    "message": "Risk status unavailable",
                    "adjustments": self.get_active_adjustments()}

        dd     = float(risk.get("drawdown_pct", 0) or 0)
        peak   = float(risk.get("peak_equity", 1000) or 1000)
        curr   = float(risk.get("current_equity", 0) or 0)
        is_punishing = self._is_punishing()

        # ── -50 % HARD RESET ──────────────────────────────────────────────────
        if dd >= self.RESET_THRESHOLD:
            lessons = self.trigger_reset(paper_engine, options_engine,
                                          peak_equity=peak, final_equity=curr,
                                          drawdown_pct=dd)
            return {
                "action":         "RESET",
                "drawdown_pct":   dd,
                "peak_equity":    peak,
                "current_equity": curr,
                "message":        f"HARD RESET at {dd:.1f}% drawdown — capital restored, lessons applied",
                "lessons":        lessons,
                "adjustments":    self.get_active_adjustments(),
            }

        # ── -15 % PUNISHMENT entry ────────────────────────────────────────────
        if not is_punishing and dd >= self.PUNISHMENT_THRESHOLD:
            recovery_target = peak * (1 - self.RECOVERY_BAND_PCT / 100)
            self._enter_punishment(peak, recovery_target)
            return {
                "action":          "ENTER_PUNISHMENT",
                "drawdown_pct":    dd,
                "peak_equity":     peak,
                "current_equity":  curr,
                "recovery_target": recovery_target,
                "message": (
                    f"⚠️ PUNISHMENT MODE — drawdown {dd:.1f}% ≥ {self.PUNISHMENT_THRESHOLD}%.  "
                    f"No new trades until equity recovers to ${recovery_target:,.2f} "
                    f"(within {self.RECOVERY_BAND_PCT}% of peak). "
                    f"Score floor raised to 75."
                ),
                "adjustments":     self.get_active_adjustments(),
            }

        # ── PUNISHMENT continuation / exit ────────────────────────────────────
        if is_punishing:
            row = self._punishment_row()
            recovery_target = float(_row_get(row, "recovery_target", 0) or 0)
            if curr >= recovery_target and recovery_target > 0:
                self._exit_punishment()
                return {
                    "action":          "EXIT_PUNISHMENT",
                    "drawdown_pct":    dd,
                    "peak_equity":     peak,
                    "current_equity":  curr,
                    "message":         f"✅ Recovered above ${recovery_target:,.2f} — punishment ended.",
                    "adjustments":     self.get_active_adjustments(),
                }
            return {
                "action":          "CONTINUE_PUNISHMENT",
                "drawdown_pct":    dd,
                "peak_equity":     peak,
                "current_equity":  curr,
                "recovery_target": recovery_target,
                "message": (
                    f"🛑 Still in PUNISHMENT — ${curr:,.2f} < ${recovery_target:,.2f} target. "
                    f"No new trades."
                ),
                "adjustments":     self.get_active_adjustments(),
            }

        # ── NORMAL ────────────────────────────────────────────────────────────
        return {
            "action":         "NORMAL",
            "drawdown_pct":   dd,
            "peak_equity":    peak,
            "current_equity": curr,
            "message":        f"Operating normally  ·  drawdown {dd:.1f}%",
            "adjustments":    self.get_active_adjustments(),
        }

    # ── Punishment helpers ────────────────────────────────────────────────────

    def _is_punishing(self) -> bool:
        try:
            row = self.conn.execute(
                "SELECT active FROM paper_punishment WHERE id=1"
            ).fetchone()
            if row is None:
                return False
            # row[0] works on both sqlite3.Row and _PgRow
            return bool(int(row[0] or 0) == 1)
        except Exception:
            return False

    def _punishment_row(self):
        try:
            row = self.conn.execute(
                "SELECT * FROM paper_punishment WHERE id=1"
            ).fetchone()
            if row is None:
                return None
            # Normalize to dict so callers can use .get() uniformly
            try:
                return dict(row)
            except Exception:
                return row
        except Exception:
            return None

    def _enter_punishment(self, peak_equity: float, recovery_target: float):
        try:
            self.conn.execute("DELETE FROM paper_punishment WHERE id=1")
            self.conn.execute(
                "INSERT INTO paper_punishment "
                "(id, active, triggered_at, peak_equity_at_trigger, recovery_target, "
                "master_score_floor, size_cap) VALUES (1, 1, ?, ?, ?, 75, 0.5)",
                (datetime.utcnow().isoformat(), peak_equity, recovery_target)
            )
        except Exception:
            pass

    def _exit_punishment(self):
        try:
            self.conn.execute("UPDATE paper_punishment SET active=0 WHERE id=1")
        except Exception:
            pass

    # ── Hard reset + lesson extraction ────────────────────────────────────────

    def trigger_reset(self, paper_engine, options_engine=None,
                      peak_equity=0.0, final_equity=0.0, drawdown_pct=0.0,
                      reason="-50% DRAWDOWN") -> dict:
        """Capital reset with lesson learning.  Force-closes everything, saves
        an analysis of failed trades, applies adjustments, restarts at $1,000."""
        import json as _json

        # ── 1. Snapshot of closed-trade history BEFORE reset ──────────────────
        try:
            rows = self.conn.execute(
                "SELECT ticker, sector, pattern, explosive_score, breakout_prob, "
                "net_pnl, gross_invested, exit_reason "
                "FROM paper_portfolio WHERE status='CLOSED'"
            ).fetchall()
            # Normalize each row to a plain dict for cross-DB consistency
            closed = []
            for r in rows or []:
                try:
                    closed.append(dict(r))
                except Exception:
                    pass
        except Exception:
            closed = []

        # ── 2. Force-close any remaining open positions ───────────────────────
        try:
            for p in paper_engine.open_positions:
                ep    = float(p.get("entry_price") or 0)
                ticker = str(p.get("ticker", ""))
                # Mark as "RESET_CLOSED" at entry (zero P&L) — fair since we don't
                # have a reliable live price here and we're nuking the account anyway
                try:
                    paper_engine.close_position(ticker, ep, "RESET_FORCED")
                except Exception:
                    pass
        except Exception:
            pass

        # ── 3. Analyse losers vs winners ──────────────────────────────────────
        losers  = [t for t in closed if (t.get("net_pnl") or 0) < 0]
        winners = [t for t in closed if (t.get("net_pnl") or 0) > 0]
        n_l, n_w = len(losers), len(winners)
        n_tot   = n_l + n_w

        avg_loss = (sum(t["net_pnl"] for t in losers) / n_l) if n_l else 0.0
        avg_win  = (sum(t["net_pnl"] for t in winners) / n_w) if n_w else 0.0
        win_rate = (n_w / n_tot * 100) if n_tot else 0.0

        def _worst(field):
            buckets = {}
            for t in losers:
                k = str(t.get(field) or "Unknown") or "Unknown"
                buckets[k] = buckets.get(k, 0) + 1
            if not buckets:
                return "—"
            return max(buckets, key=lambda k: buckets[k])

        worst_sector  = _worst("sector")
        worst_pattern = _worst("pattern")
        # Wyckoff isn't stored on trades historically; use placeholder
        worst_wyckoff = "—"

        avg_losing_score = (sum(float(t.get("explosive_score") or 0) for t in losers) / n_l
                            if n_l else 0.0)

        # ── 4. Derive adjustments to apply going forward ──────────────────────
        adjustments = self.get_active_adjustments()
        new_iter = int(adjustments.get("learning_iteration", 0)) + 1

        new_min_score = float(adjustments.get("min_master_score", 65))
        # If most losers had master/explosive scores < 75, raise the floor
        if avg_losing_score and avg_losing_score < 75:
            new_min_score = min(85.0, max(new_min_score, round(avg_losing_score + 8, 0)))

        sector_blacklist  = list(adjustments.get("sector_blacklist",  []))
        pattern_blacklist = list(adjustments.get("pattern_blacklist", []))

        # If a sector lost ≥ 60 % of its trades AND had ≥ 3 trades → blacklist
        sec_stats = {}
        for t in closed:
            k = str(t.get("sector") or "Unknown")
            sec_stats.setdefault(k, [0, 0])  # [wins, losses]
            if (t.get("net_pnl") or 0) > 0:
                sec_stats[k][0] += 1
            else:
                sec_stats[k][1] += 1
        for sec, (w, l) in sec_stats.items():
            tot = w + l
            if tot >= 3 and l / tot >= 0.6 and sec not in sector_blacklist and sec != "Unknown":
                sector_blacklist.append(sec)

        # Same for patterns
        pat_stats = {}
        for t in closed:
            k = str(t.get("pattern") or "Unknown")
            pat_stats.setdefault(k, [0, 0])
            if (t.get("net_pnl") or 0) > 0:
                pat_stats[k][0] += 1
            else:
                pat_stats[k][1] += 1
        for pat, (w, l) in pat_stats.items():
            tot = w + l
            if tot >= 3 and l / tot >= 0.6 and pat not in pattern_blacklist and pat != "Unknown":
                pattern_blacklist.append(pat)

        # Wyckoff blacklist is permanent + grows
        wyckoff_blacklist = list(adjustments.get("wyckoff_blacklist",
                                                 ["DISTRIBUTION", "MARKDOWN"]))
        for must_avoid in ("DISTRIBUTION", "MARKDOWN"):
            if must_avoid not in wyckoff_blacklist:
                wyckoff_blacklist.append(must_avoid)

        lessons_summary = (
            f"Iteration {new_iter}: closed {n_tot} trades ({n_w}W / {n_l}L, "
            f"win-rate {win_rate:.0f}%).  "
            f"Avg loss ${avg_loss:.2f} vs avg win ${avg_win:.2f}.  "
            f"Worst sector: {worst_sector}.  Worst pattern: {worst_pattern}.  "
            f"New min Master Score: {new_min_score:.0f}.  "
            f"Sectors blacklisted: {sector_blacklist or 'none'}.  "
            f"Patterns blacklisted: {pattern_blacklist or 'none'}."
        )

        # ── 5. Persist lesson row ─────────────────────────────────────────────
        try:
            self.conn.execute(
                "INSERT INTO paper_lessons "
                "(learning_iteration, reset_timestamp, reason, peak_equity, "
                "final_equity, drawdown_pct, n_trades, n_wins, n_losses, "
                "avg_win_pnl, avg_loss_pnl, win_rate_pct, worst_sector, "
                "worst_pattern, worst_wyckoff, avg_losing_score, lessons_summary) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_iter, datetime.utcnow().isoformat(), reason, peak_equity,
                 final_equity, drawdown_pct, n_tot, n_w, n_l, avg_win,
                 avg_loss, win_rate, worst_sector, worst_pattern, worst_wyckoff,
                 avg_losing_score, lessons_summary)
            )
        except Exception:
            pass

        # ── 6. Update active adjustments ──────────────────────────────────────
        try:
            self.conn.execute("DELETE FROM paper_adjustments WHERE id=1")
            self.conn.execute(
                "INSERT INTO paper_adjustments "
                "(id, learning_iteration, min_master_score, sector_blacklist, "
                "pattern_blacklist, wyckoff_blacklist, size_multiplier_cap, "
                "updated_at) VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                (new_iter, new_min_score, _json.dumps(sector_blacklist),
                 _json.dumps(pattern_blacklist), _json.dumps(wyckoff_blacklist),
                 max(0.5, 1.0 - new_iter * 0.05),    # tighten cap as iterations grow
                 datetime.utcnow().isoformat())
            )
        except Exception:
            pass

        # ── 7. Clear punishment flag (fresh start) ────────────────────────────
        self._exit_punishment()

        # ── 8. Reset paper engine to $1,000 ───────────────────────────────────
        try:
            if hasattr(paper_engine, "reset"):
                paper_engine.reset()
            else:
                # Best-effort manual reset
                paper_engine._cash         = _PAPER_BUDGET
                paper_engine._fees_paid    = 0.0
                paper_engine._realized_pnl = 0.0
                paper_engine._starting     = _PAPER_BUDGET
                paper_engine._save_state()
        except Exception:
            pass

        # ── 9. Reset options too if applicable ────────────────────────────────
        if options_engine and hasattr(options_engine, "reset"):
            try:
                options_engine.reset()
            except Exception:
                pass

        return {
            "learning_iteration": new_iter,
            "n_trades":           n_tot,
            "n_wins":             n_w,
            "n_losses":           n_l,
            "win_rate_pct":       round(win_rate, 1),
            "avg_loss_pnl":       round(avg_loss, 2),
            "avg_win_pnl":        round(avg_win,  2),
            "worst_sector":       worst_sector,
            "worst_pattern":      worst_pattern,
            "new_min_master_score": new_min_score,
            "sector_blacklist":   sector_blacklist,
            "pattern_blacklist":  pattern_blacklist,
            "lessons_summary":    lessons_summary,
        }

    # ── Active adjustments accessor ───────────────────────────────────────────

    def get_active_adjustments(self) -> dict:
        """Return the threshold adjustments the bot should apply right now."""
        import json as _json
        try:
            row = self.conn.execute(
                "SELECT * FROM paper_adjustments WHERE id=1"
            ).fetchone()
        except Exception:
            row = None
        # Sensible defaults if no row
        if not row:
            return {
                "learning_iteration":  0,
                "min_master_score":    65.0,
                "sector_blacklist":    [],
                "pattern_blacklist":   [],
                "wyckoff_blacklist":   ["DISTRIBUTION", "MARKDOWN"],
                "size_multiplier_cap": 1.0,
            }
        try:
            sl = _json.loads(_row_get(row, "sector_blacklist", "[]")  or "[]")
        except Exception:
            sl = []
        try:
            pl = _json.loads(_row_get(row, "pattern_blacklist", "[]") or "[]")
        except Exception:
            pl = []
        try:
            wl = _json.loads(_row_get(row, "wyckoff_blacklist", '["DISTRIBUTION","MARKDOWN"]')
                              or '["DISTRIBUTION","MARKDOWN"]')
        except Exception:
            wl = ["DISTRIBUTION", "MARKDOWN"]
        return {
            "learning_iteration":  int(_row_get(row, "learning_iteration", 0) or 0),
            "min_master_score":    float(_row_get(row, "min_master_score", 65) or 65),
            "sector_blacklist":    sl,
            "pattern_blacklist":   pl,
            "wyckoff_blacklist":   wl,
            "size_multiplier_cap": float(_row_get(row, "size_multiplier_cap", 1.0) or 1.0),
        }

    # ── Lesson history (for UI display) ───────────────────────────────────────

    def get_lesson_history(self, limit: int = 10) -> list:
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_lessons ORDER BY id DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
            out = []
            for r in rows or []:
                try:
                    out.append(dict(r))
                except Exception:
                    pass
            return out
        except Exception:
            return []

    # ── Punishment status accessor (for UI display) ───────────────────────────

    def get_punishment_status(self) -> dict:
        row = self._punishment_row()
        if not row or not int(_row_get(row, "active", 0) or 0):
            return {"active": False}
        return {
            "active":                 True,
            "triggered_at":           str(_row_get(row, "triggered_at", "") or ""),
            "peak_equity_at_trigger": float(_row_get(row, "peak_equity_at_trigger", 0) or 0),
            "recovery_target":        float(_row_get(row, "recovery_target", 0) or 0),
            "master_score_floor":     float(_row_get(row, "master_score_floor", 75) or 75),
            "size_cap":               float(_row_get(row, "size_cap", 0.5) or 0.5),
        }

    # ── Engine diagnostics & AI improvement suggestion ────────────────────────

    def compile_engine_diagnostics(self, paper_engine) -> dict:
        """Gather a rich snapshot of the bot's current performance so the AI
        can make informed improvement suggestions.  Captures win rates,
        score distributions, holding times, exit reasons, sector + pattern
        performance — everything that helps spot what to tweak."""
        diag = {
            "iteration":         0,
            "adjustments":       self.get_active_adjustments(),
            "punishment":        self.get_punishment_status(),
            "portfolio":         {},
            "totals":            {},
            "win_rate_pct":      0.0,
            "by_exit_reason":    {},
            "by_sector":         {},
            "by_pattern":        {},
            "score_buckets":     {"winners": [], "losers": []},
            "lesson_history":    [],
            "recent_close_avg":  {},
        }
        diag["iteration"] = diag["adjustments"].get("learning_iteration", 0)

        # ── Portfolio summary ─────────────────────────────────────────────────
        try:
            summ = paper_engine.get_summary()
            diag["portfolio"] = {
                "cash":             float(summ.get("available_cash", 0)),
                "realized_pnl":     float(summ.get("realized_pnl", 0)),
                "total_return_pct": float(summ.get("total_return_pct", 0)),
                "trades_made":      int(summ.get("trades_made", 0)),
                "n_open":           int(summ.get("open_positions", 0) or 0),
                "n_closed":         int(summ.get("n_closed", 0) or 0),
            }
        except Exception:
            pass

        # ── Closed trades analysis ────────────────────────────────────────────
        try:
            rows = self.conn.execute(
                "SELECT ticker, sector, pattern, explosive_score, breakout_prob, "
                "net_pnl, gross_invested, exit_reason, entry_date, exit_date "
                "FROM paper_portfolio WHERE status='CLOSED'"
            ).fetchall()
            closed = [dict(r) for r in (rows or [])]
        except Exception:
            closed = []

        if not closed:
            diag["totals"] = {"n_winners": 0, "n_losers": 0, "n_total": 0}
            diag["lesson_history"] = self.get_lesson_history(limit=5)
            return diag

        winners = [t for t in closed if (t.get("net_pnl") or 0) > 0]
        losers  = [t for t in closed if (t.get("net_pnl") or 0) <= 0]
        n_w, n_l = len(winners), len(losers)
        n_t = n_w + n_l

        diag["totals"] = {
            "n_winners":     n_w,
            "n_losers":      n_l,
            "n_total":       n_t,
            "sum_win_pnl":   round(sum(t.get("net_pnl", 0) for t in winners), 2),
            "sum_loss_pnl":  round(sum(t.get("net_pnl", 0) for t in losers),  2),
            "avg_win_pnl":   round(sum(t.get("net_pnl", 0) for t in winners) / n_w, 2) if n_w else 0,
            "avg_loss_pnl":  round(sum(t.get("net_pnl", 0) for t in losers) / n_l,  2) if n_l else 0,
            "biggest_win":   round(max((t.get("net_pnl", 0) for t in winners), default=0), 2),
            "biggest_loss":  round(min((t.get("net_pnl", 0) for t in losers),  default=0), 2),
        }
        diag["win_rate_pct"] = round(n_w / n_t * 100, 1) if n_t else 0

        # ── Per-bucket breakdowns ─────────────────────────────────────────────
        def _bucket(field):
            wins = {}
            losses = {}
            for t in closed:
                k = str(t.get(field) or "Unknown") or "Unknown"
                if (t.get("net_pnl") or 0) > 0:
                    wins[k] = wins.get(k, 0) + 1
                else:
                    losses[k] = losses.get(k, 0) + 1
            out = {}
            for k in set(list(wins.keys()) + list(losses.keys())):
                w, l = wins.get(k, 0), losses.get(k, 0)
                tot = w + l
                out[k] = {
                    "wins":   w,
                    "losses": l,
                    "win_rate_pct": round(w / tot * 100, 1) if tot else 0,
                    "total":  tot,
                }
            return out

        diag["by_exit_reason"] = _bucket("exit_reason")
        diag["by_sector"]      = _bucket("sector")
        diag["by_pattern"]     = _bucket("pattern")

        # ── Score distributions ───────────────────────────────────────────────
        diag["score_buckets"]["winners"] = sorted(
            [float(t.get("explosive_score") or 0) for t in winners])[:20]
        diag["score_buckets"]["losers"] = sorted(
            [float(t.get("explosive_score") or 0) for t in losers])[:20]

        # ── Lesson history (past resets) ──────────────────────────────────────
        diag["lesson_history"] = self.get_lesson_history(limit=5)

        # ── Recent closes (last 5) ────────────────────────────────────────────
        try:
            recent_rows = self.conn.execute(
                "SELECT ticker, net_pnl, exit_reason, exit_date "
                "FROM paper_portfolio WHERE status='CLOSED' "
                "ORDER BY exit_date DESC LIMIT 5"
            ).fetchall()
            diag["recent_close_avg"] = [
                {k: dict(r).get(k) for k in ("ticker", "net_pnl", "exit_reason", "exit_date")}
                for r in (recent_rows or [])
            ]
        except Exception:
            pass

        return diag

    def generate_improvement_suggestion(self, ai_analyst, paper_engine) -> dict:
        """Use the AI analyst to propose ONE concrete improvement to the
        learning engine.  Stores the suggestion in paper_suggestions.

        Returns dict: {date, suggestion, rationale, action, category, id}
        Falls back to a deterministic suggestion if AI is unavailable, so the
        feature still works without API keys."""
        diag = self.compile_engine_diagnostics(paper_engine)
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Skip if we already have a suggestion for today (idempotent daily run)
        try:
            existing = self.conn.execute(
                "SELECT id, category, suggestion, rationale, action "
                "FROM paper_suggestions WHERE suggestion_date=? "
                "ORDER BY id DESC LIMIT 1",
                (today,)
            ).fetchone()
            if existing:
                d = dict(existing)
                return {
                    "date":       today,
                    "category":   d.get("category", ""),
                    "suggestion": d.get("suggestion", ""),
                    "rationale":  d.get("rationale", ""),
                    "action":     d.get("action", ""),
                    "id":         d.get("id"),
                    "from_cache": True,
                }
        except Exception:
            pass

        # ── Build the AI prompt ──────────────────────────────────────────────
        prompt = (
            "You are tuning a self-learning paper trading bot.  Below is a JSON "
            "snapshot of the current Learning Engine state, recent performance, "
            "active blacklists, and lesson history.  Suggest ONE specific "
            "improvement that would make the bot more profitable going forward.\n\n"
            "RULES:\n"
            "1. Be concrete and actionable — name a specific parameter, threshold, "
            "blacklist entry, or new check.\n"
            "2. Cite numbers from the JSON to justify the change.\n"
            "3. Pick ONE thing — the highest-leverage improvement for THIS state.\n"
            "4. If the bot has few trades (<5 closed), suggest something to "
            "increase trade frequency without sacrificing quality.\n"
            "5. If the bot is over-trading or losing money, suggest tightening.\n"
            "6. If win rate is decent but P&L is negative, suggest sizing/exit fixes.\n\n"
            "Format your response EXACTLY like this (no extra prose):\n"
            "CATEGORY: thresholds|blacklists|sizing|exits|new_check|other\n"
            "SUGGESTION: <one sentence summary>\n"
            "RATIONALE: <2-3 sentences citing specific numbers>\n"
            "ACTION: <exact parameter / blacklist entry / code change to make>\n"
        )

        ai_response = ""
        if ai_analyst and getattr(ai_analyst, "available", False):
            try:
                ai_response = ai_analyst.chat(prompt, context=diag, max_tokens=500)
            except Exception:
                ai_response = ""

        # ── Parse AI response ────────────────────────────────────────────────
        category, suggestion, rationale, action = "", "", "", ""
        if ai_response and "SUGGESTION:" in ai_response:
            try:
                lines = [ln.strip() for ln in ai_response.splitlines() if ln.strip()]
                for ln in lines:
                    if ln.upper().startswith("CATEGORY:"):
                        category = ln.split(":", 1)[1].strip()
                    elif ln.upper().startswith("SUGGESTION:"):
                        suggestion = ln.split(":", 1)[1].strip()
                    elif ln.upper().startswith("RATIONALE:"):
                        rationale = ln.split(":", 1)[1].strip()
                    elif ln.upper().startswith("ACTION:"):
                        action = ln.split(":", 1)[1].strip()
            except Exception:
                pass

        # ── Deterministic fallback if AI failed or returned unparseable text ─
        if not suggestion:
            category, suggestion, rationale, action = self._fallback_suggestion(diag)

        # ── Persist to DB ─────────────────────────────────────────────────────
        import json as _json
        new_id = None
        try:
            cur = self.conn.execute(
                "INSERT INTO paper_suggestions "
                "(suggestion_date, learning_iteration, category, suggestion, "
                "rationale, action, diagnostics_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (today, diag.get("iteration", 0), category, suggestion, rationale,
                 action, _json.dumps(diag, default=str)[:8000],
                 datetime.utcnow().isoformat())
            )
            # Get the new ID
            try:
                last = self.conn.execute(
                    "SELECT id FROM paper_suggestions ORDER BY id DESC LIMIT 1"
                ).fetchone()
                new_id = int(dict(last).get("id")) if last else None
            except Exception:
                pass
        except Exception:
            pass

        return {
            "date":       today,
            "category":   category,
            "suggestion": suggestion,
            "rationale":  rationale,
            "action":     action,
            "id":         new_id,
            "from_cache": False,
        }

    @staticmethod
    def _fallback_suggestion(diag: dict) -> tuple:
        """Deterministic rule-based suggestion when AI is unavailable.
        Picks the most relevant heuristic given the bot's current state."""
        n_t = diag.get("totals", {}).get("n_total", 0)
        wr  = diag.get("win_rate_pct", 0)
        adj = diag.get("adjustments", {})
        ms_floor = adj.get("min_master_score", 65)
        iter_n = diag.get("iteration", 0)
        portfolio = diag.get("portfolio", {})
        n_open = portfolio.get("n_open", 0)
        realized = portfolio.get("realized_pnl", 0)

        # Too few trades — encourage activity
        if n_t < 5:
            return (
                "thresholds",
                "Lower min_master_score to widen the candidate pool",
                f"Only {n_t} trades closed so far — insufficient data to learn from. "
                f"The current floor of {ms_floor:.0f} may be filtering too aggressively.",
                f"Set LearningEngine.get_active_adjustments min_master_score to "
                f"{max(55, ms_floor - 5):.0f} (was {ms_floor:.0f}) for 1 week to gather more data.",
            )

        # Punishment active
        if diag.get("punishment", {}).get("active"):
            return (
                "risk_management",
                "Add a partial-profit-taking rule before punishment kicks in",
                f"Bot is currently in PUNISHMENT mode at iteration #{iter_n}. "
                f"Drawdown was reached without taking any profits along the way.",
                "Add rule: when any position is up >25%, automatically take 50% off "
                "and trail stop on the remainder. Locks in gains before reversals.",
            )

        # Low win rate
        if wr < 40 and n_t >= 10:
            return (
                "thresholds",
                "Raise min_master_score floor to filter out weak setups",
                f"Win rate is {wr:.0f}% across {n_t} closed trades — below the 50% "
                f"breakeven threshold. Current Master Score floor of {ms_floor:.0f} is "
                f"too lenient.",
                f"Increase min_master_score to {min(85, ms_floor + 10):.0f} "
                f"to require higher-conviction setups.",
            )

        # Decent win rate but losing money → R:R issue
        if wr >= 50 and realized < 0:
            return (
                "exits",
                "Tighten stops + extend targets to improve R:R ratio",
                f"Win rate {wr:.0f}% is good but realized P&L is negative ${realized:.2f}. "
                "Losers are bigger than winners — stops are too wide OR targets are too close.",
                "In PaperTradingEngine.open_position, tighten stop_dist floor from 2% to "
                "1.5% AND raise default target multiplier from 1.5× to 2.0× of risk.",
            )

        # High exit count via TIME_STOP → positions sitting too long
        by_exit = diag.get("by_exit_reason", {})
        ts_count = by_exit.get("TIME_STOP", {}).get("total", 0)
        if ts_count >= 3 and ts_count / n_t > 0.3:
            return (
                "exits",
                "Reduce holding window — too many positions are dying from theta",
                f"{ts_count}/{n_t} trades exited via TIME_STOP. Positions are sitting "
                f"in their range without hitting T1 — wasted capital.",
                "In check_stops_and_targets, change the TIME_STOP threshold from "
                "10 trading days to 7. Frees up slots faster.",
            )

        # Default
        return (
            "new_check",
            "Add a pre-market gap filter to skip positions with overnight news",
            f"No specific pattern detected in {n_t} trades — start with a defensive "
            "addition. Pre-market gaps >5% indicate news that often invalidates "
            "the breakout setup.",
            "Add a check in open_position: if abs(open_price - prev_close) / prev_close "
            "> 0.05, skip this entry today (news-driven, unpredictable).",
        )

    def get_latest_suggestion(self) -> dict:
        """Return the most recent improvement suggestion."""
        try:
            row = self.conn.execute(
                "SELECT * FROM paper_suggestions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                return dict(row)
        except Exception:
            pass
        return {}

    def get_suggestion_history(self, limit: int = 30) -> list:
        """Return the last N suggestions."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_suggestions ORDER BY id DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
            out = []
            for r in rows or []:
                try:
                    out.append(dict(r))
                except Exception:
                    pass
            return out
        except Exception:
            return []

    def mark_suggestion_applied(self, suggestion_id: int) -> bool:
        try:
            self.conn.execute(
                "UPDATE paper_suggestions SET applied=1 WHERE id=?",
                (int(suggestion_id),)
            )
            return True
        except Exception:
            return False

    # ── Apply suggestions automatically ───────────────────────────────────────

    def set_min_master_score(self, new_score: float) -> bool:
        """Update min_master_score in paper_adjustments table."""
        try:
            self.conn.execute(
                "UPDATE paper_adjustments SET min_master_score=?, updated_at=? "
                "WHERE id=1",
                (float(new_score), datetime.utcnow().isoformat())
            )
            return True
        except Exception:
            return False

    def set_size_multiplier_cap(self, new_cap: float) -> bool:
        """Update size_multiplier_cap in paper_adjustments table."""
        try:
            self.conn.execute(
                "UPDATE paper_adjustments SET size_multiplier_cap=?, updated_at=? "
                "WHERE id=1",
                (float(new_cap), datetime.utcnow().isoformat())
            )
            return True
        except Exception:
            return False

    def add_to_blacklist(self, kind: str, entry: str) -> bool:
        """Append *entry* to sector_blacklist / pattern_blacklist / wyckoff_blacklist.

        kind: 'sector' | 'pattern' | 'wyckoff'
        """
        import json as _json
        col_map = {
            "sector":  "sector_blacklist",
            "pattern": "pattern_blacklist",
            "wyckoff": "wyckoff_blacklist",
        }
        col = col_map.get(kind)
        if not col or not entry:
            return False
        try:
            row = self.conn.execute(
                f"SELECT {col} FROM paper_adjustments WHERE id=1"
            ).fetchone()
            current = []
            if row:
                try:
                    current = _json.loads(_row_get(row, col, "[]") or "[]")
                except Exception:
                    current = []
            if entry not in current:
                current.append(entry)
            self.conn.execute(
                f"UPDATE paper_adjustments SET {col}=?, updated_at=? WHERE id=1",
                (_json.dumps(current), datetime.utcnow().isoformat())
            )
            return True
        except Exception:
            return False

    def apply_suggestion(self, suggestion_id: int) -> dict:
        """Auto-apply a stored suggestion by parsing its action text.

        Returns dict:
          ok       — True if the change was applied to the DB
          message  — what changed (or why it couldn't)
          category — the suggestion category
          before   — previous value (if applicable)
          after    — new value (if applicable)
        """
        import re as _re
        try:
            row = self.conn.execute(
                "SELECT * FROM paper_suggestions WHERE id=?",
                (int(suggestion_id),)
            ).fetchone()
        except Exception:
            row = None
        if not row:
            return {"ok": False, "message": "Suggestion not found"}

        category = (_row_get(row, "category", "") or "").lower().strip()
        action   = (_row_get(row, "action",   "") or "").strip()
        action_l = action.lower()
        adj = self.get_active_adjustments()

        # ── THRESHOLDS — min_master_score / size_multiplier_cap ──────────────
        if category == "thresholds":
            if "min_master_score" in action_l or "master score" in action_l:
                m = _re.search(r"(\d+(?:\.\d+)?)", action)
                if m:
                    new_val = float(m.group(1))
                    if not (0 <= new_val <= 100):
                        return {"ok": False,
                                "message": f"Parsed value {new_val} is out of range 0-100"}
                    before = float(adj.get("min_master_score", 65))
                    if self.set_min_master_score(new_val):
                        self.mark_suggestion_applied(suggestion_id)
                        return {"ok": True, "category": category,
                                "before": before, "after": new_val,
                                "message": f"min_master_score changed from {before:.1f} to {new_val:.1f}"}
                return {"ok": False,
                        "message": "Could not parse a numeric value from the action text"}
            if "size_multiplier_cap" in action_l or "size cap" in action_l or "sizing cap" in action_l:
                m = _re.search(r"(\d+(?:\.\d+)?)", action)
                if m:
                    new_val = float(m.group(1))
                    if new_val > 2:    # they typed it as a percentage e.g. 75
                        new_val = new_val / 100
                    if not (0 < new_val <= 1.5):
                        return {"ok": False, "message": f"Parsed cap {new_val} out of range"}
                    before = float(adj.get("size_multiplier_cap", 1.0))
                    if self.set_size_multiplier_cap(new_val):
                        self.mark_suggestion_applied(suggestion_id)
                        return {"ok": True, "category": category,
                                "before": before, "after": new_val,
                                "message": f"size_multiplier_cap changed from {before:.2f}× to {new_val:.2f}×"}
            return {"ok": False, "message": "Threshold suggestion couldn't be auto-parsed — apply manually"}

        # ── BLACKLISTS — sector / pattern / wyckoff ─────────────────────────
        if category == "blacklists":
            # Look for quoted entries in the action text
            quoted = _re.findall(r'["\']([^"\']+)["\']', action)
            kinds = []
            if "sector" in action_l:
                kinds.append("sector")
            if "pattern" in action_l:
                kinds.append("pattern")
            if "wyckoff" in action_l:
                kinds.append("wyckoff")
            if not kinds:
                kinds = ["sector"]
            added = []
            for kind in kinds:
                for entry in quoted:
                    if self.add_to_blacklist(kind, entry):
                        added.append(f"{kind}:{entry}")
            if added:
                self.mark_suggestion_applied(suggestion_id)
                return {"ok": True, "category": category,
                        "message": f"Blacklist additions: {', '.join(added)}"}
            return {"ok": False, "message": "No quoted blacklist entry found in action text"}

        # ── UNSUPPORTED categories ─────────────────────────────────────────
        return {
            "ok": False, "category": category,
            "message": ("This suggestion category requires manual code changes — "
                        "I can help you implement it but can't auto-apply.")
        }

    # ══════════════════════════════════════════════════════════════════════════
    # CONTINUOUS LEARNING — adapts after every closed trade, not just resets
    # ══════════════════════════════════════════════════════════════════════════

    # Configurable nudge amounts (small, conservative — accumulate over many trades)
    _NUDGE_WIN_THRESHOLD       = 60   # win-rate %  ≥ this → become more aggressive
    _NUDGE_LOSS_THRESHOLD      = 30   # win-rate %  ≤ this → become more conservative
    _NUDGE_LOOKBACK_TRADES     = 10   # how many recent closes to consider
    _NUDGE_STEP_AGGR           = 1.0  # min_master_score decrease per aggressive nudge
    _NUDGE_STEP_CAUT           = 2.0  # min_master_score increase per cautious nudge
    _NUDGE_MIN_MASTER          = 45.0
    _NUDGE_MAX_MASTER          = 85.0

    # Probationary blacklist (faster reversible blacklist)
    _PROBATION_LOSS_COUNT      = 2    # 2 losses in same bucket → probation
    _BLACKLIST_LOSS_COUNT      = 3    # 3 losses in same bucket → blacklist

    def _log_learn_event(self, event_type: str, detail: str = "",
                         before: str = "", after: str = ""):
        try:
            self.conn.execute(
                "INSERT INTO paper_learning_log "
                "(event_at, event_type, detail, before_value, after_value) "
                "VALUES (?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), event_type[:64],
                 str(detail)[:512], str(before)[:200], str(after)[:200])
            )
        except Exception:
            pass

    def _learn_state(self) -> dict:
        """Read (or seed) the persistent learn-state row."""
        try:
            row = self.conn.execute(
                "SELECT * FROM paper_learn_state WHERE id=1"
            ).fetchone()
            if row:
                return dict(row)
        except Exception:
            pass
        try:
            self.conn.execute(
                "INSERT INTO paper_learn_state (id, last_processed_trade_id) "
                "VALUES (1, 0)"
            )
        except Exception:
            pass
        return {"id": 1, "last_processed_trade_id": 0,
                "recent_wins": 0, "recent_losses": 0,
                "nudges_made": 0, "last_nudge_at": None, "notes": ""}

    def _bump_signal_perf(self, signal_type: str, signal_value: str,
                          is_win: bool, pnl: float):
        """Increment win/loss counters for a (type, value) signal cell."""
        signal_value = (signal_value or "").strip() or "Unknown"
        try:
            # Try update first
            row = self.conn.execute(
                "SELECT id, n_trades, n_wins, n_losses, sum_win_pnl, sum_loss_pnl "
                "FROM paper_signal_performance "
                "WHERE signal_type=? AND signal_value=?",
                (signal_type, signal_value)
            ).fetchone()
            if row:
                rid       = _row_get(row, "id")
                n_t       = int(_row_get(row, "n_trades",     0) or 0) + 1
                n_w       = int(_row_get(row, "n_wins",       0) or 0) + (1 if is_win else 0)
                n_l       = int(_row_get(row, "n_losses",     0) or 0) + (0 if is_win else 1)
                s_w_pnl   = float(_row_get(row, "sum_win_pnl",  0) or 0) + (pnl if is_win else 0)
                s_l_pnl   = float(_row_get(row, "sum_loss_pnl", 0) or 0) + (pnl if not is_win else 0)
                self.conn.execute(
                    "UPDATE paper_signal_performance "
                    "SET n_trades=?, n_wins=?, n_losses=?, "
                    "sum_win_pnl=?, sum_loss_pnl=?, last_updated=? "
                    "WHERE id=?",
                    (n_t, n_w, n_l, s_w_pnl, s_l_pnl,
                     datetime.utcnow().isoformat(), rid)
                )
            else:
                self.conn.execute(
                    "INSERT INTO paper_signal_performance "
                    "(signal_type, signal_value, n_trades, n_wins, n_losses, "
                    "sum_win_pnl, sum_loss_pnl, last_updated) "
                    "VALUES (?, ?, 1, ?, ?, ?, ?, ?)",
                    (signal_type, signal_value,
                     1 if is_win else 0,
                     0 if is_win else 1,
                     pnl if is_win else 0,
                     pnl if not is_win else 0,
                     datetime.utcnow().isoformat())
                )
        except Exception:
            pass

    def record_trade_outcome(self, trade: dict):
        """Update signal performance stats + trigger threshold/blacklist nudges
        based on this single closed trade.  Called from process_recent_closes()."""
        try:
            pnl    = float(trade.get("net_pnl", 0) or 0)
            is_win = pnl > 0

            # Bump every signal dimension we have on the trade
            for sig_type, sig_value in [
                ("sector",       trade.get("sector",  "")),
                ("pattern",      trade.get("pattern", "")),
                ("score_bucket", self._bucket_score(trade.get("explosive_score", 0))),
                ("prob_bucket",  self._bucket_prob (trade.get("breakout_prob",   0))),
                ("exit_reason",  trade.get("exit_reason", "")),
            ]:
                self._bump_signal_perf(sig_type, str(sig_value or ""),
                                        is_win, pnl)

            # ── Adaptive blacklisting per sector / pattern ──────────────────
            for sig_type, kind in [("sector", "sector"), ("pattern", "pattern")]:
                self._maybe_promote_blacklist(sig_type, kind,
                                              str(trade.get(sig_type, "") or ""))

            # ── Threshold nudge based on rolling win rate ───────────────────
            self._maybe_nudge_threshold(is_win)
        except Exception:
            pass

    @staticmethod
    def _bucket_score(score) -> str:
        try:
            s = float(score or 0)
        except Exception:
            return "Unknown"
        if   s >= 85: return "85+"
        elif s >= 75: return "75-84"
        elif s >= 65: return "65-74"
        elif s >= 50: return "50-64"
        else:          return "<50"

    @staticmethod
    def _bucket_prob(prob) -> str:
        try:
            p = float(prob or 0)
        except Exception:
            return "Unknown"
        if   p >= 80: return "80+"
        elif p >= 70: return "70-79"
        elif p >= 60: return "60-69"
        else:          return "<60"

    def _maybe_promote_blacklist(self, sig_type: str, kind: str, value: str):
        """If a signal value has accumulated too many losses, escalate it.
        Two-tier system:
          probation (size cap 0.3× for that value) → at _PROBATION_LOSS_COUNT
          blacklist (auto-skip)                    → at _BLACKLIST_LOSS_COUNT
        """
        if not value or value == "Unknown":
            return
        try:
            row = self.conn.execute(
                "SELECT n_trades, n_wins, n_losses FROM paper_signal_performance "
                "WHERE signal_type=? AND signal_value=?",
                (sig_type, value)
            ).fetchone()
            if not row:
                return
            n_t = int(_row_get(row, "n_trades", 0) or 0)
            n_w = int(_row_get(row, "n_wins",   0) or 0)
            n_l = int(_row_get(row, "n_losses", 0) or 0)
            wr  = n_w / n_t * 100 if n_t else 0

            adj = self.get_active_adjustments()
            bl_key  = f"{kind}_blacklist"
            current = list(adj.get(bl_key, []))

            # Promote to blacklist?
            if n_l >= self._BLACKLIST_LOSS_COUNT and wr < 30 and value not in current:
                self.add_to_blacklist(kind, value)
                self._log_learn_event(
                    "BLACKLIST_ADD",
                    f"{kind}='{value}' added to blacklist after {n_l}L/{n_w}W (WR {wr:.0f}%)",
                    before=", ".join(current) or "none",
                    after=", ".join(current + [value]),
                )
        except Exception:
            pass

    def _maybe_nudge_threshold(self, last_was_win: bool):
        """Rolling-window threshold adjustment.  After every NUDGE_LOOKBACK_TRADES
        closes, recompute win rate and nudge min_master_score accordingly."""
        st = self._learn_state()
        wins   = int(st.get("recent_wins",   0) or 0)
        losses = int(st.get("recent_losses", 0) or 0)
        if last_was_win:
            wins += 1
        else:
            losses += 1

        # Save rolling window
        try:
            self.conn.execute(
                "UPDATE paper_learn_state SET recent_wins=?, recent_losses=? WHERE id=1",
                (wins, losses)
            )
        except Exception:
            pass

        total = wins + losses
        if total < self._NUDGE_LOOKBACK_TRADES:
            return

        # We've hit the lookback window — compute & nudge
        wr = wins / total * 100 if total else 0
        adj = self.get_active_adjustments()
        cur_min = float(adj.get("min_master_score", 65))
        new_min = cur_min

        if wr <= self._NUDGE_LOSS_THRESHOLD:
            # Be more conservative
            new_min = min(self._NUDGE_MAX_MASTER, cur_min + self._NUDGE_STEP_CAUT)
            if new_min > cur_min:
                self.set_min_master_score(new_min)
                self._log_learn_event(
                    "NUDGE_RAISE_SCORE",
                    f"Last {total} trades WR {wr:.0f}% → raised min_master_score",
                    before=f"{cur_min:.0f}",
                    after=f"{new_min:.0f}",
                )
        elif wr >= self._NUDGE_WIN_THRESHOLD:
            # Be more aggressive
            new_min = max(self._NUDGE_MIN_MASTER, cur_min - self._NUDGE_STEP_AGGR)
            if new_min < cur_min:
                self.set_min_master_score(new_min)
                self._log_learn_event(
                    "NUDGE_LOWER_SCORE",
                    f"Last {total} trades WR {wr:.0f}% → lowered min_master_score",
                    before=f"{cur_min:.0f}",
                    after=f"{new_min:.0f}",
                )

        # Reset rolling window + bump nudge count
        try:
            self.conn.execute(
                "UPDATE paper_learn_state SET recent_wins=0, recent_losses=0, "
                "nudges_made = COALESCE(nudges_made,0) + 1, last_nudge_at=? "
                "WHERE id=1",
                (datetime.utcnow().isoformat(),)
            )
        except Exception:
            pass

    def process_recent_closes(self, max_per_cycle: int = 50) -> dict:
        """Find paper trades that have closed since the last call and update
        learning stats for each.  Idempotent — uses last_processed_trade_id
        to avoid re-processing.

        Returns dict with counts: n_processed, n_wins, n_losses.
        """
        st = self._learn_state()
        last_id = int(st.get("last_processed_trade_id", 0) or 0)

        try:
            rows = self.conn.execute(
                "SELECT id, ticker, pattern, sector, explosive_score, "
                "breakout_prob, net_pnl, exit_reason "
                "FROM paper_portfolio "
                "WHERE status='CLOSED' AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (last_id, int(max_per_cycle))
            ).fetchall()
        except Exception:
            return {"n_processed": 0, "n_wins": 0, "n_losses": 0, "error": "db read failed"}

        n_wins = 0
        n_losses = 0
        max_seen_id = last_id
        for r in rows or []:
            try:
                trade = dict(r)
            except Exception:
                continue
            pnl = float(trade.get("net_pnl", 0) or 0)
            if pnl > 0:
                n_wins += 1
            elif pnl < 0:
                n_losses += 1
            self.record_trade_outcome(trade)
            tid = int(trade.get("id") or 0)
            if tid > max_seen_id:
                max_seen_id = tid

        # Persist the watermark
        if max_seen_id > last_id:
            try:
                self.conn.execute(
                    "UPDATE paper_learn_state SET last_processed_trade_id=? WHERE id=1",
                    (max_seen_id,)
                )
            except Exception:
                pass

        return {"n_processed": len(rows or []), "n_wins": n_wins,
                "n_losses": n_losses, "watermark": max_seen_id}

    def get_signal_performance(self, signal_type: str = None,
                                  min_trades: int = 1) -> list:
        """Return per-signal performance rows for the UI / diagnostics."""
        try:
            if signal_type:
                rows = self.conn.execute(
                    "SELECT * FROM paper_signal_performance "
                    "WHERE signal_type=? AND n_trades>=? "
                    "ORDER BY n_trades DESC, n_wins DESC",
                    (signal_type, int(min_trades))
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM paper_signal_performance "
                    "WHERE n_trades>=? "
                    "ORDER BY signal_type, n_trades DESC",
                    (int(min_trades),)
                ).fetchall()
            out = []
            for r in rows or []:
                try:
                    d = dict(r)
                except Exception:
                    continue
                n_t = int(d.get("n_trades", 0) or 0)
                n_w = int(d.get("n_wins",   0) or 0)
                d["win_rate_pct"] = round(n_w / n_t * 100, 1) if n_t else 0.0
                d["expectancy"]   = (
                    round((float(d.get("sum_win_pnl") or 0) +
                           float(d.get("sum_loss_pnl") or 0)) / n_t, 2)
                    if n_t else 0.0
                )
                out.append(d)
            return out
        except Exception:
            return []

    def get_recent_learning_events(self, limit: int = 20) -> list:
        """Recent threshold nudges, blacklist additions, etc., for display."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_learning_log ORDER BY id DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
            return [dict(r) for r in rows or []]
        except Exception:
            return []
