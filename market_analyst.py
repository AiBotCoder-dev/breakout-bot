"""
market_analyst.py — Institutional-grade market briefing (probabilistic, regime-aware).

Transforms the bot from a binary signal-chaser into a professional desk analyst.
Instead of "buy/sell", it produces a weighted-probability view of the market the
way a hedge-fund analyst would: multi-timeframe structure, market internals,
regime classification, bull/bear/neutral probabilities, key levels, risk
assessment, recommended strategy, and explicit invalidation factors.

It SYNTHESISES inputs the bot already computes (regime detector, VIX, sector
heat, breadth) rather than inventing new indicators — the institutional edge is
in the synthesis and the honesty, not in a magic signal.

Output (generate_briefing) maps 1:1 to a professional trader report:
  regime · bull/bear/neutral % · bias · confidence · key levels · risk ·
  recommended strategy · trade opportunities · supporting reasons ·
  invalidation factors.
"""

from __future__ import annotations

from datetime import datetime, date, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except Exception:                       # pragma: no cover
    np = pd = yf = None


# Representative large-cap breadth basket (fast; ~40 names across sectors)
BREADTH_BASKET = [
    "AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","TSLA","AVGO","NFLX","ORCL",
    "CRM","ADBE","INTC","QCOM","JPM","BAC","WFC","GS","V","MA","UNH","LLY","JNJ",
    "ABBV","MRK","WMT","COST","HD","MCD","NKE","PG","KO","CAT","BA","XOM","CVX",
    "GE","HON","UPS",
]
# Ratio ETFs for internals
RATIO_TICKERS = ["SPY","RSP","XLY","XLP","HYG","TLT","^VIX"]


def _hist(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    if yf is None:
        return None
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw.dropna(subset=["Close"])
    except Exception:
        return None


def _rsi(s: pd.Series, p: int = 14) -> float:
    d = s.diff(); up = d.clip(lower=0).rolling(p).mean(); dn = (-d.clip(upper=0)).rolling(p).mean()
    v = (100 - 100 / (1 + up / dn.replace(0, np.nan))).iloc[-1]
    return float(v) if pd.notna(v) else 50.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. MULTI-TIMEFRAME STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
def _swings(highs: np.ndarray, lows: np.ndarray, lookback: int = 5) -> dict:
    """Detect higher-highs/higher-lows vs lower-highs/lower-lows over recent swings."""
    n = len(highs)
    if n < lookback * 3:
        return {"hh": False, "hl": False, "lh": False, "ll": False}
    # compare last two ~lookback-window extremes
    recent_hi = highs[-lookback:].max();  prior_hi = highs[-2*lookback:-lookback].max()
    recent_lo = lows[-lookback:].min();   prior_lo = lows[-2*lookback:-lookback].min()
    return {"hh": recent_hi > prior_hi, "hl": recent_lo > prior_lo,
            "lh": recent_hi < prior_hi, "ll": recent_lo < prior_lo}


def analyze_structure(df: pd.DataFrame, label: str) -> dict:
    """Classify trend structure for a given timeframe dataframe."""
    if df is None or len(df) < 25:
        return {"timeframe": label, "trend": "Unknown", "detail": "insufficient data"}
    c = df["Close"]; price = float(c.iloc[-1])
    sma20 = float(c.iloc[-20:].mean())
    sma50 = float(c.iloc[-50:].mean()) if len(c) >= 50 else sma20
    sw = _swings(df["High"].values, df["Low"].values)
    rng_hi = float(df["High"].iloc[-20:].max()); rng_lo = float(df["Low"].iloc[-20:].min())
    close_in_range = (price - rng_lo) / (rng_hi - rng_lo) * 100 if rng_hi > rng_lo else 50
    vol_now = float(df["Volume"].iloc[-5:].mean()) if "Volume" in df else 0
    vol_avg = float(df["Volume"].iloc[-20:].mean()) if "Volume" in df else 0
    vol_trend = "rising" if vol_avg and vol_now > vol_avg * 1.1 else (
                "falling" if vol_avg and vol_now < vol_avg * 0.9 else "flat")
    rsi = _rsi(c)
    ret20 = (price / float(c.iloc[-21]) - 1) * 100 if len(c) > 21 else 0

    above20, above50 = price > sma20, price > sma50
    # Trend classification
    if sw["hh"] and sw["hl"] and above20 and above50:
        trend = "Strong Bull Trend"
    elif (sw["hh"] or sw["hl"]) and above50:
        trend = "Weak Bull Trend"
    elif sw["lh"] and sw["ll"] and not above20 and not above50:
        trend = "Strong Bear Trend"
    elif (sw["lh"] or sw["ll"]) and not above50:
        trend = "Weak Bear Trend"
    elif abs(ret20) < 3 and 25 < close_in_range < 75:
        trend = "Range Bound"
    else:
        trend = "Transitional"

    # numeric lean for the probability model: +1 bull ... -1 bear
    lean = {"Strong Bull Trend": 1.0, "Weak Bull Trend": 0.5, "Range Bound": 0.0,
            "Transitional": 0.0, "Weak Bear Trend": -0.5, "Strong Bear Trend": -1.0
            }.get(trend, 0.0)
    return {
        "timeframe": label, "trend": trend, "lean": lean,
        "price": round(price, 2), "sma20": round(sma20, 2), "sma50": round(sma50, 2),
        "above20": above20, "above50": above50,
        "hh": sw["hh"], "hl": sw["hl"], "lh": sw["lh"], "ll": sw["ll"],
        "close_in_range_pct": round(close_in_range, 0),
        "volume_trend": vol_trend, "rsi": round(rsi, 0),
        "ret_20bar_pct": round(ret20, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. MARKET INTERNALS
# ══════════════════════════════════════════════════════════════════════════════
def _breadth() -> dict:
    """% of breadth basket above 50/200-SMA (participation)."""
    above50 = above200 = n = 0
    def _check(t):
        df = _hist(t, "1y", "1d")
        if df is None or len(df) < 200:
            return None
        c = df["Close"]; price = float(c.iloc[-1])
        return (price > float(c.iloc[-50:].mean()), price > float(c.iloc[-200:].mean()))
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(_check, BREADTH_BASKET):
            if r is None:
                continue
            n += 1; above50 += int(r[0]); above200 += int(r[1])
    return {
        "n": n,
        "pct_above_50":  round(above50 / n * 100, 0) if n else None,
        "pct_above_200": round(above200 / n * 100, 0) if n else None,
    }


def _internals() -> dict:
    """Breadth + equal-vs-cap weight + risk-on/off ratios + VIX."""
    out = {"breadth": _breadth()}
    # ratio ETFs
    px = {}
    for t in RATIO_TICKERS:
        df = _hist(t, "3mo", "1d")
        if df is not None and not df.empty:
            px[t] = df["Close"]
    def _ratio_trend(a, b):
        if a in px and b in px:
            r = (px[a] / px[b]).dropna()
            if len(r) > 21:
                return float(r.iloc[-1] / r.iloc[-21] - 1) * 100
        return None
    out["equal_vs_cap_20d"]   = _ratio_trend("RSP", "SPY")    # >0 = healthy broad breadth
    out["risk_appetite_20d"]  = _ratio_trend("XLY", "XLP")    # >0 = risk-on (discretionary>staples)
    out["credit_20d"]         = _ratio_trend("HYG", "TLT")    # >0 = credit risk-on
    out["vix"]                = float(px["^VIX"].iloc[-1]) if "^VIX" in px else None
    out["vix_20d_chg"]        = (float(px["^VIX"].iloc[-1] / px["^VIX"].iloc[-21] - 1) * 100
                                 if "^VIX" in px and len(px["^VIX"]) > 21 else None)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. KEY LEVELS
# ══════════════════════════════════════════════════════════════════════════════
def _key_levels(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    out = {}
    try:
        out["prev_day_high"] = round(float(daily["High"].iloc[-2]), 2)
        out["prev_day_low"]  = round(float(daily["Low"].iloc[-2]), 2)
        out["today_open"]    = round(float(daily["Open"].iloc[-1]), 2)
        out["last"]          = round(float(daily["Close"].iloc[-1]), 2)
    except Exception:
        pass
    try:
        out["prev_week_high"] = round(float(weekly["High"].iloc[-2]), 2)
        out["prev_week_low"]  = round(float(weekly["Low"].iloc[-2]), 2)
    except Exception:
        pass
    # Support/resistance from 60-day swing extremes + round levels
    try:
        recent = daily.iloc[-60:]
        out["resistance"] = round(float(recent["High"].max()), 2)
        out["support"]    = round(float(recent["Low"].min()), 2)
        # nearer intraday pivots: 20-day
        r20 = daily.iloc[-20:]
        out["near_resistance"] = round(float(r20["High"].max()), 2)
        out["near_support"]    = round(float(r20["Low"].min()), 2)
    except Exception:
        pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MARKET ANALYST — synthesis
# ══════════════════════════════════════════════════════════════════════════════
class MarketAnalyst:
    def __init__(self, conn=None, symbol: str = "SPY"):
        self.conn = conn
        self.symbol = symbol

    def generate_briefing(self, progress=None) -> dict:
        if progress:
            try: progress("Fetching multi-timeframe data…")
            except Exception: pass
        weekly = _hist(self.symbol, "2y", "1wk")
        daily  = _hist(self.symbol, "1y", "1d")
        intraday = _hist(self.symbol, "5d", "1h")

        w = analyze_structure(weekly, "Weekly")
        d = analyze_structure(daily, "Daily")
        it = analyze_structure(intraday, "Intraday (1h)")

        if progress:
            try: progress("Computing market internals + breadth…")
            except Exception: pass
        internals = _internals()
        levels = _key_levels(daily, weekly) if (daily is not None and weekly is not None) else {}

        # Reuse the existing regime detector
        regime = {"regime": "UNKNOWN", "label": "Unknown"}
        try:
            import trading_scanner as ts
            if daily is not None:
                regime = ts.MarketRegimeDetector.detect(daily)
        except Exception:
            pass

        # ── Probability model (weighted factor lean) ─────────────────────────
        factors = []
        def add(name, lean, weight, detail):
            factors.append({"name": name, "lean": lean, "weight": weight, "detail": detail})

        add("Weekly structure", w.get("lean", 0), 25, w.get("trend", "?"))
        add("Daily structure",  d.get("lean", 0), 20, d.get("trend", "?"))
        add("Intraday structure", it.get("lean", 0), 8, it.get("trend", "?"))

        # Momentum (SPY 20d)
        mom = regime.get("ret_20d", 0) or 0
        add("Momentum (20d)", float(np.clip(mom / 5.0, -1, 1)), 12, f"{mom:+.1f}% 20d")

        # Breadth
        b200 = internals["breadth"].get("pct_above_200")
        if b200 is not None:
            b_lean = (b200 - 50) / 50.0   # 50%→0, 100%→+1, 0%→-1
            add("Breadth (% > 200SMA)", float(np.clip(b_lean, -1, 1)), 13,
                f"{b200:.0f}% of basket above 200SMA")

        # Equal-vs-cap weight (broad participation)
        ev = internals.get("equal_vs_cap_20d")
        if ev is not None:
            add("Breadth health (RSP/SPY)", float(np.clip(ev / 2.0, -1, 1)), 6,
                f"equal-weight {ev:+.1f}% vs cap-weight (20d)")

        # Risk appetite
        ra = internals.get("risk_appetite_20d")
        if ra is not None:
            add("Risk appetite (XLY/XLP)", float(np.clip(ra / 3.0, -1, 1)), 6,
                f"discretionary/staples {ra:+.1f}% (20d)")
        cr = internals.get("credit_20d")
        if cr is not None:
            add("Credit (HYG/TLT)", float(np.clip(cr / 3.0, -1, 1)), 5,
                f"high-yield/treasuries {cr:+.1f}% (20d)")

        # Volatility (VIX) — high VIX leans bearish/uncertain
        vix = internals.get("vix")
        if vix is not None:
            v_lean = -float(np.clip((vix - 18) / 22.0, -1, 1))  # 18→0, 40→-1, low→+
            add("Volatility (VIX)", v_lean, 10, f"VIX {vix:.1f}")

        # Weighted net lean → probabilities
        tot_w = sum(f["weight"] for f in factors) or 1
        net = sum(f["lean"] * f["weight"] for f in factors) / tot_w   # -1..+1
        # Map net lean to bull/bear/neutral probabilities
        bull = max(0.0, net) ** 0.9
        bear = max(0.0, -net) ** 0.9
        neutral_strength = 1 - abs(net)
        raw = np.array([bull + 0.15, bear + 0.15, neutral_strength + 0.10])
        probs = raw / raw.sum()
        p_bull, p_bear, p_neutral = [round(float(x) * 100, 0) for x in probs]

        bias = ("Bullish" if p_bull >= max(p_bear, p_neutral) else
                "Bearish" if p_bear >= max(p_bull, p_neutral) else "Neutral")
        # Confidence = dominant prob × factor agreement
        agree = np.mean([1 if np.sign(f["lean"]) == np.sign(net) or f["lean"] == 0 else 0
                         for f in factors]) if factors else 0.5
        confidence = round(float(max(p_bull, p_bear, p_neutral) * (0.6 + 0.4 * agree)), 0)

        # ── Regime classification (spec's regime list) ───────────────────────
        mkt_regime = self._classify_regime(regime, internals, w, d)

        # ── Risk assessment ──────────────────────────────────────────────────
        risk = self._risk(vix, internals, w, d)

        # ── Recommended strategy (maps regime → bot's engines) ───────────────
        strategy = self._recommend(mkt_regime, bias, internals)

        # ── Reasons + invalidation ───────────────────────────────────────────
        reasons = self._reasons(factors, w, d, internals, regime)
        invalidation = self._invalidation(bias, levels, internals)

        # ── Trade opportunities (point to the validated engines) ─────────────
        opportunities = self._opportunities(mkt_regime, bias, vix)

        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "structure": {"weekly": w, "daily": d, "intraday": it},
            "internals": internals,
            "regime_raw": regime,
            "market_regime": mkt_regime,
            "prob_bull": p_bull, "prob_bear": p_bear, "prob_neutral": p_neutral,
            "bias": bias, "confidence": confidence,
            "key_levels": levels,
            "risk": risk,
            "recommended_strategy": strategy,
            "opportunities": opportunities,
            "factors": factors,
            "reasons": reasons,
            "invalidation": invalidation,
        }

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _classify_regime(regime, internals, w, d) -> dict:
        vix = internals.get("vix") or 18
        trend_up = regime.get("regime") in ("STRONG BULL", "BULL")
        trend_dn = regime.get("regime") == "BEAR"
        hi_vol = vix >= 25
        lo_vol = vix < 15
        wk = w.get("trend", "")
        rng = "Range Bound" in wk or "Range Bound" in d.get("trend", "")
        tags = []
        if trend_up:  tags.append("Trending Bull")
        if trend_dn:  tags.append("Trending Bear")
        if hi_vol:    tags.append("High Volatility")
        if lo_vol:    tags.append("Low Volatility")
        if rng:       tags.append("Mean-Reversion / Consolidation")
        if not trend_up and not trend_dn and not rng:
            tags.append("Transitional")
        # risk on/off
        ra = internals.get("risk_appetite_20d")
        if ra is not None:
            tags.append("Risk-On" if ra > 0 else "Risk-Off")
        primary = (tags[0] if tags else "Undetermined")
        return {"primary": primary, "tags": tags, "vix": round(vix, 1)}

    @staticmethod
    def _risk(vix, internals, w, d) -> dict:
        score = 0
        if vix is not None:
            score += 0 if vix < 15 else 1 if vix < 20 else 2 if vix < 30 else 3
        if internals.get("vix_20d_chg") and internals["vix_20d_chg"] > 25:
            score += 1
        if d.get("trend", "").startswith("Strong Bear") or w.get("trend", "").startswith("Strong Bear"):
            score += 1
        b200 = internals["breadth"].get("pct_above_200")
        if b200 is not None and b200 < 40:
            score += 1
        level = ("Low" if score <= 1 else "Moderate" if score <= 2
                 else "Elevated" if score <= 4 else "Extreme")
        return {"level": level, "score": score}

    @staticmethod
    def _recommend(mkt_regime, bias, internals) -> dict:
        primary = mkt_regime["primary"]; vix = mkt_regime["vix"]
        if "Trending Bull" in primary:
            s = ("Momentum longs (validated p=0.004). Trade liquid leaders + "
                 "PEAD post-earnings drift. Calls on strength. Puts holstered.")
            engines = ["Momentum Leaders", "PEAD", "Momentum Calls"]
        elif "Trending Bear" in primary:
            s = ("Capital preservation. Reduce size. Hunt put setups on high-beta "
                 "names (regime gate now OPEN). Wait for panic capitulation to buy.")
            engines = ["Put Engine", "Panic Detector"]
        elif "High Volatility" in primary:
            s = ("Volatility elevated — smaller size, wider stops. Best edge is the "
                 "Panic Detector (buy extreme fear). Avoid chasing.")
            engines = ["Panic Detector", "Short-Term Reversal"]
        elif "Mean-Reversion" in primary or "Consolidation" in primary:
            s = ("Range conditions — fade extremes, don't chase breakouts (they "
                 "fail in ranges). Lighter activity, wait for resolution.")
            engines = ["Short-Term Reversal"]
        else:
            s = ("Mixed/transitional — require higher conviction (prob ≥ 65%). "
                 "Reduce frequency, prioritize capital preservation.")
            engines = ["Best Trades NOW (high-quality only)"]
        return {"summary": s, "engines": engines}

    @staticmethod
    def _opportunities(mkt_regime, bias, vix) -> list:
        ops = []
        if bias == "Bullish":
            ops.append("Long momentum leaders on pullbacks to rising 20-day average")
            ops.append("Slightly-OTM calls on the strongest liquid names (Best Trades tab)")
        elif bias == "Bearish":
            ops.append("Puts on high-beta crash-prone names (COIN/PLTR/SMCI) — regime permitting")
            ops.append("Defensive: trim longs into resistance, raise cash")
        else:
            ops.append("Stand aside / reduce size — wait for directional confirmation")
        if vix is not None and vix >= 30:
            ops.append("⚡ Panic-buy setup: extreme fear historically → strong 20-60d forward returns")
        return ops

    @staticmethod
    def _reasons(factors, w, d, internals, regime) -> list:
        out = []
        for f in sorted(factors, key=lambda x: abs(x["lean"]) * x["weight"], reverse=True)[:6]:
            direction = "bullish" if f["lean"] > 0.1 else "bearish" if f["lean"] < -0.1 else "neutral"
            out.append(f"{f['name']}: {f['detail']} → {direction}")
        return out

    @staticmethod
    def _invalidation(bias, levels, internals) -> list:
        out = []
        if bias == "Bullish":
            if levels.get("prev_day_low"):
                out.append(f"Daily close below prior-day low (${levels['prev_day_low']}) "
                           f"breaks the short-term structure")
            if levels.get("prev_week_low"):
                out.append(f"Weekly close below prior-week low (${levels['prev_week_low']}) "
                           f"flips the weekly trend bearish")
            out.append("VIX spiking >25 with breadth collapsing would invalidate the bullish lean")
        elif bias == "Bearish":
            if levels.get("prev_day_high"):
                out.append(f"Daily close above prior-day high (${levels['prev_day_high']}) "
                           f"negates the bearish structure")
            out.append("Breadth thrust (>70% back above 50SMA) + VIX collapse would flip bullish")
        else:
            out.append("A decisive close outside the recent range (either direction) resolves the neutral stance")
        return out
