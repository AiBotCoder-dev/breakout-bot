"""
sector_analysis.py — Sector rotation, hype detection, and fundamentals divergence.

WHAT THIS DOES
--------------
Three connected analyses, all data-driven (no opinions):

1. SECTOR HEAT — relative strength of every major sector vs SPY across 1/3/6
   month windows, combined with:
     • LEADER DENSITY — how many of the bot's top momentum names cluster in
       each sector (the AI-complex signature when AMD/MU/SMCI/etc lit up)
     • BREADTH — % of sector members above their 200-SMA
     • COMPOSITE HEAT SCORE — 0-100 grade per sector

2. EMERGING SECTORS — sectors with positive RS but NOT yet at concentration
   peak. This is the "what's quietly rotating in" signal — the configuration
   semis showed in mid-2023 before the AI explosion really got going.

3. FUNDAMENTALS DIVERGENCE — stocks with strong fundamentals (low PEG, growing
   EPS, healthy FCF, expanding margins) trading at WEAK prices. The "great
   business, market hasn't noticed yet" screen. yfinance .info gives us
   enough to do this honestly, with caveats noted in the dashboard.

HONEST LIMITS
-------------
• yfinance fundamentals data is sometimes stale or incomplete; this isn't an
  institutional-grade quant screen.
• "Next sector to explode" is genuinely hard; this module reports what's
  observably rotating in. Causation/timing of the actual boom is unknowable.
"""

from __future__ import annotations

import math
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except Exception:                       # pragma: no cover
    np = pd = yf = None


# ══════════════════════════════════════════════════════════════════════════════
# SECTOR DEFINITIONS — ETF + representative basket per sector
# ══════════════════════════════════════════════════════════════════════════════
SECTORS = {
    "Semis / AI Hardware": {
        "etf": "SMH",
        "members": ["NVDA","AMD","MU","AVGO","ASML","TSM","LRCX","KLAC","AMAT",
                    "QCOM","INTC","TXN","MRVL","ARM","ON","MCHP","SWKS","SMCI"],
    },
    "Mega-cap Tech": {
        "etf": "QQQ",
        "members": ["AAPL","MSFT","GOOGL","AMZN","META","NFLX","ORCL","ADBE","CRM"],
    },
    "Software / Cloud": {
        "etf": "IGV",
        "members": ["NOW","CRWD","DDOG","NET","PANW","SNPS","CDNS","ZS","SNOW",
                    "MDB","TEAM","WDAY","INTU","ADSK"],
    },
    "Financials": {
        "etf": "XLF",
        "members": ["JPM","BAC","WFC","GS","MS","C","SCHW","AXP","V","MA","BLK","SPGI"],
    },
    "Healthcare / Pharma": {
        "etf": "XLV",
        "members": ["UNH","LLY","JNJ","ABBV","MRK","PFE","TMO","ABT","DHR","AMGN",
                    "ISRG","VRTX"],
    },
    "Energy": {
        "etf": "XLE",
        "members": ["XOM","CVX","COP","SLB","EOG","MPC","PSX","OXY","HAL","DVN"],
    },
    "Industrials / Defense": {
        "etf": "XLI",
        "members": ["CAT","DE","BA","GE","HON","UPS","RTX","LMT","NOC","GD","TT"],
    },
    "Consumer Discretionary": {
        "etf": "XLY",
        "members": ["AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","BKNG","ABNB","DKNG"],
    },
    "Consumer Staples": {
        "etf": "XLP",
        "members": ["WMT","PG","KO","PEP","COST","MO","PM","CL"],
    },
    "Utilities / Nuclear": {
        "etf": "XLU",
        "members": ["NEE","SO","DUK","CEG","VST","AEP","D","SRE","XEL","NRG"],
    },
    "Communication Services": {
        "etf": "XLC",
        "members": ["GOOGL","META","NFLX","DIS","CMCSA","T","VZ","TMUS"],
    },
    "Crypto / Bitcoin Proxy": {
        "etf": None,
        "members": ["COIN","MSTR","MARA","RIOT","CLSK","HUT","HOOD"],
    },
    "Quantum Computing": {
        "etf": None,
        "members": ["IONQ","RGTI","QUBT","ARQQ"],
    },
    "Drones / Autonomy": {
        "etf": None,
        "members": ["AVAV","KTOS","RCAT","UMAC","ONDS","AIRO"],
    },
    "Materials / Mining": {
        "etf": "XLB",
        "members": ["LIN","FCX","NEM","NUE","SCCO","APD","ECL"],
    },
    "Biotech (Small/Mid)": {
        "etf": "XBI",
        "members": ["VRTX","REGN","ALNY","BIIB","MRNA","BNTX","ILMN","VKTX","ARWR"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _hist(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    if yf is None:
        return None
    try:
        raw = yf.download(ticker, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw.dropna(subset=["Close"])
    except Exception:
        return None


def _return(close: pd.Series, days: int) -> float | None:
    if close is None or len(close) <= days:
        return None
    try:
        return float(close.iloc[-1] / close.iloc[-1 - days] - 1) * 100
    except Exception:
        return None


def _info(ticker: str) -> dict:
    if yf is None:
        return {}
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTOR HEAT
# ══════════════════════════════════════════════════════════════════════════════
def sector_relative_strength(spy_close: pd.Series, sec_close: pd.Series) -> dict:
    """% return of sector minus SPY at 21/63/126 trading days."""
    out = {}
    for label, n in [("rs_1m", 21), ("rs_3m", 63), ("rs_6m", 126)]:
        sec_r = _return(sec_close, n)
        spy_r = _return(spy_close, n)
        out[label] = None if (sec_r is None or spy_r is None) else round(sec_r - spy_r, 2)
        out[label.replace("rs", "ret")] = sec_r
    return out


def sector_breadth(members: list) -> dict:
    """% of members trading above their 200-SMA + average 6m return."""
    if not members:
        return {"breadth_above_200": None, "avg_ret_6m": None, "n_evaluated": 0}
    rets = []
    above = 0
    n_eval = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_hist, t, "1y"): t for t in members}
        for fut in as_completed(futs):
            df = fut.result()
            if df is None or len(df) < 210:
                continue
            n_eval += 1
            c = df["Close"]
            sma200 = c.iloc[-200:].mean()
            if c.iloc[-1] > sma200:
                above += 1
            r6 = _return(c, 126)
            if r6 is not None:
                rets.append(r6)
    return {
        "breadth_above_200": round(above / n_eval * 100, 1) if n_eval else None,
        "avg_ret_6m":        round(sum(rets) / len(rets), 2) if rets else None,
        "n_evaluated":       n_eval,
    }


def leader_density(conn) -> dict:
    """For each sector, how many of the bot's top-30 momentum names belong to it."""
    try:
        from momentum_strategy import MomentumStrategy
        ranked = MomentumStrategy(conn).rank(top_n=30, min_mom_6m=0.0)
    except Exception:
        return {}
    by_sec = {label: 0 for label in SECTORS}
    for r in ranked:
        tk = r["ticker"]
        for label, info in SECTORS.items():
            if tk in info["members"] or tk == info.get("etf"):
                by_sec[label] += 1
                break
    return by_sec


def _heat_score(rs_3m: float | None, leader_count: int, breadth: float | None) -> int:
    """0-100 composite. Penalizes negative RS, rewards leader clustering + breadth."""
    s = 0
    if rs_3m is not None:
        if rs_3m >= 15:   s += 40
        elif rs_3m >= 8:  s += 30
        elif rs_3m >= 3:  s += 20
        elif rs_3m >= 0:  s += 10
        else:             s -= max(-10, int(rs_3m))
    if leader_count >= 6: s += 35
    elif leader_count >= 3: s += 25
    elif leader_count >= 1: s += 10
    if breadth is not None:
        if breadth >= 80: s += 25
        elif breadth >= 60: s += 15
        elif breadth >= 40: s += 5
    return max(0, min(100, s))


def sector_heat_report(conn, progress=None) -> list:
    """Full sector heat table: ETF, RS, breadth, leader density, composite score."""
    spy_df = _hist("SPY", "1y")
    if spy_df is None:
        return []
    spy_close = spy_df["Close"]
    densities = leader_density(conn)

    rows = []
    items = list(SECTORS.items())
    for i, (label, info) in enumerate(items):
        if progress:
            try:
                progress(i + 1, len(items), label)
            except Exception:
                pass
        etf = info.get("etf")
        rs_data = {"rs_1m": None, "rs_3m": None, "rs_6m": None,
                   "ret_1m": None, "ret_3m": None, "ret_6m": None}
        if etf:
            etf_df = _hist(etf, "1y")
            if etf_df is not None:
                rs_data = sector_relative_strength(spy_close, etf_df["Close"])
        # If no ETF, derive RS from average member returns
        b = sector_breadth(info["members"])
        if not etf and b["avg_ret_6m"] is not None and b.get("n_evaluated", 0) > 0:
            spy_6m = _return(spy_close, 126)
            if spy_6m is not None:
                rs_data["rs_6m"] = round(b["avg_ret_6m"] - spy_6m, 2)
                rs_data["ret_6m"] = b["avg_ret_6m"]
        leaders = densities.get(label, 0)
        heat = _heat_score(rs_data["rs_3m"] if rs_data["rs_3m"] is not None else rs_data.get("rs_6m"),
                           leaders, b["breadth_above_200"])
        rows.append({
            "sector":         label,
            "etf":            etf,
            **rs_data,
            "breadth_above_200": b["breadth_above_200"],
            "avg_ret_6m_basket": b["avg_ret_6m"],
            "n_evaluated":    b["n_evaluated"],
            "leaders_in_top30": leaders,
            "heat_score":     heat,
        })
    rows.sort(key=lambda r: r["heat_score"], reverse=True)
    return rows


def emerging_sectors(rows: list, max_results: int = 5) -> list:
    """
    Sectors with POSITIVE RS but NOT yet at peak concentration. The configuration
    a sector shows BEFORE its boom: relative strength building, leader count low
    (i.e., not yet hyped), breadth healthy.
    Filters: rs_3m > 0, leaders_in_top30 <= 3, breadth >= 50, heat 30-65 (warming
    but not red-hot).
    """
    out = []
    for r in rows:
        rs = r.get("rs_3m") if r.get("rs_3m") is not None else r.get("rs_6m")
        b  = r.get("breadth_above_200")
        if rs is None or rs <= 0:
            continue
        if (r.get("leaders_in_top30") or 0) > 3:
            continue
        if b is not None and b < 50:
            continue
        if not (30 <= r["heat_score"] <= 65):
            continue
        out.append(r)
    out.sort(key=lambda r: (r.get("rs_3m") or r.get("rs_6m") or 0), reverse=True)
    return out[:max_results]


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTALS DIVERGENCE — strong business, weak price
# ══════════════════════════════════════════════════════════════════════════════
def _fundamental_snapshot(ticker: str) -> dict | None:
    info = _info(ticker)
    if not info:
        return None
    df = _hist(ticker, "1y")
    if df is None or len(df) < 180:
        return None
    c = df["Close"]
    ret_6m = _return(c, 126)
    ret_3m = _return(c, 63)
    try:
        pe_fwd = float(info.get("forwardPE") or 0) or None
        pe_trail = float(info.get("trailingPE") or 0) or None
        peg = float(info.get("pegRatio") or 0) or None
        eps_growth = float(info.get("earningsQuarterlyGrowth") or 0) or None
        rev_growth = float(info.get("revenueGrowth") or 0) or None
        gross_m = float(info.get("grossMargins") or 0) or None
        op_m = float(info.get("operatingMargins") or 0) or None
        profit_m = float(info.get("profitMargins") or 0) or None
        roe = float(info.get("returnOnEquity") or 0) or None
        fcf = float(info.get("freeCashflow") or 0) or None
        mkt_cap = float(info.get("marketCap") or 0) or None
        fcf_yield = (fcf / mkt_cap * 100) if (fcf and mkt_cap and mkt_cap > 0) else None
        sector = str(info.get("sector") or "")
    except Exception:
        return None

    return {
        "ticker":     ticker.upper(),
        "sector":     sector,
        "price":      round(float(c.iloc[-1]), 2),
        "ret_3m":     ret_3m,
        "ret_6m":     ret_6m,
        "pe_fwd":     pe_fwd,
        "pe_trail":   pe_trail,
        "peg":        peg,
        "eps_growth_qoq": eps_growth,
        "rev_growth": rev_growth,
        "gross_m":    gross_m,
        "op_m":       op_m,
        "profit_m":   profit_m,
        "roe":        roe,
        "fcf_yield_pct": fcf_yield,
    }


def _divergence_score(s: dict) -> int:
    """
    0-100. Higher = stronger fundamentals + weaker recent price (divergence).
    Reward: low PE, low PEG, growing EPS, healthy margins, high FCF yield
    Penalty: stock already up a lot (already discovered)
    """
    pts = 0
    # Valuation (lower = better)
    if s.get("pe_fwd"):
        if s["pe_fwd"] < 15: pts += 20
        elif s["pe_fwd"] < 25: pts += 12
        elif s["pe_fwd"] < 40: pts += 4
    if s.get("peg"):
        if 0 < s["peg"] < 1.0: pts += 20
        elif s["peg"] < 1.5: pts += 12
        elif s["peg"] < 2.0: pts += 4
    # Growth
    if s.get("eps_growth_qoq"):
        if s["eps_growth_qoq"] > 0.30: pts += 20
        elif s["eps_growth_qoq"] > 0.15: pts += 14
        elif s["eps_growth_qoq"] > 0.05: pts += 6
    if s.get("rev_growth"):
        if s["rev_growth"] > 0.20: pts += 12
        elif s["rev_growth"] > 0.10: pts += 6
    # Quality
    if s.get("op_m"):
        if s["op_m"] > 0.25: pts += 10
        elif s["op_m"] > 0.15: pts += 6
        elif s["op_m"] > 0.05: pts += 2
    if s.get("fcf_yield_pct"):
        if s["fcf_yield_pct"] > 5: pts += 12
        elif s["fcf_yield_pct"] > 2: pts += 6
    # Price-divergence reward (cheap because nobody's noticed)
    r6 = s.get("ret_6m")
    if r6 is not None:
        if -25 < r6 < 0:  pts += 10   # mildly down — best divergence zone
        elif 0 <= r6 < 10: pts += 6   # flat
        elif r6 >= 30:    pts -= 10   # already hot
    return max(0, min(100, pts))


def find_hidden_gems(universe: list | None = None, top_n: int = 15,
                     progress=None) -> list:
    """
    The SOFI profile: STRONG fundamentals + WEAK technicals (below 200-SMA).
    These are 'great business, broken chart' names — not buys YET, but prime
    reversal-watch candidates. Flags reversal_ready when price reclaims the
    50-SMA (the validated turn trigger).
    """
    if universe is None:
        try:
            from momentum_strategy import LIQUID_UNIVERSE
            universe = LIQUID_UNIVERSE
        except Exception:
            universe = []
    results = []
    uni = [t for t in universe if "." not in t]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_fundamental_snapshot, t): t for t in uni}
        for i, fut in enumerate(as_completed(futs), 1):
            if progress:
                try: progress(i, len(futs), futs[fut])
                except Exception: pass
            s = fut.result()
            if not s:
                continue
            # Strong fundamentals filter
            pe = s.get("pe_fwd"); peg = s.get("peg"); eps_g = s.get("eps_growth_qoq")
            op = s.get("op_m"); rev_g = s.get("rev_growth")
            strong = 0
            if pe and pe < 30: strong += 1
            if peg and 0 < peg < 1.8: strong += 1
            if eps_g and eps_g > 0.10: strong += 1
            if rev_g and rev_g > 0.10: strong += 1
            if op and op > 0.10: strong += 1
            if strong < 3:                  # need genuinely strong fundamentals
                continue
            # Weak technicals: pull price vs 200/50 SMA
            df = _hist(s["ticker"], "1y")
            if df is None or len(df) < 210:
                continue
            c = df["Close"]; price = float(c.iloc[-1])
            sma50 = float(c.iloc[-50:].mean()); sma200 = float(c.iloc[-200:].mean())
            below_200 = price < sma200
            if not below_200:               # must be technically weak (the divergence)
                continue
            drawdown = (price / float(c.iloc[-252:].max()) - 1) * 100 if len(c) >= 252 else 0
            reversal_ready = price > sma50   # reclaimed the 50 = turn may be starting
            results.append({
                "ticker": s["ticker"], "sector": s.get("sector", ""),
                "price": round(price, 2), "pe_fwd": pe, "peg": peg,
                "eps_growth_qoq": eps_g, "rev_growth": rev_g, "op_m": op,
                "drawdown_pct": round(drawdown, 1),
                "below_200": below_200, "reversal_ready": reversal_ready,
                "sma50": round(sma50, 2), "sma200": round(sma200, 2),
                "fundamental_strength": strong,
            })
    # reversal-ready first, then deepest discount
    results.sort(key=lambda r: (not r["reversal_ready"], r["drawdown_pct"]))
    return results[:top_n]


def find_fundamental_divergence(universe: list | None = None,
                                top_n: int = 15, min_score: int = 50,
                                progress=None) -> list:
    """Scan universe, score each on fundamentals + price divergence, return top N."""
    if universe is None:
        try:
            from momentum_strategy import LIQUID_UNIVERSE
            universe = LIQUID_UNIVERSE
        except Exception:
            universe = []
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_fundamental_snapshot, t): t for t in universe if "." not in t}
        for i, fut in enumerate(as_completed(futs), 1):
            if progress:
                try:
                    progress(i, len(futs), futs[fut])
                except Exception:
                    pass
            s = fut.result()
            if not s:
                continue
            s["divergence_score"] = _divergence_score(s)
            if s["divergence_score"] >= min_score:
                results.append(s)
    results.sort(key=lambda r: r["divergence_score"], reverse=True)
    return results[:top_n]
