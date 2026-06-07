"""
command_center.py — One-click "what do I do today" aggregator.

The bot grew a lot of engines. This collapses them into ONE prioritized daily
action view. run_morning_scan() runs the essential live checks (market read,
panic, top momentum names) and READS the rest from what the monitor already
persisted (best options, whale picks, VIP catalysts, goal), then synthesises a
single GO / CAUTION / STAND-DOWN verdict and a short action line.

Priority order (most important first):
  1. Market read    — bias / regime / risk  (go/no-go)
  2. Panic signal   — highest-conviction buy if extreme fear is firing
  3. Top stocks     — strongest momentum leaders to buy
  4. Top option     — best validated options play (read from persisted scan)
  5. Catalysts      — fresh VIP posts + whale picks
  6. Goal           — am I on pace
"""

from __future__ import annotations

from datetime import datetime, timedelta


def _read(conn, sql, args=()):
    try:
        return conn.execute(sql, args).fetchall()
    except Exception:
        return []


def _g(r, key, idx):
    return r.get(key) if hasattr(r, "get") else r[idx]


def run_morning_scan(conn, progress=None, watchlist=None) -> dict:
    out = {}

    # 1) Market read (live, ~60s) ─────────────────────────────────────────────
    if progress:
        try: progress("Reading the market (structure + internals + regime)…")
        except Exception: pass
    market = {}
    try:
        from market_analyst import MarketAnalyst
        b = MarketAnalyst(conn).generate_briefing()
        market = {
            "bias": b["bias"], "regime": b["market_regime"]["primary"],
            "risk": b["risk"]["level"], "confidence": b["confidence"],
            "prob_bull": b["prob_bull"], "prob_bear": b["prob_bear"],
            "prob_neutral": b["prob_neutral"],
            "strategy": b["recommended_strategy"]["summary"],
            "vix": b["internals"].get("vix"),
            "invalidation": (b["invalidation"][0] if b["invalidation"] else ""),
        }
    except Exception as e:
        market = {"bias": "?", "regime": "?", "risk": "?", "error": str(e)}
    out["market"] = market

    # 2) Panic signal (live, cheap) ───────────────────────────────────────────
    if progress:
        try: progress("Checking for panic / capitulation edge…")
        except Exception: pass
    panic = []
    try:
        from panic_detector import PanicDetector
        st = PanicDetector(conn).status()
        panic = [s for s in st.get("signatures", []) if s.get("currently_fired")]
    except Exception:
        pass
    out["panic"] = panic

    # 3) Top momentum stocks to buy (live, ~60s) ──────────────────────────────
    if progress:
        try: progress("Ranking momentum leaders to buy…")
        except Exception: pass
    top_stocks = []
    try:
        from momentum_strategy import MomentumStrategy
        ranked = MomentumStrategy(conn).rank(top_n=6, min_mom_6m=0.05)
        for r in ranked[:6]:
            top_stocks.append({
                "ticker": r["ticker"], "price": r["price"],
                "mom_6m": r["mom_6m"], "mom_3m": r["mom_3m"], "rsi": r["rsi"],
                "stop": r["stop"], "extended": r.get("extended", False),
            })
    except Exception:
        pass
    out["top_stocks"] = top_stocks

    # 4) Top options play (read persisted best_options_trades) ─────────────────
    if progress:
        try: progress("Reading best options setups…")
        except Exception: pass
    options = []
    for r in _read(conn, "SELECT ticker, strike, option_type, expiry, dte, premium, "
                   "quality_score, quality_grade, decision, sources FROM "
                   "best_options_trades ORDER BY quality_score DESC LIMIT 3"):
        options.append({
            "ticker": _g(r, "ticker", 0), "strike": _g(r, "strike", 1),
            "type": _g(r, "option_type", 2), "expiry": _g(r, "expiry", 3),
            "dte": _g(r, "dte", 4), "premium": _g(r, "premium", 5),
            "score": _g(r, "quality_score", 6), "grade": _g(r, "quality_grade", 7),
            "decision": _g(r, "decision", 8), "sources": _g(r, "sources", 9),
        })
    out["options"] = options

    # 5) Fresh catalysts (VIP posts + whale picks) ────────────────────────────
    catalysts = []
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    for r in _read(conn, "SELECT vip_name, title, text, tickers, sentiment FROM "
                   "vip_posts WHERE fetched_at >= ? AND tickers <> '' "
                   "ORDER BY fetched_at DESC LIMIT 4", (cutoff,)):
        catalysts.append({
            "type": "VIP", "who": _g(r, "vip_name", 0),
            "text": (_g(r, "title", 1) or _g(r, "text", 2) or "")[:120],
            "tickers": _g(r, "tickers", 3), "sentiment": _g(r, "sentiment", 4)})
    whale = []
    for r in _read(conn, "SELECT ticker, whale_score, key_signal, entry_now, stop, "
                   "target FROM whale_watchlist ORDER BY whale_score DESC LIMIT 3"):
        whale.append({
            "ticker": _g(r, "ticker", 0), "score": _g(r, "whale_score", 1),
            "signal": _g(r, "key_signal", 2), "entry": _g(r, "entry_now", 3),
            "stop": _g(r, "stop", 4), "target": _g(r, "target", 5)})
    out["catalysts"] = catalysts
    out["whale"] = whale

    # 6) Goal status (read persisted) ─────────────────────────────────────────
    goal = {}
    try:
        from goal_tracker import GoalTracker
        import trading_scanner as ts
        paper = ts.PaperTradingEngine(conn)
        goal = GoalTracker(conn).scorecard(paper)
    except Exception:
        pass
    out["goal"] = goal

    # 7) Macro event risk (jobs/CPI/PCE/FOMC) ─────────────────────────────────
    macro = {}
    try:
        from macro_engine import event_risk, upcoming_events
        macro = event_risk()
        macro["upcoming"] = upcoming_events(days_ahead=14)[:4]
    except Exception:
        pass
    out["macro"] = macro

    # ── Synthesise verdict ────────────────────────────────────────────────────
    out["verdict"], out["action"] = _verdict(out)
    out["as_of"] = datetime.utcnow().isoformat()
    return out


def _verdict(out) -> tuple:
    m = out.get("market", {})
    bias = m.get("bias", "?"); risk = m.get("risk", "?")
    macro = out.get("macro", {})
    # Imminent binary macro event overrides everything except an active panic-buy
    if not out.get("panic") and macro.get("level") == "HIGH":
        return ("⏸ WAIT — MAJOR DATA IMMINENT",
                f"{macro.get('advice','Major macro release imminent.')} "
                f"Hold dry powder — the data can override every technical setup.")
    if out.get("panic"):
        return ("🚨 AGGRESSIVE BUY",
                "Panic signal firing — historically the strongest buy. Scale into "
                "SPY / momentum leaders / calls on a 20-60 day horizon.")
    if bias == "Bearish" or risk == "Extreme":
        return ("🛑 STAND DOWN / DEFENSIVE",
                "Bias bearish or risk extreme. Preserve capital — no new longs, "
                "trim into strength, consider hedges. Wait for the setup to improve.")
    if bias == "Bullish" and risk in ("Low", "Moderate"):
        n = len(out.get("top_stocks", []))
        return ("✅ GO — BUY MOMENTUM LEADERS",
                f"Bullish + manageable risk. Buy the top momentum names below "
                f"({n} ranked). Take the A/B options setups if any. Normal sizing.")
    return ("⚠️ CAUTION — HIGH CONVICTION ONLY",
            "Mixed / neutral / elevated risk. Trade only the highest-conviction "
            "setups (prob ≥ 65%). Reduce size and frequency. Capital preservation first.")
