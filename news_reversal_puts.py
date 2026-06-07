"""
news_reversal_puts.py — Puts on the edge that SURVIVED testing: news reversals.

overextended_backtest.py was blunt: shorting overextended stocks LOSES —
parabolic (+4.8%/5d) and stretched (+4.3%/5d) names KEEP RISING ("overbought
gets more overbought"). So this engine deliberately does NOT short strength.

The put edge is EVENT-DRIVEN. A stock gaps down on bad news regardless of the
chart. This engine surfaces puts when:

  1. NEWS_REVERSAL  — a fresh HIGH-IMPACT NEGATIVE catalyst (news agent or a
     bearish VIP/Trump-Fed post) on a liquid name, CONFIRMED by price actually
     breaking down (below the 20-EMA / gapping down). News + confirmation, not
     news alone.
  2. RSI2_PULLBACK  — extreme 2-day overbought (RSI2 > 98): a WEAK 1-day mean-
     reversion edge (61% down-rate in testing). Secondary, 1-day puts only.

Each setup produces an affordable weekly PUT via catalyst_options, sized to the
account. Honest: news puts are high-variance and you're fighting upward drift —
the edge is the catalyst + confirmation, not the chart looking 'too high'.
"""

from __future__ import annotations

from datetime import datetime, timedelta, date

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except Exception:                       # pragma: no cover
    np = pd = yf = None

from momentum_strategy import LIQUID_UNIVERSE

# Backtested stats for honest UI
SETUP_STATS = {
    "NEWS_REVERSAL": {"label": "Negative catalyst + price breaking down",
                      "edge": "event-driven (the real put edge)", "horizon": "1-5d"},
    "RSI2_PULLBACK": {"label": "Extreme 2-day overbought (RSI2>98)",
                      "edge": "weak 1-day mean-reversion (61% down-rate)", "horizon": "1d"},
}


def _rsi(s, p=2):
    d = s.diff(); up = d.clip(lower=0).rolling(p).mean(); dn = (-d.clip(upper=0)).rolling(p).mean()
    v = (100 - 100/(1 + up/dn.replace(0, np.nan))).iloc[-1]
    return float(v) if pd.notna(v) else 50.0


def _price_breaking_down(ticker: str) -> dict | None:
    """Confirm a reversal: price below 20-EMA and/or red today after a run."""
    if yf is None:
        return None
    try:
        raw = yf.download(ticker, period="3mo", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 25:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        c = raw["Close"].dropna()
        price = float(c.iloc[-1]); prev = float(c.iloc[-2])
        ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
        ret1 = price/prev - 1 if prev else 0
        rsi2 = _rsi(c, 2)
        return {"price": round(price, 2), "ema20": round(ema20, 2),
                "below_ema20": price < ema20, "ret1": ret1, "rsi2": rsi2,
                "breaking_down": price < ema20 or ret1 < -0.01}
    except Exception:
        return None


class NewsReversalPuts:
    def __init__(self, conn, account: float = 200.0):
        self.conn = conn
        self.account = account

    # ── 1) news-driven reversals ──────────────────────────────────────────────
    def find_news_reversals(self, hours: int = 24) -> list:
        """Liquid names with a fresh negative catalyst AND price confirming the break."""
        cands = {}   # ticker -> catalyst text

        # a) bearish VIP posts (Trump/Fed) in last `hours`
        try:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            rows = self.conn.execute(
                "SELECT vip_name, title, text, tickers FROM vip_posts "
                "WHERE fetched_at >= ? AND sentiment='bearish' AND tickers <> ''",
                (cutoff,)).fetchall()
            for r in rows:
                def g(k, i): return r.get(k) if hasattr(r, "get") else r[i]
                for tk in str(g("tickers", 3) or "").split(","):
                    tk = tk.strip().upper()
                    if tk and "." not in tk:
                        cands.setdefault(tk, f"{g('vip_name',0)}: "
                                         f"{(g('title',1) or g('text',2) or '')[:90]}")
        except Exception:
            pass

        # b) high-impact negative news (news agent)
        try:
            from news_agent import NewsAgent
            na = NewsAgent(self.conn, ai_analyst=None)
            for tk in list(LIQUID_UNIVERSE):
                if "." in tk:
                    continue
                imp = na.get_news_impact(tk, hours=hours)
                if imp and imp.get("should_skip"):   # should_skip = bad news flag
                    top = imp.get("top_event") or {}
                    cands.setdefault(tk, str(top.get("headline", "negative news"))[:90])
        except Exception:
            pass

        # Confirm the reversal with price action
        out = []
        for tk, catalyst in list(cands.items())[:20]:
            chk = _price_breaking_down(tk)
            if not chk or not chk["breaking_down"]:
                continue
            out.append({"ticker": tk, "setup": "NEWS_REVERSAL",
                        "catalyst": catalyst, "price": chk["price"],
                        "below_ema20": chk["below_ema20"], "ret1": chk["ret1"],
                        "stats": SETUP_STATS["NEWS_REVERSAL"]})
        return out

    # ── 2) RSI2 extreme pullback (weak secondary) ─────────────────────────────
    def find_rsi2_pullbacks(self, universe=None, max_names: int = 6) -> list:
        out = []
        uni = [t for t in (universe or LIQUID_UNIVERSE) if "." not in t]
        for t in uni:
            chk = _price_breaking_down(t)
            if not chk:
                continue
            if chk["rsi2"] > 98:
                out.append({"ticker": t, "setup": "RSI2_PULLBACK",
                            "catalyst": f"RSI2 {chk['rsi2']:.0f} — extreme overbought",
                            "price": chk["price"], "rsi2": chk["rsi2"],
                            "stats": SETUP_STATS["RSI2_PULLBACK"]})
            if len(out) >= max_names:
                break
        return out

    # ── put play for a setup ──────────────────────────────────────────────────
    def put_play(self, ticker: str, catalyst: str) -> str | None:
        try:
            from catalyst_options import catalyst_to_weekly_alert
            return catalyst_to_weekly_alert(ticker, "bearish",
                                            f"PUT setup — {catalyst}", self.account)
        except Exception:
            return None

    def scan(self) -> list:
        """All firing put setups (news reversals first, then RSI2 pullbacks)."""
        return self.find_news_reversals() + self.find_rsi2_pullbacks()


if __name__ == "__main__":
    print("News-reversal put engine — scanning (no DB writes)\n")
    print("NOTE: shorting overextension was backtest-REJECTED (parabolic/stretched")
    print("names keep rising). This engine only acts on news + confirmation.\n")
    class _NoConn:
        def execute(self, *a, **k):
            class _R:
                def fetchall(self): return []
            return _R()
    eng = NewsReversalPuts(_NoConn())
    rsi = eng.find_rsi2_pullbacks(max_names=4)
    print(f"RSI2 extreme-overbought names right now: {len(rsi)}")
    for r in rsi:
        print(f"  {r['ticker']}: RSI2 {r.get('rsi2',0):.0f}  @ ${r['price']}")
