"""
options_event_exits.py — External-factor auto-close for open options positions.

WHAT IT DOES
------------
The existing exit engine (momentum_options.auto_exit) handles the MECHANICAL
exits: +100% take profit, -50% stop loss, DTE ≤ 2. This module handles the
EXTERNAL-EVENT exits — situations where something happened in the world that
changes the trade's risk/reward, even if the price hasn't yet moved enough to
trip the mechanical rule.

The triggers it watches (all use data already in the bot):

  1. VIP NEGATIVE      — a tracked VIP (Trump / Fed) posted negatively about the
                         underlying in the last 6 hours.
  2. HIGH-IMPACT NEWS  — a bearish high-impact news headline classified by the
                         news agent in the last 24h with the ticker in its
                         affected list.
  3. EARNINGS CREEP    — the next earnings date is now within 5 days. The setup
                         was supposed to skip these (EARNINGS_AVOID_DAYS = 7),
                         but date can change or position can be held longer than
                         expected. Force-close to escape IV crush.
  4. TREND BREAK       — underlying closed below its 50-SMA. The bullish thesis
                         that triggered the trade has broken; the option becomes
                         a falling knife.
  5. VIX SPIKE         — VIX jumped into the FEAR regime since the trade was
                         opened. Speculative long-premium plays get crushed
                         when correlations go to 1.

Runs from monitor.py once per cycle, BEFORE the auto-entry block. Returns the
list of closures so monitor can tally + send Telegram.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta

try:
    import yfinance as yf
    import pandas as pd
except Exception:                       # pragma: no cover
    yf = pd = None


# ── Tunable thresholds ────────────────────────────────────────────────────────
VIP_LOOKBACK_HOURS    = 6
NEWS_LOOKBACK_HOURS   = 24
NEWS_IMPACT_THRESHOLD = 7         # 0-10 scale (matches news_agent classification)
EARNINGS_FORCE_DAYS   = 5
VIX_FEAR_THRESHOLD    = 25        # VIX above this = FEAR regime


# ══════════════════════════════════════════════════════════════════════════════
# TRIGGER CHECKS
# ══════════════════════════════════════════════════════════════════════════════
def _vip_negative_recent(conn, ticker: str) -> dict | None:
    """Any tracked-VIP post in last VIP_LOOKBACK_HOURS that mentions ticker
       and classifies as bearish?"""
    try:
        cutoff = (datetime.utcnow() - timedelta(hours=VIP_LOOKBACK_HOURS)).isoformat()
        rows = conn.execute(
            "SELECT vip_handle, vip_name, posted_at, fetched_at, title, text, "
            "tickers, sentiment, url FROM vip_posts "
            "WHERE fetched_at >= ? AND sentiment='bearish' "
            "AND (tickers LIKE ? OR tickers LIKE ? OR tickers LIKE ?)",
            (cutoff, f"{ticker}", f"{ticker},%", f"%,{ticker}")
        ).fetchall()
    except Exception:
        return None
    for r in rows:
        def g(k):
            return r.get(k) if hasattr(r, "get") else None
        tcsv = str(g("tickers") or "")
        if ticker.upper() in [t.strip().upper() for t in tcsv.split(",")]:
            return {
                "trigger":  "VIP_NEGATIVE",
                "vip":      g("vip_name") or g("vip_handle"),
                "headline": (g("title") or g("text") or "")[:180],
                "url":      g("url") or "",
            }
    return None


def _negative_news_recent(conn, ticker: str) -> dict | None:
    """High-impact bearish news in the last NEWS_LOOKBACK_HOURS mentioning ticker?"""
    try:
        from news_agent import NewsAgent
        impact = NewsAgent(conn, ai_analyst=None).get_news_impact(
            ticker, hours=NEWS_LOOKBACK_HOURS)
    except Exception:
        return None
    if not impact or not impact.get("should_skip"):
        return None
    top = impact.get("top_event") or {}
    return {
        "trigger":   "HIGH_IMPACT_NEGATIVE_NEWS",
        "headline":  str(top.get("headline", ""))[:180],
        "impact":    impact.get("max_impact"),
        "sentiment": "bearish",
    }


def _earnings_creep(conn, ticker: str) -> dict | None:
    """Earnings now within EARNINGS_FORCE_DAYS — IV crush is imminent."""
    try:
        from earnings_engine import EarningsCalendar
        cal = EarningsCalendar(conn).get(ticker)
    except Exception:
        return None
    if not cal:
        return None
    next_e = cal.get("next_earnings")
    if not next_e:
        return None
    try:
        d = date.fromisoformat(next_e[:10])
        days = (d - date.today()).days
        if 0 <= days <= EARNINGS_FORCE_DAYS:
            return {"trigger": "EARNINGS_CREEP",
                    "earnings_date": next_e,
                    "days": days}
    except Exception:
        return None
    return None


def _trend_break(ticker: str) -> dict | None:
    """Underlying closed below 50-SMA — bullish thesis broken."""
    try:
        from momentum_strategy import MomentumStrategy
        chk = MomentumStrategy().should_exit(ticker)
    except Exception:
        return None
    if not chk or not chk.get("exit"):
        return None
    return {"trigger": "TREND_BREAK_50SMA",
            "price":  chk.get("price"),
            "sma50":  chk.get("sma50")}


def _vix_spike() -> dict | None:
    """VIX in FEAR regime — close speculative options."""
    if yf is None:
        return None
    try:
        v = float(yf.Ticker("^VIX").fast_info["last_price"])
        if v >= VIX_FEAR_THRESHOLD:
            return {"trigger": "VIX_FEAR", "vix": round(v, 1)}
    except Exception:
        return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class EventDrivenExitEngine:
    def __init__(self, conn, strategies: list | None = None):
        self.conn = conn
        # Only act on bot-originated positions, not manual entries
        self.strategies = strategies or ["momentum_call"]

    def check_all_open(self, paper_options, telegram_sender=None) -> list:
        """
        For each open position in tracked strategies, evaluate triggers and
        close the position if any fires. Returns list of close-result dicts.
        """
        try:
            placeholders = ",".join("?" * len(self.strategies))
            rows = self.conn.execute(
                f"SELECT * FROM options_positions "
                f"WHERE status='OPEN' AND strategy IN ({placeholders})",
                tuple(self.strategies),
            ).fetchall()
        except Exception:
            return []
        if not rows:
            return []

        # Market-wide check (cheap, done once)
        vix_trigger = _vix_spike()

        closed = []
        for r in rows:
            def g(k):
                return r.get(k) if hasattr(r, "get") else None
            pid    = int(g("id") or 0)
            tk     = str(g("ticker") or "").upper()
            strike = float(g("strike") or 0)
            expiry = str(g("expiry") or "")[:10]
            otype  = str(g("option_type") or "call")
            if not pid or not tk:
                continue

            # Evaluate triggers in cheap-to-expensive order; first hit wins.
            trig = (
                vix_trigger
                or _vip_negative_recent(self.conn, tk)
                or _earnings_creep(self.conn, tk)
                or _negative_news_recent(self.conn, tk)
                or _trend_break(tk)
            )
            if not trig:
                continue

            # Get current premium for the close (best-effort)
            try:
                from momentum_options import MomentumOptionsStrategy
                cur = MomentumOptionsStrategy._current_premium(tk, expiry, strike) or 0.0
            except Exception:
                cur = 0.0

            res = paper_options.close(pid, cur, f"EVENT:{trig['trigger']}")
            if res.get("ok"):
                closed.append({
                    "ticker":     tk,
                    "contract":   str(g("contract_symbol") or ""),
                    "strike":     strike,
                    "expiry":     expiry,
                    "option_type": otype,
                    "trigger":    trig["trigger"],
                    "detail":     trig,
                    "exit_price": cur,
                    "net_pnl":    res.get("net_pnl", 0),
                })
                # Telegram notify
                if telegram_sender:
                    msg = self._format_alert(tk, strike, expiry, otype,
                                             cur, res.get("net_pnl", 0), trig)
                    try: telegram_sender(msg)
                    except Exception: pass
                # Console log
                print(f"    🛡  EVENT EXIT {tk:6s} ${strike:.0f}{otype[0].upper()} "
                      f"@ ${cur:.2f}  reason={trig['trigger']}  "
                      f"net=${res.get('net_pnl', 0):+.0f}")
        return closed

    @staticmethod
    def _format_alert(tk, strike, expiry, otype, exit_px, pnl, trig):
        emoji = {"VIX_FEAR": "😱", "VIP_NEGATIVE": "🚨",
                 "EARNINGS_CREEP": "📅", "HIGH_IMPACT_NEGATIVE_NEWS": "📰",
                 "TREND_BREAK_50SMA": "📉"}.get(trig["trigger"], "🛡")
        detail_lines = []
        if "headline" in trig and trig["headline"]:
            detail_lines.append(f"\"{trig['headline'][:200]}\"")
        if "vip" in trig:
            detail_lines.append(f"From: <b>{trig['vip']}</b>")
        if "days" in trig:
            detail_lines.append(f"Earnings in <b>{trig['days']}d</b> "
                                f"({trig.get('earnings_date','')})")
        if "vix" in trig:
            detail_lines.append(f"VIX = <b>{trig['vix']}</b> (FEAR regime)")
        if "sma50" in trig and "price" in trig:
            detail_lines.append(
                f"Underlying ${trig['price']:.2f} broke below 50-SMA ${trig['sma50']:.2f}")
        detail_block = "\n" + "\n".join(detail_lines) if detail_lines else ""
        return (
            f"{emoji} <b>EVENT EXIT: {tk}</b>\n"
            f"<b>{tk}</b> ${strike:.0f}{otype[0].upper()} exp {expiry}\n"
            f"Reason: <b>{trig['trigger']}</b>{detail_block}\n"
            f"Exit: ${exit_px:.2f}  ·  Net P&L: <b>${pnl:+.0f}</b>"
        )
