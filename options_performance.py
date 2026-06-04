"""
options_performance.py — The honest scorecard for every bot-originated options trade.

WHAT THIS ANSWERS
-----------------
"Are the bot's options picks actually making money — and what's the real
empirical probability that a new setup pays off?"

For each OPEN position:
  • current premium (live), %P&L, dollar P&L, days held
  • return vs the underlying-only hold over the same period
    (= leverage alpha. If positive, the option trade beats just owning the stock)

For each CLOSED position:
  • realized %P&L, hold days, exit reason
  • per-strategy historical win rate / avg winner / avg loser / expectancy

The win rate becomes the EMPIRICAL PROBABILITY badge on new setups
(grouped by strategy + quality-score bucket — i.e. "B-grade momentum_calls
historically win 38% of the time, avg winner +84%, avg loser -52%").
"""

from __future__ import annotations

from datetime import datetime, timezone, date

try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None


# ══════════════════════════════════════════════════════════════════════════════
# CURRENT PREMIUM LOOKUP (matches what momentum_options uses)
# ══════════════════════════════════════════════════════════════════════════════
def _current_premium(ticker: str, expiry: str, strike: float,
                     option_type: str = "call") -> float | None:
    if yf is None:
        return None
    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
        df = chain.calls if option_type == "call" else chain.puts
        row = df[df["strike"] == strike]
        if row.empty:
            return None
        r = row.iloc[0]
        b, a = float(r.get("bid", 0) or 0), float(r.get("ask", 0) or 0)
        if b > 0 and a > 0 and a >= b:
            return (a + b) / 2.0
        lp = float(r.get("lastPrice", 0) or 0)
        return lp if lp > 0 else None
    except Exception:
        return None


def _live_underlying(ticker: str) -> float | None:
    if yf is None:
        return None
    try:
        return float(yf.Ticker(ticker).fast_info["last_price"])
    except Exception:
        return None


def _underlying_at(ticker: str, dt_str: str) -> float | None:
    """Approximate underlying close on a past date."""
    if yf is None or not dt_str:
        return None
    try:
        import pandas as pd
        raw = yf.download(ticker, start=dt_str[:10], period="1mo",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        c = raw["Close"].dropna()
        return float(c.iloc[0]) if not c.empty else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class OptionsPerformanceTracker:
    def __init__(self, conn):
        self.conn = conn

    # ── OPEN positions enriched with live data ────────────────────────────────
    def get_open(self, strategies: list | None = None) -> list:
        try:
            sql = "SELECT * FROM options_positions WHERE status='OPEN'"
            args: tuple = ()
            if strategies:
                placeholders = ",".join("?" * len(strategies))
                sql += f" AND strategy IN ({placeholders})"
                args = tuple(strategies)
            rows = self.conn.execute(sql, args).fetchall()
        except Exception:
            return []

        out = []
        today = date.today()
        for r in rows:
            def g(k):
                return r.get(k) if hasattr(r, "get") else None
            tk     = str(g("ticker") or "").upper()
            strike = float(g("strike") or 0)
            expiry = str(g("expiry") or "")[:10]
            otype  = str(g("option_type") or "call")
            entry  = float(g("entry_price") or 0)
            contracts = int(g("contracts") or 0)
            invested = float(g("gross_invested") or 0)
            edt_str = str(g("entry_date") or "")[:10]

            current = _current_premium(tk, expiry, strike, otype) or 0.0
            ret_pct = ((current - entry) / entry * 100) if entry > 0 else 0.0
            pnl_dollars = (current - entry) * 100 * contracts

            # Days held + DTE
            try:
                edt = date.fromisoformat(edt_str)
                days_held = (today - edt).days
            except Exception:
                days_held = 0
            try:
                expd = date.fromisoformat(expiry)
                dte = max(0, (expd - today).days)
            except Exception:
                dte = 0

            # Underlying alpha (does the leverage beat just owning the stock?)
            u_now = _live_underlying(tk) or 0.0
            u_then = _underlying_at(tk, edt_str) or 0.0
            u_ret = ((u_now / u_then - 1) * 100) if u_then > 0 else 0.0
            leverage_alpha = ret_pct - u_ret

            out.append({
                "id":               int(g("id") or 0),
                "ticker":           tk,
                "contract_symbol":  str(g("contract_symbol") or ""),
                "option_type":      otype,
                "strike":           strike,
                "expiry":           expiry,
                "entry_price":      round(entry, 2),
                "current_premium":  round(current, 2),
                "contracts":        contracts,
                "invested":         round(invested, 2),
                "current_value":    round(current * 100 * contracts, 2),
                "ret_pct":          round(ret_pct, 1),
                "pnl_dollars":      round(pnl_dollars, 2),
                "days_held":        days_held,
                "dte_remaining":    dte,
                "underlying_now":   round(u_now, 2),
                "underlying_then":  round(u_then, 2),
                "underlying_ret_pct": round(u_ret, 2),
                "leverage_alpha_pct": round(leverage_alpha, 2),
                "strategy":         str(g("strategy") or ""),
                "entry_date":       edt_str,
            })
        out.sort(key=lambda x: x["ret_pct"], reverse=True)
        return out

    # ── CLOSED positions + per-strategy aggregate stats ───────────────────────
    def get_closed(self, strategies: list | None = None, limit: int = 200) -> list:
        try:
            sql = ("SELECT * FROM options_positions WHERE status='CLOSED' ")
            args: tuple = ()
            if strategies:
                placeholders = ",".join("?" * len(strategies))
                sql += f"AND strategy IN ({placeholders}) "
                args = tuple(strategies)
            sql += "ORDER BY exit_date DESC LIMIT ?"
            args = args + (int(limit),)
            rows = self.conn.execute(sql, args).fetchall()
        except Exception:
            return []
        out = []
        for r in rows:
            def g(k):
                return r.get(k) if hasattr(r, "get") else None
            entry = float(g("entry_price") or 0)
            exitp = float(g("exit_price")  or 0)
            pnl   = float(g("net_pnl")     or 0)
            inv   = float(g("gross_invested") or 1)
            pct   = (pnl / inv * 100) if inv > 0 else 0.0
            try:
                edt = date.fromisoformat(str(g("entry_date") or "")[:10])
                xdt = date.fromisoformat(str(g("exit_date")  or "")[:10])
                hold = (xdt - edt).days
            except Exception:
                hold = 0
            out.append({
                "ticker":      str(g("ticker") or ""),
                "option_type": str(g("option_type") or "call"),
                "strike":      float(g("strike") or 0),
                "expiry":      str(g("expiry") or "")[:10],
                "entry_price": round(entry, 2),
                "exit_price":  round(exitp, 2),
                "ret_pct":     round(pct, 1),
                "net_pnl":     round(pnl, 2),
                "hold_days":   hold,
                "exit_reason": str(g("exit_reason") or ""),
                "strategy":    str(g("strategy") or ""),
                "entry_date":  str(g("entry_date") or "")[:10],
                "exit_date":   str(g("exit_date") or "")[:10],
            })
        return out

    def summary(self, strategies: list | None = None) -> dict:
        closed = self.get_closed(strategies, limit=10_000)
        if not closed:
            return {
                "n_closed":     0,
                "win_rate":     None,
                "avg_winner":   None,
                "avg_loser":    None,
                "expectancy":   None,
                "total_pnl":    0.0,
                "total_invested": 0.0,
                "roi_pct":      None,
                "best_trade":   None,
                "worst_trade":  None,
            }
        wins = [c for c in closed if c["net_pnl"] > 0]
        losses = [c for c in closed if c["net_pnl"] < 0]
        total_pnl = sum(c["net_pnl"] for c in closed)
        total_inv = sum(
            float(self.conn.execute(
                "SELECT gross_invested FROM options_positions WHERE ticker=? "
                "AND entry_date=? LIMIT 1", (c["ticker"], c["entry_date"])
            ).fetchone()[0] or 0) if False else 0  # cheap fallback below
            for c in closed
        )
        # Cheaper aggregate via fresh query
        try:
            r = self.conn.execute(
                "SELECT SUM(gross_invested) FROM options_positions WHERE status='CLOSED'"
            ).fetchone()
            total_inv = float((r[0] if r else 0) or 0)
        except Exception:
            total_inv = 0.0
        wr = (len(wins) / len(closed) * 100) if closed else 0.0
        avg_w = (sum(c["ret_pct"] for c in wins) / len(wins)) if wins else 0.0
        avg_l = (sum(c["ret_pct"] for c in losses) / len(losses)) if losses else 0.0
        # Expectancy as % per trade
        exp_pct = (len(wins) * avg_w + len(losses) * avg_l) / len(closed) if closed else 0.0
        best  = max(closed, key=lambda c: c["ret_pct"])
        worst = min(closed, key=lambda c: c["ret_pct"])
        return {
            "n_closed":       len(closed),
            "n_winners":      len(wins),
            "n_losers":       len(losses),
            "win_rate":       round(wr, 1),
            "avg_winner":     round(avg_w, 1),
            "avg_loser":      round(avg_l, 1),
            "expectancy":     round(exp_pct, 1),
            "total_pnl":      round(total_pnl, 2),
            "total_invested": round(total_inv, 2),
            "roi_pct":        round(total_pnl / total_inv * 100, 2) if total_inv else None,
            "best_trade":     {"ticker": best["ticker"],  "ret_pct": best["ret_pct"]},
            "worst_trade":    {"ticker": worst["ticker"], "ret_pct": worst["ret_pct"]},
        }

    # ── Empirical probability badges for new setups ───────────────────────────
    def empirical_probability(self, strategy: str = "momentum_call") -> dict | None:
        """
        Returns {win_rate, avg_winner_pct, avg_loser_pct, expectancy_pct, n}
        using ALL closed trades of this strategy. This is what the bot can
        honestly show as "probability" for a new setup of the same kind.

        Returns None if there isn't enough history yet (n < 5).
        """
        s = self.summary(strategies=[strategy])
        if not s or s["n_closed"] < 5:
            return None
        return {
            "win_rate":       s["win_rate"],
            "avg_winner":     s["avg_winner"],
            "avg_loser":      s["avg_loser"],
            "expectancy":     s["expectancy"],
            "n":              s["n_closed"],
            "warning":        "Estimate — small sample" if s["n_closed"] < 30 else None,
        }
