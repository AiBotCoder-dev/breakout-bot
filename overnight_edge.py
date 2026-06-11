"""
overnight_edge.py — The validated "news happens after hours" strategy.

THE EDGE (creative_strategies_backtest.py, 10y):
  Nearly ALL of equity returns accrue close->open, not open->close:
    SPY overnight +164% vs intraday +59%   ·  QQQ +255% vs +104%
    NVDA(5y) +547% vs +128%                ·  TSLA(5y) +158% vs -22%
  News (earnings, Fed, geopolitics) lands outside regular hours; price gaps to
  the new reality at the open. This strategy reads ZERO charts: buy the close,
  sell the open, hold only through the news window.

MECHANICS (rides the existing 10-min monitor cycles):
  BUY  — in the last cycles before the close (15:40-15:59 ET), market-buy a
         fixed notional of OVERNIGHT_TICKER (default QQQ) if no position open.
  SELL — at the FIRST cycle after the next open (any time the market is open
         and the position was opened on a previous day), market-sell it all.

TRACKED 100% SEPARATELY:
  Its own table (overnight_edge_log) + its own scorecard + 🌙-tagged Telegram.
  Never touches the options journal, so the user sees exactly what this method
  contributes on its own.

Env knobs (set via workflow env / secrets):
  OVERNIGHT_EDGE          on|off   (default on)
  OVERNIGHT_TICKER        default QQQ
  OVERNIGHT_NOTIONAL_FRAC fraction of effective equity per night (default 0.20)
"""

from __future__ import annotations

import os
from datetime import datetime

BUY_WINDOW_START = (15, 40)    # ET
BUY_WINDOW_END   = (15, 59)


def _now_et():
    import trading_scanner as ts
    return ts.MarketClock.now_et()


def _market_open_now(now_et) -> bool:
    """Regular session only (9:30-16:00 ET, Mon-Fri)."""
    if now_et.weekday() >= 5:
        return False
    hm = (now_et.hour, now_et.minute)
    return (9, 30) <= hm < (16, 0)


class OvernightEdge:
    def __init__(self, conn, broker):
        self.conn = conn
        self.broker = broker
        self.ticker = os.environ.get("OVERNIGHT_TICKER", "QQQ").strip().upper() or "QQQ"
        try:
            self.frac = float(os.environ.get("OVERNIGHT_NOTIONAL_FRAC", "0.20") or 0.20)
        except Exception:
            self.frac = 0.20
        self.enabled = (os.environ.get("OVERNIGHT_EDGE", "on").strip().lower()
                        not in ("off", "0", "false", "no"))
        self._ensure_tables()

    def _ensure_tables(self):
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS overnight_edge_log (
                    id          BIGSERIAL PRIMARY KEY,
                    ticker      TEXT,
                    status      TEXT,            -- OPEN / CLOSED / ERROR
                    entry_date  TEXT,            -- ET date of the buy
                    entry_time  TEXT,
                    entry_price REAL,
                    notional    REAL,
                    qty         REAL,
                    exit_date   TEXT,
                    exit_time   TEXT,
                    exit_price  REAL,
                    pnl         REAL,
                    pnl_pct     REAL
                )
            """)
        except Exception as e:
            print(f"  [overnight] table init failed: {e}")

    # ── state ──────────────────────────────────────────────────────────────────
    def _open_row(self):
        try:
            r = self.conn.execute(
                "SELECT * FROM overnight_edge_log WHERE status='OPEN' "
                "ORDER BY id DESC LIMIT 1").fetchone()
            return r
        except Exception:
            return None

    # ── the cycle hook ─────────────────────────────────────────────────────────
    def step(self, effective_equity: float, telegram=None) -> dict:
        """
        Called once per monitor cycle. Decides: SELL at the open, BUY at the
        close, or nothing. Returns {"action": ...} for logging.
        """
        if not self.enabled:
            return {"action": "disabled"}
        now = _now_et()
        today = now.strftime("%Y-%m-%d")
        if not _market_open_now(now):
            return {"action": "market_closed"}

        row = self._open_row()
        def g(k):
            return row.get(k) if (row and hasattr(row, "get")) else None

        # ── SELL: position from a previous day + market open => exit now ──────
        if row and str(g("entry_date") or "") < today:
            tk = str(g("ticker") or self.ticker)
            try:
                # actual fill data from the live position
                pos = next((p for p in self.broker.get_positions()
                            if p.get("ticker") == tk), None)
                cur = (pos or {}).get("current") or self.broker.get_price(tk) or 0
                avg = (pos or {}).get("avg_entry") or float(g("entry_price") or 0)
                qty = (pos or {}).get("qty") or float(g("qty") or 0)
                res = self.broker.close_position(tk)
                if not res.get("ok"):
                    print(f"  🌙 overnight SELL failed: {res.get('error')}")
                    return {"action": "sell_failed", "error": res.get("error")}
                pnl = (cur - avg) * qty if (cur and avg and qty) else 0.0
                pnl_pct = (cur / avg - 1) * 100 if (cur and avg) else 0.0
                self.conn.execute(
                    "UPDATE overnight_edge_log SET status='CLOSED', exit_date=?, "
                    "exit_time=?, exit_price=?, pnl=?, pnl_pct=? WHERE id=?",
                    (today, now.strftime("%H:%M"), round(float(cur), 4),
                     round(float(pnl), 2), round(float(pnl_pct), 3),
                     int(g("id") or 0)))
                print(f"  🌙 OVERNIGHT SELL {tk} @ ~${cur:.2f}  "
                      f"P&L ${pnl:+.2f} ({pnl_pct:+.2f}%)")
                if telegram:
                    sc = self.scorecard()
                    telegram(
                        f"🌙 <b>OVERNIGHT EDGE — SOLD AT OPEN</b>\n"
                        f"{tk}: in ${avg:.2f} → out ~${cur:.2f}\n"
                        f"Night P&L: <b>${pnl:+.2f} ({pnl_pct:+.2f}%)</b>\n"
                        f"Method record: {sc['wins']}/{sc['n']} nights green · "
                        f"total ${sc['total_pnl']:+.2f}\n"
                        f"<i>Tracked separately — this is the overnight method "
                        f"only, not the options book.</i>")
                return {"action": "sold", "pnl": pnl, "pnl_pct": pnl_pct}
            except Exception as e:
                print(f"  🌙 overnight sell error: {e}")
                return {"action": "sell_error", "error": str(e)}

        # ── BUY: last cycles before the close, no open position ───────────────
        hm = (now.hour, now.minute)
        if row is None and BUY_WINDOW_START <= hm <= BUY_WINDOW_END:
            notional = max(1.0, round(effective_equity * self.frac, 2))
            try:
                px = self.broker.get_price(self.ticker) or 0
                res = self.broker.submit_notional_buy(self.ticker, notional)
                if not res.get("ok"):
                    print(f"  🌙 overnight BUY failed: {res.get('error')}")
                    return {"action": "buy_failed", "error": res.get("error")}
                qty = (notional / px) if px > 0 else 0
                self.conn.execute(
                    "INSERT INTO overnight_edge_log "
                    "(ticker, status, entry_date, entry_time, entry_price, "
                    " notional, qty) VALUES (?,?,?,?,?,?,?)",
                    (self.ticker, "OPEN", today, now.strftime("%H:%M"),
                     round(float(px), 4), notional, round(qty, 6)))
                print(f"  🌙 OVERNIGHT BUY {self.ticker} ${notional:.0f} notional "
                      f"@ ~${px:.2f} — selling at tomorrow's open")
                if telegram:
                    telegram(
                        f"🌙 <b>OVERNIGHT EDGE — BOUGHT THE CLOSE</b>\n"
                        f"{self.ticker}: ${notional:.0f} notional @ ~${px:.2f}\n"
                        f"Plan: sell at tomorrow's open (holding through the "
                        f"news window — that's where the return lives).\n"
                        f"<i>Backtest: QQQ overnight +255% vs +104% intraday "
                        f"(10y). Tracked separately.</i>")
                return {"action": "bought", "notional": notional}
            except Exception as e:
                print(f"  🌙 overnight buy error: {e}")
                return {"action": "buy_error", "error": str(e)}

        return {"action": "hold" if row else "wait"}

    # ── separate scorecard (the user's "exactly how this affects us" view) ────
    def scorecard(self) -> dict:
        try:
            rows = self.conn.execute(
                "SELECT pnl, pnl_pct FROM overnight_edge_log "
                "WHERE status='CLOSED'").fetchall()
        except Exception:
            rows = []
        pnls = []
        for r in rows:
            v = r.get("pnl") if hasattr(r, "get") else r[0]
            if v is not None:
                pnls.append(float(v))
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        return {
            "n": n, "wins": wins,
            "win_rate": round(100 * wins / n, 1) if n else 0.0,
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / n, 2) if n else 0.0,
        }

    def history(self, limit: int = 30) -> list:
        try:
            rows = self.conn.execute(
                "SELECT * FROM overnight_edge_log ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        except Exception:
            return []
        out = []
        for r in rows:
            def g(k):
                return r.get(k) if hasattr(r, "get") else None
            out.append({k: g(k) for k in
                        ("ticker", "status", "entry_date", "entry_price",
                         "notional", "exit_date", "exit_price", "pnl", "pnl_pct")})
        return out
