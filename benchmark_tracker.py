"""
benchmark_tracker.py — Forward "momentum vs SPY" scorecard.

WHY
---
The MCPT backtest said cross-sectional momentum has a real edge (p=0.004, +93% over
beta). But a backtest is a backtest. The only thing that actually matters is whether
the LIVE strategy beats just buying SPY going forward. This records the bot's real
mark-to-market equity once per day alongside SPY, so the edge (or lack of it)
accumulates as honest, forward, out-of-sample evidence.

Baseline = the first snapshot taken (i.e. when the momentum pivot went live), so the
comparison answers: "since we switched strategies, are we beating SPY?"

  bot_return = equity_now / equity_baseline - 1
  spy_return = spy_now   / spy_baseline   - 1
  alpha      = bot_return - spy_return        (the number that matters)
"""

from __future__ import annotations

from datetime import datetime, timezone

try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None


class BenchmarkTracker:
    def __init__(self, conn):
        self.conn = conn
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_benchmark (
                    snapshot_date TEXT PRIMARY KEY,
                    snapshot_at   TEXT,
                    bot_equity    REAL,
                    spy_price     REAL,
                    n_positions   INTEGER
                )
            """)
        except Exception as e:
            print(f"  [benchmark] table init failed: {e}")

    # ── live prices (fast_info = same reliable source as the dashboard) ─────────
    @staticmethod
    def _live(ticker: str) -> float | None:
        if yf is None:
            return None
        try:
            return float(yf.Ticker(ticker).fast_info["last_price"])
        except Exception:
            return None

    def _bot_equity(self, paper) -> float:
        """True mark-to-market equity: cash + Σ(shares × live price)."""
        try:
            summary = paper.get_summary()
            cash = float(summary.get("available_cash", 0) or 0)
        except Exception:
            cash = 0.0
        mv = 0.0
        for p in paper.open_positions:
            px = self._live(p["ticker"]) or float(p.get("entry_price") or 0)
            mv += float(p.get("shares") or 0) * px
        return cash + mv

    # ── record one snapshot (idempotent per calendar day) ───────────────────────
    def snapshot(self, paper) -> dict | None:
        spy = self._live("SPY")
        if not spy:
            print("  [benchmark] SPY price unavailable — skipping snapshot.")
            return None
        equity = self._bot_equity(paper)
        today  = datetime.now().strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()
        n_pos   = len(paper.open_positions)

        try:
            existing = self.conn.execute(
                "SELECT snapshot_date FROM paper_benchmark WHERE snapshot_date=?",
                (today,)).fetchone()
            if existing:
                self.conn.execute(
                    "UPDATE paper_benchmark SET snapshot_at=?, bot_equity=?, "
                    "spy_price=?, n_positions=? WHERE snapshot_date=?",
                    (now_iso, equity, spy, n_pos, today))
            else:
                self.conn.execute(
                    "INSERT INTO paper_benchmark "
                    "(snapshot_date, snapshot_at, bot_equity, spy_price, n_positions) "
                    "VALUES (?,?,?,?,?)",
                    (today, now_iso, equity, spy, n_pos))
        except Exception as e:
            print(f"  [benchmark] snapshot write failed: {e}")
            return None

        print(f"  [benchmark] equity ${equity:,.2f} | SPY ${spy:.2f} | {n_pos} pos")
        return {"date": today, "bot_equity": equity, "spy_price": spy}

    # ── read the scorecard (+ normalized series for charting) ───────────────────
    def get_scorecard(self) -> dict | None:
        try:
            rows = self.conn.execute(
                "SELECT snapshot_date, bot_equity, spy_price, n_positions "
                "FROM paper_benchmark ORDER BY snapshot_date ASC"
            ).fetchall()
        except Exception:
            return None
        if not rows:
            return None

        def g(r, key, idx):
            return r.get(key) if hasattr(r, "get") else r[idx]

        series = []
        for r in rows:
            series.append({
                "date": g(r, "snapshot_date", 0),
                "bot_equity": float(g(r, "bot_equity", 1) or 0),
                "spy_price": float(g(r, "spy_price", 2) or 0),
                "n_positions": int(g(r, "n_positions", 3) or 0),
            })

        base = series[0]
        last = series[-1]
        b_eq0, s_p0 = base["bot_equity"], base["spy_price"]
        if b_eq0 <= 0 or s_p0 <= 0:
            return None

        # normalized return curves (% since inception)
        for s in series:
            s["bot_ret_pct"] = (s["bot_equity"] / b_eq0 - 1) * 100
            s["spy_ret_pct"] = (s["spy_price"] / s_p0 - 1) * 100

        bot_ret = (last["bot_equity"] / b_eq0 - 1) * 100
        spy_ret = (last["spy_price"] / s_p0 - 1) * 100
        return {
            "inception": base["date"],
            "as_of": last["date"],
            "days_tracked": len(series),
            "baseline_equity": b_eq0,
            "current_equity": last["bot_equity"],
            "bot_return_pct": round(bot_ret, 2),
            "spy_return_pct": round(spy_ret, 2),
            "alpha_pct": round(bot_ret - spy_ret, 2),
            "winning": bot_ret > spy_ret,
            "series": series,
        }
