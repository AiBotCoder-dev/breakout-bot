"""
goal_tracker.py — The Disciplined Plan: $500 → $1,500 in 12 months.

WHY THIS EXISTS
---------------
Without a visible goal, every day looks like a flat line and you start FOMOing
into trades you shouldn't. This tracker translates the paper portfolio's % return
into your REAL account ($500 → $1,500), shows whether you're on pace, and
surfaces the single most important number each day: "am I winning the year?"

It doesn't open trades — discipline does. This is the scoreboard.

HOW IT WORKS
------------
- You configure: start capital ($500), goal ($1,500), horizon (12 months).
- Each cycle/day it reads the paper portfolio's % return since the day you
  started (the START_DATE row in the goal_state table, set automatically
  on first call), applies that % to your real start capital, and computes:
    current_real_value, days_elapsed, days_remaining, required_monthly_pace,
    actual_monthly_pace, on_track ("ahead" / "on pace" / "behind").
- The dashboard panel shows it as a single scoreboard with the pace gauge
  and a projection (at current pace, you'll end at $X).
"""

from __future__ import annotations

from datetime import datetime, date, timedelta


class GoalTracker:
    """
    Reads the paper portfolio's % return since `goal_start` and projects it
    against the real-account goal. Persists config in `goal_state`.
    """

    def __init__(self, conn,
                 start_capital: float = 500.0,
                 goal_capital:  float = 1500.0,
                 horizon_days:  int   = 365):
        self.conn          = conn
        self.start_capital = float(start_capital)
        self.goal_capital  = float(goal_capital)
        self.horizon_days  = int(horizon_days)
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS goal_state (
                    id              INTEGER PRIMARY KEY,
                    start_date      TEXT,
                    start_capital   REAL,
                    goal_capital    REAL,
                    horizon_days    INTEGER,
                    baseline_equity REAL
                )
            """)
        except Exception as e:
            print(f"  [goal] table init failed: {e}")

    def _load_state(self) -> dict | None:
        try:
            row = self.conn.execute(
                "SELECT * FROM goal_state WHERE id=1").fetchone()
        except Exception:
            return None
        if not row:
            return None
        def g(k):
            return row.get(k) if hasattr(row, "get") else None
        return {
            "start_date":      str(g("start_date") or ""),
            "start_capital":   float(g("start_capital") or 0),
            "goal_capital":    float(g("goal_capital") or 0),
            "horizon_days":    int(g("horizon_days") or 0),
            "baseline_equity": float(g("baseline_equity") or 0),
        }

    def _save_state(self, start_date: str, baseline_equity: float):
        try:
            self.conn.execute(
                "INSERT INTO goal_state "
                "(id, start_date, start_capital, goal_capital, horizon_days, baseline_equity) "
                "VALUES (1,?,?,?,?,?) "
                "ON CONFLICT (id) DO UPDATE SET "
                "start_date=excluded.start_date, start_capital=excluded.start_capital, "
                "goal_capital=excluded.goal_capital, horizon_days=excluded.horizon_days, "
                "baseline_equity=excluded.baseline_equity",
                (str(start_date), self.start_capital, self.goal_capital,
                 self.horizon_days, float(baseline_equity)))
        except Exception as e:
            print(f"  [goal] save failed: {e}")

    # ── public API ─────────────────────────────────────────────────────────────
    def reset(self, paper_equity: float):
        """Re-baseline the journey to today's paper equity. Use sparingly."""
        self._save_state(date.today().isoformat(), paper_equity)

    def update_config(self, start_capital: float | None = None,
                      goal_capital: float | None = None,
                      horizon_days: int | None = None):
        """Change goal/horizon without resetting the baseline."""
        st = self._load_state() or {}
        if start_capital is not None:
            self.start_capital = float(start_capital)
        if goal_capital is not None:
            self.goal_capital = float(goal_capital)
        if horizon_days is not None:
            self.horizon_days = int(horizon_days)
        self._save_state(
            st.get("start_date") or date.today().isoformat(),
            st.get("baseline_equity") or 0,
        )

    def scorecard(self, paper) -> dict:
        """
        Read paper equity, translate to real-account dollars, compute pace.
        On first call ever, baselines today's paper equity as the starting point.
        """
        # current paper equity (true mark-to-market via the existing helper)
        try:
            from benchmark_tracker import BenchmarkTracker
            paper_equity = BenchmarkTracker(self.conn)._bot_equity(paper)
        except Exception:
            summary = paper.get_summary()
            paper_equity = float(summary.get("total_capital", 0) or 0)

        st = self._load_state()
        if not st or not st.get("baseline_equity") or st["baseline_equity"] <= 0:
            self.reset(paper_equity)
            st = self._load_state() or {}

        baseline = float(st.get("baseline_equity") or paper_equity)
        start_dt = date.fromisoformat(st.get("start_date") or date.today().isoformat())

        # paper return → applied to real start capital
        paper_ret_pct = (paper_equity / baseline - 1) * 100 if baseline else 0.0
        real_value    = self.start_capital * (1 + paper_ret_pct / 100)
        gain_dollars  = real_value - self.start_capital

        # Time math
        today = date.today()
        days_elapsed   = max(1, (today - start_dt).days)
        days_remaining = max(0, self.horizon_days - days_elapsed)

        # Required pace: monthly compound rate to take start → goal in horizon
        # (1+r)^(horizon/30) = goal/start
        years_needed = self.horizon_days / 30.0
        try:
            req_monthly = (self.goal_capital / self.start_capital) ** (1 / years_needed) - 1
        except Exception:
            req_monthly = 0.0

        # Actual pace: equivalent monthly compound rate so far
        try:
            elapsed_months = days_elapsed / 30.0
            actual_monthly = (real_value / self.start_capital) ** (1 / max(elapsed_months, 0.1)) - 1
        except Exception:
            actual_monthly = 0.0

        # Projection: at current monthly pace, where do we end at horizon?
        try:
            projected = self.start_capital * ((1 + actual_monthly) ** years_needed)
        except Exception:
            projected = real_value

        if actual_monthly >= req_monthly * 1.10:
            status = "ahead"
        elif actual_monthly >= req_monthly * 0.85:
            status = "on_pace"
        else:
            status = "behind"

        return {
            "start_date":           start_dt.isoformat(),
            "start_capital":        round(self.start_capital, 2),
            "goal_capital":         round(self.goal_capital, 2),
            "horizon_days":         self.horizon_days,
            "days_elapsed":         days_elapsed,
            "days_remaining":       days_remaining,
            "paper_baseline_equity": round(baseline, 2),
            "paper_equity_now":     round(paper_equity, 2),
            "paper_return_pct":     round(paper_ret_pct, 2),
            "real_value_now":       round(real_value, 2),
            "real_gain_dollars":    round(gain_dollars, 2),
            "required_monthly_pct": round(req_monthly * 100, 2),
            "actual_monthly_pct":   round(actual_monthly * 100, 2),
            "projected_end_value":  round(projected, 2),
            "status":               status,        # "ahead" / "on_pace" / "behind"
            "pct_of_goal":          round((real_value - self.start_capital)
                                          / (self.goal_capital - self.start_capital) * 100, 1)
                                    if self.goal_capital > self.start_capital else 0.0,
        }
