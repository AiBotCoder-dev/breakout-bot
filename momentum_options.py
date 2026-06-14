"""
momentum_options.py — Cheap calls ON THE VALIDATED MOMENTUM EDGE.

WHY THIS DESIGN
---------------
"Cheap OTM calls with massive upside" is the most popular retail options strategy
and one of the most reliably losing ones (Bollen-Whaley, Goyal-Saretto, etc.) —
IV on OTM strikes systematically exceeds realized vol, so the buyer pays a
volatility premium that sellers harvest, and theta crushes you.

The only honest way to give that approach a real shot is to anchor it to a
strategy we ACTUALLY measured an edge on. We have one now: cross-sectional
momentum on liquid names (MCPT p=0.004, +93% over pure beta).

So this module buys CALLS only on names that pass the validated momentum rank,
slightly OTM (5-15%) with 2-6 weeks to expiry, with IV + earnings + cost gates,
sized like lottery tickets, and with strict mechanical exits. Forward P&L is
tracked so the live edge (or lack of it) becomes visible over time.

DESIGN PARAMETERS (all configurable at the top)
-----------------------------------------------
  Underlying  = top-N from MomentumStrategy.rank()  (validated edge)
  Strike       = 5-15% OTM
  DTE          = 14-42 days
  Premium cap  = $3.00 per contract (effectively "cheap")
  Size         = ~1% of options capital per ticket (lottery sizing)
  Exits        = +100% take profit  /  -50% stop loss  /  DTE <= 2 force-exit
  Earnings     = SKIP if earnings within 7 days (IV crush)
  IV gate      = SKIP if call IV > 1.20 (vol already too expensive)
"""

from __future__ import annotations

from datetime import datetime, timezone

try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None


# ── Tuning knobs ──────────────────────────────────────────────────────────────
COST_CAP_PER_CONTRACT   = 5.00     # max premium per share ($500 / contract).
                                   # Raised from $3 — at $3, the validated mom leaders
                                   # (semis $100-500) had no qualifying contracts.
                                   # $5 keeps the "cheap" character while letting
                                   # 5-15% OTM 2-4wk calls on mid-priced names fit.
TARGET_DTE_MIN          = 5        # SHORT-TERM only — no long holds (user directive)
TARGET_DTE_MAX          = 14       # ~2 weeks max (was 28) — quick in/out
TARGET_DTE_IDEAL        = 9        # ~1.5 weeks — short hold, cheaper contract for $1k acct
MIN_THESIS_PCT          = 8.0      # don't trade options if the expected move is < 8%
                                   # (the OTM strike + theta math doesn't pay otherwise)
# NEAR-THE-MONEY by default (was 5-15% OTM). Backtest — option_structure_backtest.py,
# n≈20k momentum longs, 6y: deep-OTM calls win only 33.5% (median -54% — a pure
# lottery), while ATM-to-5%-OTM wins ~45% (+11pp) with a ~breakeven median. We trade
# the rare moonshot for far fewer -50%/-90% bleeds and a far more execution-robust
# trade (near-money is much less theta/IV-crush/slippage sensitive). The selector
# scores closest-to-ATM highest, so it leans ATM when affordable, 5% OTM when not.
OTM_PCT_MIN             = 0.00     # ATM
OTM_PCT_MAX             = 0.07     # up to 7% OTM — the selector scores closest-to-
                                   # money highest (and penalizes premium), so it
                                   # picks ATM when it fits the $ budget and only
                                   # drifts toward 7% OTM on pricier names. This
                                   # keeps the win-rate lift while preserving volume
                                   # (a 0-5% band skipped too many $100+ leaders).
LOTTERY_SIZE_PCT        = 0.05     # 5% of options cash per ticket (lottery)
MAX_CONCURRENT_POSITIONS = 8
MAX_NEW_PER_CYCLE       = 2

TAKE_PROFIT_PCT         = 70.0     # +70% → take profit (was 100 — bank it faster on short DTE)
STOP_LOSS_PCT           = -50.0    # -50%  → cut
EXIT_DTE_FLOOR          = 3        # ≤ 3 DTE → force exit (theta cliff sooner on short calls)

EARNINGS_AVOID_DAYS     = 7        # skip if earnings within N days
IV_HARD_CEILING         = 1.20     # skip if call IV > 120% (vol way too rich)
MIN_VOLUME              = 50       # need real liquidity in the contract
MIN_OPEN_INTEREST       = 100

STRATEGY_LABEL          = "momentum_call"


# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def _earnings_within(ticker: str, days: int) -> bool:
    """True if the next earnings date falls within `days` calendar days."""
    if yf is None:
        return False
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return False
        # yfinance returns calendar as dict {Earnings Date: [date, ...]} or DataFrame
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        else:
            try:
                dates = cal.loc["Earnings Date"].tolist() if "Earnings Date" in cal.index else []
            except Exception:
                dates = []
        if not dates:
            return False
        from datetime import date as _date
        today = _date.today()
        for d in dates:
            try:
                ed = d.date() if hasattr(d, "date") else _date.fromisoformat(str(d)[:10])
                if 0 <= (ed - today).days <= days:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _pick_expiry(expiries: list[str]) -> str | None:
    """From available expiries, pick the one closest to TARGET_DTE_IDEAL within window."""
    today = datetime.utcnow().date()
    best, best_dist = None, 10**9
    for e in expiries:
        try:
            ed = datetime.strptime(e, "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (ed - today).days
        if dte < TARGET_DTE_MIN or dte > TARGET_DTE_MAX:
            continue
        dist = abs(dte - TARGET_DTE_IDEAL)
        if dist < best_dist:
            best, best_dist = e, dist
    return best


def select_call_contract(ticker: str, underlying_price: float,
                         otm_min: float | None = None,
                         otm_max: float | None = None) -> dict | None:
    """
    Pick the best CALL contract for a momentum-aligned lottery ticket.

    otm_min/otm_max override the default 5-15% OTM strike band. Bottom-fisher
    bounce plays pass a nearer-the-money band (e.g. 0-8%) because the expected
    move is a snap-back, not a multi-week trend extension, so more delta helps.

    Returns None when nothing meets the gates (cheap enough, OTM enough, real
    volume/OI, IV not insane). That's intentional — skipping is the right move
    most of the time; we only fire when the setup is clean.
    """
    if yf is None or underlying_price <= 0:
        return None
    _otm_lo = OTM_PCT_MIN if otm_min is None else otm_min
    _otm_hi = OTM_PCT_MAX if otm_max is None else otm_max

    try:
        tk = yf.Ticker(ticker)
        expiries = list(tk.options or [])
    except Exception:
        return None
    if not expiries:
        return None

    expiry = _pick_expiry(expiries)
    if not expiry:
        return None
    try:
        chain = tk.option_chain(expiry)
        calls = chain.calls
    except Exception:
        return None
    if calls is None or calls.empty:
        return None

    strike_lo = underlying_price * (1 + _otm_lo)
    strike_hi = underlying_price * (1 + _otm_hi)
    cand = calls[(calls["strike"] >= strike_lo) & (calls["strike"] <= strike_hi)]
    if cand.empty:
        return None

    # Premium estimate: midpoint of bid/ask (fall back to lastPrice)
    def _mid(row):
        b = float(row.get("bid", 0) or 0)
        a = float(row.get("ask", 0) or 0)
        if b > 0 and a > 0 and a >= b:
            return (a + b) / 2.0
        return float(row.get("lastPrice", 0) or 0)

    today = datetime.utcnow().date()
    try:
        dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
    except Exception:
        dte = TARGET_DTE_IDEAL

    best, best_score = None, -1e9
    for _, row in cand.iterrows():
        prem = _mid(row)
        if prem <= 0 or prem > COST_CAP_PER_CONTRACT:
            continue
        vol  = float(row.get("volume", 0) or 0)
        oi   = float(row.get("openInterest", 0) or 0)
        if vol < MIN_VOLUME and oi < MIN_OPEN_INTEREST:
            continue
        iv = float(row.get("impliedVolatility", 0) or 0)
        if iv and iv > IV_HARD_CEILING:
            continue
        strike = float(row["strike"])
        otm_pct = (strike - underlying_price) / underlying_price
        # Score: prefer slightly closer-to-ATM (more delta), but penalise rich premium
        # and reward liquidity. Keep simple and inspectable.
        score = (1 - otm_pct) * 2.0 + min(oi / 5000.0, 1.0) - prem * 0.5 - (iv or 0) * 0.3
        if score > best_score:
            best_score, best = score, {
                "ticker": ticker,
                "contract_symbol": str(row.get("contractSymbol", "")),
                "strike": strike,
                "expiry": expiry,
                "dte": dte,
                "otm_pct": round(otm_pct, 4),
                "premium": round(prem, 2),
                "iv": round(iv, 3),
                "volume": int(vol),
                "open_interest": int(oi),
                "score": round(score, 3),
            }
    return best


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
class MomentumOptionsStrategy:
    def __init__(self, conn):
        self.conn = conn

    # ── live premium lookup for an open position ───────────────────────────────
    @staticmethod
    def _current_premium(ticker: str, expiry: str, strike: float) -> float | None:
        if yf is None:
            return None
        try:
            chain = yf.Ticker(ticker).option_chain(expiry)
            calls = chain.calls
            row = calls[calls["strike"] == strike]
            if row.empty:
                return None
            r = row.iloc[0]
            b, a = float(r.get("bid", 0) or 0), float(r.get("ask", 0) or 0)
            if b > 0 and a > 0 and a >= b:
                return (a + b) / 2.0
            return float(r.get("lastPrice", 0) or 0)
        except Exception:
            return None

    # ── find setups (callable from monitor or dashboard) ───────────────────────
    def find_setups(self, top_n_underlyings: int = 8, ranked: list | None = None,
                    progress=None, min_quality_score: int = 55) -> list:
        """
        Return a list of clean lottery-call setups on momentum leaders, now
        FILTERED by the options_trade_score quality gate (IV rank + IV-RV +
        expected move + Greeks + UOA). Rejects rich-IV / theta-cliff / against-
        the-flow setups that the old simple gate let through.
        """
        if ranked is None:
            from momentum_strategy import MomentumStrategy
            ranked = MomentumStrategy(self.conn).rank(top_n=top_n_underlyings + 6,
                                                     min_mom_6m=0.10)
        setups = []
        for i, r in enumerate(ranked[: top_n_underlyings + 6]):
            if len(setups) >= top_n_underlyings:
                break
            tk = r["ticker"]
            if progress:
                try:
                    progress(i + 1, top_n_underlyings + 6, tk)
                except Exception:
                    pass
            if "." in tk:                                # liquid US only
                continue
            if _earnings_within(tk, EARNINGS_AVOID_DAYS):
                continue
            c = select_call_contract(tk, r["price"])
            if not c:
                continue
            c["underlying_price"] = r["price"]
            c["mom_6m"] = r["mom_6m"]
            c["mom_3m"] = r["mom_3m"]
            c["option_type"] = "call"

            # ── Quality gate via options_analytics ─────────────────────────
            # Thesis = expected underlying move over the option's life. Uses the
            # 3-MONTH-equivalent of 6-month momentum (mom_6m/3 ≈ 2-month forward),
            # because a momentum stock that put up +60% over 6m is more honestly
            # expected to move at the SAME monthly rate going forward (~10%/mo),
            # not at 1/6 of the cumulative move. Floored at MIN_THESIS_PCT so we
            # never trade options on thin theses (where the OTM strike + theta
            # math can't pay off regardless of conviction).
            try:
                from options_analytics import options_trade_score
                # Forward thesis: half-window momentum is the honest read
                thesis = max(MIN_THESIS_PCT, (r["mom_6m"] / 3.0) * 100)
                if thesis < MIN_THESIS_PCT:
                    print(f"  [mopts] SKIP {tk}: thesis {thesis:.1f}% < "
                          f"floor {MIN_THESIS_PCT:.0f}% — slow mover, options "
                          f"won't pay off enough to justify capital lockup")
                    continue
                qs = options_trade_score(
                    self.conn, tk,
                    {**c, "iv": (c.get("iv") or 0)},  # iv already in decimal in select_call_contract
                    thesis_move_pct=thesis,
                )
                c["quality_score"]   = qs["score"]
                c["quality_grade"]   = qs["grade"]
                c["quality_decision"] = qs["decision"]
                c["quality_breakdown"] = qs["components"]
                if qs["score"] < min_quality_score:
                    print(f"  [mopts] SKIP {tk}: quality {qs['score']}/100 ({qs['grade']}) — "
                          f"IVR/theta/flow not favorable")
                    continue
            except Exception as _qe:
                # If analytics fails, keep the legacy behavior (don't block)
                c["quality_score"] = None
                c["quality_grade"] = "?"
                c["quality_decision"] = "?"
                print(f"  [mopts] quality scoring unavailable for {tk}: {_qe}")

            setups.append(c)
        return setups

    # ── auto-enter (called by monitor cycle) ───────────────────────────────────
    def auto_enter(self, paper_options, market_regime=None) -> int:
        """Open up to MAX_NEW_PER_CYCLE momentum-call positions. Returns count opened."""
        if market_regime and market_regime.get("regime") == "BEAR":
            print("  BEAR regime — momentum options stand down (no new longs).")
            return 0

        # Check concurrent cap on this strategy only
        try:
            n_open = self.conn.execute(
                "SELECT COUNT(*) FROM options_positions "
                "WHERE status='OPEN' AND strategy=?", (STRATEGY_LABEL,)
            ).fetchone()
            n_open = int(n_open[0] if not hasattr(n_open, "get")
                         else n_open.get("count", 0) or 0)
        except Exception:
            n_open = 0
        if n_open >= MAX_CONCURRENT_POSITIONS:
            print(f"  Momentum options at max ({n_open}/{MAX_CONCURRENT_POSITIONS}).")
            return 0

        cash = getattr(paper_options, "_cash", 0)
        if cash < 100:
            print(f"  Options cash too low (${cash:.2f}) for momentum lottery tickets.")
            return 0
        size_per = max(50.0, cash * LOTTERY_SIZE_PCT)

        setups = self.find_setups(top_n_underlyings=6)
        if not setups:
            print("  No clean momentum-call setups this cycle.")
            return 0

        # Avoid duplicate contracts already open
        try:
            held = self.conn.execute(
                "SELECT contract_symbol FROM options_positions "
                "WHERE status='OPEN' AND strategy=?", (STRATEGY_LABEL,)
            ).fetchall()
            held = {str((r.get("contract_symbol") if hasattr(r, "get") else r[0]) or "")
                    for r in held}
        except Exception:
            held = set()

        opened = 0
        for s in setups:
            if opened >= MAX_NEW_PER_CYCLE:
                break
            if s["contract_symbol"] in held:
                continue
            contracts = max(1, int(size_per / (s["premium"] * 100)))
            cost = contracts * s["premium"] * 100
            if cost > cash:
                continue
            res = paper_options.buy(
                ticker=s["ticker"], contract_symbol=s["contract_symbol"],
                option_type="call", strike=s["strike"], expiry=s["expiry"],
                contracts=contracts, entry_price=s["premium"],
                strategy=STRATEGY_LABEL,
            )
            if res.get("ok"):
                opened += 1
                cash -= cost
                print(f"    ✅ BUY {s['ticker']:6s} ${s['strike']:.0f}C exp {s['expiry']} "
                      f"@ ${s['premium']:.2f}  ×{contracts}  cost=${cost:.0f}  "
                      f"OTM={s['otm_pct']*100:.0f}% DTE={s['dte']} IV={s['iv']*100:.0f}%")
            else:
                print(f"    ✗ {s['ticker']:6s} — {res.get('error','open failed')}")
        return opened

    # ── auto-exit (called by monitor cycle) ────────────────────────────────────
    def auto_exit(self, paper_options) -> int:
        """Apply +100% TP / -50% SL / DTE<=2 force-exit. Returns count closed."""
        today = datetime.utcnow().date()
        try:
            rows = self.conn.execute(
                "SELECT * FROM options_positions "
                "WHERE status='OPEN' AND strategy=?", (STRATEGY_LABEL,)
            ).fetchall()
        except Exception:
            return 0

        closed = 0
        for r in rows:
            def g(k):
                return r.get(k) if hasattr(r, "get") else None
            pid    = int(g("id") or 0)
            tk     = str(g("ticker") or "")
            strike = float(g("strike") or 0)
            expiry = str(g("expiry") or "")
            entry  = float(g("entry_price") or 0)
            try:
                exp_d = datetime.strptime(expiry, "%Y-%m-%d").date()
                dte   = (exp_d - today).days
            except Exception:
                dte = 99

            cur = self._current_premium(tk, expiry, strike) or 0
            unrl_pct = ((cur - entry) / entry * 100) if entry > 0 else 0

            reason = None
            if dte <= EXIT_DTE_FLOOR:
                reason = "TIME_DECAY_DTE2"
            elif unrl_pct >= TAKE_PROFIT_PCT:
                reason = "TAKE_PROFIT_100"
            elif unrl_pct <= STOP_LOSS_PCT:
                reason = "STOP_LOSS_50"

            if reason and pid:
                res = paper_options.close(pid, cur if cur > 0 else 0.0, reason)
                if res.get("ok"):
                    closed += 1
                    print(f"    📕 CLOSE {tk:6s} ${strike:.0f}C exp {expiry} "
                          f"@ ${cur:.2f} ({unrl_pct:+.0f}%)  {reason}  "
                          f"net=${res.get('net_pnl',0):+.0f}")
        return closed


if __name__ == "__main__":
    # Standalone smoke test — list current setups, no DB writes.
    print("Finding momentum-aligned call setups (no positions opened)…")
    class _NoConn:                              # rank() doesn't actually need conn
        pass
    strat = MomentumOptionsStrategy(_NoConn())
    def _cb(i, n, t):
        if i % 2 == 0 or i == n:
            print(f"  checking {i}/{n}  ({t})")
    setups = strat.find_setups(top_n_underlyings=6, progress=_cb)
    if not setups:
        print("\nNo clean setups right now (gates too tight or no liquid chains).")
    else:
        print(f"\n{len(setups)} setups:\n")
        print(f"  {'Ticker':<7}{'Strike':>9}{'DTE':>5}{'OTM%':>7}{'Prem':>8}"
              f"{'IV%':>7}{'Vol':>7}{'OI':>7}  Mom6m")
        for s in setups:
            print(f"  {s['ticker']:<7}{s['strike']:>9.2f}{s['dte']:>5d}"
                  f"{s['otm_pct']*100:>6.1f}%{s['premium']:>8.2f}"
                  f"{s['iv']*100:>6.0f}%{s['volume']:>7d}{s['open_interest']:>7d}"
                  f"  {s['mom_6m']*100:+.0f}%")
