"""
ghost_exits.py — Shadow-test alternative EXIT policies on every real trade.

The reviewer's best idea (#3): the post-exit data shows our -50% stop cut winners,
but that's REGRET (it ignores the losers the stop saved us from). The only honest
way to settle it is to let alternative exits run in shadow on the SAME entries and
compare the full distribution after enough trades.

For every real option BUY, we record a ghost and then, each cycle, evaluate what
these exit policies WOULD have yielded (read-only — changes nothing about the live
trade, which keeps using the current exit manager as the control):

  A  NO_STOP_TIME   — no stop at all; exit only at DTE<=2 (pure "let the thesis
                      play out; theta is the only enemy")
  B  WIDE_STOP_80   — hard stop at -80% (vs live -50%), then trail 30% after +50%
  C  TRAIL_AFTER_30 — no hard stop; once up +30% trail 30% below peak; else DTE<=2

Compare A/B/C against the REAL trade's journal outcome (the -50% control). After
~50-100 closed ghosts, `report()` tells us — with live data — whether a wider or
no stop actually beats the current one on mean/median/win, not on regret.
"""

from __future__ import annotations


def _ensure(conn):
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ghost_exits (
                contract_symbol TEXT PRIMARY KEY,
                underlying TEXT, entry_premium REAL, entry_at TEXT,
                dte0 INTEGER, peak_prem REAL,
                a_done INTEGER DEFAULT 0, a_pnl REAL,
                b_done INTEGER DEFAULT 0, b_pnl REAL,
                c_done INTEGER DEFAULT 0, c_pnl REAL
            )""")
    except Exception:
        pass


def record(conn, contract_symbol, underlying, entry_premium, dte=None):
    """Log a ghost at entry (read-only shadow of the real buy)."""
    if not entry_premium or entry_premium <= 0:
        return
    _ensure(conn)
    from datetime import datetime, timezone
    try:
        conn.execute(
            "INSERT INTO ghost_exits (contract_symbol, underlying, entry_premium, "
            "entry_at, dte0, peak_prem) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT (contract_symbol) DO NOTHING",
            (contract_symbol, str(underlying or "").upper(), float(entry_premium),
             datetime.now(timezone.utc).isoformat(),
             int(dte) if dte is not None else None, float(entry_premium)))
    except Exception:
        pass


def _dte_of(symbol):
    try:
        from broker import AlpacaPaperBroker
        from datetime import date
        p = AlpacaPaperBroker.parse_occ_symbol(symbol)
        if p:
            return (p["expiry"] - date.today()).days
    except Exception:
        pass
    return None


def update(conn, broker):
    """Evaluate the 3 ghost policies for every still-open ghost. Cheap: one
    option quote per open ghost per cycle. Records each policy's exit % when its
    condition first triggers."""
    _ensure(conn)
    try:
        rows = conn.execute(
            "SELECT * FROM ghost_exits WHERE a_done=0 OR b_done=0 OR c_done=0"
        ).fetchall()
    except Exception:
        return 0
    n = 0
    for r in rows:
        d = dict(r) if hasattr(r, "keys") else {}
        sym = d.get("contract_symbol"); entry = float(d.get("entry_premium") or 0)
        if not sym or entry <= 0:
            continue
        q = broker.get_option_quote(sym) if broker else None
        dte = _dte_of(sym)
        cur = q["mid"] if q else None
        # expired / no quote near expiry -> settle at ~intrinsic-floor (0 for OTM)
        if cur is None:
            if dte is not None and dte <= 0:
                cur = 0.0
            else:
                continue          # transient no-quote; try next cycle
        peak = max(float(d.get("peak_prem") or entry), cur)
        pct = (cur / entry - 1) * 100
        peakpct = (peak / entry - 1) * 100
        upd = {"peak_prem": peak}

        # A: no stop, exit only at DTE<=2
        if not d.get("a_done") and dte is not None and dte <= 2:
            upd["a_done"], upd["a_pnl"] = 1, round(pct, 1)
        # B: -80% hard stop, trail 30% after +50%
        if not d.get("b_done"):
            if pct <= -80:
                upd["b_done"], upd["b_pnl"] = 1, -80.0
            elif peakpct >= 50 and cur <= peak * 0.70:
                upd["b_done"], upd["b_pnl"] = 1, round((peak*0.70/entry-1)*100, 1)
            elif dte is not None and dte <= 2:
                upd["b_done"], upd["b_pnl"] = 1, round(pct, 1)
        # C: no hard stop; trail 30% after +30%
        if not d.get("c_done"):
            if peakpct >= 30 and cur <= peak * 0.70:
                upd["c_done"], upd["c_pnl"] = 1, round((peak*0.70/entry-1)*100, 1)
            elif dte is not None and dte <= 2:
                upd["c_done"], upd["c_pnl"] = 1, round(pct, 1)

        sets = ", ".join(f"{k}=?" for k in upd)
        try:
            conn.execute(f"UPDATE ghost_exits SET {sets} WHERE contract_symbol=?",
                         (*upd.values(), sym))
            n += 1
        except Exception:
            pass
    return n


def report(conn) -> dict:
    """Win rate / mean / median per ghost policy vs the real (-50%) control."""
    _ensure(conn)
    import statistics as st
    out = {}
    for pol, col in [("A_no_stop_time", "a_pnl"), ("B_wide_80_stop", "b_pnl"),
                     ("C_trail_after_30", "c_pnl")]:
        try:
            rows = conn.execute(
                f"SELECT {col} v FROM ghost_exits WHERE {col} IS NOT NULL").fetchall()
            vals = [float(x.get("v") if hasattr(x, "get") else x[0]) for x in rows]
        except Exception:
            vals = []
        if vals:
            out[pol] = {"n": len(vals),
                        "win": round(100*sum(1 for v in vals if v > 0)/len(vals), 0),
                        "mean": round(sum(vals)/len(vals), 0),
                        "median": round(st.median(vals), 0)}
        else:
            out[pol] = {"n": 0, "win": 0, "mean": 0, "median": 0}
    # real control from the journal (-50% stop policy)
    try:
        rows = conn.execute("SELECT pnl_pct FROM broker_trade_journal "
                            "WHERE status='CLOSED' AND pnl_pct IS NOT NULL").fetchall()
        vals = [float(x.get("pnl_pct") if hasattr(x, "get") else x[0]) for x in rows]
        if vals:
            out["REAL_-50_control"] = {"n": len(vals),
                "win": round(100*sum(1 for v in vals if v > 0)/len(vals), 0),
                "mean": round(sum(vals)/len(vals), 0),
                "median": round(st.median(vals), 0)}
    except Exception:
        pass
    return out


# ── PRE-REGISTERED ADOPTION RULE (reviewer-locked, no cherry-picking) ─────────
# Settled with the second-opinion reviewer: when >=MIN_RESOLVED ghosts have
# resolved, score each policy and adopt the highest scorer that passes ALL hard
# constraints. If none passes, DON'T adopt — surface that and keep the control.
MIN_RESOLVED = 80          # ~40 per policy: the floor for a directional read

# ── CONSTRAINTS (re-specified 2026-08-11 — the original set was UNSATISFIABLE) ──
# The first version rejected any policy whose worst trade was below -70% and any
# with >20% of trades below -50%. For a LONG-OPTION book those are structural: a
# bought option can always go to -100%, so EVERY conceivable policy failed and the
# rule could never adopt anything. It fired at 93/80 resolved and rejected all
# three — including policy C, whose median was -5% against the control's ~-52%.
#
# The fix: judge a policy AGAINST THE CONTROL rather than against absolute floors,
# because -100% tails are the instrument's nature, not a policy defect. What we
# actually care about is: is it profitable, does it beat what we already do, and
# does it not blow out the tail relative to the control?
MIN_PER_POLICY   = 25      # a policy needs its own sample before it can be adopted
DQ_MEAN_LT       = 0.0     # mean return < 0                      -> REJECT
DQ_MUST_BEAT_CONTROL = True  # mean must exceed the live control   -> else why switch
DQ_TAIL_VS_CONTROL   = 1.25  # tail rate > 1.25x the control's     -> REJECT

# Composite score weights. The old "ruin" term keyed off the worst single trade,
# which is ~-100% for every long-option policy — a degenerate component that added
# nothing. Its weight is redistributed to Sortino/median/tail.
W_SORTINO, W_MEDIAN, W_WIN, W_TAIL = 0.35, 0.30, 0.15, 0.20


def _pol_stats(vals):
    import statistics as st
    a = sorted(float(v) for v in vals)
    n = len(a)
    if n == 0:
        return None
    mean = sum(a) / n
    dn = [v for v in a if v < 0]
    dd = (sum(v * v for v in dn) / len(dn)) ** 0.5 if dn else 1e-9
    sortino = mean / dd if dd > 0 else 0.0
    worst = a[0]
    tail50 = 100.0 * sum(1 for v in a if v < -50) / n
    win = 100.0 * sum(1 for v in a if v > 0) / n
    return {"n": n, "mean": mean, "median": st.median(a), "sortino": sortino,
            "worst": worst, "tail50": tail50, "win": win}


def decision(conn) -> dict:
    """Apply the pre-registered rule. Returns {ready, resolved, winner, table,
    note} — read-only; it recommends, it does not change the live policy."""
    _ensure(conn)
    policies = {"A_no_stop_time": "a_pnl", "B_wide_80_stop": "b_pnl",
                "C_trail_after_30": "c_pnl"}
    stats = {}; total_resolved = 0
    for name, col in policies.items():
        try:
            rows = conn.execute(
                f"SELECT {col} v FROM ghost_exits WHERE {col} IS NOT NULL").fetchall()
            vals = [float(x.get("v") if hasattr(x, "get") else x[0]) for x in rows]
        except Exception:
            vals = []
        s = _pol_stats(vals)
        if s:
            stats[name] = s
            total_resolved += s["n"]

    # normalise components to [0,1] across the candidate policies, then weight.
    def _norm(key, invert=False):
        xs = [s[key] for s in stats.values()]
        lo, hi = (min(xs), max(xs)) if xs else (0, 1)
        span = (hi - lo) or 1.0
        return {n: ((hi - s[key]) if invert else (s[key] - lo)) / span
                for n, s in stats.items()}

    n_sort = _norm("sortino"); n_med = _norm("median"); n_win = _norm("win")
    n_tail = _norm("tail50", invert=True)          # lower tail = better

    # the live control (the real -50%-stop trades) is the bar to clear
    ctrl = None
    try:
        rows = conn.execute("SELECT pnl_pct FROM broker_trade_journal "
                            "WHERE status='CLOSED' AND pnl_pct IS NOT NULL").fetchall()
        ctrl = _pol_stats([float(x.get("pnl_pct") if hasattr(x, "get") else x[0])
                           for x in rows])
    except Exception:
        ctrl = None

    table = []
    for name, s in stats.items():
        reasons = []
        if s["n"] < MIN_PER_POLICY:
            reasons.append(f"n={s['n']}<{MIN_PER_POLICY}")
        if s["mean"] < DQ_MEAN_LT:
            reasons.append(f"mean {s['mean']:+.0f}%<0")
        if ctrl and DQ_MUST_BEAT_CONTROL and s["mean"] <= ctrl["mean"]:
            reasons.append(f"mean {s['mean']:+.0f}% <= control {ctrl['mean']:+.0f}%")
        if ctrl and ctrl["tail50"] > 0 and s["tail50"] > ctrl["tail50"] * DQ_TAIL_VS_CONTROL:
            reasons.append(f"tail {s['tail50']:.0f}% > {DQ_TAIL_VS_CONTROL:g}x control "
                           f"{ctrl['tail50']:.0f}%")
        score = (W_SORTINO * n_sort[name] + W_MEDIAN * n_med[name]
                 + W_WIN * n_win[name] + W_TAIL * n_tail[name])
        table.append({"policy": name, **{k: round(s[k], 1) for k in
                      ("n", "mean", "median", "sortino", "worst", "tail50", "win")},
                      "score": round(score, 3), "passes": not reasons,
                      "fails": reasons})
    table.sort(key=lambda r: (r["passes"], r["score"]), reverse=True)

    ready = total_resolved >= MIN_RESOLVED
    winner = None; note = ""
    eligible = [r for r in table if r["passes"]]
    if not ready:
        note = (f"{total_resolved}/{MIN_RESOLVED} resolved ghosts — "
                "accumulating; rule not yet armed.")
    elif not eligible:
        note = ("Armed, but no policy beats the live control on the re-specified "
                "constraints — keep the -50% control.")
    else:
        winner = eligible[0]["policy"]
        note = (f"ADOPT {winner} (score {eligible[0]['score']}; mean "
                f"{eligible[0]['mean']:+.0f}% vs control "
                f"{ctrl['mean']:+.0f}%)" if ctrl else f"ADOPT {winner}")
    return {"ready": ready, "resolved": total_resolved, "min_resolved": MIN_RESOLVED,
            "winner": winner, "table": table, "note": note,
            "control": ({k: round(ctrl[k], 1) for k in
                         ("n", "mean", "median", "sortino", "tail50", "win")}
                        if ctrl else None)}
