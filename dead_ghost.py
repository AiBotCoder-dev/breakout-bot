"""
dead_ghost.py — Keep watching every option we cut as "DEAD". Did any come back?

WHY: the 3-sigma dead-option exit frees slots, but the backtest says it COSTS ~0.7pp
of expectancy, because this is a fat-tailed strategy and 3-sigma recoveries do happen
— and those resurrections are exactly the monster wins. That estimate came from a
model. This measures the real thing.

Every DEAD_OPTION exit is recorded here and then tracked forward on live quotes until
it would otherwise have been closed. Two numbers get captured, because they answer
different questions:

  peak_after_cut   the best the contract EVER traded after we sold it
                   -> "did it reverse at all?" (the emotional question)
  value_at_policy  what it was worth at the point policy C would actually have
                   exited anyway (DTE<=2)
                   -> "what did cutting actually COST us?" (the real question)

The honest metric is value_at_policy, not the peak: a contract that spiked briefly and
died again was never money we would have collected, because policy C holds to DTE<=2
rather than selling into a spike. Reporting both keeps that distinction visible.

After ~20-30 settled ghosts, report() answers whether DEAD_SIGMA=3.0 is too tight,
about right, or should be switched off entirely. Read-only; never trades.
"""

from __future__ import annotations

from datetime import date, datetime, timezone


def _ensure(conn):
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dead_ghosts (
                contract_symbol TEXT PRIMARY KEY,
                underlying TEXT, strike REAL, opt_type TEXT, expiry TEXT,
                cut_at TEXT, cut_premium REAL, entry_premium REAL,
                cut_pct REAL, reach_sigma REAL,
                peak_after_cut REAL, value_at_policy REAL,
                settled INTEGER DEFAULT 0, settled_at TEXT
            )""")
    except Exception:
        pass


def record(conn, contract_symbol, underlying, cut_premium, entry_premium,
           cut_pct=None, reach_sigma=None):
    """Log a contract at the moment the dead-exit closes it."""
    _ensure(conn)
    try:
        from broker import AlpacaPaperBroker as _B
        p = _B.parse_occ_symbol(str(contract_symbol or ""))
        if not p:
            return
        conn.execute(
            "INSERT INTO dead_ghosts (contract_symbol, underlying, strike, opt_type, "
            "expiry, cut_at, cut_premium, entry_premium, cut_pct, reach_sigma, "
            "peak_after_cut, value_at_policy, settled) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,0) ON CONFLICT DO NOTHING",
            (contract_symbol, str(underlying or "").upper(), float(p["strike"]),
             p["type"], str(p["expiry"]),
             datetime.now(timezone.utc).isoformat(),
             float(cut_premium or 0), float(entry_premium or 0),
             float(cut_pct) if cut_pct is not None else None,
             float(reach_sigma) if reach_sigma is not None else None,
             float(cut_premium or 0)))
    except Exception:
        pass


def update(conn, broker) -> int:
    """Track live quotes on every unsettled ghost. Settles at DTE<=2 (where policy C
    would have exited anyway) or once expired. Returns rows touched."""
    _ensure(conn)
    try:
        rows = conn.execute("SELECT * FROM dead_ghosts WHERE settled=0").fetchall()
    except Exception:
        return 0
    n = 0
    for r in rows:
        d = dict(r) if hasattr(r, "keys") else {}
        sym = d.get("contract_symbol")
        if not sym:
            continue
        try:
            exp = date.fromisoformat(str(d.get("expiry"))[:10])
        except Exception:
            continue
        dte = (exp - date.today()).days
        q = broker.get_option_quote(sym) if broker else None
        cur = q["mid"] if q else None

        peak = float(d.get("peak_after_cut") or d.get("cut_premium") or 0)
        if cur is not None:
            peak = max(peak, cur)

        upd = {"peak_after_cut": peak}
        # settle where policy C would have exited anyway
        if dte <= 2:
            settle = cur
            if settle is None and dte < 0:
                # expired with no quote -> intrinsic against the underlying
                try:
                    px = broker.get_price(d.get("underlying")) if broker else None
                    K = float(d.get("strike") or 0)
                    if px:
                        settle = (max(0.0, px - K) if d.get("opt_type") == "call"
                                  else max(0.0, K - px))
                except Exception:
                    settle = None
            if settle is not None:
                upd["value_at_policy"] = round(float(settle), 4)
                upd["settled"] = 1
                upd["settled_at"] = date.today().isoformat()
        sets = ", ".join(f"{k}=?" for k in upd)
        try:
            conn.execute(f"UPDATE dead_ghosts SET {sets} WHERE contract_symbol=?",
                         (*upd.values(), sym))
            n += 1
        except Exception:
            pass
    return n


def report(conn) -> dict:
    """Was cutting them right? Compares what we got against what holding would have."""
    _ensure(conn)
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM dead_ghosts WHERE settled=1").fetchall()]
    except Exception:
        rows = []
    out = {"settled": len(rows), "open": 0}
    try:
        r = conn.execute("SELECT COUNT(*) c FROM dead_ghosts WHERE settled=0").fetchone()
        out["open"] = int((r.get("c") if hasattr(r, "get") else r[0]) or 0)
    except Exception:
        pass
    if not rows:
        out["verdict"] = "no settled dead-ghosts yet — accumulating."
        return out

    recovered = 0; cost_pts = []; peak_recovered = 0
    for d in rows:
        entry = float(d.get("entry_premium") or 0)
        cut = float(d.get("cut_premium") or 0)
        held = float(d.get("value_at_policy") or 0)
        peak = float(d.get("peak_after_cut") or 0)
        if entry <= 0:
            continue
        if held > cut:
            recovered += 1
        if peak > cut * 1.5:
            peak_recovered += 1
        # what cutting cost, in points of the ENTRY premium (comparable to pnl_pct)
        cost_pts.append((held - cut) / entry * 100)
    n = len(cost_pts)
    if n:
        avg = sum(cost_pts) / n
        out.update({"n": n,
                    "recovered_by_policy_exit": recovered,
                    "ever_spiked_50pct": peak_recovered,
                    "avg_cost_of_cutting_pts": round(avg, 1)})
        if avg > 3:
            out["verdict"] = (f"⚠️ cutting COST {avg:+.1f} pts/trade on average "
                              f"({recovered}/{n} recovered) — raise DEAD_SIGMA or "
                              "set DEAD_SIGMA=0 to stop cutting.")
        elif avg < -3:
            out["verdict"] = (f"cutting SAVED {-avg:.1f} pts/trade ({recovered}/{n} "
                              "recovered) — the exit is earning its place.")
        else:
            out["verdict"] = (f"roughly neutral ({avg:+.1f} pts/trade, {recovered}/{n} "
                              "recovered) — keep it for the freed capacity.")
        if n < 20:
            out["verdict"] += f"  [n={n}, still thin — needs ~20-30]"
    return out
