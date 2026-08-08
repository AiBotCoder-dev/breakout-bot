"""
leveraged_shadow_backfill.py — What WOULD the shadow have made since inception?

The live shadow marked a dead CAD feed from 2026-07-26 (stuck at $9,991.99), so its
recorded record is void. This reconstructs the counterfactual on the now-correct
synthetic pricing (US underlyings, 2x daily + drag) and answers two things:

  A) FULL COUNTERFACTUAL — run the real strategy (weekly top-2 by 63d momentum,
     uptrend filter, broad-index airbag, 8bps/side) from inception to today with
     FRESH data. This is the honest "what the system would have done".
  B) WHAT IT ACTUALLY HELD — the live book bought HEU+HFU (chosen on stale July-17
     data). Marked correctly, what would those specific picks have returned?

Benchmarks the same window vs SPY / QQQ buy-hold.
"""

from __future__ import annotations

from datetime import date
import leveraged_shadow as lv

START = date(2026, 7, 26)      # shadow inception
EQUITY0 = 10_000.0
ACTUAL_PICKS = ["HEU.TO", "HFU.TO"]     # what the live (stale) book bought


def _fmt(x):
    return f"${x:,.2f}"


def run():
    hist = lv._history(lv.UNIVERSE)
    if not hist:
        print("no data"); return
    cal = [d for d in hist["SPY"].index if d.date() >= START]
    if not cal:
        print("no trading days since inception"); return
    print(f"Window: {cal[0].date()} -> {cal[-1].date()}  ({len(cal)} trading days)\n")

    # ── A) full counterfactual: run the strategy properly ────────────────────
    cash = EQUITY0
    holdings = {}                      # ticker -> shares
    last_rebal = None
    curve = []
    for d in cal:
        # mark
        eq = cash + sum(sh * float(hist[t]["Close"].loc[d])
                        for t, sh in holdings.items() if d in hist[t].index)
        # broad gate (SPY & EWC above 200sma, evaluated at this date)
        gate = True
        for idx in lv.BROAD_GATE_TICKERS:
            df = hist.get(idx)
            if df is None or d not in df.index:
                gate = False; break
            px = float(df["Close"].loc[d]); s200 = float(df["sma200"].loc[d])
            if not (px == px and s200 == s200 and px > s200):
                gate = False; break

        due = (last_rebal is None) or ((d.date() - last_rebal).days >= lv.REBAL_DAYS)
        if not gate:
            if holdings:                       # airbag -> liquidate
                for t, sh in list(holdings.items()):
                    cash += sh * float(hist[t]["Close"].loc[d]) * (1 - lv.COST_BPS/10_000)
                holdings = {}
        elif due:
            ranked = []
            for t in lv.UNIVERSE:
                df = hist.get(t)
                if df is None or d not in df.index:
                    continue
                px = float(df["Close"].loc[d]); s50 = float(df["sma50"].loc[d])
                s200 = float(df["sma200"].loc[d]); m = float(df["mom"].loc[d])
                if m == m and px > s50 and px > s200:
                    ranked.append((m, t, px))
            ranked.sort(reverse=True)
            target = {t: px for _, t, px in ranked[:lv.TOP_K]}
            # sell what's not in target
            for t in list(holdings):
                if t not in target:
                    cash += holdings.pop(t) * float(hist[t]["Close"].loc[d]) * (1 - lv.COST_BPS/10_000)
            if target:
                eq_now = cash + sum(sh * float(hist[t]["Close"].loc[d])
                                    for t, sh in holdings.items())
                per = eq_now / len(target)
                for t, px in target.items():
                    want = per / px
                    delta = want - holdings.get(t, 0.0)
                    if abs(delta) * px >= 1.0:
                        cash -= delta * px + abs(delta) * px * (lv.COST_BPS/10_000)
                        holdings[t] = want
            last_rebal = d.date()
        eq = cash + sum(sh * float(hist[t]["Close"].loc[d])
                        for t, sh in holdings.items() if d in hist[t].index)
        curve.append((d.date(), eq, sorted(holdings)))

    end_eq = curve[-1][1]
    print("=" * 78)
    print(" A) FULL COUNTERFACTUAL — the strategy run properly since inception")
    print("=" * 78)
    print(f"  start {_fmt(EQUITY0)}   ->   end {_fmt(end_eq)}   "
          f"({(end_eq/EQUITY0-1)*100:+.2f}%)")
    print(f"  final holdings: {', '.join(curve[-1][2]) or 'CASH'}")
    peak = EQUITY0; mdd = 0.0
    for _, e, _h in curve:
        peak = max(peak, e); mdd = min(mdd, (e-peak)/peak)
    print(f"  max drawdown along the way: {mdd*100:.2f}%")
    print("\n  equity path:")
    for dt, e, h in curve:
        print(f"    {dt}  {_fmt(e):>12}   [{', '.join(h) or 'CASH'}]")

    # ── B) what the live book actually held (marked correctly) ───────────────
    print("\n" + "=" * 78)
    print(" B) THE ACTUAL (stale-chosen) PICKS — HEU + HFU, marked correctly")
    print("=" * 78)
    d0, d1 = cal[0], cal[-1]
    per = EQUITY0 / len(ACTUAL_PICKS)
    tot = 0.0
    for t in ACTUAL_PICKS:
        c = hist[t]["Close"]
        p0 = float(c.loc[d0]); p1 = float(c.loc[d1])
        val = per * (p1/p0)
        tot += val
        print(f"  {t:8s} (via {lv.PROXY[t]}) {p0:8.2f} -> {p1:8.2f}  "
              f"{(p1/p0-1)*100:+6.2f}%   {_fmt(per)} -> {_fmt(val)}")
    print(f"  total: {_fmt(EQUITY0)} -> {_fmt(tot)}  ({(tot/EQUITY0-1)*100:+.2f}%)")

    # ── benchmarks ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" BENCHMARKS (same window)")
    print("=" * 78)
    for b in ("SPY", "QQQ"):
        c = hist[b]["Close"]
        r = (float(c.loc[d1])/float(c.loc[d0]) - 1) * 100
        print(f"  {b} buy&hold {r:+6.2f}%   ({_fmt(EQUITY0)} -> {_fmt(EQUITY0*(1+r/100))})")
    print("\n  CAVEAT: ~2 weeks is NOISE, not evidence. And synthetic 2x flatters the")
    print("  real CAD products (tracking friction), so treat these as an upper bound.")


if __name__ == "__main__":
    run()
