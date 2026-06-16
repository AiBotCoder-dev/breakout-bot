"""
full_bot_backtest.py — Replay the BOT'S call decision process over 5y / 5 stocks.

Marks every bar where the bot WOULD buy a call (its two validated call edges) and
measures accuracy two ways:
  1. Directional  — did the underlying rise over the 10-day hold? (the thesis)
  2. Option       — would a NEAR-MONEY call (the bot's new default structure)
                    actually have MADE money, NET of real-world friction?

Option P&L is now reported BOTH ways so the cost of reality is visible:
  GROSS — the old optimistic lie: filled at the BS mid-price, IV held flat.
  NET   — what you'd really keep: buy at the ask, sell at the bid, pay
          commission each side, and let IV crush over the hold.
See buy_fill() / sell_fill() and the IV_CRUSH constant below for the model.

The bot's CALL process (validated edges only — the ones it actually trades):
  MOMENTUM  : price > rising 50SMA > rising 200SMA AND 6-mo momentum > +10%
              (the MCPT p=0.004 cross-sectional momentum entry)
  BOTTOM-FISHER : RSI(14) < 30 AND within 6% of the 60-day low AND today closes
              green (the MCPT p=0.005 oversold-at-support entry)

Non-overlapping (one trade at a time per name). Lists every signal so you can
eyeball the points, then reports per-stock + aggregate accuracy.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

STOCKS = ["NVDA", "AAPL", "TSLA", "AMD", "JPM"]
HOLD = 10           # trading days (≈ the bot's 2-week option window)
R = 0.04
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * N(d1) - K * math.exp(-R * T) * N(d2)


# ---------------------------------------------------------------------------
# Execution friction & volatility assumptions  (the REALISTIC backtest)
# ---------------------------------------------------------------------------
# The original version told three silent lies, each of which inflates option
# win rates. They are made explicit (and tunable) here:
#   1. fills at the mid-price        -> real options cross a bid/ask spread
#   2. zero commissions              -> brokers charge ~$0.65 / contract / side
#   3. IV held flat entry -> exit    -> IV mean-reverts/bleeds ("IV crush")
# Tune these to your own broker & names; the point is that they are NON-ZERO.

HALF_SPREAD_PCT = 0.025        # half the bid/ask spread as a % of mid (~5% round trip)
MIN_HALF_SPREAD = 0.02         # $/share floor for the half-spread (cheap opts trade wide)
COMMISSION_PER_SHARE = 0.0065  # $0.65 / contract ÷ 100 shares, charged on EACH side
IV_CRUSH = 0.10                # exit IV = entry IV × (1 − IV_CRUSH): vol mean-reversion

# Entry IV = realized vol × ENTRY_IV_PREMIUM. The original code hard-coded 1.10.
# A LIVE snapshot (option_iv_calibration.py, 2026-06-16) measured the real near-money
# IV/realized ratio at ~0.87 for these names — i.e. CURRENTLY implied < realized.
# That ratio is regime-dependent and NOT recoverable from price history, so the honest
# tool is the sensitivity sweep (`python full_bot_backtest.py sweep`), not one number.
ENTRY_IV_PREMIUM = 0.87


def _half_spread(mid: float) -> float:
    """Half the bid/ask spread in $/share: the larger of a % of mid or a $ floor."""
    return max(mid * HALF_SPREAD_PCT, MIN_HALF_SPREAD)


def buy_fill(mid: float) -> float:
    """Price you ACTUALLY pay to open a call: ask + commission."""
    return mid + _half_spread(mid) + COMMISSION_PER_SHARE


def sell_fill(mid: float) -> float:
    """Price you ACTUALLY collect to close a call: bid − commission (never < 0)."""
    return max(0.0, mid - _half_spread(mid) - COMMISSION_PER_SHARE)


def _rsi(c, p=14):
    d = np.diff(c, prepend=c[0]); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    out = np.full_like(c, 50.0, float); ag = al = 0.0
    for i in range(1, len(c)):
        if i <= p: ag = (ag*(i-1)+g[i])/i; al = (al*(i-1)+l[i])/i
        else: ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+l[i])/p
        out[i] = 100.0 if al < 1e-12 else 100.0 - 100.0/(1.0+ag/al)
    return out


def _load(t, years=5):
    end = datetime.now(); start = end - timedelta(days=int(years*365.25)+260)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.dropna(subset=["Close"]).copy()
    c = df["Close"]
    df["sma50"] = c.rolling(50).mean(); df["sma200"] = c.rolling(200).mean()
    df["rv20"] = np.log(c/c.shift()).rolling(20).std()*np.sqrt(252)
    df["rsi"] = _rsi(c.values, 14)
    df["low60"] = df["Low"].rolling(60).min()
    df["mom6"] = c/c.shift(126) - 1
    return df.dropna()


def _scan_stock(df, entry_iv_premium=ENTRY_IV_PREMIUM):
    """Walk one name's history; return its non-overlapping call signals.

    Each signal tuple: (date, setup, spot, dir_ret%, opt_net%, opt_gross%).
    `entry_iv_premium` scales realized vol into the assumed entry IV — the single
    biggest unknown in the whole backtest, which is why it's a parameter you can sweep.
    """
    C = df["Close"].values
    s50 = df["sma50"].values; s200 = df["sma200"].values
    rsi = df["rsi"].values; low60 = df["low60"].values
    mom6 = df["mom6"].values; rv = df["rv20"].values
    idx = df.index
    n = len(C); i = 210
    sigs = []
    while i < n - HOLD:
        setup = None
        if (C[i] > s50[i] > s200[i] and s50[i] > s50[i-10]
                and s200[i] > s200[i-20] and mom6[i] > 0.10):
            setup = "MOMENTUM"
        elif (rsi[i] < 30 and low60[i] > 0 and (C[i]/low60[i]-1) <= 0.06
              and C[i] > C[i-1]):
            setup = "BOTTOM_FISHER"
        if setup:
            S0 = C[i]; S1 = C[i+HOLD]
            dir_ret = (S1/S0 - 1) * 100
            iv0 = max(0.15, min(1.5, rv[i]*entry_iv_premium))  # entry IV (calibrated premium)
            iv1 = max(0.05, iv0 * (1.0 - IV_CRUSH))            # exit IV after the crush
            K = S0 * 1.03            # near-money call (the bot's default band)
            mid_e = bs_call(S0, K, 21/365, iv0)                # theoretical mid at entry
            mid_x = bs_call(S1, K, 11/365, iv1)                # theoretical mid at exit
            # GROSS = the old frictionless lie (mid-to-mid); NET = ask in / bid out + comm
            opt_gross = (mid_x/mid_e - 1) * 100 if mid_e > 0.01 else 0.0
            cost = buy_fill(mid_e); proceeds = sell_fill(mid_x)
            opt_net = (proceeds/cost - 1) * 100 if cost > 0.01 else 0.0
            sigs.append((idx[i].date(), setup, round(S0, 2),
                         round(dir_ret, 1), round(opt_net, 0), round(opt_gross, 0)))
            i += HOLD               # non-overlapping
        else:
            i += 1
    return sigs


def _load_all(years=5):
    """Download every name once so run() and sweep() can reuse the same data.

    `years` controls how much history is pulled; walk_forward.py asks for more
    (10y) so it has enough post-warmup data to cut multiple OOS folds.
    """
    dfs = {}
    for t in STOCKS:
        try:
            dfs[t] = _load(t, years)
        except Exception as e:
            print(f"{t}: load failed {e}")
    return dfs


def run(dfs=None, entry_iv_premium=ENTRY_IV_PREMIUM):
    print(f"Replaying the bot's CALL process - {len(STOCKS)} stocks, 5y, {HOLD}d hold")
    print(f"(entry IV = realized vol x {entry_iv_premium:.2f}, IV crush {IV_CRUSH*100:.0f}%, "
          f"~{HALF_SPREAD_PCT*200:.0f}% round-trip spread + commission)\n")
    if dfs is None:
        dfs = _load_all()
    agg_dir, agg_opt, agg_gross, agg_setup = [], [], [], []

    for t in STOCKS:
        if t not in dfs:
            continue
        sigs = _scan_stock(dfs[t], entry_iv_premium)
        for s in sigs:
            agg_dir.append(s[3]); agg_opt.append(s[4])
            agg_gross.append(s[5]); agg_setup.append(s[1])
        d = np.array([s[3] for s in sigs])
        o = np.array([s[4] for s in sigs]); g = np.array([s[5] for s in sigs])
        if len(sigs):
            print(f"=== {t}: {len(sigs)} call signals ===")
            print(f"   directional win {100*(d>0).mean():.0f}%  |  "
                  f"option win NET {100*(o>0).mean():.0f}% "
                  f"(gross {100*(g>0).mean():.0f}%)  |  "
                  f"avg option NET {o.mean():+.0f}%  median {np.median(o):+.0f}%")
            # show a handful of example signal points (NET = after friction + IV crush)
            for s in sigs[:6]:
                tag = "WIN " if s[4] > 0 else "loss"
                print(f"     {s[0]}  {s[1]:<13} ${s[2]:>8.2f}  10d {s[3]:+5.1f}%  "
                      f"-> call NET {s[4]:+4.0f}% (gross {s[5]:+4.0f}%)  [{tag}]")
            print()

    d = np.array(agg_dir); o = np.array(agg_opt); g = np.array(agg_gross)
    print("=" * 70)
    print(f" AGGREGATE - {len(d)} call signals across {len(STOCKS)} stocks, 5y")
    print("=" * 70)
    print(f"  Directional accuracy (stock up in 10d)  : {100*(d>0).mean():.1f}%")
    print( "  --- frictionless (OLD, the optimistic lie) ---")
    print(f"  OPTION win rate  (mid-to-mid, flat IV)  : {100*(g>0).mean():.1f}%")
    print(f"  Option mean / median return             : {g.mean():+.0f}% / {np.median(g):+.0f}%")
    print( "  --- realistic (NET of spread + comm + crush) ---")
    print(f"  OPTION win rate  (ask in / bid out)     : {100*(o>0).mean():.1f}%")
    print(f"  Option mean / median return             : {o.mean():+.0f}% / {np.median(o):+.0f}%")
    print(f"  Option expectancy per trade             : {o.mean():+.1f}%")
    print(f"  >>> COST OF REALITY: {100*(g>0).mean() - 100*(o>0).mean():.1f} pts of win rate, "
          f"{o.mean() - g.mean():+.0f}% avg return <<<")
    print()
    for st in ("MOMENTUM", "BOTTOM_FISHER"):
        idxs = [k for k in range(len(agg_setup)) if agg_setup[k] == st]
        mn = np.array([agg_opt[k] for k in idxs])
        gs = np.array([agg_gross[k] for k in idxs])
        if len(mn):
            print(f"    {st:<14} n={len(mn):<4} option win NET {100*(mn>0).mean():.0f}% "
                  f"(gross {100*(gs>0).mean():.0f}%)  avg NET {mn.mean():+.0f}% "
                  f"(gross {gs.mean():+.0f}%)")


def sweep(premiums=(0.80, 0.90, 1.00, 1.10, 1.20, 1.40)):
    """Re-price every signal across a RANGE of entry-IV premiums.

    The whole point: option P&L is highly sensitive to entry IV, and entry IV is
    NOT recoverable from price data. So we show the entire range instead of
    pretending one multiplier is right. Data is loaded once and reused per premium.
    """
    print("ENTRY-IV SENSITIVITY SWEEP - the option edge vs an IV we can't actually pin down")
    print("(live calibration today put the real near-money IV/realized ratio near 0.87,")
    print(" but it swings hard by regime - so treat the option win rate as a RANGE.)\n")
    dfs = _load_all()
    print(f"  {'IV premium':>10} | {'NET win%':>8} | {'NET avg%':>8} | "
          f"{'gross win%':>10} | {'gross avg%':>10} | {'n':>4}")
    print("  " + "-" * 66)
    for p in premiums:
        net, gross = [], []
        for t in STOCKS:
            if t not in dfs:
                continue
            for s in _scan_stock(dfs[t], p):
                net.append(s[4]); gross.append(s[5])
        net = np.array(net); gross = np.array(gross)
        if not len(net):
            continue
        flag = "  <- live-calibrated" if abs(p - ENTRY_IV_PREMIUM) < 1e-9 else ""
        print(f"  {p:>10.2f} | {100*(net>0).mean():>7.1f}% | {net.mean():>+7.0f}% | "
              f"{100*(gross>0).mean():>9.1f}% | {gross.mean():>+9.0f}% | {len(net):>4}{flag}")
    print("\n  Takeaway: the NET edge moves materially with the entry-IV assumption, and")
    print("  that assumption is NOT knowable from price history. Only REAL historical")
    print("  option chains can settle it. Until then, the option win rate is a RANGE,")
    print("  not a single number - size and expectations accordingly.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        run()
