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

# Debit (bull-call) SPREAD structure — long a near-money call, short a further-OTM
# call at the SAME expiry. Caps the home-run tail but (a) raises win rate, (b) cuts
# the typical loss, and (c) is largely vega-neutral, so it sidesteps the unknowable
# entry IV that dominates the naked-call P&L (see BACKTEST_CHANGES.md section 6).
# Strikes are multiples of spot at entry; used by _price_spread() / compare().
SPREAD_LONG_OTM  = 0.00     # long leg strike  = S0 * (1 + this)  -> ATM (highest delta)
SPREAD_SHORT_OTM = 0.10     # short leg strike = S0 * (1 + this)  -> +10% OTM (the cap)


def _half_spread(mid: float) -> float:
    """Half the bid/ask spread in $/share: the larger of a % of mid or a $ floor."""
    return max(mid * HALF_SPREAD_PCT, MIN_HALF_SPREAD)


def buy_fill(mid: float) -> float:
    """Price you ACTUALLY pay to open a call: ask + commission."""
    return mid + _half_spread(mid) + COMMISSION_PER_SHARE


def sell_fill(mid: float) -> float:
    """Price you ACTUALLY collect to close a call: bid − commission (never < 0)."""
    return max(0.0, mid - _half_spread(mid) - COMMISSION_PER_SHARE)


# ── option-structure pricers ───────────────────────────────────────────────
# Each returns (net%, gross%) for the hold. NET crosses the bid/ask spread and
# pays commission on EVERY leg — a spread pays friction on 4 legs (2 per side),
# and that extra cost is the honest price of the structure, included here so the
# comparison isn't rigged in the spread's favour.
def _price_naked(S0, S1, iv0, iv1):
    """The CURRENT structure: one near-money (3% OTM) long call."""
    K = S0 * 1.03
    mid_e = bs_call(S0, K, 21/365, iv0)
    mid_x = bs_call(S1, K, 11/365, iv1)
    gross = (mid_x/mid_e - 1) * 100 if mid_e > 0.01 else 0.0
    cost = buy_fill(mid_e); proceeds = sell_fill(mid_x)
    net = (proceeds/cost - 1) * 100 if cost > 0.01 else 0.0
    return net, gross


def _price_spread(S0, S1, iv0, iv1):
    """Debit spread: long S0*(1+LONG_OTM), short S0*(1+SHORT_OTM), same expiry."""
    Kl = S0 * (1 + SPREAD_LONG_OTM); Ks = S0 * (1 + SPREAD_SHORT_OTM)
    le = bs_call(S0, Kl, 21/365, iv0); se = bs_call(S0, Ks, 21/365, iv0)   # entry mids
    lx = bs_call(S1, Kl, 11/365, iv1); sx = bs_call(S1, Ks, 11/365, iv1)   # exit  mids
    # GROSS: mid-to-mid on the net debit
    debit_g = le - se; value_g = lx - sx
    gross = (value_g/debit_g - 1) * 100 if debit_g > 0.01 else 0.0
    # NET: open = pay ask on long, collect bid on short; close = sell long at bid,
    # buy the short back at ask. Friction therefore hits all four legs.
    debit_n = buy_fill(le) - sell_fill(se)
    value_n = max(0.0, sell_fill(lx) - buy_fill(sx))
    net = (value_n/debit_n - 1) * 100 if debit_n > 0.01 else 0.0
    return net, gross


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


def _scan_stock(df, entry_iv_premium=ENTRY_IV_PREMIUM, structure="naked",
                with_features=False):
    """Walk one name's history; return its non-overlapping call signals.

    Each signal tuple: (date, setup, spot, dir_ret%, opt_net%, opt_gross%[, feats]).
    `entry_iv_premium` scales realized vol into the assumed entry IV — the single
    biggest unknown in the whole backtest, which is why it's a parameter you can sweep.
    When `with_features=True` a 7th element is appended: the winner-gate feature dict
    at the signal bar (used by gate() to validate the meta-label filter). Existing
    callers index s[0..5] and are unaffected by the extra element.
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
            # GROSS = frictionless mid-to-mid; NET = ask in / bid out + commission on
            # every leg. `structure` swaps the naked call for the ATM/+10% debit spread.
            if structure == "spread":
                opt_net, opt_gross = _price_spread(S0, S1, iv0, iv1)
            else:
                opt_net, opt_gross = _price_naked(S0, S1, iv0, iv1)
            rec = [idx[i].date(), setup, round(S0, 2),
                   round(dir_ret, 1), round(opt_net, 0), round(opt_gross, 0)]
            if with_features:
                # mirror _price_naked: a 3%-OTM call, ~21 DTE at entry
                lo20 = float(np.min(C[max(0, i-19):i+1]))
                hi20 = float(np.max(C[max(0, i-19):i+1]))
                rng_pos = (C[i]-lo20)/(hi20-lo20) if hi20 > lo20 else 0.5
                rec.append({
                    "in_uptrend": bool(C[i] > s50[i] > s200[i]),
                    "mom_6m": float(mom6[i]),
                    "rng_pos": float(rng_pos),
                    "rv": float(rv[i]),
                    "otm_pct": 0.03,
                    "dte": 21,
                })
            sigs.append(tuple(rec))
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


def compare(entry_iv_premium=ENTRY_IV_PREMIUM):
    """Head-to-head: the CURRENT naked near-money call vs the ATM/+10% DEBIT SPREAD,
    NET of friction + IV crush, on the SAME signals. This is the gate that has to
    pass before any live structure change is worth building.
    """
    print("STRUCTURE COMPARISON - naked near-money call vs ATM/+10% debit spread")
    print(f"(NET of bid/ask + commission on EVERY leg, IV crush {IV_CRUSH*100:.0f}%, "
          f"entry IV = realized x {entry_iv_premium:.2f})\n")
    dfs = _load_all()
    res = {}
    for struct in ("naked", "spread"):
        net, gross, dirr, setups = [], [], [], []
        for t in STOCKS:
            if t not in dfs:
                continue
            for s in _scan_stock(dfs[t], entry_iv_premium, struct):
                dirr.append(s[3]); net.append(s[4]); gross.append(s[5]); setups.append(s[1])
        res[struct] = (np.array(net), np.array(gross), np.array(dirr), setups)

    print(f"  {'structure':<22} | {'n':>5} | {'NET win%':>8} | {'NET mean%':>9} | "
          f"{'NET median%':>11} | {'gross win%':>10}")
    print("  " + "-" * 80)
    for label, key in (("naked call (CURRENT)", "naked"), ("debit spread ATM/+10%", "spread")):
        o, g, _, _ = res[key]
        if not len(o):
            continue
        print(f"  {label:<22} | {len(o):>5} | {100*(o>0).mean():>7.1f}% | "
              f"{o.mean():>+8.0f}% | {np.median(o):>+10.0f}% | {100*(g>0).mean():>9.1f}%")

    no, _, _, nsets = res["naked"]; so = res["spread"][0]
    if len(no) and len(so):
        print()
        print(f"  >>> WIN-RATE:  spread {100*(so>0).mean():.1f}%  vs  naked "
              f"{100*(no>0).mean():.1f}%   ({100*(so>0).mean()-100*(no>0).mean():+.1f} pts) <<<")
        print(f"  >>> MEDIAN  :  spread {np.median(so):+.0f}%  vs  naked "
              f"{np.median(no):+.0f}%   ({np.median(so)-np.median(no):+.0f} pts) <<<")
        print(f"  >>> MEAN    :  spread {so.mean():+.0f}%  vs  naked {no.mean():+.0f}%  "
              f"(spreads CAP the fat-tail mean - expected) <<<")
        print()
        for st in ("MOMENTUM", "BOTTOM_FISHER"):
            ni = [k for k in range(len(nsets)) if nsets[k] == st]
            if not ni:
                continue
            nn = no[ni]; ss = so[ni]
            print(f"    {st:<14} n={len(ni):<4} win  naked {100*(nn>0).mean():.0f}% "
                  f"-> spread {100*(ss>0).mean():.0f}%   median  naked {np.median(nn):+.0f}% "
                  f"-> spread {np.median(ss):+.0f}%")
    print("\n  Verdict rule: prefer the spread if it raises NET win rate AND median without")
    print("  pushing expectancy negative. If ONLY the mean falls, that's the tail being")
    print("  capped - acceptable for a steadier curve on a small account.")


def gate(entry_iv_premium=ENTRY_IV_PREMIUM):
    """Validate the WINNER GATE (winner_gate.evaluate) the same disciplined way
    `compare` validated the spread: run the SAME signals, split them into what the
    gate would TAKE vs SKIP, and show win rate / mean / median for each set. To be
    worth enforcing, the gate must lift win rate and median on the TAKEN set while
    the SKIPPED set is meaningfully worse (i.e. it really is removing losers).
    """
    from winner_gate import evaluate as _wge, REACH_MAX, CHASE_MAX, MOM_MIN
    print("WINNER-GATE VALIDATION - does the meta-label filter separate winners from losers?")
    print(f"(naked near-money call, NET of friction + IV crush, entry IV = realized x "
          f"{entry_iv_premium:.2f})")
    print(f"(thresholds: reach<={REACH_MAX}, range<={CHASE_MAX:.0%}, mom_6m>={MOM_MIN:.0%}, "
          f"uptrend required)\n")
    dfs = _load_all()
    all_net, taken, skipped, reason_ct = [], [], [], {}
    for t in STOCKS:
        if t not in dfs:
            continue
        for s in _scan_stock(dfs[t], entry_iv_premium, "naked", with_features=True):
            net = s[4]; feats = s[6]
            all_net.append(net)
            r = _wge(feats)
            if r["passed"]:
                taken.append(net)
            else:
                skipped.append(net)
                for rs in r["reasons"]:
                    k = rs.split("(")[0].strip()
                    reason_ct[k] = reason_ct.get(k, 0) + 1
    A = np.array(all_net); TK = np.array(taken); SK = np.array(skipped)

    def line(label, x):
        if not len(x):
            print(f"  {label:<26} n=0"); return
        print(f"  {label:<26} n={len(x):>5}  win {100*(x>0).mean():>5.1f}%  "
              f"mean {x.mean():>+5.0f}%  median {np.median(x):>+5.0f}%")

    print(f"  {'set':<26} {'n':>6}  win-rate / mean / median")
    print("  " + "-" * 70)
    line("ALL signals (no gate)", A)
    line("GATE TAKES (passed)", TK)
    line("GATE SKIPS (rejected)", SK)
    if len(TK) and len(A):
        print()
        print(f"  >>> WIN-RATE: gated {100*(TK>0).mean():.1f}%  vs  ungated "
              f"{100*(A>0).mean():.1f}%   ({100*(TK>0).mean()-100*(A>0).mean():+.1f} pts) <<<")
        print(f"  >>> MEDIAN  : gated {np.median(TK):+.0f}%  vs  ungated "
              f"{np.median(A):+.0f}%   ({np.median(TK)-np.median(A):+.0f} pts) <<<")
        print(f"  >>> KEPT {len(TK)}/{len(A)} signals ({100*len(TK)/len(A):.0f}%); the "
              f"removed {len(SK)} averaged {SK.mean():+.0f}% (the losers it filters out) <<<")
        print("\n  why signals were skipped:")
        for k, v in sorted(reason_ct.items(), key=lambda x: -x[1]):
            print(f"    {k:<42} {v}")
    print("\n  Verdict rule: enforce the gate if it lifts NET win rate AND median on the")
    print("  taken set, and the skipped set is clearly worse. Otherwise keep shadow-only.")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "sweep":
        sweep()
    elif arg == "compare":
        compare()
    elif arg == "gate":
        gate()
    else:
        run()
