"""
creative_strategies_backtest.py — Unconventional, NON-price-action strategies.

THE USER'S THESIS: markets now trade on news, making candles/price-action less
readable. So instead of reading price, these strategies exploit STRUCTURE —
when money mechanically moves, how news physically arrives, and where flows
concentrate. None of them care what the chart "looks like".

TESTS (all on free yfinance data, with baselines):
  A. OVERNIGHT EDGE      — news drops after hours; is the close→open session
                           where the return lives vs open→close?
  B. TURN-OF-MONTH       — 401k/pension inflows land at month end; hold only
                           last day + first 3 days of each month.
  C. PRE-FOMC DRIFT      — Lucca-Moench: equities drift UP in the 24h before
                           Fed announcements (documented since 1994).
  D. OPEX WEEK           — monthly options expiry (3rd Friday) week flows.
  E. VIX TERM STRUCTURE  — ^VIX vs ^VIX3M: contango = risk-on regime, backwardation
                           = crash regime. A regime filter that ignores price entirely.
  F. NEWS GAPS           — when news gaps a stock ±2%+ at the open, does it
                           CONTINUE (buy the gap) or FADE (fade the gap)?
  G. VOLUME-SPIKE NEWS PROXY — volume >2.5x average = "news happened here";
                           direction of the close that day → forward edge?

Honest output: mean returns vs baseline, win rates, sample sizes. No cherry-picking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

YEARS_IDX = 10          # index-level tests
YEARS_XS  = 5           # cross-sectional tests

XS_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","TSLA","AVGO","NFLX",
    "CRM","ADBE","INTC","QCOM","MU","ORCL","PLTR","COIN","SHOP","UBER",
    "PYPL","ROKU","DKNG","RBLX","JPM","BAC","GS","V","MA","UNH","LLY",
    "ABBV","MRK","PFE","WMT","COST","HD","MCD","NKE","DIS","CAT","BA",
    "GE","XOM","CVX","FCX","SOFI","HOOD","SMCI","MRVL",
]

# FOMC decision days (2-day meeting announcement dates), 2019-2026 YTD
FOMC_DATES = [
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
    "2026-01-28","2026-03-18","2026-04-29",
]


def _dl(t, years):
    end = datetime.now(); start = end - timedelta(days=int(years*365.25)+30)
    raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.dropna(subset=["Close"])


def _ann(mean_daily, n_per_year=252):
    return ((1 + mean_daily) ** n_per_year - 1) * 100


def _fmt(label, arr, base=None):
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        print(f"    {label:<34s} n=0"); return
    extra = ""
    if base is not None and len(base):
        extra = f"  edge {100*(a.mean()-np.asarray(base).mean()):+.3f}%/d"
    print(f"    {label:<34s} n={len(a):<5d} mean {a.mean()*100:+.3f}%  "
          f"win {100*(a>0).mean():.1f}%{extra}")


# ── A. OVERNIGHT vs INTRADAY ──────────────────────────────────────────────────
def test_overnight():
    print("\nA. OVERNIGHT EDGE — who earns the return: the night or the day?")
    for t in ["SPY", "QQQ", "NVDA", "TSLA"]:
        df = _dl(t, YEARS_IDX if t in ("SPY","QQQ") else YEARS_XS)
        if df is None: continue
        o, c = df["Open"].values, df["Close"].values
        overnight = o[1:] / c[:-1] - 1          # close -> next open
        intraday  = c[1:] / o[1:] - 1           # open -> close
        cum_on  = (np.prod(1 + overnight) - 1) * 100
        cum_id  = (np.prod(1 + intraday) - 1) * 100
        print(f"  {t}: ({len(overnight)} days)")
        print(f"    overnight total {cum_on:+.0f}%  (ann {_ann(overnight.mean()):+.1f}%)  "
              f"win {100*(overnight>0).mean():.1f}%")
        print(f"    intraday  total {cum_id:+.0f}%  (ann {_ann(intraday.mean()):+.1f}%)  "
              f"win {100*(intraday>0).mean():.1f}%")


# ── B. TURN-OF-MONTH ──────────────────────────────────────────────────────────
def test_turn_of_month():
    print("\nB. TURN-OF-MONTH — hold only last trading day + first 3 of each month")
    df = _dl("SPY", YEARS_IDX)
    r = df["Close"].pct_change().dropna()
    idx = r.index
    month = pd.Series(idx.month, index=idx)
    is_last = month != month.shift(-1)          # last trading day of month
    pos_in_month = pd.Series(np.arange(len(idx)), index=idx)
    first3 = month != month.shift(1)
    f3 = first3 | first3.shift(1, fill_value=False) | first3.shift(2, fill_value=False)
    tom_mask = (is_last | f3).values
    _fmt("turn-of-month days", r.values[tom_mask], r.values)
    _fmt("all other days", r.values[~tom_mask])
    n_tom = tom_mask.sum()
    print(f"    capital deployed only {100*n_tom/len(r):.0f}% of days "
          f"(ann if TOM-only: {_ann(r.values[tom_mask].mean(), int(252*n_tom/len(r))):+.1f}%)")


# ── C. PRE-FOMC DRIFT ─────────────────────────────────────────────────────────
def test_fomc():
    print("\nC. PRE-FOMC DRIFT — long SPY only into Fed announcement days")
    df = _dl("SPY", 8)
    r = df["Close"].pct_change().dropna()
    dates = pd.to_datetime(FOMC_DATES)
    dset = set(d.date() for d in dates)
    on_day, day_before, baseline = [], [], list(r.values)
    rd = list(r.index.date)
    for i, d in enumerate(rd):
        if d in dset:
            on_day.append(r.values[i])
            if i > 0: day_before.append(r.values[i-1])
    _fmt("FOMC announcement day", on_day, baseline)
    _fmt("day BEFORE announcement", day_before, baseline)
    both = on_day + day_before
    _fmt("2-day pre/announce combo", both, baseline)


# ── D. OPEX WEEK ──────────────────────────────────────────────────────────────
def test_opex():
    print("\nD. OPEX WEEK — week containing the monthly 3rd-Friday expiry")
    df = _dl("SPY", YEARS_IDX)
    r = df["Close"].pct_change().dropna()
    def is_opex_week(d):
        # third Friday of d's month
        first = d.replace(day=1)
        fridays = [first + timedelta(days=x) for x in range(31)
                   if (first + timedelta(days=x)).month == d.month
                   and (first + timedelta(days=x)).weekday() == 4]
        tf = fridays[2]
        return tf - timedelta(days=4) <= d <= tf
    mask = np.array([is_opex_week(d) for d in r.index.date])
    _fmt("OPEX-week days", r.values[mask], r.values)
    _fmt("non-OPEX days", r.values[~mask])


# ── E. VIX TERM STRUCTURE REGIME ──────────────────────────────────────────────
def test_vix_structure():
    print("\nE. VIX TERM STRUCTURE — ^VIX/^VIX3M ratio as a no-price regime filter")
    spy = _dl("SPY", YEARS_IDX); vix = _dl("^VIX", YEARS_IDX); v3m = _dl("^VIX3M", YEARS_IDX)
    if vix is None or v3m is None:
        print("    VIX data unavailable"); return
    ratio = (vix["Close"] / v3m["Close"]).rename("ratio")
    r = spy["Close"].pct_change().rename("spy")
    dfj = pd.concat([ratio, r], axis=1).dropna()
    nxt = dfj["spy"].shift(-1).dropna()
    dfj = dfj.iloc[:-1]
    base = nxt.values
    for lo, hi, lbl in [(0, 0.85, "steep contango (<0.85) — calm"),
                        (0.85, 1.0, "mild contango (0.85-1.0)"),
                        (1.0, 9, "BACKWARDATION (>1.0) — stress")]:
        m = (dfj["ratio"].values >= lo) & (dfj["ratio"].values < hi)
        _fmt(lbl, nxt.values[m], base)
    # simple regime strategy: long only when in contango
    m_con = dfj["ratio"].values < 1.0
    eq_strat = np.prod(1 + nxt.values[m_con]) - 1
    eq_bh = np.prod(1 + base) - 1
    print(f"    long-only-in-contango total {eq_strat*100:+.0f}% vs buy-hold {eq_bh*100:+.0f}% "
          f"(in market {100*m_con.mean():.0f}% of days)")


# ── F. NEWS GAPS ──────────────────────────────────────────────────────────────
def test_gaps():
    print("\nF. NEWS GAPS — stock gaps ±2%+ at open: follow it or fade it?")
    res = {"up_d0": [], "up_f5": [], "dn_d0": [], "dn_f5": []}
    def one(t):
        df = _dl(t, YEARS_XS)
        if df is None or len(df) < 100: return None
        o, c = df["Open"].values, df["Close"].values
        out = {"up_d0": [], "up_f5": [], "dn_d0": [], "dn_f5": []}
        for i in range(1, len(o) - 6):
            gap = o[i] / c[i-1] - 1
            d0  = c[i] / o[i] - 1                  # open -> close same day
            f5  = c[i+5] / c[i] - 1                # close -> 5d later
            if gap >= 0.02:
                out["up_d0"].append(d0); out["up_f5"].append(f5)
            elif gap <= -0.02:
                out["dn_d0"].append(d0); out["dn_f5"].append(f5)
        return out
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed({ex.submit(one, t): t for t in XS_UNIVERSE}):
            o = fut.result()
            if o:
                for k in res: res[k].extend(o[k])
    _fmt("GAP UP  >> same-day open->close", res["up_d0"])
    _fmt("GAP UP  >> next 5 days", res["up_f5"])
    _fmt("GAP DOWN >> same-day open->close", res["dn_d0"])
    _fmt("GAP DOWN >> next 5 days", res["dn_f5"])


# ── G. VOLUME-SPIKE NEWS PROXY ────────────────────────────────────────────────
def test_volume_spike():
    print("\nG. VOLUME SPIKE (news proxy) — vol >2.5x avg; trade WITH the close direction?")
    res = {"green_f5": [], "green_f10": [], "red_f5": [], "red_f10": [], "base5": []}
    def one(t):
        df = _dl(t, YEARS_XS)
        if df is None or len(df) < 100: return None
        c, v = df["Close"].values, df["Volume"].values
        v20 = pd.Series(v).rolling(20).mean().values
        out = {k: [] for k in res}
        for i in range(21, len(c) - 11):
            f5  = c[i+5]  / c[i] - 1
            f10 = c[i+10] / c[i] - 1
            out["base5"].append(f5)
            if v20[i] > 0 and v[i] > 2.5 * v20[i]:
                day = c[i] / c[i-1] - 1
                if day > 0.01:
                    out["green_f5"].append(f5); out["green_f10"].append(f10)
                elif day < -0.01:
                    out["red_f5"].append(f5);   out["red_f10"].append(f10)
        return out
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed({ex.submit(one, t): t for t in XS_UNIVERSE}):
            o = fut.result()
            if o:
                for k in res: res[k].extend(o[k])
    _fmt("news day GREEN >> fwd 5d", res["green_f5"], res["base5"])
    _fmt("news day GREEN >> fwd 10d", res["green_f10"])
    _fmt("news day RED   >> fwd 5d", res["red_f5"], res["base5"])
    _fmt("news day RED   >> fwd 10d", res["red_f10"])


if __name__ == "__main__":
    print("=" * 78)
    print(" CREATIVE / STRUCTURAL STRATEGY LAB — no price-action, no candles")
    print("=" * 78)
    test_overnight()
    test_turn_of_month()
    test_fomc()
    test_opex()
    test_vix_structure()
    test_gaps()
    test_volume_spike()
    print("\ndone.")
