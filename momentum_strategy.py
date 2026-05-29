"""
momentum_strategy.py — Liquid trend + momentum strategy (the pivot).

WHY THIS REPLACES THE PATTERN ENGINE
------------------------------------
We MCPT-tested the old chart-pattern engine and it had no edge: worse than random
on large caps (p=0.81), only weakly positive and not significant on small caps
(p=0.23). Worse, the bot was auto-trading thin Canadian micro-caps (.TO/.V/.NE)
with bad data and no liquidity — guaranteeing slippage and stuck positions.

This module implements the one edge that is actually documented to work for retail
in a market like this: TREND + CROSS-SECTIONAL MOMENTUM on LIQUID names.
  1. Absolute trend filter — only go long names above their 200-day SMA (keeps you
     out of downtrends; this single rule is the backbone of trend-following).
  2. Above 50-day SMA — confirm the medium-term trend too.
  3. Cross-sectional momentum — rank the liquid universe by 6-month (and 3-month)
     return; buy the strongest.
  4. Don't chase blow-offs — skip names that are wildly extended (RSI>80 or far
     above the 20-EMA).
Exit on trend break (close below the 50-day SMA) or the existing ATR/stop engine.

The universe is intentionally restricted to deep-liquidity US large-caps + ETFs.
No micro-caps, no OTC, no foreign listings — liquidity IS risk management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None


# ══════════════════════════════════════════════════════════════════════════════
# LIQUID UNIVERSE — deep-liquidity US large/mega-caps + core ETFs
# ══════════════════════════════════════════════════════════════════════════════
LIQUID_UNIVERSE = [
    # Broad / index / sector ETFs (the trend engine's bread and butter)
    "SPY", "QQQ", "IWM", "DIA", "MDY",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLC", "XLRE",
    "SMH", "SOXX", "XBI", "ITB", "KRE", "GLD", "SLV",
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL",
    "AMD", "ADBE", "CRM", "NFLX", "INTC", "CSCO", "QCOM", "TXN", "AMAT", "MU",
    "INTU", "NOW", "PANW", "SNPS", "CDNS", "LRCX", "KLAC", "ASML", "ARM",
    # Comms / consumer / internet
    "DIS", "CMCSA", "T", "VZ", "TMUS", "ABNB", "UBER", "BKNG", "SBUX", "MCD",
    "NKE", "LOW", "HD", "TGT", "COST", "WMT", "PG", "KO", "PEP", "PM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "V", "MA", "PYPL", "BLK", "SPGI",
    # Healthcare / pharma
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN", "ISRG", "VRTX",
    # Industrials / energy / materials
    "CAT", "DE", "BA", "GE", "HON", "UPS", "RTX", "LMT", "XOM", "CVX", "COP", "SLB", "FCX", "LIN",
    # High-beta / momentum favorites (still liquid)
    "PLTR", "COIN", "MSTR", "SHOP", "SQ", "SOFI", "HOOD", "DKNG", "RBLX", "SNOW", "CRWD", "DDOG", "NET", "MRVL", "SMCI",
]


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    dn = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def momentum_signal(df: pd.DataFrame, idx: int | None = None) -> dict | None:
    """
    Compute the trend+momentum signal as of bar `idx` (default: last bar).
    `df` must have at least 210 bars of OHLC(V) with a Close column.

    Returns a dict with the trend filters, momentum, and a composite score, or
    None if there isn't enough history.
    """
    if df is None or len(df) < 210:
        return None
    close = df["Close"]
    if idx is None:
        idx = len(df) - 1
    if idx < 200:
        return None

    window = close.iloc[: idx + 1]
    price = float(window.iloc[-1])
    if price <= 0:
        return None

    sma50  = float(window.iloc[-50:].mean())
    sma200 = float(window.iloc[-200:].mean())
    ema20  = float(window.ewm(span=20, adjust=False).mean().iloc[-1])

    # Momentum: total return over ~6m (126 bars) and ~3m (63 bars)
    def _ret(n):
        if idx - n < 0:
            return 0.0
        p0 = float(window.iloc[-n - 1])
        return (price - p0) / p0 if p0 > 0 else 0.0

    mom_6m = _ret(126)
    mom_3m = _ret(63)
    rsi    = _rsi(window)

    above_200 = price > sma200
    above_50  = price > sma50
    in_uptrend = above_200 and above_50 and sma50 > sma200   # golden alignment
    extended  = (rsi > 80) or (ema20 > 0 and (price - ema20) / ema20 > 0.20)

    # Composite momentum score (cross-sectional ranking key): blend 6m/3m, lightly
    # penalise extension so we prefer strong-but-not-parabolic names.
    score = 0.6 * mom_6m + 0.4 * mom_3m
    if extended:
        score *= 0.5

    return {
        "price": round(price, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "ema20": round(ema20, 2),
        "mom_6m": round(mom_6m, 4),
        "mom_3m": round(mom_3m, 4),
        "rsi": round(rsi, 1),
        "above_200": above_200,
        "above_50": above_50,
        "in_uptrend": in_uptrend,
        "extended": extended,
        "score": round(score, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST STRATEGY (for MCPT validation) — pure time-series trend-following
# ══════════════════════════════════════════════════════════════════════════════
def trend_following_trades(df: pd.DataFrame, **_) -> list:
    """
    Long-only trend-following replay for MCPT. Enter when price is above both the
    50- and 200-day SMA (50>200 alignment); exit on a close below the 50-day SMA.
    Returns realized trade returns. Signature matches permutation_test's trades_fn.
    """
    if df is None or len(df) < 210:
        return []
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)
    s = pd.Series(close)
    sma50  = s.rolling(50).mean().to_numpy()
    sma200 = s.rolling(200).mean().to_numpy()

    trades, in_pos, entry = [], False, 0.0
    for i in range(200, n):
        if np.isnan(sma200[i]) or np.isnan(sma50[i]):
            continue
        if not in_pos:
            if close[i] > sma200[i] and close[i] > sma50[i] and sma50[i] > sma200[i]:
                in_pos, entry = True, close[i]
        else:
            if close[i] < sma50[i] and entry > 0:
                trades.append((close[i] - entry) / entry)
                in_pos = False
    if in_pos and entry > 0:
        trades.append((close[-1] - entry) / entry)
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY (live ranking + exit rule)
# ══════════════════════════════════════════════════════════════════════════════
class MomentumStrategy:
    def __init__(self, conn=None, universe: list | None = None):
        self.conn = conn
        self.universe = [t.upper() for t in (universe or LIQUID_UNIVERSE)]

    @staticmethod
    def _download(ticker: str) -> pd.DataFrame | None:
        if yf is None:
            return None
        try:
            raw = yf.download(ticker, period="2y", interval="1d",
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty or len(raw) < 210:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            return raw.dropna(subset=["Close"])
        except Exception:
            return None

    def rank(self, top_n: int = 10, min_mom_6m: float = 0.0,
             progress=None) -> list:
        """
        Rank the liquid universe. Returns the top-N names that are in a confirmed
        uptrend, sorted by momentum score, each with entry/stop/target.
        """
        rows = []
        total = len(self.universe)
        for i, t in enumerate(self.universe):
            if progress:
                try:
                    progress(i + 1, total, t)
                except Exception:
                    pass
            df = self._download(t)
            if df is None:
                continue
            sig = momentum_signal(df)
            if not sig:
                continue
            if not sig["in_uptrend"]:
                continue
            if sig["mom_6m"] < min_mom_6m:
                continue

            # ATR(14) for a trend-based stop
            high, low, close = df["High"], df["Low"], df["Close"]
            tr = pd.concat([(high - low),
                            (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1] or (sig["price"] * 0.02))

            price = sig["price"]
            # Stop: the tighter-protecting of "just below 50SMA" vs "2*ATR", but
            # never wider than ~12% (liquid names don't need huge stops).
            stop = max(sig["sma50"] * 0.99, price - 2 * atr, price * 0.88)
            stop = min(stop, price * 0.97)          # at least ~3% room
            target = price * 1.25                   # ride the trend; trailing does the rest

            rows.append({
                "ticker": t,
                "score": sig["score"],
                "mom_6m": sig["mom_6m"],
                "mom_3m": sig["mom_3m"],
                "rsi": sig["rsi"],
                "price": price,
                "entry": price,
                "stop": round(stop, 2),
                "target": round(target, 2),
                "sma50": sig["sma50"],
                "sma200": sig["sma200"],
                "extended": sig["extended"],
            })

        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:top_n]

    def should_exit(self, ticker: str) -> dict | None:
        """Trend-break exit: close below the 50-day SMA ends the trend trade."""
        df = self._download(ticker)
        if df is None:
            return None
        sig = momentum_signal(df)
        if not sig:
            return None
        if not sig["above_50"]:
            return {"exit": True, "reason": "TREND_BREAK_50SMA",
                    "price": sig["price"], "sma50": sig["sma50"]}
        return {"exit": False, "price": sig["price"], "sma50": sig["sma50"]}


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL VALIDATION — the RIGHT test for momentum selection
# ══════════════════════════════════════════════════════════════════════════════
def cross_sectional_backtest(universe: list | None = None, years: int = 3,
                             mom: int = 126, rebal: int = 21, top_n: int = 10,
                             n_null: int = 500, seed: int = 42) -> dict:
    """
    Does buying the STRONGEST-momentum names beat (a) the equal-weight universe
    (pure beta) and (b) randomly picking names from the same uptrending set?

    - Each rebalance (every `rebal` days): rank names above their 200-SMA by
      `mom`-day return (past data only), hold top_n equal-weight for `rebal` days.
    - Benchmark = equal-weight ALL names with forward data (beta).
    - Null = pick top_n RANDOM uptrending names, repeated n_null times.
    p-value = fraction of random-selection portfolios that match/beat momentum.
    """
    if yf is None:
        raise RuntimeError("yfinance unavailable")
    uni = [t.upper() for t in (universe or LIQUID_UNIVERSE)]
    raw = yf.download(uni, period=f"{years}y", interval="1d",
                      progress=False, auto_adjust=True)
    px = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    px = px.dropna(how="all")
    # keep names with >=90% history coverage
    px = px.dropna(axis=1, thresh=int(len(px) * 0.9))
    tickers = list(px.columns)
    if len(tickers) < top_n + 5 or len(px) < 220:
        raise RuntimeError("insufficient data for cross-sectional test")

    sma200 = px.rolling(200).mean()
    vals   = px.to_numpy()
    sma_v  = sma200.to_numpy()
    n_days = len(px)
    col_ix = {tk: j for j, tk in enumerate(tickers)}
    rng = np.random.default_rng(seed)

    strat, bench = [], []
    null = [[] for _ in range(n_null)]

    i = 200
    while i + rebal < n_days:
        if i - mom < 0:
            i += rebal; continue
        cur  = vals[i]
        past = vals[i - mom]
        fwd_row = vals[i + rebal]

        elig, momv, fwd = [], {}, {}
        for tk, j in col_ix.items():
            c, p0, f, sm = cur[j], past[j], fwd_row[j], sma_v[i][j]
            if not (np.isfinite(c) and c > 0):
                continue
            if np.isfinite(f) and f > 0:
                fwd[tk] = f / c - 1.0
            if np.isfinite(sm) and c > sm and np.isfinite(p0) and p0 > 0:
                elig.append(tk)
                momv[tk] = c / p0 - 1.0

        elig_fwd = [tk for tk in elig if tk in fwd]
        if len(momv) >= top_n and fwd and len(elig_fwd) >= top_n:
            top = sorted(momv, key=momv.get, reverse=True)[:top_n]
            strat.append(float(np.mean([fwd[tk] for tk in top if tk in fwd])))
            bench.append(float(np.mean(list(fwd.values()))))
            for k in range(n_null):
                pick = rng.choice(elig_fwd, size=top_n, replace=False)
                null[k].append(float(np.mean([fwd[tk] for tk in pick])))
        i += rebal

    def comp(rs):
        return float(np.prod([1.0 + r for r in rs]) - 1.0) if rs else 0.0

    n_periods   = len(strat)
    strat_total = comp(strat)
    bench_total = comp(bench)
    null_totals = np.array([comp(r) for r in null]) if n_periods else np.array([0.0])
    p_val = (1 + int(np.sum(null_totals >= strat_total))) / (n_null + 1)

    yrs = (n_periods * rebal) / 252.0 if n_periods else 1
    def cagr(tot):
        return (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0

    return {
        "tickers": len(tickers), "periods": n_periods, "years": round(yrs, 2),
        "rebal_days": rebal, "mom_days": mom, "top_n": top_n, "n_null": n_null,
        "strat_total": round(strat_total, 4), "strat_cagr": round(cagr(strat_total), 4),
        "bench_total": round(bench_total, 4), "bench_cagr": round(cagr(bench_total), 4),
        "momentum_premium": round(strat_total - bench_total, 4),
        "null_total_mean": round(float(null_totals.mean()), 4),
        "null_total_pct95": round(float(np.percentile(null_totals, 95)), 4),
        "p_value_vs_random": round(p_val, 4),
    }


def print_cs_report(r: dict):
    sep = "=" * 64
    print(f"\n{sep}")
    print("  CROSS-SECTIONAL MOMENTUM TEST — pick strongest vs random/benchmark")
    print(sep)
    print(f"  Universe: {r['tickers']} names | {r['periods']} rebalances over "
          f"~{r['years']}y | top {r['top_n']} every {r['rebal_days']}d | "
          f"{r['mom_days']}d momentum")
    print(f"\n  MOMENTUM (buy strongest) : total {r['strat_total']*100:+.1f}%  "
          f"(CAGR {r['strat_cagr']*100:+.1f}%)")
    print(f"  BENCHMARK (equal-weight) : total {r['bench_total']*100:+.1f}%  "
          f"(CAGR {r['bench_cagr']*100:+.1f}%)  <- pure beta")
    print(f"  Momentum premium         : {r['momentum_premium']*100:+.1f}% over benchmark")
    print(f"\n  RANDOM-SELECTION NULL    : mean {r['null_total_mean']*100:+.1f}%  "
          f"| 95th pct {r['null_total_pct95']*100:+.1f}%")
    print(f"  p-value (random >= momentum): {r['p_value_vs_random']}")
    p = r["p_value_vs_random"]
    prem = r["momentum_premium"]
    verdict = ("STRONG selection edge (p<=0.05) AND beats beta" if p <= 0.05 and prem > 0 else
               "Selection edge significant but ~= beta"        if p <= 0.05 else
               "Beats beta but selection not significant"      if prem > 0 else
               "NO selection edge — momentum picking ~ random")
    print(f"\n  VERDICT: {verdict}")
    print(sep + "\n")


if __name__ == "__main__":
    print("Ranking liquid momentum universe...")
    strat = MomentumStrategy()
    def _cb(i, total, t):
        if i % 20 == 0 or i == total:
            print(f"  scanned {i}/{total}")
    top = strat.rank(top_n=15, progress=_cb)
    print(f"\nTop {len(top)} momentum names in confirmed uptrends:\n")
    print(f"  {'Ticker':<7}{'6m%':>8}{'3m%':>8}{'RSI':>6}{'Price':>10}{'Stop':>9}  Flags")
    for r in top:
        flag = "EXTENDED" if r["extended"] else ""
        print(f"  {r['ticker']:<7}{r['mom_6m']*100:>7.1f}%{r['mom_3m']*100:>7.1f}%"
              f"{r['rsi']:>6.0f}{r['price']:>10.2f}{r['stop']:>9.2f}  {flag}")
