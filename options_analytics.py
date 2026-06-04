"""
options_analytics.py — The data + math that gives options buyers actual edge.

WHAT THIS MODULE PROVIDES
-------------------------
The five things documented to matter for retail options profitability — none of
which the bot was using before:

  1. IV Rank / HV Rank       — "is vol historically cheap or expensive?"
                                The single biggest filter; buying calls at IVR > 70
                                is the most documented losing pattern in retail.
  2. Expected Move           — what the market is pricing in via the ATM straddle.
                                Only trade when your thesis exceeds the implied move.
  3. IV–RV Spread            — Goyal-Saretto (2009): names where IV < realized
                                vol systematically outperform. The "cheap options"
                                edge.
  4. Black-Scholes Greeks    — Δ Γ Θ ν.  Proper risk + sizing, not raw $.
  5. Unusual Options Activity — volume/OI anomalies as a (noisy) smart-money tell.

Plus `options_trade_score(...)` that fuses all of the above into one 0-100 quality
grade so the bot only fires when the math is actually favorable.

HONEST LIMITATIONS
------------------
True IV Rank needs 52 weeks of historical IV data, which yfinance does not
provide. We use two approximations:
  (a) HV Rank from 52-week realized-volatility distribution (immediate proxy)
  (b) iv_snapshots table — the bot snaps current ATM IV daily and the rank
      improves as the snapshot history grows. After ~60 days the rank is
      meaningful; after 252 days it's true IV Rank.
The dashboard surfaces which one is being used so you always know.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, date, timedelta

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except Exception:                       # pragma: no cover
    yf = pd = np = None


# ── Defaults ──────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.05    # fallback if ^TNX fetch fails
TRADING_DAYS   = 252


# ══════════════════════════════════════════════════════════════════════════════
# REALIZED VOLATILITY + HV RANK (immediate IVR proxy)
# ══════════════════════════════════════════════════════════════════════════════
def realized_vol(ticker: str, lookback: int = 30) -> float | None:
    """Close-to-close annualized realized vol over `lookback` trading days (decimal)."""
    if yf is None:
        return None
    try:
        raw = yf.download(ticker, period="3mo", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        c = raw["Close"].dropna()
        if len(c) < lookback + 2:
            return None
        rets = np.log(c / c.shift(1)).dropna().iloc[-lookback:]
        return float(rets.std() * math.sqrt(TRADING_DAYS))
    except Exception:
        return None


def hv_rank(ticker: str, lookback_window: int = 30) -> dict | None:
    """
    52-week realized-vol percentile rank.
    Returns {hv_now, hv_min, hv_max, hv_rank_pct, hv_percentile_pct}.
    Used as an IV-Rank proxy until iv_snapshots history is built up.
    """
    if yf is None:
        return None
    try:
        raw = yf.download(ticker, period="1y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < lookback_window + 50:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        c = raw["Close"].dropna()
        log_ret = np.log(c / c.shift(1)).dropna()
        rv_series = (log_ret.rolling(lookback_window).std() *
                     math.sqrt(TRADING_DAYS)).dropna()
        if rv_series.empty:
            return None
        hv_now = float(rv_series.iloc[-1])
        hv_min = float(rv_series.min())
        hv_max = float(rv_series.max())
        rank_pct = 0.0 if hv_max == hv_min else (hv_now - hv_min) / (hv_max - hv_min) * 100
        pct      = float((rv_series < hv_now).mean() * 100)
        return {
            "hv_now":            round(hv_now * 100, 1),
            "hv_min":            round(hv_min * 100, 1),
            "hv_max":            round(hv_max * 100, 1),
            "hv_rank_pct":       round(rank_pct, 1),
            "hv_percentile_pct": round(pct, 1),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# IV SNAPSHOTS — building real IV Rank over time
# ══════════════════════════════════════════════════════════════════════════════
def _atm_iv(ticker: str) -> dict | None:
    """Pull nearest-expiry ATM call+put IV. Returns None on failure."""
    if yf is None:
        return None
    try:
        tk = yf.Ticker(ticker)
        exps = list(tk.options or [])
        if not exps:
            return None
        chain = tk.option_chain(exps[0])
        spot = float(tk.fast_info["last_price"])
        def _atm(df):
            if df is None or df.empty:
                return None
            d = df.assign(_d=(df["strike"] - spot).abs()).sort_values("_d").iloc[0]
            iv = float(d.get("impliedVolatility", 0) or 0)
            return iv if iv > 0 else None
        call_iv = _atm(chain.calls)
        put_iv  = _atm(chain.puts)
        avg = None
        if call_iv and put_iv:
            avg = (call_iv + put_iv) / 2
        elif call_iv:
            avg = call_iv
        elif put_iv:
            avg = put_iv
        if avg is None:
            return None
        return {
            "spot":    round(spot, 2),
            "expiry":  exps[0],
            "iv_call": round((call_iv or 0) * 100, 2),
            "iv_put":  round((put_iv or 0) * 100, 2),
            "iv_avg":  round(avg * 100, 2),
        }
    except Exception:
        return None


def snapshot_iv(conn, tickers: list, progress=None) -> int:
    """
    Persist today's ATM IV for each ticker into iv_snapshots (one row per ticker
    per calendar day). Returns count written. Run from monitor once per day so
    the bot builds its own historical IV record.
    """
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS iv_snapshots (
                ticker         TEXT,
                snapshot_date  TEXT,
                spot           REAL,
                iv_avg         REAL,
                iv_call        REAL,
                iv_put         REAL,
                PRIMARY KEY (ticker, snapshot_date)
            )
        """)
    except Exception as e:
        print(f"  [opt-an] table init failed: {e}")
        return 0

    today = date.today().isoformat()
    n = 0
    for i, t in enumerate(tickers):
        if progress:
            try:
                progress(i + 1, len(tickers), t)
            except Exception:
                pass
        d = _atm_iv(t)
        if not d:
            continue
        try:
            conn.execute(
                "INSERT INTO iv_snapshots "
                "(ticker, snapshot_date, spot, iv_avg, iv_call, iv_put) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT (ticker, snapshot_date) DO UPDATE SET "
                "spot=excluded.spot, iv_avg=excluded.iv_avg, "
                "iv_call=excluded.iv_call, iv_put=excluded.iv_put",
                (t.upper(), today, d["spot"], d["iv_avg"], d["iv_call"], d["iv_put"])
            )
            n += 1
        except Exception:
            continue
    return n


def iv_rank(conn, ticker: str, fallback_hv: bool = True) -> dict | None:
    """
    Return {iv_now, iv_min, iv_max, iv_rank_pct, iv_percentile_pct, source, n}.

    Tries the bot's own iv_snapshots first — accurate but needs history. Falls
    back to HV Rank when we don't yet have enough snapshots (< 30 days).
    """
    snap = _atm_iv(ticker)
    iv_now = snap["iv_avg"] if snap else None

    rows = []
    try:
        rows = conn.execute(
            "SELECT iv_avg FROM iv_snapshots WHERE ticker=? "
            "AND snapshot_date >= ? ORDER BY snapshot_date ASC",
            (ticker.upper(),
             (date.today() - timedelta(days=365)).isoformat())
        ).fetchall()
    except Exception:
        rows = []
    vals = []
    for r in rows:
        v = r.get("iv_avg") if hasattr(r, "get") else r[0]
        try:
            vals.append(float(v))
        except Exception:
            continue

    if len(vals) >= 30 and iv_now is not None:
        lo, hi = min(vals), max(vals)
        rank = 0.0 if hi == lo else (iv_now - lo) / (hi - lo) * 100
        pct  = float(sum(1 for v in vals if v < iv_now) / len(vals) * 100)
        return {
            "iv_now":            round(iv_now, 1),
            "iv_min":            round(lo, 1),
            "iv_max":            round(hi, 1),
            "iv_rank_pct":       round(rank, 1),
            "iv_percentile_pct": round(pct, 1),
            "source":            "iv_snapshots",
            "n":                 len(vals),
        }

    if not fallback_hv:
        return None
    hv = hv_rank(ticker)
    if not hv:
        return None
    return {
        "iv_now":            iv_now if iv_now is not None else hv["hv_now"],
        "iv_min":            hv["hv_min"],
        "iv_max":            hv["hv_max"],
        "iv_rank_pct":       hv["hv_rank_pct"],
        "iv_percentile_pct": hv["hv_percentile_pct"],
        "source":            "hv_proxy",
        "n":                 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED MOVE FROM ATM STRADDLE
# ══════════════════════════════════════════════════════════════════════════════
def expected_move(ticker: str, expiry: str | None = None) -> dict | None:
    """
    1-sigma expected move implied by the ATM straddle.
    Returns {expiry, dte, spot, straddle_price, exp_move_dollars, exp_move_pct,
             upper, lower}.

    Rule of thumb: ATM straddle ≈ 1.25 × 1-sigma expected move. We use the
    cleaner straight straddle ≈ expected move (more conservative).
    """
    if yf is None:
        return None
    try:
        tk = yf.Ticker(ticker)
        exps = list(tk.options or [])
        if not exps:
            return None
        if expiry is None:
            expiry = exps[0]
        if expiry not in exps:
            expiry = exps[0]
        chain = tk.option_chain(expiry)
        spot = float(tk.fast_info["last_price"])
        def _atm_mid(df):
            if df is None or df.empty:
                return None
            d = df.assign(_d=(df["strike"] - spot).abs()).sort_values("_d").iloc[0]
            b, a = float(d.get("bid", 0) or 0), float(d.get("ask", 0) or 0)
            if b > 0 and a > 0 and a >= b:
                return (a + b) / 2
            return float(d.get("lastPrice", 0) or 0)
        c = _atm_mid(chain.calls)
        p = _atm_mid(chain.puts)
        if not c or not p:
            return None
        straddle = c + p
        try:
            d_exp = datetime.strptime(expiry, "%Y-%m-%d").date()
            dte = max(0, (d_exp - date.today()).days)
        except Exception:
            dte = 0
        em_pct = (straddle / spot * 100) if spot else 0.0
        return {
            "expiry":           expiry,
            "dte":              dte,
            "spot":             round(spot, 2),
            "straddle_price":   round(straddle, 2),
            "exp_move_dollars": round(straddle, 2),
            "exp_move_pct":     round(em_pct, 2),
            "upper":            round(spot + straddle, 2),
            "lower":            round(spot - straddle, 2),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# IV-RV SPREAD (Goyal-Saretto cheap-options edge)
# ══════════════════════════════════════════════════════════════════════════════
def iv_rv_spread(ticker: str) -> dict | None:
    """
    {iv_pct, rv30_pct, spread_pct, signal}.
    spread > 0  → options expensive relative to realized vol (sellers favored)
    spread < 0  → options cheap (buyers favored — the Goyal-Saretto edge)
    """
    snap = _atm_iv(ticker)
    rv = realized_vol(ticker, lookback=30)
    if not snap or rv is None:
        return None
    iv_pct = snap["iv_avg"]
    rv_pct = round(rv * 100, 1)
    spread = round(iv_pct - rv_pct, 1)
    signal = ("cheap"     if spread < -3 else
              "neutral"   if abs(spread) <= 3 else
              "expensive" if spread <= 10 else "rich")
    return {"iv_pct": iv_pct, "rv30_pct": rv_pct,
            "spread_pct": spread, "signal": signal}


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES GREEKS
# ══════════════════════════════════════════════════════════════════════════════
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes_greeks(spot: float, strike: float, dte_days: int,
                         iv: float, option_type: str = "call",
                         r: float = RISK_FREE_RATE) -> dict | None:
    """
    Δ Γ Θ ν for a European option. IV in decimal (e.g. 0.45 = 45%).
    Theta returned as $/day, Vega as $/1pt-of-IV (i.e. /100 of decimal IV move).
    """
    if spot <= 0 or strike <= 0 or iv <= 0 or dte_days <= 0:
        return None
    t = dte_days / 365.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    is_call = option_type.lower() == "call"
    if is_call:
        delta = _norm_cdf(d1)
        theta_year = (- (spot * _norm_pdf(d1) * iv) / (2 * sqrt_t)
                      - r * strike * math.exp(-r * t) * _norm_cdf(d2))
    else:
        delta = _norm_cdf(d1) - 1.0
        theta_year = (- (spot * _norm_pdf(d1) * iv) / (2 * sqrt_t)
                      + r * strike * math.exp(-r * t) * _norm_cdf(-d2))

    gamma = _norm_pdf(d1) / (spot * iv * sqrt_t)
    vega  = spot * _norm_pdf(d1) * sqrt_t / 100   # per 1 IV point
    theta_day = theta_year / 365.0

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 5),
        "theta_per_day": round(theta_day, 4),
        "vega_per_pt":   round(vega, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# UNUSUAL OPTIONS ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════
def unusual_options_activity(ticker: str,
                             min_volume: int = 200,
                             vol_oi_ratio: float = 3.0,
                             max_results: int = 8) -> list:
    """
    Scan the current chain for contracts whose volume / OI ratio is unusually
    high — a (noisy) smart-money flow signal. Returns ranked list of contracts.
    """
    if yf is None:
        return []
    try:
        tk = yf.Ticker(ticker)
        exps = list(tk.options or [])
        if not exps:
            return []
    except Exception:
        return []

    out = []
    for exp in exps[:4]:                # nearest 4 expiries
        try:
            ch = tk.option_chain(exp)
        except Exception:
            continue
        for kind, df in (("call", ch.calls), ("put", ch.puts)):
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                def _f(v, default=0.0):
                    try:
                        f = float(v)
                        return f if f == f else default     # NaN check
                    except Exception:
                        return default
                vol = _f(row.get("volume", 0))
                oi  = _f(row.get("openInterest", 0))
                if vol < min_volume:
                    continue
                ratio = vol / oi if oi > 0 else vol
                if ratio < vol_oi_ratio:
                    continue
                b, a = _f(row.get("bid", 0)), _f(row.get("ask", 0))
                prem = ((a + b) / 2) if (b > 0 and a > 0) else _f(row.get("lastPrice", 0))
                iv_  = _f(row.get("impliedVolatility", 0))
                out.append({
                    "contract": str(row.get("contractSymbol", "")),
                    "expiry":   exp,
                    "strike":   _f(row.get("strike", 0)),
                    "type":     kind,
                    "volume":   int(vol),
                    "open_interest": int(oi),
                    "vol_oi_ratio": round(ratio, 2),
                    "premium":  round(prem, 2),
                    "iv_pct":   round(iv_ * 100, 1),
                })
    out.sort(key=lambda r: r["vol_oi_ratio"], reverse=True)
    return out[:max_results]


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE TRADE-QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════
def options_trade_score(conn, ticker: str, contract: dict,
                        thesis_move_pct: float | None = None) -> dict:
    """
    Score 0-100 for "should I take this options trade?" The math the bot uses
    to filter `momentum_options` setups.

    `contract` expected keys: strike, expiry, dte, premium, iv (decimal),
                              option_type ('call'/'put'), underlying_price.
    `thesis_move_pct`: if you have a directional thesis, pass the expected
                      underlying move % so we can compare it to implied move.
    """
    pts: dict = {}
    total = 0

    # ── IV Rank (cheap = good) — up to 35 pts ─────────────────────────────────
    ivr = iv_rank(conn, ticker)
    if ivr:
        r = ivr["iv_rank_pct"]
        s = (35 if r <= 30 else 25 if r <= 50 else 15 if r <= 70 else 5 if r <= 85 else 0)
        pts["iv_rank"] = {"pts": s, "max": 35,
                          "label": f"IVR {r:.0f}% ({ivr['source']})"}
    else:
        pts["iv_rank"] = {"pts": 10, "max": 35, "label": "IV rank unavailable (neutral)"}
    total += pts["iv_rank"]["pts"]

    # ── IV-RV spread (Goyal-Saretto edge) — up to 20 pts ──────────────────────
    spr = iv_rv_spread(ticker)
    if spr:
        sig = spr["signal"]
        s = (20 if sig == "cheap" else 12 if sig == "neutral" else
             6 if sig == "expensive" else 0)
        pts["iv_rv_spread"] = {"pts": s, "max": 20,
                               "label": f"IV-RV {spr['spread_pct']:+.1f}pt ({sig})"}
    else:
        pts["iv_rv_spread"] = {"pts": 8, "max": 20, "label": "IV-RV unavailable"}
    total += pts["iv_rv_spread"]["pts"]

    # ── Expected move vs thesis — up to 20 pts ────────────────────────────────
    em = expected_move(ticker, contract.get("expiry"))
    if em and thesis_move_pct is not None:
        if thesis_move_pct > em["exp_move_pct"] * 1.3:
            s, msg = 20, f"thesis {thesis_move_pct:.1f}% >> implied {em['exp_move_pct']:.1f}%"
        elif thesis_move_pct > em["exp_move_pct"]:
            s, msg = 12, f"thesis {thesis_move_pct:.1f}% > implied {em['exp_move_pct']:.1f}%"
        else:
            s, msg = 3, f"thesis {thesis_move_pct:.1f}% ≤ implied {em['exp_move_pct']:.1f}%"
        pts["expected_move"] = {"pts": s, "max": 20, "label": msg}
    elif em:
        pts["expected_move"] = {"pts": 10, "max": 20,
                                "label": f"implied {em['exp_move_pct']:.1f}% (no thesis)"}
    else:
        pts["expected_move"] = {"pts": 8, "max": 20, "label": "expected move n/a"}
    total += pts["expected_move"]["pts"]

    # ── Greeks sanity (theta/premium ratio) — up to 15 pts ────────────────────
    g = black_scholes_greeks(
        contract.get("underlying_price", 0) or 0,
        contract.get("strike", 0) or 0,
        int(contract.get("dte", 0) or 0),
        float(contract.get("iv", 0) or 0),
        contract.get("option_type", "call"),
    )
    if g and contract.get("premium"):
        # daily theta as % of premium per day — lower = better
        theta_burn_pct = abs(g["theta_per_day"]) / max(contract["premium"], 0.01) * 100
        if theta_burn_pct < 1.0:
            s, msg = 15, f"θ burn {theta_burn_pct:.1f}%/day · Δ {g['delta']:+.2f}"
        elif theta_burn_pct < 2.5:
            s, msg = 10, f"θ burn {theta_burn_pct:.1f}%/day · Δ {g['delta']:+.2f}"
        elif theta_burn_pct < 5.0:
            s, msg = 5,  f"θ burn {theta_burn_pct:.1f}%/day · Δ {g['delta']:+.2f}"
        else:
            s, msg = 0,  f"θ burn {theta_burn_pct:.1f}%/day TOO HIGH · Δ {g['delta']:+.2f}"
        pts["greeks"] = {"pts": s, "max": 15, "label": msg}
    else:
        pts["greeks"] = {"pts": 7, "max": 15, "label": "greeks unavailable"}
    total += pts["greeks"]["pts"]

    # ── Unusual options activity (smart-money confirmation) — up to 10 pts ────
    uoa = unusual_options_activity(ticker, min_volume=200, vol_oi_ratio=3.0)
    same_side_uoa = [u for u in uoa if u["type"] == contract.get("option_type", "call")]
    if len(same_side_uoa) >= 3:
        pts["uoa"] = {"pts": 10, "max": 10,
                      "label": f"{len(same_side_uoa)} unusual same-side contracts"}
    elif len(same_side_uoa) >= 1:
        pts["uoa"] = {"pts": 5, "max": 10,
                      "label": f"{len(same_side_uoa)} unusual same-side contract(s)"}
    else:
        pts["uoa"] = {"pts": 0, "max": 10, "label": "no unusual flow"}
    total += pts["uoa"]["pts"]

    grade = ("A+" if total >= 85 else "A" if total >= 75 else
             "B" if total >= 60 else "C" if total >= 45 else
             "D" if total >= 30 else "F")
    decision = ("BUY" if total >= 65 else "WATCH" if total >= 50 else "SKIP")

    return {
        "score": int(total),
        "grade": grade,
        "decision": decision,
        "components": pts,
    }


def full_report(conn, ticker: str) -> dict:
    """Convenience: every analytic in one dict, for the Options Lab dashboard."""
    return {
        "ticker":   ticker.upper(),
        "iv_rank":  iv_rank(conn, ticker),
        "exp_move": expected_move(ticker),
        "iv_rv":    iv_rv_spread(ticker),
        "uoa":      unusual_options_activity(ticker),
        "atm_iv":   _atm_iv(ticker),
    }
