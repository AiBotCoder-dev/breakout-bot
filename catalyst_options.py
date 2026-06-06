"""
catalyst_options.py — "News just hit → here's the exact weekly option to buy."

For a small account ($200) that wants to trade NEWS via WEEKLY options. When a
catalyst fires (Trump/Fed post mentioning a ticker, or high-impact news on a
liquid name), this finds an AFFORDABLE weekly contract that fits the account and
produces a precise ENTER-NOW Telegram alert: ticker, contract, premium, total
cost, expiry, direction, and the catalyst reason.

Calibrated to the account size — at $200 you can only afford contracts with
premium <= ~$2.00 (1 contract = premium x 100). The finder enforces that.

HONEST NOTES (surfaced in every alert):
  • Weekly options on news are HIGH variance: fast theta + IV-crush risk. Size
    is one ticket; expect frequent full losses, occasional big wins.
  • Latency is the monitor cycle (~5 min), not instant. Treat the alert as
    "this setup exists" — verify the price is still reasonable before entering.
  • Direction follows the catalyst sentiment: bullish news -> call, bearish -> put.
"""

from __future__ import annotations

from datetime import datetime, date

try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None


# ── Account calibration ───────────────────────────────────────────────────────
ACCOUNT_SIZE          = 200.0     # your tradeable capital
MAX_CONTRACT_COST     = 200.0     # never suggest a contract costing more than this
WEEKLY_DTE_MIN        = 1
WEEKLY_DTE_MAX        = 9          # ~1.5 weeks max — keep it "weekly"
WEEKLY_DTE_IDEAL      = 5
MIN_VOL, MIN_OI       = 50, 100    # real liquidity so you can actually exit
IV_CEILING            = 2.50       # news names run hot; cap the truly insane
# Strike selection: slightly OTM both directions (cheap but real delta on a move)
OTM_MIN, OTM_MAX      = 0.01, 0.10


def _pick_weekly_expiry(expiries: list) -> str | None:
    today = date.today(); best, bd = None, 10**9
    for e in expiries:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        except Exception:
            continue
        if WEEKLY_DTE_MIN <= dte <= WEEKLY_DTE_MAX and abs(dte - WEEKLY_DTE_IDEAL) < bd:
            best, bd = e, abs(dte - WEEKLY_DTE_IDEAL)
    return best


def find_affordable_weekly(ticker: str, direction: str = "bullish",
                           max_cost: float = MAX_CONTRACT_COST) -> dict | None:
    """
    Find a weekly call (bullish) or put (bearish) that fits `max_cost`.
    Returns the contract dict or None if nothing affordable/liquid exists.
    """
    if yf is None:
        return None
    is_call = direction != "bearish"
    try:
        tk = yf.Ticker(ticker)
        spot = float(tk.fast_info["last_price"])
        exp = _pick_weekly_expiry(list(tk.options or []))
        if not exp or spot <= 0:
            return None
        chain = tk.option_chain(exp)
        df = chain.calls if is_call else chain.puts
        if df is None or df.empty:
            return None
    except Exception:
        return None

    today = date.today()
    try:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
    except Exception:
        dte = WEEKLY_DTE_IDEAL

    # Strike band: slightly OTM in the trade direction
    if is_call:
        lo, hi = spot * (1 + OTM_MIN), spot * (1 + OTM_MAX)
    else:
        lo, hi = spot * (1 - OTM_MAX), spot * (1 - OTM_MIN)
    cand = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
    if cand.empty:
        # fall back to nearest-to-money strikes
        cand = df.iloc[(df["strike"] - spot).abs().argsort()[:5]]

    best, best_score = None, -1e9
    for _, r in cand.iterrows():
        def _f(v, d=0.0):
            try:
                x = float(v); return x if x == x else d
            except Exception:
                return d
        b, a = _f(r.get("bid")), _f(r.get("ask"))
        prem = (a + b) / 2 if (b > 0 and a > 0) else _f(r.get("lastPrice"))
        if prem <= 0:
            continue
        cost = prem * 100
        if cost > max_cost:                 # MUST fit the account
            continue
        vol, oi = _f(r.get("volume")), _f(r.get("openInterest"))
        if vol < MIN_VOL and oi < MIN_OI:
            continue
        iv = _f(r.get("impliedVolatility"))
        if iv and iv > IV_CEILING:
            continue
        strike = _f(r.get("strike"))
        # prefer closest-to-money (more delta) among affordable, decent liquidity
        score = -abs(strike - spot) + min(oi / 5000, 1)
        if score > best_score:
            best_score = score
            best = {
                "ticker": ticker.upper(), "direction": "call" if is_call else "put",
                "contract_symbol": str(r.get("contractSymbol", "")),
                "strike": strike, "expiry": exp, "dte": dte,
                "premium": round(prem, 2), "cost": round(cost, 2),
                "iv": round(iv, 3), "volume": int(vol), "open_interest": int(oi),
                "spot": round(spot, 2),
                "contracts_affordable": int(max_cost // cost) if cost > 0 else 0,
            }
    return best


def format_alert(ticker: str, direction: str, catalyst: str,
                 contract: dict | None, account: float = ACCOUNT_SIZE) -> str:
    """Build the ENTER-NOW Telegram message."""
    arrow = "📈 CALL" if direction != "bearish" else "📉 PUT"
    head = (f"💥 <b>NEWS OPTION SETUP — {ticker}</b>  {arrow}\n"
            f"<b>Catalyst:</b> {catalyst[:180]}\n")
    if not contract:
        return (head +
                f"\n⚠️ No weekly contract fits your ${account:.0f} account "
                f"(all affordable strikes too illiquid or too pricey). "
                f"Skip — don't force it.")
    tp = round(contract["premium"] * 2.0, 2)      # +100%
    sl = round(contract["premium"] * 0.5, 2)      # -50%
    n = max(1, contract["contracts_affordable"])
    return (
        head +
        f"\n<b>ENTER:</b> {contract['ticker']} ${contract['strike']:.0f}"
        f"{contract['direction'][0].upper()} exp {contract['expiry']} "
        f"({contract['dte']}d weekly)\n"
        f"<b>Premium:</b> ${contract['premium']:.2f}  →  "
        f"<b>${contract['cost']:.0f} per contract</b> "
        f"({n}x fits your ${account:.0f})\n"
        f"Underlying ${contract['spot']:.2f} · IV {contract['iv']*100:.0f}% · "
        f"vol {contract['volume']} / OI {contract['open_interest']}\n"
        f"\n<b>Plan:</b> exit +100% (≈${tp:.2f}) · stop -50% (≈${sl:.2f}) · "
        f"don't hold into expiry day.\n"
        f"<i>⚠️ Weekly news options are high-variance (fast theta + IV crush). "
        f"One ticket only. Verify price is still near ${contract['premium']:.2f} "
        f"before entering — alert may be up to ~5 min old.</i>"
    )


def catalyst_to_weekly_alert(ticker: str, sentiment: str, catalyst: str,
                             account: float = ACCOUNT_SIZE) -> str | None:
    """
    One-call helper: given a ticker + catalyst sentiment, find the affordable
    weekly contract and return the formatted alert (or None if not liquid/US).
    """
    t = str(ticker or "").upper().strip()
    if not t or "." in t:
        return None
    direction = "bearish" if sentiment == "bearish" else "bullish"
    contract = find_affordable_weekly(t, direction, account)
    # If nothing affordable, still alert (so the user knows the catalyst hit) but
    # say no contract fits — better than silence on real news.
    return format_alert(t, direction, catalyst, contract, account)


if __name__ == "__main__":
    # Smoke test: pretend NVDA got bullish news; find the affordable weekly call.
    import re
    for tk, sent in [("NVDA", "bullish"), ("AMD", "bullish"), ("SOFI", "bullish")]:
        c = find_affordable_weekly(tk, "bullish")
        print(f"\n{tk}:")
        if c:
            print(f"  ${c['strike']:.0f}C exp {c['expiry']} ({c['dte']}d) "
                  f"prem ${c['premium']:.2f} = ${c['cost']:.0f}/contract "
                  f"({c['contracts_affordable']}x fits $200)")
        else:
            print("  no affordable weekly contract under $200")
