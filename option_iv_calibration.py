"""
option_iv_calibration.py - measure the REAL entry-IV premium and option spreads.

WHY THIS EXISTS
---------------
full_bot_backtest.py prices the ENTRY implied vol as `realized_vol * 1.1`. That
1.1 multiplier is a guess, and it's almost certainly too low: when you buy
near-money calls on momentum names, you pay an *implied* vol that sits well above
trailing *realized* vol (the "vol risk premium"). Understating entry IV makes
every entry look cheaper than it really is -> inflated option returns.

True HISTORICAL option chains are not available for free (yfinance only exposes
the CURRENT chain). So this script does the next best thing: it samples the LIVE
chain today and measures, per name:
  - real implied vol of the ~3% OTM, ~21-DTE call (matches the backtest structure)
  - trailing 20d realized vol (same definition the backtest uses)
  - the implied / realized RATIO  -> the realistic entry-IV multiplier
  - the real bid/ask spread %      -> validates HALF_SPREAD_PCT in the backtest

LIMITATION: this is a single SNAPSHOT (today), not a historical average. The vol
risk premium changes with regime. Treat the output as a sanity-calibration of the
*level*, then plug a defensible multiplier into full_bot_backtest.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

STOCKS = ["NVDA", "AAPL", "TSLA", "AMD", "JPM"]
TARGET_DTE = 21        # match the backtest's ~21-day entry tenor
OTM = 0.03             # 3% OTM near-money call (the bot's default band)


def realized_vol(close: pd.Series, win: int = 20) -> float:
    r = np.log(close / close.shift())
    return float(r.tail(win).std() * np.sqrt(252))


def nearest_expiry(options, target_dte: int):
    today = datetime.now().date()
    best = None
    for e in options:
        dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        if dte <= 0:
            continue
        if best is None or abs(dte - target_dte) < abs(best[1] - target_dte):
            best = (e, dte)
    return best


def run():
    print(f"Calibrating entry IV + spreads vs the backtest's rv*1.1 / 5% assumptions\n")
    rows = []
    for t in STOCKS:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="3mo")["Close"]
            spot = float(hist.iloc[-1])
            rv = realized_vol(hist)
            exp = nearest_expiry(tk.options, TARGET_DTE)
            if not exp:
                print(f"{t}: no usable expiries"); continue
            e, dte = exp
            calls = tk.option_chain(e).calls
            K = spot * (1 + OTM)
            row = calls.iloc[(calls["strike"] - K).abs().argmin()]
            iv = float(row["impliedVolatility"])
            bid, ask = float(row["bid"]), float(row["ask"])
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(row["lastPrice"])
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else float("nan")
            ratio = iv / rv if rv > 0 else float("nan")
            rows.append(dict(t=t, iv=iv, rv=rv, ratio=ratio, spread_pct=spread_pct))
            print(f"{t:5s} spot {spot:9.2f}  {dte:2d}DTE  K {float(row['strike']):9.2f}  "
                  f"IV {iv*100:5.1f}%  RV {rv*100:5.1f}%  IV/RV {ratio:4.2f}  "
                  f"spread {spread_pct:5.1f}% (bid {bid:.2f}/ask {ask:.2f})")
        except Exception as ex:
            print(f"{t}: failed - {ex}")

    if rows:
        df = pd.DataFrame(rows)
        print("=" * 70)
        print(f"  Median IV/RV ratio (real entry-IV premium) : {df['ratio'].median():.2f}")
        print(f"  Mean   IV/RV ratio                         : {df['ratio'].mean():.2f}")
        print(f"  Median real round-trip spread              : {df['spread_pct'].median():.1f}%")
        print("  ----------------------------------------------------------------")
        print("  Backtest currently assumes: IV/RV = 1.10,  round-trip spread = 5.0%")
        print(f"  => suggested ENTRY_IV_PREMIUM for full_bot_backtest.py: "
              f"{df['ratio'].median():.2f}")


if __name__ == "__main__":
    run()
