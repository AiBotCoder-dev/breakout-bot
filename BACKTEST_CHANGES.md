# Backtest realism fix — `full_bot_backtest.py`

**Date:** 2026-06-16
**File changed:** `full_bot_backtest.py` (the canonical "replay the bot's CALL process" backtest)
**Goal:** Stop the options backtest from lying. Model real-world execution friction
(bid/ask spread + commissions) and IV crush, and report **frictionless (GROSS)**
vs **realistic (NET)** numbers side by side so the cost of reality is visible.

---

## 1. Why this was the #1 problem

The original backtest priced options with Black–Scholes and reported a win rate as
if you could trade at the theoretical mid-price with no costs and with implied
volatility frozen for the whole hold. That is not how options trade. Three silent
lies, each of which **inflates** the reported edge:

| # | Lie | Reality |
|---|-----|---------|
| 1 | Fills at the **mid-price** | You buy at the **ask**, sell at the **bid**. Short-dated equity options routinely cross a 3–5%+ spread. |
| 2 | **Zero commissions** | Brokers charge ~$0.65 per contract, **each side**. |
| 3 | **IV held flat** entry→exit | Implied vol mean-reverts and bleeds after a move ("IV crush"). A long call loses value when IV drops, *even if the stock rises*. |

Original pricing (the exact lines that were wrong):

```python
iv = max(0.15, min(1.5, rv[i]*1.1))           # one IV, used for BOTH legs
e  = bs_call(S0, K, 21/365, iv)               # entry  = mid
x  = bs_call(S1, K, 11/365, iv)               # exit   = mid, SAME iv
opt_ret = (x/e - 1) * 100 if e > 0.01 else 0  # no spread, no commission
```

---

## 2. Exactly what changed

### a) Added explicit, tunable friction + vol constants

```python
HALF_SPREAD_PCT      = 0.025    # half the bid/ask spread as a % of mid (~5% round trip)
MIN_HALF_SPREAD      = 0.02     # $/share floor for the half-spread (cheap opts trade wide)
COMMISSION_PER_SHARE = 0.0065   # $0.65 / contract / 100 shares, charged on EACH side
IV_CRUSH             = 0.10     # exit IV = entry IV x (1 - IV_CRUSH): vol mean-reversion
```

### b) Added fill helpers — you pay the ask, sell the bid, pay commission both ways

```python
def _half_spread(mid):                 # larger of a % of mid or a $ floor
    return max(mid * HALF_SPREAD_PCT, MIN_HALF_SPREAD)

def buy_fill(mid):                     # what you ACTUALLY pay to open
    return mid + _half_spread(mid) + COMMISSION_PER_SHARE

def sell_fill(mid):                    # what you ACTUALLY collect to close (never < 0)
    return max(0.0, mid - _half_spread(mid) - COMMISSION_PER_SHARE)
```

### c) Model entry and exit IV separately, and compute GROSS vs NET per trade

**Before:**
```python
iv = max(0.15, min(1.5, rv[i]*1.1))
e = bs_call(S0, K, 21/365, iv); x = bs_call(S1, K, 11/365, iv)
opt_ret = (x/e - 1) * 100 if e > 0.01 else 0.0
```

**After:**
```python
iv0 = max(0.15, min(1.5, rv[i]*1.1))     # entry IV (realized-vol proxy)
iv1 = max(0.05, iv0 * (1.0 - IV_CRUSH))  # exit IV after the crush
mid_e = bs_call(S0, K, 21/365, iv0)      # theoretical mid at entry
mid_x = bs_call(S1, K, 11/365, iv1)      # theoretical mid at exit

opt_gross = (mid_x/mid_e - 1) * 100 if mid_e > 0.01 else 0.0   # the OLD lie, kept for contrast
cost = buy_fill(mid_e); proceeds = sell_fill(mid_x)
opt_net   = (proceeds/cost - 1) * 100 if cost > 0.01 else 0.0  # what you'd really keep
```

### d) Report both numbers (per stock and aggregate), plus an explicit "cost of reality"

The aggregate now prints the frictionless win rate/return, the realistic win
rate/return, and the difference between them. Per-stock and per-example lines show
`NET (gross ...)` so you can eyeball where friction bites.

### e) Minor: replaced two Unicode separators (`—`, `·`) in `print()`s with ASCII
(`-`, `|`) so the output renders cleanly in the Windows console instead of `�`.

---

## 3. Measured impact (actual run, 5 stocks, 5y, 199 signals)

```
 AGGREGATE - 199 call signals across 5 stocks, 5y
  Directional accuracy (stock up in 10d)  : 60.8%
  --- frictionless (OLD, the optimistic lie) ---
  OPTION win rate  (mid-to-mid, flat IV)  : 39.7%
  Option mean / median return             : +45% / -30%
  --- realistic (NET of spread + comm + crush) ---
  OPTION win rate  (ask in / bid out)     : 39.2%
  Option mean / median return             : +37% / -34%
  >>> COST OF REALITY: 0.5 pts of win rate, -8% avg return <<<
```

### How to read this honestly

- **The headline edge is much weaker than "directional accuracy" suggests.** The
  underlying rises 60.8% of the time, but the near-money call only makes money
  **~39%** of the time — because it has to overcome the 3% OTM strike + theta. Most
  trades are small losses (median return is **negative**); the positive *average*
  is carried by a few big winners (e.g. AMD +539%). This is a fat right tail, not a
  reliable win rate.
- **Friction cost here was smaller than I first guessed** (~0.5 pts of win rate,
  ~8% of average return), and that's an honest, useful finding: for *long* options,
  the bid/ask spread and commissions are **not** the dominant cost — direction,
  theta, and IV are. A 5% round-trip spread only flips trades that sat right at
  breakeven, and the return distribution is bimodal, so few flip.
- **The bigger remaining lie is the synthetic *entry* IV** (`realized vol x 1.1`).
  In reality, when momentum is hot and everyone is buying calls, *implied* vol is
  elevated — you pay up at entry — and it collapses after the move. This change
  models the *exit* crush but still understates the *entry* IV. Fixing that
  properly needs **real historical option chains / an IV surface**, not Black–Scholes
  on realized vol. That's the next step.

---

## 4. What is STILL not modeled (do not over-trust this backtest yet)

- **Real historical option prices / IV surface** — entry IV is still a realized-vol
  proxy, and there's no volatility skew (OTM strikes trade at higher IV).
- **Partial fills / liquidity** — assumes you always get filled at one price.
- **Early exits / stops** — holds a fixed 10 days; the live bot exits dynamically.
- **Survivorship bias** — universe is 5 mega-cap survivors (NVDA/AAPL/TSLA/AMD/JPM)
  in a bull window. No delisted or crashed names.
- **Out-of-sample / walk-forward** — thresholds are still tuned on the same window
  they're tested on. (This was suggestion #2 in the review.)

---

## 5. How to run

```powershell
# deps (yfinance/numpy/pandas were NOT installed locally — install once):
python -m pip install yfinance numpy pandas
python full_bot_backtest.py
```

Tune `HALF_SPREAD_PCT`, `MIN_HALF_SPREAD`, `COMMISSION_PER_SHARE`, and `IV_CRUSH`
at the top of the file to match your own broker and the liquidity of the names you
actually trade. Wider spreads (less liquid tickers, retail fills) will widen the
gap between GROSS and NET.
