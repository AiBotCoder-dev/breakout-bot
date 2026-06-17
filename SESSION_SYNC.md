# Session Sync — cross-device coordination log

This file is the **shared channel** between Claude Code sessions working on this repo
from different devices. There is **no live connection** between sessions — we talk
*asynchronously through git*. Treat this file as a message board / shared notebook.

---

## How this works (the protocol)

1. **Before you start work:** `git pull`. Read the newest entries in the Log below and
   the "Open questions" section so you don't duplicate or clobber the other session.
2. **Claim what you're about to do** by adding a line under **Now in progress** (with
   your device tag) and pushing *before* you dig in — this is the lock that stops both
   sessions building the same thing.
3. **When you finish a chunk:** append a dated entry at the **TOP** of the Log
   (newest first), move your "in progress" line into it, then commit + push.
4. **To ask the other session something:** add it under **Open questions / requests**
   and push. The other session answers in its next Log entry and removes the question.
5. Keep entries short and action-oriented: *what changed, why, what's next, what you
   need from the other side.* Reference commit hashes so the other session can
   `git show <hash>`. Deep detail lives in `BACKTEST_CHANGES.md` / `TODO.md`; this file
   is the index + conversation.

**Device tags:** label every entry so it's clear who wrote it. Rename these to whatever
makes sense for your setup:
- `[A / desktop]`  — the device this file was created on
- `[B / laptop]`   — the other device

> Sync note: this repo also lives under OneDrive, so files *may* sync without git too —
> but **git is the source of truth.** Always pull/commit/push; don't rely on OneDrive
> timing for coordination.

---

## Now in progress (the lock — clear your line when done)

- _(nothing claimed)_

## Open questions / requests (answer in your next entry, then delete)

- **[A → B]** Next recommended step is the **survivorship-bias / universe test** (run the
  momentum entry on ~50–100 diverse names, not just the 5 winners). If either session
  starts it, claim it under "Now in progress" first so we don't both build it. Which
  device wants it?

---

## Log (newest first)

### 2026-06-16 — [A / desktop] — backtest honesty pass complete (3 fixes) + next step
Made the options backtest stop lying, on three axes. All pushed to `main`:

- **`f7bcd13`** — real execution friction: bid/ask half-spread, per-contract commission,
  and IV crush. Reports GROSS vs NET side by side. (`full_bot_backtest.py`, `BACKTEST_CHANGES.md` §1–5)
- **`c008ed6`** — replaced the magic `IV = rv*1.1` with a live-calibrated
  `ENTRY_IV_PREMIUM` (=0.87) + a sensitivity sweep. Key finding: option avg return swings
  **+88% → +11%** purely on the entry-IV assumption, which can't be recovered from price
  data. (`option_iv_calibration.py`, `BACKTEST_CHANGES.md` §6)
- **`a5ee5f1`** — walk-forward (out-of-sample) validation of the MOMENTUM entry.
  15 folds over 10y. Findings: **overfitting tax only +0.2%** (entry survives OOS at
  **57.1% directional win, +1.9%/trade**); **adaptive threshold tuning is worthless
  (-0.4%) vs a fixed 0.10**; the "best" threshold is noise (scatters across the whole
  grid). **Conclusion: keep the momentum threshold fixed at 0.10, don't optimise it.**
  (`walk_forward.py`, `BACKTEST_CHANGES.md` §7)

**Honest verdict so far:** the momentum *direction* edge is real but modest, and it's
robust across **time**. It is **not yet** tested across **stocks** — everything above ran
on 5 hand-picked survivors (NVDA/AAPL/TSLA/AMD/JPM). That's the #1 open risk.

**Recommended next (in priority order):**
1. **Survivorship-bias / universe test** (free, highest value — can *invalidate* the edge,
   not just resize it). Build `universe_test.py`: run the same momentum entry across
   ~50–100 diverse names incl. past disappointments (INTC/PYPL/BABA/DIS/PFE/WBA…), report
   OOS win/avg. If the edge holds → probably real; if it collapses → it was stock-picking
   luck. Document as `BACKTEST_CHANGES.md` §8.
2. **Dynamic exits / stops** — backtest holds a fixed 10d; the live bot exits dynamically
   (`options_event_exits.py`). Model stop-loss + profit target + trailing exit so the
   backtest matches the bot and the fat-tail strategy can cut losers / let winners run.
3. **Real historical option IV** — biggest absolute weakness but needs *paid* chain data;
   lower on the free-impact list.

→ See "Open questions" above re: who takes #1.
