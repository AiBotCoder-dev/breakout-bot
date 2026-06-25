# GO-LIVE SHEET — turning the validated fixes ON

All fixes are built, OFF by default, and additive. To deploy, set the GitHub
**Actions secrets** below (repo → **Settings → Secrets and variables → Actions →
New repository secret**). No code change or redeploy needed — the next monitor run
(every ~10 min during market hours) picks them up. To roll any of them back, delete
the secret (or set it to the default shown).

The evidence: 67 round-trips, **9% win rate, −$3,047 realized**. The journal pins the
loss to specific signal sources, and the structure backtest pins the rest to the
far-OTM wrapper. These three secrets target exactly that.

---

## Set these 3 secrets

| Secret | Value | What it does |
|---|---|---|
| `BLOCK_SIGNAL_SOURCES` | `pead,bottom_fisher` | Stops trading the two dead sources — **PEAD calls (0/8, −$307)** and **bottom-fisher (0/3)**. Counter-trend/event sources that contradict the momentum edge. Momentum combos still trade. |
| `MIN_TRADES_PER_DAY` | `0` | Turns off the **data-floor "explore" churn** (17% win, −$230). The bot stops force-filling low-quality tickets just to hit a daily count. |
| `OPTION_STRUCTURE` | `spread` | Routes each surviving CALL entry into the validated **ATM/+10% debit spread** (backtest NET: +7pp win rate, typical trade −32% → −4%, capped losers). |

That's it. The exit manager (the only profitable component — trailing stops banked
AMD +$235, INTC +$106) is untouched. Churn is already controlled by the existing
rebuy cooldown.

---

## Day-1 watch list (first session after you set them)
- **First spread fill:** confirm the `mleg` order fills cleanly and the Telegram
  alert reads "CALL DEBIT SPREAD" with a short leg + net debit. Watch one full
  open→close cycle to verify the exit manager closes **both legs together** (it
  buys back the short hedge; you should never be left holding a naked short).
- **Source block working:** the logs will print `⊘ <ticker> — sources [...] all
  blocked, skip` for PEAD/bottom-fisher setups.
- **No data-floor line:** you should no longer see `📊 Data floor: topping up …`.
- **Ledger:** `data/ledger_summary.json` updates after each close — check win rate
  and `by_source` trend over the next 1–2 weeks.

## Rollback (instant, per-fix)
- Naked calls again → delete `OPTION_STRUCTURE` (or set `naked`).
- Re-enable a source → remove it from `BLOCK_SIGNAL_SOURCES`.
- Restore the data floor → set `MIN_TRADES_PER_DAY` back to `3`.

## Realistic expectation
This should move the bot from "clearly losing" toward **breakeven-to-modestly-
positive** — not a windfall. The spread caps the typical loser; cutting PEAD/
bottom-fisher/explore removes the worst drag. Re-check the ledger after ~2 weeks /
20+ new trades, then run `python train_meta_model.py` (with `DATABASE_URL`) to see
if any feature now separates winners well enough to enforce the winner gate.

## Not yet wired (future code change, optional)
- Tighten the live momentum bar back to the validated `mom_6m ≥ 0.10` (the scan
  currently uses 0.05). Needs a small code change if you want it; flag it and it's
  a quick follow-up.
