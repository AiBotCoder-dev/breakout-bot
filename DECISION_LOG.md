# Decision Log — Leveraged Shadow → Real Money

A pre-registered record of decisions, so they're made once, in advance, on evidence —
and not re-litigated emotionally in the moment. Nothing here is a directive to trade;
the human operator makes every real-money decision themselves.

> **Not financial advice.** This is the operator's own systematic framework and a
> record of backtests run on it. Claude is not a licensed financial advisor. The
> decision to deploy real capital — whether, when, and how much — is entirely the
> operator's. Dollar figures below are the operator's chosen parameters, not a
> recommendation.

---

## 1. Finalized architecture (as of 2026-07-26) — DONE, stop adding

The design is frozen after a full reviewer-driven validation cycle. Further triggers
were tested and rejected on evidence (see §3). "Done" means: let it run, don't
re-architect.

| Layer | Vehicle | Rule | Status |
|-------|---------|------|--------|
| Core | CAD 2x sector rotation (HQU/HSU/HXU/HEU/HFU/HGU) | Weekly top-2 by 63d momentum, each held name must be > its own 50 & 200 SMA | ✅ live shadow |
| Master gate (airbag) | Cash | Force full cash if SPY OR XIU < 200-SMA; checked DAILY | ✅ live shadow |
| Overnight | QQQ close→open | Harvest only when core is LONG; overweight first 3 days after a re-entry | ✅ live module (overnight_edge.py) |
| Options sleeve | Convex events | VIX > 25 only | ✅ currently gated OFF |
| Ghost exits | 4 exit policies A/B/C vs control | Pre-registered adoption at 80 resolved trades, composite score + hard DQs | ✅ read-only (ghost_exits.decision) |

Vehicle = Wealthsimple-tradeable CAD-listed 2x ETFs (no 1.5% FX fee; 2x is the
leverage-sweep optimum). Every rebalance priced with the real Wealthsimple cost
(bid/ask spread; $0 commission; no FX on CAD).

---

## 2. The evidence behind the design (so the "why" is recorded)

| Finding | Number | Test |
|---------|--------|------|
| Selection alpha is real at the SECTOR level (not stock) | 1x rotation Sortino 1.06 vs QQQ 0.86 | aggregation_test.py |
| Real 3x ETFs lose ~half their theoretical leverage to decay | synth3x +4816% vs real 3x +2513% | aggregation_test.py |
| Leverage optimum is ~1.5–2x, not 3x (real-vehicle Calmar flat past it) | Sortino peaks 1.5x; real 3x Calmar 1.25 ≈ 1x 1.26 | leverage_tune.py |
| US 3x on Wealthsimple basic is unviable (1.5% FX each way) | Sortino 1.12 → 0.53 with FX | wealthsimple_backtest.py |
| CAD 2x rotation beats buy-hold on real fees | +2306% Sortino 1.17 vs HQU 0.73 | wealthsimple_backtest.py |
| Broad-index airbag halves drawdown | maxDD −50% → −23%, Calmar 1.20 → 2.05 | broad_gate_test.py |
| Trend filter survived 2022 (not whipsaw, not "smarter SPY") | 2022 −0.3% vs QQQ −32.6%; corr median 0.63 | regime_diagnostics.py |
| Overnight edge is trend-following, not fear-driven | uptrend +29%/yr, downtrend −12%/yr | overnight_regime_test.py |
| VIX circuit breaker adds nothing — REJECTED | 0 drawdown reduction, 4/5 false positives | vix_circuit_test.py |

Two-layer gate on the full window (per-ETF 50/200 + broad 200): **Calmar ~3.0** on a
2x book — the binding drawdown was Aug-2024 (yen carry unwind, VIX only 28), and −21%
is accepted as the **irreducible risk budget of 2x through a correlation spike**, not
a hole to plug. Any trigger that would catch it fires 20+ times in normal bulls.

---

## 3. Pre-registered REAL-MONEY graduation gate

The rule for flipping from paper shadow to real capital. Decided in advance; ALL must
pass. This is a floor for *considering* it — the operator still makes the call.

1. **≥ 60 calendar days** of live shadow performance recorded.
2. **Shadow max drawdown < −25%** over that window (wider than backtest −21% for variance).
3. **Shadow Sortino > 1.0** over that window.
4. **Execution-fidelity check** — the operator has paper-traded the weekly rebalance in
   their own Wealthsimple account for ≥ 4 consecutive weeks with < ~1% tracking error
   vs the shadow (proves the manual copy is actually executable on time).
5. **Operator confirms** they can commit to the weekly rebalance reliably.
6. **Start tiny:** deploy at most **$500** (half of target). Scale to the ~$1k target
   only after **≥ 30 days** of clean real execution.

### Honest limitation on this gate (recorded, not hidden)
Criteria 1–3 are **weak evidence on their own.** The strategy's entire risk profile
lives in the rare drawdown, and a calm 60-day window can pass every threshold simply
because no stress event occurred — telling you nothing about the −21% behavior. So:
- Passing the gate is a *permission to start small*, NOT proof the edge is real live.
- **The first real drawdown is the actual test.** Do not scale up until the book has
  survived at least one genuine −10%+ shadow drawdown and behaved as the backtest says
  (airbag fired, rotation rotated).
- Criterion 4 (execution fidelity) is doing more real work than 1–3: the biggest live
  risk at $1k is not the strategy, it's manual-execution slippage (missed rebalances,
  wrong sizing, late fills).

---

## 4. Open threads
- None on architecture (frozen).
- Overnight overlay is validated but not yet expressed in the shadow's paper P&L
  (currently a separate live module); optional future integration.
- Ghost-exit adoption rule armed but not yet triggered (0/80 resolved).
