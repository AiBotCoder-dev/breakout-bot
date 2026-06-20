"""
train_meta_model.py — learn the WINNER/LOSER classifier from the trade journal.

This is the third stage of the meta-labeling system (after the journal logger and
the rule-based winner_gate). Once `broker_trade_journal` has accumulated enough
CLOSED trades — each carrying its pre-trade feature vector and its realized outcome
— this script:

  1. Pulls the closed trades + features + win/loss label.
  2. Reports which features actually SEPARATE winners from losers (univariate:
     win rate above vs below the median of each feature). Useful even on small N.
  3. Validates the live rule-based gate in SHADOW: win rate of trades the gate
     flagged 'take' vs 'skip' (gate_passed column).
  4. If there are enough labelled trades, fits a logistic regression (numpy-only,
     no sklearn dependency) to output a learned P(win), and prints the standardized
     coefficients + in-sample accuracy/AUC. Those weights can later replace the
     hand-set winner_gate thresholds.

DATA SOURCE (read-only):
  • DATABASE_URL set  → reads the live Postgres journal (same DB the bot writes).
  • else: pass a sqlite file path as argv[1] for local testing.

Run:  python train_meta_model.py            (uses DATABASE_URL)
      python train_meta_model.py local.db   (sqlite file)
"""
from __future__ import annotations
import os
import sys
import math

FEATURES = ["mom_6m", "mom_3m", "rng_pos", "rv_at_entry", "reach",
            "dte_at_entry", "iv_at_entry", "otm_pct", "quality_score",
            "in_uptrend", "gate_score"]
MIN_TRADES = 40  # below this, only the univariate report runs (model needs data)


def _rows():
    """Return list of dict rows of CLOSED journal trades (read-only)."""
    cols = ("pnl_pct, gate_passed, " + ", ".join(FEATURES))
    sql = (f"SELECT {cols} FROM broker_trade_journal "
           f"WHERE status='CLOSED' AND pnl_pct IS NOT NULL")
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        import psycopg2
        import psycopg2.extras
        cx = psycopg2.connect(url, sslmode="require")
        cur = cx.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        cx.close()
        return rows
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        import sqlite3
        cx = sqlite3.connect(sys.argv[1]); cx.row_factory = sqlite3.Row
        rows = [dict(r) for r in cx.execute(sql).fetchall()]
        cx.close()
        return rows
    print("No data source. Set DATABASE_URL (live journal) or pass a sqlite path.")
    return None


def _f(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def univariate(rows):
    print(f"\n{'='*72}\n  WHICH FEATURES SEPARATE WINNERS FROM LOSERS  (n={len(rows)})\n{'='*72}")
    y = [1 if _f(r["pnl_pct"]) and _f(r["pnl_pct"]) > 0 else 0 for r in rows]
    base = sum(y) / len(y) * 100
    print(f"  overall win rate: {base:.0f}%  ({sum(y)}W / {len(y)-sum(y)}L)\n")
    print(f"  {'feature':<16}{'win% LOW half':>15}{'win% HIGH half':>16}{'spread':>9}")
    print("  " + "-" * 56)
    out = []
    for feat in FEATURES:
        vals = [(_f(r.get(feat)), yi) for r, yi in zip(rows, y) if _f(r.get(feat)) is not None]
        if len(vals) < 8:
            continue
        vals.sort(key=lambda x: x[0])
        mid = len(vals) // 2
        lo = vals[:mid]; hi = vals[mid:]
        wl = sum(v[1] for v in lo) / len(lo) * 100
        wh = sum(v[1] for v in hi) / len(hi) * 100
        out.append((abs(wh - wl), feat, wl, wh))
    for spread, feat, wl, wh in sorted(out, reverse=True):
        print(f"  {feat:<16}{wl:>14.0f}%{wh:>15.0f}%{wh-wl:>+8.0f}")
    if not out:
        print("  (not enough non-null feature values yet)")


def shadow_gate(rows):
    pairs = [(r.get("gate_passed"), _f(r["pnl_pct"])) for r in rows
             if r.get("gate_passed") is not None and _f(r["pnl_pct"]) is not None]
    if not pairs:
        print("\n  [shadow gate] no gate_passed labels yet — accumulating.")
        return
    took = [p for g, p in pairs if int(g) == 1]
    skip = [p for g, p in pairs if int(g) == 0]
    print(f"\n  SHADOW GATE on live trades ({len(pairs)} labelled):")
    if took:
        print(f"    gate TAKE : n={len(took):>4}  win {sum(1 for p in took if p>0)/len(took)*100:>4.0f}%  "
              f"avg {sum(took)/len(took):+.0f}%")
    if skip:
        print(f"    gate SKIP : n={len(skip):>4}  win {sum(1 for p in skip if p>0)/len(skip)*100:>4.0f}%  "
              f"avg {sum(skip)/len(skip):+.0f}%   (the trades it would have removed)")


def logistic(rows):
    import numpy as np
    data = []
    for r in rows:
        xs = [_f(r.get(f)) for f in FEATURES]
        if any(x is None for x in xs):
            continue
        y = 1.0 if _f(r["pnl_pct"]) > 0 else 0.0
        data.append((xs, y))
    if len(data) < MIN_TRADES:
        print(f"\n  [model] {len(data)} fully-featured closed trades; need >= {MIN_TRADES} "
              f"to fit a stable model. Univariate report above is the read for now.")
        return
    X = np.array([d[0] for d in data], float)
    y = np.array([d[1] for d in data], float)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
    w = np.zeros(Xs.shape[1])
    for _ in range(4000):
        p = 1 / (1 + np.exp(-Xs @ w))
        w -= 0.1 * (Xs.T @ (p - y)) / len(y)
    p = 1 / (1 + np.exp(-Xs @ w))
    acc = ((p > 0.5) == y).mean() * 100
    order = np.argsort(-np.abs(w[1:]))
    pos = np.sum((p[y == 1][:, None] > p[y == 0][None, :])) if (y == 1).any() and (y == 0).any() else 0
    auc = pos / ((y == 1).sum() * (y == 0).sum()) if (y == 1).any() and (y == 0).any() else float("nan")
    print(f"\n{'='*72}\n  LEARNED META-MODEL (logistic, n={len(data)})\n{'='*72}")
    print(f"  in-sample accuracy {acc:.0f}%   AUC {auc:.2f}   (bias {w[0]:+.2f})")
    print(f"  standardized coefficients (|larger| = stronger separator):")
    for i in order:
        print(f"    {FEATURES[i]:<16}{w[i+1]:>+7.2f}")
    print("  (positive coef -> higher feature raises P(win). These weights can later")
    print("   replace the hand-set winner_gate thresholds once AUC is convincingly >0.5.)")


def main():
    rows = _rows()
    if rows is None:
        return
    if not rows:
        print("Journal has no CLOSED trades yet — let the bot run, then re-run this.")
        return
    univariate(rows)
    shadow_gate(rows)
    logistic(rows)


if __name__ == "__main__":
    main()
