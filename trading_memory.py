#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trading_memory.py — Episodic Memory Agent for the breakout bot.
=================================================================
An AI-augmented memory system that learns from every closed trade:

  1. EPISODIC MEMORY  — stores full context of every opened trade
                        (master score, Wyckoff, sector, pattern, regime, VIX)
  2. REFLECTION       — AI generates a 2-3 sentence post-close note for
                        each trade ("what did we learn from this?")
  3. SIMILARITY       — finds the N most-similar past trades to any
                        candidate using categorical scoring (no embeddings)
  4. MEMORY CONFIDENCE — multiplier added to Master Score based on the
                        win rate of similar past trades
  5. CRITIC AGENT     — pre-trade AI sanity check that can soft-veto
                        catastrophically bad setups

DESIGN PRINCIPLES (per user's constraints):
  • Runs on cloud infrastructure already in place (Supabase + Groq + GitHub Actions)
  • Zero PC compute load
  • Single AI call per trade open + single per close (cheap on Groq quota)
  • Falls back to deterministic behaviour when AI is off
  • Idempotent — safe to re-run any cycle
  • Builds memory gradually — degrades gracefully with <3 similar past trades

This replaces nothing — it AUGMENTS the existing 12-multiplier Master Score
with a 13th 'memory' multiplier and an advisory critic step.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional


# ── Row access compatible with sqlite3.Row + PgAdapter._PgRow ─────────────────

def _row_get(row, key, default=None):
    if row is None:
        return default
    try:
        if hasattr(row, "get") and callable(row.get):
            return row.get(key, default)
    except Exception:
        pass
    try:
        return dict(row).get(key, default)
    except Exception:
        try:
            return row[key]
        except Exception:
            return default


# ══════════════════════════════════════════════════════════════════════════════
# TRADING MEMORY AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TradingMemoryAgent:
    """Persistent episodic memory + AI-powered critic for the trading bot."""

    _INIT_SQL = """
CREATE TABLE IF NOT EXISTS paper_trade_memory (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id            INTEGER UNIQUE,
    ticker              TEXT,
    entry_date          TIMESTAMP,
    exit_date           TIMESTAMP,
    status              TEXT DEFAULT 'OPEN',
    master_score        REAL,
    explosive_score     REAL,
    breakout_prob       REAL,
    wyckoff_phase       TEXT,
    sector              TEXT,
    pattern             TEXT,
    market_regime       TEXT,
    vix_level           REAL,
    vix_regime          TEXT,
    net_pnl             REAL,
    won                 INTEGER,
    days_held           INTEGER,
    exit_reason         TEXT,
    pnl_pct             REAL,
    reflection          TEXT,
    reflection_at       TIMESTAMP,
    context_text        TEXT,
    created_at          TIMESTAMP,
    updated_at          TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tmem_ticker  ON paper_trade_memory(ticker);
CREATE INDEX IF NOT EXISTS idx_tmem_status  ON paper_trade_memory(status);
CREATE INDEX IF NOT EXISTS idx_tmem_won     ON paper_trade_memory(won);

CREATE TABLE IF NOT EXISTS paper_critic_reviews (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    reviewed_at         TIMESTAMP,
    ticker              TEXT,
    master_score_pre    REAL,
    n_similar           INTEGER,
    similar_wins        INTEGER,
    similar_losses      INTEGER,
    similar_win_rate    REAL,
    verdict             TEXT,
    confidence          REAL,
    reasoning           TEXT,
    was_overridden      INTEGER DEFAULT 0,
    actual_outcome      TEXT
);

CREATE TABLE IF NOT EXISTS paper_memory_state (
    id                          INTEGER PRIMARY KEY,
    last_processed_open_id      INTEGER DEFAULT 0,
    last_processed_close_id     INTEGER DEFAULT 0,
    n_reflections_generated     INTEGER DEFAULT 0,
    n_critic_reviews            INTEGER DEFAULT 0,
    last_lifecycle_at           TIMESTAMP
);
"""

    # ── Similarity scoring weights (sum ≤ 100) ────────────────────────────────
    _SIM_WEIGHTS = {
        "same_sector":         20,
        "same_pattern":        20,
        "same_wyckoff":        15,
        "same_regime":         10,
        "same_vix_regime":     10,
        "score_within_10":     15,
        "prob_within_10":      10,
    }

    # ── Memory confidence multiplier table ────────────────────────────────────
    # Maps win-rate of similar past trades → multiplier
    _MULT_TIERS = [
        (0.80,  1.12),    # 80%+ similar trades won → 12% boost
        (0.65,  1.06),
        (0.55,  1.03),
        (0.45,  1.00),    # neutral
        (0.30,  0.92),
        (0.00,  0.85),    # <30% wins → 15% penalty
    ]

    # Critic veto threshold — only suggests SKIP when this many similar trades
    # AND their loss rate is overwhelming
    _CRITIC_VETO_MIN_SIMILAR   = 5
    _CRITIC_VETO_LOSS_RATE     = 0.80   # 80%+ losers in similar past

    def __init__(self, conn, ai_analyst=None):
        self.conn = conn
        self.ai   = ai_analyst
        try:
            self.conn.executescript(self._INIT_SQL)
        except Exception:
            pass
        # Seed state row if missing
        try:
            row = self.conn.execute(
                "SELECT id FROM paper_memory_state WHERE id=1"
            ).fetchone()
            if not row:
                self.conn.execute(
                    "INSERT INTO paper_memory_state (id) VALUES (1)"
                )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # ENTRY / CLOSE recording
    # ─────────────────────────────────────────────────────────────────────────

    def remember_at_entry(self, trade_id: int, context: dict) -> bool:
        """Called when a trade opens.  *context* should include as much as
        is known at entry: ticker, master_score, wyckoff_phase, sector,
        pattern, market_regime, vix_level, vix_regime, explosive_score,
        breakout_prob.

        Idempotent — re-calling for the same trade_id updates instead of
        duplicating.
        """
        try:
            ctx_text = self._build_context_text(context)
            row = self.conn.execute(
                "SELECT id FROM paper_trade_memory WHERE trade_id=?",
                (int(trade_id),)
            ).fetchone()
            if row:
                # Already have it
                return False

            self.conn.execute(
                "INSERT INTO paper_trade_memory "
                "(trade_id, ticker, entry_date, status, master_score, "
                "explosive_score, breakout_prob, wyckoff_phase, sector, "
                "pattern, market_regime, vix_level, vix_regime, "
                "context_text, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (int(trade_id),
                 str(context.get("ticker", "")).upper(),
                 str(context.get("entry_date", datetime.utcnow().isoformat())),
                 "OPEN",
                 float(context.get("master_score", 0) or 0),
                 float(context.get("explosive_score", 0) or 0),
                 float(context.get("breakout_prob", 0) or 0),
                 str(context.get("wyckoff_phase", "")),
                 str(context.get("sector", "")),
                 str(context.get("pattern", "")),
                 str(context.get("market_regime", "")),
                 float(context.get("vix_level", 0) or 0),
                 str(context.get("vix_regime", "")),
                 ctx_text,
                 datetime.utcnow().isoformat(),
                 datetime.utcnow().isoformat())
            )
            return True
        except Exception:
            return False

    def update_at_close(self, trade_id: int, outcome: dict) -> bool:
        """Called when a trade closes.  *outcome*: net_pnl, days_held,
        exit_reason, exit_date, pnl_pct."""
        try:
            pnl = float(outcome.get("net_pnl", 0) or 0)
            won = 1 if pnl > 0 else 0
            self.conn.execute(
                "UPDATE paper_trade_memory "
                "SET status='CLOSED', exit_date=?, net_pnl=?, won=?, "
                "days_held=?, exit_reason=?, pnl_pct=?, updated_at=? "
                "WHERE trade_id=?",
                (str(outcome.get("exit_date", datetime.utcnow().isoformat())),
                 pnl, won,
                 int(outcome.get("days_held", 0) or 0),
                 str(outcome.get("exit_reason", "")),
                 float(outcome.get("pnl_pct", 0) or 0),
                 datetime.utcnow().isoformat(),
                 int(trade_id))
            )
            return True
        except Exception:
            return False

    def _build_context_text(self, context: dict) -> str:
        """Pre-formatted text used for human display + future embedding lookup."""
        parts = []
        for key, label in [
            ("ticker",         "Ticker"),
            ("sector",         "Sector"),
            ("pattern",        "Pattern"),
            ("wyckoff_phase",  "Wyckoff"),
            ("market_regime",  "Regime"),
            ("vix_regime",     "VIX"),
            ("master_score",   "MasterScore"),
            ("explosive_score","ExpScore"),
            ("breakout_prob",  "BreakoutProb"),
        ]:
            v = context.get(key, "")
            if v:
                parts.append(f"{label}={v}")
        return "  ·  ".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # SIMILARITY SEARCH (categorical, no embeddings needed)
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar_trades(self, candidate: dict, limit: int = 5,
                              min_similarity: int = 20) -> list:
        """Return the *limit* most-similar past CLOSED trades.

        Similarity score 0-100 from categorical matches:
            same sector       +20
            same pattern      +20
            same Wyckoff      +15
            same regime       +10
            same VIX regime   +10
            score within 10pt +15
            prob within 10pt  +10
        """
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_trade_memory WHERE status='CLOSED' "
                "ORDER BY id DESC LIMIT 500"
            ).fetchall()
        except Exception:
            return []

        scored = []
        c_sector  = str(candidate.get("sector",        "")).strip()
        c_pattern = str(candidate.get("pattern",       "")).strip()
        c_wyck    = str(candidate.get("wyckoff_phase", "")).strip()
        c_regime  = str(candidate.get("market_regime", "")).strip()
        c_vix     = str(candidate.get("vix_regime",    "")).strip()
        c_score   = float(candidate.get("master_score", 0) or 0)
        c_prob    = float(candidate.get("breakout_prob", 0) or 0)

        for r in rows or []:
            try:
                d = dict(r)
            except Exception:
                continue
            sim = 0
            if c_sector and str(d.get("sector", "")).strip() == c_sector:
                sim += self._SIM_WEIGHTS["same_sector"]
            if c_pattern and str(d.get("pattern", "")).strip() == c_pattern:
                sim += self._SIM_WEIGHTS["same_pattern"]
            if c_wyck and str(d.get("wyckoff_phase", "")).strip() == c_wyck:
                sim += self._SIM_WEIGHTS["same_wyckoff"]
            if c_regime and str(d.get("market_regime", "")).strip() == c_regime:
                sim += self._SIM_WEIGHTS["same_regime"]
            if c_vix and str(d.get("vix_regime", "")).strip() == c_vix:
                sim += self._SIM_WEIGHTS["same_vix_regime"]
            ms = float(d.get("master_score", 0) or 0)
            bp = float(d.get("breakout_prob", 0) or 0)
            if c_score > 0 and ms > 0 and abs(c_score - ms) <= 10:
                sim += self._SIM_WEIGHTS["score_within_10"]
            if c_prob > 0 and bp > 0 and abs(c_prob - bp) <= 10:
                sim += self._SIM_WEIGHTS["prob_within_10"]

            if sim >= min_similarity:
                d["similarity"] = sim
                scored.append(d)

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:limit]

    # ─────────────────────────────────────────────────────────────────────────
    # MEMORY CONFIDENCE (multiplier for Master Score)
    # ─────────────────────────────────────────────────────────────────────────

    def memory_confidence(self, candidate: dict, limit: int = 8) -> dict:
        """Compute a multiplier based on win rate of N most-similar past trades.

        Returns dict:
            multiplier        — 0.85 - 1.12 (1.0 when neutral or no data)
            n_similar         — count of comparable past trades found
            wins              — count of winners among similar
            losses            — count of losers among similar
            win_rate          — float 0-1
            top_similar       — list of 3 most similar past trades
            reasoning         — plain-English summary
        """
        similar = self.find_similar_trades(candidate, limit=limit)

        if len(similar) < 3:
            return {
                "multiplier":   1.0,
                "n_similar":    len(similar),
                "wins":         sum(1 for s in similar if s.get("won")),
                "losses":       sum(1 for s in similar if not s.get("won")),
                "win_rate":     0.0,
                "top_similar":  similar[:3],
                "reasoning":    (f"Only {len(similar)} similar past trades — "
                                  "memory too sparse for confidence weighting"),
            }

        wins  = sum(1 for s in similar if s.get("won"))
        wr    = wins / len(similar)

        # Pick multiplier tier
        mult = 1.0
        for thresh, m in self._MULT_TIERS:
            if wr >= thresh:
                mult = m
                break

        return {
            "multiplier":   mult,
            "n_similar":    len(similar),
            "wins":         wins,
            "losses":       len(similar) - wins,
            "win_rate":     round(wr, 3),
            "top_similar":  similar[:3],
            "reasoning":    (f"{wins}/{len(similar)} similar past trades won "
                              f"({wr*100:.0f}%) → {mult:.2f}× confidence"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # AI REFLECTION (post-close)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_reflection(self, trade_id: int) -> str:
        """Use AI to write a 2-3 sentence post-mortem on a closed trade.
        Stored on the memory row.  Returns the reflection text (or '' if AI off)."""
        try:
            row = self.conn.execute(
                "SELECT * FROM paper_trade_memory WHERE trade_id=?",
                (int(trade_id),)
            ).fetchone()
            if not row:
                return ""
            d = dict(row)
        except Exception:
            return ""

        # Skip if already reflected
        if d.get("reflection"):
            return str(d["reflection"])

        if not self.ai or not getattr(self.ai, "available", False):
            return ""

        prompt = (
            "Reflect on this closed paper trade in 2-3 plain-English sentences. "
            "Focus on: the KEY reason it won or lost, and ONE actionable "
            "pattern the bot should learn from this for similar future setups.\n\n"
            "Trade details:\n"
            f"  Ticker:        {d.get('ticker', '?')}\n"
            f"  Sector:        {d.get('sector', '?')}\n"
            f"  Pattern:       {d.get('pattern', '?')}\n"
            f"  Wyckoff phase: {d.get('wyckoff_phase', '?')}\n"
            f"  Market regime: {d.get('market_regime', '?')}\n"
            f"  VIX regime:    {d.get('vix_regime', '?')}\n"
            f"  Master Score:  {d.get('master_score', 0):.0f}/100\n"
            f"  Result:        {'WIN' if d.get('won') else 'LOSS'}  "
            f"P&L: ${d.get('net_pnl', 0):+.2f}  ({d.get('pnl_pct', 0):+.1f}%)\n"
            f"  Days held:     {d.get('days_held', '?')}\n"
            f"  Exit reason:   {d.get('exit_reason', '?')}\n\n"
            "Respond with only the 2-3 sentences — no headers, no JSON."
        )

        try:
            reflection = self.ai.chat(prompt, max_tokens=200) or ""
            reflection = reflection.strip()
            if reflection:
                self.conn.execute(
                    "UPDATE paper_trade_memory "
                    "SET reflection=?, reflection_at=? WHERE trade_id=?",
                    (reflection[:1000], datetime.utcnow().isoformat(),
                     int(trade_id))
                )
                # Bump counter
                try:
                    self.conn.execute(
                        "UPDATE paper_memory_state SET n_reflections_generated = "
                        "COALESCE(n_reflections_generated, 0) + 1 WHERE id=1"
                    )
                except Exception:
                    pass
            return reflection
        except Exception:
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # CRITIC AGENT (pre-trade)
    # ─────────────────────────────────────────────────────────────────────────

    def critic_review(self, candidate: dict, master_result: dict) -> dict:
        """One-shot AI critic that reviews a candidate against similar past
        trades.  Can soft-VETO catastrophically bad setups.

        Returns dict:
            verdict       — 'BUY' / 'WATCH' / 'SKIP'
            confidence    — 0.0 - 1.0
            reasoning     — one-sentence explanation
            n_similar     — how many past trades were considered
            similar_win_rate — win rate of those past trades
            was_ai_call   — True if AI was queried, False if rule-based fallback
        """
        similar = self.find_similar_trades(candidate, limit=8)
        n_sim   = len(similar)
        wins    = sum(1 for s in similar if s.get("won"))
        wr      = wins / n_sim if n_sim else 0.5

        # ── Deterministic veto rule first (works without AI) ────────────────
        # Only skip when we have STRONG evidence: 5+ similar past trades
        # AND 80%+ of them lost
        if n_sim >= self._CRITIC_VETO_MIN_SIMILAR and (1 - wr) >= self._CRITIC_VETO_LOSS_RATE:
            result = {
                "verdict":          "SKIP",
                "confidence":       0.85,
                "reasoning":        f"{n_sim - wins}/{n_sim} similar past trades lost — strong avoidance signal",
                "n_similar":        n_sim,
                "similar_win_rate": round(wr, 3),
                "was_ai_call":      False,
            }
            self._log_critic_review(candidate, master_result, result, similar)
            return result

        # ── If AI is off, return advisory NEUTRAL ──────────────────────────
        if not self.ai or not getattr(self.ai, "available", False):
            verdict = "BUY" if wr >= 0.5 else "WATCH"
            result = {
                "verdict":          verdict,
                "confidence":       0.5,
                "reasoning":        (f"{wins}/{n_sim} similar past trades won "
                                      f"({wr*100:.0f}%)" if n_sim
                                      else "No memory data yet"),
                "n_similar":        n_sim,
                "similar_win_rate": round(wr, 3) if n_sim else None,
                "was_ai_call":      False,
            }
            self._log_critic_review(candidate, master_result, result, similar)
            return result

        # ── AI critic call ─────────────────────────────────────────────────
        similar_summary = ""
        for i, s in enumerate(similar[:5], 1):
            wn   = "Won " if s.get("won") else "Lost"
            pnl  = float(s.get("net_pnl", 0) or 0)
            held = int(s.get("days_held", 0) or 0)
            similar_summary += (
                f"  {i}. {s.get('ticker', '?')}  {wn}  "
                f"${pnl:+.2f}  {held}d  "
                f"(sim {s.get('similarity', 0)}/100, "
                f"{s.get('pattern','?')} / {s.get('sector','?')})\n"
            )

        prompt = (
            "You are the bot's critic.  The bot wants to enter this paper trade.  "
            "Cross-check against the bot's past similar trades.\n\n"
            "CANDIDATE:\n"
            f"  Ticker:        {candidate.get('ticker', '?')}\n"
            f"  Master Score:  {candidate.get('master_score', 0):.0f}/100  "
            f"({master_result.get('grade', '?')} → {master_result.get('decision', '?')})\n"
            f"  Sector:        {candidate.get('sector', '?')}\n"
            f"  Pattern:       {candidate.get('pattern', '?')}\n"
            f"  Wyckoff:       {candidate.get('wyckoff_phase', '?')}\n"
            f"  Regime:        {candidate.get('market_regime', '?')}\n\n"
            f"SIMILAR PAST TRADES ({n_sim} found):\n"
            f"{similar_summary}"
            f"  Overall: {wins} wins / {n_sim - wins} losses ({wr*100:.0f}% win rate)\n\n"
            "Output STRICT JSON only:\n"
            '  {"verdict": "BUY"|"WATCH"|"SKIP", "confidence": 0.0-1.0, '
            '"reasoning": "one sentence"}\n\n'
            "Rules:\n"
            "- BUY if past similar trades won >=55%\n"
            "- WATCH if mixed (40-54%)\n"
            "- SKIP if past similar trades lost >=60%\n"
            "- Be tough — your job is to catch bad trades the Master Score missed."
        )

        verdict, confidence, reasoning = "BUY", 0.5, ""
        try:
            resp = self.ai.chat(prompt, max_tokens=200) or ""
            # Extract JSON
            if "{" in resp and "}" in resp:
                resp = resp[resp.index("{"): resp.rindex("}") + 1]
                data = json.loads(resp)
                verdict    = str(data.get("verdict",    "BUY")).upper()
                confidence = float(data.get("confidence", 0.5))
                reasoning  = str(data.get("reasoning",   ""))[:300]
            if verdict not in ("BUY", "WATCH", "SKIP"):
                verdict = "BUY"
        except Exception:
            verdict = "BUY" if wr >= 0.5 else "WATCH"
            reasoning = f"AI critic failed — defaulting based on win rate {wr*100:.0f}%"

        result = {
            "verdict":          verdict,
            "confidence":       confidence,
            "reasoning":        reasoning,
            "n_similar":        n_sim,
            "similar_win_rate": round(wr, 3) if n_sim else None,
            "was_ai_call":      True,
        }
        self._log_critic_review(candidate, master_result, result, similar)
        return result

    def _log_critic_review(self, candidate: dict, master_result: dict,
                            result: dict, similar: list):
        try:
            wins = sum(1 for s in similar if s.get("won"))
            self.conn.execute(
                "INSERT INTO paper_critic_reviews "
                "(reviewed_at, ticker, master_score_pre, n_similar, "
                "similar_wins, similar_losses, similar_win_rate, verdict, "
                "confidence, reasoning) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(),
                 str(candidate.get("ticker", "")),
                 float(candidate.get("master_score", 0) or 0),
                 len(similar),
                 wins,
                 len(similar) - wins,
                 result.get("similar_win_rate"),
                 result.get("verdict", ""),
                 float(result.get("confidence", 0.5)),
                 str(result.get("reasoning", ""))[:500])
            )
            try:
                self.conn.execute(
                    "UPDATE paper_memory_state SET n_critic_reviews = "
                    "COALESCE(n_critic_reviews, 0) + 1 WHERE id=1"
                )
            except Exception:
                pass
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE PROCESSOR — idempotent, called every monitor cycle
    # ─────────────────────────────────────────────────────────────────────────

    def process_lifecycle_events(self, generate_reflections: bool = True,
                                    max_per_cycle: int = 20) -> dict:
        """Find any paper_portfolio rows that newly opened or closed since
        last cycle, and update memory accordingly.

        Idempotent — uses watermarks in paper_memory_state.
        """
        try:
            state_row = self.conn.execute(
                "SELECT * FROM paper_memory_state WHERE id=1"
            ).fetchone() or {}
            last_open  = int(_row_get(state_row, "last_processed_open_id",  0) or 0)
            last_close = int(_row_get(state_row, "last_processed_close_id", 0) or 0)
        except Exception:
            last_open  = 0
            last_close = 0

        n_opens_recorded   = 0
        n_closes_updated   = 0
        n_reflections      = 0
        max_open_id        = last_open
        max_close_id       = last_close

        # ── New opens ───────────────────────────────────────────────────────
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_portfolio "
                "WHERE id > ? ORDER BY id ASC LIMIT ?",
                (last_open, int(max_per_cycle))
            ).fetchall()
            for r in rows or []:
                try:
                    d = dict(r)
                except Exception:
                    continue
                tid = int(d.get("id") or 0)
                context = {
                    "ticker":          d.get("ticker", ""),
                    "entry_date":      d.get("entry_date"),
                    "explosive_score": d.get("explosive_score", 0),
                    "breakout_prob":   d.get("breakout_prob",   0),
                    "sector":          d.get("sector",  ""),
                    "pattern":         d.get("pattern", ""),
                    # These would come from monitor.py for richer context
                    # but the basics work without them
                    "master_score":    d.get("master_score") or 0,
                    "wyckoff_phase":   d.get("wyckoff_phase") or "",
                    "market_regime":   d.get("market_regime") or "",
                    "vix_regime":      d.get("vix_regime") or "",
                }
                if self.remember_at_entry(tid, context):
                    n_opens_recorded += 1
                if tid > max_open_id:
                    max_open_id = tid
        except Exception:
            pass

        # ── New closes (or transition from OPEN to CLOSED) ──────────────────
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_portfolio "
                "WHERE status='CLOSED' AND id > ? ORDER BY id ASC LIMIT ?",
                (last_close, int(max_per_cycle))
            ).fetchall()
            for r in rows or []:
                try:
                    d = dict(r)
                except Exception:
                    continue
                tid = int(d.get("id") or 0)
                outcome = {
                    "net_pnl":     d.get("net_pnl", 0),
                    "exit_date":   d.get("exit_date"),
                    "exit_reason": d.get("exit_reason", ""),
                    "pnl_pct":     d.get("net_pnl_pct", 0),
                    "days_held":   self._compute_days_held(d),
                }
                if self.update_at_close(tid, outcome):
                    n_closes_updated += 1
                # Generate reflection if AI is on
                if generate_reflections:
                    if self.generate_reflection(tid):
                        n_reflections += 1
                if tid > max_close_id:
                    max_close_id = tid
        except Exception:
            pass

        # ── Persist watermarks ──────────────────────────────────────────────
        try:
            self.conn.execute(
                "UPDATE paper_memory_state SET "
                "last_processed_open_id=?, last_processed_close_id=?, "
                "last_lifecycle_at=? WHERE id=1",
                (max_open_id, max_close_id, datetime.utcnow().isoformat())
            )
        except Exception:
            pass

        return {
            "opens_recorded":  n_opens_recorded,
            "closes_updated":  n_closes_updated,
            "reflections":     n_reflections,
            "open_watermark":  max_open_id,
            "close_watermark": max_close_id,
        }

    @staticmethod
    def _compute_days_held(d: dict) -> int:
        try:
            ed = d.get("entry_date")
            xd = d.get("exit_date")
            if not ed or not xd:
                return 0
            if isinstance(ed, str):
                ed_dt = datetime.strptime(ed[:10], "%Y-%m-%d")
            elif hasattr(ed, "year"):
                ed_dt = datetime(ed.year, ed.month, ed.day)
            else:
                return 0
            if isinstance(xd, str):
                xd_dt = datetime.strptime(xd[:10], "%Y-%m-%d")
            elif hasattr(xd, "year"):
                xd_dt = datetime(xd.year, xd.month, xd.day)
            else:
                return 0
            return max(0, (xd_dt - ed_dt).days)
        except Exception:
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # READ APIs (for UI / diagnostics)
    # ─────────────────────────────────────────────────────────────────────────

    def get_recent_reflections(self, limit: int = 15) -> list:
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_trade_memory "
                "WHERE reflection IS NOT NULL AND reflection != '' "
                "ORDER BY reflection_at DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
            return [dict(r) for r in rows or []]
        except Exception:
            return []

    def get_recent_critic_reviews(self, limit: int = 15) -> list:
        try:
            rows = self.conn.execute(
                "SELECT * FROM paper_critic_reviews "
                "ORDER BY id DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
            return [dict(r) for r in rows or []]
        except Exception:
            return []

    def get_memory_stats(self) -> dict:
        try:
            n_total = int(self.conn.execute(
                "SELECT COUNT(*) FROM paper_trade_memory"
            ).fetchone()[0] or 0)
            n_closed = int(self.conn.execute(
                "SELECT COUNT(*) FROM paper_trade_memory WHERE status='CLOSED'"
            ).fetchone()[0] or 0)
            n_refl = int(self.conn.execute(
                "SELECT COUNT(*) FROM paper_trade_memory "
                "WHERE reflection IS NOT NULL AND reflection != ''"
            ).fetchone()[0] or 0)
            n_critic = int(self.conn.execute(
                "SELECT COUNT(*) FROM paper_critic_reviews"
            ).fetchone()[0] or 0)
            n_critic_skips = int(self.conn.execute(
                "SELECT COUNT(*) FROM paper_critic_reviews WHERE verdict='SKIP'"
            ).fetchone()[0] or 0)
            return {
                "n_total_memories":  n_total,
                "n_closed":          n_closed,
                "n_open":            n_total - n_closed,
                "n_reflections":     n_refl,
                "n_critic_reviews":  n_critic,
                "n_critic_skips":    n_critic_skips,
            }
        except Exception:
            return {"n_total_memories": 0}
