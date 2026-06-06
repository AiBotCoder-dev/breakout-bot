#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_briefing.py — Sends a morning AI briefing to Telegram.
============================================================
Runs once per trading day (8:30 AM ET / 13:30 UTC) via GitHub Actions.

Fetches:
  - Latest market context (VIX, Fear & Greed, SPY regime, economic events)
  - Top 5 picks from the most recent breakout scan
  - Current portfolio status & risk gates

Sends a concise AI-written briefing to Telegram for the day ahead.

Required env vars:
  DATABASE_URL       — Supabase connection
  TELEGRAM_BOT_TOKEN — for sending the briefing
  TELEGRAM_CHAT_ID
  GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY — any one
"""

import os
import sys
import traceback
from datetime import datetime

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import trading_scanner as ts

# Reuse the PgAdapter + send_telegram from monitor.py
from monitor import PgAdapter, get_connection, send_telegram

try:
    from ai_engine import AIAnalyst
    _AI = AIAnalyst()
except Exception:
    _AI = None


def build_briefing_context(conn) -> dict:
    """Assemble everything the AI needs to write a useful briefing."""
    ctx: dict = {
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    try:
        ctx.update(ts.get_market_context_full() or {})
    except Exception:
        pass

    # Top scan picks
    try:
        row = conn.execute(
            "SELECT scan_id FROM calls ORDER BY scan_timestamp DESC LIMIT 1"
        ).fetchone()
        if row:
            rows = conn.execute(
                "SELECT ticker, explosive_score, breakout_prob, pattern_detected, "
                "entry_price, target_price, stop_loss "
                "FROM calls WHERE scan_id=? "
                "ORDER BY explosive_score DESC LIMIT 8",
                (row[0],)
            ).fetchall()
            ctx["top_picks"] = [{k: r.get(k) for k in r.keys()} for r in rows]
    except Exception:
        pass

    # Portfolio
    try:
        p = ts.PaperTradingEngine(conn)
        ctx["portfolio_summary"] = p.get_summary()
        ctx["open_positions"]    = p.open_positions[:6]
        ctx["risk_status"]       = ts.get_portfolio_risk_status(p)
    except Exception:
        pass

    # Options portfolio
    try:
        op = ts.OptionsPaperEngine(conn)
        ctx["open_options"] = op.get_positions("OPEN")[:6]
    except Exception:
        pass

    return ctx


def send_command_center_brief():
    """
    Send the one-glance 'what to buy today' command-center brief to Telegram.
    Runs FIRST and independently of the AI prose briefing — it only needs the DB
    + live market data, so it sends even when no AI key is configured.
    """
    print("  Building Command Center morning brief…")
    try:
        cc_conn = get_connection()
    except Exception as exc:
        print(f"  CC DB connection failed: {exc}")
        return
    try:
        from command_center import run_morning_scan
        try:
            from monitor import OPTIONS_AUTO_WATCHLIST as _WL
        except Exception:
            _WL = None
        cc = run_morning_scan(cc_conn, progress=lambda m: print(f"    …{m}"),
                              watchlist=_WL)

        m = cc.get("market", {})
        stocks = cc.get("top_stocks", [])
        opts = [o for o in cc.get("options", []) if str(o.get("decision")) == "BUY"]
        whale = cc.get("whale", [])
        goal = cc.get("goal", {})

        # Buy list (skip extended names)
        buy_lines = []
        for s in stocks[:5]:
            flag = " ⚠️ext" if s.get("extended") else ""
            buy_lines.append(f"  • <b>{s['ticker']}</b> ${s['price']:.2f} "
                             f"(6mo {s['mom_6m']*100:+.0f}%, stop ${s['stop']:.2f}){flag}")
        buy_block = "\n".join(buy_lines) if buy_lines else "  (none ranked)"

        opt_block = ""
        if opts:
            opt_block = "\n<b>🎰 Options (BUY-grade):</b>\n" + "\n".join(
                f"  • {o['ticker']} ${float(o['strike'] or 0):.0f}"
                f"{str(o['type'])[0].upper()} exp {o['expiry']} "
                f"${float(o['premium'] or 0):.2f} ({o['grade']})" for o in opts[:3])

        whale_block = ""
        if whale:
            whale_block = "\n<b>🐋 Smart money:</b> " + ", ".join(
                f"{w['ticker']}({w['score']})" for w in whale[:3])

        panic_block = ""
        if cc.get("panic"):
            panic_block = ("\n🚨 <b>PANIC SIGNAL FIRING</b> — highest-conviction buy. "
                           "Scale into SPY/momentum/calls.")

        goal_block = ""
        if goal:
            goal_block = (f"\n<b>🎯 Goal:</b> ${goal.get('real_value_now',0):,.0f} / "
                          f"${goal.get('goal_capital',0):,.0f} ({goal.get('status','?')})")

        msg = (
            f"🌅 <b>MORNING BRIEF — what to do today</b>\n"
            f"{datetime.utcnow().strftime('%a %d %b %Y')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>{cc.get('verdict','?')}</b>\n"
            f"{cc.get('action','')}\n"
            f"{panic_block}\n"
            f"\n<b>Market:</b> {m.get('bias','?')} bias · {m.get('regime','?')} · "
            f"{m.get('risk','?')} risk · VIX {m.get('vix',0):.1f}\n"
            f"\n<b>📈 Stocks to buy:</b>\n{buy_block}\n"
            f"{opt_block}"
            f"{whale_block}"
            f"{goal_block}"
        )
        if len(msg) > 4000:
            msg = msg[:3990] + "…"
        sent = send_telegram(msg)
        print(f"  Command Center brief → Telegram: {'✓ sent' if sent else '✗ failed'}")
    except Exception as exc:
        print(f"  WARN command center brief failed: {exc}")
        traceback.print_exc()
    finally:
        try: cc_conn.close()
        except Exception: pass


def main():
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"{'='*60}")
    print(f"  Daily AI Briefing  —  {now_utc}")
    print(f"{'='*60}")

    # ── FIRST: the actionable 'what to buy today' command-center brief ────────
    # Sent regardless of AI config (it needs only DB + market data).
    send_command_center_brief()

    if _AI is None or not _AI.available:
        print("  AI not configured (no GROQ_API_KEY / GEMINI_API_KEY / "
              "OPENROUTER_API_KEY) — skipping briefing.")
        send_telegram(
            "🤖 <b>Daily Briefing — AI not configured</b>\n"
            "Add a free API key as a GitHub secret to enable AI briefings:\n"
            "• GROQ_API_KEY (recommended)\n"
            "• GEMINI_API_KEY\n"
            "• OPENROUTER_API_KEY"
        )
        return

    print(f"  AI provider: {_AI.provider}")

    print("  Connecting to Supabase…")
    try:
        conn = get_connection()
    except Exception as exc:
        print(f"  ERROR: DB connection failed: {exc}")
        send_telegram(f"⚠️ <b>Daily Briefing DB Error</b>\n{exc}")
        sys.exit(1)

    print("  Building briefing context…")
    ctx = build_briefing_context(conn)
    top_picks = ctx.pop("top_picks", [])
    print(f"  Context: {len(top_picks)} top picks · "
          f"VIX={ctx.get('vix', {}).get('vix') if ctx.get('vix') else '?'} · "
          f"Regime={(ctx.get('regime') or {}).get('label', '?')}")

    print("  Generating briefing…")
    briefing = _AI.daily_briefing(ctx, top_picks=top_picks, max_tokens=600)

    # ── Send to Telegram ──────────────────────────────────────────────────────
    header = (
        f"🌅 <b>Morning Briefing</b> — {datetime.utcnow().strftime('%a %d %b %Y')}\n"
        f"VIX: {ctx.get('vix', {}).get('vix') if ctx.get('vix') else '—'}  ·  "
        f"Regime: {(ctx.get('regime') or {}).get('label', '—')}  ·  "
        f"F&G: {(ctx.get('fg') or {}).get('score', '—')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )
    msg = header + briefing
    # Telegram has a 4096 char limit
    if len(msg) > 4000:
        msg = msg[:3990] + "…"

    sent = send_telegram(msg)
    print(f"  Telegram: {'✓ sent' if sent else '✗ failed'}")
    print(f"  AI latency: {_AI.last_latency_ms} ms")

    # ── Daily Learning Engine improvement suggestion ──────────────────────────
    print()
    print("  Generating daily Learning Engine improvement suggestion…")
    try:
        paper      = ts.PaperTradingEngine(conn)
        learning   = ts.LearningEngine(conn)
        suggestion = learning.generate_improvement_suggestion(_AI, paper)

        if suggestion.get("suggestion"):
            from_cache = suggestion.get("from_cache", False)
            print(f"  {'(cached)' if from_cache else '(fresh)'} {suggestion['category']} — "
                  f"{suggestion['suggestion']}")

            # Build a clean Telegram payload — separate from the briefing
            sm = (
                f"💡 <b>Daily Learning Engine Suggestion</b>\n"
                f"{datetime.utcnow().strftime('%a %d %b %Y')}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🏷  <b>Category:</b> {suggestion.get('category', '—')}\n\n"
                f"📌 <b>Suggestion:</b>\n{suggestion.get('suggestion', '—')}\n\n"
                f"🧠 <b>Rationale:</b>\n{suggestion.get('rationale', '—')}\n\n"
                f"⚙️  <b>Action:</b>\n<code>{suggestion.get('action', '—')}</code>"
            )
            if len(sm) > 4000:
                sm = sm[:3990] + "…"
            sent_s = send_telegram(sm)
            print(f"  Suggestion → Telegram: {'✓ sent' if sent_s else '✗ failed'}")
        else:
            print("  Suggestion generator returned empty result — skipping.")
    except Exception as _sx:
        print(f"  WARN suggestion generation failed: {_sx}")
        traceback.print_exc()

    conn.close()
    print("  Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}")
        traceback.print_exc()
        try:
            send_telegram(f"💥 <b>Daily Briefing Error</b>\n{str(exc)[:400]}")
        except Exception:
            pass
        sys.exit(1)
