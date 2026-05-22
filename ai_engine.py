#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_engine.py — Free AI assistant for the breakout bot.
========================================================
A single ``AIAnalyst`` class that auto-detects which free LLM provider is
configured and routes requests to it.  Designed so the entire app keeps
working even when no AI key is set — `ai.available` is just ``False`` and
every call returns a polite "AI not configured" message instead of
crashing.

Providers (set any ONE in env vars / Streamlit secrets):
  GROQ_API_KEY       — Groq (Llama 3.3 70B, very fast, 14 400 req/day free)
  GEMINI_API_KEY     — Google Gemini 1.5 Flash (high quality, 15 RPM free)
  OPENROUTER_API_KEY — OpenRouter (access to many free models)

Order of preference: GROQ → GEMINI → OPENROUTER.

How to get a key (all FREE, no credit card):
  Groq      : https://console.groq.com/keys  (recommended)
  Gemini    : https://aistudio.google.com/app/apikey
  OpenRouter: https://openrouter.ai/keys

Streamlit Cloud: paste the key under your app's Settings → Secrets:
    GROQ_API_KEY = "gsk_xxx..."

GitHub Actions: add the key as a repository Secret with the same name.

Public API
----------
    ai = AIAnalyst()
    ai.available                 # bool — True if any provider is configured
    ai.provider                  # e.g. 'groq' / 'gemini' / 'openrouter'
    ai.chat(prompt, context={})  # general Q&A
    ai.explain_trade(trade)      # 2-sentence rationale for a single trade
    ai.analyze_position(pos)     # risk read on an open position
    ai.daily_briefing(ctx)       # morning market commentary
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

try:
    import requests
except ImportError:                                                     # pragma: no cover
    requests = None  # type: ignore


# ── System prompt — the AI's persona ──────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are an expert trading analyst embedded inside a paper-trading bot "
    "for short-term breakout setups and 0DTE–7DTE options strategies. "
    "You think like a market maker and institutional desk: Wyckoff phases, "
    "gamma exposure, VWAP, multi-timeframe confirmation, and disciplined "
    "risk management. Keep replies concise (≤ 150 words unless asked for "
    "detail). Always cite specific numbers from the data provided. Never "
    "fabricate prices or signals — if a number is missing, say so. End "
    "trade recommendations with a clear BUY / WATCH / SKIP verdict. "
    "Remember: this is paper trading only — no real money at stake."
)


class AIAnalyst:
    """Provider-agnostic AI analyst with auto-failover."""

    # ── Model defaults per provider — picks the fastest free option ──────────
    _GROQ_MODEL       = "llama-3.3-70b-versatile"
    _GEMINI_MODEL     = "gemini-1.5-flash"
    _OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

    def __init__(self, prefer: Optional[str] = None):
        self.last_error: Optional[str] = None
        self.last_latency_ms: int = 0

        # Resolve provider
        gk = os.environ.get("GROQ_API_KEY", "").strip()
        gem = os.environ.get("GEMINI_API_KEY", "").strip()
        orx = os.environ.get("OPENROUTER_API_KEY", "").strip()

        order = ["groq", "gemini", "openrouter"]
        if prefer in order:
            order.remove(prefer)
            order.insert(0, prefer)

        self.provider: Optional[str] = None
        self.api_key: Optional[str]  = None
        for p in order:
            if p == "groq" and gk:
                self.provider, self.api_key = "groq", gk
                break
            if p == "gemini" and gem:
                self.provider, self.api_key = "gemini", gem
                break
            if p == "openrouter" and orx:
                self.provider, self.api_key = "openrouter", orx
                break

        self.available: bool = bool(self.provider and self.api_key and requests is not None)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def chat(self, prompt: str, context: Optional[dict] = None,
             max_tokens: int = 600) -> str:
        """Send a free-form question, optionally with structured context."""
        if not self.available:
            return ("🤖 AI not configured.  Set GROQ_API_KEY, GEMINI_API_KEY, or "
                    "OPENROUTER_API_KEY in your environment / Streamlit secrets. "
                    "See ai_engine.py docstring for free signup links.")

        ctx_str = ""
        if context:
            try:
                ctx_str = "\n\nContext (JSON):\n" + json.dumps(
                    context, indent=2, default=str)[:6000]
            except Exception:
                ctx_str = f"\n\nContext: {str(context)[:2000]}"

        return self._call(prompt + ctx_str, max_tokens=max_tokens)

    def explain_trade(self, trade: dict, max_tokens: int = 200) -> str:
        """Generate a short 2-3 sentence rationale for one trade entry."""
        if not self.available:
            return ""

        prompt = (
            "Explain in 2-3 sentences why this paper trade was auto-entered. "
            "Focus on the strongest signal. End with one specific risk to monitor.\n"
            f"\nTrade JSON:\n{json.dumps(trade, indent=2, default=str)[:3000]}"
        )
        return self._call(prompt, max_tokens=max_tokens)

    def analyze_position(self, position: dict, market_ctx: Optional[dict] = None,
                         max_tokens: int = 300) -> str:
        """Read on a single open position — should we hold, take profit, cut?"""
        if not self.available:
            return ""

        prompt = (
            "Analyse this open paper-trading position and give a one-paragraph "
            "verdict on whether to HOLD, TAKE PROFIT, or CUT LOSS. Reference the "
            "current P&L, days to expiry (if options), and current market regime.\n"
            f"\nPosition:\n{json.dumps(position, indent=2, default=str)[:2000]}"
        )
        if market_ctx:
            prompt += f"\n\nMarket context:\n{json.dumps(market_ctx, indent=2, default=str)[:2000]}"
        return self._call(prompt, max_tokens=max_tokens)

    def daily_briefing(self, market_ctx: dict, top_picks: Optional[list] = None,
                       max_tokens: int = 500) -> str:
        """Morning market commentary tailored to the bot's strategy."""
        if not self.available:
            return ""

        prompt = (
            "Write a concise morning briefing (8-12 short bullet points) for a "
            "short-term breakout / 0DTE-7DTE options trader. Cover: market "
            "regime, VIX level, top setup of the day, top risks, and one "
            "actionable trade idea.\n"
            f"\nMarket context:\n{json.dumps(market_ctx, indent=2, default=str)[:3000]}"
        )
        if top_picks:
            prompt += f"\n\nTop scan picks today:\n{json.dumps(top_picks[:8], indent=2, default=str)[:2500]}"
        return self._call(prompt, max_tokens=max_tokens)

    # ─────────────────────────────────────────────────────────────────────────
    # PROVIDER ROUTERS
    # ─────────────────────────────────────────────────────────────────────────

    def _call(self, prompt: str, max_tokens: int = 600) -> str:
        if not self.available:
            return ""
        t0 = time.time()
        try:
            if self.provider == "groq":
                out = self._call_groq(prompt, max_tokens)
            elif self.provider == "gemini":
                out = self._call_gemini(prompt, max_tokens)
            elif self.provider == "openrouter":
                out = self._call_openrouter(prompt, max_tokens)
            else:
                out = ""
            self.last_latency_ms = int((time.time() - t0) * 1000)
            self.last_error = None
            return out.strip()
        except Exception as exc:
            self.last_error = f"{self.provider}: {exc}"
            self.last_latency_ms = int((time.time() - t0) * 1000)
            return f"⚠️ AI call failed ({self.provider}): {str(exc)[:200]}"

    def _call_groq(self, prompt: str, max_tokens: int) -> str:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":     self._GROQ_MODEL,
                "messages":  [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens":  max_tokens,
                "temperature": 0.3,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str, max_tokens: int) -> str:
        # Gemini bakes the system prompt into the user message
        full = f"{_SYSTEM_PROMPT}\n\n---\n\n{prompt}"
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._GEMINI_MODEL}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": full}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": max_tokens,
                },
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Gemini response: {str(data)[:300]}")

    def _call_openrouter(self, prompt: str, max_tokens: int) -> str:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/AiBotCoder-dev/breakout-bot",
                "X-Title":       "Breakout Bot",
            },
            json={
                "model":     self._OPENROUTER_MODEL,
                "messages":  [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens":  max_tokens,
                "temperature": 0.3,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]
