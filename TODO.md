# 📋 TODO / Future Improvements

Tracked manually so they don't get lost between sessions.
See also: the **Self-Learning Engine's daily suggestions** for AI-generated improvement ideas based on real performance data.

---

## 🗞 News Tracker (high priority)

Add a per-ticker news feed that surfaces material events in real time and feeds into the Master Score as a new signal.

**Sources (all free):**
- Yahoo Finance RSS (`https://finance.yahoo.com/rss/headline?s=AAPL`)
- MarketWatch RSS
- Finnhub free tier (60 calls/min) — has structured news endpoints
- Benzinga RSS (high signal-to-noise on actionable news)

**What it should do:**
- Pull last 24 h of headlines per held ticker
- Run headline through LLM sentiment classifier (POSITIVE / NEGATIVE / NEUTRAL with confidence)
- Tag for category: earnings / analyst / SEC filing / acquisition / lawsuit / regulatory / product / macro
- Cache aggressively (15 min TTL) to avoid hammering RSS endpoints
- Surface in:
  - **Streamlit** — collapsible news panel per open position with sentiment badges
  - **Master Score** — new component worth ±5 points based on net 24h sentiment
  - **Telegram alerts** — push when a major negative headline lands on an open position (auto-exit suggestion)
  - **Daily briefing** — top 3 catalysts to watch today

**Tech notes:**
- Use `feedparser` (already in requirements.txt) for RSS parsing
- Use existing AI engine for sentiment classification (Groq free tier handles 14k/day)
- Store classified headlines in a `paper_news_events` table for retrospective analysis
- Detect news + price-move combination → potential trade signal

**Why this matters:**
- Most blowups come from news the bot didn't see (earnings miss, downgrade, lawsuit)
- Top funds have entire teams dedicated to news flow — even basic automation here is alpha

---

## Other ideas (not yet prioritized)

- [ ] Walk-forward backtesting harness (separate workflow on weekends)
- [ ] Position sizing based on per-ticker historical volatility, not just account fraction
- [ ] After-hours / pre-market gap scanner
- [ ] Cross-asset risk signals (TLT, GLD, DXY relative strength)
- [ ] Discord webhook in addition to Telegram
- [ ] Sector rotation visualization in app
- [ ] PDF tearsheet export of monthly performance
