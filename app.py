"""
Trading Scanner — Streamlit Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import sqlite3
import sys, os, io, argparse, contextlib, re as _re
from datetime import datetime, timedelta
import numpy as np

# ── AI engine (optional — works without an API key, falls back silently) ──────
# Streamlit Cloud secrets are accessible via st.secrets, NOT as os.environ vars.
# Copy them across so ai_engine.py (which reads from os.environ) finds them.
# Also walks nested TOML sections in case the key was put inside [database] etc.
def _hoist_ai_secrets():
    try:
        import streamlit as _st_for_secrets
        _wanted = {"GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY",
                   "ALPACA_API_KEY", "ALPACA_API_SECRET",
                   # Broker (paper trading) secrets — needed so the Live Broker
                   # dashboard can read them (broker.py reads os.environ).
                   "ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET", "BROKER_MODE",
                   "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}

        def _walk(obj, depth=0):
            if depth > 3:
                return
            try:
                for _k in obj:
                    try:
                        _v = obj[_k]
                    except Exception:
                        continue
                    # Top-level match
                    if _k in _wanted and isinstance(_v, str) and _v.strip():
                        if _k not in os.environ:
                            os.environ[_k] = _v.strip()
                    # Recurse into nested sections (dict-like)
                    elif hasattr(_v, "__iter__") and not isinstance(_v, (str, bytes)):
                        _walk(_v, depth + 1)
            except Exception:
                pass

        try:
            _walk(_st_for_secrets.secrets)
        except Exception:
            pass
    except Exception:
        pass

_hoist_ai_secrets()

try:
    from ai_engine import AIAnalyst
    _AI = AIAnalyst()
except Exception:
    _AI = None


# ── PostgreSQL adapter (makes psycopg2 look like sqlite3) ─────────────────────
class _PgRow:
    """sqlite3.Row-compatible row backed by psycopg2 results."""
    __slots__ = ("_cols", "_vals")

    def __init__(self, cols, vals):
        self._cols = list(cols)
        self._vals = list(vals)

    def __getitem__(self, k):
        if isinstance(k, int):
            return self._vals[k]
        try:
            return self._vals[self._cols.index(k)]
        except ValueError:
            raise KeyError(k)

    def keys(self):
        return self._cols

    def get(self, k, default=None):
        try:
            return self._vals[self._cols.index(k)]
        except (ValueError, IndexError):
            return default

    def __iter__(self):
        return iter(self._vals)


class _PgCursor:
    """sqlite3 cursor wrapper for psycopg2."""

    def __init__(self, cur):
        self._cur = cur

    def _cols(self):
        return [d[0] for d in self._cur.description] if self._cur.description else []

    def fetchone(self):
        row = self._cur.fetchone()
        if row is None:
            return None
        cols = self._cols()
        return _PgRow(cols, row)

    def fetchall(self):
        cols = self._cols()
        return [_PgRow(cols, r) for r in self._cur.fetchall()]

    def __iter__(self):
        cols = self._cols()
        for row in self._cur:
            yield _PgRow(cols, row)


class _FakeCursor:
    """Returns a fixed scalar — used to fake last_insert_rowid()."""

    def __init__(self, value):
        self._v = value

    def fetchone(self):
        return (self._v,)

    def fetchall(self):
        return [(self._v,)]


class PgAdapter:
    """
    Wraps a psycopg2 connection to present a sqlite3-compatible API.
    Handles: ?→%s, AUTOINCREMENT→BIGSERIAL, INSERT OR REPLACE→ON CONFLICT,
             executescript(), last_insert_rowid(), DATETIME→TIMESTAMP.
    """

    _AUTOINCREMENT    = _re.compile(r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b", _re.I)
    _INSERT_OR_REPLACE = _re.compile(r"\bINSERT\s+OR\s+REPLACE\b", _re.I)
    _LAST_ROWID       = _re.compile(r"SELECT\s+last_insert_rowid\(\)", _re.I)
    _INTO_TABLE       = _re.compile(r"\bINTO\s+(calls|portfolio)\b", _re.I)

    def __init__(self, pg_conn):
        self._conn    = pg_conn
        self._last_id = None
        # autocommit=True — each statement is its own transaction; a failed
        # statement never poisons the connection for subsequent ones.
        try:
            self._conn.autocommit = True
        except Exception:
            pass

    def _adapt(self, sql: str) -> str:
        sql = sql.replace("?", "%s")
        sql = self._AUTOINCREMENT.sub("BIGSERIAL PRIMARY KEY", sql)
        sql = sql.replace("DATETIME", "TIMESTAMP")
        return sql

    def execute(self, sql: str, params=()):
        if self._LAST_ROWID.search(sql.strip()):
            return _FakeCursor(self._last_id)

        # psycopg2 blocks EVERY command — including cursor() — when the
        # connection is in a failed-transaction state.  Since we commit()
        # after every successful execute(), rolling back here only ever
        # clears a previously failed statement, never valid in-progress work.
        try:
            self._conn.rollback()
        except Exception:
            pass

        is_replace = bool(self._INSERT_OR_REPLACE.search(sql))
        adapted    = self._adapt(sql)

        if is_replace:
            adapted = self._INSERT_OR_REPLACE.sub("INSERT", adapted)
            adapted = adapted.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"

        is_insert      = adapted.strip().upper().startswith("INSERT")
        add_returning  = (
            is_insert
            and not is_replace
            and "RETURNING" not in adapted.upper()
            and bool(self._INTO_TABLE.search(adapted))
        )
        if add_returning:
            adapted = adapted.rstrip().rstrip(";") + " RETURNING id"

        cur = self._conn.cursor()
        try:
            cur.execute(adapted, params or ())
        except Exception:
            # Roll back the failed transaction so the connection stays usable
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

        if add_returning:
            row           = cur.fetchone()
            self._last_id = row[0] if row else None

        self.commit()
        return _PgCursor(cur)

    def executescript(self, sql: str):
        adapted = self._adapt(sql)
        for stmt in _re.split(r";[ \t]*\n?", adapted):
            stmt = stmt.strip()
            if not stmt or stmt.startswith("--"):
                continue
            # Rollback any prior failed state, then get a fresh cursor per statement
            try:
                self._conn.rollback()
            except Exception:
                pass
            try:
                cur = self._conn.cursor()
                cur.execute(stmt)
                self.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass

    def commit(self):
        try:
            if not getattr(self._conn, "autocommit", False):
                self._conn.commit()
        except Exception:
            pass

    def close(self):
        self._conn.close()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Scanner Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import scanner ────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
import trading_scanner as ts

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  .stApp { background-color: #0d1117; }
  section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 6px; color: #8b949e; font-size: 0.85rem; }
  .stTabs [aria-selected="true"] { background: #1c2333 !important; color: #e6edf3 !important; }

  /* ── KPI cards ── */
  .kpi-card {
    background: #1c2333; border: 1px solid #30363d;
    border-radius: 10px; padding: 18px 20px; text-align: center;
  }
  .kpi-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.07em; }
  .kpi-value { font-size: 1.9rem; font-weight: 700; margin: 4px 0 0; }
  .kpi-green  { color: #3fb950; } .kpi-red { color: #f85149; }
  .kpi-yellow { color: #e3b341; } .kpi-blue { color: #58a6ff; }

  /* ── Section headers ── */
  .section-hdr {
    font-size: 0.72rem; font-weight: 700; color: #8b949e;
    letter-spacing: 0.09em; text-transform: uppercase;
    border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-bottom: 14px;
  }

  /* ── Badges ── */
  .badge { display:inline-block; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
  .badge-green  { background:#1a4428; color:#3fb950; }
  .badge-red    { background:#4c1b1b; color:#f85149; }
  .badge-yellow { background:#4a3b00; color:#e3b341; }
  .badge-blue   { background:#0d2d6b; color:#58a6ff; }

  /* ── Stock cards ── */
  .stock-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 16px 18px; margin-bottom: 4px; position: relative; overflow: hidden;
    transition: border-color 0.15s;
  }
  .stock-card:hover { border-color: #58a6ff; }
  .stock-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
  .card-high::before   { background: linear-gradient(90deg,#3fb950,#26a641); }
  .card-medium::before { background: linear-gradient(90deg,#e3b341,#d4a017); }
  .card-low::before    { background: linear-gradient(90deg,#f85149,#da3633); }

  .card-ticker { font-size:1.35rem; font-weight:800; color:#e6edf3; letter-spacing:-0.02em; }
  .card-price  { font-size:1.05rem; color:#8b949e; margin-left:8px; }
  .card-pattern { font-size:0.82rem; color:#58a6ff; margin-top:5px; font-weight:600; }
  .card-explosive { font-size:0.76rem; color:#e3b341; margin-top:3px; }
  .card-divider { border:none; border-top:1px solid #21262d; margin:10px 0; }

  .trade-grid {
    display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:6px; margin:6px 0 8px;
  }
  .tg-label { font-size:0.62rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:2px; }
  .tg-val   { font-size:0.92rem; font-weight:700; color:#e6edf3; }
  .tg-green { color:#3fb950 !important; }
  .tg-red   { color:#f85149 !important; }
  .tg-blue  { color:#58a6ff !important; }

  .card-move     { font-size:0.76rem; color:#3fb950; margin-top:4px; }
  .card-catalyst { font-size:0.76rem; color:#e3b341; margin-top:5px; }
  .card-risk     { font-size:0.74rem; color:#f85149; margin-top:3px; }

  /* ── Market context bar ── */
  .ctx-bar {
    display:flex; gap:12px; background:#161b22; border:1px solid #30363d;
    border-radius:10px; padding:12px 18px; margin-bottom:16px; flex-wrap:wrap;
  }
  .ctx-item { text-align:center; min-width:80px; }
  .ctx-label { font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.06em; }
  .ctx-val   { font-size:1.1rem; font-weight:700; color:#e6edf3; }

  /* ── Page header ── */
  .page-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .page-title { font-size:1.6rem; font-weight:800; color:#e6edf3; margin:0; }
  .page-sub   { font-size:0.82rem; color:#8b949e; margin-top:4px; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer { visibility: hidden; }
  [data-testid="stToolbar"] { display:none; }

  /* ── Keep the sidebar REOPEN button always visible + high-contrast ──
     On the dark theme the collapsed-sidebar reopen control renders as a
     near-invisible dark icon, so it looks like the sidebar can't be reopened.
     Streamlit 1.5x names this control "stExpandSidebarButton" (aria-label
     "expandSidebar"); older versions used "collapsedControl". Target both, plus
     the header that contains it, and style it as a clearly-clickable chip. */
  [data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"],
  [aria-label="expandSidebar"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    background: #1c2333 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
  }
  [data-testid="stExpandSidebarButton"] svg,
  [data-testid="stExpandSidebarButton"] *,
  [data-testid="stSidebarCollapsedControl"] svg,
  [data-testid="collapsedControl"] svg,
  [aria-label="expandSidebar"] svg,
  [aria-label="expandSidebar"] * {
    color: #e6edf3 !important;
    fill: #e6edf3 !important;
  }
  /* The collapsed reopen button lives in the header — make sure the header
     stays rendered (we only hide the top-right toolbar, not the whole header). */
  [data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Bulletproof sidebar-reopen button (JS, version-independent) ───────────────
# CSS alone proved unreliable (Streamlit's collapsed-control testid changes
# between versions and the dark theme hides the icon). This injects a floating
# "☰" button into the PARENT document that is always visible when the sidebar
# is collapsed; clicking it programmatically clicks Streamlit's real expand
# button (a JS .click() works even if that button is visually hidden). A polling
# loop keeps our button alive across Streamlit reruns.
st.components.v1.html(
    """
<script>
(function () {
  const doc = window.parent && window.parent.document ? window.parent.document : document;
  const BTN_ID = "ccd-sidebar-reopen";

  function findExpandButton() {
    const sels = [
      '[data-testid="stExpandSidebarButton"]',
      '[data-testid="stSidebarCollapsedControl"] button',
      '[data-testid="stSidebarCollapsedControl"]',
      '[data-testid="collapsedControl"] button',
      '[data-testid="collapsedControl"]',
      '[aria-label="expandSidebar"]',
      'button[title="Expand sidebar"]'
    ];
    for (const s of sels) { const el = doc.querySelector(s); if (el) return el; }
    return null;
  }

  function sidebarIsCollapsed() {
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb) return true;
    const w = sb.getBoundingClientRect().width;
    if (w < 40) return true;
    const ariaExp = sb.getAttribute('aria-expanded');
    if (ariaExp === 'false') return true;
    return false;
  }

  function ensureButton() {
    let b = doc.getElementById(BTN_ID);
    if (!b) {
      b = doc.createElement('button');
      b.id = BTN_ID;
      b.textContent = '\\u2630 Menu';
      b.title = 'Open sidebar';
      Object.assign(b.style, {
        position: 'fixed', top: '10px', left: '10px', zIndex: '2147483647',
        background: '#1c2333', color: '#e6edf3', border: '1px solid #30363d',
        borderRadius: '8px', padding: '7px 13px', fontSize: '15px',
        fontWeight: '600', cursor: 'pointer', lineHeight: '1',
        boxShadow: '0 2px 8px rgba(0,0,0,0.4)'
      });
      b.addEventListener('click', function () {
        const exp = findExpandButton();
        if (exp) { exp.click(); }
      });
      doc.body.appendChild(b);
    }
    return b;
  }

  function tick() {
    try {
      const b = ensureButton();
      b.style.display = sidebarIsCollapsed() ? 'inline-block' : 'none';
    } catch (e) { /* ignore */ }
  }

  setInterval(tick, 400);
  tick();
})();
</script>
    """,
    height=0,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi(label: str, value: str, colour: str = "blue"):
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value kpi-{colour}">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def badge(text: str, colour: str = "blue"):
    return f'<span class="badge badge-{colour}">{text}</span>'


@st.cache_data(ttl=30)  # cache live prices for 30 seconds — fresh-ish + cheap
def _fetch_live_prices(tickers: tuple, _salt: int = 0) -> dict:
    """Return {ticker: last_price} for all tickers.

    Uses the unified data_providers layer:
      1. Alpaca Markets   (official live IEX feed, no rate limiting)
      2. yfinance          (fallback, can be rate-limited on cloud)

    Alpaca is dramatically more reliable on Streamlit Cloud than yfinance
    — set ALPACA_API_KEY + ALPACA_API_SECRET in Streamlit secrets to enable.
    Sign up free: https://alpaca.markets/
    """
    _ = _salt   # intentionally unused — purely a cache-key differentiator
    if not tickers:
        return {}

    # ── Try the unified data layer first (Alpaca → yfinance) ─────────────────
    try:
        from data_providers import get_live_prices as _dp_get
        prices_dp = _dp_get(tickers, allow_yfinance_fallback=False)
        if prices_dp and len(prices_dp) == len(tickers):
            return prices_dp
    except Exception:
        prices_dp = {}

    # If Alpaca got some but not all, fill in the rest via yfinance below
    prices: dict = dict(prices_dp)
    remaining = set(t.upper() for t in tickers) - set(prices.keys())
    if not remaining:
        return prices

    def _extract(raw, target_tickers):
        """Pull last Close per ticker from a yfinance dataframe."""
        out = {}
        if raw is None or raw.empty:
            return out
        if isinstance(raw.columns, pd.MultiIndex):
            _price_labels = {"Open", "High", "Low", "Close", "Volume"}
            lvl = 0 if _price_labels & set(raw.columns.get_level_values(0)) else 1
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(lvl)
            if isinstance(raw.columns, pd.MultiIndex):
                try:
                    raw = raw.xs("Close", axis=1, level=0)
                except Exception:
                    return out
        if isinstance(raw, pd.DataFrame):
            if "Close" in raw.columns:
                s = raw["Close"].dropna()
                if not s.empty and len(target_tickers) == 1:
                    out[next(iter(target_tickers))] = float(s.iloc[-1])
            else:
                for tk in target_tickers:
                    if tk in raw.columns:
                        s = raw[tk].dropna()
                        if not s.empty:
                            out[tk] = float(s.iloc[-1])
        elif isinstance(raw, pd.Series):
            s = raw.dropna()
            if not s.empty and target_tickers:
                out[next(iter(target_tickers))] = float(s.iloc[-1])
        return out

    # ── Strategy 1: 5-min bars over 2 days ────────────────────────────────────
    try:
        raw = yf.download(list(remaining), period="2d", interval="5m",
                          progress=False, auto_adjust=True, threads=True)
        prices.update(_extract(raw, remaining))
        remaining -= set(prices.keys())
    except Exception:
        pass

    # ── Strategy 2: daily bars over 5 days (for tickers still missing) ────────
    if remaining:
        try:
            raw = yf.download(list(remaining), period="5d", interval="1d",
                              progress=False, auto_adjust=True, threads=True)
            prices.update(_extract(raw, remaining))
            remaining -= set(prices.keys())
        except Exception:
            pass

    # ── Strategy 3: per-ticker fast_info ──────────────────────────────────────
    for tk in list(remaining):
        try:
            v = getattr(yf.Ticker(tk).fast_info, "last_price", None)
            if v and float(v) > 0:
                prices[tk] = float(v)
                remaining.discard(tk)
        except Exception:
            pass

    # ── Strategy 4: per-ticker 1d download (slowest, last resort) ────────────
    for tk in list(remaining):
        try:
            raw = yf.download(tk, period="5d", interval="1d",
                              progress=False, auto_adjust=True)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                if "Close" in raw.columns:
                    s = raw["Close"].dropna()
                    if not s.empty:
                        prices[tk] = float(s.iloc[-1])
                        remaining.discard(tk)
        except Exception:
            pass

    return prices

def _render_stock_card(r: dict):
    ticker   = r.get("ticker", "")
    price    = r.get("price", 0) or 0
    prob     = r.get("probability", 0) or 0
    pattern  = r.get("pattern", "No Pattern") or "No Pattern"
    score    = r.get("explosive_score", 0) or 0
    grade    = r.get("explosive_grade", "") or ""
    stop     = r.get("stop_price") or 0
    target   = r.get("tgt_price") or 0
    t1       = r.get("t1_price") or 0
    rr       = r.get("rr", 0) or 0
    move_lo  = r.get("move_low", 0) or 0
    move_hi  = r.get("move_high", 0) or 0
    catalyst = r.get("top_flag", "") or ""
    earn     = r.get("earnings_risk", "No") or "No"
    flt      = r.get("expl_float", 0) or 0
    pct52    = (r.get("pct_52w", 0) or 0) * 100
    timing   = r.get("timing", {}) or {}
    min_d    = timing.get("min_days", "")
    max_d    = timing.get("max_days", "")
    avwap    = r.get("above_avwap")

    if prob >= 75:
        card_cls, p_bg, p_col, conf = "card-high",   "#1a4428", "#3fb950", "HIGH CONF"
    elif prob >= 60:
        card_cls, p_bg, p_col, conf = "card-medium", "#4a3b00", "#e3b341", "MEDIUM"
    else:
        card_cls, p_bg, p_col, conf = "card-low",    "#4c1b1b", "#f85149", "SPECULATIVE"

    expl_html = (f'<div class="card-explosive">⚡ Explosive Score {score:.0f}/100'
                 f'{"  ·  Grade " + grade if grade else ""}</div>') if score >= 40 else ""

    fee_adj_rr     = r.get("fee_adj_rr", 0) or 0
    fee_adj_reward = r.get("fee_adj_reward_pct", 0) or 0
    tf_pass        = r.get("trade_filter_pass", True)  # default True for legacy results
    raw_rr_val     = r.get("raw_rr", rr) or rr

    stop_str   = f"${stop:.2f}"   if stop   else "—"
    tgt_str    = f"${target:.2f}" if target else "—"
    t1_str     = f"${t1:.2f}"    if t1     else "—"
    rr_str     = f"{rr:.1f}:1"   if rr     else "—"

    move_html = ""
    if move_lo or move_hi:
        dur = f"  ·  {min_d}–{max_d} day hold" if min_d and max_d else ""
        move_html = f'<div class="card-move">↑ Estimated move: +{move_lo:.0f}% – {move_hi:.0f}%{dur}</div>'

    cat_html = f'<div class="card-catalyst">⚡ Catalyst: {catalyst}</div>' if catalyst else ""

    # Fee-adjusted R/R block
    if fee_adj_rr or fee_adj_reward:
        tf_color  = "#3fb950" if tf_pass else "#f85149"
        tf_label  = "✅ PASS" if tf_pass else "❌ FAIL"
        fee_html  = (
            f'<div style="margin-top:8px; padding:7px 10px; background:#0d1117; '
            f'border-radius:6px; font-size:0.74rem;">'
            f'<span style="color:#8b949e;">Fee-Adj R/R: </span>'
            f'<span style="color:#58a6ff; font-weight:700;">{fee_adj_rr:.1f}:1</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:#8b949e;">Reward: </span>'
            f'<span style="color:#58a6ff; font-weight:700;">+{fee_adj_reward:.1f}%</span>'
            f'&nbsp;&nbsp;&nbsp;'
            f'<span style="background:{"#1a4428" if tf_pass else "#4c1b1b"}; '
            f'color:{tf_color}; padding:2px 8px; border-radius:10px; '
            f'font-weight:700;">{tf_label}</span>'
            f'</div>'
        )
    else:
        fee_html = ""

    risks = []
    if earn and earn != "No":
        risks.append(f"⚠ Earnings risk: {earn}")
    if flt and flt < 5_000_000:
        risks.append("⚠ Very low float — high volatility possible")
    if pct52 > 90:
        risks.append("⚠ Extended near 52-week high — late entry risk")
    if avwap is False:
        risks.append("⚠ Trading below AVWAP — bearish bias")
    risk_html = "".join(f'<div class="card-risk">{rk}</div>' for rk in risks)

    st.markdown(f"""
    <div class="stock-card {card_cls}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
        <div>
          <span class="card-ticker">{ticker}</span>
          <span class="card-price">${price:.2f}</span>
        </div>
        <div style="background:{p_bg};color:{p_col};padding:4px 12px;border-radius:20px;
                    font-size:0.72rem;font-weight:700;white-space:nowrap;flex-shrink:0;">
          {prob}% &nbsp;{conf}
        </div>
      </div>
      <div class="card-pattern">{pattern}</div>
      {expl_html}
      <hr class="card-divider">
      <div class="tg-label" style="margin-bottom:6px;">Trade Plan</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:6px;margin:6px 0 8px;">
        <div><div class="tg-label">Entry</div>
             <div class="tg-val">${price:.2f}</div></div>
        <div><div class="tg-label">Stop</div>
             <div class="tg-val tg-red">{stop_str}</div></div>
        <div><div class="tg-label">T1 (50%)</div>
             <div class="tg-val tg-blue">{t1_str}</div></div>
        <div><div class="tg-label">Target</div>
             <div class="tg-val tg-green">{tgt_str}</div></div>
        <div><div class="tg-label">R : R</div>
             <div class="tg-val tg-blue">{rr_str}</div></div>
      </div>
      {move_html}
      {cat_html}
      {fee_html}
      {risk_html}
    </div>
    """, unsafe_allow_html=True)

    if st.button("📈 View Chart", key=f"card_{ticker}", use_container_width=True):
        st.session_state.selected = ticker
        st.rerun()

def _render_market_clock():
    """Live ET market clock with session quality and countdown — updates every second via JS."""
    session = ts.MarketClock.get_session()
    q       = session["quality"]
    color   = session["color"]
    name    = session["name"]
    advice  = ts.MarketClock.get_advice()
    rem     = session.get("remaining_seconds", 0)
    nxt     = session.get("next_prime_seconds", 0)

    # Choose what the countdown tracks
    counting_down_to = ("Session ends" if q in ("PRIME", "SECONDARY", "CAUTION")
                        else "Next prime")

    _BADGE = {
        "PRIME":     ("#1a4428", "#3fb950"),
        "SECONDARY": ("#0d2d6b", "#58a6ff"),
        "CAUTION":   ("#4a3b00", "#e3b341"),
        "AVOID":     ("#4c1b1b", "#f85149"),
        "PREMARKET": ("#2a1957", "#6e40c9"),
        "CLOSED":    ("#21262d", "#8b949e"),
    }
    bg, fg = _BADGE.get(q, ("#21262d", "#8b949e"))
    # Server-side end-time in ms so the JS countdown is accurate immediately
    countdown_secs = rem if q in ("PRIME", "SECONDARY", "CAUTION", "AVOID", "PREMARKET") else nxt

    html = f"""<!DOCTYPE html>
<html><head><style>
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:0; background:#0d1117;
         font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
  .card {{background:#161b22;border:1px solid {color};border-radius:10px;padding:13px 15px;}}
  .lbl  {{font-size:9px;color:#8b949e;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px;}}
  .time {{font-size:26px;font-weight:800;color:#e6edf3;font-family:monospace;letter-spacing:.03em;}}
  .date {{font-size:10px;color:#8b949e;margin-top:1px;margin-bottom:8px;}}
  .badge{{display:inline-block;background:{bg};color:{fg};padding:3px 11px;
          border-radius:10px;font-size:10px;font-weight:700;margin-bottom:7px;}}
  .adv  {{font-size:10px;color:#8b949e;line-height:1.45;margin-bottom:8px;}}
  .cd-hdr{{font-size:9px;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;}}
  .cd-row{{display:flex;gap:6px;}}
  .cd-box{{flex:1;background:#0d1117;border-radius:6px;padding:5px 0;text-align:center;}}
  .cd-val{{font-size:18px;font-weight:700;color:#e6edf3;font-family:monospace;}}
  .cd-lbl{{font-size:8px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;}}
</style></head><body>
<div class="card">
  <div class="lbl">Market Clock (Eastern Time)</div>
  <div class="time" id="ct">--:--:-- --</div>
  <div class="date" id="cd">Loading…</div>
  <div class="badge">{q} &nbsp;·&nbsp; {name}</div>
  <div class="adv">{advice}</div>
  <div class="cd-hdr">{counting_down_to}</div>
  <div class="cd-row">
    <div class="cd-box"><div class="cd-val" id="ch">--</div><div class="cd-lbl">Hours</div></div>
    <div class="cd-box"><div class="cd-val" id="cm">--</div><div class="cd-lbl">Mins</div></div>
    <div class="cd-box"><div class="cd-val" id="cs">--</div><div class="cd-lbl">Secs</div></div>
  </div>
</div>
<script>
(function(){{
  var target = Date.now() + {countdown_secs} * 1000;
  function pad(n){{ return String(n).padStart(2,'0'); }}
  function tick(){{
    var now = new Date();
    // Live ET clock
    document.getElementById('ct').innerHTML =
      now.toLocaleTimeString('en-US',{{timeZone:'America/New_York',hour:'numeric',
        minute:'2-digit',second:'2-digit',hour12:true}});
    document.getElementById('cd').innerHTML =
      now.toLocaleDateString('en-US',{{timeZone:'America/New_York',
        weekday:'short',month:'short',day:'numeric',year:'numeric'}});
    // Countdown
    var rem = Math.max(0, Math.floor((target - Date.now())/1000));
    var h = Math.floor(rem/3600), r = rem%3600, m = Math.floor(r/60), s = r%60;
    document.getElementById('ch').innerHTML = pad(h);
    document.getElementById('cm').innerHTML = pad(m);
    document.getElementById('cs').innerHTML = pad(s);
  }}
  setInterval(tick, 1000); tick();
}})();
</script></body></html>"""

    st.html(html)


def _render_regime_banner(regime: dict):
    """Horizontal banner showing current market regime."""
    if not regime or regime.get("regime") == "UNKNOWN":
        return
    color = regime["color"]
    r20   = regime.get("ret_20d", 0)
    vol   = regime.get("hist_vol", 0)
    a200  = "Above" if regime.get("above_200") else "Below"
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid {color};border-radius:8px;
                padding:9px 16px;margin-bottom:14px;display:flex;
                justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
      <div>
        <span style="color:{color};font-weight:700;font-size:0.88rem;">
          📊 {regime['label']} Market Regime</span>
        <span style="color:#8b949e;font-size:0.77rem;margin-left:10px;">
          SPY {r20:+.1f}% (20d) &nbsp;·&nbsp; {a200} 200d SMA &nbsp;·&nbsp; Vol {vol:.0f}%</span>
      </div>
      <span style="color:#8b949e;font-size:0.76rem;">{regime['advice']}</span>
    </div>
    """, unsafe_allow_html=True)


def _is_streamlit_cloud() -> bool:
    return (
        os.environ.get("STREAMLIT_SHARING_MODE") == "1"
        or os.environ.get("HOME", "").startswith("/home/appuser")
        or os.path.exists("/mount/src")
    )

def _supabase_url() -> str:
    try:
        return st.secrets["database"]["url"]
    except Exception:
        return ""

@st.cache_resource
def get_db():
    db_url   = _supabase_url()
    on_cloud = _is_streamlit_cloud()

    if db_url:
        import psycopg2
        pg  = psycopg2.connect(db_url, sslmode="require")
        conn = PgAdapter(pg)
        mode = "postgres"
    elif on_cloud:
        raw = sqlite3.connect(":memory:", check_same_thread=False)
        raw.row_factory = sqlite3.Row
        conn = raw
        mode = "memory"
    else:
        raw = sqlite3.connect(
            os.path.join(_DIR, "trading_scanner_history.db"),
            check_same_thread=False,
        )
        raw.row_factory = sqlite3.Row
        conn = raw
        mode = "local"

    logger = ts.CallLogger.__new__(ts.CallLogger)
    logger.conn = conn
    logger._init_schema()
    return conn, mode

def _df(query: str, conn, params=()):
    try:
        rows = conn.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception:
        return pd.DataFrame()


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("results", []),
    ("market_ctx", {}),
    ("selected", None),
    ("scan_ran", False),
    ("regime", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Scan configuration
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Trading Scanner")
    st.caption("Configure and launch your scan below.")

    # ── Live Market Clock ──────────────────────────────────────────────────────
    _render_market_clock()
    st.divider()

    # Shared DB connection for the sidebar — same cached instance used elsewhere
    try:
        conn, _sb_db_mode = get_db()
    except Exception as _db_exc:
        conn = None
        _sb_db_mode = f"error: {_db_exc}"

    # ══════════════════════════════════════════════════════════════════════════
    # 🔧 SIDEBAR STATUS (full diagnostics live in the 🎛 Management tab)
    # ══════════════════════════════════════════════════════════════════════════
    try:
        _sb_pe   = ts.PaperTradingEngine(conn)
        _sb_risk = ts.get_portfolio_risk_status(_sb_pe)
        _sb_le   = ts.LearningEngine(conn)
        _sb_adj  = _sb_le.get_active_adjustments()
        _sb_pun  = _sb_le.get_punishment_status()
        _sb_lvl  = _sb_risk.get("risk_level", "NORMAL")
        _icon_map = {"NORMAL": "🟢", "CAUTION": "🟡", "HALT": "🛑", "EMERGENCY_HALT": "🚨"}
        _sb_icon = _icon_map.get(_sb_lvl, "⚪")
        _pun_tag = "  ·  🛑 PUNISH" if _sb_pun.get("active") else ""
        _line = (
            f"{_sb_icon} <b>{_sb_lvl}</b>  ·  "
            f"DD {_sb_risk.get('drawdown_pct', 0):.1f}%  ·  "
            f"Learn #{_sb_adj.get('learning_iteration', 0)}"
            f"{_pun_tag}"
        )
        st.markdown(
            "<div style='background:#161b22;border-radius:6px;"
            "padding:8px 12px;margin-bottom:6px;font-size:0.85em'>"
            + _line + "</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass
    st.caption("💡 Full diagnostics live in the **🎛 Management** tab.")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # 🤖 AI ANALYST CHAT
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🤖 AI Analyst", expanded=False):
        if _AI is None or not _AI.available:
            st.info(
                "**AI not configured.** Add a free API key in Streamlit "
                "Settings → Secrets to enable:\n\n"
                "- `GROQ_API_KEY` (recommended — fastest, free 14 k/day):  "
                "https://console.groq.com/keys\n"
                "- `GEMINI_API_KEY` (Google, free 1 M tokens/day):  "
                "https://aistudio.google.com/app/apikey\n"
                "- `OPENROUTER_API_KEY`:  https://openrouter.ai/keys",
                icon="🔑",
            )

            # ── Debug: show what we actually detect ─────────────────────────
            with st.expander("🔧 Debug — why isn't this working?", expanded=True):
                _dbg = {}
                for _k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
                    _v_env = os.environ.get(_k, "")
                    _v_sec = ""
                    try:
                        if _k in st.secrets:
                            _v_sec = str(st.secrets[_k])
                    except Exception as _e:
                        _v_sec = f"<err: {type(_e).__name__}>"
                    _dbg[_k] = {
                        "in os.environ": bool(_v_env),
                        "env preview":   (_v_env[:6] + "…") if _v_env else "—",
                        "in st.secrets": bool(_v_sec) and "<err" not in _v_sec,
                        "secrets preview": (_v_sec[:6] + "…") if _v_sec and "<err" not in _v_sec else _v_sec or "—",
                    }
                st.json(_dbg)

                # ── Show ALL top-level keys in st.secrets ───────────────────
                st.markdown("**All top-level keys Streamlit currently sees in `st.secrets`:**")
                try:
                    _all_keys = []
                    for _k in st.secrets:
                        _v = st.secrets[_k]
                        if isinstance(_v, dict):
                            _all_keys.append(f"`[{_k}]` (section) — sub-keys: "
                                             f"{list(_v.keys())}")
                        else:
                            _preview = str(_v)[:6] + "…" if _v else "(empty)"
                            _all_keys.append(f"`{_k}` = {_preview}")
                    if _all_keys:
                        for _line in _all_keys:
                            st.markdown(f"- {_line}")
                    else:
                        st.warning("st.secrets is **empty** — no secrets are saved!")
                except Exception as _se:
                    st.error(f"Could not read st.secrets at all: {_se}")

                st.caption(
                    "**If GROQ_API_KEY is NOT in the list above** → "
                    "1) Open dashboard → ⚙ Settings → Secrets, "
                    "2) Confirm the key is there, "
                    "3) Click **Save** (this is the step usually missed), "
                    "4) Click ⋮ → **Reboot app** at the top of the dashboard."
                )
        else:
            st.caption(f"⚡ Powered by **{_AI.provider.upper()}**")

            # Init chat history
            if "ai_chat_history" not in st.session_state:
                st.session_state.ai_chat_history = []

            # Display previous messages
            for msg in st.session_state.ai_chat_history[-10:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Quick-action buttons
            _qa1, _qa2 = st.columns(2)
            _qbtn = None
            if _qa1.button("📊 Portfolio Risk", key="ai_qa_risk",
                           use_container_width=True):
                _qbtn = "What's the biggest risk in my current paper portfolio right now?"
            if _qa2.button("🌅 Morning Brief", key="ai_qa_brief",
                           use_container_width=True):
                _qbtn = "Give me a short market briefing for today's trading."
            _qa3, _qa4 = st.columns(2)
            if _qa3.button("🎯 Top Setup", key="ai_qa_setup",
                           use_container_width=True):
                _qbtn = "What's the best high-conviction setup from my latest scan?"
            if _qa4.button("🧠 Coach Me", key="ai_qa_coach",
                           use_container_width=True):
                _qbtn = ("Review my closed paper trades and give me 3 specific "
                         "improvements I can make.")

            # Chat input
            _ai_input = st.chat_input("Ask the AI…", key="ai_chat_input")
            _user_msg = _ai_input or _qbtn

            if _user_msg:
                # ── Build context that gives AI awareness ─────────────────────
                _ai_ctx: dict = {}
                try:
                    _ai_ctx["vix"] = ts.get_vix_level()
                except Exception:
                    pass
                # SPY market regime (computed fresh; previously referenced an
                # undefined `_ctx` so this silently never populated)
                try:
                    import yfinance as _yf_reg
                    _spy = _yf_reg.download("SPY", period="1y", interval="1d",
                                            progress=False, auto_adjust=True)
                    if _spy is not None and not _spy.empty:
                        if isinstance(_spy.columns, pd.MultiIndex):
                            _spy.columns = _spy.columns.get_level_values(0)
                        _reg = ts.MarketRegimeDetector.detect(_spy)
                        _ai_ctx["regime"] = _reg.get("regime") or _reg.get("label")
                except Exception:
                    pass
                # Recent scan picks
                try:
                    rows = conn.execute(
                        "SELECT ticker, explosive_score, breakout_prob, "
                        "pattern_detected, entry_price, target_price, stop_loss "
                        "FROM calls ORDER BY scan_timestamp DESC LIMIT 5"
                    ).fetchall()
                    _ai_ctx["latest_scan"] = [
                        {k: r.get(k) for k in r.keys()} for r in rows
                    ]
                except Exception:
                    pass
                # Open paper positions
                try:
                    _p = ts.PaperTradingEngine(conn)
                    _ai_ctx["open_positions"] = _p.open_positions[:8]
                    _ai_ctx["portfolio_summary"] = _p.get_summary()
                    _ai_ctx["risk_status"] = ts.get_portfolio_risk_status(_p)
                except Exception:
                    pass
                # Open options positions
                try:
                    _op = ts.OptionsPaperEngine(conn)
                    _ai_ctx["open_options"] = _op.get_positions("OPEN")[:8]
                except Exception:
                    pass

                # Append user msg
                st.session_state.ai_chat_history.append(
                    {"role": "user", "content": _user_msg}
                )
                with st.chat_message("user"):
                    st.markdown(_user_msg)

                # Call AI
                with st.chat_message("assistant"):
                    with st.spinner("Analyst thinking…"):
                        _reply = _AI.chat(_user_msg, context=_ai_ctx)
                    st.markdown(_reply)
                    if _AI.last_latency_ms:
                        st.caption(f"⚡ {_AI.last_latency_ms} ms · {_AI.provider}")

                st.session_state.ai_chat_history.append(
                    {"role": "assistant", "content": _reply}
                )
                st.rerun()

            if st.session_state.ai_chat_history and st.button(
                    "🗑 Clear Chat", key="ai_clear_btn"):
                st.session_state.ai_chat_history = []
                st.rerun()

    # 📰 News Tracker moved to 🎛 Management tab
    st.divider()

    universe = st.selectbox(
        "Universe",
        ["all", "canadian", "us", "tsx", "tsxv", "cse", "neo",
         "nasdaq", "nyse", "cboe", "otc", "smallcap", "russell2000"],
        help="Which exchanges to pull tickers from",
    )

    watchlist_raw = st.text_input(
        "Custom Watchlist (overrides Universe)",
        placeholder="AAPL, RY.TO, NVDA, SHOP.TO …",
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        min_prob = st.number_input("Min Prob %", 0, 100, 0, step=5)
    with col2:
        min_expl = st.number_input("Min Score", 0, 100, 0, step=5)

    top_n = st.slider("Top N results", 5, 50, 15)

    st.divider()
    no_earnings   = st.checkbox("Skip earnings risk")
    catalyst_only = st.checkbox("Catalyst only")
    squeeze_only  = st.checkbox("Squeeze only")
    paper_only    = st.checkbox("Paper mode (don't log)", value=False)

    st.divider()
    run_btn = st.button("🚀  Run Scan", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    # ── Trading-window advisory ────────────────────────────────────────────────
    _session_now = ts.MarketClock.get_session()
    _sq          = _session_now["quality"]
    if _sq == "AVOID":
        st.sidebar.warning(
            f"⚠️ **{_session_now['name']}** — scanning now but paper bot will "
            f"hold off on new entries. Clears in "
            f"{_session_now['remaining_seconds'] // 60}m "
            f"{_session_now['remaining_seconds'] % 60}s.",
            icon="🕐",
        )
    elif _sq == "CAUTION":
        st.sidebar.info(
            "⚠️ Midday chop window — paper bot uses stricter thresholds "
            "(prob ≥ 72%, explosive ≥ 75).",
            icon="🌤️",
        )

    conn, _mode = get_db()

    # Pre-scan: refresh pending outcomes silently
    logger  = ts.CallLogger.__new__(ts.CallLogger)
    logger.conn = conn
    logger.MIN_SCORE = 40
    logger.MIN_PROB  = 55
    with contextlib.suppress(Exception):
        logger.update_calls()

    args = argparse.Namespace(
        universe      = universe,
        watchlist     = watchlist_raw.replace(" ", "") if watchlist_raw.strip() else None,
        file          = None,
        pattern       = None,
        min_prob      = min_prob,
        min_explosive = min_expl,
        no_earnings   = no_earnings,
        catalyst_only = catalyst_only,
        biotech       = False,
        squeeze       = squeeze_only,
        max_float     = None,
        max_cap       = None,
        no_otc        = False,
        min_rs        = 0,
        earnings_quality = False,
        premarket     = False,
        top           = top_n,
        export        = False,
        debug         = False,
    )

    scanner        = ts.BreakoutScanner()
    scanner.args   = args
    scanner._debug = False

    status = st.sidebar.status("Running scan…", expanded=True)
    with status:
        # ── Progress bar widget that updates live as the scanner progresses ──
        _scan_bar = st.progress(0, text="🚀 Starting scan…")
        _scan_log = st.empty()

        # Phase-weighted progress ranges (each phase gets a slice of 0-100%):
        #   Phase 1 (Universe):   0 → 10 %
        #   Phase 2 (Filter):    10 → 20 %
        #   Phase 3 (Catalyst):  20 → 50 %
        #   Phase 4 (Analysis):  50 → 99 %
        _phase_ranges = {1: (0, 10), 2: (10, 20), 3: (20, 50), 4: (50, 99)}
        _phase_names  = {1: "Universe", 2: "Filter", 3: "News Catalyst", 4: "Analysis"}

        def _on_scan_progress(phase: int, current: int, total: int, message: str):
            lo, hi = _phase_ranges.get(phase, (0, 100))
            if total and total > 0:
                frac = max(0.0, min(1.0, current / total))
            else:
                frac = 0.0
            pct = int(lo + (hi - lo) * frac)
            label = f"Phase {phase}/4 · {_phase_names.get(phase, '?')}: {message}"
            try:
                _scan_bar.progress(pct, text=label[:200])
            except Exception:
                pass

        scanner.progress_callback = _on_scan_progress

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                scanner.run()
            except Exception as e:
                st.error(f"Scan error: {e}")

        try:
            _scan_bar.progress(100, text=f"✓ Scan complete — {len(scanner.results)} stocks qualified")
        except Exception:
            pass
        st.write(f"Done — {len(scanner.results)} stocks qualified")

    status.update(label="Scan complete ✓", state="complete", expanded=False)

    # Save results
    st.session_state.results    = scanner.results
    st.session_state.scan_ran   = True
    st.session_state.market_ctx = {
        "spy_raw":   getattr(scanner, "_universe_raw",      0),
        "filtered":  getattr(scanner, "_universe_filtered",  0),
        "advanced":  getattr(scanner, "_universe_advanced",  0),
    }
    st.session_state.regime = getattr(scanner, "regime", {})

    # Post-scan: log to portfolio tracker
    if not paper_only and scanner.results:
        import uuid as _u
        meta = {
            "scan_id":      str(_u.uuid4())[:8],
            "spy":          getattr(scanner, "spy", None),
            "universe_size": len(scanner.results),
        }
        n = logger.log_scan_results(scanner.results, meta)
        if n:
            st.sidebar.success(f"Logged {n} new call(s) to tracker.")

    # Post-scan: auto-trade paper portfolio
    all_results = scanner.results + getattr(scanner, "rejected_results", [])
    paper   = ts.PaperTradingEngine(conn)
    opened  = paper.auto_trade(all_results)
    last_s  = getattr(paper, "_last_session", {})
    last_sq = last_s.get("quality", "UNKNOWN") if isinstance(last_s, dict) else "UNKNOWN"

    if opened:
        session_tag = f" [{last_sq}]" if last_sq != "UNKNOWN" else ""
        st.sidebar.success(
            f"💼 Paper bot opened {len(opened)} position(s){session_tag}: "
            + ", ".join(o["ticker"] for o in opened)
        )
    elif not paper_only:
        if last_sq == "AVOID":
            rem = last_s.get("remaining_seconds", 0) if isinstance(last_s, dict) else 0
            st.sidebar.warning(
                f"🕐 Paper bot paused — {last_s.get('name','Opening/Closing Volatility')}. "
                f"Resumes in {rem // 60}m {rem % 60}s.",
            )
        elif last_sq == "CAUTION":
            st.sidebar.info(
                "⚠️ Midday window — paper bot used stricter thresholds. "
                "Check Scan Results to see if any signals qualify."
            )
        else:
            n_pass = sum(1 for r in all_results if r.get("trade_filter_pass"))
            n_qual = sum(1 for r in all_results
                         if r.get("trade_filter_pass")
                         and (r.get("explosive_score", 0) >= 70
                              or r.get("probability", 0) >= 65))
            port   = paper.get_summary()
            if port["open_positions"] >= ts._PAPER_MAX_POSITIONS:
                st.sidebar.info("💼 Paper portfolio full (5/5 positions).")
            elif port["available_cash"] < ts._PAPER_MIN_CASH:
                st.sidebar.info(f"💼 Not enough cash (${port['available_cash']:.2f}).")
            elif n_pass == 0:
                st.sidebar.info("💼 No trades passed fee-adj R/R ≥ 2.0 + reward ≥ 20% filter.")
            elif n_qual == 0:
                st.sidebar.info(
                    f"💼 {n_pass} trade(s) passed R/R filter but none had "
                    f"explosive ≥ 70 or prob ≥ 65%."
                )


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
(tab_today, tab_mgmt, tab_analyst, tab_whale, tab_dash, tab_results, tab_chart,
 tab_analytics, tab_paper, tab_fees, tab_options) = st.tabs([
    "🌅  Today",
    "🎛  Management",
    "🏛  Market Analyst",
    "🐋  Whale Watch",
    "📊  Dashboard",
    "🔍  Scan Results",
    "📈  Stock Chart",
    "📉  Analytics",
    "💼  Paper Portfolio",
    "💰  Fees Tracker",
    "⚡  Options",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB — TODAY (one-click command center: what to buy today)
# ══════════════════════════════════════════════════════════════════════════════
with tab_today:
    conn, _td_mode = get_db()
    st.markdown("## 🌅 Today — Your Daily Command Center")
    st.caption("One click. The whole bot distilled into: should I trade, what to "
               "buy, and am I on track. Run this each morning before the open.")

    if st.button("🌅 Run Morning Scan", key="cc_btn", type="primary",
                 use_container_width=True):
        try:
            from command_center import run_morning_scan
            from monitor import OPTIONS_AUTO_WATCHLIST as _CC_WL
            _ccp = st.empty()
            def _cc_cb(msg):
                _ccp.caption(f"⏳ {msg}")
            with st.spinner("Running the full morning scan (~2 min)…"):
                _cc = run_morning_scan(conn, progress=_cc_cb, watchlist=_CC_WL)
            _ccp.empty()
            st.session_state["_cc"] = _cc
        except Exception as _cce:
            st.error(f"Morning scan failed: {_cce}")

    _cc = st.session_state.get("_cc")
    if not _cc:
        st.info("Hit **Run Morning Scan** for today's complete picture in one view.",
                icon="🌅")
    else:
        # ── VERDICT banner ───────────────────────────────────────────────────
        _vcolor = ("#3fb950" if "GO" in _cc["verdict"] or "BUY" in _cc["verdict"]
                   else "#f85149" if "STAND DOWN" in _cc["verdict"]
                   else "#e3b341")
        st.markdown(
            f"<div style='background:#161b22;border-left:8px solid {_vcolor};"
            f"border-radius:10px;padding:16px 20px;margin:8px 0'>"
            f"<div style='font-size:1.5em;font-weight:800;color:{_vcolor}'>{_cc['verdict']}</div>"
            f"<div style='color:#c9d1d9;margin-top:6px'>{_cc['action']}</div>"
            f"</div>", unsafe_allow_html=True)

        _m = _cc["market"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bias", _m.get("bias", "?"))
        c2.metric("Regime", _m.get("regime", "?"))
        c3.metric("Risk", _m.get("risk", "?"))
        c4.metric("VIX", f"{_m.get('vix', 0):.1f}" if _m.get("vix") else "?")

        # ── Macro event risk (jobs/CPI/PCE/FOMC) ─────────────────────────────
        _macro = _cc.get("macro") or {}
        if _macro:
            _ml = _macro.get("level", "LOW")
            _mc = {"HIGH": "#f85149", "ELEVATED": "#e3b341"}.get(_ml, "#3fb950")
            st.markdown(
                f"<div style='background:#161b22;border-left:5px solid {_mc};"
                f"border-radius:6px;padding:8px 14px;margin:6px 0;color:#c9d1d9'>"
                f"<b>🌐 Macro event risk: {_ml}</b> — {_macro.get('advice','')}"
                f"</div>", unsafe_allow_html=True)
            _up = _macro.get("upcoming") or []
            if _up:
                st.caption("Upcoming: " + " · ".join(
                    f"{e['name']} ({e['days_away']}d{'≈' if e.get('approx_date') else ''})"
                    for e in _up))

        # ── Panic (if firing) ────────────────────────────────────────────────
        if _cc.get("panic"):
            st.markdown("### 🚨 PANIC SIGNAL — highest-conviction buy")
            for p in _cc["panic"]:
                st.success(f"**{p['label']}** — {p['win_rate_60d']:.0f}% win rate / "
                           f"+{p['mean_return_60d']:.1f}% mean over 60 days. "
                           f"Scale into SPY/momentum/calls.", icon="🚨")

        # ── Buy list ─────────────────────────────────────────────────────────
        _cb1, _cb2 = st.columns(2)
        with _cb1:
            st.markdown("### 📈 Stocks to Buy (momentum leaders)")
            if _cc.get("top_stocks"):
                import pandas as _pdt
                df = _pdt.DataFrame([{
                    "Ticker": s["ticker"], "Price": f"${s['price']:.2f}",
                    "6mo": f"{s['mom_6m']*100:+.0f}%", "RSI": f"{s['rsi']:.0f}",
                    "Stop": f"${s['stop']:.2f}",
                    "": "⚠️ext" if s.get("extended") else "",
                } for s in _cc["top_stocks"]])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No momentum leaders ranked.")
        with _cb2:
            st.markdown("### 🎰 Options Play (best validated setup)")
            if _cc.get("options"):
                import pandas as _pdo
                df = _pdo.DataFrame([{
                    "Ticker": o["ticker"],
                    "Contract": f"${float(o['strike'] or 0):.0f}{str(o['type'])[0].upper()}",
                    "Exp": o["expiry"], "Prem": f"${float(o['premium'] or 0):.2f}",
                    "Grade": o["grade"], "Decision": o["decision"],
                } for o in _cc["options"]])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption("Only act on BUY-grade. WATCH = not yet.")
            else:
                st.caption("No options setups passed quality gate (correct on a "
                           "low-conviction day).")

        # ── Catalysts + Goal ─────────────────────────────────────────────────
        _cc1, _cc2 = st.columns(2)
        with _cc1:
            st.markdown("### 🐋 Smart-Money & Catalysts")
            if _cc.get("whale"):
                for w in _cc["whale"]:
                    st.markdown(f"- **{w['ticker']}** ({w['score']}/100) — "
                                f"{str(w['signal'])[:50]} · entry ${w['entry']}")
            if _cc.get("catalysts"):
                for c in _cc["catalysts"]:
                    _se = {"bullish":"🟢","bearish":"🔴"}.get(c["sentiment"], "⚪")
                    st.markdown(f"- {_se} **{c['who']}** [{c['tickers']}]: "
                                f"{c['text'][:70]}")
            if not _cc.get("whale") and not _cc.get("catalysts"):
                st.caption("No fresh catalysts. Refresh Whale Watch / VIP tabs for detail.")
        with _cc2:
            st.markdown("### 🎯 Goal Progress")
            _g = _cc.get("goal") or {}
            if _g:
                st.metric(f"${_g.get('start_capital',500):.0f} → "
                          f"${_g.get('goal_capital',1500):.0f}",
                          f"${_g.get('real_value_now',0):,.0f}",
                          delta=f"{_g.get('status','?')} · "
                                f"{_g.get('pct_of_goal',0):.0f}% there")
                st.caption(f"Pace {_g.get('actual_monthly_pct',0):+.1f}%/mo · "
                           f"need {_g.get('required_monthly_pct',0):+.1f}%/mo · "
                           f"day {_g.get('days_elapsed',0)} of "
                           f"{_g.get('days_elapsed',0)+_g.get('days_remaining',0)}")
            else:
                st.caption("Set your goal in the Management tab.")

        st.caption(f"Scan completed {_cc['as_of'][:19]} UTC · "
                   f"Market read: {_m.get('strategy','')[:120]}")

    # ── LIVE BROKER (Alpaca paper) — real account track record ───────────────
    st.divider()
    st.markdown("### 🏦 Live Broker — Alpaca Paper Account")
    try:
        from broker import AlpacaPaperBroker
        _bk = AlpacaPaperBroker()
        _bk_test = _bk.test_connection()
    except Exception as _bke:
        _bk = None
        _bk_test = {"ok": False, "error": str(_bke)}

    if not _bk_test.get("ok"):
        st.info(
            "**Not connected yet.** To trade a real broker paper account "
            "(honest fills + a clean track record to judge profitability):\n"
            "1. Create a free account at **alpaca.markets** → Paper Trading\n"
            "2. In the Alpaca dashboard, set your paper starting equity (e.g. $200)\n"
            "3. Generate **paper** API keys\n"
            "4. Add them as secrets: `ALPACA_PAPER_KEY`, `ALPACA_PAPER_SECRET` "
            "(Streamlit secrets + GitHub repo secrets)\n"
            "5. Set `BROKER_MODE=alpaca_paper` (GitHub secret/env) to let the bot "
            "auto-trade it.\n\n"
            f"_Status: {_bk_test.get('error','not configured')}_", icon="🏦")
    else:
        _acct = _bk.get_account()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity", f"${_acct.get('equity',0):,.2f}",
                  delta=f"{_acct.get('day_pnl_pct',0):+.2f}% today")
        c2.metric("Cash", f"${_acct.get('cash',0):,.2f}")
        c3.metric("Buying power", f"${_acct.get('buying_power',0):,.2f}")
        c4.metric("Day P&L", f"${_acct.get('day_pnl',0):+,.2f}")
        st.caption(f"Account {_bk_test.get('account_number','?')} · status "
                   f"{_acct.get('status','?')} · this is a REAL broker paper "
                   f"account (simulated money, real fills).")

        # Options-enabled status (you trade options, so this is the key check)
        try:
            _ost = _bk.options_status()
            if _ost.get("can_buy_longs"):
                st.success(f"✅ Options ready — {_ost['msg']}", icon="🎰")
            else:
                st.warning(f"⚠️ {_ost['msg']}", icon="🎰")
        except Exception:
            pass

        # Option positions (your focus)
        _opos = _bk.get_option_positions()
        st.markdown("**Open OPTION positions (live from Alpaca):**")
        if _opos:
            import pandas as _pdop
            df = _pdop.DataFrame([{
                "Contract": p["symbol"], "Underlying": p["underlying"],
                "Qty": f"{p['qty']:.0f}", "Entry": f"${p['avg_entry']:.2f}",
                "Now": f"${p['current']:.2f}", "Value": f"${p['market_value']:,.0f}",
                "P&L": f"${p['unrealized_pnl']:+,.0f}",
                "P&L %": f"{p['unrealized_pct']:+.1f}%",
            } for p in _opos])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No open option positions in the Alpaca paper account.")

        _pos = _bk.get_positions()
        st.markdown("**Open STOCK positions (live from Alpaca):**")
        if _pos:
            import pandas as _pdbk
            df = _pdbk.DataFrame([{
                "Ticker": p["ticker"], "Qty": f"{p['qty']:.0f}",
                "Entry": f"${p['avg_entry']:.2f}", "Now": f"${p['current']:.2f}",
                "Value": f"${p['market_value']:,.0f}",
                "P&L": f"${p['unrealized_pnl']:+,.0f}",
                "P&L %": f"{p['unrealized_pct']:+.1f}%",
            } for p in _pos])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No open positions in the Alpaca paper account.")

        _ords = _bk.get_orders(limit=10)
        if _ords:
            with st.expander("Recent orders", expanded=False):
                import pandas as _pdo2
                st.dataframe(_pdo2.DataFrame([{
                    "Ticker": o["ticker"], "Side": o["side"], "Qty": o["qty"],
                    "Type": o["type"], "Class": o["order_class"],
                    "Status": o["status"], "Filled@": o["filled_avg_price"],
                    "When": o["submitted_at"],
                } for o in _ords]), use_container_width=True, hide_index=True)

    # ── TRADE JOURNAL — attribution analytics (drives optimization) ───────────
    st.markdown("### 📓 Trade Journal — What's Actually Working")
    st.caption("Every broker option trade tagged by quality band, DTE, sector, "
               "and momentum. After ~2-4 weeks this tells you WHICH trades make "
               "money — so optimization is driven by evidence, not guesses.")
    try:
        from trade_journal import analyze as _tj_analyze, recent as _tj_recent
        _ja = _tj_analyze(conn)
    except Exception as _tje:
        _ja = None
        st.caption(f"(journal not available: {_tje})")
    if _ja:
        _ov = _ja.get("overall", {})
        if _ov.get("n", 0) == 0:
            st.info(f"No closed broker trades yet ({_ja.get('n_open',0)} open). "
                    "Breakdowns populate as trades close — check back after a few "
                    "days of the autonomous run.", icon="📓")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Closed trades", _ov["n"])
            c2.metric("Win rate", f"{_ov['win_rate']:.0f}%")
            c3.metric("Expectancy", f"{_ov['expectancy']:+.0f}%/trade")
            c4.metric("Total P&L", f"${_ov['total_pnl']:+,.0f}")
            st.caption(f"Avg win {_ov['avg_win']:+.0f}% · avg loss "
                       f"{_ov['avg_loss']:+.0f}% · {_ja.get('n_open',0)} still open")

            import pandas as _pdj
            def _bd_table(title, d):
                if not d:
                    return
                st.markdown(f"**{title}**")
                rows = [{"Bucket": k, "N": v.get("n",0),
                         "Win%": f"{v.get('win_rate',0):.0f}%",
                         "Expectancy": f"{v.get('expectancy',0):+.0f}%",
                         "Total P&L": f"${v.get('total_pnl',0):+,.0f}"}
                        for k, v in sorted(d.items(),
                                           key=lambda x: -(x[1].get('expectancy') or 0))]
                st.dataframe(_pdj.DataFrame(rows), use_container_width=True, hide_index=True)
            _jc1, _jc2 = st.columns(2)
            with _jc1:
                _bd_table("By quality band", _ja.get("by_quality"))
                _bd_table("By DTE", _ja.get("by_dte"))
                _bd_table("By exit reason", _ja.get("by_exit"))
            with _jc2:
                _bd_table("By sector", _ja.get("by_sector"))
                _bd_table("By momentum", _ja.get("by_momentum"))
            st.caption("Read it like this: if a bucket has negative expectancy "
                       "across a decent N, stop taking those trades. That's the "
                       "evidence-based optimization.")

    # ── MORNING ROUTINE checklist ────────────────────────────────────────────
    st.divider()
    st.markdown("### ✅ Morning Routine (in order)")
    st.markdown(
        "1. **Run Morning Scan** above ☝️ — read the verdict first.\n"
        "2. **Check the verdict:** GO = trade normally · CAUTION = high-conviction "
        "only · STAND DOWN = no new longs · PANIC = aggressive buy.\n"
        "3. **If 🚨 PANIC is firing** — that's the strongest signal. Scale into "
        "SPY/momentum/calls and stop here.\n"
        "4. **Buy from the momentum list** (top 2-3 not flagged ⚠️extended), on "
        "pullbacks toward their rising average. Set stops as shown.\n"
        "5. **Options:** only take BUY-grade plays. Skip WATCH. Lottery sizing (~5%).\n"
        "6. **Scan catalysts:** check 🐋 Whale Watch + 📣 VIP Feed for fresh "
        "Trump/Fed/insider moves on your names.\n"
        "7. **Glance at 🔄 Reversal Finder** (Market Analyst tab) for TRIGGERED "
        "turnarounds.\n"
        "8. **Check goal pace** — if behind, do NOT force trades to catch up. "
        "Discipline over FOMO.\n"
        "9. **Set alerts & walk away.** Telegram pings you on panic, VIP posts, "
        "and new A-grade options. Let the setups come to you."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — BOT MANAGEMENT DASHBOARD
#   Consolidated control panel — system status, engine diagnostics,
#   learning controls, news flow, configuration. Replaces sidebar clutter.
# ══════════════════════════════════════════════════════════════════════════════
with tab_mgmt:
    conn, _mgmt_mode = get_db()

    st.markdown('<div class="section-hdr">🎛 Bot Management Dashboard</div>',
                unsafe_allow_html=True)
    st.caption(
        "Single-pane-of-glass view: every engine status, all controls, "
        "and pipeline diagnostics in one place."
    )

    _mgmt_paper = ts.PaperTradingEngine(conn)
    _mgmt_ops   = ts.OptionsPaperEngine(conn)
    _mgmt_le    = ts.LearningEngine(conn)

    # ── SECTION 1 — SYSTEM STATUS OVERVIEW ───────────────────────────────────
    # ── DISCIPLINED PATH — goal-progress scoreboard ───────────────────────────
    st.markdown("### 🎯 Disciplined Path — $500 → $1,500 (12 mo)")
    st.caption(
        "The disciplined-plan scoreboard. Reads the paper portfolio's % return "
        "and applies it to your real account. The single most important question "
        "each day: am I on pace to hit my goal? Configure your real numbers "
        "below — defaults are $500 → $1,500 over 365 days (~9%/month compound)."
    )

    _gcfg1, _gcfg2, _gcfg3, _gcfg4 = st.columns(4)
    with _gcfg1:
        _g_start = st.number_input("Start capital ($)", 100.0, 100000.0, 500.0,
                                   step=50.0, key="g_start")
    with _gcfg2:
        _g_goal = st.number_input("Goal ($)", 100.0, 1000000.0, 1500.0,
                                  step=100.0, key="g_goal")
    with _gcfg3:
        _g_days = st.number_input("Horizon (days)", 30, 730, 365,
                                  step=30, key="g_days")
    with _gcfg4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _g_apply = st.button("💾 Apply", key="g_apply", use_container_width=True)

    try:
        from goal_tracker import GoalTracker
        import trading_scanner as _ts_gt
        _paper_gt = _ts_gt.PaperTradingEngine(conn)
        _gt = GoalTracker(conn, start_capital=_g_start, goal_capital=_g_goal,
                          horizon_days=int(_g_days))
        if _g_apply:
            _gt.update_config(_g_start, _g_goal, int(_g_days))
            st.toast("Goal configuration saved.", icon="✅")
        _sc = _gt.scorecard(_paper_gt)
    except Exception as _gte:
        _sc = None
        st.warning(f"Goal tracker unavailable: {_gte}")

    if _sc:
        # Big stat row
        _cA, _cB, _cC, _cD = st.columns(4)
        _cA.metric("Real value now", f"${_sc['real_value_now']:,.0f}",
                   delta=f"{_sc['paper_return_pct']:+.1f}% since start")
        _cB.metric("Goal", f"${_sc['goal_capital']:,.0f}",
                   delta=f"{_sc['pct_of_goal']:.0f}% of journey")
        _cC.metric("Monthly pace",  f"{_sc['actual_monthly_pct']:+.2f}%",
                   delta=f"need {_sc['required_monthly_pct']:+.2f}%")
        _proj_diff = _sc["projected_end_value"] - _sc["goal_capital"]
        _cD.metric("Projected end", f"${_sc['projected_end_value']:,.0f}",
                   delta=f"{'+' if _proj_diff >= 0 else ''}${_proj_diff:,.0f} vs goal",
                   delta_color=("normal" if _proj_diff >= 0 else "inverse"))

        # Pace banner
        _status_color = {"ahead": "#3fb950", "on_pace": "#58a6ff",
                         "behind": "#f85149"}.get(_sc["status"], "#8b949e")
        _status_text = {"ahead": "AHEAD OF PACE",
                        "on_pace": "ON PACE",
                        "behind": "BEHIND PACE"}.get(_sc["status"], "?")
        _msg = (
            f"<b>{_status_text}</b> · Day {_sc['days_elapsed']} of "
            f"{_sc['days_elapsed']+_sc['days_remaining']} · "
            f"{_sc['days_remaining']} days remaining · "
            f"+${_sc['real_gain_dollars']:+,.0f} so far"
        )
        st.markdown(
            f"<div style='background:#161b22;border-left:5px solid {_status_color};"
            f"border-radius:6px;padding:10px 14px;margin:6px 0 12px 0;color:#c9d1d9'>"
            f"{_msg}</div>", unsafe_allow_html=True,
        )

        # Honest plan-context reminder
        with st.expander("📋 Disciplined-plan rules (tap to view)", expanded=False):
            st.markdown(
                "- **80% momentum stocks** (top 4–6 from Momentum Leaders, rotated monthly)\n"
                "- **15% whale-validated swings** (1–2 names from Whale Watch with fresh 13D / cluster / heavy insider)\n"
                "- **5% catalyst options only** — fire only on tier-1 VIP alerts (Trump posts about a momentum-leader ticker)\n"
                "- **Max 10–12% per position** · cut at −10–15% · exit on close < 50-SMA\n"
                "- **Skip options most months.** Skip new trades into the weekend. Stand down in BEAR regime.\n"
                "- **No FOMO, no averaging down.** The scorecard tells the truth — trust it."
            )
        st.caption(f"Baseline: ${_sc['paper_baseline_equity']:,.0f} on "
                   f"{_sc['start_date']} · paper equity now: ${_sc['paper_equity_now']:,.0f}")
        st.divider()

    # ── PANIC DETECTOR — "blood in the streets" buy signal ────────────────────
    st.markdown("### 🚨 Panic Detector — Buy-Fear Signal")
    st.caption(
        "12y of SPY+VIX backtested signatures. When fear hits documented panic "
        "levels, history says: buy. Each row shows current vs trigger, plus the "
        "historical edge if the signature fires. Telegram alerts fire automatically "
        "on first activation; reset when conditions normalize."
    )
    try:
        from panic_detector import PanicDetector
        _ps = PanicDetector(conn).status()
    except Exception as _pe:
        _ps = None
        st.warning(f"Panic detector unavailable: {_pe}")

    if _ps and _ps.get("snap"):
        _snap = _ps["snap"]
        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric("SPY", f"${_snap['spy']:.2f}",
                   delta=f"{_snap['spy_ret_1d']*100:+.2f}% today")
        _c2.metric("VIX", f"{_snap['vix']:.1f}",
                   delta=("LOW"     if _snap['vix'] < 15 else
                          "NORMAL"  if _snap['vix'] < 20 else
                          "ELEVATED" if _snap['vix'] < 30 else
                          "FEAR"    if _snap['vix'] < 40 else "EXTREME"),
                   delta_color=("normal" if _snap['vix'] < 30 else "inverse"))
        _c3.metric("RSI (14d)", f"{_snap['spy_rsi14']:.0f}",
                   delta=("oversold" if _snap['spy_rsi14'] < 30 else
                          "neutral"  if _snap['spy_rsi14'] < 70 else "overbought"))
        _fires = sum(1 for s in _ps["signatures"] if s["currently_fired"])
        _c4.metric("Signatures firing", f"{_fires} of {len(_ps['signatures'])}",
                   delta=("🚨 BUY-FEAR" if _fires else "no panic signal"),
                   delta_color=("normal" if _fires else "off"))

        import pandas as _pdpn
        rows = []
        for s in _ps["signatures"]:
            status = "🚨 ACTIVE" if s["currently_fired"] else "—"
            rows.append({
                "Status": status,
                "Signature": s["label"],
                "Historical n": s["n_historical"],
                "20d Win%": f"{s['win_rate_20d']:.0f}%",
                "20d Mean": f"{s['mean_return_20d']:+.1f}%",
                "60d Win%": f"{s['win_rate_60d']:.0f}%",
                "60d Mean": f"{s['mean_return_60d']:+.1f}%",
                "60d worst (p10)": f"{s['worst_60d_p10']:+.1f}%",
            })
        st.dataframe(_pdpn.DataFrame(rows), use_container_width=True, hide_index=True)

        if _fires:
            st.markdown(
                f"<div style='background:#3d0e0e;border-left:5px solid #f85149;"
                f"border-radius:6px;padding:10px 14px;margin:6px 0;color:#c9d1d9'>"
                f"<b>🚨 PANIC SIGNAL ACTIVE.</b> Historical edge has been strong "
                f"every time this fired. Consider scaling INTO momentum leaders, "
                f"SPY, or quality call options on a 20–60 day horizon. "
                f"Telegram alert sent.</div>", unsafe_allow_html=True)
        else:
            st.caption("No panic signature currently firing. Detector watches each "
                       "cycle; you'll get a Telegram alert the moment one trips.")
    else:
        st.info("Loading panic detector...", icon="⏳")

    st.divider()

    st.markdown("### 🚦 System Status Overview")
    _stat_cols = st.columns(6)

    with _stat_cols[0]:
        st.metric("Database", "🟢 ONLINE" if conn is not None else "🔴 OFFLINE",
                  delta=_mgmt_mode, delta_color="off")
    with _stat_cols[1]:
        _ai_ok = _AI is not None and _AI.available
        st.metric("AI Analyst",
                  "🟢 " + _AI.provider.upper() if _ai_ok else "⚪ OFF",
                  delta=None if _ai_ok else "optional", delta_color="off")
    with _stat_cols[2]:
        try:
            from data_providers import alpaca_available, tradingview_ta_available
            _alpaca_on = alpaca_available()
            _tv_on     = tradingview_ta_available()
        except Exception:
            _alpaca_on = False
            _tv_on     = False
        st.metric("Live Prices",
                  "🟢 ALPACA" if _alpaca_on else "🟡 YFINANCE",
                  delta=None if _alpaca_on else "add ALPACA_API_KEY",
                  delta_color="off")
    with _stat_cols[3]:
        st.metric("TradingView TA", "🟢 ON" if _tv_on else "⚪ OFF",
                  delta=None if _tv_on else "auto-installs", delta_color="off")
    with _stat_cols[4]:
        try:
            _adj_top = _mgmt_le.get_active_adjustments()
            _pun_top = _mgmt_le.get_punishment_status()
            _le_st = "🛑 PUNISH" if _pun_top.get("active") else f"🟢 iter #{_adj_top.get('learning_iteration', 0)}"
        except Exception:
            _le_st = "⚪ —"
        st.metric("Learning Engine", _le_st)
    with _stat_cols[5]:
        try:
            _re_top = ts.get_portfolio_risk_status(_mgmt_paper)
            _re_lvl = _re_top.get("risk_level", "NORMAL")
            _re_e = {"NORMAL":"🟢","CAUTION":"🟡","HALT":"🛑","EMERGENCY_HALT":"🚨"}.get(_re_lvl,"⚪")
            st.metric("Risk Engine", f"{_re_e} {_re_lvl}",
                      delta=f"DD {_re_top.get('drawdown_pct', 0):.1f}%")
        except Exception:
            st.metric("Risk Engine", "⚪ —")

    st.divider()

    # ── SECTION 2 — PORTFOLIO + LEARNING ENGINE (2-column) ───────────────────
    _mc_a, _mc_b = st.columns(2)

    with _mc_a:
        st.markdown("### 💼 Portfolio Snapshot")
        try:
            _ps = _mgmt_paper.get_summary()
            _os_ = _mgmt_ops.get_summary()
            _p_cols = st.columns(2)
            with _p_cols[0]:
                st.markdown("**Stocks**")
                st.markdown(
                    f"<div style='background:#161b22;border-radius:6px;padding:8px 12px'>"
                    f"💰 Cash <b>${_ps.get('available_cash', 0):,.2f}</b><br>"
                    f"📂 Open <b>{_ps.get('open_positions', 0)}/{_ps.get('max_positions', 5)}</b><br>"
                    f"💹 Realized <b>${_ps.get('realized_pnl', 0):+,.2f}</b><br>"
                    f"📊 Trades <b>{_ps.get('trades_made', 0)}</b>"
                    f"</div>", unsafe_allow_html=True)
            with _p_cols[1]:
                st.markdown("**Options**")
                st.markdown(
                    f"<div style='background:#161b22;border-radius:6px;padding:8px 12px'>"
                    f"💰 Cash <b>${_os_.get('available_cash', 0):,.2f}</b><br>"
                    f"📂 Open <b>{_os_.get('n_open', 0)}</b><br>"
                    f"💹 Realized <b>${_os_.get('realized_pnl', 0):+,.2f}</b><br>"
                    f"📊 Trades <b>{_os_.get('trades_made', 0)}</b>"
                    f"</div>", unsafe_allow_html=True)
        except Exception as _pse:
            st.error(f"Portfolio error: {_pse}")

        st.markdown("### 🛡 Risk Engine")
        try:
            _rs = ts.get_portfolio_risk_status(_mgmt_paper)
            _lvl = _rs.get("risk_level", "NORMAL")
            _col = {"NORMAL":"#3fb950","CAUTION":"#e3b341","HALT":"#f85149","EMERGENCY_HALT":"#a371f7"}.get(_lvl,"#8b949e")
            st.markdown(
                f"<div style='background:#161b22;border:2px solid {_col};border-radius:8px;padding:10px 14px'>"
                f"<b style='color:{_col}'>{_lvl}</b>  ·  "
                f"DD <b>{_rs.get('drawdown_pct', 0):.2f}%</b>  ·  "
                f"Max sector <b>{_rs.get('max_sector_pct', 0):.0f}%</b>  ·  "
                f"Sizing <b>{_rs.get('size_multiplier', 1.0):.2f}×</b>"
                f"<br><small style='color:#8b949e'>{_rs.get('reason', '')}</small>"
                f"</div>", unsafe_allow_html=True)
            _sec_exp = _rs.get("sector_exposure", {})
            _n_unk = int(_rs.get("n_unknown_sector", 0) or 0)
            _unk_pct = float(_rs.get("unknown_sector_pct", 0) or 0)

            # Show Backfill button if any positions have unknown sector
            if _n_unk > 0:
                _bf_c1, _bf_c2 = st.columns([3, 1])
                with _bf_c1:
                    st.info(
                        f"ℹ️ {_n_unk} legacy position(s) missing sector data "
                        f"({_unk_pct:.0f}% of book). Click → to fetch sectors from yfinance.",
                        icon="ℹ️",
                    )
                with _bf_c2:
                    if st.button("🔧 Backfill Sectors",
                                  key="mgmt_backfill_sectors",
                                  type="primary",
                                  help="One-shot: fetches sector via yfinance for "
                                       "positions opened before sector tracking was added"):
                        with st.spinner(f"Fetching sectors for {_n_unk} ticker(s)…"):
                            _bf = ts.backfill_position_sectors(conn, limit=200)
                            if _bf.get("updated", 0) > 0:
                                st.success(_bf.get("message", "Done"),
                                           icon="✅")
                                if _bf.get("details"):
                                    st.caption("Updated: " + ", ".join(_bf["details"][:15]))
                                st.rerun()
                            else:
                                st.warning(_bf.get("message", "No updates"))

            if _sec_exp:
                st.markdown("**Sector exposure**")
                for _sec, _pct in sorted(_sec_exp.items(), key=lambda x: -x[1])[:8]:
                    # Unknown bucket gets a different visual treatment
                    if _sec == "Unknown":
                        _bc = "#6e7681"
                        _label = "⚪ Unknown (legacy)"
                    elif _pct > 40:
                        _bc = "#f85149"
                        _label = _sec
                    elif _pct > 25:
                        _bc = "#e3b341"
                        _label = _sec
                    else:
                        _bc = "#58a6ff"
                        _label = _sec
                    st.markdown(
                        f"<div style='display:flex;gap:8px;align-items:center;margin:2px 0'>"
                        f"<div style='width:160px;font-size:0.78em'>{_label}</div>"
                        f"<div style='flex:1;background:#21262d;border-radius:3px;height:8px'>"
                        f"<div style='width:{min(100, _pct)}%;background:{_bc};height:8px;border-radius:3px'></div></div>"
                        f"<div style='width:42px;font-size:0.78em;text-align:right'>{_pct:.0f}%</div>"
                        f"</div>", unsafe_allow_html=True)
        except Exception as _rse:
            st.warning(f"Risk status unavailable: {_rse}")

    with _mc_b:
        st.markdown("### 🧠 Self-Learning Engine")
        try:
            _adj = _mgmt_le.get_active_adjustments()
            _pun = _mgmt_le.get_punishment_status()
            st.markdown(
                f"**Iter #{_adj.get('learning_iteration', 0)}**  ·  "
                f"Min Master Score <b>{_adj.get('min_master_score', 65):.0f}</b>  ·  "
                f"Size Cap <b>{_adj.get('size_multiplier_cap', 1.0):.2f}×</b>"
            )
            if _pun.get("active"):
                st.error(
                    f"🛑 PUNISHMENT MODE — recovery target ${_pun.get('recovery_target', 0):,.2f}, "
                    f"floor {_pun.get('master_score_floor', 75):.0f}, cap {_pun.get('size_cap', 0.5):.2f}×",
                    icon="🛑")
            for _bl_label, _bl in [("Sectors", _adj.get("sector_blacklist")),
                                    ("Patterns", _adj.get("pattern_blacklist")),
                                    ("Wyckoff", _adj.get("wyckoff_blacklist"))]:
                if _bl:
                    st.caption(f"🚫 {_bl_label}: {', '.join(_bl)}")

            # ── Continuous learning panel: signal performance + recent nudges ─
            with st.expander("📈 Continuous Learning (per-trade signal performance)",
                              expanded=False):
                # Process any new closes that the monitor hasn't picked up yet
                if st.button("🔄 Process Recent Closes",
                             key="mgmt_proc_closes", use_container_width=True,
                             help="Run continuous learning on any trades closed since last monitor cycle"):
                    with st.spinner("Processing closed trades..."):
                        _lcr = _mgmt_le.process_recent_closes(max_per_cycle=200)
                        st.success(f"Processed {_lcr['n_processed']} "
                                    f"({_lcr['n_wins']}W / {_lcr['n_losses']}L)")
                        st.rerun()

                # Per-signal performance
                _sig_types = ["sector", "pattern", "score_bucket",
                              "prob_bucket", "exit_reason"]
                _sig_labels = {"sector": "🏭 Sector", "pattern": "📐 Pattern",
                               "score_bucket": "💯 Score Bucket",
                               "prob_bucket": "🎯 Prob Bucket",
                               "exit_reason": "🚪 Exit Reason"}
                for _st_name in _sig_types:
                    rows = _mgmt_le.get_signal_performance(signal_type=_st_name, min_trades=1)
                    if not rows:
                        continue
                    st.markdown(f"**{_sig_labels.get(_st_name, _st_name)}**")
                    _df_rows = []
                    for r in rows[:10]:
                        wr  = r.get("win_rate_pct", 0)
                        n_t = r.get("n_trades", 0)
                        _icon = "🟢" if wr >= 60 else "🟡" if wr >= 40 else "🔴"
                        _df_rows.append({
                            "Value":     r.get("signal_value", "?"),
                            "Trades":    n_t,
                            "Wins":      r.get("n_wins", 0),
                            "Losses":    r.get("n_losses", 0),
                            "Win %":     f"{_icon} {wr:.0f}%",
                            "Expectancy": f"${r.get('expectancy', 0):+.2f}",
                        })
                    st.dataframe(pd.DataFrame(_df_rows), hide_index=True,
                                 use_container_width=True)

                # Recent learning events log
                st.markdown("**🪄 Recent Auto-Adjustments**")
                _events = _mgmt_le.get_recent_learning_events(limit=10)
                if _events:
                    for e in _events:
                        _t = str(e.get("event_at", ""))[:19]
                        _et = e.get("event_type", "?")
                        _dt = e.get("detail", "")
                        _bv = e.get("before_value", "")
                        _av = e.get("after_value", "")
                        _ic = {
                            "NUDGE_RAISE_SCORE": "📈",
                            "NUDGE_LOWER_SCORE": "📉",
                            "BLACKLIST_ADD":     "🚫",
                        }.get(_et, "•")
                        _change = (f"  ·  <code>{_bv} → {_av}</code>"
                                   if _bv and _av else "")
                        st.markdown(
                            f"{_ic} <small style='color:#8b949e'>{_t}</small>  "
                            f"<b>{_et}</b>{_change}<br>"
                            f"<small>{_dt}</small>",
                            unsafe_allow_html=True)
                else:
                    st.caption("No auto-adjustments yet — learning kicks in after the first 10 closed trades.")

            st.markdown("**💡 Today's Improvement Suggestion**")
            _latest = _mgmt_le.get_latest_suggestion()
            _today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
            _has_today = (_latest and str(_latest.get("suggestion_date", "")).startswith(_today))
            if _has_today and _latest.get("suggestion"):
                _ap = bool(_latest.get("applied", 0))
                _bc = "#3fb950" if _ap else "#58a6ff"
                _tag = "✅ Applied" if _ap else "🆕 New"
                st.markdown(
                    f"<div style='background:#161b22;border-left:4px solid {_bc};padding:8px 12px;border-radius:4px;margin:4px 0'>"
                    f"<small style='color:#8b949e'>{_tag}  ·  <b>{_latest.get('category', '?')}</b></small><br>"
                    f"<b>{_latest.get('suggestion', '—')}</b><br>"
                    f"<small style='color:#c9d1d9'>{_latest.get('rationale', '')}</small><br>"
                    f"<code style='font-size:0.8em'>{_latest.get('action', '')}</code>"
                    f"</div>", unsafe_allow_html=True)
                if not _ap:
                    _abc = st.columns(2)
                    with _abc[0]:
                        if st.button("⚡ Apply Now", key=f"mgmt_apply_{_latest.get('id')}",
                                     type="primary", use_container_width=True):
                            try:
                                _r = _mgmt_le.apply_suggestion(_latest.get("id"))
                                if _r.get("ok"):
                                    st.success(f"✅ {_r['message']}")
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.warning(_r.get("message", "Couldn't apply"))
                            except Exception as _e:
                                st.error(f"Apply failed: {_e}")
                    with _abc[1]:
                        if st.button("✓ Mark Manual", key=f"mgmt_mark_{_latest.get('id')}",
                                     use_container_width=True):
                            _mgmt_le.mark_suggestion_applied(_latest.get("id"))
                            st.rerun()
            else:
                st.caption("No suggestion yet today.")
                if st.button("🪄 Generate Now", key="mgmt_gen_sug"):
                    with st.spinner("AI analysing..."):
                        try:
                            _new = _mgmt_le.generate_improvement_suggestion(_AI, _mgmt_paper)
                            if _new.get("suggestion"):
                                st.success(f"💡 {_new['suggestion']}")
                                st.rerun()
                        except Exception as _e:
                            st.error(f"Generation failed: {_e}")
        except Exception as _le_e:
            st.warning(f"Learning engine unavailable: {_le_e}")

    st.divider()

    # ── SECTION 3 — NEWS + POSITIONS HEALTH ──────────────────────────────────
    _nc_a, _nc_b = st.columns(2)

    with _nc_a:
        st.markdown("### 📰 News Tracker")
        try:
            from news_agent import NewsAgent as _MgmtNA
            _mna = _MgmtNA(conn, ai_analyst=_AI)
            _mna_hours = st.slider("Lookback (h)", 1, 72, 24, key="mgmt_news_hours")
            _nbc1, _nbc2 = st.columns([3, 1])
            with _nbc1:
                _pulse = _mna.get_market_pulse(hours=_mna_hours)
                _mood = _pulse.get("mood", "NEUTRAL")
                _mood_map = {"BULLISH":("#3fb950","🟢"),"MILDLY_BULLISH":("#58a6ff","🔵"),
                             "NEUTRAL":("#8b949e","⚪"),"MILDLY_BEARISH":("#e3b341","🟡"),
                             "BEARISH":("#f85149","🔴")}
                _mc, _me = _mood_map.get(_mood, ("#8b949e","⚪"))
                st.markdown(
                    f"<div style='background:#161b22;border-left:4px solid {_mc};padding:6px 10px;border-radius:4px'>"
                    f"{_me} <b>{_mood.replace('_', ' ')}</b>  ·  "
                    f"net {_pulse.get('net_sentiment', 0):+.2f}  ·  "
                    f"{_pulse.get('n_events', 0)} events"
                    f"</div>", unsafe_allow_html=True)
            with _nbc2:
                if st.button("🔄 Pull", key="mgmt_news_pull", use_container_width=True):
                    with st.spinner("Fetching..."):
                        _rr = _mna.run_cycle(max_per_source=15, max_new=20)
                        st.success(f"+{_rr['new']} new")
            _news = _mna.get_recent_market_news(hours=_mna_hours, min_impact=5, limit=8)
            for _ev in _news[:6]:
                _s = _ev.get("sentiment", "NEUTRAL")
                _e_emoji = {"POSITIVE":"✅","NEGATIVE":"🛑"}.get(_s, "⚪")
                _bc = "#3fb950" if _s=="POSITIVE" else "#f85149" if _s=="NEGATIVE" else "#8b949e"
                st.markdown(
                    f"<div style='border-left:3px solid {_bc};padding:4px 8px;margin:2px 0;font-size:0.85em'>"
                    f"{_e_emoji} <span style='color:#8b949e'>{_ev.get('category','?')} · "
                    f"{int(_ev.get('impact_score') or 0)}/10 · {_ev.get('source', '')}</span><br>"
                    f"{(_ev.get('headline') or '')[:100]}"
                    f"</div>", unsafe_allow_html=True)
        except Exception as _ne:
            st.warning(f"News tracker unavailable: {_ne}")

    with _nc_b:
        st.markdown("### 📊 Open Positions")
        try:
            _open_stk = _mgmt_paper.open_positions
            if _open_stk:
                _live_tk = tuple(p["ticker"] for p in _open_stk)
                _salt = st.session_state.get("price_refresh_salt", 0)
                _live_p = _fetch_live_prices(_live_tk, _salt=_salt)
                _ph = []
                _tot_unrl = 0.0
                _tot_cost = 0.0
                for _p in _open_stk:
                    _tk = _p.get("ticker", "?")
                    _ep = float(_p.get("entry_price") or 0)
                    _sh = float(_p.get("shares") or 0)
                    _gi = float(_p.get("gross_invested") or 0)
                    _st_p = float(_p.get("stop_loss") or 0)
                    _tg = float(_p.get("target_price") or 0)
                    _cur = float(_live_p.get(_tk, 0))
                    if _cur > 0 and _gi > 0:
                        _cv = _sh * _cur
                        _u = _cv - _gi
                        _up = _u / _gi * 100
                        _tot_unrl += _u
                        _tot_cost += _gi
                        _flag = ""
                        if _st_p > 0 and _cur <= _st_p:
                            _flag = "🛑"
                        elif _tg > 0 and _cur >= _tg:
                            _flag = "🎯"
                        _ph.append({"Ticker": _tk, "Cur": f"${_cur:.2f}",
                                    "P&L": f"${_u:+.1f}({_up:+.1f}%)",
                                    "Stop": f"${_st_p:.2f}", "Tgt": f"${_tg:.2f}",
                                    "Flag": _flag})
                    else:
                        _ph.append({"Ticker": _tk, "Cur": "?", "P&L": "—",
                                    "Stop": f"${_st_p:.2f}", "Tgt": f"${_tg:.2f}", "Flag": "—"})
                st.dataframe(pd.DataFrame(_ph), hide_index=True, use_container_width=True)
                if _tot_cost > 0:
                    _c = "#3fb950" if _tot_unrl >= 0 else "#f85149"
                    st.markdown(
                        f"<b>Total unrealized:</b> "
                        f"<span style='color:{_c}'>${_tot_unrl:+.2f} ({_tot_unrl/_tot_cost*100:+.2f}%)</span>",
                        unsafe_allow_html=True)
                if st.button("🔄 Refresh Prices", key="mgmt_refresh_p"):
                    st.session_state["price_refresh_salt"] = st.session_state.get("price_refresh_salt", 0) + 1
                    try:
                        _fetch_live_prices.clear()
                    except Exception:
                        pass
                    st.rerun()
            else:
                st.info("No open stock positions.", icon="📂")
        except Exception as _pe:
            st.warning(f"Positions error: {_pe}")

    st.divider()

    # ── SECTION 3.38 — MOMENTUM LEADERS (the active, MCPT-validated strategy) ─
    st.markdown("### 🚀 Momentum Leaders — Active Strategy")
    st.caption(
        "The bot's PRIMARY stock strategy after MCPT validation: buy the strongest "
        "liquid names in a confirmed uptrend (above 50- & 200-day SMA). "
        "Cross-sectional momentum tested at p=0.004 — beats pure beta by ~93% and "
        "random selection 99.6% of the time. Exits on a close below the 50-day SMA."
    )
    _mc1, _mc2 = st.columns([3, 1])
    with _mc1:
        st.markdown(
            "<div style='background:#0d2818;border-left:4px solid #3fb950;"
            "border-radius:6px;padding:8px 12px;font-size:0.85em'>"
            "✅ <b>Replaced</b> the chart-pattern engine (no edge: p≈0.23–0.81) and "
            "the illiquid micro-cap universe. Trades liquid US large-caps + ETFs only."
            "</div>", unsafe_allow_html=True,
        )
    with _mc2:
        _mom_run = st.button("🚀 Rank Leaders", key="mgmt_mom_btn",
                             type="primary", use_container_width=True)

    if _mom_run:
        try:
            from momentum_strategy import MomentumStrategy
            _mp = st.progress(0.0, text="Scanning liquid universe…")
            def _mom_cb(i, total, t):
                _mp.progress(min(i / max(total, 1), 1.0), text=f"Scoring {t} ({i}/{total})")
            with st.spinner("Ranking momentum leaders…"):
                _mom_res = MomentumStrategy(conn).rank(top_n=15, min_mom_6m=0.0,
                                                       progress=_mom_cb)
            _mp.empty()
            st.session_state["_mom_last"] = _mom_res
        except Exception as _me:
            st.error(f"Momentum scan error: {_me}")

    _mom_show = st.session_state.get("_mom_last")
    if _mom_show:
        st.success(f"{len(_mom_show)} liquid uptrend leaders", icon="🚀")
        import pandas as _pdm
        # Earnings badges (read cached only — no API calls per row)
        try:
            from earnings_engine import EarningsCalendar as _EC
            _ec = _EC(conn)
        except Exception:
            _ec = None

        def _earn_cell(tkr):
            if _ec is None:
                return "—"
            row = _ec.get(tkr)
            if not row or not row.get("next_earnings"):
                return "—"
            try:
                from datetime import date as _dt
                d = _dt.fromisoformat(row["next_earnings"][:10])
                dd = (d - _dt.today()).days
                if dd < 0:
                    return "—"
                if dd <= 7:
                    return f"⚠️ {dd}d"
                if dd <= 21:
                    return f"📅 {dd}d"
                return f"{dd}d"
            except Exception:
                return "—"

        _rows = [{
            "Ticker": r["ticker"],
            "6-mo %": f"{r['mom_6m']*100:+.0f}%",
            "3-mo %": f"{r['mom_3m']*100:+.0f}%",
            "RSI": f"{r['rsi']:.0f}",
            "Price": f"${r['price']:.2f}",
            "Stop": f"${r['stop']:.2f}",
            "Earnings": _earn_cell(r["ticker"]),
            "Flag": "⚠️ extended" if r["extended"] else "",
        } for r in _mom_show]
        st.dataframe(_pdm.DataFrame(_rows), use_container_width=True, hide_index=True)
        st.caption("**Earnings col:** `⚠️ Xd` = within 7d (IV-crush risk on long calls) · "
                   "`📅 Xd` = within 21d · plain `Xd` = far out · `—` = no cached date "
                   "(refresh PEAD scanner in 🐋 Whale Watch tab to populate)")
    else:
        st.info("Hit **Rank Leaders** to see the current momentum leaders.", icon="📊")

    st.divider()

    # ── SECTION 3.39 — MOMENTUM vs SPY SCORECARD (live forward edge) ──────────
    st.markdown("### 📊 Momentum vs SPY — Live Scorecard")
    st.caption(
        "The real test: since the momentum pivot went live, is the bot beating a "
        "simple buy-and-hold of SPY? Updated once per day from true mark-to-market "
        "equity. This is forward, out-of-sample evidence — the backtest's promissory note."
    )
    try:
        from benchmark_tracker import BenchmarkTracker
        _bsc = BenchmarkTracker(conn).get_scorecard()
    except Exception as _bse:
        _bsc = None
        st.warning(f"Scorecard unavailable: {_bse}")

    if _bsc:
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Bot (momentum)", f"{_bsc['bot_return_pct']:+.2f}%")
        _c2.metric("SPY (buy & hold)", f"{_bsc['spy_return_pct']:+.2f}%")
        _alpha = _bsc["alpha_pct"]
        _c3.metric("Alpha vs SPY", f"{_alpha:+.2f}%",
                   delta=("beating SPY" if _bsc["winning"] else "trailing SPY"),
                   delta_color=("normal" if _bsc["winning"] else "inverse"))
        st.caption(f"Since {_bsc['inception']} · {_bsc['days_tracked']} day(s) tracked "
                   f"· equity ${_bsc['current_equity']:,.0f}")

        _ser = _bsc.get("series") or []
        if len(_ser) >= 2:
            import pandas as _pdb
            # NOTE: must NOT use the name `_df` here — it would shadow the
            # module-level _df() query helper (Streamlit tab blocks share module
            # scope), breaking every later _df(...) call with "DataFrame is not
            # callable". Use a distinct local name.
            _bench_df = _pdb.DataFrame({
                "date": [s["date"] for s in _ser],
                "Bot (momentum)": [round(s["bot_ret_pct"], 2) for s in _ser],
                "SPY": [round(s["spy_ret_pct"], 2) for s in _ser],
            }).set_index("date")
            st.line_chart(_bench_df)
        else:
            st.caption("📈 Equity curve appears once 2+ daily snapshots exist.")
    else:
        st.info("No snapshots yet. The monitor records one per day during market "
                "hours — the scorecard fills in starting the next trading session.",
                icon="⏳")

    st.divider()

    # ── SECTION 3.395 — MOMENTUM CALLS (cheap-leverage version of the edge) ───
    st.markdown("### 🎰 Momentum Calls — Cheap Leverage")
    st.caption(
        "The options bot: slightly-OTM (5–15%) calls, 1–6 weeks to expiry, "
        "premium ≤ $5, ONLY on names that pass the validated momentum rank. "
        "Auto-exits at +100% take profit / −50% stop / DTE ≤ 2 (theta cliff). "
        "Lottery sizing — expect most to lose 100%, the winners pay for the misses."
    )

    _moc1, _moc2 = st.columns([3, 1])
    with _moc2:
        _mo_btn = st.button("🔎 Scan Setups", key="mgmt_mopt_btn",
                            type="primary", use_container_width=True)

    if _mo_btn:
        try:
            from momentum_options import MomentumOptionsStrategy
            _mp = st.progress(0.0, text="Searching option chains…")
            def _mo_cb(i, n, t):
                _mp.progress(min(i / max(n, 1), 1.0), text=f"Checking {t} ({i}/{n})")
            with st.spinner("Finding lottery setups on momentum leaders…"):
                _mo_res = MomentumOptionsStrategy(conn).find_setups(
                    top_n_underlyings=8, progress=_mo_cb)
            _mp.empty()
            st.session_state["_mopt_last"] = _mo_res
        except Exception as _me:
            st.error(f"Setup scan error: {_me}")

    _mo_show = st.session_state.get("_mopt_last")
    if _mo_show:
        import pandas as _pdo
        _rows = [{
            "Ticker": s["ticker"],
            "Strike": f"${s['strike']:.2f}",
            "Expiry": s["expiry"],
            "DTE": s["dte"],
            "OTM%": f"{s['otm_pct']*100:+.1f}%",
            "Prem": f"${s['premium']:.2f}",
            "IV": f"{s['iv']*100:.0f}%",
            "Vol/OI": f"{s['volume']}/{s['open_interest']}",
            "Mom 6m": f"{s.get('mom_6m',0)*100:+.0f}%",
        } for s in _mo_show]
        st.success(f"{len(_mo_show)} clean lottery setup(s)", icon="🎰")
        st.dataframe(_pdo.DataFrame(_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Hit **Scan Setups** to see today's momentum-aligned call candidates.", icon="🔎")

    # Open momentum-options positions table
    try:
        _rows = conn.execute(
            "SELECT ticker, strike, expiry, entry_price, contracts, entry_date, "
            "gross_invested FROM options_positions "
            "WHERE status='OPEN' AND strategy='momentum_call' "
            "ORDER BY entry_date DESC"
        ).fetchall()
    except Exception:
        _rows = []
    if _rows:
        def _g(r, k, i): return r.get(k) if hasattr(r, "get") else r[i]
        import pandas as _pdo2
        _data = [{
            "Ticker":   _g(r, "ticker", 0),
            "Strike":   f"${float(_g(r, 'strike', 1) or 0):.2f}",
            "Expiry":   _g(r, "expiry", 2),
            "Entry $":  f"${float(_g(r, 'entry_price', 3) or 0):.2f}",
            "×":        int(_g(r, "contracts", 4) or 0),
            "Opened":   _g(r, "entry_date", 5),
            "Cost":     f"${float(_g(r, 'gross_invested', 6) or 0):.0f}",
        } for r in _rows]
        st.markdown("**Open momentum-call positions:**")
        st.dataframe(_pdo2.DataFrame(_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── SECTION 3.40 — UNIFIED SCANNER (one brain, every universe) ───────────
    st.markdown("### 🎯 Unified Scanner")
    st.caption(
        "ONE orchestrator that feeds every universe into the same Master-Score brain "
        "(it already fuses breakout · entry-quality · Wyckoff · MTF · news · squeeze · "
        "whale · early-momentum). Picks up catalyst plays — earnings & news gaps like "
        "SNOW, RCAT, UMAC — that the technical breakout scan never sees."
    )

    _us_modes = {
        "⚡ Smart (movers + watchlist + breakout)": "smart",
        "🚀 Top Movers only (today's real gainers)": "movers",
        "📈 Latest breakout scan": "breakout",
        "🌍 Universal (ALL exchanges — slow)": "universal",
    }
    _usc1, _usc2, _usc3 = st.columns([3, 1, 1])
    with _usc1:
        _us_mode_lbl = st.selectbox("Universe", list(_us_modes.keys()),
                                    key="mgmt_unified_mode")
    with _usc2:
        _us_min = st.number_input("Min score", 0, 100, 55, key="mgmt_unified_min")
    with _usc3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _us_run = st.button("🎯 Run Scan", key="mgmt_unified_btn",
                            type="primary", use_container_width=True)

    _us_mode = _us_modes[_us_mode_lbl]
    if _us_mode == "universal":
        st.warning(
            "Universal mode scans every US + Canadian + OTC ticker. It can take "
            "several minutes and is rate-limited by yfinance — best run occasionally, "
            "not every refresh.", icon="⏳",
        )

    if _us_run:
        try:
            from unified_scanner import run_unified_scan
            try:
                from monitor import OPTIONS_AUTO_WATCHLIST as _US_WL
            except Exception:
                _US_WL = []
            _prog = st.progress(0.0, text="Building universe…")
            def _us_cb(i, total, tk):
                _prog.progress(min(i / max(total, 1), 1.0),
                               text=f"Scoring {tk}  ({i}/{total})")
            with st.spinner(f"Running {_us_mode} scan…"):
                _us_res = run_unified_scan(
                    conn, mode=_us_mode, min_score=int(_us_min),
                    market_regime=None, watchlist=_US_WL, progress=_us_cb,
                )
            _prog.empty()
            st.session_state["_unified_last"] = _us_res
        except Exception as _use:
            st.error(f"Unified scan error: {_use}")

    # Show latest persisted snapshot (survives reruns)
    _us_show = st.session_state.get("_unified_last")
    if _us_show is None:
        try:
            from unified_scanner import get_latest_unified
            _us_show = get_latest_unified(conn, limit=60)
        except Exception:
            _us_show = []

    if _us_show:
        _n_buy = sum(1 for r in _us_show if r.get("decision") == "BUY")
        st.success(f"{len(_us_show)} candidate(s) ≥ score · {_n_buy} BUY", icon="🎯")
        for r in _us_show[:30]:
            _dec = r.get("decision", "SKIP")
            _col = {"BUY": "#3fb950", "WATCH": "#e3b341"}.get(_dec, "#8b949e")
            _bias = r.get("bias", "bullish")
            _bias_ic = "🟢" if _bias == "bullish" else "🔴"
            _pct = r.get("pct_change")
            _pct_txt = f"{_pct:+.1f}% today" if _pct not in (None, "") else ""
            _srcs = r.get("sources") or []
            _src_txt = " · ".join(s for s in _srcs if s)
            with st.expander(
                    f"{_bias_ic} **{r.get('ticker','?')}** — {r.get('score',0)}/100  "
                    f"({r.get('grade','?')})  ·  {_dec}  {('· ' + _pct_txt) if _pct_txt else ''}",
                    expanded=False,
            ):
                st.markdown(
                    f"<div style='background:#161b22;border-left:4px solid {_col};"
                    f"border-radius:6px;padding:8px 12px;margin-bottom:6px'>"
                    f"<b style='color:{_col}'>{_dec}</b> · bias <b>{_bias}</b> · "
                    f"size ×{r.get('size_multiplier',0):.2f} · sources: "
                    f"<code>{_src_txt or 'n/a'}</code><br>"
                    f"<span style='color:#8b949e;font-size:0.85em'>"
                    f"{r.get('summary','')}</span></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No unified scan run yet. Pick a universe and hit **Run Scan**.", icon="📭")

    st.divider()

    # ── SECTION 3.45 — EARLY MOMENTUM SCANNER (pre-explosion detector) ───────
    st.markdown("### 🔮 Pre-Explosion Scanner")
    st.caption(
        "Catches stocks in the 1-3 day window BEFORE they explode. "
        "Looks for: coiled spring (BB compression building) · volume base · "
        "StockTwits velocity acceleration · MTF transition · short covering · "
        "fresh catalyst. Each individual signal is noisy — the edge comes from clusters."
    )

    _em_c1, _em_c2 = st.columns([3, 1])
    with _em_c1:
        _em_input = st.text_area(
            "Universe to scan (comma-separated)",
            value="UPS, SOFI, HIMS, KLAR, SPY, QQQ, AAPL, NVDA, TSLA, AMD, "
                  "PLTR, AMC, GME, F, T, BAC, MSFT, META, AMZN, GOOG",
            key="mgmt_early_input",
            height=80,
        )
    with _em_c2:
        _em_min = st.number_input("Min score", 0, 100, 30,
                                    key="mgmt_early_minscore")
        _em_run = st.button("🔮 Scan for Pre-Explosion",
                              key="mgmt_early_btn",
                              type="primary",
                              use_container_width=True)

    if _em_run:
        _tickers = [t.strip().upper() for t in _em_input.split(",") if t.strip()]
        if _tickers:
            with st.spinner(f"Scanning {len(_tickers)} tickers for pre-explosion signals…"):
                try:
                    from early_momentum import EarlyMomentumScanner
                    _ems = EarlyMomentumScanner(conn=conn)
                    _em_results = _ems.scan_universe(_tickers, min_score=_em_min)
                except Exception as _eme:
                    st.error(f"Scanner error: {_eme}")
                    _em_results = []

            if _em_results:
                st.success(f"Found {len(_em_results)} candidate(s) ≥ {_em_min}",
                           icon="🎯")
                for r in _em_results:
                    _tier = r.get("tier", "QUIET")
                    _meta = {
                        "IMMINENT": ("#f85149", "🚨", "Move 1-2d likely"),
                        "BUILDING": ("#e3b341", "🔥", "3-5d setup"),
                        "WATCH":    ("#58a6ff", "👀", "Early signals"),
                        "QUIET":    ("#8b949e", "⚪", "No signal"),
                    }
                    _col, _ic, _sub = _meta.get(_tier, ("#8b949e", "⚪", ""))
                    with st.expander(
                            f"{_ic} **{r['ticker']}** — {r['score']}/100  ·  "
                            f"{_tier}  ·  {r['n_signals']} signal(s) firing",
                            expanded=(_tier in ("IMMINENT", "BUILDING")),
                    ):
                        st.markdown(
                            f"<div style='background:#161b22;border-left:4px solid {_col};"
                            f"border-radius:6px;padding:10px 14px;margin-bottom:8px'>"
                            f"<b style='color:{_col}'>{_tier}</b>: {r['outlook']}"
                            f"</div>", unsafe_allow_html=True,
                        )
                        # Per-signal breakdown bars
                        _sig_names = {
                            "coiled_spring":   "🌀 Coiled Spring",
                            "volume_base":     "💼 Volume Base",
                            "velocity":        "🚀 Velocity",
                            "mtf_transition":  "🔄 MTF Transition",
                            "short_covering":  "🔥 Short Covering",
                            "fresh_catalyst":  "📰 Fresh Catalyst",
                        }
                        _max_pts = {"coiled_spring": 18, "volume_base": 18,
                                    "velocity": 18, "mtf_transition": 15,
                                    "short_covering": 15, "fresh_catalyst": 16}
                        for sk, sv in (r.get("signals") or {}).items():
                            _pts = sv.get("score", 0)
                            _mx  = _max_pts.get(sk, 18)
                            _pct = int(_pts / _mx * 100) if _mx else 0
                            _bc  = ("#3fb950" if _pct >= 60 else
                                    "#e3b341" if _pct >= 30 else
                                    "#30363d" if _pct == 0 else "#58a6ff")
                            st.markdown(
                                f"<div style='display:flex;gap:8px;align-items:center;margin:2px 0'>"
                                f"<div style='width:160px;font-size:0.8em;color:#8b949e'>"
                                f"{_sig_names.get(sk, sk)}</div>"
                                f"<div style='flex:1;background:#21262d;border-radius:3px;height:8px'>"
                                f"<div style='width:{_pct}%;background:{_bc};height:8px;border-radius:3px'></div></div>"
                                f"<div style='width:48px;font-size:0.78em;text-align:right'>{_pts}/{_mx}</div>"
                                f"</div>"
                                f"<div style='font-size:0.72em;color:#6e7681;margin-left:168px'>"
                                f"{sv.get('label', '')}</div>",
                                unsafe_allow_html=True,
                            )
            else:
                st.info(f"No tickers scored ≥ {_em_min}. Try lower threshold or different universe.",
                        icon="📭")

    st.divider()

    # ── SECTION 3.55 — WHALE INTELLIGENCE (follow the money) ─────────────────
    st.markdown("### 🐋 Whale Intelligence")
    st.caption(
        "Five free sources of institutional/insider/political activity: "
        "Quiver Quant (Congressional trades) · SEC EDGAR (13D/13G/Form 4) · "
        "USASpending.gov (federal contracts) · StockTwits (trader sentiment). "
        "Feeds into Master Score as a 14th multiplier (1.00× to 1.10×)."
    )

    _wi_c1, _wi_c2 = st.columns([3, 1])
    with _wi_c1:
        _wi_ticker = st.text_input("Ticker to investigate",
                                     value="NVDA",
                                     key="mgmt_whale_input",
                                     placeholder="e.g. AAPL, NVDA, GME")
    with _wi_c2:
        _wi_run = st.button("🔍 Investigate",
                              key="mgmt_whale_btn",
                              type="primary",
                              use_container_width=True)

    if _wi_run and _wi_ticker:
        with st.spinner(f"Querying SEC EDGAR + Quiver + StockTwits + USASpending for {_wi_ticker.upper()}…"):
            try:
                from whale_intelligence import WhaleIntelligence
                _wi_report = WhaleIntelligence().full_report(_wi_ticker.upper())

                # ── Composite score banner ────────────────────────────────
                _ws = _wi_report.get("whale_score", 0)
                _wc = ("#3fb950" if _ws >= 60 else
                        "#58a6ff" if _ws >= 35 else
                        "#e3b341" if _ws >= 15 else "#8b949e")
                st.markdown(
                    f"<div style='background:#161b22;border:2px solid {_wc};"
                    f"border-radius:8px;padding:14px 18px;margin:8px 0'>"
                    f"<div style='font-size:0.85em;color:#8b949e'>WHALE SCORE</div>"
                    f"<div style='font-size:2em;font-weight:bold;color:{_wc}'>"
                    f"{_ws}/100</div>"
                    f"</div>", unsafe_allow_html=True,
                )

                # ── Flags row ─────────────────────────────────────────────
                if _wi_report.get("flags"):
                    st.markdown("**🚩 Detected signals:**")
                    for f in _wi_report["flags"][:10]:
                        st.markdown(f"- {f}")

                # ── Four-column sub-panels ────────────────────────────────
                _wp1, _wp2 = st.columns(2)

                # Congressional
                with _wp1:
                    with st.container(border=True):
                        st.markdown("**🏛️ Congressional Trading**")
                        _cg = _wi_report["congress"]
                        st.metric("Congress score", f"{_cg.get('congress_score', 0)}/35")
                        st.caption(f"{_cg.get('n_buyers_30d', 0)} unique buyers in last 30 days")
                        if _cg.get("cluster_detected"):
                            st.success("🏛️ CLUSTER BUY DETECTED")
                        for p in (_cg.get("purchases") or [])[:4]:
                            st.markdown(
                                f"- **{p.get('representative','?')}** "
                                f"({p.get('party','?')}) — "
                                f"`{p.get('amount','?')}` on {p.get('date','?')}"
                            )
                        if _cg.get("sales"):
                            st.warning(f"⚠️ {len(_cg['sales'])} recent SALE(s)")

                # SEC Filings
                with _wp2:
                    with st.container(border=True):
                        st.markdown("**📑 SEC EDGAR Filings**")
                        _sec = _wi_report["sec_filings"]
                        st.metric("SEC score", f"{_sec.get('sec_score', 0)}/50")
                        _sm1, _sm2 = st.columns(2)
                        _sm1.metric("13D filings", _sec.get("n_13d_30d", 0))
                        _sm2.metric("13G filings", _sec.get("n_13g_30d", 0))
                        _sm3, _sm4 = st.columns(2)
                        _sm3.metric("Form 4s", _sec.get("n_form4_30d", 0))
                        _sm4.metric("Unique insiders", _sec.get("n_unique_insiders", 0))
                        if _sec.get("has_activist"):
                            st.success(f"🐋 **ACTIVIST:** {_sec.get('activist_name', '?')[:60]}")
                        for f in (_sec.get("recent_filings") or [])[:3]:
                            st.caption(f"• {f.get('form','?')} · {f.get('filed_at','?')} · "
                                        f"{f.get('filer','?')[:50]}")

                _wp3, _wp4 = st.columns(2)

                # Gov contracts
                with _wp3:
                    with st.container(border=True):
                        st.markdown("**🇺🇸 Government Contracts**")
                        _gov = _wi_report["gov_contracts"]
                        st.metric("Gov score", f"{_gov.get('gov_score', 0)}/35")
                        if _gov.get("biggest_value", 0) > 0:
                            st.metric("Biggest contract",
                                       f"${_gov['biggest_value']/1e6:.1f}M")
                        for c in (_gov.get("contracts") or [])[:3]:
                            st.caption(f"• ${c.get('amount', 0)/1e6:.2f}M · "
                                        f"{c.get('agency','?')[:30]} · "
                                        f"{c.get('date','?')}")

                # StockTwits
                with _wp4:
                    with st.container(border=True):
                        st.markdown("**📱 StockTwits Sentiment**")
                        _st = _wi_report["stocktwits"]
                        _bp = float(_st.get("bull_pct", 0) or 0)
                        st.metric("Bull %",
                                    f"{_bp*100:.0f}%",
                                    f"{_st.get('n_messages', 0)} msgs")
                        _v = float(_st.get("velocity_ratio", 1) or 1)
                        st.metric("Velocity", f"{_v:.1f}× normal",
                                    "Going viral 🔥" if _v >= 3 else None)
                        st.caption(_st.get("summary", ""))

            except Exception as _we:
                st.error(f"Whale investigation failed: {_we}")
    st.divider()

    # ── SECTION 3.6 — TRADING MEMORY AGENT (AI episodic memory + critic) ─────
    st.markdown("### 🧠 Trading Memory Agent")
    st.caption(
        "Episodic memory of every closed trade · AI reflection after each close · "
        "pre-trade critic that vetoes setups where similar past trades have a "
        "high loss rate. Plugs into the Master Score as a 13th multiplier."
    )

    try:
        from trading_memory import TradingMemoryAgent
        _mgmt_tma = TradingMemoryAgent(conn, ai_analyst=_AI)

        _tms = _mgmt_tma.get_memory_stats()
        _m1, _m2, _m3, _m4, _m5 = st.columns(5)
        _m1.metric("Trades in memory", _tms.get("n_total_memories", 0))
        _m2.metric("Closed (learnable)", _tms.get("n_closed", 0))
        _m3.metric("AI Reflections", _tms.get("n_reflections", 0))
        _m4.metric("Critic Reviews",   _tms.get("n_critic_reviews", 0))
        _m5.metric("Critic Vetoes",    _tms.get("n_critic_skips", 0))

        _mc1, _mc2 = st.columns([1, 1])
        with _mc1:
            if st.button("🔄 Sync Memory Lifecycle",
                         key="mgmt_mem_sync",
                         help="Record any new opens and reflect on new closes (idempotent)"):
                with st.spinner("Syncing memory + generating reflections..."):
                    _r = _mgmt_tma.process_lifecycle_events(
                        generate_reflections=True, max_per_cycle=100
                    )
                    st.success(
                        f"+{_r['opens_recorded']} opens · "
                        f"{_r['closes_updated']} closes · "
                        f"{_r['reflections']} reflections",
                        icon="✅",
                    )
                    st.rerun()
        with _mc2:
            _test_ticker = st.text_input(
                "Test critic on ticker", value="SPY",
                key="mgmt_mem_test_ticker",
                help="Ask the critic about a hypothetical entry",
            )
            if st.button("🧪 Run Critic", key="mgmt_mem_critic_btn"):
                with st.spinner(f"Critic reviewing {_test_ticker}..."):
                    _critic_test = _mgmt_tma.critic_review(
                        candidate={
                            "ticker": _test_ticker.upper(),
                            "master_score": 70,
                            "sector": "Technology",
                            "pattern": "Cup and Handle",
                            "wyckoff_phase": "ACCUMULATION",
                        },
                        master_result={"grade": "B", "decision": "BUY"},
                    )
                    _v = _critic_test.get("verdict", "?")
                    _vc = {"BUY": "#3fb950", "WATCH": "#e3b341",
                           "SKIP": "#f85149"}.get(_v, "#8b949e")
                    st.markdown(
                        f"<div style='background:#161b22;border-left:4px solid {_vc};"
                        f"padding:10px 14px;border-radius:4px'>"
                        f"<b style='color:{_vc};font-size:1.2em'>{_v}</b>  "
                        f"(confidence {_critic_test.get('confidence', 0):.2f})<br>"
                        f"<small>{_critic_test.get('reasoning', '')}</small><br>"
                        f"<small style='color:#8b949e'>"
                        f"Based on {_critic_test.get('n_similar', 0)} similar past trades · "
                        f"WR {(_critic_test.get('similar_win_rate') or 0)*100:.0f}% · "
                        f"AI call: {_critic_test.get('was_ai_call', False)}"
                        f"</small></div>",
                        unsafe_allow_html=True,
                    )

        # ── Recent AI reflections ─────────────────────────────────────────
        _refls = _mgmt_tma.get_recent_reflections(limit=8)
        if _refls:
            with st.expander(f"💭 Recent AI Reflections ({len(_refls)})",
                              expanded=False):
                for r in _refls:
                    _w   = bool(r.get("won"))
                    _ic  = "✅" if _w else "🛑"
                    _col = "#3fb950" if _w else "#f85149"
                    _tk  = r.get("ticker", "?")
                    _pnl = float(r.get("net_pnl", 0) or 0)
                    _refl = (r.get("reflection") or "")[:400]
                    _ts  = str(r.get("reflection_at") or "")[:19]
                    st.markdown(
                        f"<div style='background:#0d1117;border-left:3px solid {_col};"
                        f"padding:8px 12px;margin:4px 0;border-radius:4px'>"
                        f"<div style='font-size:0.78em;color:#8b949e'>"
                        f"{_ic} <b>{_tk}</b>  ·  ${_pnl:+.2f}  ·  "
                        f"{r.get('pattern', '?')} / {r.get('sector', '?')}  ·  "
                        f"{_ts}</div>"
                        f"<div style='margin-top:4px;font-size:0.9em'>{_refl}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Recent critic reviews ────────────────────────────────────────
        _critics = _mgmt_tma.get_recent_critic_reviews(limit=10)
        if _critics:
            with st.expander(f"⚖️ Recent Critic Reviews ({len(_critics)})",
                              expanded=False):
                for c in _critics:
                    _v   = c.get("verdict", "?")
                    _col = {"BUY": "#3fb950", "WATCH": "#e3b341",
                            "SKIP": "#f85149"}.get(_v, "#8b949e")
                    _ic  = {"BUY": "✅", "WATCH": "👁",
                            "SKIP": "🛑"}.get(_v, "?")
                    _tk  = c.get("ticker", "?")
                    _ms  = float(c.get("master_score_pre", 0) or 0)
                    _ns  = int(c.get("n_similar", 0) or 0)
                    _wr  = float(c.get("similar_win_rate", 0) or 0)
                    _rsn = (c.get("reasoning") or "")[:300]
                    _ts  = str(c.get("reviewed_at") or "")[:19]
                    st.markdown(
                        f"<div style='border-left:3px solid {_col};"
                        f"padding:6px 10px;margin:3px 0;font-size:0.85em'>"
                        f"{_ic} <b>{_tk}</b>  ·  Master {_ms:.0f}/100  ·  "
                        f"<span style='color:{_col}'><b>{_v}</b></span>  ·  "
                        f"{_ns} similar (WR {_wr*100:.0f}%)<br>"
                        f"<small style='color:#8b949e'>{_rsn}</small>  "
                        f"<small style='color:#6e7681'>· {_ts}</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    except Exception as _tme:
        st.warning(f"Memory agent unavailable: {_tme}")

    st.divider()

    # ── SECTION 3.5 — SQUEEZE SCANNER (2-10x hunter) ─────────────────────────
    st.markdown("### 🚀 Squeeze Scanner — 2-10× Hunter")
    st.caption(
        "Combines short interest + float tightness + days-to-cover + "
        "Bollinger compression + volume velocity + catalyst proximity into "
        "ONE explosive-move score. EXPLOSIVE category = 2-10x potential setups."
    )

    _sc_c1, _sc_c2 = st.columns([3, 1])
    with _sc_c1:
        _sc_input = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL, NVDA, AMC, GME, BBBY, TSLA, AMD, PLTR",
            key="mgmt_squeeze_input",
            help="Add tickers you want to scan for squeeze potential",
        )
    with _sc_c2:
        _sc_min = st.number_input("Min score", 0, 100, 40,
                                    key="mgmt_squeeze_minscore")

    if st.button("🔍 Scan for Squeezes", key="mgmt_squeeze_btn", type="primary"):
        _tickers = [t.strip().upper() for t in _sc_input.split(",") if t.strip()]
        if _tickers:
            with st.spinner(f"Scanning {len(_tickers)} tickers..."):
                _results = ts.SqueezeScanner.scan_many(_tickers, conn=conn, min_score=_sc_min)
            if _results:
                st.success(f"Found {len(_results)} candidate(s) ≥ {_sc_min}", icon="🎯")
                for r in _results:
                    _cat = r["category"]
                    _meta = {
                        "EXPLOSIVE":  ("#f85149", "🔥", r["upside_estimate"]),
                        "HIGH":       ("#e3b341", "⚡", r["upside_estimate"]),
                        "MODERATE":   ("#58a6ff", "📊", r["upside_estimate"]),
                        "LOW":        ("#8b949e", "⚪", r["upside_estimate"]),
                    }
                    _col, _ic, _up = _meta.get(_cat, ("#8b949e", "⚪", ""))
                    with st.expander(f"{_ic} **{r['ticker']}** — Score {r['squeeze_score']}/100  ·  "
                                      f"{_cat}  ·  {_up}",
                                      expanded=(_cat in ("EXPLOSIVE","HIGH"))):
                        st.markdown(
                            f"<div style='background:#161b22;border-left:4px solid {_col};"
                            f"border-radius:6px;padding:10px 14px'>"
                            f"{r.get('recommendation','')}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        _kf1, _kf2, _kf3, _kf4 = st.columns(4)
                        _kf1.metric("Short %", f"{r.get('shortable_float_pct') or 0:.1f}%")
                        _kf2.metric("Days to Cover", f"{r.get('days_to_cover') or 0:.1f}")
                        _kf3.metric("Float", f"{(r.get('float_shares') or 0)/1e6:.1f}M")
                        _kf4.metric("Avg Volume", f"{(r.get('avg_volume') or 0)/1e6:.1f}M")

                        # Component breakdown bars
                        st.markdown("**Component breakdown:**")
                        for cn, cv in r.get("components", {}).items():
                            _pts = cv.get("pts", 0)
                            _mx  = cv.get("max", 1)
                            _pct = int(_pts / _mx * 100) if _mx else 0
                            _bc  = ("#3fb950" if _pct >= 70 else
                                    "#e3b341" if _pct >= 40 else "#f85149"
                                    if _pct > 0 else "#30363d")
                            st.markdown(
                                f"<div style='display:flex;gap:8px;align-items:center;margin:2px 0'>"
                                f"<div style='width:140px;font-size:0.78em;color:#8b949e'>"
                                f"{cn.replace('_',' ').title()}</div>"
                                f"<div style='flex:1;background:#21262d;border-radius:3px;height:8px'>"
                                f"<div style='width:{_pct}%;background:{_bc};height:8px;border-radius:3px'></div></div>"
                                f"<div style='width:48px;font-size:0.78em;text-align:right'>{_pts}/{_mx}</div>"
                                f"</div>"
                                f"<div style='font-size:0.72em;color:#6e7681;margin-left:148px'>{cv.get('label','')}</div>",
                                unsafe_allow_html=True,
                            )
                        if r.get("bb_squeeze"):
                            bb = r["bb_squeeze"]
                            st.caption(
                                f"📐 BB width {bb.get('bb_width_pct')}%  ·  "
                                f"pct rank {bb.get('pct_rank'):.0f}  ·  "
                                f"strength {bb.get('squeeze_strength')}  ·  "
                                f"{bb.get('days_in_squeeze')} days in squeeze"
                            )
            else:
                st.info(f"No tickers met the threshold of {_sc_min}. Try lower threshold or add more tickers.",
                        icon="📭")

    st.divider()

    # ── SECTION 3.7 — OPTIONS ENGINE STATUS + DIAGNOSTIC ─────────────────────
    st.markdown("### 💰 Options Engine Status")
    try:
        _opx_ops    = ts.OptionsPaperEngine(conn)
        _opx_summ   = _opx_ops.get_summary()
        _opx_open   = _opx_ops.get_positions("OPEN")
        _opx_closed = _opx_ops.get_positions("CLOSED")

        _ox1, _ox2, _ox3, _ox4 = st.columns(4)
        _ox1.metric("Options Cash", f"${_opx_summ.get('available_cash', 0):,.2f}")
        _ox2.metric("Open Positions", _opx_summ.get("n_open", 0))
        _ox3.metric("Closed", len(_opx_closed))
        _ox4.metric("Total Trades", _opx_summ.get("trades_made", 0))

        if _opx_summ.get("trades_made", 0) == 0:
            st.warning(
                "⚠️ Options engine hasn't traded yet. Click **🩺 Diagnose** "
                "below to see exactly which gate is blocking it.",
                icon="⚠️",
            )

        _od_c1, _od_c2 = st.columns(2)
        with _od_c1:
            if st.button("🩺 Diagnose Why No Options Trades",
                          key="mgmt_diag_options_btn",
                          type="primary",
                          use_container_width=True):
                with st.spinner("Walking through every gate the auto-entry uses…"):
                    _rep = []
                    # Gate 1: cash
                    _ok = _opx_summ.get("available_cash", 0) >= 20
                    _rep.append(("Cash ≥ $20", _ok,
                                  f"${_opx_summ.get('available_cash', 0):.2f}"))
                    # Gate 2: VIX
                    try:
                        _vix = ts.get_vix_level()
                        _vr  = (_vix or {}).get("regime", "?")
                        _rep.append(("VIX not EXTREME", _vr != "EXTREME",
                                      f"VIX {(_vix or {}).get('vix','?')} ({_vr})"))
                    except Exception as _e:
                        _rep.append(("VIX check", True, f"err: {_e}"))
                    # Gate 3: scan plays
                    try:
                        _plays = ts.get_scan_options_plays(conn, top_n=10)
                        _rep.append(("Scan plays exist (≥1)", len(_plays) > 0,
                                      f"{len(_plays)} from latest scan"))
                    except Exception:
                        _plays = []
                        _rep.append(("Scan plays exist", False, "load failed"))
                    # Gate 4: pre-filter on scan
                    from monitor import OPTIONS_MIN_SCORE, OPTIONS_MIN_PROB
                    _passing = [p for p in _plays
                                if float(p.get("explosive_score", 0)) >= OPTIONS_MIN_SCORE
                                and float(p.get("breakout_prob",   0)) >= OPTIONS_MIN_PROB]
                    _rep.append((f"Scan plays pass pre-filter "
                                  f"(score≥{OPTIONS_MIN_SCORE}, prob≥{OPTIONS_MIN_PROB}%)",
                                  len(_passing) > 0,
                                  f"{len(_passing)} of {len(_plays)}"))
                    # Gate 5: watchlist available
                    from monitor import OPTIONS_AUTO_WATCHLIST, OPTIONS_MIN_CASH
                    _rep.append((f"Watchlist ({len(OPTIONS_AUTO_WATCHLIST)} tickers)",
                                  True,
                                  f"Always evaluated: {', '.join(OPTIONS_AUTO_WATCHLIST[:6])}…"))
                    # Gate 6: check ALL watchlist tickers, not just first 6
                    _affordable    = []
                    _too_expensive = []
                    _yf_failed     = []
                    _strat_e       = ts.OptionsStrategyEngine()
                    _cash_now      = _opx_summ.get("available_cash", 0)
                    for _tk in OPTIONS_AUTO_WATCHLIST:
                        try:
                            _sugs = _strat_e.suggest(_tk, "bullish")
                            if not _sugs:
                                _yf_failed.append(_tk)
                                continue
                            _cheapest = min(
                                (float(s.get("mid", 0) or 0) for s in _sugs
                                  if float(s.get("mid", 0) or 0) > 0),
                                default=0
                            )
                            if _cheapest <= 0:
                                _yf_failed.append(_tk)
                                continue
                            _cost = _cheapest * 100
                            if _cost <= _cash_now:
                                _affordable.append((_tk, _cost))
                            else:
                                _too_expensive.append((_tk, _cost))
                        except Exception:
                            _yf_failed.append(_tk)
                    _rep.append((
                        f"Affordable watchlist plays (full {len(OPTIONS_AUTO_WATCHLIST)})",
                        len(_affordable) > 0,
                        f"{len(_affordable)} affordable · "
                        f"{len(_too_expensive)} too expensive · "
                        f"{len(_yf_failed)} yfinance unavailable",
                    ))

                    # Render
                    for label, ok, detail in _rep:
                        _icon = "✅" if ok else "🛑"
                        st.markdown(f"{_icon}  **{label}** — {detail}")

                    # Verdict
                    st.divider()
                    if not _rep[0][1]:
                        st.error(f"**Bottleneck:** Options cash ${_cash_now:.2f} "
                                  f"is below the ${OPTIONS_MIN_CASH:.0f} minimum.",
                                  icon="💰")
                    elif not _rep[1][1]:
                        st.warning("**Bottleneck:** VIX is EXTREME, options auto-entry paused.",
                                    icon="⚡")
                    elif not _rep[5][1]:
                        if len(_yf_failed) >= len(OPTIONS_AUTO_WATCHLIST) - 2:
                            st.error(
                                "**Bottleneck:** yfinance failed for nearly all watchlist "
                                "tickers — likely rate-limiting Streamlit Cloud's IP.  "
                                "This is temporary; Alpaca live data isn't affected.  "
                                "The next GitHub Actions monitor cycle should work.",
                                icon="🌐",
                            )
                        else:
                            st.error(
                                f"**Bottleneck:** No watchlist plays affordable on ${_cash_now:.2f}.  "
                                "All 31 tickers in the watchlist need ≥$6 to enter. "
                                "Either reset the options account (Paper Trades tab) or wait for "
                                "existing positions to close.",
                                icon="❌",
                            )
                    elif not _rep[2][1] and not _rep[5][1]:
                        st.warning("**Bottleneck:** No candidates anywhere.", icon="📭")
                    else:
                        st.success(
                            f"✅ All gates show GREEN.  **{len(_affordable)} watchlist tickers** "
                            f"are affordable on your ${_cash_now:.2f} cash.  Auto-entry should "
                            f"trigger on next monitor cycle (within 5 min during market hours).",
                            icon="✅",
                        )
                        # Show specific affordable tickers
                        if _affordable:
                            _afford_str = ", ".join(
                                f"{tk} (${c:.0f})" for tk, c in
                                sorted(_affordable, key=lambda x: x[1])[:12]
                            )
                            st.caption(f"**Cheapest affordable:** {_afford_str}")

        with _od_c2:
            if st.button("🧪 Dry-Run Auto-Entry Now",
                          key="mgmt_dryrun_opts",
                          use_container_width=True,
                          help="Simulate what auto_enter_options would do RIGHT NOW"):
                with st.spinner("Running auto_enter_options() in test mode…"):
                    import io as _io
                    import contextlib as _cl
                    _buf = _io.StringIO()
                    try:
                        import monitor as _mo
                        from trading_scanner import LearningEngine as _LE
                        _le_local = _LE(conn)
                        _vn = ts.get_vix_level()
                        _ses = ts.MarketClock.get_session()
                        with _cl.redirect_stdout(_buf):
                            _mo.auto_enter_options(
                                conn, _vn, _ses,
                                _ses.get("quality", "NORMAL"),
                                risk_mult=1.0,
                                market_regime=None,
                                learning_adjustments=_le_local.get_active_adjustments(),
                            )
                        st.code(_buf.getvalue() or "(no output)", language="text")
                    except Exception as _dre:
                        st.error(f"Dry-run failed: {_dre}")

        # Open options positions table
        if _opx_open:
            st.markdown("**Open Options Positions:**")
            _opt_rows = []
            for p in _opx_open[:10]:
                _opt_rows.append({
                    "Ticker":   p.get("ticker"),
                    "Strategy": p.get("strategy", "?"),
                    "Strike":   f"${float(p.get('strike', 0) or 0):.0f}",
                    "Type":     str(p.get("option_type", "?")).upper(),
                    "Expiry":   str(p.get("expiry", "")),
                    "Entry":    f"${float(p.get('entry_price', 0) or 0):.2f}",
                    "Current":  f"${float(p.get('current_price', 0) or 0):.2f}" if p.get('current_price') else "—",
                    "Unr P&L":  (f"${float(p.get('unrealized_pnl', 0) or 0):+.2f} "
                                   f"({float(p.get('unrealized_pct', 0) or 0):+.1f}%)"
                                   if p.get('unrealized_pnl') is not None else "—"),
                })
            st.dataframe(pd.DataFrame(_opt_rows), hide_index=True,
                          use_container_width=True)

    except Exception as _oxe:
        st.warning(f"Options engine unavailable: {_oxe}")

    st.divider()

    # ── SECTION 3.5 — OPTIONS CALLOUTS (SOCIAL AGGREGATOR) ───────────────────
    st.markdown("### 📢 Options Callouts — Social Watchlist")
    st.caption(
        "Aggregates call/put ideas from StockTwits (incl. trending) + Reddit "
        "(10 subs + the WSB daily-thread comments), snapshots the premium when "
        "called, then tracks forward P&L. **Research only.** Backtest of chasing "
        "social hype was ~zero/negative edge — so a source must EARN trust: it's "
        "only 'actionable' after ≥55% win over ≥20 closed callouts."
    )
    try:
        from options_callouts import OptionsCalloutTracker
        _oct = OptionsCalloutTracker(conn)
        _oc_stats = _oct.get_stats()

        # ── Win-rate gate: per-source proof before trust ────────────────────
        _wr = _oct.get_source_winrates()
        st.markdown("**🚦 Source win-rate gate** "
                    "(forward-measured — callouts aren't trusted until proven):")
        if _wr:
            import pandas as _pdwr
            st.dataframe(_pdwr.DataFrame([{
                "Source": s["source"], "Closed": s["n_closed"],
                "Win%": f"{s['win_rate']:.0f}%", "Avg P&L": f"{s['avg_pnl']:+.0f}%",
                "Status": s["status"],
            } for s in _wr]), use_container_width=True, hide_index=True)
            _act = _oct.get_actionable_callouts(limit=20)
            if _act:
                st.success(f"✅ {len(_act)} ACTIONABLE callout(s) from proven sources",
                           icon="✅")
            else:
                st.info("No source has cleared the win-rate gate yet — all callouts "
                        "are in observation mode (tracked, not actionable). This is "
                        "correct until the data proves a source.", icon="👀")
        else:
            st.caption("No closed callouts yet — sources start in observation. "
                       "The gate populates as callouts resolve.")

        _ocm = st.columns(5)
        _ocm[0].metric("Total Tracked", _oc_stats.get("n_total", 0))
        _ocm[1].metric("Open", _oc_stats.get("n_open", 0))
        _ocm[2].metric("Closed", _oc_stats.get("n_closed", 0))
        _ocm[3].metric("Wins", _oc_stats.get("n_wins", 0))
        _ocm[4].metric("Win Rate", f"{_oc_stats.get('overall_win_rate', 0):.0f}%")

        _oc_btns = st.columns(2)
        with _oc_btns[0]:
            if st.button("📥 Pull Latest Callouts", key="mgmt_oc_pull",
                          use_container_width=True):
                _inc_yt = st.session_state.get("oc_include_yt", False)
                with st.spinner("Fetching social callouts + snapshotting premiums"
                                + (" + YouTube transcripts (slow)" if _inc_yt else "") + "…"):
                    try:
                        _trk = _oct.track_outcomes(max_per_cycle=40)
                        _ing = _oct.ingest_callouts(include_reddit=True,
                                                    include_youtube=_inc_yt)
                        st.success(
                            f"Ingested {_ing.get('new_stored',0)} new "
                            f"(fetched {_ing.get('fetched',0)}, "
                            f"dup {_ing.get('skipped_duplicate',0)}, "
                            f"no-premium {_ing.get('skipped_no_premium',0)}). "
                            f"Tracked {_trk.get('updated',0)}, "
                            f"closed {_trk.get('expired',0)}."
                        )
                        st.rerun()
                    except Exception as _pe:
                        st.error(f"Pull failed: {_pe}")
        with _oc_btns[1]:
            if st.button("🔄 Refresh P&L Only", key="mgmt_oc_track",
                          use_container_width=True):
                with st.spinner("Updating live premiums on open callouts..."):
                    try:
                        _trk = _oct.track_outcomes(max_per_cycle=60)
                        st.success(f"Updated {_trk.get('updated',0)}, "
                                   f"closed {_trk.get('expired',0)}.")
                        st.rerun()
                    except Exception as _te:
                        st.error(f"Refresh failed: {_te}")

        st.checkbox("Include YouTube finance-video transcripts when pulling "
                    "(spoken callouts; slower; may be IP-blocked on cloud)",
                    value=False, key="oc_include_yt")

        # ── Active callouts (live P&L) ───────────────────────────────────────
        _active = _oct.get_active_callouts(limit=40)
        if _active:
            st.markdown("**🟢 Active Callouts (live P&L):**")
            _ac_rows = []
            for c in _active:
                _pnl = float(c.get("pnl_pct", 0) or 0)
                _ac_rows.append({
                    "Ticker":  c.get("ticker", ""),
                    "Type":    str(c.get("option_type", "")).upper(),
                    "Strike":  f"${float(c['strike']):.0f}" if c.get("strike") else "ATM",
                    "Expiry":  str(c.get("expiry", "") or "—"),
                    "Caller":  f"@{c.get('username','?')}",
                    "Source":  c.get("source", ""),
                    "Entry $": f"${float(c.get('entry_premium',0) or 0):.2f}",
                    "Now $":   f"${float(c.get('last_premium',0) or 0):.2f}",
                    "P&L %":   f"{_pnl:+.0f}%",
                })
            st.dataframe(pd.DataFrame(_ac_rows), hide_index=True,
                          use_container_width=True)
        else:
            st.info("No active callouts yet. Click **Pull Latest Callouts** to "
                    "scan StockTwits now (the monitor also does this every cycle).")

        # ── Recent winners + Leaderboard side by side ────────────────────────
        _oc_lr = st.columns(2)
        with _oc_lr[0]:
            st.markdown("**🔥 Recent Winners (last 24h, open & up >10%):**")
            _winners = _oct.get_recent_winners(hours=24, limit=10)
            if _winners:
                _w_rows = [{
                    "Ticker": w.get("ticker", ""),
                    "Type":   str(w.get("option_type", "")).upper(),
                    "P&L %":  f"{float(w.get('pnl_pct',0) or 0):+.0f}%",
                    "Caller": f"@{w.get('username','?')}",
                } for w in _winners]
                st.dataframe(pd.DataFrame(_w_rows), hide_index=True,
                              use_container_width=True)
            else:
                st.caption("No standout winners in the last 24h.")
        with _oc_lr[1]:
            st.markdown("**🏆 Caller Leaderboard (closed callouts):**")
            _lb = _oct.get_leaderboard(min_callouts=3)
            if _lb:
                _lb_rows = [{
                    "Caller":   f"@{l.get('username','?')}",
                    "Source":   l.get("source", ""),
                    "Calls":    l.get("n_total", 0),
                    "Win %":    f"{l.get('win_rate_pct',0):.0f}%",
                    "Avg P&L":  f"{float(l.get('avg_pnl_pct',0) or 0):+.0f}%",
                } for l in _lb[:10]]
                st.dataframe(pd.DataFrame(_lb_rows), hide_index=True,
                              use_container_width=True)
            else:
                st.caption("Leaderboard builds as callouts close (need 3+ "
                           "closed callouts per caller). Give it a few days.")
    except Exception as _oce:
        st.warning(f"Options callouts unavailable: {_oce}")

    st.divider()

    # ── SECTION 4 — DIAGNOSTICS & MANUAL TRIGGERS ────────────────────────────
    st.markdown("### 🩺 Engine Diagnostics & Manual Triggers")
    _diag_cols = st.columns(4)

    with _diag_cols[0]:
        st.markdown("**Stops/Targets**")
        st.caption("Manual check_stops_and_targets")
        if st.button("🔄 Check Now", key="mgmt_stops", use_container_width=True):
            with st.spinner("..."):
                try:
                    _closed = _mgmt_paper.check_stops_and_targets()
                    st.success(f"Closed {len(_closed)}" if _closed else "None breached")
                except Exception as _x:
                    st.error(f"{_x}")

    with _diag_cols[1]:
        st.markdown("**Options Expiry**")
        st.caption("Close expired worthless contracts")
        if st.button("⏰ Run", key="mgmt_exp", use_container_width=True):
            try:
                _ex = _mgmt_ops.expire_check()
                st.success(f"Expired {len(_ex)}")
            except Exception as _x:
                st.error(f"{_x}")

    with _diag_cols[2]:
        st.markdown("**News Pull**")
        st.caption("Fetch + classify headlines")
        if st.button("📰 Pull", key="mgmt_npull", use_container_width=True):
            try:
                from news_agent import NewsAgent as _NA2
                with st.spinner("..."):
                    _r = _NA2(conn, ai_analyst=_AI).run_cycle()
                    st.success(f"+{_r['new']} new")
            except Exception as _x:
                st.error(f"{_x}")

    with _diag_cols[3]:
        st.markdown("**Pipeline Test**")
        st.caption("End-to-end dry-run on SPY")
        if st.button("🧪 Test", key="mgmt_pipe", use_container_width=True):
            with st.spinner("Running pipeline on SPY..."):
                import time as _tm
                _rep = []
                _tk = "SPY"
                _t = _tm.time()
                try:
                    _p_test = _fetch_live_prices((_tk,))
                    _rep.append(("Live price", bool(_p_test.get(_tk, 0) > 0),
                                 f"${_p_test.get(_tk, 0):.2f}", _tm.time()-_t))
                except Exception as _e:
                    _rep.append(("Live price", False, str(_e), 0))
                _t = _tm.time()
                try:
                    _w = ts.detect_wyckoff_phase(_tk)
                    _rep.append(("Wyckoff", True,
                                 f"{_w.get('phase','?')} ({_w.get('confidence',0)}%)", _tm.time()-_t))
                except Exception as _e:
                    _rep.append(("Wyckoff", False, str(_e), 0))
                _t = _tm.time()
                try:
                    _mtf = ts.confirm_multi_timeframe(_tk, "bullish")
                    _rep.append(("Multi-TF", True,
                                 f"{_mtf.get('aligned',0)}/3 ({_mtf.get('grade','?')})", _tm.time()-_t))
                except Exception as _e:
                    _rep.append(("Multi-TF", False, str(_e), 0))
                _t = _tm.time()
                try:
                    _m = ts.compute_master_score(_tk, None, "bullish", conn, skip_slow_checks=False)
                    _rep.append(("Master Score", True,
                                 f"{_m.get('score',0)}/100 ({_m.get('grade','?')}) {_m.get('decision','?')}",
                                 _tm.time()-_t))
                except Exception as _e:
                    _rep.append(("Master Score", False, str(_e), 0))
                _t = _tm.time()
                try:
                    _sugs = ts.OptionsStrategyEngine().suggest(_tk, "bullish")
                    _rep.append(("Options strat", len(_sugs) > 0,
                                 f"{len(_sugs)} suggestions", _tm.time()-_t))
                except Exception as _e:
                    _rep.append(("Options strat", False, str(_e), 0))

                for label, ok, detail, elapsed in _rep:
                    icon = "✅" if ok else "🛑"
                    st.markdown(f"{icon} **{label}** — {detail}  "
                               f"<span style='color:#8b949e;font-size:0.8em'>({elapsed*1000:.0f}ms)</span>",
                               unsafe_allow_html=True)
                if all(r[1] for r in _rep):
                    st.success("✅ All pipeline stages functional", icon="✅")

    st.divider()

    # ── SECTION 5 — CONFIGURATION ────────────────────────────────────────────
    with st.expander("⚙️ Configuration & Tunable Parameters", expanded=False):
        _cfg = st.columns(2)
        with _cfg[0]:
            st.markdown("**Learning-controlled (DB)**")
            try:
                _ac = _mgmt_le.get_active_adjustments()
                st.markdown(
                    f"- Min Master Score: <b>{_ac['min_master_score']:.0f}</b><br>"
                    f"- Size Cap: <b>{_ac['size_multiplier_cap']:.2f}×</b><br>"
                    f"- Sector Blacklist: <b>{len(_ac['sector_blacklist'])}</b><br>"
                    f"- Pattern Blacklist: <b>{len(_ac['pattern_blacklist'])}</b><br>"
                    f"- Wyckoff Blacklist: <b>{len(_ac['wyckoff_blacklist'])}</b><br>"
                    f"- Iteration: <b>#{_ac['learning_iteration']}</b>",
                    unsafe_allow_html=True)
            except Exception:
                pass
        with _cfg[1]:
            st.markdown("**Hard-coded (monitor.py)**")
            try:
                from monitor import (STOCK_MIN_SCORE, STOCK_MIN_PROB,
                                     OPTIONS_MIN_SCORE, OPTIONS_MIN_PROB,
                                     AUTO_MAX_ENTRIES_PER_RUN, STOCK_CHASE_LIMIT_PCT)
                st.markdown(
                    f"- Stock pre-filter score: <b>{STOCK_MIN_SCORE}</b><br>"
                    f"- Stock pre-filter prob: <b>{STOCK_MIN_PROB}%</b><br>"
                    f"- Options pre-filter score: <b>{OPTIONS_MIN_SCORE}</b><br>"
                    f"- Options pre-filter prob: <b>{OPTIONS_MIN_PROB}%</b><br>"
                    f"- Max new entries/cycle: <b>{AUTO_MAX_ENTRIES_PER_RUN}</b><br>"
                    f"- Chase limit: <b>{STOCK_CHASE_LIMIT_PCT}%</b>",
                    unsafe_allow_html=True)
            except Exception:
                pass
        st.markdown("**Drawdown thresholds (LearningEngine)**")
        st.markdown(
            f"- Punishment Mode: <b>≥ {ts.LearningEngine.PUNISHMENT_THRESHOLD}%</b><br>"
            f"- Hard Reset: <b>≥ {ts.LearningEngine.RESET_THRESHOLD}%</b><br>"
            f"- Recovery band: within <b>{ts.LearningEngine.RECOVERY_BAND_PCT}%</b> of peak",
            unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 0.5 — WHALE WATCH ("follow the money" idea generator)
#   Public smart-money signals — Congressional disclosures, SEC 13D/13G activist
#   filings, Form 4 insider buys, federal contracts — aggregated into one ranked
#   watchlist with concrete entry/stop/target. Forward scorecard measures
#   whether the picks actually beat SPY.
# ──────────────────────────────────────────────────────────────────────────────
with tab_analyst:
    conn, _ma_mode = get_db()
    st.markdown("## 🏛 Market Analyst — Institutional Briefing")
    st.caption(
        "Thinks like a desk analyst, not an indicator chaser: multi-timeframe "
        "structure (weekly/daily/intraday), market internals + breadth, regime "
        "classification, and a weighted bull/bear/neutral PROBABILITY view — "
        "never a binary 'buy'. Capital preservation and probabilities over "
        "trade frequency."
    )

    if st.button("🏛 Generate Briefing", key="ma_btn", type="primary"):
        try:
            from market_analyst import MarketAnalyst
            with st.spinner("Analyzing structure, internals, breadth, regime… (~60s)"):
                _brief = MarketAnalyst(conn).generate_briefing()
            st.session_state["_market_brief"] = _brief
        except Exception as _mae:
            st.error(f"Briefing failed: {_mae}")

    _brief = st.session_state.get("_market_brief")
    if not _brief:
        st.info("Hit **Generate Briefing** for the full professional market read. "
                "Also delivered to Telegram each morning before the open.", icon="🏛")
    else:
        # ── Headline: probabilities + bias + confidence ──────────────────────
        _bias = _brief["bias"]
        _bias_color = {"Bullish": "#3fb950", "Bearish": "#f85149",
                       "Neutral": "#e3b341"}.get(_bias, "#8b949e")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Bullish", f"{_brief['prob_bull']:.0f}%")
        c2.metric("🔴 Bearish", f"{_brief['prob_bear']:.0f}%")
        c3.metric("⚪ Neutral", f"{_brief['prob_neutral']:.0f}%")
        c4.metric("Confidence", f"{_brief['confidence']:.0f}%")
        st.markdown(
            f"<div style='background:#161b22;border-left:6px solid {_bias_color};"
            f"border-radius:8px;padding:12px 16px;margin:8px 0;color:#c9d1d9'>"
            f"<span style='font-size:1.2em'><b>Market Bias: "
            f"<span style='color:{_bias_color}'>{_bias.upper()}</span></b></span>  ·  "
            f"Regime: <b>{_brief['market_regime']['primary']}</b>  ·  "
            f"Risk: <b>{_brief['risk']['level']}</b><br>"
            f"<span style='color:#8b949e;font-size:0.9em'>"
            f"Regime tags: {' · '.join(_brief['market_regime']['tags'])}</span>"
            f"</div>", unsafe_allow_html=True)

        # ── Multi-timeframe structure ────────────────────────────────────────
        st.markdown("### 📐 Multi-Timeframe Structure")
        _sc = st.columns(3)
        for _i, _tf in enumerate(["weekly", "daily", "intraday"]):
            s = _brief["structure"][_tf]
            _tc = ("#3fb950" if "Bull" in s["trend"] else
                   "#f85149" if "Bear" in s["trend"] else "#e3b341")
            with _sc[_i]:
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;"
                    f"border-radius:8px;padding:10px 12px'>"
                    f"<b>{s['timeframe']}</b><br>"
                    f"<span style='color:{_tc};font-weight:700'>{s['trend']}</span><br>"
                    f"<span style='font-size:0.8em;color:#8b949e'>"
                    f"RSI {s.get('rsi',0):.0f} · close {s.get('close_in_range_pct',0):.0f}% "
                    f"of range · vol {s.get('volume_trend','?')}<br>"
                    f"{'HH ' if s.get('hh') else ''}{'HL ' if s.get('hl') else ''}"
                    f"{'LH ' if s.get('lh') else ''}{'LL' if s.get('ll') else ''}</span>"
                    f"</div>", unsafe_allow_html=True)

        # ── Internals + key levels ───────────────────────────────────────────
        _ic1, _ic2 = st.columns(2)
        with _ic1:
            st.markdown("### 🔬 Market Internals")
            _intr = _brief["internals"]; _br = _intr["breadth"]
            _vix_chg = _intr.get("vix_20d_chg")
            _vix_chg_txt = f" ({_vix_chg:+.0f}% 20d)" if _vix_chg is not None else ""
            st.markdown(
                f"- **Breadth:** {_br.get('pct_above_50','?')}% above 50-SMA · "
                f"{_br.get('pct_above_200','?')}% above 200-SMA\n"
                f"- **VIX:** {_intr.get('vix',0):.1f}{_vix_chg_txt}\n"
                f"- **Breadth health (RSP/SPY):** {_intr.get('equal_vs_cap_20d','?')}% (20d)\n"
                f"- **Risk appetite (XLY/XLP):** {_intr.get('risk_appetite_20d','?')}% (20d)\n"
                f"- **Credit (HYG/TLT):** {_intr.get('credit_20d','?')}% (20d)")
        with _ic2:
            st.markdown("### 🎯 Key Levels")
            _lv = _brief["key_levels"]
            st.markdown(
                f"- **Prev week:** H ${_lv.get('prev_week_high','?')} · "
                f"L ${_lv.get('prev_week_low','?')}\n"
                f"- **Prev day:** H ${_lv.get('prev_day_high','?')} · "
                f"L ${_lv.get('prev_day_low','?')}\n"
                f"- **Resistance:** ${_lv.get('near_resistance','?')} (near) · "
                f"${_lv.get('resistance','?')} (60d)\n"
                f"- **Support:** ${_lv.get('near_support','?')} (near) · "
                f"${_lv.get('support','?')} (60d)\n"
                f"- **Last:** ${_lv.get('last','?')}")

        # ── Recommended strategy + opportunities ─────────────────────────────
        st.markdown("### ♟ Recommended Strategy")
        st.info(_brief["recommended_strategy"]["summary"], icon="♟")
        st.caption("Active engines for this regime: " +
                   " · ".join(_brief["recommended_strategy"]["engines"]))
        if _brief["opportunities"]:
            st.markdown("**Trade opportunities:**")
            for o in _brief["opportunities"]:
                st.markdown(f"- {o}")

        # ── Reasons + invalidation ───────────────────────────────────────────
        _rc1, _rc2 = st.columns(2)
        with _rc1:
            st.markdown("### ✅ Reasons Supporting")
            for r in _brief["reasons"]:
                st.markdown(f"- {r}")
        with _rc2:
            st.markdown("### ⚠️ Would Invalidate This View")
            for r in _brief["invalidation"]:
                st.markdown(f"- {r}")

        st.caption(f"Generated {_brief['as_of'][:19]} UTC · symbol {_brief['symbol']} · "
                   f"This is probabilistic analysis, not a directional guarantee.")

    # ── MACRO ENGINE — economic calendar + event risk + interpretation ───────
    st.divider()
    st.markdown("### 🌐 Macro Engine — What Moves the Market")
    st.caption("The 'human logic' layer: macro data (jobs/CPI/PCE/FOMC) drives the "
               "market more than charts. Knows what's coming, flags event risk so "
               "you don't get blindsided, and explains the likely driver of recent moves.")
    try:
        from macro_engine import upcoming_events as _ue, event_risk as _erk, explain_recent as _exr
        _er = _erk()
        _erc = {"HIGH": "#f85149", "ELEVATED": "#e3b341"}.get(_er["level"], "#3fb950")
        st.markdown(
            f"<div style='background:#161b22;border-left:5px solid {_erc};"
            f"border-radius:6px;padding:10px 14px;margin:6px 0;color:#c9d1d9'>"
            f"<b>Event risk: {_er['level']}</b> · next: {_er.get('next_event','—')} "
            f"in {_er.get('days_away','?')}d<br>"
            f"<span style='color:#8b949e;font-size:0.9em'>{_er['advice']}</span></div>",
            unsafe_allow_html=True)
        # Why did the market move recently
        try:
            _vix_now = None; _spy5 = None
            import yfinance as _yfm
            _spydf = _yfm.download("SPY", period="10d", progress=False, auto_adjust=True)
            if _spydf is not None and not _spydf.empty:
                if hasattr(_spydf.columns, "get_level_values"):
                    _spydf.columns = _spydf.columns.get_level_values(0)
                _cl = _spydf["Close"].dropna()
                if len(_cl) >= 6:
                    _spy5 = float(_cl.iloc[-1]/_cl.iloc[-6]-1)*100
            try:
                _vix_now = float(_yfm.Ticker("^VIX").fast_info["last_price"])
            except Exception:
                pass
            st.info("🔍 " + _exr(vix=_vix_now, spy_5d_pct=_spy5), icon="🧭")
        except Exception:
            pass
        # Calendar
        _evs = _ue(days_ahead=21)
        if _evs:
            import pandas as _pdm
            st.dataframe(_pdm.DataFrame([{
                "Date": e["date"], "In": f"{e['days_away']}d",
                "Event": e["name"] + (" ≈" if e.get("approx_date") else ""),
                "Impact": "🔴"*e["impact"] if e["impact"]>=5 else "🟠"*e["impact"],
                "Time": e["time"],
            } for e in _evs]), use_container_width=True, hide_index=True)
            st.caption("≈ = approximate date (agencies set exact day; verify). "
                       "Impact 🔴🔴🔴🔴🔴 = can move the whole market.")
    except Exception as _me:
        st.caption(f"(macro engine unavailable: {_me})")

    # ── HIDDEN GEMS — strong fundamentals, weak technicals (the SOFI profile) ─
    st.divider()
    st.markdown("### 💎 Hidden Gems — Strong Fundamentals, Weak Technicals")
    st.caption("The SOFI profile: great business, broken chart. NOT buys yet — "
               "prime reversal-watch candidates. ✅ 'reversal-ready' = reclaimed "
               "the 50-SMA (the validated turn trigger fired).")
    if st.button("💎 Scan Hidden Gems", key="gems_btn", type="primary"):
        try:
            from sector_analysis import find_hidden_gems
            _gp = st.progress(0.0, text="Scanning fundamentals + technicals…")
            def _g_cb(i, n, t):
                _gp.progress(min(i/max(n,1), 1.0), text=f"{t} ({i}/{n})")
            with st.spinner("Finding strong-fundamentals / weak-technicals names…"):
                _gems = find_hidden_gems(progress=_g_cb)
            _gp.empty()
            st.session_state["_gems"] = _gems
        except Exception as _ge:
            st.error(f"Scan failed: {_ge}")
    _gems = st.session_state.get("_gems")
    if _gems is None:
        st.info("Hit **Scan Hidden Gems** to find SOFI-type names — strong "
                "fundamentals trading at weak technicals.", icon="💎")
    elif not _gems:
        st.info("No clean hidden-gem candidates right now.", icon="📭")
    else:
        import pandas as _pdg
        st.success(f"{len(_gems)} hidden gem(s)", icon="💎")
        st.dataframe(_pdg.DataFrame([{
            "Ticker": g["ticker"], "Sector": (g["sector"] or "")[:18],
            "Price": f"${g['price']:.2f}", "Drawdown": f"{g['drawdown_pct']:.0f}%",
            "Fwd P/E": f"{g['pe_fwd']:.0f}" if g.get("pe_fwd") else "—",
            "EPS g": f"{(g['eps_growth_qoq'] or 0)*100:+.0f}%",
            "Rev g": f"{(g['rev_growth'] or 0)*100:+.0f}%",
            "50SMA": f"${g['sma50']:.2f}", "200SMA": f"${g['sma200']:.2f}",
            "Status": "✅ reversal-ready" if g["reversal_ready"] else "👀 basing/down",
        } for g in _gems]), use_container_width=True, hide_index=True)
        st.caption("Watch the 👀 names for the 50-SMA reclaim → they flip to "
                   "✅ reversal-ready, which is the validated entry trigger.")

    # ── REVERSAL FINDER — downtrend -> base -> uptrend watchlist ──────────────
    st.divider()
    st.markdown("### 🔄 Reversal Finder — Downtrend → Base → Uptrend")
    st.caption(
        "Finds stocks turning the corner: fell hard, based tightly without new "
        "lows, then reclaimed a flattening/rising 50-SMA. Backtest (88 names, 5y): "
        "the 50-SMA-reclaim version wins 62.6% over 60 days (median +5.1%) — but "
        "the naive base-BREAKOUT version was REJECTED (it underperforms). "
        "Watchlist / idea generator, not an auto-trader — the edge is modest."
    )
    if st.button("🔄 Scan Reversals", key="rev_btn", type="primary"):
        try:
            from reversal_finder import ReversalFinder
            _rp = st.progress(0.0, text="Scanning for reversals…")
            def _rev_cb(i, n, t):
                _rp.progress(min(i/max(n,1), 1.0), text=f"{t} ({i}/{n})")
            with st.spinner("Scanning liquid universe for downtrend→base→uptrend…"):
                _rev_res = ReversalFinder(conn).scan(progress=_rev_cb)
            _rp.empty()
            st.session_state["_rev_res"] = _rev_res
        except Exception as _re:
            st.error(f"Reversal scan failed: {_re}")

    _rev_res = st.session_state.get("_rev_res")
    if _rev_res is None:
        st.info("Hit **Scan Reversals** to find stocks turning from downtrend to "
                "uptrend. Staged: TRIGGERED (entry), BASING (watch), EXTENDED (chase risk).",
                icon="🔄")
    elif not _rev_res:
        st.info("No reversal setups in the universe right now.", icon="📭")
    else:
        _stage_meta = {"TRIGGERED": ("#3fb950", "🎯", "Just reclaimed 50-SMA — the validated entry"),
                       "BASING":    ("#58a6ff", "👀", "Based, downtrend arrested, not yet turned — watch"),
                       "EXTENDED":  ("#e3b341", "⚠️", "Already ran post-trigger — chase risk")}
        for _stage in ["TRIGGERED", "BASING", "EXTENDED"]:
            _rows = [r for r in _rev_res if r["stage"] == _stage]
            if not _rows:
                continue
            _col, _ic, _desc = _stage_meta[_stage]
            st.markdown(f"<div style='color:{_col};font-weight:700;margin-top:8px'>"
                        f"{_ic} {_stage} ({len(_rows)}) — <span style='font-weight:400;"
                        f"color:#8b949e'>{_desc}</span></div>", unsafe_allow_html=True)
            import pandas as _pdr
            df = _pdr.DataFrame([{
                "Ticker": r["ticker"],
                "Price": f"${r['price']:.2f}",
                "Drawdown": f"{r['drawdown_pct']:.0f}%",
                "Base range": f"{r['base_range_pct']:.0f}%",
                "200-slope": f"{r['slope200_20d']:+.1f}%",
                "Improving": "✓" if r["slope_improving"] else "",
                "Entry": f"${r['entry']:.2f}",
                "Stop": f"${r['stop']:.2f}",
                "Target": f"${r['target']:.2f}",
            } for r in _rows])
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("Discipline: act on TRIGGERED (50-SMA reclaim), watch BASING for "
                   "the trigger, avoid chasing EXTENDED. Stop below the base.")


with tab_whale:
    conn, _ww_mode = get_db()

    # ── VIP FEED — Trump / Fed instant-alert section ──────────────────────────
    st.markdown("## 📣 VIP Feed — Trump · Fed (instant Telegram alerts)")
    st.caption(
        "Polls Trump's Truth Social and the Federal Reserve press feed every "
        "monitor cycle. When a post mentions a stock (cashtag or known company "
        "name) — or any Fed release lands — a Telegram alert fires in real time. "
        "All posts persist below so you can scroll the recent feed and see "
        "which were market-relevant."
    )

    _vc1, _vc2, _vc3 = st.columns([2, 1, 1])
    with _vc2:
        _vip_tonly = st.toggle("Only posts with tickers", value=False, key="vip_tonly")
    with _vc3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _vip_poll = st.button("🔄 Poll Now", key="vip_poll",
                              type="primary", use_container_width=True)
    with _vc1:
        st.info(
            "Auto-polls every cycle. Hit **Poll Now** to fetch immediately. "
            "Telegram alerts fire for: (a) any Trump post that mentions a stock, "
            "(b) every Fed press release.",
            icon="📡",
        )

    if _vip_poll:
        try:
            from vip_news_monitor import VipNewsMonitor
            # Re-use monitor's send_telegram via env if configured
            def _vip_tg(msg):
                tok  = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
                chat = os.environ.get("TELEGRAM_CHAT_ID",   "").strip()
                if not tok or not chat:
                    return False
                try:
                    import requests as _rq
                    r = _rq.post(
                        f"https://api.telegram.org/bot{tok}/sendMessage",
                        data={"chat_id": chat, "text": msg,
                              "parse_mode": "HTML",
                              "disable_web_page_preview": True},
                        timeout=10)
                    return r.status_code == 200
                except Exception:
                    return False
            with st.spinner("Polling Trump + Fed feeds…"):
                _rep = VipNewsMonitor(conn).run_cycle(telegram_sender=_vip_tg)
            for h, s in _rep.items():
                st.success(f"{h}: fetched {s['fetched']} · "
                           f"new {s['new']} · alerted {s['alerted']}", icon="📡")
        except Exception as _vpe:
            st.error(f"Poll failed: {_vpe}")

    try:
        from vip_news_monitor import VipNewsMonitor as _VNM
        _vip_recent = _VNM(conn).get_recent(limit=80, ticker_only=_vip_tonly)
    except Exception as _vre:
        _vip_recent = []
        st.warning(f"VIP feed unavailable: {_vre}")

    if _vip_recent:
        for p in _vip_recent[:40]:
            _sent_emoji = {"bullish": "🟢", "bearish": "🔴",
                           "neutral": "⚪"}.get(p["sentiment"], "⚪")
            _src_emoji  = "🚨" if p["vip_handle"] == "DJT" else "🏦"
            _alert_tag  = "🔔" if p.get("alerted") else "·"
            _tickers    = " ".join(f"`${t}`" for t in p["tickers"]) or "—"
            with st.container():
                st.markdown(
                    f"{_src_emoji} **{p['vip_name']}**  {_sent_emoji} "
                    f"{_alert_tag}  · {p['posted_at'] or p['fetched_at'][:19]}",
                )
                _snippet = (p["title"] or p["text"])[:380]
                st.markdown(f"<div style='color:#c9d1d9;font-size:0.92em;"
                            f"padding:4px 0'>{_snippet}</div>",
                            unsafe_allow_html=True)
                cols = st.columns([3, 2])
                cols[0].markdown(f"Tickers: {_tickers}")
                if p["url"]:
                    cols[1].markdown(f"[Open ↗]({p['url']})")
                st.markdown("<hr style='margin:6px 0;border:0;"
                            "border-top:1px solid #21262d'>", unsafe_allow_html=True)
    else:
        st.info("No VIP posts yet — hit **Poll Now** or wait for the next monitor "
                "cycle.", icon="📭")

    st.divider()

    # ── PEAD SCANNER (the actual earnings edge) ────────────────────────────────
    st.markdown("## 📅 PEAD Scanner — Post-Earnings Drift")
    st.caption(
        "Trades AFTER earnings, not INTO them. PEAD (Post-Earnings Announcement "
        "Drift) is the academically-documented anomaly where stocks that BEAT + "
        "GAPPED UP keep drifting up for 2–12 weeks. This is the safe, validated "
        "earnings edge — none of the IV-crush risk of buying calls into the print."
    )

    _pc1, _pc2 = st.columns([3, 1])
    with _pc2:
        _pead_btn = st.button("📅 Refresh PEAD", key="pead_btn",
                              type="primary", use_container_width=True)
    with _pc1:
        st.info(
            "Refreshes the earnings calendar for the liquid universe (~125 names) "
            "then scores each on PEAD criteria: beat magnitude × post-earnings "
            "gap × trend-intact × days-since (sweet spot 7–21d). Takes a couple "
            "of minutes on first run; cached after that.",
            icon="ℹ️",
        )

    if _pead_btn:
        try:
            from earnings_engine import PEADScanner
            _pp = st.progress(0.0, text="Refreshing earnings calendar…")
            def _pead_cb(i, n, t):
                _pp.progress(min(i / max(n, 1), 1.0), text=f"{t} ({i}/{n})")
            with st.spinner("Scanning PEAD setups…"):
                _pead_rows = PEADScanner(conn).scan(progress=_pead_cb)
            _pp.empty()
            st.success(f"PEAD scan complete: {len(_pead_rows)} candidate(s)", icon="📅")
        except Exception as _pee:
            st.error(f"PEAD scan failed: {_pee}")

    try:
        from earnings_engine import PEADScanner as _PS
        _pead_show = _PS(conn).get_latest()
    except Exception:
        _pead_show = []

    if _pead_show:
        import pandas as _pdpe
        df = _pdpe.DataFrame([{
            "Ticker":   r["ticker"],
            "Score":    r["score"],
            "Beat":     f"{r['surprise_pct']:+.1f}%",
            "Gap":      f"{r['gap_pct']:+.1f}%",
            "Days":     r["days_since"],
            "Trend":    "✓ intact" if r["trend_intact"] else "wobbly",
            "Entry":    f"${r['entry_now']:.2f}",
            "Stop":     f"${r['stop']:.2f}",
            "Target":   f"${r['target']:.2f}",
        } for r in _pead_show])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("Score ≥ 80 = strong PEAD setup · 60–79 = solid · 50–59 = marginal. "
                   "Higher = better confluence of beat, gap, trend, and timing.")
    else:
        st.info("No PEAD candidates cached. Hit **Refresh PEAD** to scan.", icon="📭")

    st.divider()

    st.markdown("## 🐋 Whale Watch — Follow the Public Smart Money")
    st.caption(
        "Legal analog to insider info: stocks where smart money has PUBLICLY shown "
        "their hand — Congressional STOCK-Act disclosures, SEC 13D/13G activist "
        "filings, Form 4 insider open-market buys, federal contract awards. "
        "Information is public; the edge is in aggregating it and acting before "
        "the broader market does. Forward scorecard below tells you whether it "
        "actually works — don't trust the thesis, trust the data."
    )

    _wc1, _wc2, _wc3 = st.columns([2, 1, 1])
    with _wc2:
        _ww_min = st.number_input("Min whale score", 0, 100, 20, key="ww_min")
    with _wc3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _ww_run = st.button("🐋 Refresh Watchlist", key="ww_refresh",
                            type="primary", use_container_width=True)
    with _wc1:
        st.info(
            "Refreshing scans ~125 liquid US names through 4 whale sources "
            "(Quiver, SEC EDGAR, USASpending, StockTwits). Takes a couple of "
            "minutes on a cold cache; subsequent runs are much faster.",
            icon="ℹ️",
        )

    if _ww_run:
        try:
            from whale_watch import WhaleWatchlist
            _wp = st.progress(0.0, text="Scanning whale sources…")
            def _ww_cb(i, n, t):
                _wp.progress(min(i / max(n, 1), 1.0),
                             text=f"{t}  ({i}/{n})")
            with st.spinner("Aggregating whale signals — this can take a few minutes…"):
                wl = WhaleWatchlist(conn)
                rows = wl.build(min_score=int(_ww_min), progress=_ww_cb)
                wl.persist(rows)
                wl.update_outcomes()
            _wp.empty()
            st.success(f"Watchlist refreshed: {len(rows)} names with whale signal ≥ {_ww_min}",
                       icon="🐋")
        except Exception as _wwe:
            st.error(f"Watchlist refresh failed: {_wwe}")

    # ── Current watchlist ─────────────────────────────────────────────────────
    try:
        from whale_watch import WhaleWatchlist
        _wl = WhaleWatchlist(conn)
        _rows = _wl.get_latest()
        _outc = _wl.get_outcomes()
    except Exception as _wle:
        _rows, _outc = [], {"picks": [], "summary": {}}
        st.warning(f"Watchlist unavailable: {_wle}")

    st.markdown("### 📋 Current Watchlist")
    if _rows:
        import pandas as _pdw
        df = _pdw.DataFrame([{
            "Ticker":     r["ticker"],
            "Score":      f"{r['whale_score']}/100",
            "Key signal": r["key_signal"][:60],
            "Price":      f"${r['price_now']:.2f}",
            "Entry now":  f"${r['entry_now']:.2f}",
            "Pullback":   f"${r['entry_pullback']:.2f}",
            "Stop":       f"${r['stop']:.2f}",
            "Target":     f"${r['target']:.2f}",
            "Risk":       f"{r['risk_pct']:.1f}%",
            "Reward":     f"{r['reward_pct']:.1f}%",
            "R:R":        f"{r['rr']:.1f}",
        } for r in _rows])
        st.dataframe(df, use_container_width=True, hide_index=True, height=420)

        # Per-pick detail expanders
        st.markdown("### 🔎 Per-Pick Details")
        for r in _rows[:15]:
            with st.expander(
                f"🐋 **{r['ticker']}** — score {r['whale_score']}/100  ·  "
                f"entry ${r['entry_now']:.2f}  stop ${r['stop']:.2f}  "
                f"target ${r['target']:.2f}  (R:R {r['rr']:.1f})",
                expanded=False,
            ):
                _flags = [f for f in (r.get("flags") or []) if f]
                if _flags:
                    for f in _flags:
                        st.markdown(f"- {f}")
                else:
                    st.caption("No detailed flags captured this run.")
                st.markdown(
                    f"<small>"
                    f"<b>Entry now:</b> ${r['entry_now']:.2f}   |   "
                    f"<b>Pullback entry:</b> ${r['entry_pullback']:.2f} (EMA20 / 50-SMA)   |   "
                    f"<b>Stop:</b> ${r['stop']:.2f} (1.5×ATR)   |   "
                    f"<b>Target:</b> ${r['target']:.2f} (recent high or +3×ATR)"
                    f"</small>", unsafe_allow_html=True,
                )
    else:
        st.info("Empty watchlist. Hit **Refresh Watchlist** to populate it. "
                "First run takes ~2 min while the whale-source caches warm up.",
                icon="📭")

    st.divider()

    # ── Forward scorecard ─────────────────────────────────────────────────────
    st.markdown("### 📊 Forward Scorecard — Whale Picks vs SPY")
    _summ = _outc.get("summary") or {}
    _picks = _outc.get("picks") or []
    if _summ:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Picks tracked", _summ["n_picks"])
        c2.metric("Avg pick return", f"{_summ['avg_pick_return']:+.2f}%")
        c3.metric("Avg SPY return",  f"{_summ['avg_spy_return']:+.2f}%")
        c4.metric("Alpha vs SPY",    f"{_summ['avg_alpha']:+.2f}%",
                  delta=(f"{_summ['win_rate']:.0f}% beat SPY"))
        st.caption("Each pick's entry was snapped at first detection; SPY is "
                   "snapped at the same moment. Alpha = pick return − SPY return.")
        import pandas as _pdw2
        df2 = _pdw2.DataFrame([{
            "Ticker":   p["ticker"],
            "Detected": p["detected_at"],
            "Days":     p["days"],
            "Entry":    f"${p['entry']:.2f}",
            "Last":     f"${p['last']:.2f}",
            "Pick %":   f"{p['ticker_ret']:+.2f}%",
            "SPY %":    f"{p['spy_ret']:+.2f}%",
            "Alpha":    f"{p['alpha']:+.2f}%",
        } for p in _picks])
        st.dataframe(df2, use_container_width=True, hide_index=True)
    else:
        st.info("No picks tracked yet — refresh the watchlist once and the "
                "forward scorecard starts accumulating from that snapshot.",
                icon="⏳")

    st.markdown(
        "<small style='color:#8b949e'>"
        "Honest caveat: post-STOCK-Act studies are MIXED on whether "
        "Congressional-trade-following still beats the market; 13D activist "
        "filings and Form 4 insider buys have firmer documented edges. The "
        "scorecard above is the only thing that settles it for your bot."
        "</small>", unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PORTFOLIO DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
with tab_dash:
    conn, _mode = get_db()

    if _mode == "postgres":
        st.success("Connected to Supabase — portfolio history persists across devices.", icon="🗄️")
    elif _mode == "memory":
        st.warning(
            "**Cloud mode:** Portfolio history resets on app restart. "
            "Add Supabase credentials to Streamlit secrets to persist data.",
            icon="☁️",
        )

    tracker  = ts.PortfolioTracker(conn)
    analyzer = ts.PerformanceAnalyzer(conn)
    s        = analyzer.summary()
    port     = tracker.get_portfolio_summary()
    stk      = analyzer.get_streak()
    spy      = analyzer.benchmark_vs_spy()

    st.markdown('<div class="section-hdr">Overall Performance</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Total Calls",  str(s["total"] + s["pending"]), "blue")
    with c2:
        wr   = s["win_rate"]
        col  = "green" if wr >= 55 else ("yellow" if wr >= 45 else "red")
        kpi("Win Rate", f"{wr:.1f}%", col)
    with c3:
        pf   = s["profit_factor"]
        col  = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
        kpi("Profit Factor", f"{pf:.2f}x", col)
    with c4:
        exp  = s["expectancy"]
        kpi("Expectancy", f"{exp:+.1f}%", "green" if exp > 0 else "red")
    with c5:
        alp  = spy.get("avg_alpha", 0)
        kpi("vs SPY α", f"{alp:+.1f}%" if spy.get("n") else "N/A",
            "green" if alp > 0 else "red")

    st.markdown("<br>", unsafe_allow_html=True)

    # W/L/BE counts + streak
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Win / Loss / Break-even</div>'
            f'<div style="font-size:1.3rem;font-weight:700;margin-top:6px;">'
            f'<span class="kpi-green">{s["wins"]}</span> / '
            f'<span class="kpi-red">{s["losses"]}</span> / '
            f'<span class="kpi-yellow">{s["breakeven"]}</span></div></div>',
            unsafe_allow_html=True)
    with c2:
        icon = "🟢" if stk["direction"] == "WIN" else "🔴"
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Current Streak</div>'
            f'<div style="font-size:1.3rem;font-weight:700;margin-top:6px;">'
            f'{icon} {stk["current"]} '
            f'{"wins" if stk["direction"] == "WIN" else "losses"}</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Best / Worst streak</div>'
            f'<div style="font-size:1.3rem;font-weight:700;margin-top:6px;">'
            f'<span class="kpi-green">{stk["longest_win"]}W</span> / '
            f'<span class="kpi-red">{stk["longest_loss"]}L</span></div></div>',
            unsafe_allow_html=True)
    with c4:
        rp  = port["realized_pnl"]
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Realized P&L (virtual)</div>'
            f'<div style="font-size:1.3rem;font-weight:700;margin-top:6px;'
            f'color:{"#3fb950" if rp >= 0 else "#f85149"};">'
            f'{"+" if rp >= 0 else ""}${rp:,.0f}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Open positions table
    st.markdown('<div class="section-hdr">Open Virtual Positions</div>',
                unsafe_allow_html=True)

    # ── Refresh control: update_calls() fetches live prices for all pending
    # virtual positions and recomputes current_price / current_pct / outcomes.
    # Cached for 5 min so re-renders don't trigger 104 yfinance calls.
    _vp_c1, _vp_c2, _vp_c3 = st.columns([2, 1, 1])
    with _vp_c1:
        st.caption("Live P&L tracking on every breakout call.  "
                   "Click 🔄 to fetch fresh prices for all open positions.")
    with _vp_c2:
        _vp_last = st.session_state.get("virtual_pos_refreshed_at", "Never")
        st.caption(f"Last refresh: **{_vp_last}**")
    with _vp_c3:
        if st.button("🔄 Refresh All Prices", key="dash_refresh_virtual",
                     type="primary", use_container_width=True,
                     help="Fetches live prices for all open virtual positions (~30s for 100 tickers)"):
            with st.spinner("Refreshing live prices for all open positions…"):
                try:
                    # Build a CallLogger bound to this tab's connection. `logger`
                    # was previously referenced here but only ever existed as a
                    # local inside get_db(), so this raised NameError on click.
                    _logger = ts.CallLogger.__new__(ts.CallLogger)
                    _logger.conn = conn
                    _logger._init_schema()
                    _n = _logger.update_calls()
                    st.session_state["virtual_pos_refreshed_at"] = (
                        datetime.utcnow().strftime("%H:%M:%S UTC")
                    )
                    st.success(f"Updated {_n} position(s)", icon="✅")
                    st.rerun()
                except Exception as _re:
                    st.error(f"Refresh failed: {_re}")

    # ── Cached duration predictor (per ticker+target, 1 h TTL) ───────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_duration_prediction(ticker: str, entry: float, target: float,
                                       pattern: str) -> dict:
        try:
            return ts.predict_duration_to_target(
                ticker, entry, target, pattern=pattern
            ) or {}
        except Exception:
            return {}

    open_pos = tracker.get_open_positions()
    if not open_pos:
        st.info("No open positions. Run a scan to start tracking.", icon="📂")
    else:
        today = datetime.now().date()
        rows = []
        for p in open_pos:
            ep  = p.get("entry_price") or 0
            cur = p.get("current_price") or ep
            pct = p.get("current_pct") or 0

            # ── Date parsing: handle Postgres date objects + string formats ──
            _ed = p.get("entry_date")
            day_n = "?"
            try:
                if _ed is None:
                    pass
                elif isinstance(_ed, str):
                    _ed_dt = datetime.strptime(_ed[:10], "%Y-%m-%d").date()
                    day_n = (today - _ed_dt).days + 1
                elif hasattr(_ed, "year"):
                    _ed_dt = _ed.date() if hasattr(_ed, "hour") else _ed
                    day_n = (today - _ed_dt).days + 1
                else:
                    _ed_dt = pd.Timestamp(_ed).date()
                    day_n = (today - _ed_dt).days + 1
            except Exception:
                day_n = "?"

            # ── Expected duration: use stored est_duration_max, else predict ─
            _expected = None
            _est_max = p.get("est_duration_max")
            if _est_max and int(_est_max) > 0:
                _expected = int(_est_max)
            else:
                # Fall back to predictor (cached per ticker+target+pattern)
                _pred = _cached_duration_prediction(
                    p.get("ticker", ""),
                    float(p.get("entry_price") or 0),
                    float(p.get("target_price") or 0),
                    str(p.get("pattern_detected") or "")
                )
                if _pred and _pred.get("predicted_days"):
                    _expected = int(_pred["predicted_days"])

            # ── Format "Day #" cell: D5 / 8d with color/icon based on ratio ─
            if isinstance(day_n, int) and _expected:
                _ratio = day_n / _expected
                if _ratio < 0.5:
                    _day_icon = "🟢"
                elif _ratio < 0.8:
                    _day_icon = "🔵"
                elif _ratio <= 1.0:
                    _day_icon = "🟡"
                elif _ratio <= 1.5:
                    _day_icon = "🟠 STALE"
                else:
                    _day_icon = "🔴 EXPIRED"
                day_cell = f"{_day_icon} D{day_n}/{_expected}d"
            elif isinstance(day_n, int):
                day_cell = f"D{day_n}"
            else:
                day_cell = "?"
            tgt = p.get("target_price")
            stp = p.get("stop_loss")
            if tgt and cur >= tgt * 0.95:   status_flag = "🎯 Near Target"
            elif stp and cur <= stp * 1.05: status_flag = "⚠️ Near Stop"
            elif pct > 15:                  status_flag = "🚀 Running"
            elif pct > 0:                   status_flag = "✅ Up"
            elif pct < -8:                  status_flag = "❌ Down"
            else:                           status_flag = "➖ Flat"
            rows.append({
                "Ticker": p["ticker"],
                "Entry $": f"${ep:.2f}",
                "Current $": f"${cur:.2f}",
                "P&L %": f"{pct:+.1f}%",
                "Day (act/exp)": day_cell,
                "Target": f"${tgt:.2f}" if tgt else "—",
                "Stop": f"${stp:.2f}" if stp else "—",
                "Pattern": (p.get("pattern_detected") or "—")[:16],
                "Status": status_flag,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Recent closed calls
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Recent Closed Calls</div>',
                unsafe_allow_html=True)
    hist_df = _df("""
        SELECT ticker, entry_price, outcome_price, outcome_pct,
               actual_duration, pattern_detected, result, entry_date
        FROM calls WHERE result IN ('WIN','LOSS','BREAKEVEN')
        ORDER BY scan_timestamp DESC LIMIT 20
    """, conn)
    if hist_df.empty:
        st.info("No closed calls yet.", icon="📋")
    else:
        hist_df["P&L %"] = hist_df["outcome_pct"].apply(
            lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
        hist_df["Days"]   = hist_df["actual_duration"].apply(
            lambda x: f"{int(x)}d" if pd.notna(x) else "?")
        hist_df["Result"] = hist_df["result"].map(
            {"WIN": "✅ WIN", "LOSS": "❌ LOSS", "BREAKEVEN": "➖ EVEN"})
        hist_df["Entry"] = hist_df["entry_price"].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "—")
        hist_df["Exit"] = hist_df["outcome_price"].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "—")
        st.dataframe(
            hist_df[["ticker","Entry","Exit","P&L %","Days","pattern_detected","Result"]]
                .rename(columns={"ticker":"Ticker","pattern_detected":"Pattern"}),
            use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — SCAN RESULTS
# ──────────────────────────────────────────────────────────────────────────────
with tab_results:
    if not st.session_state.scan_ran:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px;">
          <div style="font-size:3rem; margin-bottom:16px;">🔍</div>
          <div style="font-size:1.2rem; font-weight:700; color:#e6edf3; margin-bottom:8px;">
            No scan results yet
          </div>
          <div style="font-size:0.9rem; color:#8b949e;">
            Configure your universe and filters in the sidebar, then click
            <strong style="color:#58a6ff;">🚀 Run Scan</strong> to find opportunities.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        results = st.session_state.results
        ctx     = st.session_state.market_ctx

        # ── Market regime banner ──────────────────────────────────────────────
        _render_regime_banner(st.session_state.regime)

        # ── Market context bar ────────────────────────────────────────────────
        raw      = ctx.get("spy_raw", 0)
        filtered = ctx.get("filtered", 0)
        advanced = ctx.get("advanced", 0)
        n_qual   = len(results)
        n_expl   = sum(1 for r in results if (r.get("explosive_score") or 0) >= 40)
        n_high   = sum(1 for r in results if (r.get("probability") or 0) >= 75)

        st.markdown(f"""
        <div class="ctx-bar">
          <div class="ctx-item">
            <div class="ctx-label">Universe</div>
            <div class="ctx-val">{raw:,}</div>
          </div>
          <div style="color:#30363d; font-size:1.2rem; align-self:center;">→</div>
          <div class="ctx-item">
            <div class="ctx-label">Quick filter</div>
            <div class="ctx-val">{filtered:,}</div>
          </div>
          <div style="color:#30363d; font-size:1.2rem; align-self:center;">→</div>
          <div class="ctx-item">
            <div class="ctx-label">Deep analyzed</div>
            <div class="ctx-val">{advanced:,}</div>
          </div>
          <div style="color:#30363d; font-size:1.2rem; align-self:center;">→</div>
          <div class="ctx-item">
            <div class="ctx-label">Qualified</div>
            <div class="ctx-val" style="color:#58a6ff;">{n_qual}</div>
          </div>
          <div style="margin-left:auto; display:flex; gap:16px; align-items:center;">
            <div class="ctx-item">
              <div class="ctx-label">⚡ Explosive</div>
              <div class="ctx-val" style="color:#e3b341;">{n_expl}</div>
            </div>
            <div class="ctx-item">
              <div class="ctx-label">🟢 High conf</div>
              <div class="ctx-val" style="color:#3fb950;">{n_high}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not results:
            st.warning("No stocks qualified with the current filters.", icon="🚫")
        else:
            # ── Sort & filter controls ────────────────────────────────────────
            fc1, fc2, fc3 = st.columns([3, 1.4, 1.4])
            with fc1:
                search = st.text_input("🔎 Filter by ticker or pattern", "",
                                       placeholder="e.g.  AAPL  or  Cup")
            with fc2:
                sort_by = st.selectbox("Sort by",
                    ["Probability", "Explosive Score", "R:R Ratio"])
            with fc3:
                show_only = st.selectbox("Show",
                    ["All results", "High conf only (≥75%)", "Explosive only"])

            # Apply search
            view = results
            if search.strip():
                s = search.strip().upper()
                view = [r for r in view if
                        s in (r.get("ticker") or "").upper() or
                        s in (r.get("pattern") or "").upper()]

            # Apply show filter
            if show_only == "High conf only (≥75%)":
                view = [r for r in view if (r.get("probability") or 0) >= 75]
            elif show_only == "Explosive only":
                view = [r for r in view if (r.get("explosive_score") or 0) >= 40]

            # Apply sort
            if sort_by == "Explosive Score":
                view = sorted(view, key=lambda x: x.get("explosive_score") or 0, reverse=True)
            elif sort_by == "R:R Ratio":
                view = sorted(view, key=lambda x: x.get("rr") or 0, reverse=True)
            else:
                view = sorted(view, key=lambda x: x.get("probability") or 0, reverse=True)

            if not view:
                st.info("No results match your filter.", icon="🔍")
            else:
                st.caption(f"Showing {len(view)} stock{'s' if len(view) != 1 else ''}")
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Card grid (3 columns) ─────────────────────────────────────
                cols_per_row = 3
                for i in range(0, len(view), cols_per_row):
                    batch = view[i : i + cols_per_row]
                    cols  = st.columns(cols_per_row)
                    for col, r in zip(cols, batch):
                        with col:
                            _render_stock_card(r)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — STOCK CHART
# ──────────────────────────────────────────────────────────────────────────────
with tab_chart:
    # Ticker picker — pre-fill from selection
    all_tickers = [r["ticker"] for r in st.session_state.results]
    default_val = st.session_state.selected or (all_tickers[0] if all_tickers else "")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        ticker_input = st.text_input(
            "Ticker symbol", value=default_val,
            placeholder="AAPL, RY.TO, NVDA …")
    with col_b:
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
    with col_c:
        interval = st.selectbox("Candle", ["1d", "1wk"], index=0)

    if ticker_input.strip():
        tk = ticker_input.strip().upper()

        # Find matching result for overlays
        result = next((r for r in st.session_state.results
                       if r["ticker"] == tk), None)

        with st.spinner(f"Loading {tk}…"):
            try:
                raw = yf.download(tk, period=period, interval=interval,
                                  auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.dropna()
            except Exception as e:
                raw = pd.DataFrame()
                st.error(f"Could not fetch data for {tk}: {e}")

        if raw.empty:
            st.warning(f"No price data returned for **{tk}**.")
        else:
            # ── Build chart ───────────────────────────────────────────────────
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.60, 0.20, 0.20],
                vertical_spacing=0.02,
                subplot_titles=(f"{tk}  |  OHLC + MAs", "Volume", "RSI (14)"),
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=raw.index,
                open=raw["Open"], high=raw["High"],
                low=raw["Low"],   close=raw["Close"],
                name="Price",
                increasing_line_color="#3fb950",
                decreasing_line_color="#f85149",
                increasing_fillcolor="#3fb950",
                decreasing_fillcolor="#f85149",
            ), row=1, col=1)

            # Moving averages
            for period_ma, colour in [(20,"#58a6ff"), (50,"#e3b341"), (200,"#ff7b72")]:
                if len(raw) >= period_ma:
                    ma = raw["Close"].rolling(period_ma).mean()
                    fig.add_trace(go.Scatter(
                        x=raw.index, y=ma, name=f"SMA{period_ma}",
                        line=dict(color=colour, width=1.2),
                        opacity=0.85,
                    ), row=1, col=1)

            # Entry / Stop / Target overlays from scan result
            if result:
                ep  = result.get("price")
                stp = result.get("stop_price")
                tgt = result.get("tgt_price")
                last_date = raw.index[-1]
                span = [raw.index[max(0, len(raw)-30)], last_date]
                for val, label, color in [
                    (ep,  "Entry", "#58a6ff"),
                    (stp, "Stop",  "#f85149"),
                    (tgt, "Target","#3fb950"),
                ]:
                    if val:
                        fig.add_shape(type="line", x0=span[0], x1=span[1],
                                      y0=val, y1=val, xref="x", yref="y",
                                      line=dict(color=color, width=1.5, dash="dash"),
                                      row=1, col=1)
                        fig.add_annotation(
                            x=last_date, y=val, text=f" {label} ${val:.2f}",
                            font=dict(color=color, size=11),
                            showarrow=False, xanchor="left", row=1, col=1)

            # Bollinger Bands (20, 2σ)
            if len(raw) >= 20:
                mid  = raw["Close"].rolling(20).mean()
                std  = raw["Close"].rolling(20).std()
                ub   = mid + 2 * std
                lb   = mid - 2 * std
                for band, name in [(ub, "BB Upper"), (lb, "BB Lower")]:
                    fig.add_trace(go.Scatter(
                        x=raw.index, y=band, name=name,
                        line=dict(color="#6e40c9", width=0.8, dash="dot"),
                        opacity=0.5,
                    ), row=1, col=1)

            # Volume bars
            colors = ["#3fb950" if c >= o else "#f85149"
                      for c, o in zip(raw["Close"], raw["Open"])]
            fig.add_trace(go.Bar(
                x=raw.index, y=raw["Volume"], name="Volume",
                marker_color=colors, opacity=0.7,
            ), row=2, col=1)

            # RSI
            delta = raw["Close"].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = 100 - (100 / (1 + rs))
            fig.add_trace(go.Scatter(
                x=raw.index, y=rsi, name="RSI",
                line=dict(color="#e3b341", width=1.3),
            ), row=3, col=1)
            for level, col_ in [(70,"#f85149"),(30,"#3fb950"),(50,"#8b949e")]:
                fig.add_shape(type="line",
                              x0=raw.index[0], x1=raw.index[-1],
                              y0=level, y1=level, xref="x3", yref="y3",
                              line=dict(color=col_, width=0.8, dash="dash"))

            # ── Layout ────────────────────────────────────────────────────────
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9", family="monospace"),
                legend=dict(bgcolor="#161b22", bordercolor="#30363d",
                            borderwidth=1, font=dict(size=10)),
                xaxis_rangeslider_visible=False,
                height=700,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            for ax in ["xaxis", "xaxis2", "xaxis3",
                       "yaxis", "yaxis2", "yaxis3"]:
                fig.update_layout(**{ax: dict(
                    gridcolor="#21262d", zerolinecolor="#30363d",
                    tickfont=dict(size=10),
                )})

            st.plotly_chart(fig, use_container_width=True)

            # ── Breakdown panel ───────────────────────────────────────────────
            if result:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="section-hdr">{tk} — Signal Breakdown</div>',
                            unsafe_allow_html=True)
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.metric("Breakout Prob", f"{result.get('probability') or 0}%")
                with c2: st.metric("Pattern",       (result.get("pattern") or "—")[:18])
                with c3: st.metric("RSI",           f"{result.get('rsi') or 0:.1f}")
                with c4: st.metric("R:R",           f"{result.get('rr') or 0:.1f}:1")
                with c5: st.metric("Earnings Risk", result.get("earnings_risk") or "No")

                # Probability modifier breakdown
                pos = result.get("pos_mods", [])
                neg = result.get("neg_mods", [])
                if pos or neg:
                    cm1, cm2 = st.columns(2)
                    with cm1:
                        st.markdown("**✅ Positive modifiers**")
                        for m in pos:
                            st.markdown(f"- {m}")
                    with cm2:
                        st.markdown("**❌ Negative modifiers**")
                        for m in neg:
                            st.markdown(f"- {m}")

                thesis = result.get("thesis", "")
                if thesis:
                    st.markdown(f"> {thesis}")
    else:
        st.info("Select a ticker from the Scan Results table, or type one above.",
                icon="📈")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
with tab_analytics:
    conn, _mode = get_db()
    analyzer = ts.PerformanceAnalyzer(conn)
    s        = analyzer.summary()

    if s["total"] == 0:
        st.info("No resolved calls yet. Run scans to start building history.", icon="📉")
    else:
        c1, c2 = st.columns(2)

        # ── Pattern win-rate bar chart ─────────────────────────────────────────
        with c1:
            st.markdown('<div class="section-hdr">Win Rate by Pattern</div>',
                        unsafe_allow_html=True)
            pats = analyzer.pattern_ranking()
            if pats:
                df_p = pd.DataFrame(pats)
                fig_p = px.bar(
                    df_p[df_p["total"] >= 1],
                    x="win_rate", y="pattern", orientation="h",
                    color="win_rate", color_continuous_scale="RdYlGn",
                    range_color=[0, 100],
                    text=df_p["total"].apply(lambda n: f"n={n}"),
                    labels={"win_rate": "Win Rate %", "pattern": ""},
                )
                fig_p.add_vline(x=50, line_dash="dash",
                                line_color="#8b949e", annotation_text="50%")
                fig_p.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"), showlegend=False,
                    coloraxis_showscale=False, height=350,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_p, use_container_width=True)

        # ── Catalyst win-rate bar chart ────────────────────────────────────────
        with c2:
            st.markdown('<div class="section-hdr">Win Rate by Catalyst</div>',
                        unsafe_allow_html=True)
            cats = analyzer.catalyst_ranking()
            if cats:
                df_c = pd.DataFrame(cats)
                fig_c = px.bar(
                    df_c[df_c["total"] >= 1],
                    x="win_rate", y="catalyst", orientation="h",
                    color="win_rate", color_continuous_scale="RdYlGn",
                    range_color=[0, 100],
                    text=df_c["total"].apply(lambda n: f"n={n}"),
                    labels={"win_rate": "Win Rate %", "catalyst": ""},
                )
                fig_c.add_vline(x=50, line_dash="dash", line_color="#8b949e")
                fig_c.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"), showlegend=False,
                    coloraxis_showscale=False, height=350,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_c, use_container_width=True)

        # ── Probability calibration chart ──────────────────────────────────────
        st.markdown('<div class="section-hdr">Probability Calibration — Expected vs Actual Win Rate</div>',
                    unsafe_allow_html=True)
        cals = analyzer.score_calibration()
        if cals:
            df_cal = pd.DataFrame(cals)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=df_cal["expected"], y=df_cal["expected"],
                name="Perfect calibration",
                line=dict(color="#8b949e", dash="dash"), mode="lines",
            ))
            fig_cal.add_trace(go.Scatter(
                x=df_cal["expected"], y=df_cal["actual"],
                name="Actual", mode="lines+markers+text",
                text=df_cal["range"],
                textposition="top center",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=10,
                            color=["#3fb950" if abs(d) < 10 else "#f85149"
                                   for d in df_cal["delta"]]),
            ))
            fig_cal.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"), height=320,
                xaxis_title="Expected Win Rate %",
                yaxis_title="Actual Win Rate %",
                legend=dict(bgcolor="#161b22"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            for ax in ["xaxis", "yaxis"]:
                fig_cal.update_layout(**{ax: dict(gridcolor="#21262d")})
            st.plotly_chart(fig_cal, use_container_width=True)
        else:
            st.caption("Need at least 5 resolved calls per probability bucket to show calibration.")

        # ── Cumulative P&L curve ───────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Cumulative P&L (Virtual $10K / position)</div>',
                    unsafe_allow_html=True)
        pnl_df = _df("""
            SELECT scan_timestamp, outcome_pct, result
            FROM calls WHERE result IN ('WIN','LOSS','BREAKEVEN')
            AND outcome_pct IS NOT NULL
            ORDER BY scan_timestamp ASC
        """, conn)
        if not pnl_df.empty:
            pnl_df["pnl_$"] = pnl_df["outcome_pct"] * 10_000
            pnl_df["cum"]   = pnl_df["pnl_$"].cumsum()
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=list(range(len(pnl_df))), y=pnl_df["cum"],
                fill="tozeroy",
                line=dict(color="#58a6ff", width=2),
                fillcolor="rgba(88,166,255,0.1)",
                name="Cumulative P&L",
            ))
            fig_pnl.add_hline(y=0, line_color="#30363d")
            fig_pnl.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"), height=300,
                xaxis_title="Trade #", yaxis_title="Cumulative P&L ($)",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        # ── Improvement suggestions ────────────────────────────────────────────
        tips = analyzer.suggest_improvements()
        if tips:
            st.markdown('<div class="section-hdr">Self-Improvement Suggestions</div>',
                        unsafe_allow_html=True)
            for tip in tips:
                st.warning(tip, icon="💡")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — PAPER PORTFOLIO
# ──────────────────────────────────────────────────────────────────────────────
# ── Live position monitor (runs every 60 s, independent of scan) ─────────────
# Defined outside `with tab_paper` so @st.fragment can decorate it at module level.
try:
    @st.fragment(run_every=60)
    def _live_position_monitor():
        """Auto-runs every 60 s. Checks open positions against 1-min intraday
        prices and closes any that have hit their stop or target."""
        _conn, _ = get_db()
        _paper   = ts.PaperTradingEngine(_conn)
        _pos     = _paper.open_positions

        session  = ts.MarketClock.get_session()
        is_open  = session.get("is_open", False)

        col_l, col_r = st.columns([3, 1])
        with col_r:
            if st.button("🔄 Check Now", key="check_now_btn", use_container_width=True):
                _closed = _paper.check_stops_and_targets()
                if _closed:
                    for c in _closed:
                        em = "🎯" if "TARGET" in (c.get("exit_reason") or "") else "🛑"
                        st.success(f"{em} **{c['ticker']}** closed — "
                                   f"{c.get('exit_reason','')} @ ${c.get('exit_price',0):.2f}")
                    st.rerun()
                else:
                    st.info("No stops or targets hit.", icon="✅")

        with col_l:
            now_str = datetime.now().strftime("%H:%M:%S")
            if not _pos:
                st.caption(f"⬜ Live monitor — no open positions  ({now_str})")
                return

            if is_open:
                # Market is open → check stops/targets
                _closed = _paper.check_stops_and_targets()
                if _closed:
                    for c in _closed:
                        em = "🎯" if "TARGET" in (c.get("exit_reason") or "") else "🛑"
                        st.success(f"{em} **{c['ticker']}** auto-closed — "
                                   f"{c.get('exit_reason','')} @ ${c.get('exit_price',0):.2f}  "
                                   f"| Net P&L: {'+'if (c.get('net_pnl') or 0)>=0 else ''}"
                                   f"${c.get('net_pnl',0):.2f}")
                    st.rerun()
                else:
                    tickers_str = ", ".join(p["ticker"] for p in _pos)
                    st.caption(
                        f"🟢 Live monitor **active** — watching {len(_pos)} position(s): "
                        f"{tickers_str}  |  last checked {now_str}  "
                        f"(refreshes every 60 s)"
                    )
            else:
                win = session.get("name", "Closed")
                st.caption(
                    f"🌙 Live monitor **paused** — market is {win}  "
                    f"({now_str})  |  {len(_pos)} position(s) held overnight"
                )

except AttributeError:
    # Streamlit < 1.37 — define a no-op; manual Check Now button still works via fallback below
    def _live_position_monitor():
        pass


with tab_paper:
    conn, _mode = get_db()
    paper = ts.PaperTradingEngine(conn)
    s     = paper.get_summary()
    positions = paper.open_positions

    # ── Live monitor bar ──────────────────────────────────────────────────────
    _live_position_monitor()

    st.markdown("<hr style='border-color:#21262d;margin:4px 0 12px 0'>",
                unsafe_allow_html=True)

    # ── Fetch live prices (cached 30 s) ──────────────────────────────────────
    live_prices: dict = {}
    if positions:
        tickers_tuple = tuple(p["ticker"] for p in positions)
        _price_salt = st.session_state.get("price_refresh_salt", 0)
        with st.spinner("Fetching live prices…"):
            live_prices = _fetch_live_prices(tickers_tuple, _salt=_price_salt)

    # Build per-position live data — only include positions WITH a live price
    # in the totals.  Without a live price we cannot honestly say what the
    # unrealized P&L is, so we mark it as N/A rather than show fake math.
    live_pos_map: dict = {}
    total_live_value = 0.0
    total_cost_basis = 0.0
    _live_count      = 0
    for p in positions:
        tk     = p["ticker"]
        ep     = float(p.get("entry_price") or 0)
        shares = float(p.get("shares") or 0)
        gross  = float(p.get("gross_invested") or 0)
        cur    = live_prices.get(tk)
        has_lv = cur is not None and cur > 0

        if has_lv:
            cur_val = shares * cur
            unr_pnl = cur_val - gross
            unr_pct = (unr_pnl / gross * 100) if gross else 0.0
            total_live_value += cur_val
            total_cost_basis += gross
            _live_count += 1
        else:
            # No live price → don't fabricate a value
            cur     = ep   # display fallback only
            cur_val = None
            unr_pnl = None
            unr_pct = None

        live_pos_map[tk] = {
            "current_price":  cur,
            "current_value":  cur_val,
            "unrealized_pnl": unr_pnl,
            "unrealized_pct": unr_pct,
            "has_live_price": has_lv,
        }

    # If NO positions had live prices, totals are unknown — don't show fake P&L
    if _live_count == 0:
        total_unrealized = None
        unr_pct_total    = None
    else:
        total_unrealized = total_live_value - total_cost_basis
        unr_pct_total    = (total_unrealized / total_cost_basis * 100) if total_cost_basis else 0.0
    prices_fetched = _live_count > 0
    _missing_count = len(positions) - _live_count

    # Mark-to-market total (cash + current value of positions at live prices).
    # When live prices are missing for some positions, fall back to cost basis
    # for those — better than leaving them out entirely.
    _mtm_positions_val = 0.0
    for p in positions:
        tk    = p["ticker"]
        _lpm  = live_pos_map.get(tk, {})
        if _lpm.get("has_live_price") and _lpm.get("current_value") is not None:
            _mtm_positions_val += float(_lpm["current_value"])
        else:
            _mtm_positions_val += float(p.get("gross_invested") or 0)
    mtm_total = s["available_cash"] + _mtm_positions_val
    mtm_ret   = mtm_total - s["starting_capital"]

    # ── Header KPIs ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Paper Portfolio — $1,000 Starting Capital</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    ret_col = "green" if s["total_return"] >= 0 else "red"
    mtm_col = "green" if mtm_ret >= 0 else "red"
    with c1: kpi("MTM Portfolio",
                 f"${mtm_total:,.2f}",
                 mtm_col)
    with c2: kpi("Total Return",
                 f"{'+'if s['total_return']>=0 else ''}${s['total_return']:,.2f}",
                 ret_col)
    with c3: kpi("Return %",
                 f"{'+'if s['total_return_pct']>=0 else ''}{s['total_return_pct']:.1f}%",
                 ret_col)
    with c4: kpi("Fees Paid",  f"${s['total_fees_paid']:.2f}", "red")
    with c5: kpi("Open Slots",
                 f"{s['open_positions']}/{s['max_positions']}", "yellow")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Available Cash",  f"${s['available_cash']:,.2f}", "blue")
    with c2: kpi("Invested",        f"${s['invested_value']:,.2f}", "yellow")
    with c3: kpi("Realized P&L",
                 f"{'+'if s['realized_pnl']>=0 else ''}${s['realized_pnl']:,.2f}",
                 "green" if s["realized_pnl"] >= 0 else "red")
    with c4: kpi("Fee Drag",        f"-{s['fee_drag_pct']:.2f}%", "red")

    # ── Unrealized P&L row ───────────────────────────────────────────────────
    if positions:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        if total_unrealized is None:
            # No live prices at all — show N/A honestly
            with c1: kpi("Unrealized P&L", "N/A — no live prices", "yellow")
            with c2: kpi("Unrealized %",   "—",                    "yellow")
        else:
            unr_col = "green" if total_unrealized >= 0 else "red"
            with c1: kpi("Unrealized P&L",
                         f"{'+'if total_unrealized>=0 else ''}${total_unrealized:,.2f}",
                         unr_col)
            with c2: kpi("Unrealized %",
                         f"{'+'if unr_pct_total>=0 else ''}{unr_pct_total:.2f}%",
                         unr_col)
        with c3: kpi("Cost Basis (Open)", f"${sum(float(p.get('gross_invested') or 0) for p in positions):,.2f}", "yellow")

        _pl1, _pl2 = st.columns([4, 1])
        with _pl1:
            if prices_fetched and _missing_count == 0:
                st.caption("✅ Live prices for all positions — auto-refresh every 30 s")
            elif prices_fetched and _missing_count > 0:
                st.caption(f"⚠️ Live prices for {_live_count}/{len(positions)} positions  ·  "
                           f"{_missing_count} position(s) could not be fetched")
            else:
                st.error(
                    "🚨 **yfinance returned no prices for any position.**  "
                    "Streamlit Cloud sometimes gets rate-limited by Yahoo. "
                    "Click 🔄 Refresh now to retry, or wait a minute and try again. "
                    "If this persists, the issue is upstream — your bot logic is fine.",
                    icon="🚨",
                )
        with _pl2:
            if st.button("🔄 Refresh now", key="paper_refresh_prices",
                         help="Force a fresh price fetch (bypasses cache)"):
                st.session_state["price_refresh_salt"] = (
                    st.session_state.get("price_refresh_salt", 0) + 1
                )
                try:
                    _fetch_live_prices.clear()
                except Exception:
                    pass
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Portfolio Pie Chart ───────────────────────────────────────────────────
    if positions or s["total_fees_paid"] > 0:
        labels, values, colors = [], [], []
        for p in positions:
            invested_val = (p.get("gross_invested") or 0) - (p.get("buy_fee") or 0)
            labels.append(p["ticker"])
            values.append(max(invested_val, 0))
            colors.append("#3fb950")

        labels.append("Available Cash")
        values.append(max(s["available_cash"], 0))
        colors.append("#30363d")

        if s["total_fees_paid"] > 0:
            labels.append("Fees Paid")
            values.append(s["total_fees_paid"])
            colors.append("#f85149")

        col_pie, col_stats = st.columns([1, 1])
        with col_pie:
            st.markdown('<div class="section-hdr">Capital Allocation</div>',
                        unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(size=12, color="#e6edf3"),
                hovertemplate="<b>%{label}</b><br>$%{value:.2f}<br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                font=dict(color="#c9d1d9"), height=320,
                showlegend=False, margin=dict(l=10, r=10, t=20, b=10),
                annotations=[dict(text=f"${s['total_capital']:,.0f}",
                                  font=dict(size=18, color="#e6edf3"), showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_stats:
            st.markdown('<div class="section-hdr">Position Breakdown</div>',
                        unsafe_allow_html=True)
            total_v = sum(values)
            for label, val, col in zip(labels, values, colors):
                pct = val / total_v * 100 if total_v > 0 else 0
                dot = ("🟢" if label not in ("Available Cash","Fees Paid") else
                       ("⬜" if label == "Available Cash" else "🔴"))
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 0;border-bottom:1px solid #21262d;font-size:0.85rem;">'
                    f'<span>{dot} {label}</span>'
                    f'<span style="color:#e6edf3;font-weight:600;">'
                    f'${val:,.2f} &nbsp;<span style="color:#8b949e;">'
                    f'({pct:.1f}%)</span></span></div>',
                    unsafe_allow_html=True)
            st.markdown(
                f'<div style="margin-top:12px;font-size:0.76rem;color:#8b949e;">'
                f'Break-even move per trade: '
                f'<strong style="color:#e3b341;">+{s["break_even_pct"]:.2f}%</strong>'
                f' (1.5% buy + 1.5% sell fees)</div>',
                unsafe_allow_html=True)
    else:
        st.info("No paper trades yet. Run a scan with auto-trade enabled to start.", icon="💼")

    # ── Open Positions Detail ─────────────────────────────────────────────────
    if positions:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-hdr">Open Positions</div>',
                    unsafe_allow_html=True)
        for p in positions:
            ep     = p.get("entry_price") or 0
            tgt    = p.get("target_price") or 0
            stp    = p.get("stop_loss") or 0
            shares = p.get("shares") or 0
            gross  = p.get("gross_invested") or 0
            bf     = p.get("buy_fee") or 0
            far    = p.get("fee_adj_rr") or 0
            tk     = p["ticker"]

            lp       = live_pos_map.get(tk, {})
            cur      = lp.get("current_price") or ep
            unr      = lp.get("unrealized_pnl") or 0
            unr_p    = lp.get("unrealized_pct") or 0
            cur_val  = lp.get("current_value") or (shares * ep)
            has_lv   = lp.get("has_live_price", False)
            price_lbl = "Live Price" if has_lv else "Est Price"
            unr_sign  = "+" if unr >= 0 else ""

            with st.expander(
                f"**{tk}** — entered @ ${ep:.2f}  |  "
                f"{price_lbl}: ${cur:.2f}  |  "
                f"Unrealized P&L: {unr_sign}${unr:.2f} ({unr_sign}{unr_p:.1f}%)",
                expanded=True,
            ):
                # ── Live P&L row ──────────────────────────────────────────────
                lc1, lc2, lc3 = st.columns(3)
                with lc1:
                    chg_pct = (cur - ep) / ep * 100 if ep else 0
                    st.metric(price_lbl, f"${cur:.2f}",
                              delta=f"{chg_pct:+.2f}%")
                with lc2:
                    st.metric("Unrealized P&L",
                              f"{unr_sign}${unr:.2f}",
                              delta=f"{unr_sign}{unr_p:.2f}%")
                with lc3:
                    st.metric("Current Value", f"${cur_val:.2f}")

                st.divider()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entry",     f"${ep:.2f}")
                    st.metric("Shares",    f"{shares:.4f}")
                with col2:
                    st.metric("Stop Loss", f"${stp:.2f}" if stp else "—",
                              delta=f"-{(ep-stp)/ep*100:.1f}%" if stp and ep else None,
                              delta_color="inverse")
                    st.metric("Invested",  f"${gross:.2f}")
                with col3:
                    st.metric("Target",    f"${tgt:.2f}" if tgt else "—",
                              delta=f"+{(tgt-ep)/ep*100:.1f}%" if tgt and ep else None)
                    st.metric("Buy Fee",   f"-${bf:.2f}")
                with col4:
                    st.metric("Fee-Adj R/R", f"{far:.1f}:1" if far else "—")
                    est_sell = shares * (tgt or ep) * 0.015
                    st.metric("Est Sell Fee", f"-${est_sell:.2f}")

                st.caption(f"Pattern: {p.get('pattern','—')}  |  "
                           f"Prob: {p.get('breakout_prob',0):.0f}%  |  "
                           f"Explosive Score: {p.get('explosive_score',0):.0f}")

        # Horizontal range chart
        st.markdown('<div class="section-hdr">Position Range — Stop vs Entry vs Target</div>',
                    unsafe_allow_html=True)
        tickers_p = [p["ticker"] for p in positions]
        entries   = [p.get("entry_price") or 0 for p in positions]
        stops     = [p.get("stop_loss") or 0   for p in positions]
        targets   = [p.get("target_price") or 0 for p in positions]

        fig_range = go.Figure()
        for i, (tk, ep, stp, tgt) in enumerate(zip(tickers_p, entries, stops, targets)):
            if not ep:
                continue
            # Full range bar
            fig_range.add_trace(go.Bar(
                y=[tk], x=[tgt - stp], base=[stp], orientation="h",
                marker=dict(color="rgba(88,166,255,0.15)",
                            line=dict(color="#30363d", width=1)),
                showlegend=False, hoverinfo="skip",
            ))
            for val, col, sym, lbl in [
                (stp, "#f85149", "circle", "Stop"),
                (ep,  "#58a6ff", "diamond", "Entry"),
                (tgt, "#3fb950", "circle", "Target"),
            ]:
                if val:
                    fig_range.add_trace(go.Scatter(
                        y=[tk], x=[val], mode="markers",
                        marker=dict(size=12, color=col, symbol=sym),
                        name=lbl, showlegend=(i == 0),
                        hovertemplate=f"{lbl}: ${val:.2f}<extra></extra>",
                    ))
        fig_range.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"), height=120 + len(positions) * 50,
            xaxis=dict(title="Price", gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            barmode="overlay", margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_range, use_container_width=True)

    # ── Closed Trades ─────────────────────────────────────────────────────────
    closed_rows = conn.execute(
        "SELECT * FROM paper_portfolio WHERE status='CLOSED' ORDER BY exit_date DESC"
    ).fetchall()
    if closed_rows:
        st.markdown('<div class="section-hdr">Closed Paper Trades</div>',
                    unsafe_allow_html=True)
        rows = []
        for r in closed_rows:
            r = dict(r)
            rows.append({
                "Ticker":   r["ticker"],
                "Entry":    f"${r.get('entry_price') or 0:.2f}",
                "Exit":     f"${r.get('exit_price') or 0:.2f}",
                "Shares":   f"{r.get('shares') or 0:.3f}",
                "Fees":     f"${(r.get('buy_fee') or 0)+(r.get('sell_fee') or 0):.2f}",
                "Gross P&L":f"{'+'if (r.get('gross_pnl') or 0)>=0 else ''}${r.get('gross_pnl') or 0:.2f}",
                "Net P&L":  f"{'+'if (r.get('net_pnl') or 0)>=0 else ''}${r.get('net_pnl') or 0:.2f}",
                "Net %":    f"{'+'if (r.get('net_pnl_pct') or 0)>=0 else ''}{r.get('net_pnl_pct') or 0:.1f}%",
                "Reason":   r.get("exit_reason","—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — FEES TRACKER
# ──────────────────────────────────────────────────────────────────────────────
with tab_fees:
    conn, _mode = get_db()
    _PAPER_BUDGET = ts._PAPER_BUDGET

    try:
        fee_rows = conn.execute(
            "SELECT * FROM fees ORDER BY transaction_date ASC, id ASC"
        ).fetchall()
        fee_data = [dict(r) for r in fee_rows]
    except Exception:
        fee_data = []

    total_fees  = sum(r.get("fee_amount") or 0 for r in fee_data)
    buy_fees    = sum(r.get("fee_amount") or 0 for r in fee_data
                      if r.get("transaction_type") == "BUY")
    sell_fees   = sum(r.get("fee_amount") or 0 for r in fee_data
                      if r.get("transaction_type") == "SELL")
    n_buy       = sum(1 for r in fee_data if r.get("transaction_type") == "BUY")
    n_sell      = sum(1 for r in fee_data if r.get("transaction_type") == "SELL")

    paper_s = ts.PaperTradingEngine(conn).get_summary()
    try:
        gross_pnl_row = conn.execute(
            "SELECT SUM(gross_pnl) FROM paper_portfolio WHERE status='CLOSED'"
        ).fetchone()
        gross_pnl_total = float(gross_pnl_row[0] or 0) if gross_pnl_row else 0.0
    except Exception:
        gross_pnl_total = 0.0

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Fee Summary</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Total Fees Paid",  f"${total_fees:.2f}", "red")
    with c2: kpi("Buy Fees",  f"${buy_fees:.2f}  ({n_buy} trades)", "red")
    with c3: kpi("Sell Fees", f"${sell_fees:.2f}  ({n_sell} trades)", "red")
    with c4:
        drag = total_fees / _PAPER_BUDGET * 100 if _PAPER_BUDGET else 0
        kpi("Capital Drag", f"-{drag:.2f}%", "red")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        avg = total_fees / max(len(fee_data), 1)
        kpi("Avg Fee / Transaction", f"${avg:.2f}", "yellow")
    with c2:
        eaten = total_fees / gross_pnl_total * 100 if gross_pnl_total > 0 else 0
        kpi("% of Gross Profits Eaten",
            f"{eaten:.1f}%" if gross_pnl_total > 0 else "N/A", "yellow")
    with c3:
        be = ts._FEE_ENGINE.break_even_move(1.0)["break_even_pct"]
        kpi("Break-even Move / Trade", f"+{be:.2f}%", "yellow")

    st.markdown("<br>", unsafe_allow_html=True)

    if fee_data:
        # ── Cumulative fee chart ──────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Cumulative Fees vs P&L Over Time</div>',
                    unsafe_allow_html=True)

        dates      = [r.get("transaction_date","") for r in fee_data]
        cum_fees   = []
        running    = 0.0
        for r in fee_data:
            running += r.get("fee_amount") or 0
            cum_fees.append(running)

        fig_fees = go.Figure()
        fig_fees.add_trace(go.Scatter(
            x=dates, y=cum_fees, name="Cumulative Fees",
            line=dict(color="#f85149", width=2),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.08)",
        ))

        # Equity snapshots
        try:
            snap_rows = conn.execute(
                "SELECT snapshot_date, realized_pnl, portfolio_return_pct "
                "FROM paper_trading ORDER BY snapshot_date ASC"
            ).fetchall()
        except Exception:
            snap_rows = []
        if snap_rows:
            snap_dates = [r["snapshot_date"] for r in snap_rows]
            snap_pnl   = [r["realized_pnl"]  for r in snap_rows]
            fig_fees.add_trace(go.Scatter(
                x=snap_dates, y=snap_pnl, name="Realized P&L",
                line=dict(color="#3fb950", width=2),
            ))

        fig_fees.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"), height=280,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title="$"),
            legend=dict(bgcolor="#161b22"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_fees, use_container_width=True)

        # ── Full fee log ──────────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Full Fee Transaction Log</div>',
                    unsafe_allow_html=True)
        log_rows = []
        for i, r in enumerate(fee_data, 1):
            t_type = r.get("transaction_type","—")
            log_rows.append({
                "#":        i,
                "Date":     r.get("transaction_date","—"),
                "Ticker":   r.get("ticker","—"),
                "Type":     t_type,
                "Gross":    f"${r.get('gross_amount') or 0:.2f}",
                "Fee":      f"${r.get('fee_amount') or 0:.2f}",
                "Net":      f"${r.get('net_amount') or 0:.2f}",
                "Running Total": f"${r.get('running_total') or 0:.2f}",
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        # Per-ticker fee summary
        st.markdown('<div class="section-hdr">Fee Cost by Ticker</div>',
                    unsafe_allow_html=True)
        ticker_fees: dict = {}
        for r in fee_data:
            tk  = r.get("ticker","?")
            amt = r.get("fee_amount") or 0
            ticker_fees[tk] = ticker_fees.get(tk, 0) + amt
        if ticker_fees:
            df_tf = pd.DataFrame(
                sorted(ticker_fees.items(), key=lambda x: x[1], reverse=True),
                columns=["Ticker", "Total Fees ($)"]
            )
            fig_tf = px.bar(df_tf, x="Ticker", y="Total Fees ($)",
                            color="Total Fees ($)",
                            color_continuous_scale=["#3fb950","#e3b341","#f85149"])
            fig_tf.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"), height=250,
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_tf, use_container_width=True)
    else:
        st.info("No fee transactions recorded yet. "
                "Paper trades will appear here once the portfolio is active.", icon="💰")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — OPTIONS TRADING (Aggressive Fast-Money Focus)
# ══════════════════════════════════════════════════════════════════════════════

with tab_options:
    st.markdown(
        '<div class="section-hdr">⚡ Options Trading — Weekly Plays (5–7 DTE Sweet Spot)</div>',
        unsafe_allow_html=True,
    )

    # ── Market context bar (VIX · F&G · SPY regime · events · risk engine) ───
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_market_context():
        import yfinance as _yf_l
        _reg = None
        try:
            _spy = _yf_l.download("SPY", period="1y", interval="1d",
                                  progress=False, auto_adjust=True)
            if not _spy.empty:
                if isinstance(_spy.columns, pd.MultiIndex):
                    _spy.columns = _spy.columns.get_level_values(0)
                _reg = ts.MarketRegimeDetector.detect(_spy)
        except Exception:
            pass
        return {
            "vix":    ts.get_vix_level(),
            "fg":     ts.get_fear_greed(),
            "events": ts.get_economic_events(days_ahead=7),
            "regime": _reg,
        }

    _ctx    = _get_market_context()
    _vix    = _ctx["vix"]
    _fg     = _ctx["fg"]
    _eco    = _ctx["events"]
    _regime = _ctx.get("regime")

    _mc1, _mc2, _mc3, _mc4 = st.columns([1, 1, 1, 2])

    with _mc1:
        if _vix:
            _vix_colors = {"LOW": "🟢", "NORMAL": "🟡", "ELEVATED": "🟠", "EXTREME": "🔴"}
            _vix_icon   = _vix_colors.get(_vix["regime"], "⚪")
            st.metric(
                label=f"{_vix_icon} VIX — {_vix['regime']}",
                value=f"{_vix['vix']:.2f}",
                help=_vix["advice"],
            )
        else:
            st.metric("VIX", "—", help="Could not fetch VIX")

    with _mc2:
        if _fg:
            st.metric(
                label=f"{_fg['emoji']} Fear & Greed",
                value=f"{_fg['score']:.0f} / 100",
                delta=_fg["rating"],
                help="0–25 Extreme Fear · 26–45 Fear · 46–55 Neutral · 56–75 Greed · 76–100 Extreme Greed",
            )
        else:
            st.metric("Fear & Greed", "—", help="Could not fetch index")

    with _mc3:
        if _regime:
            _reg_icons = {"STRONG BULL": "🟢", "BULL": "🔵", "NEUTRAL": "🟡",
                          "RECOVERING": "🟠", "BEAR": "🔴", "UNKNOWN": "⚪"}
            _ri = _reg_icons.get(_regime["regime"], "⚪")
            st.metric(
                label=f"{_ri} SPY Regime",
                value=_regime["label"],
                delta=f"vs 200d {_regime['dist_200_pct']:+.1f}%",
                help=_regime["advice"],
            )
        else:
            st.metric("SPY Regime", "—", help="Could not fetch SPY data")

    with _mc4:
        # ── Portfolio Risk Engine ─────────────────────────────────────────────
        try:
            _re_paper = ts.PaperTradingEngine(conn)
            _re_risk  = ts.get_portfolio_risk_status(_re_paper)
            _re_level = _re_risk["risk_level"]
            _re_colors = {
                "NORMAL":         ("#3fb950", "✅"),
                "CAUTION":        ("#e3b341", "⚠️"),
                "HALT":           ("#f85149", "🛑"),
                "EMERGENCY_HALT": ("#a371f7", "🚨"),
            }
            _re_c, _re_e = _re_colors.get(_re_level, ("#8b949e", "❔"))
            st.markdown(
                f"<div style='background:#161b22;border:2px solid {_re_c};"
                f"border-radius:8px;padding:8px 12px;height:100%'>"
                f"<div style='color:{_re_c};font-weight:bold;font-size:0.95em'>"
                f"{_re_e} Risk Engine: {_re_level}</div>"
                f"<div style='font-size:0.8em;color:#c9d1d9;margin-top:2px'>"
                f"Drawdown <b>{_re_risk['drawdown_pct']:.1f}%</b>  ·  "
                f"Sector cap <b>{_re_risk['max_sector_pct']:.0f}%</b>  ·  "
                f"Sizing <b>{_re_risk['size_multiplier']:.2f}×</b></div>"
                f"<div style='font-size:0.75em;color:#8b949e;margin-top:4px'>"
                f"{_re_risk['reason']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        except Exception as _re_exc:
            st.info(f"Risk engine unavailable: {_re_exc}", icon="🛡")

    if _eco:
        _eco_lines = "  |  ".join(
            f"**{e['title']}** {e['date']} {e['time']}" for e in _eco[:4]
        )
        st.warning(
            f"🗓 **High-impact events this week:** {_eco_lines}  "
            "— IV can spike or crush around these. Check before entering.",
            icon="🗓",
        )

    st.divider()

    # ── Ticker + bias controls (shared across all sub-tabs) ───────────────────
    _oc1, _oc2, _oc3 = st.columns([2, 1, 1])
    with _oc1:
        opt_ticker = st.text_input(
            "Ticker symbol", value="SPY", key="opt_ticker_input",
            placeholder="e.g. SPY, AAPL, TSLA",
        ).upper().strip()
    with _oc2:
        opt_bias = st.selectbox("Directional bias", ["Bullish", "Bearish"],
                                key="opt_bias_sel")
    with _oc3:
        opt_contracts = st.number_input("Contracts (paper)", min_value=1,
                                        max_value=50, value=1, step=1,
                                        key="opt_contracts_input")

    (ot_best, ot_short, ot_puts, ot_perf, ot_lab, ot_chain, ot_unusual,
     ot_strategy, ot_paper, ot_smart) = st.tabs([
        "🎯  Best Trades NOW",
        "⚡  Short-Term",
        "🔻  Puts",
        "📊  Performance",
        "🧪  Options Lab",
        "⛓  Chain Viewer",
        "🔥  Unusual Activity",
        "🎯  Strategy Builder",
        "💰  Paper Trades",
        "🏛  Smart Money",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB — PUT ENGINE (bearish, regime-gated)
    # ─────────────────────────────────────────────────────────────────────────
    with ot_puts:
        # ── NEWS-REVERSAL PUTS (the validated put edge) ──────────────────────
        st.markdown("### 📰 News-Reversal Puts — The Validated Put Edge")
        st.caption(
            "Backtest verdict: shorting OVEREXTENDED stocks LOSES — parabolic "
            "(+4.8%/5d) and stretched (+4.3%/5d) names keep RISING ('overbought "
            "gets more overbought'). The real put edge is EVENT-DRIVEN: a stock "
            "gaps down on bad news regardless of the chart. This fires only on a "
            "negative catalyst CONFIRMED by price breaking down."
        )
        if st.button("📰 Scan News-Reversal Puts", key="nrp_btn", type="primary"):
            try:
                from news_reversal_puts import NewsReversalPuts
                with st.spinner("Scanning negative catalysts + RSI2 extremes…"):
                    _nrp = NewsReversalPuts(conn).scan()
                st.session_state["_nrp"] = _nrp
            except Exception as _ne:
                st.error(f"Scan failed: {_ne}")
        _nrp = st.session_state.get("_nrp")
        if _nrp is None:
            st.info("Hit **Scan News-Reversal Puts**. Most days this is empty — it "
                    "only fires on a real negative catalyst that price is confirming.",
                    icon="📰")
        elif not _nrp:
            st.success("No news-reversal put setups firing — correct on a calm day. "
                       "Don't short strength just because it looks 'too high'.", icon="✅")
        else:
            st.warning(f"{len(_nrp)} put setup(s)", icon="🔻")
            for r in _nrp:
                _stt = r["stats"]
                with st.expander(f"🔻 **{r['ticker']}** @ ${r['price']:.2f} — "
                                 f"{_stt['label']}", expanded=True):
                    st.markdown(f"- **Setup:** {_stt['label']} ({_stt['edge']})\n"
                                f"- **Catalyst:** {r['catalyst']}\n"
                                f"- **Horizon:** {_stt['horizon']}")
                    st.caption("News puts are high-variance — you're fighting upward "
                               "drift. The edge is the catalyst + confirmation, not the "
                               "chart looking high. Affordable weekly put suggestion is "
                               "sent to Telegram automatically by the monitor.")
        st.divider()

        st.markdown("### 🔻 Put Engine — Bearish (Regime-Gated)")
        st.caption(
            "The bearish counterpart to the call engines. Backtesting was blunt: "
            "most bearish setups LOSE because markets drift up and weakness bounces. "
            "So puts are GATED to where they actually work — bear/neutral regime, "
            "real catalysts, and the 2 setups that survived testing."
        )
        st.markdown(
            "<div style='background:#3d1414;border-left:5px solid #f85149;"
            "border-radius:6px;padding:10px 14px;margin:6px 0;color:#c9d1d9;font-size:0.88em'>"
            "<b>⚠️ Honest truth from the backtest:</b> shorting breakdowns (+0.84%/3d), "
            "weakness (+0.70%/3d), and death crosses (+1.45%/5d) all <b>made money for "
            "the other side</b> — they bounce. Puts are NOT a standalone trend strategy. "
            "This engine holsters itself in confirmed bull markets and only activates "
            "on bearish regime + catalysts + the 2 validated setups (overbought-in-"
            "downtrend, failed gap-up)."
            "</div>", unsafe_allow_html=True)

        try:
            from put_engine import market_allows_puts, PutEngine
            _gate = market_allows_puts()
        except Exception as _pe:
            _gate = {"allowed": False, "regime": "?", "reason": str(_pe)}

        _gc = "#3fb950" if not _gate.get("allowed") else "#f85149"
        st.markdown(
            f"<div style='background:#161b22;border-left:5px solid {_gc};"
            f"border-radius:6px;padding:10px 14px;margin:6px 0;color:#c9d1d9'>"
            f"<b>Regime gate:</b> {_gate.get('regime','?')}  ·  "
            f"puts <b>{'ACTIVE' if _gate.get('allowed') else 'HOLSTERED'}</b><br>"
            f"<span style='color:#8b949e;font-size:0.88em'>{_gate.get('reason','')}</span>"
            f"</div>", unsafe_allow_html=True)

        _pc1, _pc2 = st.columns([2, 1])
        with _pc1:
            _force = st.checkbox(
                "Force scan (override gate — only for a specific bearish catalyst "
                "you're acting on, e.g. bad CPI print, Fed shock, tariff news)",
                value=False, key="put_force")
        with _pc2:
            _put_btn = st.button("🔻 Scan Puts", key="put_scan_btn",
                                 type="primary", use_container_width=True)

        if _put_btn:
            try:
                _pp = st.progress(0.0, text="Scanning downtrend names…")
                def _p_cb(i, n, t):
                    _pp.progress(min(i/max(n,1), 1.0), text=f"{t} ({i}/{n})")
                with st.spinner("Ranking bearish candidates + selecting puts…"):
                    _put_res = PutEngine(conn).scan(top_n=8, progress=_p_cb,
                                                    force=_force)
                _pp.empty()
                st.session_state["_put_res"] = _put_res
            except Exception as _pse:
                st.error(f"Put scan failed: {_pse}")

        _put_res = st.session_state.get("_put_res")
        if _put_res is None:
            st.info("Hit **Scan Puts**. In a bull market this stays empty by design — "
                    "the discipline is NOT shorting strength.", icon="⏳")
        elif not _put_res.get("plays"):
            if not _put_res.get("allowed"):
                st.success("Puts holstered by the regime gate — correct in a bull "
                           "market. Tick **Force scan** only if you have a specific "
                           "bearish catalyst.", icon="✅")
            else:
                st.info("Regime allows puts but no downtrend candidates with clean "
                        "contracts right now.", icon="📭")
        else:
            st.warning(f"{len(_put_res['plays'])} put play(s) — regime "
                       f"{_put_res.get('regime')}", icon="🔻")
            import pandas as _pdp
            df = _pdp.DataFrame([{
                "Ticker": p["ticker"],
                "6-mo %": f"{p['mom_6m']*100:+.0f}%",
                "Vol%": f"{p.get('realized_vol', 0):.0f}",
                "RSI": f"{p['rsi']:.0f}",
                "Strike": f"${p['contract']['strike']:.0f}P",
                "Expiry": p["contract"]["expiry"],
                "DTE": p["contract"]["dte"],
                "Prem": f"${p['contract']['premium']:.2f}",
                "1 ctr": f"${p['contract']['premium']*100:.0f}",
                "IV": f"{p['contract']['iv']*100:.0f}%",
                "Quality": f"{p['contract'].get('quality_score','?')}",
            } for p in _put_res["plays"]])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Put exits mirror calls: +100% TP / −50% stop. Remember the "
                       "backtest — even in a bear regime, individual-name puts are a "
                       "coin-flip with a fat tail, not a sure thing.")

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB — SHORT-TERM REVERSAL OPTIONS (backtested fat-tail setups)
    # ─────────────────────────────────────────────────────────────────────────
    with ot_short:
        st.markdown("### ⚡ Short-Term Reversal Calls — Backtested Setups")
        st.caption(
            "Short-DTE (1–7 day) calls on the ONLY short-horizon setups that "
            "survived a 3-year backtest with a real fat right tail: market "
            "capitulation + single-name reversals. Uses ATM/slightly-ITM "
            "contracts (high delta) because a 2–3 day hold has no time for a "
            "far-OTM lottery to come good."
        )
        st.markdown(
            "<div style='background:#3d2a0e;border-left:5px solid #e3b341;"
            "border-radius:6px;padding:10px 14px;margin:6px 0;color:#c9d1d9;font-size:0.88em'>"
            "<b>⚠️ Honest expectancy:</b> this is a defined-risk LOTTERY, not a "
            "money printer. The underlying edge is ~1–1.3% over baseline; the "
            "'crazy returns' live only in the right tail (p90 +12% underlying → "
            "+200–400% on the call) and only on rare high-conviction days. Expect "
            "most tickets to lose. Setups that FAILED the backtest (momentum-thrust "
            "chasing, inside-day breaks, oversold bounces) are deliberately excluded."
            "</div>", unsafe_allow_html=True)

        # Backtested edge reference table
        with st.expander("📊 The backtested edge behind each setup", expanded=False):
            st.markdown(
                "| Setup | Horizon | Win% | Forward edge | Right tail |\n"
                "|---|---|---|---|---|\n"
                "| Market capitulation (SPY −5% day) | 1–2d | **83%** | +3.4%/day | calls +200–400% |\n"
                "| Extreme fear (VIX ≥ 40) | 5–20d | **90%** | +8% (20d) | strongest signal |\n"
                "| Gap-down reversal (gap<−2%, RSI<40) | 2–3d | 54% | +1.3% vs base | p90 +12% |\n"
                "| Post-panic bounce (prior −4%, today green) | 2–3d | 51% | +1.0% vs base | p90 +12% |\n"
            )

        if st.button("⚡ Scan Short-Term Setups", key="st_scan_btn",
                     type="primary"):
            try:
                from short_term_options import ShortTermOptionsStrategy
                _sp = st.progress(0.0, text="Scanning for reversal setups…")
                def _st_cb(i, n, t):
                    _sp.progress(min(i / max(n, 1), 1.0), text=f"{t} ({i}/{n})")
                with st.spinner("Checking market panic + single-name reversals…"):
                    _st_plays = ShortTermOptionsStrategy(conn).scan(progress=_st_cb)
                _sp.empty()
                st.session_state["_st_plays"] = _st_plays
            except Exception as _se:
                st.error(f"Scan failed: {_se}")

        _st_plays = st.session_state.get("_st_plays")
        if _st_plays is None:
            st.info("Hit **Scan Short-Term Setups**. Most days this finds nothing — "
                    "the edge only exists on capitulation/reversal days, and the "
                    "discipline is waiting for them.", icon="⏳")
        elif not _st_plays:
            st.success("No short-term reversal setups firing right now — that's the "
                       "correct answer on a calm day. Don't force a trade.", icon="✅")
        else:
            st.success(f"{len(_st_plays)} short-term setup(s) firing", icon="⚡")
            for p in _st_plays:
                c = p["contract"]; stt = p["stats"]
                with st.expander(
                        f"⚡ **{p['ticker']}** — {stt['label']}  ·  "
                        f"${c['strike']:.0f}C exp {c['expiry']} ({c['dte']}d)  "
                        f"${c['premium']:.2f}", expanded=True):
                    st.markdown(
                        f"- **Setup:** {stt['label']}  ·  hist win **{stt['win']}%**  ·  "
                        f"forward {stt['fwd']}\n"
                        f"- **Contract:** {c['strike']:.0f}C exp {c['expiry']} "
                        f"({c['dte']}d, {c['moneyness']:+.1f}% moneyness)  ·  "
                        f"IV {c['iv']*100:.0f}%  ·  vol {c['volume']} / OI {c['open_interest']}\n"
                        f"- **Cost:** ${c['premium']:.2f}/sh = **${c['premium']*100:.0f} per contract**\n"
                        f"- **Exit rules:** +75% take profit · −40% stop · hard time-stop 3 days\n"
                        f"- **Tail:** {stt['tail']}")
        st.caption("Exit discipline is everything here: +75% TP / −40% stop / "
                   "force-close after 3 trading days. The edge is measured over "
                   "1–3 days; holding longer just feeds theta.")

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB 1 — OPTIONS PERFORMANCE TRACKER + EMPIRICAL PROBABILITY
    # ─────────────────────────────────────────────────────────────────────────
    with ot_perf:
        st.markdown("### 📊 Options Performance — Honest Scorecard")
        st.caption(
            "Every bot-originated options trade tracked: live %P&L, days held, "
            "and — critically — return vs the underlying-only hold over the "
            "same period. If the leverage isn't adding alpha, the scorecard "
            "tells you. Empirical win rate from closed trades becomes the "
            "PROBABILITY badge for new setups."
        )

        try:
            from options_performance import OptionsPerformanceTracker
            _opt = OptionsPerformanceTracker(conn)
            _opens  = _opt.get_open(strategies=["momentum_call"])
            _closed = _opt.get_closed(strategies=["momentum_call"], limit=100)
            _summ   = _opt.summary(strategies=["momentum_call"])
            _emp    = _opt.empirical_probability("momentum_call")
        except Exception as _oe:
            _opens, _closed, _summ, _emp = [], [], {}, None
            st.warning(f"Performance tracker unavailable: {_oe}")

        # ── Aggregate scoreboard ─────────────────────────────────────────────
        if _summ and _summ.get("n_closed", 0) > 0:
            c1, c2, c3, c4 = st.columns(4)
            wr = _summ.get("win_rate") or 0
            c1.metric("Win rate", f"{wr:.0f}%",
                      delta=f"{_summ.get('n_winners',0)}W / {_summ.get('n_losers',0)}L")
            c2.metric("Avg winner", f"{_summ.get('avg_winner') or 0:+.0f}%")
            c3.metric("Avg loser",  f"{_summ.get('avg_loser')  or 0:+.0f}%")
            exp = _summ.get("expectancy")
            c4.metric("Expectancy/trade",
                      f"{exp:+.1f}%" if exp is not None else "—",
                      delta=(f"Total P&L ${_summ.get('total_pnl',0):+.0f}"
                             if _summ.get("total_pnl") is not None else None),
                      delta_color=("normal" if (_summ.get('total_pnl') or 0) >= 0 else "inverse"))
            best = _summ.get("best_trade") or {}
            worst = _summ.get("worst_trade") or {}
            st.caption(f"Best: {best.get('ticker','—')} {best.get('ret_pct',0):+.0f}%  ·  "
                       f"Worst: {worst.get('ticker','—')} {worst.get('ret_pct',0):+.0f}%  ·  "
                       f"ROI on capital: "
                       f"{(_summ.get('roi_pct') if _summ.get('roi_pct') is not None else 0):+.1f}%")
        else:
            st.info("No closed trades yet. Stats appear once the bot has "
                    "completed its first options trade.", icon="📭")

        # ── Empirical probability badge for NEW setups ───────────────────────
        if _emp:
            warning = f" ({_emp['warning']})" if _emp.get("warning") else ""
            st.markdown(
                f"<div style='background:#161b22;border-left:5px solid #58a6ff;"
                f"border-radius:6px;padding:10px 14px;margin:6px 0;color:#c9d1d9'>"
                f"<b>Empirical probability for new momentum_call setups:</b> "
                f"win rate <b>{_emp['win_rate']}%</b> · avg winner "
                f"<b>{_emp['avg_winner']:+.0f}%</b> · avg loser "
                f"<b>{_emp['avg_loser']:+.0f}%</b> · expectancy "
                f"<b>{_emp['expectancy']:+.1f}%</b> per trade · n={_emp['n']}{warning}"
                f"</div>", unsafe_allow_html=True)
        elif _summ.get("n_closed", 0) > 0:
            st.caption(f"Empirical probability appears after 5+ closed trades "
                       f"(currently {_summ['n_closed']}).")

        # ── Open positions table ────────────────────────────────────────────
        st.markdown("### 🟢 Open Options Positions")
        if _opens:
            import pandas as _pdop
            df = _pdop.DataFrame([{
                "Ticker":     o["ticker"],
                "Contract":   f"${o['strike']:.0f}{o['option_type'][0].upper()}",
                "Expiry":     o["expiry"],
                "DTE":        o["dte_remaining"],
                "Days held":  o["days_held"],
                "Entry":      f"${o['entry_price']:.2f}",
                "Now":        f"${o['current_premium']:.2f}",
                "%P&L":       f"{o['ret_pct']:+.1f}%",
                "$P&L":       f"${o['pnl_dollars']:+.0f}",
                "Underlying %": f"{o['underlying_ret_pct']:+.1f}%",
                "Leverage α": f"{o['leverage_alpha_pct']:+.1f}pt",
            } for o in _opens])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("**Leverage α** = how much the option position is "
                       "outperforming just owning the stock. Negative means "
                       "you'd have been better off in the underlying.")
        else:
            st.info("No open options positions.", icon="📭")

        # ── Closed positions table ──────────────────────────────────────────
        st.markdown("### 📜 Closed Options Trades")
        if _closed:
            import pandas as _pdcl
            df = _pdcl.DataFrame([{
                "Entry":     c["entry_date"],
                "Exit":      c["exit_date"],
                "Held":      c["hold_days"],
                "Ticker":    c["ticker"],
                "Contract":  f"${c['strike']:.0f}{c['option_type'][0].upper()}",
                "Entry$":    f"${c['entry_price']:.2f}",
                "Exit$":     f"${c['exit_price']:.2f}",
                "%P&L":      f"{c['ret_pct']:+.1f}%",
                "$P&L":      f"${c['net_pnl']:+.0f}",
                "Reason":    c["exit_reason"],
            } for c in _closed])
            st.dataframe(df, use_container_width=True, hide_index=True, height=300)
        else:
            st.info("No closed trades yet.", icon="📭")

        st.divider()
        st.markdown(
            "<small style='color:#8b949e'>"
            "<b>Event exits ON:</b> the monitor closes options on these external "
            "factors → VIP negative post · high-impact negative news · earnings "
            "within 5 days (creep) · underlying breaks 50-SMA · VIX spike to "
            "FEAR (≥25). Each exit fires a Telegram alert with the reason."
            "</small>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB 0 — BEST OPTIONS TRADES NOW (the meta-scanner)
    # ─────────────────────────────────────────────────────────────────────────
    with ot_best:
        st.markdown("### 🎯 Best Options Trades — Right Now")
        st.caption(
            "ONE ranked list that pulls underlying candidates from EVERY signal "
            "source (momentum leaders · PEAD · whale picks · VIP-mentioned · "
            "movers), picks the best contract per name, scores via the quality "
            "engine (IVR · IV-RV · expected move · Greeks · UOA). Telegram "
            "fires for new A-grade setups automatically."
        )
        _bc1, _bc2, _bc3 = st.columns([3, 1, 1])
        with _bc2:
            _bs_min = st.number_input("Min quality", 0, 100, 50, key="bs_min")
        with _bc3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            _bs_run = st.button("🎯 Run Scan", key="bs_btn",
                                type="primary", use_container_width=True)
        with _bc1:
            st.info(
                "Scan takes ~60–90s (pulls option chains + scores 20–40 "
                "candidates). The monitor also runs this hourly during market "
                "hours — Telegram alerts fire for any NEW setup with quality ≥ 70.",
                icon="📡",
            )

        if _bs_run:
            try:
                from options_scanner import OptionsScanner
                _bp = st.progress(0.0, text="Gathering candidates from all sources…")
                def _bs_cb(i, n, t):
                    _bp.progress(min(i / max(n, 1), 1.0), text=f"Scoring {t} ({i}/{n})")
                with st.spinner("Pulling chains + scoring across all signal sources…"):
                    _bs_res = OptionsScanner(conn).scan(
                        min_quality=int(_bs_min), progress=_bs_cb)
                    OptionsScanner(conn).persist(_bs_res)
                _bp.empty()
                st.success(f"Scan complete: {len(_bs_res)} setup(s) "
                           f"at quality ≥ {_bs_min}", icon="🎯")
            except Exception as _bse:
                st.error(f"Scan failed: {_bse}")

        try:
            from options_scanner import OptionsScanner as _OS
            _bs_show = _OS(conn).get_latest()
        except Exception:
            _bs_show = []

        if _bs_show:
            import pandas as _pdb
            # Live-compute thesis math for display (cheap — just expected_move)
            def _enrich(r):
                try:
                    from options_analytics import expected_move
                    em = expected_move(r["ticker"], r["expiry"]) or {}
                    em_pct = em.get("exp_move_pct") or 0
                    thesis = r.get("thesis_pct") or 0
                    edge_x = (thesis / em_pct) if em_pct > 0 else 0
                    return em_pct, edge_x
                except Exception:
                    return 0, 0
            _enriched = [(r, *_enrich(r)) for r in _bs_show]

            df = _pdb.DataFrame([{
                "Score":     r["quality_score"],
                "Grade":     r["quality_grade"],
                "Ticker":    r["ticker"],
                "Strike":    f"${r['strike']:.0f}",
                "Type":      r["option_type"].upper(),
                "Expiry":    r["expiry"],
                "DTE":       r["dte"],
                "Prem":      f"${r['premium']:.2f}",
                "1 contract": f"${r['premium']*100:.0f}",
                "IV":        f"{r['iv_pct']:.0f}%",
                "Thesis":    f"{r.get('thesis_pct',0):.1f}%",
                "Implied":   f"{em_pct:.1f}%",
                "Edge ×":    f"{edge_x:.1f}×" if edge_x else "—",
                "Under":     f"${r['underlying_price']:.2f}",
                "Sources":   " · ".join(r["sources"]),
                "Decision":  r["decision"],
            } for (r, em_pct, edge_x) in _enriched])
            st.dataframe(df, use_container_width=True, hide_index=True, height=420)
            st.caption(
                "**Thesis** = expected underlying move over the option's life "
                "(3-mo-equiv of momentum, ≥ 8% floor).  **Implied** = what the "
                "ATM straddle is pricing in.  **Edge ×** = thesis ÷ implied — "
                "≥ 1.5× = real edge, < 1.0× = seller-favored.  Score now includes "
                "a capital-efficiency component ($/100/day) so slow-grind trades "
                "get filtered out."
            )

            # Per-setup expanders with the contract symbol + alert status
            st.markdown("### 🔎 Per-Trade Detail")
            for r in _bs_show[:15]:
                _g_color = {"A+": "#3fb950", "A": "#3fb950", "B": "#58a6ff",
                            "C": "#e3b341"}.get(r["quality_grade"], "#8b949e")
                _alert_tag = "🔔 alerted" if r.get("alerted") else "·"
                with st.expander(
                        f"{r['quality_grade']}  **{r['ticker']}** ${r['strike']:.0f}"
                        f"{r['option_type'][0].upper()} exp {r['expiry']}  "
                        f"·  score {r['quality_score']}/100  ·  {_alert_tag}",
                        expanded=False,
                ):
                    st.markdown(
                        f"<div style='background:#161b22;border-left:4px solid {_g_color};"
                        f"border-radius:6px;padding:10px 14px;margin-bottom:6px;color:#c9d1d9'>"
                        f"<b style='color:{_g_color}'>{r['decision']}</b> · "
                        f"contract <code>{r['contract_symbol']}</code><br>"
                        f"<b>Sources:</b> {' · '.join(r['sources'])}<br>"
                        f"<b>Premium:</b> ${r['premium']:.2f} per share "
                        f"(= <b>${r['premium']*100:.0f} per contract</b>)<br>"
                        f"<b>Underlying:</b> ${r['underlying_price']:.2f}  ·  "
                        f"<b>IV:</b> {r['iv_pct']:.0f}%  ·  "
                        f"<b>DTE:</b> {r['dte']}d  ·  "
                        f"<b>Thesis:</b> {r['thesis_pct']:.1f}% move"
                        f"</div>", unsafe_allow_html=True,
                    )
            st.caption("Grade scale: **A+** = exceptional · **A** = strong setup · "
                       "**B** = solid · **C** = marginal · **D/F** = filtered out. "
                       "Quality ≥ 70 = Telegram alert.")
        else:
            st.info("No scan results yet. Hit **Run Scan** above, or wait for "
                    "the monitor's hourly auto-scan during market hours.", icon="📭")

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB 0 — OPTIONS LAB (the new analytics)
    # ─────────────────────────────────────────────────────────────────────────
    with ot_lab:
        st.markdown("### 🧪 Options Lab — IV Rank · Expected Move · Greeks · Flow")
        st.caption(
            "The data the bot uses to filter options trades. IV Rank tells you "
            "if vol is cheap or expensive. Expected Move shows what the market "
            "is pricing in. IV-RV spread is the Goyal-Saretto cheap-options "
            "edge. Greeks are real risk. UOA is smart-money flow."
        )
        _ol_t = (opt_ticker or "NVDA").strip().upper()
        if st.button("🧪 Analyze", key="ol_btn", type="primary"):
            try:
                from options_analytics import full_report
                with st.spinner(f"Pulling chain + computing analytics for {_ol_t}…"):
                    _ol = full_report(conn, _ol_t)
                st.session_state["_ol_last"] = _ol
            except Exception as _ole:
                st.error(f"Analysis failed: {_ole}")
        _ol = st.session_state.get("_ol_last")

        if _ol:
            _ivr = _ol.get("iv_rank") or {}
            _em  = _ol.get("exp_move") or {}
            _rv  = _ol.get("iv_rv") or {}
            _uoa = _ol.get("uoa") or []

            # IV Rank
            st.markdown(f"#### IV Analysis — {_ol['ticker']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("IV now", f"{_ivr.get('iv_now', 0):.1f}%")
            c2.metric("IV Rank", f"{_ivr.get('iv_rank_pct', 0):.0f}%",
                      delta=("low — buy zone"  if _ivr.get('iv_rank_pct', 100) <= 30 else
                             "medium"          if _ivr.get('iv_rank_pct', 0) <= 70 else
                             "HIGH — avoid buys"),
                      delta_color=("normal" if _ivr.get('iv_rank_pct', 100) <= 50 else "inverse"))
            c3.metric("Source",
                      _ivr.get("source", "?"),
                      delta=f"{_ivr.get('n', 0)} snapshots")
            c4.metric("52-wk range",
                      f"{_ivr.get('iv_min', 0):.0f}–{_ivr.get('iv_max', 0):.0f}%")

            # Expected Move
            if _em:
                st.markdown(f"#### Expected Move ({_em.get('dte', 0)}d to "
                            f"{_em.get('expiry', '')})")
                c1, c2, c3 = st.columns(3)
                c1.metric("Spot", f"${_em['spot']:.2f}")
                c2.metric("ATM Straddle", f"${_em['straddle_price']:.2f}",
                          delta=f"±{_em['exp_move_pct']:.1f}%")
                c3.metric("Implied range",
                          f"${_em['lower']:.2f} – ${_em['upper']:.2f}")
                st.caption(f"Your thesis must move the stock MORE than "
                           f"±{_em['exp_move_pct']:.1f}% over {_em['dte']}d to "
                           f"beat the straddle. Less than that, sellers win.")

            # IV-RV Spread
            if _rv:
                _sig_color = {"cheap": "#3fb950", "neutral": "#58a6ff",
                              "expensive": "#e3b341", "rich": "#f85149"}.get(_rv.get("signal"), "#8b949e")
                _sig_msg = {"cheap": "BUYER FAVORED — Goyal-Saretto edge",
                            "neutral": "Fairly priced",
                            "expensive": "Seller favored",
                            "rich": "VOL TOO RICH — avoid long premium"}.get(_rv.get("signal"), "")
                st.markdown(
                    f"<div style='background:#161b22;border-left:5px solid {_sig_color};"
                    f"border-radius:6px;padding:10px 14px;margin:6px 0'>"
                    f"<b>IV-RV Spread:</b> IV {_rv['iv_pct']}% vs realized {_rv['rv30_pct']}% "
                    f"= <b>{_rv['spread_pct']:+.1f}pt</b>  →  <b style='color:{_sig_color}'>"
                    f"{_rv.get('signal','').upper()}</b>  ·  {_sig_msg}"
                    f"</div>", unsafe_allow_html=True)

            # UOA
            st.markdown("#### Unusual Options Activity (smart-money flow)")
            if _uoa:
                import pandas as _pdol
                df = _pdol.DataFrame([{
                    "Type":      u["type"].upper(),
                    "Strike":    f"${u['strike']:.2f}",
                    "Expiry":    u["expiry"],
                    "Vol":       u["volume"],
                    "OI":        u["open_interest"],
                    "Vol/OI":    f"{u['vol_oi_ratio']:.1f}×",
                    "Premium":   f"${u['premium']:.2f}",
                    "IV":        f"{u['iv_pct']:.0f}%",
                } for u in _uoa])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption("Vol/OI > 3× = unusual. Concentrated calls = bullish positioning; "
                           "concentrated puts = bearish or hedge.")
            else:
                st.info("No unusual flow detected on the nearest 4 expiries.", icon="📭")
        else:
            st.info(f"Enter a ticker above and hit **Analyze** to see IV Rank, "
                    f"Expected Move, IV-RV spread, and Unusual Activity for any "
                    f"name. Defaults to {_ol_t}.", icon="🧪")

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB 1 — CHAIN VIEWER
    # ─────────────────────────────────────────────────────────────────────────
    with ot_chain:
        st.markdown('<div class="section-hdr">Full Options Chain + Greeks</div>',
                    unsafe_allow_html=True)
        if not opt_ticker:
            st.info("Enter a ticker above to load the chain.", icon="⛓")
        else:
            @st.cache_data(ttl=120, show_spinner=False)
            def _load_chain(tk, exp):
                return ts.OptionsChainAnalyzer(tk).get_chain(exp)

            @st.cache_data(ttl=3600, show_spinner=False)
            def _load_exps(tk):
                return ts.OptionsChainAnalyzer(tk).expirations()

            @st.cache_data(ttl=3600, show_spinner=False)
            def _load_ivrank(tk):
                return ts.OptionsChainAnalyzer(tk).iv_rank()

            with st.spinner(f"Loading expiries for {opt_ticker}…"):
                _exps = _load_exps(opt_ticker)

            if not _exps:
                st.warning(f"No options data found for **{opt_ticker}**. "
                           "Try a different ticker.", icon="⚠️")
            else:
                _cc1, _cc2 = st.columns([3, 1])
                with _cc1:
                    selected_expiry = st.selectbox("Expiry", _exps,
                                                   key="chain_expiry_sel")
                with _cc2:
                    chain_side = st.radio("Show", ["Calls", "Puts", "Both"],
                                         horizontal=True, key="chain_side_radio")

                with st.spinner(f"Fetching {opt_ticker} chain…"):
                    _chain_data = _load_chain(opt_ticker, selected_expiry)

                if not _chain_data:
                    st.error("Could not load chain data. Try again.", icon="❌")
                else:
                    spot    = _chain_data.get("spot", 0)
                    dte_val = _chain_data.get("dte", 0)
                    mp      = ts.OptionsChainAnalyzer(opt_ticker).max_pain(selected_expiry)
                    cp_r    = ts.OptionsChainAnalyzer(opt_ticker).cp_ratio(selected_expiry)

                    # ── IV rank ribbon ────────────────────────────────────────
                    _ivr = _load_ivrank(opt_ticker)
                    _m1, _m2, _m3, _m4, _m5 = st.columns(5)
                    _m1.metric("Spot", f"${spot:.2f}")
                    _m2.metric("DTE", dte_val)
                    _m3.metric("Max Pain", f"${mp:.2f}" if mp else "—")
                    _m4.metric("C/P Ratio", f"{cp_r:.2f}" if cp_r else "—",
                               help=">2 = bullish flow  |  <0.5 = bearish flood")
                    _m5.metric(
                        "IV Rank",
                        f"{_ivr.get('iv_rank', '—')}%" if _ivr else "—",
                        help="0=cheap vol  |  100=expensive vol",
                    )

                    # ── Chain table ───────────────────────────────────────────
                    def _fmt_chain(df, side):
                        keep = ["strike", "mid", "bid", "ask", "volume",
                                "openInterest", "iv_pct",
                                "delta", "gamma", "theta", "vega",
                                "itm", "moneyness"]
                        df = df[[c for c in keep if c in df.columns]].copy()
                        rename = {
                            "strike": "Strike", "mid": "Mid", "bid": "Bid",
                            "ask": "Ask", "volume": "Vol",
                            "openInterest": "OI", "iv_pct": "IV %",
                            "delta": "Δ", "gamma": "Γ", "theta": "Θ",
                            "vega": "V", "itm": "ITM", "moneyness": "M/S",
                        }
                        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
                        return df

                    if chain_side in ("Calls", "Both"):
                        st.markdown("**📈 CALLS**")
                        c_df = _fmt_chain(_chain_data["calls"], "call")
                        st.dataframe(
                            c_df.style.apply(
                                lambda row: ["background-color:#1a3a1a" if row.get("ITM") else "" for _ in row],
                                axis=1
                            ) if "ITM" in c_df.columns else c_df,
                            use_container_width=True, hide_index=True,
                        )

                    if chain_side in ("Puts", "Both"):
                        st.markdown("**📉 PUTS**")
                        p_df = _fmt_chain(_chain_data["puts"], "put")
                        st.dataframe(p_df, use_container_width=True, hide_index=True)

                    # ── Open Interest bar chart ───────────────────────────────
                    st.markdown('<div class="section-hdr">Open Interest by Strike</div>',
                                unsafe_allow_html=True)
                    c_oi = _chain_data["calls"][["strike", "openInterest"]].copy()
                    p_oi = _chain_data["puts"][["strike",  "openInterest"]].copy()
                    c_oi.columns = ["Strike", "OI"]
                    p_oi.columns = ["Strike", "OI"]
                    fig_oi = go.Figure()
                    fig_oi.add_bar(x=c_oi["Strike"], y=c_oi["OI"],
                                   name="Calls", marker_color="#3fb950")
                    fig_oi.add_bar(x=p_oi["Strike"], y=-p_oi["OI"],
                                   name="Puts",  marker_color="#f85149")
                    if spot:
                        fig_oi.add_vline(x=spot, line_color="#e3b341",
                                         line_dash="dash",
                                         annotation_text=f"Spot ${spot:.0f}")
                    if mp:
                        fig_oi.add_vline(x=mp, line_color="#58a6ff",
                                         line_dash="dot",
                                         annotation_text=f"Max Pain ${mp:.0f}")
                    fig_oi.update_layout(
                        barmode="overlay", bargap=0,
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font=dict(color="#c9d1d9"), height=320,
                        xaxis=dict(gridcolor="#21262d", title="Strike"),
                        yaxis=dict(gridcolor="#21262d", title="Open Interest"),
                        legend=dict(bgcolor="#161b22"),
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_oi, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB 2 — UNUSUAL ACTIVITY SCANNER
    # ─────────────────────────────────────────────────────────────────────────
    with ot_unusual:
        st.markdown('<div class="section-hdr">🔥 Unusual Options Flow — 0DTE to 7DTE Only</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Scans **0DTE–7DTE contracts** for institutional flow signals — "
            "focus on 5–7 DTE (weekly) contracts that confirm your breakout play. "
            "Tags: SWEEP (vol > 5× OI = new money) · BLOCK (>1,000 contracts) · "
            "GAMMA_RIP (OTM call with volume) · PUT_SWEEP (bearish institutional hedge)."
        )

        # ── Ticker source ─────────────────────────────────────────────────────
        _ua_src = st.radio(
            "Ticker source",
            ["📡 Latest scan results", "✏️ Enter manually"],
            horizontal=True, key="ua_src_radio",
        )

        if _ua_src == "📡 Latest scan results":
            # Auto-load tickers from the most recent breakout scan
            @st.cache_data(ttl=120, show_spinner=False)
            def _get_scan_tickers_for_ua(_conn_id):
                try:
                    row = conn.execute(
                        "SELECT scan_id FROM calls ORDER BY scan_timestamp DESC LIMIT 1"
                    ).fetchone()
                    if not row:
                        return []
                    rows = conn.execute(
                        "SELECT ticker FROM calls WHERE scan_id=? "
                        "ORDER BY explosive_score DESC LIMIT 15",
                        (row[0],)
                    ).fetchall()
                    return [r[0] for r in rows if r[0]]
                except Exception:
                    return []

            _scan_tks = _get_scan_tickers_for_ua(id(conn))
            if _scan_tks:
                st.info(
                    f"📡 **{len(_scan_tks)} tickers** loaded from latest scan: "
                    + ", ".join(_scan_tks),
                    icon="📡",
                )
                scan_tickers_raw = ", ".join(_scan_tks)
            else:
                st.warning("No scan results found. Run a breakout scan first, or switch to manual entry.", icon="⚠️")
                scan_tickers_raw = opt_ticker
        else:
            scan_tickers_raw = st.text_input(
                "Tickers to scan (comma-separated)",
                value=opt_ticker,
                key="unusual_scan_tickers_manual",
                placeholder="SPY, QQQ, AAPL, TSLA, NVDA",
            )

        scan_btn = st.button("🔍 Scan 0DTE–7DTE Flow Now", use_container_width=False,
                             key="unusual_scan_btn")

        if scan_btn:
            scan_list = [t.strip().upper() for t in scan_tickers_raw.split(",") if t.strip()]
            if not scan_list:
                st.warning("Enter at least one ticker.", icon="⚠️")
            else:
                with st.spinner(f"Scanning {len(scan_list)} ticker(s) for 0DTE–7DTE unusual flow…"):
                    scanner_ua = ts.AggressiveOptionsScanner()
                    alerts     = scanner_ua.scan_many(scan_list)

                if not alerts:
                    st.info(
                        "No unusual activity found in 0–7 DTE contracts. "
                        "Markets may be quiet, or no short-dated expiries exist yet today.",
                        icon="🔇",
                    )
                else:
                    st.success(f"**{len(alerts)} alert(s) found!**", icon="🔥")

                    rows_ua = []
                    for a in alerts:
                        bias_icon = "🟢" if a["bias"] == "BULLISH" else "🔴"
                        tag_str   = a.get("tags", "")
                        rows_ua.append({
                            "Ticker":  a["ticker"],
                            "Bias":    f"{bias_icon} {a['bias']}",
                            "Type":    a["type"],
                            "Strike":  f"${a['strike']:.2f}",
                            "Expiry":  a["expiry"],
                            "DTE":     a["dte"],
                            "Volume":  f"{a['volume']:,}",
                            "OI":      f"{a['open_interest']:,}",
                            "Vol/OI":  f"{a['vol_oi_ratio']:.1f}×",
                            "Mid":     f"${a['mid_price']:.2f}",
                            "IV %":    f"{a['iv_pct']:.1f}%",
                            "Tags":    tag_str,
                        })
                    st.dataframe(pd.DataFrame(rows_ua),
                                 use_container_width=True, hide_index=True)

                    # ── Volume bar chart ──────────────────────────────────────
                    fig_ua = go.Figure()
                    for a in alerts[:20]:
                        color = "#3fb950" if a["bias"] == "BULLISH" else "#f85149"
                        lbl   = f"{a['ticker']} ${a['strike']:.0f}{a['type'][0]} ({a['dte']}d)"
                        fig_ua.add_bar(x=[lbl], y=[a["volume"]],
                                       marker_color=color,
                                       text=a.get("tags", ""),
                                       textposition="outside",
                                       name=a.get("tags", ""))
                    fig_ua.update_layout(
                        showlegend=False,
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font=dict(color="#c9d1d9"), height=320,
                        xaxis=dict(gridcolor="#21262d", tickangle=-35),
                        yaxis=dict(gridcolor="#21262d", title="Volume (contracts)"),
                        margin=dict(l=10, r=10, t=30, b=80),
                        title=dict(text="0DTE–7DTE Unusual Flow — Top Contracts by Volume",
                                   font=dict(color="#c9d1d9")),
                    )
                    st.plotly_chart(fig_ua, use_container_width=True)

        else:
            st.info("Choose a ticker source and press **Scan** to detect short-dated unusual flow.", icon="🔍")

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB 3 — STRATEGY BUILDER
    # ─────────────────────────────────────────────────────────────────────────
    with ot_strategy:
        st.markdown('<div class="section-hdr">🎯 Strategy Builder — Weekly Plays (5–7 DTE Focus)</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Generates three plays per ticker targeting the **current week's Friday expiry (~5–7 DTE)**. "
            "**Weekly ATM** (full delta, highest probability) · "
            "**Weekly OTM** (2% out-of-money, cheaper entry, needs bigger move) · "
            "**Mid-Week ATM** (2–4 DTE, faster decay — use only with high conviction). "
            "Earnings warnings are shown automatically for every play."
        )

        # ── Mode selector ─────────────────────────────────────────────────────
        _sb_mode = st.radio(
            "Build from",
            ["📡 Latest scan results (all tickers)", "🎯 Single ticker"],
            horizontal=True, key="strat_mode_radio",
        )

        ops_engine_strat = ts.OptionsPaperEngine(conn)

        def _render_strategy_cards(ticker, suggestions, scan_meta=None):
            """Render enriched strategy cards with PoP, EQ, Conviction, and Sizing."""
            if not suggestions:
                st.warning(
                    f"No 0–7 DTE plays available for **{ticker}** "
                    "(market may be closed or no short-dated expiries exist).",
                    icon="⚠️",
                )
                return

            # ── Quick position sizing inputs (once per ticker render) ─────────
            _vix_data    = _ctx.get("vix")
            _closed_hist = ops_engine_strat.get_positions("CLOSED")

            for sg in suggestions:
                cost_str = (f"${sg['entry_cost']:.2f}"
                            if isinstance(sg.get("entry_cost"), (int, float)) else "—")
                prof_str = (f"${sg['max_profit']:.2f}"
                            if isinstance(sg.get("max_profit"), (int, float))
                            else str(sg.get("max_profit", "—")))
                loss_str = (f"${sg['max_loss']:.2f}"
                            if isinstance(sg.get("max_loss"), (int, float)) else "—")
                dte_badge = (
                    f"{'🔴' if sg.get('dte', 99) == 0 else '🟡' if sg.get('dte', 99) <= 3 else '🟢'} "
                    f"{sg.get('dte', '?')} DTE"
                )
                pop_val = sg.get("pop")
                pop_str = f"{pop_val:.0f}%" if pop_val is not None else "—"

                exp_title = (
                    f"**{ticker}** · {sg['strategy']}  {dte_badge}  "
                    f"| Cost {cost_str}  Max Loss {loss_str}  Max Profit {prof_str}"
                    f"  PoP {pop_str}"
                )
                with st.expander(exp_title, expanded=True):

                    # ── Scan meta ─────────────────────────────────────────────
                    if scan_meta:
                        _mc1, _mc2, _mc3 = st.columns(3)
                        _mc1.metric("Breakout Score",
                                    f"{scan_meta.get('explosive_score', 0):.0f}")
                        _mc2.metric("Breakout Prob",
                                    f"{scan_meta.get('breakout_prob', 0):.0f}%")
                        _mc3.metric("Pattern",
                                    scan_meta.get("pattern_detected", "—") or "—")
                        st.divider()

                    # ── Earnings warning ──────────────────────────────────────
                    _expiry_str = sg.get("expiry", "")
                    if _expiry_str:
                        try:
                            _ew = ts.check_earnings_in_window(ticker, _expiry_str)
                            if _ew.get("has_earnings"):
                                st.error(_ew["warning"], icon="⚠️")
                        except Exception:
                            pass

                    # ── Trade legs ────────────────────────────────────────────
                    st.markdown("**Trade:**")
                    for leg in sg.get("legs", []):
                        st.code(leg)

                    # ── Row 1: Core metrics ───────────────────────────────────
                    _r1a, _r1b, _r1c, _r1d = st.columns(4)
                    _r1a.metric("Entry Cost", cost_str)
                    _r1b.metric("Max Loss",   loss_str)
                    _r1c.metric("Max Profit", prof_str)
                    _r1d.metric("DTE",        sg.get("dte", "—"))

                    # ── Row 2: Probability & EV metrics ──────────────────────
                    _r2a, _r2b, _r2c, _r2d = st.columns(4)
                    _r2a.metric(
                        "PoP at Expiry",
                        f"{sg['pop']:.0f}%" if sg.get("pop") is not None else "—",
                        help="Probability of Profit: odds of finishing beyond break-even at expiry",
                    )
                    _r2b.metric(
                        "Prob ITM",
                        f"{sg.get('prob_itm', 50):.0f}%",
                        help="Approximate probability the option expires in-the-money (≈ |delta|)",
                    )
                    _r2c.metric(
                        "Break-even Move",
                        f"{sg.get('breakeven_move_pct', 0):.1f}%",
                        help="Stock must move this % from current price to break even at expiry",
                    )
                    ev_sig = sg.get("ev_signal", "—")
                    _r2d.metric(
                        "EV Signal",
                        ev_sig,
                        help="Favorable = IV Rank < 30 (cheap options). Unfavorable = IV Rank > 60.",
                    )

                    st.divider()

                    # ── Dynamic Sizing recommendation ─────────────────────────
                    _mid_price = sg.get("mid", 0)
                    if _mid_price > 0:
                        _summ_now = ops_engine_strat.get_summary()
                        _sizing   = ts.get_position_sizing(
                            account_cash  = _summ_now["available_cash"],
                            mid_price     = _mid_price,
                            vix_data      = _vix_data,
                            closed_trades = _closed_hist,
                        )
                        _sz_color = "#3fb950" if _sizing["contracts"] > 0 else "#f85149"
                        st.markdown(
                            f"<div style='background:#161b22;border:1px solid #30363d;"
                            f"border-radius:6px;padding:8px 12px;margin:4px 0'>"
                            f"<b>📐 Sizing:</b> "
                            f"<span style='color:{_sz_color}'>{_sizing['advice']}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        _rec_contracts = _sizing.get("contracts", 1)
                    else:
                        _rec_contracts = opt_contracts

                    # ── On-demand Analysis: MASTER SCORE (unified) ──────────
                    _ana_key = (
                        f"analysis_{ticker}_{sg['strategy'].replace(' ','_')}"
                        f"_{sg.get('expiry','')}"
                    )
                    _ana_btn_key = f"analyze_btn_{_ana_key}"
                    if st.button("🎯 Compute MASTER SCORE",
                                 key=_ana_btn_key,
                                 help="Unified 0-100 score: scan + entry quality + Wyckoff + "
                                      "volume + multi-timeframe + market regime + VIX + earnings"):
                        with st.spinner(f"Running full Master Score analysis on {ticker}…"):
                            try:
                                _eq    = ts.get_entry_quality(ticker, opt_bias.lower())
                                _master = ts.compute_master_score(
                                    ticker, expiry=sg.get("expiry", ""),
                                    bias=opt_bias.lower(), conn=conn,
                                    vix_data=_vix_data,
                                    market_regime=_regime,
                                    entry_quality=_eq,
                                )
                                st.session_state[_ana_key] = {
                                    "eq": _eq, "master": _master,
                                }
                            except Exception as _ae:
                                st.warning(f"Analysis error: {_ae}", icon="⚠️")

                    if _ana_key in st.session_state:
                        _ana    = st.session_state[_ana_key]
                        _eq     = _ana.get("eq",     {})
                        _master = _ana.get("master", {})

                        # ── MASTER SCORE banner ─────────────────────────────
                        _ms_score = _master.get("score", 0)
                        _ms_grade = _master.get("grade", "?")
                        _ms_dec   = _master.get("decision", "?")
                        _ms_mult  = _master.get("size_multiplier", 0)
                        _ms_color = {
                            "A+": "#3fb950", "A": "#3fb950", "B": "#58a6ff",
                            "C":  "#e3b341", "D": "#f85149", "F": "#f85149",
                        }.get(_ms_grade, "#8b949e")
                        _dec_emoji = {"BUY": "✅", "WATCH": "👁", "SKIP": "🛑"}.get(_ms_dec, "?")

                        st.markdown(
                            f"<div style='background:linear-gradient(90deg,#161b22,#0d1117);"
                            f"border:3px solid {_ms_color};border-radius:12px;"
                            f"padding:16px 22px;margin:8px 0'>"
                            f"<div style='font-size:0.9em;color:#8b949e'>MASTER SCORE</div>"
                            f"<div style='font-size:2.5em;font-weight:bold;color:{_ms_color};"
                            f"line-height:1.1'>"
                            f"{_ms_score:.0f}<span style='font-size:0.5em;color:#8b949e'>/100</span>  "
                            f"<span style='font-size:0.6em'>{_ms_grade}</span></div>"
                            f"<div style='font-size:1.1em;color:{_ms_color};margin-top:4px'>"
                            f"{_dec_emoji} <b>{_ms_dec}</b>  ·  Size {_ms_mult:.1f}×</div>"
                            f"<div style='color:#c9d1d9;margin-top:6px'>"
                            f"{_master.get('summary','')}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # ── Component breakdown ─────────────────────────────
                        st.markdown("**Component breakdown:**")
                        _comp_names = {
                            "breakout_scan":   "Breakout Scan",
                            "entry_quality":   "Entry Quality",
                            "wyckoff":         "Wyckoff Phase",
                            "inst_volume":     "Institutional Volume",
                            "multi_tf":        "Multi-Timeframe",
                            "market_regime":   "SPY Regime",
                            "vix":             "VIX Environment",
                            "earnings_safety": "Earnings Safety",
                        }
                        for _ck, _cv in (_master.get("components") or {}).items():
                            _pts = _cv.get("pts", 0)
                            _max = _cv.get("max", 1)
                            _pct = int(_pts / _max * 100) if _max else 0
                            _bc  = ("#3fb950" if _pct >= 70
                                    else "#e3b341" if _pct >= 40 else "#f85149")
                            _nm  = _comp_names.get(_ck, _ck)
                            st.markdown(
                                f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
                                f"<div style='width:150px;font-size:0.8em;color:#8b949e'>{_nm}</div>"
                                f"<div style='flex:1;background:#21262d;border-radius:3px;height:9px'>"
                                f"<div style='width:{_pct}%;background:{_bc};height:9px;border-radius:3px'></div></div>"
                                f"<div style='width:55px;font-size:0.78em;text-align:right'>{_pts}/{_max}</div>"
                                f"</div>"
                                f"<div style='font-size:0.72em;color:#6e7681;margin-left:158px'>{_cv.get('label','')}</div>",
                                unsafe_allow_html=True,
                            )

                        # ── Entry Quality sub-detail (optional expander) ───
                        if _eq:
                            with st.expander("📊 Entry Quality sub-signals (used inside Master Score)", expanded=False):
                                _eq_score = _eq.get("score", 0)
                                _eq_grade = _eq.get("grade", "—")
                                _eq_color = ("#3fb950" if _eq_grade == "A" else
                                             "#58a6ff" if _eq_grade == "B" else
                                             "#e3b341" if _eq_grade == "C" else "#f85149")
                                st.markdown(
                                    f"<div style='color:{_eq_color}'><b>Entry Quality {_eq_score}/100 {_eq_grade}</b> — "
                                    f"{_eq.get('advice','')}</div>",
                                    unsafe_allow_html=True,
                                )
                                _max_map = {"time_of_day": 30, "vwap": 25,
                                            "extension": 20, "volume": 15, "iv_rank": 10}
                                for _ck, _cv in (_eq.get("components") or {}).items():
                                    _pts = _cv.get("pts", 0)
                                    _max = _max_map.get(_ck, 20)
                                    _pct = int(_pts / _max * 100) if _max else 0
                                    _bar_c = ("#3fb950" if _pct >= 70 else
                                              "#e3b341" if _pct >= 40 else "#f85149")
                                    st.markdown(
                                        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
                                        f"<div style='width:90px;font-size:0.75em;color:#8b949e'>"
                                        f"{_ck.replace('_',' ').title()}</div>"
                                        f"<div style='flex:1;background:#21262d;border-radius:3px;height:8px'>"
                                        f"<div style='width:{_pct}%;background:{_bar_c};height:8px;border-radius:3px'></div></div>"
                                        f"<div style='width:50px;font-size:0.75em;text-align:right'>{_pts}/{_max}</div>"
                                        f"</div>"
                                        f"<div style='font-size:0.7em;color:#6e7681;margin-left:98px'>{_cv.get('label','')}</div>",
                                        unsafe_allow_html=True,
                                    )

                    st.divider()

                    # ── Paper Trade button ────────────────────────────────────
                    btn_key = (
                        f"pbuy_{ticker}_{sg['strategy'].replace(' ', '_')}"
                        f"_{sg.get('strike', 0):.0f}_{sg.get('expiry', '')}"
                    )
                    _use_contracts = max(1, _rec_contracts if _rec_contracts > 0 else opt_contracts)
                    if st.button(
                        f"💼 Paper Trade — {_use_contracts}× contract(s)"
                        + (" (recommended)" if _rec_contracts != opt_contracts else ""),
                        key=btn_key,
                    ):
                        mid        = sg.get("mid", 0)
                        cost_total = mid * 100 * _use_contracts
                        summ       = ops_engine_strat.get_summary()
                        if cost_total > summ["available_cash"]:
                            st.error(
                                f"Not enough paper cash. Need ${cost_total:.2f}, "
                                f"have ${summ['available_cash']:.2f}.",
                                icon="❌",
                            )
                        else:
                            res = ops_engine_strat.buy(
                                ticker          = ticker,
                                contract_symbol = (
                                    f"{ticker}_{sg['expiry']}_"
                                    f"{sg['opt_type']}_{sg['strike']:.0f}"
                                ),
                                option_type     = sg["opt_type"],
                                strike          = sg["strike"],
                                expiry          = sg["expiry"],
                                contracts       = _use_contracts,
                                entry_price     = mid,
                                strategy        = sg["strategy"],
                            )
                            if res.get("ok"):
                                st.success(
                                    f"✅ Bought {_use_contracts}× {sg['strategy']} on {ticker} "
                                    f"for ${res['cost']:.2f}. "
                                    f"Cash left: ${res['cash_remaining']:.2f}",
                                    icon="✅",
                                )
                                st.rerun()
                            else:
                                st.error(f"Trade failed: {res.get('error')}", icon="❌")

        # ── Mode: from scan results ───────────────────────────────────────────
        if _sb_mode == "📡 Latest scan results (all tickers)":
            scan_run_btn = st.button("📡 Load Scan Picks & Build All Plays",
                                     use_container_width=False,
                                     key="scan_strat_run_btn")
            if scan_run_btn:
                with st.spinner("Loading latest scan picks and generating 0DTE–7DTE plays…"):
                    scan_plays = ts.get_scan_options_plays(conn, top_n=8,
                                                           bias_override=opt_bias.lower())

                if not scan_plays:
                    st.warning(
                        "No scan results in the database. "
                        "Run the breakout scanner (Scan Results tab) first.",
                        icon="⚠️",
                    )
                else:
                    total_plays = sum(len(p["suggestions"]) for p in scan_plays)
                    st.success(
                        f"**{len(scan_plays)} tickers · {total_plays} plays generated** "
                        f"from latest scan  |  bias: {opt_bias}",
                        icon="📡",
                    )
                    for sp in scan_plays:
                        st.markdown(
                            f"### {sp['ticker']}  "
                            f"<small style='color:#8b949e'>score {sp['explosive_score']:.0f} · "
                            f"prob {sp['breakout_prob']:.0f}% · "
                            f"{sp['pattern_detected'] or 'pattern'}</small>",
                            unsafe_allow_html=True,
                        )
                        _render_strategy_cards(sp["ticker"], sp["suggestions"], scan_meta=sp)
                        st.divider()
            else:
                st.info(
                    "Press **Load Scan Picks & Build All Plays** to auto-generate "
                    "0DTE–7DTE plays for every ticker from the latest breakout scan.",
                    icon="📡",
                )

        # ── Mode: single ticker ───────────────────────────────────────────────
        else:
            single_btn = st.button("⚡ Build Strategies", key="single_strat_btn")
            if single_btn:
                if not opt_ticker:
                    st.warning("Enter a ticker in the control bar above.", icon="⚠️")
                else:
                    with st.spinner(
                        f"Building 5–7 DTE plays for {opt_ticker} ({opt_bias}) "
                        "with PoP, break-even, and sizing…"
                    ):
                        engine_s    = ts.OptionsStrategyEngine()
                        suggestions = engine_s.suggest(opt_ticker, bias=opt_bias.lower())
                    st.success(
                        f"**{len(suggestions)} play(s) for {opt_ticker}** — "
                        "click 🔍 inside each card for Entry Quality & Conviction analysis",
                        icon="🎯",
                    ) if suggestions else None
                    _render_strategy_cards(opt_ticker, suggestions)
            else:
                st.info(
                    "Press **Build Strategies** to generate Weekly ATM, Weekly OTM, "
                    "and Mid-Week ATM plays with PoP, break-even move, and dynamic sizing. "
                    "Click 🔍 on any card for a full Entry Quality + Conviction score.",
                    icon="🎯",
                )

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB 4 — OPTIONS PAPER TRADES
    # ─────────────────────────────────────────────────────────────────────────
    with ot_paper:
        st.markdown('<div class="section-hdr">💰 Options Paper Portfolio — $500 Virtual Capital</div>',
                    unsafe_allow_html=True)

        ops_engine = ts.OptionsPaperEngine(conn)

        # ── Auto-expiry + exit alerts ─────────────────────────────────────────
        _expired = ops_engine.expire_check()
        if _expired:
            for _ep in _expired:
                st.warning(
                    f"⏰ **{_ep.get('ticker')} ${_ep.get('strike', 0):.0f}"
                    f"{str(_ep.get('option_type',''))[0].upper()}** expired worthless "
                    f"(P&L: ${_ep.get('net_pnl', 0):.2f})",
                    icon="⏰",
                )

        _exit_alerts = ops_engine.check_exit_alerts()
        if _exit_alerts:
            st.markdown("### 🚨 Auto-Exit Alerts")
            for _al in _exit_alerts:
                _atype = _al.get("alert_type", "")
                if _atype == "TAKE_PROFIT":
                    st.success(_al["message"], icon="🎯")
                elif _atype == "STOP_LOSS":
                    st.error(_al["message"], icon="🛑")
                else:  # TIME_DECAY
                    st.warning(_al["message"], icon="⏱")
            st.caption(
                "Auto-exit rules: close at **+50% profit** to lock gains · "
                "cut at **-50% loss** to preserve capital · "
                "exit by **2 DTE** to avoid theta destruction."
            )
            st.divider()

        ops_summary = ops_engine.get_summary()

        # ── Portfolio KPIs ────────────────────────────────────────────────────
        _pm1, _pm2, _pm3, _pm4, _pm5, _pm6 = st.columns(6)
        _pm1.metric("Cash Available",
                    f"${ops_summary['available_cash']:,.2f}")
        pnl_color = "normal" if ops_summary["realized_pnl"] >= 0 else "inverse"
        _pm2.metric("Realized P&L",
                    f"${ops_summary['realized_pnl']:+,.2f}")
        _pm3.metric("Total Return",
                    f"{ops_summary['total_return_pct']:+.1f}%")
        _pm4.metric("Win Rate",
                    f"{ops_summary['win_rate']:.0f}%",
                    f"{ops_summary['n_wins']}/{ops_summary['n_closed']} closed")
        _pm5.metric("Open Positions", ops_summary["n_open"])
        _pm6.metric("Trades Taken",   ops_summary["trades_made"])

        st.divider()

        # ── Open positions ────────────────────────────────────────────────────
        st.markdown("### 📂 Open Positions")
        open_pos = ops_engine.get_positions("OPEN")
        if not open_pos:
            st.info("No open options positions. Use the Strategy Builder tab to enter trades.", icon="📂")
        else:
            rows_op = []
            for p in open_pos:
                pnl     = p.get("unrealized_pnl")
                pct     = p.get("unrealized_pct")
                cur_p   = p.get("current_price")
                pnl_str = f"${pnl:+.2f} ({pct:+.1f}%)" if pnl is not None else "—"
                rows_op.append({
                    "ID":       p.get("id"),
                    "Ticker":   p.get("ticker"),
                    "Type":     str(p.get("option_type", "")).upper(),
                    "Strike":   f"${p.get('strike', 0):.2f}",
                    "Expiry":   p.get("expiry"),
                    "Qty":      p.get("contracts"),
                    "Entry $":  f"${p.get('entry_price', 0):.2f}",
                    "Cost":     f"${p.get('gross_invested', 0):.2f}",
                    "Current":  f"${cur_p:.2f}" if cur_p else "—",
                    "Unreal P&L": pnl_str,
                    "Strategy": p.get("strategy", "manual"),
                })
            st.dataframe(pd.DataFrame(rows_op),
                         use_container_width=True, hide_index=True)

            # ── Close position form ───────────────────────────────────────────
            st.markdown("#### Close a Position")
            _cl1, _cl2, _cl3 = st.columns([1, 2, 1])
            with _cl1:
                close_id = st.number_input("Position ID", min_value=1,
                                           step=1, key="close_pos_id")
            with _cl2:
                close_price = st.number_input("Exit premium (per share $)",
                                              min_value=0.0, step=0.01,
                                              format="%.2f", key="close_pos_price")
            with _cl3:
                close_reason = st.selectbox("Reason",
                                            ["TARGET_HIT", "STOP_OUT", "MANUAL",
                                             "EXPIRING_SOON"],
                                            key="close_pos_reason")
            if st.button("🔒 Close Position", key="close_pos_btn"):
                result = ops_engine.close(close_id, close_price, close_reason)
                if result.get("ok"):
                    pnl_v = result["net_pnl"]
                    icon  = "✅" if pnl_v >= 0 else "🛑"
                    st.success(
                        f"{icon} Position #{close_id} closed. "
                        f"P&L: ${pnl_v:+.2f}  |  Cash: ${result['cash']:.2f}",
                        icon=icon,
                    )
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('error')}", icon="❌")

        st.divider()

        # ── Closed positions history ──────────────────────────────────────────
        st.markdown("### 📜 Closed Positions History")
        closed_pos = ops_engine.get_positions("CLOSED")
        if not closed_pos:
            st.info("No closed positions yet.", icon="📜")
        else:
            rows_cl = []
            for p in closed_pos:
                pnl_v   = p.get("net_pnl") or 0
                pnl_str = f"${pnl_v:+.2f}"
                rows_cl.append({
                    "ID":       p.get("id"),
                    "Ticker":   p.get("ticker"),
                    "Type":     str(p.get("option_type", "")).upper(),
                    "Strike":   f"${p.get('strike', 0):.2f}",
                    "Expiry":   p.get("expiry"),
                    "Qty":      p.get("contracts"),
                    "Entry $":  f"${p.get('entry_price', 0):.2f}",
                    "Exit $":   f"${p.get('exit_price', 0) or 0:.2f}",
                    "Net P&L":  pnl_str,
                    "Reason":   p.get("exit_reason", "—"),
                    "Closed":   p.get("exit_date", "—"),
                    "Strategy": p.get("strategy", "manual"),
                })
            df_cl = pd.DataFrame(rows_cl)
            st.dataframe(df_cl, use_container_width=True, hide_index=True)

            # P&L waterfall chart
            if len(rows_cl) > 1:
                pnl_vals = [p.get("net_pnl") or 0 for p in closed_pos]
                labels   = [f"{p.get('ticker')} ${p.get('strike', 0):.0f}"
                            f"{str(p.get('option_type',''))[0].upper()}"
                            for p in closed_pos]
                colors   = ["#3fb950" if v >= 0 else "#f85149" for v in pnl_vals]
                fig_cl   = go.Figure(go.Bar(
                    x=labels, y=pnl_vals,
                    marker_color=colors,
                    text=[f"${v:+.0f}" for v in pnl_vals],
                    textposition="outside",
                ))
                fig_cl.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"), height=280,
                    xaxis=dict(gridcolor="#21262d", tickangle=-30),
                    yaxis=dict(gridcolor="#21262d", title="Net P&L ($)"),
                    margin=dict(l=10, r=10, t=10, b=60),
                )
                st.plotly_chart(fig_cl, use_container_width=True)

        st.divider()
        st.markdown("#### ⚠️ Reset Paper Account")
        st.caption("This permanently deletes all options paper trades and resets cash to $500.")
        if st.button("🗑 Reset Options Paper Account", key="ops_reset_btn",
                     type="secondary"):
            ops_engine.reset()
            st.success("Options paper account reset to $500.", icon="🔄")
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-TAB 5 — SMART MONEY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    with ot_smart:
        st.markdown('<div class="section-hdr">🏛 Smart Money Analysis</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Think like institutions and market makers. "
            "**Wyckoff Cycle** detects accumulation, springs, and distribution. "
            "**Institutional Volume** reads absorption, climax, and dry-up patterns. "
            "**Gamma Walls** reveal MM support/resistance from options OI. "
            "**Gamma Squeeze** flags runaway hedging setups."
        )

        _sm_ticker = st.text_input(
            "Ticker to analyse",
            value=opt_ticker or "SPY",
            key="sm_ticker_input",
            placeholder="e.g. AAPL",
        ).upper().strip()

        _sm_btn = st.button(
            "🔍 Run Full Smart Money Analysis",
            key="sm_run_btn",
            use_container_width=False,
        )

        if _sm_btn and _sm_ticker:
            with st.spinner(f"Running Wyckoff · Volume · Gamma · Squeeze analysis for {_sm_ticker}…"):
                _smw  = ts.detect_wyckoff_phase(_sm_ticker)
                _smv  = ts.analyze_institutional_volume(_sm_ticker)
                _smgw = ts.get_gamma_walls(_sm_ticker)
                _smgs = ts.detect_gamma_squeeze_setup(_sm_ticker)
            st.session_state["sm_results"] = {
                "ticker": _sm_ticker,
                "wyckoff": _smw, "volume": _smv,
                "gamma_walls": _smgw, "squeeze": _smgs,
            }

        if "sm_results" in st.session_state:
            _sr      = st.session_state["sm_results"]
            _sm_tk   = _sr["ticker"]
            _smw     = _sr["wyckoff"]
            _smv     = _sr["volume"]
            _smgw    = _sr["gamma_walls"]
            _smgs    = _sr["squeeze"]

            st.markdown(f"## {_sm_tk}")
            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # 1. WYCKOFF PHASE
            # ═══════════════════════════════════════════════════════════════
            st.markdown("### 📐 Wyckoff Market Phase")

            _phase = _smw.get("phase", "UNDEFINED")
            _phase_cfg = {
                "SPRING":         ("#3fb950", "💎", "Best long entry in the entire cycle"),
                "ACCUMULATION":   ("#58a6ff", "🔵", "Institutions quietly loading — buy the dip"),
                "MARKUP":         ("#3fb950", "📈", "Institutional demand driving trend — momentum valid"),
                "REACCUMULATION": ("#58a6ff", "🔄", "Healthy pause — continuation likely"),
                "DISTRIBUTION":   ("#e3b341", "⚠️", "Smart money selling into your strength"),
                "MARKDOWN":       ("#f85149", "📉", "No institutional floor — avoid longs"),
                "UNDEFINED":      ("#8b949e", "❓", "No clear structure — wait for clarity"),
                "INSUFFICIENT_DATA": ("#8b949e", "📭", "Not enough history"),
                "ERROR":          ("#8b949e", "💥", "Data error"),
            }
            _pc, _pe, _ptip = _phase_cfg.get(_phase, ("#8b949e", "❓", ""))
            _conf    = _smw.get("confidence", 0)
            _tradeable = _smw.get("is_tradeable", False)
            _spring  = _smw.get("spring_detected", False)

            # Big phase badge
            st.markdown(
                f"<div style='background:#161b22;border:2px solid {_pc};"
                f"border-radius:10px;padding:16px 20px;margin-bottom:8px'>"
                f"<div style='font-size:2em;font-weight:bold;color:{_pc}'>"
                f"{_pe} {_phase.replace('_',' ')}"
                f"</div>"
                f"<div style='color:#c9d1d9;margin-top:4px'>{_smw.get('description','')}</div>"
                f"<div style='margin-top:8px;padding:6px 10px;background:#21262d;"
                f"border-radius:6px;color:{_pc};font-weight:bold'>"
                f"ACTION: {_smw.get('action','')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if _spring:
                st.success(
                    "💎 **SPRING DETECTED** — Price dipped below the prior range low and "
                    "recovered. This is the highest-conviction Wyckoff long entry. "
                    "Institutions engineered this shakeout to sweep retail stops before markup.",
                    icon="💎",
                )

            _wc1, _wc2, _wc3, _wc4, _wc5 = st.columns(5)
            _wc1.metric("Phase",        _phase.replace("_", " "))
            _wc2.metric("Confidence",   f"{_conf}%")
            _wc3.metric("Position %",   f"{_smw.get('price_position_pct', 0):.0f}% of range",
                        help="0% = at range bottom, 100% = at range top")
            _wc4.metric("Volume Trend", f"{_smw.get('volume_trend', 1):.2f}×",
                        help=">1 = expanding volume vs 60-day avg; <1 = contracting")
            _wc5.metric("Tradeable?",   "✅ YES" if _tradeable else "🚫 NO")

            # Range gauge: show current price position on a horizontal bar
            _pos_pct = _smw.get("price_position_pct", 50)
            st.markdown(
                f"<div style='margin:8px 0 4px 0;font-size:0.8em;color:#8b949e'>"
                f"Price position within 60-day range "
                f"(Low ${_smw.get('period_low',0):.2f} → High ${_smw.get('period_high',0):.2f})</div>"
                f"<div style='background:#21262d;border-radius:6px;height:14px;position:relative'>"
                f"<div style='width:{_pos_pct}%;background:{_pc};"
                f"height:14px;border-radius:6px'></div>"
                f"<div style='position:absolute;top:-1px;left:{min(_pos_pct,97)}%;"
                f"font-size:0.7em;color:#fff'>▲ {_pos_pct:.0f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Wyckoff education box
            with st.expander("📚 How to use the Wyckoff Cycle — full guide", expanded=False):
                st.markdown("""
**The Wyckoff Cycle has 6 phases:**

| Phase | What's Happening | What To Do |
|---|---|---|
| **Accumulation** | Institutions buying quietly near lows — range is tight, volume building | Build long position at range lows |
| **Spring** | Fake breakdown below support, immediately reverses — a trap to shake out retail | **Best entry in the cycle** — buy the reversal with stop below spring low |
| **Markup** | Price breaks above the range on strong volume — trend is confirmed | Enter on first pullback to VWAP or 21 EMA |
| **Reaccumulation** | Mid-trend pause — volume dries up, range tightens | Hold longs; buy dip if volume confirms |
| **Distribution** | Institutions selling into retail greed at highs — closes are weak | Exit longs; do not buy; consider puts |
| **Markdown** | Price falls with no institutional support | Avoid; wait for selling climax to restart cycle |

**The Spring is the single most profitable Wyckoff signal:**
- Price violates a key support level (sweeps stops)
- Volume spikes as retail stops are triggered and shorts pile in
- Within 1–3 bars: price closes BACK above the support level
- This is institutions absorbing all the panic selling and flipping it long
- Enter on the reclaim with a tight stop below the spring low
                """)

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # 2. INSTITUTIONAL VOLUME
            # ═══════════════════════════════════════════════════════════════
            st.markdown("### 📊 Institutional Volume Footprint")

            _vpat  = _smv.get("primary_pattern", "NORMAL")
            _vscore = _smv.get("bullish_score", 50)
            _vpat_colors = {
                "BREAKOUT_CONFIRM":  "#3fb950",
                "ABSORPTION":        "#58a6ff",
                "SELLING_CLIMAX":    "#3fb950",
                "VOLUME_DRY_UP":     "#58a6ff",
                "EFFORT_NO_RESULT":  "#e3b341",
                "DISTRIBUTION_SIGN": "#f85149",
                "NORMAL":            "#8b949e",
            }
            _vc = _vpat_colors.get(_vpat, "#8b949e")

            _vv1, _vv2, _vv3, _vv4 = st.columns(4)
            _vv1.metric("Pattern",       _vpat.replace("_", " ").title())
            _vv2.metric("Bullish Score", f"{_vscore}/100")
            _vv3.metric("Volume Ratio",  f"{_smv.get('vol_ratio', 1):.2f}×",
                        help="Today's volume vs 20-day average")
            _vv4.metric("Close Strength",
                        f"{_smv.get('close_pct', 0.5)*100:.0f}%",
                        help="Where price closed within today's high-low range. >60% = bullish bar")

            # Pattern box
            st.markdown(
                f"<div style='background:#161b22;border-left:4px solid {_vc};"
                f"border-radius:6px;padding:12px 16px;margin:8px 0'>"
                f"<b style='color:{_vc}'>{_vpat.replace('_',' ')}</b><br>"
                f"{_smv.get('description','')}<br>"
                f"<span style='color:{_vc}'>{_smv.get('implication','')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Show all detected patterns
            _all_pats = _smv.get("all_patterns", [])
            if len(_all_pats) > 1:
                st.caption(f"All patterns detected: {' · '.join(p.replace('_',' ') for p in _all_pats)}")

            with st.expander("📚 Institutional Volume Pattern Guide", expanded=False):
                st.markdown("""
**How to read what institutions are doing from volume:**

| Pattern | Volume | Bar Shape | Meaning |
|---|---|---|---|
| **Breakout Confirm** | 2×+ avg | Large up bar, closes at high | Institutions are IN — real breakout |
| **Absorption** | 2×+ avg | Narrow range, mid close | Supply being soaked up — move coming |
| **Selling Climax** | 3×+ avg | Big down bar closes upper ½ | Sellers exhausted, buyers won — reversal likely |
| **Volume Dry-Up** | Below avg | Small bars on pullback | Institutions NOT selling — healthy correction |
| **Effort No Result** | 3×+ avg | Very narrow range | War between buyers/sellers — next bar decides |
| **Distribution Sign** | 2×+ avg | At highs, closes lower ½ | Institutions selling into you — get out |

**The key insight:** Institutions move billions — they **cannot** hide volume. Every time smart money loads or exits a position, volume spikes. The bar's shape tells you WHO won.
                """)

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # 3. GAMMA WALLS (Market Maker Levels)
            # ═══════════════════════════════════════════════════════════════
            st.markdown("### 🧱 Gamma Walls — Market Maker Support & Resistance")

            if _smgw:
                _spot     = _smgw.get("spot", 0)
                _mp       = _smgw.get("max_pain", 0)
                _gravity  = _smgw.get("gravity", "NEUTRAL")
                _mp_dist  = _smgw.get("max_pain_distance_pct", 0)
                _opex_wk  = _smgw.get("is_opex_week", False)
                _sqz_zone = _smgw.get("squeeze_zone", False)
                _nr       = _smgw.get("nearest_resistance")
                _ns       = _smgw.get("nearest_support")

                _gw1, _gw2, _gw3, _gw4, _gw5 = st.columns(5)
                _gw1.metric("Current Price", f"${_spot:.2f}")
                _gw2.metric("Max Pain",      f"${_mp:.2f}",
                            delta=f"{_mp_dist:+.1f}% from spot",
                            delta_color="inverse")
                _grav_emoji = {"UP": "⬆️", "DOWN": "⬇️", "NEUTRAL": "↔️"}.get(_gravity, "↔️")
                _gw3.metric("MM Gravity",    f"{_grav_emoji} {_gravity}",
                            help="Direction MMs profit from — price tends to drift this way into expiry")
                _gw4.metric("Nearest Resistance", f"${_nr:.2f}" if _nr else "—")
                _gw5.metric("Nearest Support",    f"${_ns:.2f}" if _ns else "—")

                if _opex_wk:
                    st.warning(
                        f"⚡ **OpEx Week** — {_smgw.get('days_to_expiry')}d to expiry. "
                        "Max Pain gravity is strongest now. Price tends to pin toward "
                        f"**${_mp:.2f}** into Friday close.",
                        icon="⚡",
                    )
                if _sqz_zone:
                    st.info(
                        f"🚀 **Squeeze Zone** — dominant call wall at ${_nr:.2f} is within 3% of spot. "
                        "If price pushes up, MMs must buy shares to delta-hedge — "
                        "this can accelerate the move.",
                        icon="🚀",
                    )
                if _gravity == "DOWN":
                    st.warning(
                        f"⬇️ Price is **${abs(_mp_dist):.1f}% above Max Pain** (${_mp:.2f}). "
                        "Market makers profit most if price drifts down into expiry. "
                        "Do not hold naked calls into this expiry.",
                        icon="⬇️",
                    )
                elif _gravity == "UP":
                    st.success(
                        f"⬆️ Price is **${abs(_mp_dist):.1f}% below Max Pain** (${_mp:.2f}). "
                        "Market makers profit most if price rises. "
                        "MM hedging flow supports longs into expiry.",
                        icon="⬆️",
                    )

                # Gamma wall chart — horizontal levels around spot
                _cw = _smgw.get("call_walls", [])
                _pw = _smgw.get("put_walls",  [])

                if _cw or _pw:
                    _strikes, _ois, _colors_gw, _labels_gw = [], [], [], []
                    for w in _pw[:5]:
                        _strikes.append(w["strike"])
                        _ois.append(w["oi"])
                        _colors_gw.append("#58a6ff")
                        _labels_gw.append(f"PUT WALL {w['strength']:.1f}×")
                    # Spot marker
                    _strikes.append(_spot)
                    _ois.append(0)
                    _colors_gw.append("#ffffff")
                    _labels_gw.append("SPOT")
                    # Max pain
                    _strikes.append(_mp)
                    _ois.append(0)
                    _colors_gw.append("#e3b341")
                    _labels_gw.append("MAX PAIN")
                    for w in _cw[:5]:
                        _strikes.append(w["strike"])
                        _ois.append(w["oi"])
                        _colors_gw.append("#f85149")
                        _labels_gw.append(f"CALL WALL {w['strength']:.1f}×")

                    _gw_df = pd.DataFrame({
                        "Strike": _strikes,
                        "OI":     _ois,
                        "Type":   _labels_gw,
                    }).sort_values("Strike")

                    _fig_gw = go.Figure()
                    for _, _row in _gw_df.iterrows():
                        _lc = ("#58a6ff" if "PUT" in _row["Type"]
                               else "#f85149" if "CALL" in _row["Type"]
                               else "#e3b341" if "PAIN" in _row["Type"]
                               else "#ffffff")
                        _fig_gw.add_shape(type="line",
                            x0=0, x1=max(_ois) * 1.1 if max(_ois) > 0 else 1,
                            y0=_row["Strike"], y1=_row["Strike"],
                            line=dict(color=_lc, width=2,
                                      dash="dot" if _row["OI"] == 0 else "solid"))
                        _fig_gw.add_annotation(
                            x=max(_ois) * 1.05 if max(_ois) > 0 else 0.5,
                            y=_row["Strike"],
                            text=f"${_row['Strike']:.0f} — {_row['Type']}",
                            showarrow=False, font=dict(color=_lc, size=11),
                            xanchor="right",
                        )
                    _fig_gw.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font=dict(color="#c9d1d9"),
                        height=max(300, len(_strikes) * 45),
                        xaxis=dict(visible=False),
                        yaxis=dict(title="Strike Price ($)", gridcolor="#21262d"),
                        margin=dict(l=10, r=200, t=10, b=10),
                        showlegend=False,
                    )
                    st.plotly_chart(_fig_gw, use_container_width=True)

                # Call and Put wall tables
                _gwt1, _gwt2 = st.columns(2)
                with _gwt1:
                    st.markdown("**🔴 Call Walls — Resistance**")
                    if _cw:
                        st.dataframe(pd.DataFrame([{
                            "Strike":    f"${w['strike']:.2f}",
                            "OI":        f"{w['oi']:,}",
                            "Strength":  f"{w['strength']:.1f}× avg",
                            "Distance":  f"+{w['distance_pct']:.1f}%",
                        } for w in _cw]), hide_index=True, use_container_width=True)
                        st.caption("MMs sell stock as price approaches → ceiling effect")
                    else:
                        st.info("No significant call walls above spot.")

                with _gwt2:
                    st.markdown("**🔵 Put Walls — Support**")
                    if _pw:
                        st.dataframe(pd.DataFrame([{
                            "Strike":    f"${w['strike']:.2f}",
                            "OI":        f"{w['oi']:,}",
                            "Strength":  f"{w['strength']:.1f}× avg",
                            "Distance":  f"-{w['distance_pct']:.1f}%",
                        } for w in _pw]), hide_index=True, use_container_width=True)
                        st.caption("MMs buy stock as price falls here → floor effect")
                    else:
                        st.info("No significant put walls below spot.")

                with st.expander("📚 How Gamma Walls Work — Market Maker Mechanics", expanded=False):
                    st.markdown("""
**Market Makers (MMs) must stay delta-neutral.** When you buy a call:
1. The MM sells it to you and immediately **buys shares** to hedge (delta hedging)
2. As the stock rises toward a strike with huge call OI, each contract's delta **increases**
3. The MM must buy **even more shares** to stay neutral — this buying lifts the price further
4. Near the strike, this becomes self-reinforcing → **gamma wall magnet effect**

**Call Wall** (red) = resistance ceiling:
- MMs sold these calls and hedged with shares
- As price approaches, they sell shares (delta decreases as they go deep ITM)
- Net effect: selling pressure that caps the stock near that strike

**Put Wall** (blue) = support floor:
- MMs sold these puts and shorted shares to hedge
- As price falls toward the strike, delta increases → they must buy shares to re-hedge
- Net effect: buying pressure that cushions the stock near that strike

**Max Pain** = the strike price where the maximum number of options expire worthless.
MMs are net short options, so they profit most when options expire worthless.
Into expiry, price is **magnetically pulled toward Max Pain** as MMs adjust hedges.
This effect is strongest in the **final 2 days** before expiry (OpEx).
                    """)

            else:
                st.info("No options data available — gamma walls require active options chain.", icon="📭")

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # 4. GAMMA SQUEEZE DETECTOR
            # ═══════════════════════════════════════════════════════════════
            st.markdown("### 🚀 Gamma Squeeze Detector")

            _sq_pot = _smgs.get("squeeze_potential", "LOW")
            _sq_cfg = {
                "HIGH":     ("#f85149", "🔥", "HIGH RISK — forced MM buying loop possible"),
                "MODERATE": ("#e3b341", "⚡", "MODERATE — watch for catalyst"),
                "LOW":      ("#3fb950", "✅", "LOW — no squeeze setup"),
            }
            _sqc, _sqe, _sqt = _sq_cfg.get(_sq_pot, ("#8b949e", "❓", ""))

            st.markdown(
                f"<div style='background:#161b22;border:2px solid {_sqc};"
                f"border-radius:10px;padding:14px 18px'>"
                f"<span style='font-size:1.5em;font-weight:bold;color:{_sqc}'>"
                f"{_sqe} Squeeze Potential: {_sq_pot}</span><br>"
                f"<span style='color:#c9d1d9'>{_smgs.get('description','')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            _sq1, _sq2, _sq3, _sq4 = st.columns(4)
            _sq1.metric("OTM Call Dominance",
                        f"{_smgs.get('otm_call_dominance_pct', 0):.0f}%",
                        help="% of total call OI that is OTM — higher = more squeeze fuel")
            _sq2.metric("Call/Put Ratio",
                        f"{_smgs.get('cp_ratio', 1):.2f}",
                        help=">1.5 = calls dominating; squeeze setups typically have C/P > 1.5")
            _dom_strike = _smgs.get("dominant_strike")
            _sq3.metric("Dominant Strike",
                        f"${_dom_strike:.0f}" if _dom_strike else "—",
                        help="The OTM call strike with the most open interest")
            _sq4.metric("Distance to Wall",
                        f"{_smgs.get('nearest_otm_wall_pct', 0):.1f}%",
                        help="How far the dominant OTM strike is from current price")

            if _smgs.get("what_happens_if_price_rises"):
                st.info(_smgs["what_happens_if_price_rises"], icon="🔁")

            with st.expander("📚 The Gamma Squeeze — How It Works", expanded=False):
                st.markdown("""
**The Gamma Squeeze is the most violent move in modern markets.**

The 2021 GME / AMC squeeze was NOT just a short squeeze — it was primarily a **gamma squeeze**:

**Step-by-step:**
1. Retail traders buy large quantities of **OTM call options** (cheap lottery tickets)
2. Market makers sell those calls and must **buy shares to delta-hedge**
3. Buying pressure pushes the stock price **up toward the OTM strike**
4. As price rises, the calls' delta **increases** (gamma effect)
5. MMs must buy **more shares** to stay neutral — more buying = more price increase
6. This feedback loop **accelerates** until the strike is reached
7. At the strike, all the OTM calls become ITM — MMs must buy a huge amount of stock
8. Result: **exponential move** in a matter of hours or days

**How to spot it before it happens:**
- OTM call open interest building rapidly (>60% of all calls are OTM)
- Call/Put ratio > 1.5 (calls dominating puts)
- Dominant OTM strike within 5% of current price
- Unusually high options volume relative to stock volume

**The risk:** Gamma squeezes unwind just as violently. Once MMs have hedged and the
catalyst is gone, the stock can collapse 50%+ in hours as delta unwinds.
**Trade the squeeze early or not at all.**
                """)

        else:
            st.info(
                "Enter a ticker and click **🔍 Run Full Smart Money Analysis** to see:\n\n"
                "- 📐 **Wyckoff Phase** — Are institutions accumulating, marking up, or distributing?\n"
                "- 📊 **Volume Footprint** — What does today's volume tell us about smart money activity?\n"
                "- 🧱 **Gamma Walls** — Where are the MM support/resistance levels hiding in plain sight?\n"
                "- 🚀 **Gamma Squeeze** — Is the options positioning set up for a runaway move?",
                icon="🏛",
            )
