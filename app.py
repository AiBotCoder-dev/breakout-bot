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
        _wanted = {"GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"}

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
</style>
""", unsafe_allow_html=True)


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


@st.cache_data(ttl=300)  # cache live prices for 5 minutes
def _fetch_live_prices(tickers: tuple) -> dict:
    """Return {ticker: last_price} for all tickers. Cached 5 min to avoid hammering yfinance."""
    if not tickers:
        return {}
    prices = {}
    try:
        raw = yf.download(list(tickers), period="2d", interval="5m",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return prices

        # Flatten MultiIndex → single level (yfinance ≥ 0.2.x returns MultiIndex)
        if isinstance(raw.columns, pd.MultiIndex):
            # Detect which level holds price-type labels (Open/High/Low/Close/Volume)
            _price_labels = {"Open", "High", "Low", "Close", "Volume"}
            lvl = 0 if _price_labels & set(raw.columns.get_level_values(0)) else 1
            raw.columns = raw.columns.get_level_values(lvl)
            # If still MultiIndex (level flip didn't help), try xs
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw.xs("Close", axis=1, level=0)

        # Now raw is a DataFrame or Series
        if isinstance(raw, pd.DataFrame):
            # Multi-ticker download → each column is a ticker
            if "Close" in raw.columns:
                # Single level with Close column (single ticker case)
                s = raw["Close"].dropna()
                if not s.empty and len(tickers) == 1:
                    prices[tickers[0]] = float(s.iloc[-1])
            else:
                for tk in tickers:
                    if tk in raw.columns:
                        s = raw[tk].dropna()
                        if not s.empty:
                            prices[tk] = float(s.iloc[-1])
        elif isinstance(raw, pd.Series):
            s = raw.dropna()
            if not s.empty and tickers:
                prices[tickers[0]] = float(s.iloc[-1])
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

    # ══════════════════════════════════════════════════════════════════════════
    # 🔧 PORTFOLIO HEALTH — diagnostic panel
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🔧 Portfolio Health", expanded=False):
        st.caption("Live diagnostic to verify the paper portfolio is updating.")
        try:
            _pe_diag = ts.PaperTradingEngine(conn)
            _summ    = _pe_diag.get_summary()
            _opens   = _pe_diag.open_positions

            # ── Last scan timestamp ──────────────────────────────────────────
            try:
                _last_scan = conn.execute(
                    "SELECT scan_timestamp, scan_id, COUNT(*) "
                    "FROM calls WHERE scan_id = "
                    "(SELECT scan_id FROM calls ORDER BY scan_timestamp DESC LIMIT 1) "
                    "GROUP BY scan_timestamp, scan_id"
                ).fetchone()
                if _last_scan:
                    _scan_ts = str(_last_scan[0])
                    _scan_n  = int(_last_scan[2])
                    try:
                        _dt = pd.to_datetime(_scan_ts)
                        _age_h = (pd.Timestamp.utcnow().tz_localize(None) - _dt.tz_localize(None)).total_seconds() / 3600
                        _age_str = (f"**{_age_h:.1f} h ago**" if _age_h < 48
                                    else f"**{_age_h/24:.1f} days ago** ⚠️")
                    except Exception:
                        _age_str = ""
                    st.markdown(f"📡 **Last scan:** {_scan_ts}  ·  {_scan_n} tickers  ·  {_age_str}")
                else:
                    st.warning("⚠️ No scans found in DB. Run the breakout scanner to populate signals.")
            except Exception as _e:
                st.caption(f"Scan lookup error: {_e}")

            # ── Last closed position ─────────────────────────────────────────
            try:
                _last_closed = conn.execute(
                    "SELECT ticker, exit_date, exit_reason, net_pnl FROM paper_portfolio "
                    "WHERE status='CLOSED' ORDER BY exit_date DESC LIMIT 1"
                ).fetchone()
                if _last_closed:
                    _cd = str(_last_closed[1])
                    _ct = str(_last_closed[0])
                    _cr = str(_last_closed[2])
                    _cp = float(_last_closed[3] or 0)
                    try:
                        _dt = pd.to_datetime(_cd)
                        _age_d = (pd.Timestamp.utcnow().tz_localize(None) - _dt.tz_localize(None)).total_seconds() / 86400
                        _age_str = (f"{_age_d:.1f}d ago" if _age_d < 30 else f"{_age_d:.0f}d ago ⚠️")
                    except Exception:
                        _age_str = ""
                    st.markdown(
                        f"📜 **Last close:** {_ct} on {_cd} ({_age_str})  ·  "
                        f"{_cr}  ·  ${_cp:+.2f}"
                    )
                else:
                    st.markdown("📜 **No closes yet.**")
            except Exception as _e:
                st.caption(f"Close lookup error: {_e}")

            st.markdown(
                f"💼 **Portfolio:** ${_summ['available_cash']:,.2f} cash  ·  "
                f"{_summ['n_open']}/{_summ.get('max_positions', 5)} open  ·  "
                f"Realized ${_summ['realized_pnl']:+,.2f}  ·  "
                f"Trades made {_summ['trades_made']}"
            )

            # ── Live position health: how close to stops/targets? ────────────
            if _opens:
                st.markdown("**Open positions — live stop/target distances:**")
                _ph_rows = []
                for _p in _opens:
                    _tk    = _p.get("ticker", "?")
                    _entry = float(_p.get("entry_price")  or 0)
                    _stop  = float(_p.get("stop_loss")    or 0)
                    _tgt   = float(_p.get("target_price") or 0)
                    try:
                        _cur = float(yf.Ticker(_tk).fast_info.last_price or 0)
                    except Exception:
                        _cur = 0
                    if _cur > 0 and _stop > 0 and _tgt > 0:
                        _d_stop = (_cur - _stop) / _cur * 100
                        _d_tgt  = (_tgt - _cur) / _cur * 100
                        _flag = ""
                        if _cur <= _stop * 1.005:
                            _flag = "🛑 SHOULD HAVE STOPPED"
                        elif _cur >= _tgt * 0.995:
                            _flag = "🎯 SHOULD HAVE TARGETED"
                        _ph_rows.append({
                            "Ticker": _tk,
                            "Entry":  f"${_entry:.2f}",
                            "Current": f"${_cur:.2f}",
                            "Stop":   f"${_stop:.2f}",
                            "Target": f"${_tgt:.2f}",
                            "To Stop": f"{_d_stop:+.1f}%",
                            "To Tgt":  f"{_d_tgt:+.1f}%",
                            "Status": _flag,
                        })
                    else:
                        _ph_rows.append({
                            "Ticker": _tk, "Entry": f"${_entry:.2f}",
                            "Current": "?", "Stop": f"${_stop:.2f}",
                            "Target": f"${_tgt:.2f}", "To Stop": "—",
                            "To Tgt": "—", "Status": "no live price",
                        })
                st.dataframe(pd.DataFrame(_ph_rows), hide_index=True,
                             use_container_width=True)
                _stuck = [r for r in _ph_rows if r["Status"]]
                if _stuck:
                    st.error(
                        f"⚠️ {len(_stuck)} position(s) have hit stops/targets but are still OPEN. "
                        "The monitor isn't closing them. Check GitHub Actions → Position Monitor "
                        "for failed runs.",
                        icon="🛑",
                    )
            else:
                st.info("No open positions. Either monitor hasn't found any signals to enter "
                        "or all positions have closed already.", icon="📂")

            # ── Risk Engine status ───────────────────────────────────────────
            try:
                _diag_risk = ts.get_portfolio_risk_status(_pe_diag)
                st.markdown(
                    f"🛡 **Risk Engine:** {_diag_risk['risk_level']}  ·  "
                    f"Can open: **{'YES' if _diag_risk['can_open_new'] else 'NO'}**  ·  "
                    f"Sizing {_diag_risk['size_multiplier']:.2f}×"
                )
                if not _diag_risk["can_open_new"]:
                    st.error(_diag_risk["reason"], icon="🛑")
            except Exception:
                pass

            st.caption(
                "If your scan is more than 24 h old or no positions are entering, "
                "the chase guard may be skipping stale signals. Run a fresh scan "
                "from the sidebar to refresh."
            )
        except Exception as _diag_exc:
            st.error(f"Diagnostic error: {_diag_exc}")

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
                # SPY regime from cached ctx if available
                try:
                    if "_ctx" in dir() and isinstance(_ctx, dict):
                        _ai_ctx["regime"] = _ctx.get("regime")
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
        st.write("Fetching universe…")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                scanner.run()
            except Exception as e:
                st.error(f"Scan error: {e}")

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
tab_dash, tab_results, tab_chart, tab_analytics, tab_paper, tab_fees, tab_options = st.tabs([
    "📊  Dashboard",
    "🔍  Scan Results",
    "📈  Stock Chart",
    "📉  Analytics",
    "💼  Paper Portfolio",
    "💰  Fees Tracker",
    "⚡  Options",
])


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
            try:
                day_n = (today - datetime.strptime(p["entry_date"], "%Y-%m-%d").date()).days + 1
            except Exception:
                day_n = "?"
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
                "Day #": f"D{day_n}",
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

    # ── Fetch live prices (cached 5 min) ─────────────────────────────────────
    live_prices: dict = {}
    if positions:
        tickers_tuple = tuple(p["ticker"] for p in positions)
        with st.spinner("Fetching live prices…"):
            live_prices = _fetch_live_prices(tickers_tuple)

    # Build per-position live data
    live_pos_map: dict = {}
    total_live_value = 0.0
    total_cost_basis = 0.0
    for p in positions:
        tk     = p["ticker"]
        ep     = float(p.get("entry_price") or 0)
        shares = float(p.get("shares") or 0)
        gross  = float(p.get("gross_invested") or 0)
        cur    = live_prices.get(tk)
        has_lv = cur is not None
        if not has_lv:
            cur = ep
        cur_val = shares * cur
        unr_pnl = cur_val - gross
        unr_pct = (unr_pnl / gross * 100) if gross else 0.0
        live_pos_map[tk] = {
            "current_price":  cur,
            "current_value":  cur_val,
            "unrealized_pnl": unr_pnl,
            "unrealized_pct": unr_pct,
            "has_live_price": has_lv,
        }
        total_live_value += cur_val
        total_cost_basis += gross

    total_unrealized = total_live_value - total_cost_basis
    unr_pct_total    = (total_unrealized / total_cost_basis * 100) if total_cost_basis else 0.0
    prices_fetched   = bool(live_prices)

    # Mark-to-market total (cash + current value of positions at live prices)
    mtm_total = s["available_cash"] + (total_live_value if positions else 0.0)
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
        unr_col = "green" if total_unrealized >= 0 else "red"
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Unrealized P&L",
                f"{'+'if total_unrealized>=0 else ''}${total_unrealized:,.2f}",
                unr_col)
        with c2:
            kpi("Unrealized %",
                f"{'+'if unr_pct_total>=0 else ''}{unr_pct_total:.2f}%",
                unr_col)
        with c3:
            kpi("Cost Basis (Open)",
                f"${total_cost_basis:,.2f}",
                "yellow")
        if prices_fetched:
            st.caption("✅ Live prices fetched — refreshes every 5 minutes")
        else:
            st.caption("⚠️ Live prices unavailable — showing cost basis (unrealized P&L = $0)")

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

    ot_chain, ot_unusual, ot_strategy, ot_paper, ot_smart = st.tabs([
        "⛓  Chain Viewer",
        "🔥  Unusual Activity",
        "🎯  Strategy Builder",
        "💰  Paper Trades",
        "🏛  Smart Money",
    ])

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
