"""
Trading Scanner — Streamlit Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import sqlite3
import sys, os, io, argparse, contextlib, re as _re
from datetime import datetime, timedelta
import numpy as np


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
        return self._vals[self._cols.index(k)]

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
                self._conn.commit()
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
        if len(tickers) == 1:
            df = yf.download(tickers[0], period="2d", interval="5m",
                             progress=False, auto_adjust=True)
            if not df.empty:
                prices[tickers[0]] = float(df["Close"].dropna().iloc[-1])
        else:
            df = yf.download(list(tickers), period="2d", interval="5m",
                             progress=False, auto_adjust=True)
            if not df.empty:
                close_data = df.get("Close", df)
                if isinstance(close_data, pd.DataFrame):
                    for tk in tickers:
                        if tk in close_data.columns:
                            s = close_data[tk].dropna()
                            if not s.empty:
                                prices[tk] = float(s.iloc[-1])
                elif isinstance(close_data, pd.Series):
                    s = close_data.dropna()
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
    avwap    = r.get("avwap_above")

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
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


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
tab_dash, tab_results, tab_chart, tab_analytics, tab_paper, tab_fees = st.tabs([
    "📊  Dashboard",
    "🔍  Scan Results",
    "📈  Stock Chart",
    "📉  Analytics",
    "💼  Paper Portfolio",
    "💰  Fees Tracker",
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

    fee_rows = conn.execute(
        "SELECT * FROM fees ORDER BY transaction_date ASC, id ASC"
    ).fetchall()
    fee_data = [dict(r) for r in fee_rows]

    total_fees  = sum(r.get("fee_amount") or 0 for r in fee_data)
    buy_fees    = sum(r.get("fee_amount") or 0 for r in fee_data
                      if r.get("transaction_type") == "BUY")
    sell_fees   = sum(r.get("fee_amount") or 0 for r in fee_data
                      if r.get("transaction_type") == "SELL")
    n_buy       = sum(1 for r in fee_data if r.get("transaction_type") == "BUY")
    n_sell      = sum(1 for r in fee_data if r.get("transaction_type") == "SELL")

    paper_s = ts.PaperTradingEngine(conn).get_summary()
    gross_pnl_total = conn.execute(
        "SELECT SUM(gross_pnl) FROM paper_portfolio WHERE status='CLOSED'"
    ).fetchone()
    gross_pnl_total = float(gross_pnl_total[0] or 0) if gross_pnl_total else 0.0

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
        snap_rows = conn.execute(
            "SELECT snapshot_date, realized_pnl, portfolio_return_pct "
            "FROM paper_trading ORDER BY snapshot_date ASC"
        ).fetchall()
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
