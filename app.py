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

    def _adapt(self, sql: str) -> str:
        sql = sql.replace("?", "%s")
        sql = self._AUTOINCREMENT.sub("BIGSERIAL PRIMARY KEY", sql)
        sql = sql.replace("DATETIME", "TIMESTAMP")
        return sql

    def execute(self, sql: str, params=()):
        if self._LAST_ROWID.search(sql.strip()):
            return _FakeCursor(self._last_id)

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
        cur.execute(adapted, params or ())

        if add_returning:
            row           = cur.fetchone()
            self._last_id = row[0] if row else None

        self._conn.commit()
        return _PgCursor(cur)

    def executescript(self, sql: str):
        adapted = self._adapt(sql)
        cur     = self._conn.cursor()
        for stmt in _re.split(r";[ \t]*\n?", adapted):
            stmt = stmt.strip()
            if not stmt or stmt.startswith("--"):
                continue
            try:
                cur.execute(stmt)
                self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass

    def commit(self):
        try:
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

def _render_stock_card(r: dict):
    ticker   = r.get("ticker", "")
    price    = r.get("price", 0) or 0
    prob     = r.get("probability", 0) or 0
    pattern  = r.get("pattern", "No Pattern") or "No Pattern"
    score    = r.get("explosive_score", 0) or 0
    grade    = r.get("explosive_grade", "") or ""
    stop     = r.get("stop_price") or 0
    target   = r.get("tgt_price") or 0
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
      <div class="trade-grid">
        <div><div class="tg-label">Entry</div>
             <div class="tg-val">${price:.2f}</div></div>
        <div><div class="tg-label">Stop Loss</div>
             <div class="tg-val tg-red">{stop_str}</div></div>
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
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Scan configuration
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Trading Scanner")
    st.caption("Configure and launch your scan below.")
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
with tab_paper:
    conn, _mode = get_db()
    paper = ts.PaperTradingEngine(conn)
    s     = paper.get_summary()
    positions = paper.open_positions

    # ── Header KPIs ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Paper Portfolio — $1,000 Starting Capital</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    ret_col = "green" if s["total_return"] >= 0 else "red"
    with c1: kpi("Total Capital",  f"${s['total_capital']:,.2f}", "blue")
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

            # Estimated current P&L (entry price only — no live price fetch in dashboard)
            est_sell_fee = shares * ep * 0.015
            net_pnl_est  = -bf - est_sell_fee  # at entry price, net is negative (fees only)

            with st.expander(f"**{p['ticker']}** — entered @ ${ep:.2f}  |  "
                             f"Target ${tgt:.2f}  |  Stop ${stp:.2f}", expanded=True):
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

    _PAPER_BUDGET = ts._PAPER_BUDGET
