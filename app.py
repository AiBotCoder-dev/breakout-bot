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
import sys, os, io, argparse, contextlib
from datetime import datetime, timedelta
import numpy as np

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
  /* dark background override */
  .stApp { background-color: #0d1117; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  /* KPI cards */
  .kpi-card {
    background: #1c2333;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
  }
  .kpi-label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.06em; }
  .kpi-value { font-size: 1.9rem; font-weight: 700; margin: 4px 0 0; }
  .kpi-green  { color: #3fb950; }
  .kpi-red    { color: #f85149; }
  .kpi-yellow { color: #e3b341; }
  .kpi-blue   { color: #58a6ff; }

  /* section headers */
  .section-hdr {
    font-size: 0.75rem; font-weight: 600;
    color: #8b949e; letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px; margin-bottom: 12px;
  }

  /* status badges */
  .badge {
    display: inline-block; border-radius: 4px;
    padding: 2px 8px; font-size: 0.72rem; font-weight: 600;
  }
  .badge-green  { background:#1a4428; color:#3fb950; }
  .badge-red    { background:#4c1b1b; color:#f85149; }
  .badge-yellow { background:#4a3b00; color:#e3b341; }
  .badge-blue   { background:#0d2d6b; color:#58a6ff; }

  /* hide streamlit chrome */
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

def _is_streamlit_cloud() -> bool:
    """Detect Streamlit Community Cloud environment."""
    return (
        os.environ.get("STREAMLIT_SHARING_MODE") == "1"
        or os.environ.get("HOME", "").startswith("/home/appuser")
        or os.path.exists("/mount/src")  # Community Cloud mount point
    )

@st.cache_resource
def get_db():
    on_cloud = _is_streamlit_cloud()
    if on_cloud:
        path = ":memory:"
    else:
        path = os.path.join(_DIR, "trading_scanner_history.db")
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    logger = ts.CallLogger.__new__(ts.CallLogger)
    logger.conn = conn
    logger._init_schema()
    return conn, on_cloud

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
    conn, _cloud = get_db()

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
tab_dash, tab_results, tab_chart, tab_analytics = st.tabs([
    "📊  Dashboard",
    "🔍  Scan Results",
    "📈  Stock Chart",
    "📉  Analytics",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PORTFOLIO DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
with tab_dash:
    conn, on_cloud = get_db()

    if on_cloud:
        st.warning(
            "**Cloud mode:** Portfolio history is stored in memory only and resets "
            "when the app restarts. To persist trade history, run the app locally.",
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
        st.info("Configure your scan in the sidebar and click **🚀 Run Scan**.",
                icon="🔍")
    else:
        results = st.session_state.results
        ctx     = st.session_state.market_ctx

        # Market context bar
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Universe (raw)", f"{ctx.get('spy_raw',0):,}")
        with c2: st.metric("Passed filter",  f"{ctx.get('filtered',0):,}")
        with c3: st.metric("Fully analyzed", f"{ctx.get('advanced',0):,}")
        with c4: st.metric("Qualified",      str(len(results)))

        st.divider()

        if not results:
            st.warning("No stocks qualified with current filters.", icon="🚫")
        else:
            # ── Explosive candidates ──────────────────────────────────────────
            expl = [r for r in results if r.get("explosive_score", 0) >= 40]
            expl.sort(key=lambda x: x.get("explosive_score", 0), reverse=True)

            st.markdown('<div class="section-hdr">⚡ Explosive Move Candidates</div>',
                        unsafe_allow_html=True)
            if not expl:
                st.caption("No explosive candidates (score ≥ 40) in this scan.")
            else:
                expl_rows = []
                for r in expl:
                    mc  = r.get("market_cap", 0)
                    fs  = r.get("expl_float", 0)
                    expl_rows.append({
                        "Ticker":    r["ticker"],
                        "Price":     f"${r['price']:.2f}",
                        "Mkt Cap":   (f"${mc/1e9:.1f}B" if mc >= 1e9
                                      else f"${mc/1e6:.0f}M" if mc >= 1e6 else "—"),
                        "Float":     (f"{fs/1e6:.1f}M" if fs >= 1e6
                                      else f"{fs/1e3:.0f}K" if fs > 0 else "—"),
                        "Short %":   (f"{r.get('expl_short_pct',0):.0%}"
                                      if r.get("expl_short_pct") else "—"),
                        "Score":     f"{r.get('explosive_score',0):.0f}/100",
                        "Grade":     r.get("explosive_grade", ""),
                        "Est Move":  f"+{r.get('move_low',0):.0f}%–{r.get('move_high',0):.0f}%",
                        "Pattern":   r.get("pattern", "—")[:16],
                        "Prob %":    f"{r.get('probability',0)}%",
                        "Catalyst":  (r.get("top_flag","")[:20] if r.get("top_flag") else "—"),
                    })
                df_expl = pd.DataFrame(expl_rows)
                sel_expl = st.dataframe(
                    df_expl, use_container_width=True, hide_index=True,
                    on_select="rerun", selection_mode="single-row")
                if sel_expl.selection.rows:
                    ticker = expl_rows[sel_expl.selection.rows[0]]["Ticker"]
                    st.session_state.selected = ticker
                    st.success(f"Selected **{ticker}** — switch to the 📈 Stock Chart tab to view.", icon="✅")

            # ── Breakout probability table ─────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-hdr">📊 Breakout Probability Rankings</div>',
                        unsafe_allow_html=True)

            prob_rows = []
            for r in results:
                pm  = r.get("pm_gap", 0) or 0
                prob_rows.append({
                    "Ticker":   r["ticker"],
                    "Price":    f"${r['price']:.2f}",
                    "RS":       r.get("rs_rating", "—"),
                    "EQ":       r.get("eq_grade", "—"),
                    "Flow":     r.get("options_score", "—"),
                    "Prob %":   r.get("probability", 0),
                    "Band":     r.get("conf_label", ""),
                    "Pattern":  r.get("pattern", "No Pattern")[:18],
                    "AVWAP":    ("▲ Above" if r.get("avwap_above") else "▼ Below"),
                    "PM Gap":   f"+{pm:.1%}" if pm > 0.005 else ("—" if pm == 0 else f"{pm:.1%}"),
                    "% < 52W":  f"{r.get('pct_52w',0)*100:.1f}%",
                    "R:R":      f"{r.get('rr',0):.1f}:1",
                    "Earnings": r.get("earnings_risk", "No"),
                })
            df_prob = pd.DataFrame(prob_rows)

            # colour prob column
            def _prob_colour(val):
                if val >= 75: return "background-color:#1a4428; color:#3fb950"
                if val >= 60: return "background-color:#2d3a1a; color:#e3b341"
                return "background-color:#3a1a1a; color:#f85149"

            sel_prob = st.dataframe(
                df_prob.style.map(_prob_colour, subset=["Prob %"]),
                use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="single-row")

            if sel_prob.selection.rows:
                ticker = prob_rows[sel_prob.selection.rows[0]]["Ticker"]
                st.session_state.selected = ticker
                st.success(f"Selected **{ticker}** — switch to the 📈 Stock Chart tab to view.", icon="✅")

            # ── Urgency summary (timing) ───────────────────────────────────────
            timed = [r for r in results if r.get("timing")]
            if timed:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-hdr">⏱ Trade Timing Summary</div>',
                            unsafe_allow_html=True)
                t_rows = []
                for r in timed:
                    t = r["timing"]
                    t_rows.append({
                        "Ticker":   r["ticker"],
                        "Urgency":  t.get("urgency_display", "—"),
                        "Duration": (f"{t.get('min_days','?')}–{t.get('max_days','?')}d"
                                     if t.get("min_days") else "—"),
                        "Peak Day": (f"D{t.get('peak_day')}" if t.get("peak_day") else "—"),
                        "Type":     t.get("trade_type", "—"),
                    })
                st.dataframe(pd.DataFrame(t_rows),
                             use_container_width=True, hide_index=True)


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
                with c1: st.metric("Breakout Prob", f"{result.get('probability',0)}%")
                with c2: st.metric("Pattern",       result.get("pattern","—")[:18])
                with c3: st.metric("RSI",           f"{result.get('rsi',0):.1f}")
                with c4: st.metric("R:R",           f"{result.get('rr',0):.1f}:1")
                with c5: st.metric("Earnings Risk", result.get("earnings_risk","No"))

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
    conn, _  = get_db()
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
            st.markdown('<div class="section-hdr">💡 Self-Improvement Suggestions</div>',
                        unsafe_allow_html=True)
            for tip in tips:
                st.warning(tip, icon="💡")
