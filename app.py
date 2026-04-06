"""
StockPulse India v3 — Nifty 500 Edition
Multi-factor quant model · Backtested · Portfolio optimised
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RISK_PROFILES, ANALYSIS_PERIODS
from stock_universe import fetch_index_constituents, get_all_sectors, filter_stocks
from data_fetcher import fetch_all, fetch_nifty, fetch_news, market_breadth
from technical_analysis import technical_score, relative_strength, rs_percentile, add_indicators
from fundamental_analysis import fundamental_score
from scoring_engine import recommend, rank, composite
from backtester import run_multi_period_backtest
from portfolio_optimizer import build_portfolio

st.set_page_config(page_title="StockPulse India", page_icon="◉", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
#  Theme CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#06080d;--sf:#0d1117;--sf2:#151b25;--bd:#1e2a3a;--tx:#e2e8f0;--tx2:#8b9ab5;--ac:#00d4ff;--gn:#00ff87;--rd:#ff4757;--am:#ffb700;--pu:#a855f7}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--tx)!important;font-family:'Outfit',sans-serif!important}
[data-testid="stSidebar"]{background:var(--sf)!important;border-right:1px solid var(--bd)!important}
[data-testid="stSidebar"] *{color:var(--tx)!important}
header[data-testid="stHeader"]{display:none!important}
footer{display:none!important}
#MainMenu{display:none!important}
h1,h2,h3,h4{font-family:'Outfit',sans-serif!important;font-weight:700!important;color:var(--tx)!important}
.block-container{padding-top:1.2rem!important;max-width:1500px!important}
.hero{background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(168,85,247,.08));border:1px solid var(--bd);border-radius:16px;padding:1.8rem 2.2rem;margin-bottom:1.2rem}
.hero-title{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#00d4ff,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
.hero-sub{color:var(--tx2);font-size:.9rem;margin-top:.2rem}
.glass{background:rgba(13,17,23,.85);backdrop-filter:blur(16px);border:1px solid var(--bd);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:.8rem;transition:border-color .2s}
.glass:hover{border-color:var(--ac)}
.mcard{background:var(--sf2);border:1px solid var(--bd);border-radius:12px;padding:.9rem 1rem;text-align:center}
.mcard-val{font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace}
.mcard-label{font-size:.72rem;color:var(--tx2);text-transform:uppercase;letter-spacing:1px;margin-top:.15rem}
.mc-cyan .mcard-val{color:var(--ac)}.mc-green .mcard-val{color:var(--gn)}.mc-amber .mcard-val{color:var(--am)}.mc-red .mcard-val{color:var(--rd)}.mc-purple .mcard-val{color:var(--pu)}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.badge-buy{background:rgba(0,255,135,.12);color:var(--gn);border:1px solid rgba(0,255,135,.3)}
.badge-acc{background:rgba(0,212,255,.10);color:var(--ac);border:1px solid rgba(0,212,255,.3)}
.badge-hold{background:rgba(255,183,0,.10);color:var(--am);border:1px solid rgba(255,183,0,.3)}
.badge-avoid{background:rgba(255,71,87,.10);color:var(--rd);border:1px solid rgba(255,71,87,.3)}
.sbar{height:6px;background:var(--sf2);border-radius:3px;overflow:hidden}.sbar-fill{height:100%;border-radius:3px}
div[data-testid="stMetric"]{background:var(--sf2)!important;border:1px solid var(--bd)!important;border-radius:12px!important;padding:.8rem!important}
div[data-testid="stMetric"] label{color:var(--tx2)!important;font-size:.72rem!important;text-transform:uppercase!important;letter-spacing:.5px!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--tx)!important;font-family:'JetBrains Mono',monospace!important}
div[data-testid="stExpander"]{background:var(--sf)!important;border:1px solid var(--bd)!important;border-radius:12px!important}
div[data-testid="stExpander"] summary{color:var(--tx)!important;font-weight:600!important}
.stProgress>div>div{background:var(--sf2)!important;border-radius:4px!important}
button[kind="primary"]{background:linear-gradient(135deg,#00d4ff,#a855f7)!important;border:none!important;color:#000!important;font-weight:700!important;border-radius:10px!important}
.stSelectbox [data-baseweb="select"]>div,.stMultiSelect [data-baseweb="select"]>div{background:var(--sf2)!important;border-color:var(--bd)!important;color:var(--tx)!important}
input[type="number"]{background:var(--sf2)!important;border-color:var(--bd)!important;color:var(--tx)!important}
.stTabs [data-baseweb="tab-list"]{background:var(--sf)!important;border-radius:12px;padding:4px;border:1px solid var(--bd)}
.stTabs [data-baseweb="tab"]{border-radius:10px!important;padding:6px 18px!important;color:var(--tx2)!important;font-weight:500!important}
.stTabs [aria-selected="true"]{background:var(--sf2)!important;color:var(--ac)!important}
.stTabs [data-baseweb="tab-border"],.stTabs [data-baseweb="tab-highlight"]{display:none!important}
.stDownloadButton>button{background:var(--sf2)!important;border:1px solid var(--bd)!important;color:var(--ac)!important;border-radius:10px!important}
.disclaimer{background:rgba(255,183,0,.06);border:1px solid rgba(255,183,0,.2);border-radius:10px;padding:.8rem 1rem;font-size:.78rem;color:var(--am)}
.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.6rem;margin:.8rem 0}
.detail-item{background:var(--sf2);border:1px solid var(--bd);border-radius:10px;padding:.7rem .9rem}
.detail-item .label{font-size:.7rem;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px}
.detail-item .value{font-size:1.1rem;font-weight:600;font-family:'JetBrains Mono',monospace;margin-top:.1rem}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def action_badge(a):
    c = {"STRONG BUY":"buy","BUY":"buy","ACCUMULATE":"acc","HOLD":"hold"}.get(a,"avoid")
    return f'<span class="badge badge-{c}">{a}</span>'

def risk_span(l):
    c = {"Low":"color:var(--gn)","Medium":"color:var(--am)","High":"color:var(--rd)"}.get(l,"")
    return f'<span style="{c};font-weight:600">{l}</span>'

def mc(val, lbl, cls="mc-cyan"):
    return f'<div class="mcard {cls}"><div class="mcard-val">{val}</div><div class="mcard-label">{lbl}</div></div>'

def detail_item(label, value, color=None):
    cs = f'color:{color}' if color else ''
    return f'<div class="detail-item"><div class="label">{label}</div><div class="value" style="{cs}">{value}</div></div>'

CL = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.6)",
          font=dict(family="Outfit,sans-serif",color="#8b9ab5"),
          xaxis=dict(gridcolor="rgba(30,42,58,.5)"), yaxis=dict(gridcolor="rgba(30,42,58,.5)"),
          margin=dict(l=50,r=20,t=40,b=30), legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))

# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Configuration")
    index_choice = st.selectbox("Stock Universe", ["Nifty 50", "Nifty 100", "Nifty 500"], index=0,
                                help="Nifty 500 scans ALL 500 stocks (takes ~5-8 min first time)")

    portfolio_val = st.number_input("Portfolio Value (₹)", 50000, 10_00_00_000, 5_00_000, 50000)
    risk_profile = st.selectbox("Risk Profile", list(RISK_PROFILES.keys()), index=1)
    st.caption(RISK_PROFILES[risk_profile]["desc"])

    st.markdown("---")

    # Fetch universe to populate sector filter
    universe = fetch_index_constituents(index_choice)
    all_sectors = get_all_sectors(universe) if universe else []

    cap_filter = st.multiselect("Market Cap", ["Large Cap","Mid Cap","Small Cap"],
                                default=RISK_PROFILES[risk_profile]["preferred_caps"])
    sector_filter = st.multiselect("Sectors", ["All"] + all_sectors, default=["All"])
    min_score = st.slider("Min Composite Score", 0, 100, 35)
    period_key = st.selectbox("Data Period", list(ANALYSIS_PERIODS.keys()), index=1)

    st.markdown("---")
    max_scan = st.slider("Max Stocks to Scan", 10, len(universe) if universe else 500,
                         min(len(universe), 500) if universe else 50,
                         help="More stocks = longer scan time (~2-3s per stock)")
    run_bt = st.checkbox("Run Backtest", value=True)
    run = st.button("◉  SCAN MARKET", type="primary", use_container_width=True)

    st.markdown("---")
    if universe:
        st.caption(f"Universe: **{len(universe)}** stocks loaded from NSE")
    st.markdown('<div class="disclaimer"><b>⚠ Disclaimer:</b> Educational purposes only. '
                'Not financial advice. Consult a SEBI-registered advisor.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero">
    <p class="hero-title">StockPulse India</p>
    <p class="hero-sub">{index_choice} · Multi-factor quant screener · Stage analysis · Backtested · Portfolio optimised</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  Analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def get_tickers():
    all_sec = "All" in sector_filter or not sector_filter
    filtered = filter_stocks(universe, None if all_sec else sector_filter, cap_filter if cap_filter else None)
    return list(filtered.keys())[:max_scan], filtered

def analyse(data, bench, stock_map):
    all_rs = {}
    for t, d in data.items():
        all_rs[t] = relative_strength(d["price"], bench) if bench is not None else 1.0
    rs_vals = list(all_rs.values())

    recs = []
    for t, d in data.items():
        info_tuple = stock_map.get(t, (t, "Other", "Unknown", "Unknown"))
        name, sector, cap = info_tuple[0], info_tuple[1], info_tuple[2]

        tech = technical_score(d["price"], bench)
        tech["rs"] = all_rs.get(t, 1.0)
        tech["rs_pct"] = rs_percentile(all_rs.get(t, 1.0), rs_vals)

        fund = fundamental_score(d["info"], sector)
        rec = recommend(t, name, sector, cap, tech, fund, d["info"], risk_profile, portfolio_val)
        recs.append(rec)
    return rank(recs)

# ═══════════════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════════════

if run:
    tickers, stock_map = get_tickers()
    if not tickers:
        st.warning("No stocks match filters.")
        st.stop()

    prog = st.progress(0, text=f"Scanning {len(tickers)} stocks from {index_choice}...")
    def _p(pct, txt=""):
        prog.progress(min(pct, 1.0), text=txt)

    data = fetch_all(tickers, ANALYSIS_PERIODS[period_key], _p)
    bench = fetch_nifty(ANALYSIS_PERIODS[period_key])

    prog.progress(0.88, text="Analyzing...")
    recs = analyse(data, bench, stock_map)

    prog.progress(0.93, text="Building portfolio...")
    portfolio = build_portfolio(recs, data, risk_profile, portfolio_val, min_score)

    bt = None
    if run_bt and bench is not None:
        prog.progress(0.96, text="Backtesting...")
        bt = run_multi_period_backtest(data, bench, risk_profile, top_n=10)

    prog.empty()
    st.session_state.update({"recs": recs, "data": data, "bench": bench,
                             "portfolio": portfolio, "bt": bt, "done": True,
                             "stock_map": stock_map})

# ═══════════════════════════════════════════════════════════════════════════════
#  Display
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("done"):
    recs = st.session_state["recs"]
    data = st.session_state["data"]
    bench = st.session_state["bench"]
    portfolio = st.session_state["portfolio"]
    bt = st.session_state.get("bt")
    stock_map = st.session_state.get("stock_map", {})

    top = [r for r in recs if r["composite"] >= min_score]
    buys = sum(1 for r in recs if r["action"] in ("STRONG BUY","BUY","ACCUMULATE"))
    avg_sc = np.mean([r["composite"] for r in recs]) if recs else 0
    breadth = market_breadth(data)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(mc(len(recs),"Scanned","mc-cyan"), True)
    with c2: st.markdown(mc(len(top),"Above Threshold","mc-green"), True)
    with c3: st.markdown(mc(buys,"Buy Signals","mc-green"), True)
    with c4: st.markdown(mc(f"{avg_sc:.0f}","Avg Score","mc-purple"), True)
    with c5: st.markdown(mc(f"{breadth['above_50sma_pct']:.0f}%",f"Breadth · {breadth['label']}","mc-amber"), True)
    with c6: st.markdown(mc(portfolio.get("total_stocks",0),"Portfolio","mc-cyan"), True)

    tabs = st.tabs(["◉ TOP PICKS","📊 SCREENER","🔬 DEEP DIVE","💼 PORTFOLIO","⏪ BACKTEST","📡 MARKET"])

    # ══════════ TAB 1: TOP PICKS ═══════════════════════════════════════════
    with tabs[0]:
        st.markdown(f"### Top Recommendations — {len(top)} stocks above score {min_score}")
        if not top:
            st.info("No stocks above threshold. Lower the minimum score in sidebar.")

        per_page = 20
        total_pages = max(1, (len(top) + per_page - 1) // per_page)
        page = st.number_input("Page", 1, total_pages, 1, key="tp_page") if total_pages > 1 else 1
        page_items = top[(page-1)*per_page : page*per_page]

        for i, r in enumerate(page_items):
            idx = (page-1)*per_page + i + 1
            stage_txt = r['stage'][1] if isinstance(r['stage'], tuple) else str(r['stage'])
            with st.expander(f"#{idx}  {r['name']}  ·  {r['ticker'].replace('.NS','')}  ·  Score {r['composite']:.0f}  ·  {r['action']}  ·  {stage_txt}", expanded=(i < 3)):
                # Score cards row
                sc1,sc2,sc3,sc4,sc5,sc6 = st.columns(6)
                sc1.metric("Composite", f"{r['composite']:.0f}/100")
                sc2.metric("Technical", f"{r['tech_score']:.0f}")
                sc3.metric("Fundamental", f"{r['fund_score']:.0f}")
                sc4.metric("Momentum", f"{r['momentum']:.0f}")
                sc5.metric("RS Rank", f"{r['rs_pct']:.0f}th")
                sc6.metric("Price", f"₹{r['price']:,.2f}")

                st.markdown(f"{action_badge(r['action'])} &nbsp; Risk: {risk_span(r['risk_level'])} &nbsp; "
                            f"Trend: **{r['trend']}** &nbsp; Vol: **{r['volatility']:.0f}%** &nbsp; "
                            f"Sector: **{r['sector']}** &nbsp; Cap: **{r['cap']}**", True)

                # Detailed metrics grid
                rets = r.get("returns", {})
                metrics_html = '<div class="detail-grid">'
                metrics_html += detail_item("1 Week", f"{rets.get('1w',0):+.1f}%", "#00ff87" if rets.get('1w',0)>0 else "#ff4757")
                metrics_html += detail_item("1 Month", f"{rets.get('1m',0):+.1f}%", "#00ff87" if rets.get('1m',0)>0 else "#ff4757")
                metrics_html += detail_item("3 Months", f"{rets.get('3m',0):+.1f}%", "#00ff87" if rets.get('3m',0)>0 else "#ff4757")
                metrics_html += detail_item("6 Months", f"{rets.get('6m',0):+.1f}%", "#00ff87" if rets.get('6m',0)>0 else "#ff4757")
                metrics_html += detail_item("From 52W High", f"{r['from_high']:+.1f}%")
                metrics_html += detail_item("From 52W Low", f"{r['from_low']:+.1f}%")
                for k, v in r.get("metrics", {}).items():
                    metrics_html += detail_item(k, str(v) if v else "—")
                metrics_html += '</div>'
                st.markdown(metrics_html, True)

                # Indicators + Holding
                ll, rr = st.columns(2)
                with ll:
                    st.markdown("**Technical Indicators**")
                    for ind_name, ind_data in r.get("indicators", {}).items():
                        sc = ind_data["s"]
                        st.markdown(f"`{ind_name}` **{sc:.0f}**/100 — {ind_data['r']}")
                        st.progress(max(0, min(100, int(sc))) / 100)
                with rr:
                    st.markdown("**Fundamentals**")
                    for cat, bd in r.get("fund_breakdown", {}).items():
                        sc = bd["s"]
                        st.markdown(f"`{cat}` **{sc:.0f}**/100")
                        st.progress(max(0, min(100, int(sc))) / 100)
                        for n in bd.get("n", [])[:2]:
                            st.caption(f"  · {n}")

                h1, h2 = st.columns(2)
                with h1:
                    st.info(f"**Hold:** {r['holding']}")
                    st.caption(r["hold_reason"])
                    st.caption(f"**Exit:** {r['exit']}")
                with h2:
                    st.success(f"**Allocate:** {r['alloc_pct']:.1f}% → ₹{r['alloc_amt']:,}")
                    if r.get("risk_factors"):
                        st.markdown("**Risk Factors:**")
                        for rf in r["risk_factors"][:3]:
                            st.caption(f"⚠ {rf}")

    # ══════════ TAB 2: FULL SCREENER ═══════════════════════════════════════
    with tabs[1]:
        st.markdown(f"### Full Screener — {len(recs)} stocks")
        rows = []
        for r in recs:
            rets = r.get("returns", {})
            rows.append({
                "Stock": r["name"], "Ticker": r["ticker"].replace(".NS",""),
                "Sector": r["sector"], "Cap": r["cap"],
                "Score": r["composite"], "Tech": r["tech_score"],
                "Fund": r["fund_score"], "Mom": r["momentum"],
                "RS%": r["rs_pct"],
                "Action": r["action"], "Risk": r["risk_level"],
                "Stage": (r["stage"][1].split("—")[0].strip() if isinstance(r["stage"], tuple) else str(r["stage"])),
                "Price": r["price"],
                "1W%": rets.get("1w",0), "1M%": rets.get("1m",0),
                "3M%": rets.get("3m",0), "6M%": rets.get("6m",0),
                "52WH%": r["from_high"], "Vol%": r["volatility"],
                "Hold": r["holding"], "Alloc%": r["alloc_pct"], "Alloc₹": r["alloc_amt"],
            })

        df_full = pd.DataFrame(rows)
        if df_full.empty:
            st.info("No data to display.")
        else:
            fc1, fc2 = st.columns(2)
            with fc1:
                act_filt = st.multiselect("Action Filter",
                    ["STRONG BUY","BUY","ACCUMULATE","HOLD","UNDERWEIGHT","AVOID"],
                    default=["STRONG BUY","BUY","ACCUMULATE","HOLD"], key="sf_act")
            with fc2:
                sort_by = st.selectbox("Sort by", ["Score","Tech","Fund","Mom","RS%","1W%","1M%","3M%","Price"], key="sf_sort")

            if act_filt and "Action" in df_full.columns:
                df_full = df_full[df_full["Action"].isin(act_filt)]
            df_full = df_full.sort_values(sort_by, ascending=False)

            st.dataframe(df_full, use_container_width=True, height=700,
                         column_config={
                             "Score": st.column_config.ProgressColumn("Score",min_value=0,max_value=100,format="%.0f"),
                             "Tech":  st.column_config.ProgressColumn("Tech",min_value=0,max_value=100,format="%.0f"),
                             "Fund":  st.column_config.ProgressColumn("Fund",min_value=0,max_value=100,format="%.0f"),
                             "Mom":   st.column_config.ProgressColumn("Mom",min_value=0,max_value=100,format="%.0f"),
                             "Price": st.column_config.NumberColumn(format="₹%.2f"),
                             "1W%":   st.column_config.NumberColumn(format="%.1f%%"),
                             "1M%":   st.column_config.NumberColumn(format="%.1f%%"),
                             "3M%":   st.column_config.NumberColumn(format="%.1f%%"),
                             "6M%":   st.column_config.NumberColumn(format="%.1f%%"),
                             "Alloc₹":st.column_config.NumberColumn(format="₹%d"),
                         })
            st.download_button("Download CSV", df_full.to_csv(index=False).encode(), "stockpulse_scan.csv", "text/csv")

            # Sector heatmap
            if rows:
                sec_avg = pd.DataFrame(rows).groupby("Sector")[["Score","Tech","Fund","Mom"]].agg(["mean","count"])
                sec_avg.columns = [f"{c[0]}_{c[1]}" for c in sec_avg.columns]
                sec_avg = sec_avg.sort_values("Score_mean", ascending=True)
                sec_avg_reset = sec_avg.reset_index()

                fig_s = go.Figure()
                for col, clr in [("Score_mean","#00d4ff"),("Tech_mean","#a855f7"),("Fund_mean","#00ff87"),("Mom_mean","#ffb700")]:
                    fig_s.add_trace(go.Bar(y=sec_avg_reset["Sector"], x=sec_avg_reset[col],
                                          name=col.replace("_mean",""), orientation="h",
                                          marker_color=clr, opacity=0.8))
                fig_s.update_layout(**CL, barmode="group", height=max(300, len(sec_avg)*30),
                                   title="Sector Average Scores")
                st.plotly_chart(fig_s, use_container_width=True)

    # ══════════ TAB 3: DEEP DIVE ═══════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Detailed Stock Analysis")
        stock_opts = {f"{r['name']} ({r['ticker'].replace('.NS','')}) — Score {r['composite']:.0f}": r for r in recs}
        sel = st.selectbox("Select stock", list(stock_opts.keys()), key="dd_sel")
        if sel:
            r = stock_opts[sel]
            t = r["ticker"]

            # Header metrics
            dc = st.columns(7)
            dc[0].metric("Composite", f"{r['composite']:.0f}/100")
            dc[1].metric("Technical", f"{r['tech_score']:.0f}")
            dc[2].metric("Fundamental", f"{r['fund_score']:.0f}")
            dc[3].metric("Momentum", f"{r['momentum']:.0f}")
            dc[4].metric("RS Rank", f"{r['rs_pct']:.0f}th")
            dc[5].metric("Volatility", f"{r['volatility']:.0f}%")
            dc[6].metric("Price", f"₹{r['price']:,.2f}")

            stage_txt = r['stage'][1] if isinstance(r['stage'], tuple) else str(r['stage'])
            st.markdown(f"{action_badge(r['action'])} &nbsp; Risk: {risk_span(r['risk_level'])} &nbsp; "
                        f"Stage: **{stage_txt}** &nbsp; Trend: **{r['trend']}** &nbsp; "
                        f"Sector: **{r['sector']}** &nbsp; Cap: **{r['cap']}**", True)

            # Returns & metrics grid
            rets = r.get("returns", {})
            g = '<div class="detail-grid">'
            for k, v in rets.items():
                clr = "#00ff87" if v > 0 else "#ff4757"
                g += detail_item(k, f"{v:+.1f}%", clr)
            g += detail_item("From 52W High", f"{r['from_high']:+.1f}%")
            g += detail_item("From 52W Low", f"{r['from_low']:+.1f}%")
            for k, v in r.get("metrics", {}).items():
                g += detail_item(k, str(v) if v else "—")
            g += '</div>'
            st.markdown(g, True)

            # Chart
            price_df = data.get(t, {}).get("price")
            if price_df is not None and len(price_df) > 30:
                cdf = add_indicators(price_df.copy())
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                    row_heights=[0.55,0.22,0.23])
                fig.add_trace(go.Candlestick(x=cdf.index, open=cdf["Open"], high=cdf["High"],
                    low=cdf["Low"], close=cdf["Close"], name="Price",
                    increasing_line_color="#00ff87", decreasing_line_color="#ff4757"), row=1, col=1)
                for nm, col, clr in [("EMA21","EMA21","#ffb700"),("SMA50","SMA50","#00d4ff"),("SMA200","SMA200","#a855f7")]:
                    if col in cdf.columns:
                        s = cdf[col].dropna()
                        if len(s) > 0:
                            fig.add_trace(go.Scatter(x=s.index,y=s,name=nm,line=dict(color=clr,width=1.2)),row=1,col=1)
                if "BB_U" in cdf.columns:
                    fig.add_trace(go.Scatter(x=cdf.index,y=cdf["BB_U"],line=dict(color="rgba(100,120,140,.3)",width=1,dash="dot"),showlegend=False),row=1,col=1)
                    fig.add_trace(go.Scatter(x=cdf.index,y=cdf["BB_L"],line=dict(color="rgba(100,120,140,.3)",width=1,dash="dot"),fill="tonexty",fillcolor="rgba(100,120,140,.03)",showlegend=False),row=1,col=1)
                if "MACD_H" in cdf.columns:
                    colors = ["#00ff87" if v>=0 else "#ff4757" for v in cdf["MACD_H"].fillna(0)]
                    fig.add_trace(go.Bar(x=cdf.index,y=cdf["MACD_H"],name="Hist",marker_color=colors,opacity=0.6),row=2,col=1)
                    fig.add_trace(go.Scatter(x=cdf.index,y=cdf["MACD"],name="MACD",line=dict(color="#00d4ff",width=1.2)),row=2,col=1)
                    fig.add_trace(go.Scatter(x=cdf.index,y=cdf["MACD_S"],name="Signal",line=dict(color="#ffb700",width=1.2)),row=2,col=1)
                if "RSI" in cdf.columns:
                    fig.add_trace(go.Scatter(x=cdf.index,y=cdf["RSI"],name="RSI",line=dict(color="#a855f7",width=1.5)),row=3,col=1)
                    fig.add_hline(y=70,line_dash="dot",line_color="rgba(255,71,87,.4)",row=3,col=1)
                    fig.add_hline(y=30,line_dash="dot",line_color="rgba(0,255,135,.4)",row=3,col=1)
                fig.update_layout(**CL, height=680, xaxis_rangeslider_visible=False,
                                  title=f"{r['name']} ({r['ticker'].replace('.NS','')})")
                st.plotly_chart(fig, use_container_width=True)

            # Indicator breakdown
            il, ir = st.columns(2)
            with il:
                st.markdown("#### Technical Indicators")
                for ind_nm, ind_d in r.get("indicators", {}).items():
                    sc = ind_d["s"]
                    color = "#00ff87" if sc >= 65 else ("#ffb700" if sc >= 40 else "#ff4757")
                    st.markdown(f"**{ind_nm}** — <span style='color:{color};font-weight:700'>{sc:.0f}/100</span>", True)
                    st.progress(max(0, min(100, int(sc))) / 100)
                    st.caption(ind_d["r"])
            with ir:
                st.markdown("#### Fundamental Breakdown")
                for cat, bd in r.get("fund_breakdown", {}).items():
                    sc = bd["s"]
                    color = "#00ff87" if sc >= 65 else ("#ffb700" if sc >= 40 else "#ff4757")
                    st.markdown(f"**{cat}** — <span style='color:{color};font-weight:700'>{sc:.0f}/100</span>", True)
                    st.progress(max(0, min(100, int(sc))) / 100)
                    for n in bd.get("n", []):
                        st.caption(f"  · {n}")

            # Holding + Risk
            h1, h2 = st.columns(2)
            with h1:
                st.markdown("#### Holding & Exit Strategy")
                st.info(r["holding"])
                st.caption(r["hold_reason"])
                st.markdown(f"**Exit:** {r['exit']}")
                st.markdown(f"**Allocation:** {r['alloc_pct']:.1f}% → ₹{r['alloc_amt']:,}")
            with h2:
                st.markdown("#### Risk Assessment")
                st.metric("Risk Score", f"{r['risk_score']}/100", delta=r["risk_level"])
                for rf in r.get("risk_factors", []):
                    st.caption(f"⚠ {rf}")

    # ══════════ TAB 4: PORTFOLIO ═══════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Optimised Portfolio")
        if portfolio.get("error"):
            st.warning(portfolio["error"])
        elif portfolio.get("holdings"):
            pc = st.columns(5)
            with pc[0]: st.markdown(mc(portfolio["total_stocks"],"Stocks","mc-cyan"),True)
            with pc[1]: st.markdown(mc(f"₹{portfolio['total_invested']:,}","Invested","mc-green"),True)
            with pc[2]: st.markdown(mc(f"{portfolio['avg_composite']:.0f}","Avg Score","mc-purple"),True)
            with pc[3]: st.markdown(mc(f"{portfolio['portfolio_volatility']:.0f}%","Port Vol","mc-amber"),True)
            with pc[4]: st.markdown(mc(f"{portfolio['diversification_score']}","Diversity","mc-cyan"),True)

            port_df = pd.DataFrame([{
                "Stock": h["name"], "Ticker": h["ticker"].replace(".NS",""),
                "Sector": h["sector"], "Weight%": h["weight_pct"],
                "Amount": h["amount"], "Qty": h["qty"],
                "Score": h["composite"], "Risk": h["risk_level"],
                "Stage": h["stage"], "Holding": h["holding"],
            } for h in portfolio["holdings"]])
            st.dataframe(port_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Score": st.column_config.ProgressColumn("Score",min_value=0,max_value=100,format="%.0f"),
                             "Amount": st.column_config.NumberColumn(format="₹%d"),
                             "Weight%": st.column_config.NumberColumn(format="%.1f%%"),
                         })

            sec_alloc = portfolio.get("sector_allocation", {})
            if sec_alloc:
                fig_pie = px.pie(values=list(sec_alloc.values()), names=list(sec_alloc.keys()),
                                 hole=0.55, color_discrete_sequence=px.colors.qualitative.Set2)
                fig_pie.update_layout(**CL, height=350, title="Sector Allocation")
                fig_pie.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No portfolio generated. Run scan first.")

    # ══════════ TAB 5: BACKTEST ════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Strategy Backtest")
        if bt is None:
            st.info("Enable 'Run Backtest' and scan again.")
        elif bt.get("summary"):
            summ = bt["summary"]
            bc = st.columns(4)
            with bc[0]:
                c = "mc-green" if summ["avg_return"] > 0 else "mc-red"
                st.markdown(mc(f"{summ['avg_return']:+.1f}%","Avg Return",c),True)
            with bc[1]:
                c = "mc-green" if summ["avg_alpha"] > 0 else "mc-red"
                st.markdown(mc(f"{summ['avg_alpha']:+.1f}%","Alpha vs Nifty",c),True)
            with bc[2]:
                st.markdown(mc(f"{summ['avg_win_rate']:.0f}%","Win Rate","mc-cyan"),True)
            with bc[3]:
                st.markdown(mc(f"{summ['beat_benchmark_pct']:.0f}%","Beat Nifty","mc-purple"),True)

            for p in bt.get("periods", []):
                if "error" in p:
                    st.warning(f"{p.get('label','')}: {p['error']}")
                    continue
                with st.expander(f"{p['label']}  ·  Portfolio: {p['portfolio_return']:+.1f}%  vs  Nifty: {p['benchmark_return']:+.1f}%  ·  Alpha: {p['alpha']:+.1f}%"):
                    pc = st.columns(5)
                    pc[0].metric("Portfolio", f"{p['portfolio_return']:+.1f}%")
                    pc[1].metric("Nifty", f"{p['benchmark_return']:+.1f}%")
                    pc[2].metric("Alpha", f"{p['alpha']:+.1f}%")
                    pc[3].metric("Win Rate", f"{p['win_rate']:.0f}%")
                    pc[4].metric("Sharpe", f"{p['sharpe']:.2f}")
                    if p.get("top_picks"):
                        tp_df = pd.DataFrame(p["top_picks"])[["name","sector","composite","tech_score","fund_score","actual_return"]]
                        tp_df.columns = ["Stock","Sector","Score","Tech","Fund","Return%"]
                        st.dataframe(tp_df, use_container_width=True, hide_index=True,
                                     column_config={"Return%": st.column_config.NumberColumn(format="%.1f%%")})

            valid_p = [p for p in bt.get("periods",[]) if "error" not in p]
            if valid_p:
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Bar(x=[p["label"] for p in valid_p], y=[p["portfolio_return"] for p in valid_p],
                                        name="Portfolio", marker_color="#00ff87"))
                fig_bt.add_trace(go.Bar(x=[p["label"] for p in valid_p], y=[p["benchmark_return"] for p in valid_p],
                                        name="Nifty 50", marker_color="#a855f7"))
                fig_bt.update_layout(**CL, barmode="group", height=350, title="Portfolio vs Benchmark")
                st.plotly_chart(fig_bt, use_container_width=True)

    # ══════════ TAB 6: MARKET ══════════════════════════════════════════════
    with tabs[5]:
        m1, m2 = st.columns([1, 2])
        with m1:
            st.markdown("#### Market Breadth")
            st.markdown(mc(f"{breadth['above_50sma_pct']:.0f}%", f"Above 50 SMA · {breadth['label']}", "mc-amber"), True)
            st.caption(f"{breadth['above']}/{breadth['total']} stocks above 50-day SMA")

            st.markdown("#### Score Distribution")
            scores = [r["composite"] for r in recs]
            fig_h = px.histogram(x=scores, nbins=25, color_discrete_sequence=["#00d4ff"])
            fig_h.add_vline(x=min_score, line_dash="dash", line_color="#ff4757",
                            annotation_text=f"Threshold ({min_score})")
            fig_h.update_layout(**CL, height=280, showlegend=False, xaxis_title="Score", yaxis_title="Count")
            st.plotly_chart(fig_h, use_container_width=True)

            st.markdown("#### Actions Breakdown")
            act_c = {}
            for r in recs:
                act_c[r["action"]] = act_c.get(r["action"], 0) + 1
            fig_ac = px.pie(values=list(act_c.values()), names=list(act_c.keys()), hole=0.5,
                            color_discrete_sequence=["#00ff87","#00d4ff","#a855f7","#ffb700","#ff4757","#8b0000"])
            fig_ac.update_layout(**CL, height=280)
            fig_ac.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
            st.plotly_chart(fig_ac, use_container_width=True)

        with m2:
            st.markdown("#### Latest Market News")
            news = fetch_news()
            if news:
                for a in news[:10]:
                    st.markdown(f"**[{a['title']}]({a['link']})**")
                    if a["date"]:
                        st.caption(a["date"])
                    st.markdown("---")
            else:
                st.info("Unable to fetch news.")

            st.markdown("#### Top Movers")
            movers = sorted(recs, key=lambda x: x.get("returns",{}).get("1w",0), reverse=True)
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Top Gainers (1W)**")
                for m in movers[:8]:
                    ret = m.get("returns",{}).get("1w",0)
                    st.markdown(f'<span style="color:#00ff87;font-weight:600">{m["name"]}</span> — {ret:+.1f}%', True)
            with g2:
                st.markdown("**Top Losers (1W)**")
                for m in movers[-8:]:
                    ret = m.get("returns",{}).get("1w",0)
                    st.markdown(f'<span style="color:#ff4757;font-weight:600">{m["name"]}</span> — {ret:+.1f}%', True)

else:
    # ── Welcome ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="glass" style="text-align:center;padding:2.5rem">
    <h3 style="color:var(--ac)">Select your universe & hit SCAN MARKET</h3>
    <p style="color:var(--tx2);max-width:650px;margin:.5rem auto">
        Choose Nifty 50, 100, or the full <b>Nifty 500</b> in the sidebar.
        The screener fetches live constituent lists from NSE, pulls price data
        and fundamentals, then runs multi-factor scoring on every stock.
    </p>
</div>
""", True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown('<div class="glass"><h4>Multi-Factor Scoring</h4><p style="color:var(--tx2);font-size:.85rem">'
                    'RSI, MACD, MAs, ADX, Bollinger, Volume for technicals. '
                    'P/E, ROE, Growth, Debt, Cash Flow for fundamentals. '
                    'Stage 2 detection (Minervini). Relative Strength ranking.</p></div>', True)
    with fc2:
        st.markdown('<div class="glass"><h4>Backtested</h4><p style="color:var(--tx2);font-size:.85rem">'
                    'Walk-forward backtest on historical data. '
                    'Measures win rate, alpha vs Nifty 50, Sharpe ratio. '
                    'Multiple periods tested for consistency.</p></div>', True)
    with fc3:
        st.markdown('<div class="glass"><h4>Smart Portfolio</h4><p style="color:var(--tx2);font-size:.85rem">'
                    'Inverse-volatility weighting, correlation filtering, sector caps. '
                    'Risk-parity allocation ensures portfolio-level protection. '
                    'Exact quantities and amounts for your portfolio size.</p></div>', True)

    news = fetch_news()
    if news:
        st.markdown("#### Latest Headlines")
        for a in news[:5]:
            st.markdown(f"· **[{a['title']}]({a['link']})**")
