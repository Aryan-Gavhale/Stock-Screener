"""
StockPulse India v5 — Personal Alpha Engine
Momentum + Quality strategy · Entry setups · ATR risk sizing · Regime filter
· Look-ahead-free backtest · Active positions tracker
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RISK_PROFILES, ANALYSIS_PERIODS, SECTOR_ROTATION
from stock_universe import (fetch_index_constituents, fetch_sector_constituents,
                            get_all_sectors, filter_stocks, reclassify_caps)
from data_fetcher import (fetch_all, fetch_nifty, fetch_news, market_breadth,
                          fetch_market_context, fetch_risk_news, breadth_series,
                          fetch_sector_indices, fetch_all_sector_news)
from technical_analysis import technical_score, relative_strength, rs_percentile, add_indicators
from fundamental_analysis import fundamental_score
from scoring_engine import recommend, rank, exit_signal
from risk_engine import detect_regime, portfolio_heat
from market_risk import crash_risk, accumulation_zone
from backtester import run_backtest
from sector_data import build_sector_dataset
from sector_rotation import compute_sector_forecasts
from sector_stock_classifier import classify_all_sectors
from sector_backtester import run_sector_backtest, calibrate_probabilities
from sector_ui import render_sector_radar
import positions as pos_mod

st.set_page_config(
    page_title="StockPulse India", page_icon="◉", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#06080d;--sf:#0d1117;--sf2:#151b25;--bd:#1e2a3a;--tx:#e2e8f0;--tx2:#8b9ab5;--ac:#00d4ff;--gn:#00ff87;--rd:#ff4757;--am:#ffb700;--pu:#a855f7}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--tx)!important;font-family:'Outfit',sans-serif!important}
[data-testid="stSidebar"]{background:var(--sf)!important;border-right:1px solid var(--bd)!important}
[data-testid="stSidebar"] *{color:var(--tx)!important}
header[data-testid="stHeader"]{display:block!important;background:transparent!important}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"]{display:flex!important;visibility:visible!important;opacity:1!important;z-index:999999!important}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button{color:var(--ac)!important;background:var(--sf2)!important;border:1px solid var(--bd)!important;border-radius:8px!important}
footer{display:none!important}#MainMenu{display:none!important}
h1,h2,h3,h4{font-family:'Outfit',sans-serif!important;font-weight:700!important;color:var(--tx)!important}
.block-container{padding-top:1.2rem!important;max-width:1550px!important}
.hero{background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(168,85,247,.08));border:1px solid var(--bd);border-radius:16px;padding:1.5rem 2rem;margin-bottom:1rem}
.hero-title{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#00d4ff,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
.hero-sub{color:var(--tx2);font-size:.88rem;margin-top:.2rem}
.mcard{background:var(--sf2);border:1px solid var(--bd);border-radius:12px;padding:.9rem 1rem;text-align:center}
.mcard-val{font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace}
.mcard-label{font-size:.7rem;color:var(--tx2);text-transform:uppercase;letter-spacing:1px;margin-top:.15rem}
.mc-cyan .mcard-val{color:var(--ac)}.mc-green .mcard-val{color:var(--gn)}.mc-amber .mcard-val{color:var(--am)}.mc-red .mcard-val{color:var(--rd)}.mc-purple .mcard-val{color:var(--pu)}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.b-buy{background:rgba(0,255,135,.12);color:var(--gn);border:1px solid rgba(0,255,135,.3)}
.b-watch{background:rgba(0,212,255,.10);color:var(--ac);border:1px solid rgba(0,212,255,.3)}
.b-hold{background:rgba(255,183,0,.10);color:var(--am);border:1px solid rgba(255,183,0,.3)}
.b-avoid{background:rgba(255,71,87,.10);color:var(--rd);border:1px solid rgba(255,71,87,.3)}
.b-burst{background:rgba(168,85,247,.14);color:var(--pu);border:1px solid rgba(168,85,247,.35)}
.b-setup{background:rgba(139,154,181,.12);color:var(--tx);border:1px solid var(--bd)}
.regime{border-radius:14px;padding:1rem 1.3rem;margin-bottom:1rem;border:1px solid var(--bd)}
.reg-green{background:linear-gradient(135deg,rgba(0,255,135,.10),rgba(0,255,135,.02));border-color:rgba(0,255,135,.35)}
.reg-amber{background:linear-gradient(135deg,rgba(255,183,0,.10),rgba(255,183,0,.02));border-color:rgba(255,183,0,.35)}
.reg-red{background:linear-gradient(135deg,rgba(255,71,87,.10),rgba(255,71,87,.02));border-color:rgba(255,71,87,.35)}
div[data-testid="stMetric"]{background:var(--sf2)!important;border:1px solid var(--bd)!important;border-radius:12px!important;padding:.8rem!important}
div[data-testid="stMetric"] label{color:var(--tx2)!important;font-size:.7rem!important;text-transform:uppercase!important;letter-spacing:.5px!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--tx)!important;font-family:'JetBrains Mono',monospace!important;font-size:1.3rem!important}
div[data-testid="stExpander"]{background:var(--sf)!important;border:1px solid var(--bd)!important;border-radius:12px!important}
div[data-testid="stExpander"] summary{color:var(--tx)!important;font-weight:600!important}
.stProgress>div>div{background:var(--sf2)!important;border-radius:4px!important}
button[kind="primary"]{background:linear-gradient(135deg,#00d4ff,#a855f7)!important;border:none!important;color:#000!important;font-weight:700!important;border-radius:10px!important}
.stSelectbox [data-baseweb="select"]>div,.stMultiSelect [data-baseweb="select"]>div{background:var(--sf2)!important;border-color:var(--bd)!important;color:var(--tx)!important}
input[type="number"],input[type="text"]{background:var(--sf2)!important;border-color:var(--bd)!important;color:var(--tx)!important}
.stTabs [data-baseweb="tab-list"]{background:var(--sf)!important;border-radius:12px;padding:4px;border:1px solid var(--bd);flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{border-radius:10px!important;padding:6px 14px!important;color:var(--tx2)!important;font-weight:500!important}
.stTabs [aria-selected="true"]{background:var(--sf2)!important;color:var(--ac)!important}
.stTabs [data-baseweb="tab-border"],.stTabs [data-baseweb="tab-highlight"]{display:none!important}
.stDownloadButton>button{background:var(--sf2)!important;border:1px solid var(--bd)!important;color:var(--ac)!important;border-radius:10px!important}
.disclaimer{background:rgba(255,183,0,.06);border:1px solid rgba(255,183,0,.2);border-radius:10px;padding:.7rem .9rem;font-size:.76rem;color:var(--am)}
.plan{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.5rem;margin:.6rem 0}
.plan-item{background:var(--sf2);border:1px solid var(--bd);border-radius:10px;padding:.6rem .8rem}
.plan-item .l{font-size:.66rem;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px}
.plan-item .v{font-size:1.05rem;font-weight:600;font-family:'JetBrains Mono',monospace;margin-top:.1rem}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────────
def action_badge(a):
    c = {"STRONG BUY":"b-buy","BUY":"b-buy","WATCH":"b-watch","HOLD":"b-hold"}.get(a,"b-avoid")
    return f'<span class="badge {c}">{a}</span>'

def setup_badge(s):
    return f'<span class="badge b-setup">{s}</span>'

def bucket_badge(b):
    return f'<span class="badge b-burst">BURST</span>' if b=="Burst" else f'<span class="badge b-setup">CORE</span>'

def risk_span(l):
    c={"Low":"var(--gn)","Medium":"var(--am)","High":"var(--rd)"}.get(l,"var(--tx)")
    return f'<span style="color:{c};font-weight:600">{l}</span>'

def mc(val,lbl,cls="mc-cyan"):
    return f'<div class="mcard {cls}"><div class="mcard-val">{val}</div><div class="mcard-label">{lbl}</div></div>'

def plan_item(l,v,color=None):
    cs=f'color:{color}' if color else ''
    return f'<div class="plan-item"><div class="l">{l}</div><div class="v" style="{cs}">{v}</div></div>'

def table_column_help():
    st.caption("Hidden a column? Hover over the table, click the eye icon in its top-right toolbar, then enable the column again.")

CL=dict(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(13,17,23,0.6)",
        font=dict(family="Outfit,sans-serif",color="#8b9ab5"),
        xaxis=dict(gridcolor="rgba(30,42,58,.5)"),yaxis=dict(gridcolor="rgba(30,42,58,.5)"),
        margin=dict(l=50,r=20,t=40,b=30),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    index_raw = st.selectbox("Stock Universe",["Nifty 50","Nifty 100","Nifty 500"],index=2,
                             help="Nifty 500 = full opportunity set incl. mid/small-cap bursts")
    capital = st.number_input("Capital (₹)",50000,10_00_00_000,5_00_000,50000)
    risk_profile = st.selectbox("Strategy Profile",list(RISK_PROFILES.keys()),index=1)
    st.caption(RISK_PROFILES[risk_profile]["desc"])
    st.markdown("---")
    cap_filter = st.multiselect("Market Cap",["Large Cap","Mid Cap","Small Cap"],
                                default=RISK_PROFILES[risk_profile]["preferred_caps"])
    needs_broad = ("Mid Cap" in cap_filter or "Small Cap" in cap_filter)
    if needs_broad and index_raw in ("Nifty 50","Nifty 100"):
        index_choice="Nifty 500"; st.info(f"Auto-switched to Nifty 500 ({index_raw} is large-cap only).")
    else:
        index_choice=index_raw
    universe = fetch_index_constituents(index_choice)
    all_sectors = get_all_sectors(universe) if universe else []
    sector_filter = st.multiselect("Sectors",["All"]+all_sectors,default=["All"])
    period_key = st.selectbox("Data Period",list(ANALYSIS_PERIODS.keys()),index=3,
                              help="Use 3Y/5Y for a meaningful backtest")
    sector_horizon = st.selectbox(
        "Sector Forecast Horizon",
        list(SECTOR_ROTATION["horizons"].keys()),
        index=list(SECTOR_ROTATION["horizons"]).index(
            SECTOR_ROTATION["default_horizon"]
        ),
    )
    max_scan = st.slider("Max Stocks to Scan",10,len(universe) if universe else 500,
                         min(len(universe),500) if universe else 50)
    run_bt = st.checkbox("Run Backtest (needs 3Y+)",value=True)
    run = st.button("◉  SCAN MARKET",type="primary",width="stretch")
    st.markdown("---")
    if universe: st.caption(f"Universe: **{len(universe)}** stocks · {index_choice}")
    st.markdown('<div class="disclaimer"><b>⚠</b> Personal research tool. Not advice. '
                'Always honor your stop-loss.</div>',unsafe_allow_html=True)

st.markdown(f"""<div class="hero"><p class="hero-title">StockPulse India</p>
<p class="hero-sub">{index_choice} · Momentum + Quality · Entry setups · ATR risk-sizing · Regime-filtered · Backtested</p></div>""",unsafe_allow_html=True)

# ── Pipeline ─────────────────────────────────────────────────────────────────────
def get_tickers():
    all_sec = "All" in sector_filter or not sector_filter
    filtered = filter_stocks(universe,None if all_sec else sector_filter,None)
    return list(filtered.keys())[:max_scan], filtered

def analyse(data,bench,stock_map,cap,deploy,stop_tighten=1.0,
            sector_signals=None):
    umap = reclassify_caps(stock_map,data)
    all_rs = {t:(relative_strength(d["price"],bench) if bench is not None else 1.0) for t,d in data.items()}
    rs_vals = list(all_rs.values())
    recs=[]
    for t,d in data.items():
        info_t = umap.get(t,(t,"Other","Unknown","?"))
        name,sector,capc = info_t[0],info_t[1],info_t[2]
        if cap_filter and capc not in cap_filter: continue
        tech = technical_score(d["price"],bench)
        tech["rs"]=all_rs.get(t,1.0)
        tech["rs_pct"]=rs_percentile(all_rs.get(t,1.0),rs_vals)
        fund = fundamental_score(d["info"],sector)
        recs.append(recommend(
            t,name,sector,capc,tech,fund,d["info"],risk_profile,cap,deploy,
            stop_tighten,(sector_signals or {}).get(sector)
        ))
    return rank(recs)

def build_portfolio(recs,profile,cap,deploy):
    p=RISK_PROFILES[profile]
    cands=[r for r in recs if r["action"] in ("STRONG BUY","BUY")]
    cands.sort(key=lambda r:r["composite"],reverse=True)
    max_stocks=p["max_stocks"]; max_sec=max(2,max_stocks//3)
    burst_cap=p["burst_alloc"]*cap; budget=deploy*cap
    max_heat=p["max_portfolio_heat"]*cap
    chosen=[];secct={};burst=0;spent=0;heat=0
    for r in cands:
        if len(chosen)>=max_stocks: break
        if secct.get(r["sector"],0)>=max_sec: continue
        amt=r.get("alloc_amt",0)
        if amt<=0: continue
        if r["bucket"]=="Burst" and burst+amt>burst_cap: continue
        if spent+amt>budget: continue
        if heat+r.get("risk_amt",0)>max_heat and chosen: continue
        chosen.append(r); secct[r["sector"]]=secct.get(r["sector"],0)+1
        spent+=amt; heat+=r.get("risk_amt",0)
        if r["bucket"]=="Burst": burst+=amt
    return {"holdings":chosen,"invested":round(spent),"cash":round(cap-spent),
            "heat":round(heat),"heat_pct":round(heat/cap*100,2) if cap else 0,
            "burst_amt":round(burst),"n":len(chosen)}
# ── Run ──────────────────────────────────────────────────────────────────────────
if run:
    tickers,stock_map = get_tickers()
    if not tickers:
        st.warning("No stocks match filters."); st.stop()
    prog = st.progress(0,text=f"Scanning {len(tickers)} stocks...")
    data = fetch_all(tickers,ANALYSIS_PERIODS[period_key],lambda pct,txt="":prog.progress(min(pct,1.0),text=txt))
    bench = fetch_nifty(ANALYSIS_PERIODS[period_key])
    prog.progress(0.86,text="Reading crash radar (VIX, global, breadth, news)...")
    breadth = market_breadth(data)
    bser = breadth_series(data)
    context = fetch_market_context()
    risknews = fetch_risk_news()
    crash = crash_risk(bench,data,context,risknews,bser)
    accum = accumulation_zone(bench,data,context,bser)
    regime = detect_regime(bench,breadth["above_50sma_pct"],crash)
    prog.progress(0.88,text="Ranking sector rotation and emerging leadership...")
    sector_indices = fetch_sector_indices("10y")
    sector_memberships = fetch_sector_constituents()
    sector_bench = fetch_nifty("10y")
    if sector_bench is None:
        sector_bench = bench
    sector_dataset = (
        build_sector_dataset(
            data,stock_map,sector_indices,sector_bench,sector_memberships
        )
        if sector_bench is not None else {}
    )
    preliminary_sectors = (
        compute_sector_forecasts(
            sector_dataset,sector_bench,sector_horizon,context,regime
        ) if sector_bench is not None else []
    )
    news_sectors = [item["sector"] for item in preliminary_sectors[:8]]
    sector_news = fetch_all_sector_news(news_sectors) if news_sectors else {}
    sector_forecasts = (
        compute_sector_forecasts(
            sector_dataset,sector_bench,sector_horizon,context,regime,sector_news
        ) if sector_bench is not None else []
    )
    sector_bt=None
    if run_bt and sector_bench is not None:
        prog.progress(0.91,text="Walk-forward validating sector rotation...")
        official_dataset = {
            sector:item for sector,item in sector_dataset.items()
            if item.get("source")=="Official"
        }
        sector_bt = run_sector_backtest(
            official_dataset,sector_bench,sector_horizon
        )
        if sector_bt and not sector_bt.get("error"):
            ablation = sector_bt.get("ablations",{})
            if (ablation.get("plus_seasonality_rank_ic",0)
                    <= ablation.get("price_breadth_rank_ic",0)):
                sector_forecasts = compute_sector_forecasts(
                    sector_dataset,sector_bench,sector_horizon,context,regime,
                    sector_news,disabled_overlays={"seasonality"}
                )
            sector_forecasts = calibrate_probabilities(
                sector_forecasts,sector_bt.get("calibration",[])
            )
            validated_tilt = (
                sector_bt.get("rank_ic",0) > 0
                and sector_bt.get("spread",0) > 0
            )
            for item in sector_forecasts:
                item["tilt_enabled"] = validated_tilt
                if not validated_tilt:
                    item["risks"].append(
                        "Portfolio tilt disabled: this horizon failed "
                        "out-of-sample rank/spread gates"
                    )
    sector_signals = {
        item["sector"]:item for item in sector_forecasts
    }
    prog.progress(0.93,text="Scoring stocks with capped sector evidence...")
    recs = analyse(
        data,bench,stock_map,capital,regime["deploy"],
        regime.get("stop_tighten",1.0),sector_signals
    )
    sector_tiers = classify_all_sectors(recs,sector_forecasts,sector_dataset)
    prog.progress(0.95,text="Constructing risk-controlled portfolio...")
    portfolio = build_portfolio(recs,risk_profile,capital,regime["deploy"])
    bt=None
    if run_bt and bench is not None:
        prog.progress(0.97,text="Backtesting stock strategy (look-ahead-free)...")
        bt = run_backtest(data,bench,risk_profile)
    prog.empty()
    st.session_state.update({"recs":recs,"data":data,"bench":bench,"portfolio":portfolio,
                             "bt":bt,"regime":regime,"breadth":breadth,"crash":crash,"accum":accum,
                             "context":context,"risknews":risknews,"bser":bser,
                             "sector_forecasts":sector_forecasts,
                             "sector_tiers":sector_tiers,"sector_bt":sector_bt,
                             "sector_horizon":sector_horizon,
                             "done":True,"capital":capital})

# ── Display ──────────────────────────────────────────────────────────────────────
if st.session_state.get("done"):
    recs=st.session_state["recs"]; data=st.session_state["data"]; bench=st.session_state["bench"]
    portfolio=st.session_state["portfolio"]; bt=st.session_state.get("bt")
    regime=st.session_state["regime"]; breadth=st.session_state["breadth"]
    crash=st.session_state.get("crash",{}); accum=st.session_state.get("accum",{})
    context=st.session_state.get("context",{}); risknews=st.session_state.get("risknews",{})
    bser=st.session_state.get("bser")
    sector_forecasts=st.session_state.get("sector_forecasts",[])
    sector_tiers=st.session_state.get("sector_tiers",{})
    sector_bt=st.session_state.get("sector_bt")
    active_sector_horizon=st.session_state.get(
        "sector_horizon",SECTOR_ROTATION["default_horizon"]
    )
    capital=st.session_state.get("capital",capital)

    # Regime banner (now includes Crash Radar level + auto de-risk)
    rc={"green":"reg-green","amber":"reg-amber","red":"reg-red"}[regime["color"]]
    hedge_line = f"<br><span style='color:var(--rd);font-weight:600'>⚠ Auto de-risk: {regime['hedge']}</span>" if regime.get("hedge") else ""
    st.markdown(f"""<div class="regime {rc}">
    <span style="font-size:1.1rem;font-weight:700">Regime: {regime['label']}</span>
    &nbsp;·&nbsp; <span style="color:var(--tx2)">{regime['stance']}</span><br>
    <span style="color:var(--tx2);font-size:.82rem">{' · '.join(regime['reasons'])} · Deploy {regime['deploy']*100:.0f}% of risk budget</span>{hedge_line}
    </div>""",unsafe_allow_html=True)

    buys=[r for r in recs if r["action"] in ("STRONG BUY","BUY")]
    bursts=[r for r in recs if r["bucket"]=="Burst" and r["action"] in ("STRONG BUY","BUY","WATCH")]
    dips=sorted([r for r in recs if r.get("is_dip")],key=lambda r:r["dip_score"],reverse=True)
    avg=np.mean([r["composite"] for r in recs]) if recs else 0
    crash_idx=crash.get("index",0); crash_lbl=crash.get("label","—")
    crash_cls={"green":"mc-green","amber":"mc-amber","red":"mc-red"}.get(crash.get("color","green"),"mc-cyan")

    c=st.columns(7)
    c[0].markdown(mc(len(recs),"Scanned","mc-cyan"),unsafe_allow_html=True)
    c[1].markdown(mc(len(buys),"Buy Setups","mc-green"),unsafe_allow_html=True)
    c[2].markdown(mc(len(bursts),"Burst","mc-purple"),unsafe_allow_html=True)
    c[3].markdown(mc(len(dips),"Dip Buys","mc-cyan"),unsafe_allow_html=True)
    c[4].markdown(mc(f"{crash_idx:.0f}",f"Crash·{crash_lbl}",crash_cls),unsafe_allow_html=True)
    c[5].markdown(mc(f"{breadth['above_50sma_pct']:.0f}%",f"Breadth·{breadth['label']}","mc-cyan"),unsafe_allow_html=True)
    c[6].markdown(mc(portfolio["n"],"In Portfolio","mc-green"),unsafe_allow_html=True)

    (tab_action,tab_sector,tab_crash,tab_dip,tab_burst,tab_screener,tab_deep,
     tab_port,tab_pos,tab_bt,tab_market)=st.tabs(
        ["◉ ACTION","SECTOR RADAR","🛡 CRASH RADAR","💧 DIP BUYS","🚀 BURST","📊 SCREENER","🔬 DEEP DIVE",
         "💼 PORTFOLIO","📈 MY POSITIONS","⏪ BACKTEST","📡 MARKET"])

    # ═══ TAB: ACTION ═══
    with tab_action:
        st.markdown(f"### Actionable Setups — {len(buys)} buy candidates")
        st.caption("Only names in a confirmed uptrend with a real entry trigger (breakout/pullback). Each has a stop, target, and risk-sized quantity.")
        if not buys:
            st.info("No clean buy setups right now. In a weak regime this is normal — patience preserves capital.")
        for i,r in enumerate(buys[:40]):
            g = "var(--gn)" if r["composite"]>=65 else "var(--ac)"
            with st.expander(f"#{i+1}  {r['name']} ({r['ticker'].replace('.NS','')})  ·  {r['action']}  ·  {r['setup']}  ·  Score {r['composite']:.0f}",expanded=(i<4)):
                st.markdown(f"{action_badge(r['action'])} {setup_badge(r['setup'])} {bucket_badge(r['bucket'])} "
                            f"&nbsp; Risk: {risk_span(r['risk_level'])} &nbsp; RS: **{r['rs_pct']:.0f}th** &nbsp; "
                            f"Stage: **{r['stage'][1] if isinstance(r['stage'],tuple) else r['stage']}** &nbsp; "
                            f"{r['sector']} · Sector: **{r.get('sector_status','Unrated')}** "
                            f"({r.get('sector_tilt',0):+.1f} pts)",unsafe_allow_html=True)
                st.caption(f"📍 {r['setup_note']}")
                st.markdown('<div class="plan">'
                    +plan_item("Entry",f"₹{r['entry']:,.1f}")
                    +plan_item("Stop",f"₹{r['stop']:,.1f}","#ff4757")
                    +plan_item("Stop Dist",f"-{r['stop_pct']:.1f}%","#ff4757")
                    +plan_item("Target",f"₹{r['target']:,.1f}","#00ff87")
                    +plan_item("Target",f"+{r['target_pct']:.1f}%","#00ff87")
                    +plan_item("R:R",f"1:{r['rr']:.0f}")
                    +plan_item("Qty",f"{r['qty']}")
                    +plan_item("Invest",f"₹{r['alloc_amt']:,}")
                    +plan_item("Risk",f"₹{r['risk_amt']:,} ({r['risk_pct']:.1f}%)","#ffb700")
                    +'</div>',unsafe_allow_html=True)
                sc=st.columns(5)
                sc[0].metric("Composite",f"{r['composite']:.0f}")
                sc[1].metric("Trend",f"{r['tech_score']:.0f}")
                sc[2].metric("Growth",f"{r['growth_s']:.0f}")
                sc[3].metric("Quality",f"{r['quality_s']:.0f}")
                sc[4].metric("Momentum",f"{r['momentum']:.0f}")
                st.info(f"**Plan:** {r['holding']} — {r['hold_reason']}  \n**Exit rule:** {r['exit']}")
                if r["earnings_accel"]: st.caption("⚡ Earnings accelerating (forward P/E < trailing)")
                if r["red_flags"]:
                    st.caption("⚠ " + " · ".join(r["red_flags"]))

    # ═══ TAB: SECTOR ROTATION ═══
    with tab_sector:
        render_sector_radar(
            sector_forecasts,sector_tiers,sector_bt,active_sector_horizon
        )

    # ═══ TAB: CRASH RADAR ═══
    with tab_crash:
        st.markdown("### 🛡 Crash Radar")
        st.caption("A probabilistic gauge of market-wide crash risk — NOT a prediction. It reads conditions that historically precede sharp falls so you can de-risk proactively. Expect false alarms.")
        gcol=st.columns([1,2])
        with gcol[0]:
            gfig=go.Figure(go.Indicator(mode="gauge+number",value=crash.get("index",0),
                title={"text":f"Crash Risk · {crash.get('label','—')}","font":{"size":15}},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":{"green":"#00ff87","amber":"#ffb700","red":"#ff4757"}.get(crash.get("color","green"),"#00d4ff")},
                       "steps":[{"range":[0,30],"color":"rgba(0,255,135,.12)"},
                                {"range":[30,55],"color":"rgba(255,183,0,.12)"},
                                {"range":[55,75],"color":"rgba(255,71,87,.12)"},
                                {"range":[75,100],"color":"rgba(255,71,87,.25)"}]}))
            gfig.update_layout(**{k:v for k,v in CL.items() if k not in ("xaxis","yaxis")},height=250)
            st.plotly_chart(gfig,width="stretch")
            st.info(crash.get("stance","—"))
            if regime.get("hedge"):
                st.error(f"Auto de-risk: {regime['hedge']}")
        with gcol[1]:
            comp=crash.get("components",{})
            if comp:
                labels={"trend":"Trend","breadth":"Breadth","distribution":"Distribution","divergence":"Divergence","vix":"India VIX","global":"Global","news":"News"}
                cf=go.Figure(go.Bar(x=[comp.get(k,0) for k in labels],y=[labels[k] for k in labels],orientation="h",
                    marker_color=["#ff4757" if comp.get(k,0)>=55 else ("#ffb700" if comp.get(k,0)>=30 else "#00d4ff") for k in labels]))
                cf.update_layout(**CL,height=250,xaxis_title="Risk contribution (0-100)",title="What's driving the risk")
                st.plotly_chart(cf,width="stretch")
            st.markdown("**Signals firing:**")
            for s in crash.get("signals",[]):
                st.markdown(f"- {s}")
        stt=crash.get("stats",{})
        sc=st.columns(6)
        sc[0].markdown(mc(stt.get("india_vix","—"),"India VIX","mc-amber"),unsafe_allow_html=True)
        sc[1].markdown(mc(f"{stt.get('breadth','—')}%","Breadth","mc-cyan"),unsafe_allow_html=True)
        sc[2].markdown(mc(stt.get("distribution_days","—"),"Distrib. Days","mc-red"),unsafe_allow_html=True)
        sc[3].markdown(mc(f"{stt.get('above_200dma_pct','—')}%","Nifty vs 200DMA","mc-cyan"),unsafe_allow_html=True)
        sc[4].markdown(mc(f"{stt.get('usdinr_1m','—')}%","USDINR 1M","mc-amber"),unsafe_allow_html=True)
        sc[5].markdown(mc(f"{stt.get('crude_1m','—')}%","Crude 1M","mc-purple"),unsafe_allow_html=True)
        # Nifty + 200DMA and India VIX charts
        ch=st.columns(2)
        with ch[0]:
            if bench is not None and len(bench)>60:
                nb=bench.tail(400)
                nf=go.Figure()
                nf.add_trace(go.Scatter(x=nb.index,y=nb["Close"],name="Nifty",line=dict(color="#00d4ff",width=1.6)))
                if len(bench)>=200:
                    nf.add_trace(go.Scatter(x=nb.index,y=bench["Close"].rolling(200).mean().tail(400),name="200-DMA",line=dict(color="#a855f7",width=1.2,dash="dot")))
                nf.update_layout(**CL,height=260,title="Nifty vs 200-DMA")
                st.plotly_chart(nf,width="stretch")
        with ch[1]:
            ivix=context.get("india_vix")
            if ivix is not None and len(ivix)>10:
                vv=ivix.tail(400)
                vf=go.Figure()
                vf.add_trace(go.Scatter(x=vv.index,y=vv["Close"],name="India VIX",line=dict(color="#ffb700",width=1.6)))
                vf.add_hline(y=20,line_dash="dot",line_color="rgba(255,71,87,.5)")
                vf.update_layout(**CL,height=260,title="India VIX (fear gauge)")
                st.plotly_chart(vf,width="stretch")
            elif bser is not None and len(bser)>10:
                bf=go.Figure()
                bf.add_trace(go.Scatter(x=bser.index,y=bser.values,name="Breadth",line=dict(color="#00ff87",width=1.6)))
                bf.add_hline(y=40,line_dash="dot",line_color="rgba(255,71,87,.5)")
                bf.update_layout(**CL,height=260,title="% stocks above 50-DMA (breadth)")
                st.plotly_chart(bf,width="stretch")
        rn=risknews.get("headlines",[]) if risknews else []
        if rn:
            with st.expander(f"Risk-scan headlines (news-risk {risknews.get('score',0)}/100)"):
                if risknews.get("matched"):
                    st.caption("Matched terms: " + ", ".join(risknews["matched"]))
                for a in rn[:8]:
                    st.markdown(f"- [{a['title']}]({a['link']})")
        st.markdown('<div class="disclaimer"><b>Caveat:</b> This gauge stacks odds, it does not foresee the future. '
                    'Use it to size down and tighten stops when risk is high — not to make all-or-nothing calls.</div>',unsafe_allow_html=True)

    # ═══ TAB: DIP BUYS ═══
    with tab_dip:
        st.markdown("### 💧 Dip Buys — quality on sale")
        az=accum.get("status","No")
        az_cls={"Confirmed":"reg-green","Forming":"reg-amber","Watching":"reg-red","No":"reg-green"}.get(az,"reg-amber")
        st.markdown(f"""<div class="regime {az_cls}">
        <span style="font-size:1.05rem;font-weight:700">Accumulation Zone: {az}</span>
        &nbsp;·&nbsp; <span style="color:var(--tx2)">Nifty {accum.get('drawdown',0):+.1f}% from 6-month high · {accum.get('bull_signals',0)} stabilization signals</span><br>
        <span style="color:var(--tx2);font-size:.82rem">{' · '.join(accum.get('reasons',[])[:3])}</span>
        </div>""",unsafe_allow_html=True)
        if az in ("Forming","Confirmed"):
            st.success("Market is stabilizing after a fall — this is when the dip list below is most actionable. Accumulate in tranches.")
        elif az=="Watching":
            st.warning("Market fell but hasn't turned yet. Prefer to WAIT — don't catch a falling knife. The list shows quality names to build a watchlist.")
        else:
            st.info("No broad market dip right now. These are individual names that fell hard but look stable — treat as opportunistic, not urgent.")
        st.caption("Screen: fell ≥25% from 52-week high · fundamentals intact (quality, low debt, no red flags) · showing ≥2 stabilization signals. Mean-reversion play, distinct from the momentum list.")
        if not dips:
            st.info("No quality-on-sale dip candidates in the current universe.")
        else:
            for i,r in enumerate(dips[:30]):
                with st.expander(f"#{i+1}  {r['name']} ({r['ticker'].replace('.NS','')})  ·  Dip {r['dip_score']:.0f}  ·  {r['from_high']:.0f}% from high",expanded=(i<4)):
                    st.markdown(f"{bucket_badge(r['bucket'])} &nbsp; Risk: {risk_span(r['risk_level'])} &nbsp; "
                                f"Quality: **{r['quality_s']:.0f}** &nbsp; Value: **{r['value_s']:.0f}** &nbsp; "
                                f"RS: **{r['rs_pct']:.0f}th** &nbsp; {r['sector']} · {r['cap']}",unsafe_allow_html=True)
                    st.caption(f"📍 {r['dip_note']}")
                    tr=r.get("dip_tranches",[])
                    tr_html="".join(plan_item(f"Tranche {k+1}",f"₹{lvl:,.1f}") for k,lvl in enumerate(tr))
                    st.markdown('<div class="plan">'
                        +plan_item("Price",f"₹{r['price']:,.1f}")
                        +plan_item("Fell",f"{r['from_high']:.0f}%","#ff4757")
                        +plan_item("Capit. Low",f"₹{r.get('dip_cap_low',0):,.1f}")
                        +plan_item("Stop",f"₹{r.get('dip_stop',0):,.1f}","#ff4757")
                        +tr_html
                        +'</div>',unsafe_allow_html=True)
                    if r.get("dip_signals"):
                        st.markdown("**Stabilization:** " + " · ".join(r["dip_signals"]))
                    if r["earnings_accel"]: st.caption("⚡ Earnings still growing")
                    if r["red_flags"]: st.caption("⚠ " + " · ".join(r["red_flags"]))
            drows=[{"Stock":r["name"],"Ticker":r["ticker"].replace(".NS",""),"Sector":r["sector"],
                "Dip":r["dip_score"],"Fell%":r["from_high"],"Quality":r["quality_s"],"Value":r["value_s"],
                "Price":r["price"],"CapLow":r.get("dip_cap_low"),"Stop":r.get("dip_stop")} for r in dips]
            st.dataframe(pd.DataFrame(drows),width="stretch",hide_index=True,column_config={
                "Dip":st.column_config.ProgressColumn("Dip",min_value=0,max_value=100,format="%.0f"),
                "Quality":st.column_config.ProgressColumn("Quality",min_value=0,max_value=100,format="%.0f"),
                "Fell%":st.column_config.NumberColumn(format="%.0f%%"),
                "Price":st.column_config.NumberColumn(format="₹%.1f"),
                "CapLow":st.column_config.NumberColumn(format="₹%.1f"),
                "Stop":st.column_config.NumberColumn(format="₹%.1f")})
            table_column_help()
        st.markdown('<div class="disclaimer"><b>Caveat:</b> Buying dips is mean-reversion — it fights the trend. '
                    'Only quality names with stabilization qualify here, but accumulate in tranches and honor the stop below the capitulation low.</div>',unsafe_allow_html=True)

    # ═══ TAB: BURST ═══
    with tab_burst:
        st.markdown(f"### 🚀 Burst / Multibagger Candidates — {len(bursts)}")
        st.caption("Small/mid-caps with accelerating earnings + strong relative strength near highs. High reward, high risk — size SMALL, honor stops. A few 2-3x winners carry the book.")
        burst_notice=st.session_state.pop("burst_add_notice",None)
        if burst_notice:
            st.success(burst_notice)
        if not bursts:
            st.info("No burst candidates meet the criteria in the current universe/regime.")
        else:
            brows=[]
            for r in bursts:
                rets=r.get("returns",{})
                brows.append({"Stock":r["name"],"Ticker":r["ticker"].replace(".NS",""),"Sector":r["sector"],
                    "MCap Cr":r["metrics"].get("MCap Cr","—"),"Score":r["composite"],"Setup":r["setup"],
                    "RS%":r["rs_pct"],"RevGr":r["metrics"].get("Rev Gr","—"),"EarnGr":r["metrics"].get("Earn Gr","—"),
                    "6M%":rets.get("6m",0),"Entry":r["entry"],"Stop":r["stop"],"Target":r["target"],
                    "Qty":r["qty"],"Invest":r["alloc_amt"],"Action":r["action"]})
            bdf=pd.DataFrame(brows)
            st.dataframe(bdf,width="stretch",hide_index=True,column_config={
                "Score":st.column_config.ProgressColumn("Score",min_value=0,max_value=100,format="%.0f"),
                "Entry":st.column_config.NumberColumn(format="₹%.1f"),
                "Stop":st.column_config.NumberColumn(format="₹%.1f"),
                "Target":st.column_config.NumberColumn(format="₹%.1f"),
                "Invest":st.column_config.NumberColumn(format="₹%d"),
                "6M%":st.column_config.NumberColumn(format="%.0f%%")})
            table_column_help()
            st.markdown("#### Add Burst to My Positions")
            st.caption("Adds the selected burst at its recommended entry, stop-loss, and quantity. Existing My Positions remain unchanged.")
            tracked=pos_mod.load_positions()
            tracked_tickers={str(p.get("ticker","")).upper() for p in tracked}
            for r in bursts:
                full_ticker=r["ticker"] if r["ticker"].endswith(".NS") else f"{r['ticker']}.NS"
                already_added=full_ticker.upper() in tracked_tickers
                no_quantity=r.get("qty",0)<=0
                bc=st.columns([2.4,1,1,0.8,1.5])
                bc[0].markdown(f"**{r['name']}**  \n`{full_ticker.replace('.NS','')}`")
                bc[1].metric("Entry",f"₹{r['entry']:,.1f}")
                bc[2].metric("Stop",f"₹{r['stop']:,.1f}")
                bc[3].metric("Qty",f"{r['qty']}")
                if already_added:
                    button_label="Already in My Positions"
                elif no_quantity:
                    button_label="No quantity allocated"
                else:
                    button_label="Add to My Positions"
                if bc[4].button(
                    button_label,
                    key=f"add_burst_{full_ticker}",
                    disabled=already_added or no_quantity,
                    width="stretch",
                ):
                    pos_mod.add_position(
                        full_ticker,r["name"],r["entry"],r["qty"],r["stop"]
                    )
                    st.session_state["burst_add_notice"]=(
                        f"Added {r['name']} at ₹{r['entry']:,.1f}, "
                        f"stop ₹{r['stop']:,.1f}, quantity {r['qty']}."
                    )
                    st.rerun()
                st.divider()
    # ═══ TAB: SCREENER ═══
    with tab_screener:
        st.markdown(f"### Full Screener — {len(recs)} stocks")
        rows=[]
        for r in recs:
            rets=r.get("returns",{})
            rows.append({"Stock":r["name"],"Ticker":r["ticker"].replace(".NS",""),"Sector":r["sector"],
                "Cap":r["cap"],"Bucket":r["bucket"],"Score":r["composite"],"Trend":r["tech_score"],
                "Growth":r["growth_s"],"Quality":r["quality_s"],"Value":r["value_s"],"Mom":r["momentum"],
                "RS%":r["rs_pct"],"Setup":r["setup"],"Action":r["action"],"Risk":r["risk_level"],
                "SectorStatus":r.get("sector_status","Unrated"),
                "SectorFwd":r.get("sector_forward",50),
                "SectorTilt":r.get("sector_tilt",0),
                "Stage":(r["stage"][1].split("—")[0].strip() if isinstance(r["stage"],tuple) else str(r["stage"])),
                "Price":r["price"],"Entry":r["entry"],"Stop":r["stop"],"Target":r["target"],
                "1M%":rets.get("1m",0),"3M%":rets.get("3m",0),"6M%":rets.get("6m",0),
                "52WH%":r["from_high"],"Vol%":r["volatility"]})
        df=pd.DataFrame(rows)
        if df.empty:
            st.info("No data.")
        else:
            f1,f2,f3=st.columns(3)
            with f1: af=st.multiselect("Action",["STRONG BUY","BUY","WATCH","HOLD","AVOID"],default=["STRONG BUY","BUY","WATCH"],key="scr_a")
            with f2: sf=st.multiselect("Setup",sorted(df["Setup"].unique()),default=[],key="scr_s")
            with f3: sortby=st.selectbox("Sort",["Score","Trend","Growth","Quality","Mom","RS%","3M%","6M%"],key="scr_sort")
            if af: df=df[df["Action"].isin(af)]
            if sf: df=df[df["Setup"].isin(sf)]
            df=df.sort_values(sortby,ascending=False)
            st.dataframe(df,width="stretch",height=620,column_config={
                "Score":st.column_config.ProgressColumn("Score",min_value=0,max_value=100,format="%.0f"),
                "Trend":st.column_config.ProgressColumn("Trend",min_value=0,max_value=100,format="%.0f"),
                "Growth":st.column_config.ProgressColumn("Growth",min_value=0,max_value=100,format="%.0f"),
                "Quality":st.column_config.ProgressColumn("Quality",min_value=0,max_value=100,format="%.0f"),
                "Price":st.column_config.NumberColumn(format="₹%.1f"),
                "Entry":st.column_config.NumberColumn(format="₹%.1f"),
                "Stop":st.column_config.NumberColumn(format="₹%.1f"),
                "Target":st.column_config.NumberColumn(format="₹%.1f"),
                "1M%":st.column_config.NumberColumn(format="%.0f%%"),
                "3M%":st.column_config.NumberColumn(format="%.0f%%"),
                "6M%":st.column_config.NumberColumn(format="%.0f%%")})
            table_column_help()
            st.download_button("Download CSV",df.to_csv(index=False).encode(),"stockpulse_scan.csv","text/csv")

    # ═══ TAB: DEEP DIVE ═══
    with tab_deep:
        st.markdown("### Detailed Stock Analysis")
        opts={f"{r['name']} ({r['ticker'].replace('.NS','')}) — {r['composite']:.0f}":r for r in recs}
        sel=st.selectbox("Select stock",list(opts.keys()),key="dd")
        if sel:
            r=opts[sel]; t=r["ticker"]
            st.markdown(f"{action_badge(r['action'])} {setup_badge(r['setup'])} {bucket_badge(r['bucket'])} "
                        f"&nbsp; Risk: {risk_span(r['risk_level'])} &nbsp; Stage: **{r['stage'][1] if isinstance(r['stage'],tuple) else r['stage']}** &nbsp; {r['sector']} · {r['cap']}",unsafe_allow_html=True)
            st.caption(f"📍 {r['setup_note']}")
            dc=st.columns(8)
            dc[0].metric("Composite",f"{r['composite']:.0f}")
            dc[1].metric("Trend",f"{r['tech_score']:.0f}")
            dc[2].metric("Growth",f"{r['growth_s']:.0f}")
            dc[3].metric("Quality",f"{r['quality_s']:.0f}")
            dc[4].metric("Value",f"{r['value_s']:.0f}")
            dc[5].metric("RS Rank",f"{r['rs_pct']:.0f}th")
            dc[6].metric("Volatility",f"{r['volatility']:.0f}%")
            dc[7].metric("Price",f"₹{r['price']:,.1f}")
            st.markdown('<div class="plan">'
                +plan_item("Entry",f"₹{r['entry']:,.1f}")
                +plan_item("Stop",f"₹{r['stop']:,.1f}","#ff4757")
                +plan_item("Target",f"₹{r['target']:,.1f}","#00ff87")
                +plan_item("R:R",f"1:{r['rr']:.0f}")
                +plan_item("Qty",f"{r['qty']}")
                +plan_item("Invest",f"₹{r['alloc_amt']:,}")
                +plan_item("Risk",f"₹{r['risk_amt']:,}","#ffb700")
                +'</div>',unsafe_allow_html=True)
            price_df=data.get(t,{}).get("price")
            if price_df is not None and len(price_df)>30:
                cdf=add_indicators(price_df.copy())
                fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.56,0.22,0.22])
                fig.add_trace(go.Candlestick(x=cdf.index,open=cdf["Open"],high=cdf["High"],low=cdf["Low"],close=cdf["Close"],
                    name="Price",increasing_line_color="#00ff87",decreasing_line_color="#ff4757"),row=1,col=1)
                for nm,col,clr in [("EMA21","EMA21","#ffb700"),("SMA50","SMA50","#00d4ff"),("SMA200","SMA200","#a855f7")]:
                    if col in cdf.columns:
                        s=cdf[col].dropna()
                        if len(s)>0: fig.add_trace(go.Scatter(x=s.index,y=s,name=nm,line=dict(color=clr,width=1.2)),row=1,col=1)
                fig.add_hline(y=r["stop"],line_dash="dash",line_color="#ff4757",row=1,col=1)
                fig.add_hline(y=r["target"],line_dash="dash",line_color="#00ff87",row=1,col=1)
                if "MACD_H" in cdf.columns:
                    colors=["#00ff87" if v>=0 else "#ff4757" for v in cdf["MACD_H"].fillna(0)]
                    fig.add_trace(go.Bar(x=cdf.index,y=cdf["MACD_H"],name="Hist",marker_color=colors,opacity=0.6),row=2,col=1)
                    fig.add_trace(go.Scatter(x=cdf.index,y=cdf["MACD"],name="MACD",line=dict(color="#00d4ff",width=1.2)),row=2,col=1)
                if "RSI" in cdf.columns:
                    fig.add_trace(go.Scatter(x=cdf.index,y=cdf["RSI"],name="RSI",line=dict(color="#a855f7",width=1.5)),row=3,col=1)
                    fig.add_hline(y=70,line_dash="dot",line_color="rgba(255,71,87,.4)",row=3,col=1)
                    fig.add_hline(y=30,line_dash="dot",line_color="rgba(0,255,135,.4)",row=3,col=1)
                fig.update_layout(**CL,height=640,xaxis_rangeslider_visible=False,title=f"{r['name']} — stop (red) / target (green)")
                st.plotly_chart(fig,width="stretch")
            il,ir=st.columns(2)
            with il:
                st.markdown("#### Technical")
                for nm,d0 in r.get("indicators",{}).items():
                    sc=d0["s"]; clr="#00ff87" if sc>=65 else ("#ffb700" if sc>=45 else "#ff4757")
                    st.markdown(f"**{nm}** — <span style='color:{clr};font-weight:700'>{sc:.0f}</span>",unsafe_allow_html=True)
                    st.progress(max(0,min(100,int(sc)))/100); st.caption(d0["r"])
            with ir:
                st.markdown("#### Fundamental")
                for cat,bd in r.get("fund_breakdown",{}).items():
                    sc=bd["s"]; clr="#00ff87" if sc>=65 else ("#ffb700" if sc>=45 else "#ff4757")
                    st.markdown(f"**{cat}** — <span style='color:{clr};font-weight:700'>{sc:.0f}</span>",unsafe_allow_html=True)
                    st.progress(max(0,min(100,int(sc)))/100)
                    for nn in bd.get("n",[])[:3]: st.caption(f"· {nn}")
                if r["red_flags"]:
                    st.markdown("**Red Flags:**")
                    for f in r["red_flags"]: st.caption(f"⚠ {f}")
            st.info(f"**Holding:** {r['holding']} — {r['hold_reason']}  \n**Exit rule:** {r['exit']}")

    # ═══ TAB: PORTFOLIO ═══
    with tab_port:
        st.markdown("### Risk-Sized Portfolio")
        st.caption(f"Positions sized so each risks ~{RISK_PROFILES[risk_profile]['risk_per_trade']*100:.1f}% of capital at its stop. Deployment throttled by regime ({regime['label']}).")
        add_notice=st.session_state.pop("portfolio_add_notice",None)
        if add_notice:
            st.success(add_notice)
        if not portfolio["holdings"]:
            st.info("No positions — regime and/or lack of setups keep you in cash. That's a valid, capital-preserving state.")
        else:
            pcol=st.columns(5)
            pcol[0].markdown(mc(portfolio["n"],"Positions","mc-cyan"),unsafe_allow_html=True)
            pcol[1].markdown(mc(f"₹{portfolio['invested']:,}","Invested","mc-green"),unsafe_allow_html=True)
            pcol[2].markdown(mc(f"₹{portfolio['cash']:,}","Cash","mc-amber"),unsafe_allow_html=True)
            pcol[3].markdown(mc(f"{portfolio['heat_pct']:.1f}%","Portfolio Heat","mc-red"),unsafe_allow_html=True)
            pcol[4].markdown(mc(f"₹{portfolio['burst_amt']:,}","In Bursts","mc-purple"),unsafe_allow_html=True)
            prows=[{"Stock":h["name"],"Ticker":h["ticker"].replace(".NS",""),"Bucket":h["bucket"],
                "Sector":h["sector"],"Entry":h["entry"],"Stop":h["stop"],"Target":h["target"],
                "Qty":h["qty"],"Invest":h["alloc_amt"],"Weight%":h["alloc_pct"],"Risk₹":h["risk_amt"],
                "Score":h["composite"],"Setup":h["setup"]} for h in portfolio["holdings"]]
            pdf=pd.DataFrame(prows)
            st.dataframe(pdf,width="stretch",hide_index=True,column_config={
                "Score":st.column_config.ProgressColumn("Score",min_value=0,max_value=100,format="%.0f"),
                "Entry":st.column_config.NumberColumn(format="₹%.1f"),
                "Stop":st.column_config.NumberColumn(format="₹%.1f"),
                "Target":st.column_config.NumberColumn(format="₹%.1f"),
                "Invest":st.column_config.NumberColumn(format="₹%d"),
                "Risk₹":st.column_config.NumberColumn(format="₹%d"),
                "Weight%":st.column_config.NumberColumn(format="%.1f%%")})
            table_column_help()
            st.markdown("#### Add to My Positions")
            st.caption("This only appends the selected recommendation. Your existing My Positions portfolio remains unchanged.")
            tracked=pos_mod.load_positions()
            tracked_tickers={str(p.get("ticker","")).upper() for p in tracked}
            for h in portfolio["holdings"]:
                full_ticker=h["ticker"] if h["ticker"].endswith(".NS") else f"{h['ticker']}.NS"
                already_added=full_ticker.upper() in tracked_tickers
                ac=st.columns([2.4,1,1,0.8,1.5])
                ac[0].markdown(f"**{h['name']}**  \n`{full_ticker.replace('.NS','')}`")
                ac[1].metric("Entry",f"₹{h['entry']:,.1f}")
                ac[2].metric("Stop",f"₹{h['stop']:,.1f}")
                ac[3].metric("Qty",f"{h['qty']}")
                if ac[4].button(
                    "Already in My Positions" if already_added else "Add to My Positions",
                    key=f"add_portfolio_{full_ticker}",
                    disabled=already_added,
                    width="stretch",
                ):
                    pos_mod.add_position(
                        full_ticker,h["name"],h["entry"],h["qty"],h["stop"]
                    )
                    st.session_state["portfolio_add_notice"]=(
                        f"Added {h['name']} at ₹{h['entry']:,.1f}, "
                        f"stop ₹{h['stop']:,.1f}, quantity {h['qty']}."
                    )
                    st.rerun()
                st.divider()
            sec_alloc={}
            for h in portfolio["holdings"]:
                sec_alloc[h["sector"]]=sec_alloc.get(h["sector"],0)+h["alloc_amt"]
            if sec_alloc:
                fig=px.pie(values=list(sec_alloc.values()),names=list(sec_alloc.keys()),hole=0.55,
                           color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(**CL,height=330,title="Sector Allocation")
                fig.update_traces(textposition="inside",textinfo="percent+label",textfont_size=11)
                st.plotly_chart(fig,width="stretch")
            st.caption("My Positions is your persistent portfolio and is never replaced by a new market scan.")
    # ═══ TAB: MY POSITIONS ═══
    with tab_pos:
        st.markdown("### My Positions — live exit monitor")
        st.caption("Record what you actually own. Each scan re-checks the trend and tells you HOLD / TRIM / EXIT with your current ratcheting stop.")
        opts={f"{r['name']} ({r['ticker'].replace('.NS','')})":r for r in recs}
        pick=st.selectbox("Prefill from scan (optional)",["— manual —"]+list(opts.keys()),key="pos_pick")
        d_entry=float(opts[pick]["price"]) if pick in opts else 0.0
        d_stop=float(opts[pick]["stop"]) if pick in opts else 0.0
        d_name=pick.split(" (")[0] if pick in opts else ""
        d_tk=opts[pick]["ticker"].replace(".NS","") if pick in opts else ""
        with st.form("add_pos",clear_on_submit=True):
            fc=st.columns(5)
            tk=fc[0].text_input("Ticker",d_tk)
            nm=fc[1].text_input("Name",d_name)
            entry=fc[2].number_input("Entry ₹",value=d_entry,min_value=0.0,step=1.0)
            qty=fc[3].number_input("Qty",value=0,min_value=0,step=1)
            stop=fc[4].number_input("Stop ₹",value=d_stop,min_value=0.0,step=1.0)
            if st.form_submit_button("➕ Add Position",width="stretch") and tk and qty>0:
                pos_mod.add_position(tk,nm or tk,entry,qty,stop)
                st.success(f"Added {tk}"); st.rerun()

        positions=pos_mod.load_positions()
        if not positions:
            st.info("No positions tracked yet. Add one above (or from the Portfolio tab picks).")
        else:
            rows=pos_mod.monitor_positions(positions,data,exit_signal,technical_score,bench)
            summ=pos_mod.portfolio_summary(rows)
            sc=st.columns(5)
            sc[0].markdown(mc(summ["n"],"Holdings","mc-cyan"),unsafe_allow_html=True)
            sc[1].markdown(mc(f"₹{summ['invested']:,}","Invested","mc-cyan"),unsafe_allow_html=True)
            pcls="mc-green" if summ["pnl"]>=0 else "mc-red"
            sc[2].markdown(mc(f"₹{summ['pnl']:,}","Open P&L",pcls),unsafe_allow_html=True)
            sc[3].markdown(mc(f"{summ['pnl_pct']:+.1f}%","Return",pcls),unsafe_allow_html=True)
            sc[4].markdown(mc(f"{summ['exits']} / {summ['trims']}","Exit / Trim","mc-amber"),unsafe_allow_html=True)
            for r in rows:
                if r["action"]=="EXIT":
                    st.error(f"🔴 EXIT **{r['name']}** — {r['reason']}  ·  P&L {r.get('gain_pct',0):+.1f}%  ·  stop ₹{r.get('current_stop')}")
                elif r["action"]=="TRIM":
                    st.warning(f"🟡 TRIM **{r['name']}** — {r['reason']}  ·  P&L {r.get('gain_pct',0):+.1f}%")
            prows=[{"Stock":r["name"],"Ticker":r["ticker"].replace(".NS",""),"Entry":r["entry"],
                "Current":r.get("current"),"Stop":r.get("current_stop"),"Qty":r["qty"],
                "P&L%":r.get("gain_pct"),"P&L₹":r.get("pnl_amt"),"Action":r.get("action"),
                "Reason":r.get("reason"),"Stage":r.get("stage")} for r in rows]
            st.dataframe(pd.DataFrame(prows),width="stretch",hide_index=True,column_config={
                "Entry":st.column_config.NumberColumn(format="₹%.1f"),
                "Current":st.column_config.NumberColumn(format="₹%.1f"),
                "Stop":st.column_config.NumberColumn(format="₹%.1f"),
                "P&L%":st.column_config.NumberColumn(format="%.1f%%"),
                "P&L₹":st.column_config.NumberColumn(format="₹%d")})
            table_column_help()
            rc=st.columns([3,1])
            ridx=rc[0].selectbox("Remove position",range(len(positions)),
                                 format_func=lambda i:f"{positions[i]['name']} ({positions[i]['ticker'].replace('.NS','')})",key="rm")
            if rc[1].button("🗑 Remove",width="stretch"):
                pos_mod.remove_position(ridx); st.rerun()

    # ═══ TAB: BACKTEST ═══
    with tab_bt:
        st.markdown("### Strategy Backtest — look-ahead-free")
        if bt is None:
            st.info("Enable 'Run Backtest' in the sidebar and use a 3Y or 5Y data period, then scan.")
        elif bt.get("error"):
            st.warning(bt["error"])
        else:
            s=bt["strategy"]; b=bt["benchmark"]; dec=bt["decile"]
            st.caption(f"Period {bt['period']['start']} → {bt['period']['end']} ({bt['period']['years']}y) · "
                       f"monthly rebalance into top-{RISK_PROFILES[risk_profile]['max_stocks']} momentum leaders · "
                       f"regime filter + trailing stops + 0.4%/trade costs.")
            m=st.columns(4)
            cg="mc-green" if s.get("cagr",0)>=b.get("cagr",0) else "mc-amber"
            m[0].markdown(mc(f"{s.get('cagr',0):+.1f}%","Strategy CAGR",cg),unsafe_allow_html=True)
            m[1].markdown(mc(f"{bt['alpha_cagr']:+.1f}%","Alpha vs Nifty","mc-purple"),unsafe_allow_html=True)
            m[2].markdown(mc(f"{s.get('max_drawdown',0):.1f}%","Max Drawdown","mc-red"),unsafe_allow_html=True)
            m[3].markdown(mc(f"{s.get('sharpe',0):.2f}","Sharpe","mc-cyan"),unsafe_allow_html=True)
            m2=st.columns(4)
            m2[0].markdown(mc(f"{s.get('win_rate',0):.0f}%","Win Rate","mc-cyan"),unsafe_allow_html=True)
            m2[1].markdown(mc(f"{s.get('profit_factor',0):.2f}","Profit Factor","mc-green"),unsafe_allow_html=True)
            avgw=s.get("avg_win",0);avgl=s.get("avg_loss",0)
            m2[2].markdown(mc(f"+{avgw:.0f}% / {avgl:.0f}%","Avg Win / Loss","mc-amber"),unsafe_allow_html=True)
            m2[3].markdown(mc(f"{s.get('exposure',0):.0f}%","Time Invested","mc-purple"),unsafe_allow_html=True)
            eq=bt["equity"]
            if eq["strategy"]:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=[p["date"] for p in eq["strategy"]],y=[p["value"] for p in eq["strategy"]],
                    name="Strategy",line=dict(color="#00ff87",width=2)))
                fig.add_trace(go.Scatter(x=[p["date"] for p in eq["benchmark"]],y=[p["value"] for p in eq["benchmark"]],
                    name="Nifty (buy & hold)",line=dict(color="#a855f7",width=1.6,dash="dot")))
                fig.update_layout(**CL,height=380,title="Growth of ₹1 (after costs)")
                st.plotly_chart(fig,width="stretch")
            st.markdown(f"**Benchmark (Nifty):** CAGR {b.get('cagr',0):+.1f}% · Max DD {b.get('max_drawdown',0):.1f}% · Sharpe {b.get('sharpe',0):.2f} · "
                        f"**Strategy trades:** {bt['n_trades']}")
            if dec.get("available"):
                st.markdown("#### Signal validation — forward return by decile")
                st.caption("If the momentum signal has edge, the highest-signal decile (10) should beat the lowest (1). This is the real proof the ranking predicts returns.")
                fig2=go.Figure()
                cols=["#ff4757" if v<0 else "#00d4ff" for v in dec["deciles"]]
                fig2.add_trace(go.Bar(x=[f"D{i+1}" for i in range(len(dec["deciles"]))],y=dec["deciles"],marker_color=cols))
                fig2.update_layout(**CL,height=320,title=f"Avg {dec['fwd_days']}-day forward return by signal decile ({dec['sample']} samples)",
                                   yaxis_title="Avg forward %")
                st.plotly_chart(fig2,width="stretch")
                verdict="✅ Signal has predictive edge" if dec["monotonic"] and dec["spread"]>0 else "⚠ Weak/inconsistent edge — treat picks cautiously"
                st.markdown(f"**{verdict}** — Top decile {dec['top_decile_fwd']:+.1f}% vs bottom {dec['bottom_decile_fwd']:+.1f}% "
                            f"(spread **{dec['spread']:+.1f}%** per {dec['fwd_days']} days).")
            st.markdown('<div class="disclaimer"><b>Honest caveats:</b> Fundamentals are current-snapshot (not point-in-time), '
                        'so this validates the price/momentum engine, not the fundamental tilt. Universe is current index members '
                        '(mild survivorship bias). Past performance ≠ future results.</div>',unsafe_allow_html=True)

    # ═══ TAB: MARKET ═══
    with tab_market:
        m1,m2=st.columns([1,2])
        with m1:
            st.markdown("#### Market Regime")
            st.markdown(mc(regime["label"],regime["stance"][:28],{"green":"mc-green","amber":"mc-amber","red":"mc-red"}[regime["color"]]),unsafe_allow_html=True)
            for rr in regime["reasons"]: st.caption(f"· {rr}")
            st.markdown("#### Score Distribution")
            scores=[r["composite"] for r in recs]
            fh=px.histogram(x=scores,nbins=25,color_discrete_sequence=["#00d4ff"])
            fh.update_layout(**CL,height=250,showlegend=False,xaxis_title="Composite",yaxis_title="Count")
            st.plotly_chart(fh,width="stretch")
            st.markdown("#### Actions")
            ac={}
            for r in recs: ac[r["action"]]=ac.get(r["action"],0)+1
            fa=px.pie(values=list(ac.values()),names=list(ac.keys()),hole=0.5,
                      color_discrete_sequence=["#00ff87","#00d4ff","#a855f7","#ffb700","#ff4757"])
            fa.update_layout(**CL,height=250); fa.update_traces(textposition="inside",textinfo="percent+label",textfont_size=10)
            st.plotly_chart(fa,width="stretch")
        with m2:
            st.markdown("#### Latest Market News")
            news=fetch_news()
            if news:
                for a in news[:8]:
                    st.markdown(f"**[{a['title']}]({a['link']})**")
                    if a["date"]: st.caption(a["date"])
                    st.markdown("---")
            else: st.info("Unable to fetch news.")
            st.markdown("#### Top Momentum Leaders (RS)")
            leaders=sorted(recs,key=lambda x:x["rs_pct"],reverse=True)[:10]
            for m in leaders:
                st.markdown(f'<span style="color:#00ff87;font-weight:600">{m["name"]}</span> — RS {m["rs_pct"]:.0f}th · {m["setup"]} · Score {m["composite"]:.0f}',unsafe_allow_html=True)

else:
    st.markdown("""<div style="text-align:center;padding:2.5rem;background:rgba(13,17,23,.85);border:1px solid #1e2a3a;border-radius:14px">
    <h3 style="color:#00d4ff">Your personal momentum + quality engine</h3>
    <p style="color:#8b9ab5;max-width:680px;margin:.5rem auto">
    Set your capital & profile, then <b>SCAN MARKET</b>. You'll get regime-aware buy setups with exact
    entry, stop, target and risk-sized quantity — plus a burst list for asymmetric upside, a look-ahead-free
    backtest, and a positions tracker that tells you when to sell.</p>
    <p style="color:#8b9ab5;font-size:.85rem">Tip: pick a <b>3Y or 5Y</b> data period so the backtest is meaningful.</p>
    </div>""",unsafe_allow_html=True)



