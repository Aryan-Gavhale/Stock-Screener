"""Streamlit renderer for the Sector Radar."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import positions as pos_mod


def _money(value):
    return f"₹{value:,.1f}" if isinstance(value, (int, float)) else "—"


def _add_stock_button(stock: dict, key_prefix: str):
    ticker = stock["ticker"] if stock["ticker"].endswith(".NS") else (
        f"{stock['ticker']}.NS"
    )
    tracked = {str(item.get("ticker", "")).upper()
               for item in pos_mod.load_positions()}
    already = ticker.upper() in tracked
    qty = int(stock.get("qty", 0) or 0)
    disabled = already or qty <= 0
    label = (
        "Already in My Positions" if already
        else "No quantity allocated" if qty <= 0
        else "Add to My Positions"
    )
    if st.button(
        label, key=f"{key_prefix}_{ticker}", disabled=disabled,
        width="stretch",
    ):
        pos_mod.add_position(
            ticker, stock["name"], stock["entry"], qty, stock["stop"]
        )
        st.session_state["sector_add_notice"] = (
            f"Added {stock['name']} at {_money(stock['entry'])}, "
            f"stop {_money(stock['stop'])}, quantity {qty}."
        )
        st.rerun()


def _stock_plan(stock: dict):
    columns = st.columns(6)
    columns[0].metric("Leadership", f"{stock.get('leadership_score', 0):.0f}")
    columns[1].metric("Sector RS", f"{stock.get('sector_rs_pct', 0):.0f}th")
    columns[2].metric("Entry", _money(stock.get("entry")))
    columns[3].metric("Stop", _money(stock.get("stop")))
    columns[4].metric("Target", _money(stock.get("target")))
    columns[5].metric("Qty", f"{stock.get('qty', 0)}")
    for reason in stock.get("leadership_reasons", []):
        st.caption(f"• {reason}")


def render_sector_radar(
    forecasts: list[dict],
    tiers: dict,
    backtest: dict | None,
    horizon: str,
):
    st.markdown("### Sector Rotation Radar")
    st.caption(
        f"{horizon} view. Forecasts are probabilities of outperforming Nifty, "
        "not guarantees. Price momentum and breadth form the core; live macro, "
        "news and fundamentals are capped overlays."
    )
    notice = st.session_state.pop("sector_add_notice", None)
    if notice:
        st.success(notice)
    if not forecasts:
        st.info("Sector histories are unavailable for this scan.")
        return

    booming = [item for item in forecasts if item["status"] == "Booming Now"]
    emerging = [item for item in forecasts if item["status"] == "Emerging"]
    fading = [item for item in forecasts if item["status"] in ("Fading", "Avoid")]
    metrics = st.columns(5)
    metrics[0].metric("Sectors ranked", len(forecasts))
    metrics[1].metric("Booming", len(booming))
    metrics[2].metric("Emerging", len(emerging))
    metrics[3].metric("Fading / Avoid", len(fading))
    metrics[4].metric(
        "Top forecast",
        f"{forecasts[0]['forward_rotation']:.0f}/100",
        forecasts[0]["sector"],
    )

    rows = []
    for rank, item in enumerate(forecasts, start=1):
        rows.append({
            "Rank": rank, "Sector": item["sector"], "Status": item["status"],
            "Current": item["current_strength"],
            "Forward": item["forward_rotation"],
            "Probability%": item["probability"],
            "Confidence": item["confidence"],
            "Momentum": item["momentum"],
            "Acceleration": item["acceleration"],
            "Breadth": item["breadth"],
            "Macro": item["macro_fit"],
            "Catalyst": item["catalyst"],
            "Seasonality": (
                "Zero-weighted (failed OOS)"
                if "seasonality" in item.get("disabled_overlays", [])
                else f"{item['seasonality']['expected']:+.1f}% "
                if item["seasonality"]["stable"] else "Not validated"
            ),
            "Source": item["source"],
        })
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame, hide_index=True, width="stretch",
        column_config={
            "Current": st.column_config.ProgressColumn(
                "Current", min_value=0, max_value=100, format="%.0f"
            ),
            "Forward": st.column_config.ProgressColumn(
                "Forward", min_value=0, max_value=100, format="%.0f"
            ),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=100, format="%.0f"
            ),
            "Probability%": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    st.caption(
        "Hidden a column? Hover over the table, click its eye icon, and "
        "enable the column again."
    )

    chart_frame = frame.copy()
    chart = px.scatter(
        chart_frame, x="Momentum", y="Acceleration", text="Sector",
        size="Confidence", color="Status", hover_data=["Forward", "Breadth"],
        title="Rotation map: relative momentum vs acceleration",
    )
    chart.add_vline(x=50, line_dash="dot")
    chart.add_hline(y=50, line_dash="dot")
    chart.update_traces(textposition="top center")
    chart.update_layout(
        template="plotly_dark", height=480,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,.6)",
    )
    st.plotly_chart(chart, width="stretch")

    selected_sector = st.selectbox(
        "Inspect sector", [item["sector"] for item in forecasts],
        key=f"sector_select_{horizon}",
    )
    selected = next(item for item in forecasts
                    if item["sector"] == selected_sector)
    st.markdown(
        f"#### {selected_sector}: {selected['status']} "
        f"({selected['forward_rotation']:.0f}/100 forward, "
        f"{selected['confidence']:.0f}/100 confidence)"
    )
    detail_columns = st.columns(6)
    detail_columns[0].metric("1M", f"{selected['returns']['1M']:+.1f}%")
    detail_columns[1].metric("3M", f"{selected['returns']['3M']:+.1f}%")
    detail_columns[2].metric("6M", f"{selected['returns']['6M']:+.1f}%")
    detail_columns[3].metric("Breadth", f"{selected['breadth']:.0f}")
    detail_columns[4].metric("Macro fit", f"{selected['macro_fit']:.0f}")
    detail_columns[5].metric("Data", selected["source"])
    evidence_col, risk_col = st.columns(2)
    with evidence_col:
        st.markdown("**Why it ranks here**")
        for reason in selected["reasons"]:
            st.markdown(f"- {reason}")
    with risk_col:
        st.markdown("**Risks / caveats**")
        for risk in selected["risks"]:
            st.markdown(f"- {risk}")

    sector_tiers = tiers.get(selected_sector, {})
    hero = sector_tiers.get("hero")
    st.markdown("#### Hero")
    if hero is None:
        st.info("No confirmed Hero yet. The strict trend, liquidity, setup, or confidence gates did not all pass.")
    else:
        st.success(
            f"{hero['name']} ({hero['ticker'].replace('.NS', '')}) — "
            f"leadership {hero['leadership_score']:.0f}/100"
        )
        _stock_plan(hero)
        _add_stock_button(hero, "sector_hero")

    contenders = sector_tiers.get("contenders", [])
    st.markdown(f"#### Contenders ({len(contenders)})")
    if not contenders:
        st.info("No additional stocks clear the contender gates.")
    for index, stock in enumerate(contenders, start=1):
        with st.expander(
            f"#{index} {stock['name']} · leadership "
            f"{stock['leadership_score']:.0f} · {stock.get('setup', 'NONE')}"
        ):
            _stock_plan(stock)
            _add_stock_button(stock, f"sector_contender_{index}")

    avoided = sector_tiers.get("avoid", [])
    st.markdown(f"#### Avoid / weak links ({len(avoided)})")
    if avoided:
        avoid_rows = [{
            "Stock": stock["name"],
            "Ticker": stock["ticker"].replace(".NS", ""),
            "Leadership": stock["leadership_score"],
            "Sector RS": stock["sector_rs_pct"],
            "Action": stock.get("action"),
            "Reason": " · ".join(stock["avoid_reasons"]),
        } for stock in avoided]
        st.dataframe(pd.DataFrame(avoid_rows), hide_index=True,
                     width="stretch")
    else:
        st.caption("No covered stocks currently trigger the strict avoid rules.")

    st.markdown("#### Walk-forward evidence")
    if not backtest:
        st.info("Enable Run Backtest and scan again to validate this horizon.")
        return
    if backtest.get("error"):
        st.warning(backtest["error"])
        return
    top3 = backtest["metrics"]["top3"]
    benchmark = backtest["metrics"]["benchmark"]
    bt_cols = st.columns(6)
    bt_cols[0].metric("Top-3 CAGR", f"{top3.get('cagr', 0):+.1f}%")
    bt_cols[1].metric("Alpha", f"{backtest['alpha_cagr']:+.1f}%")
    bt_cols[2].metric("Max DD", f"{top3.get('max_drawdown', 0):.1f}%")
    bt_cols[3].metric("Rank IC", f"{backtest['rank_ic']:.3f}")
    bt_cols[4].metric("Top-bottom spread", f"{backtest['spread']:+.2f}%")
    bt_cols[5].metric("Top-1 hit rate", f"{backtest['top1_hit_rate']:.1f}%")
    if backtest["rank_ic"] <= 0 or backtest["spread"] <= 0:
        st.error("This horizon has not shown reliable out-of-sample ranking power. Treat the live ordering as research only.")
    elif backtest["alpha_cagr"] <= 0:
        st.warning(
            "Ranking spread is positive, but the traded Top-3 strategy did not "
            "beat Nifty after costs. Use sector evidence as a stock-selection "
            "filter, not a standalone allocation strategy."
        )
    curves = backtest["curves"]
    figure = go.Figure()
    for name, color in (
        ("top3", "#00ff87"), ("benchmark", "#a855f7"),
        ("bottom3", "#ff4757"),
    ):
        points = curves[name]
        figure.add_trace(go.Scatter(
            x=[point["date"] for point in points],
            y=[point["value"] for point in points],
            name=name, line=dict(color=color),
        ))
    figure.update_layout(
        template="plotly_dark", height=360,
        title=f"Growth of ₹1: {backtest['period']['start']} to {backtest['period']['end']}",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,.6)",
    )
    st.plotly_chart(figure, width="stretch")
    ablation = backtest["ablations"]
    st.caption(
        f"Ablation rank IC — price/breadth: "
        f"{ablation['price_breadth_rank_ic']:.3f}; plus validated seasonality: "
        f"{ablation['plus_seasonality_rank_ic']:.3f}. "
        f"{ablation['macro_news']}"
    )
    with st.expander("Calibration, regime splits, and limitations"):
        st.markdown(f"**Brier score:** {backtest['brier']:.3f}")
        if backtest["calibration"]:
            st.dataframe(pd.DataFrame(backtest["calibration"]),
                         hide_index=True, width="stretch")
        if backtest["regime_stats"]:
            regime_rows = [
                {"Regime": name, **values}
                for name, values in backtest["regime_stats"].items()
            ]
            st.dataframe(pd.DataFrame(regime_rows), hide_index=True,
                         width="stretch")
        for limitation in backtest["limitations"]:
            st.caption(f"• {limitation}")
