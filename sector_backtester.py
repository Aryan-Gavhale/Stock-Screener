"""
Walk-forward sector rotation validation.

At every rebalance date all signals are truncated to that date. Current
fundamentals, current constituents' breadth, live news and macro overlays are
excluded from historical scores to avoid look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import COST_PARAMS, SECTOR_ROTATION
from risk_engine import equity_metrics
from sector_data import sector_close_panel
from sector_rotation import compute_sector_forecasts


def _regime_at(bench: pd.Series, idx: int) -> str:
    if idx < 200:
        return "Unknown"
    price = bench.iloc[idx]
    ma200 = bench.rolling(200).mean().iloc[idx]
    ma50 = bench.rolling(50).mean().iloc[idx]
    if price > ma200 and ma50 > ma200:
        return "Bull"
    if price < ma200:
        return "Risk-Off"
    return "Neutral"


def _curve_points(curve: pd.Series, limit=300) -> list[dict]:
    if len(curve) <= limit:
        indexes = np.arange(len(curve))
    else:
        indexes = np.linspace(0, len(curve) - 1, limit).astype(int)
    return [
        {"date": curve.index[i].strftime("%Y-%m-%d"),
         "value": round(float(curve.iloc[i]), 4)}
        for i in indexes
    ]


def _calibration(records: pd.DataFrame) -> tuple[list[dict], float]:
    if records.empty:
        return [], 0.0
    records = records.copy()
    records["outcome"] = (records["forward_excess"] > 0).astype(float)
    records["raw_probability"] = np.clip(
        0.50 + (records["score"] - 50) * 0.0065, 0.25, 0.75
    )
    brier = float(np.mean(
        (records["raw_probability"] - records["outcome"]) ** 2
    ))
    records["bin"] = pd.cut(
        records["score"], bins=[0, 40, 50, 60, 70, 100],
        labels=["0-40", "40-50", "50-60", "60-70", "70-100"],
        include_lowest=True,
    )
    rows = []
    for label, group in records.groupby("bin", observed=True):
        rows.append({
            "score_bin": str(label),
            "sample": int(len(group)),
            "hit_rate": round(float(group["outcome"].mean() * 100), 1),
            "avg_excess": round(float(group["forward_excess"].mean() * 100), 2),
        })
    return rows, round(brier, 3)


def calibrate_probabilities(
    forecasts: list[dict], calibration: list[dict]
) -> list[dict]:
    """Replace raw score mapping with historically observed bin hit rates."""
    if not calibration:
        return forecasts
    for forecast in forecasts:
        score = forecast["forward_rotation"]
        label = (
            "0-40" if score <= 40 else "40-50" if score <= 50
            else "50-60" if score <= 60 else "60-70" if score <= 70
            else "70-100"
        )
        row = next((item for item in calibration
                    if item["score_bin"] == label and item["sample"] >= 20), None)
        if row:
            forecast["probability"] = row["hit_rate"]
            forecast["probability_calibrated"] = True
        else:
            forecast["probability_calibrated"] = False
    return forecasts


def run_sector_backtest(
    sector_data: dict,
    bench_df: pd.DataFrame,
    horizon: str = "3M Swing",
    rebalance_days: int = 21,
) -> dict:
    """Monthly top-1/top-3 sector rotation vs Nifty and bottom-3."""
    panel = sector_close_panel(sector_data)
    if panel.empty or bench_df is None:
        return {"error": "Sector or benchmark history unavailable."}

    benchmark = bench_df["Close"].rename("Nifty")
    aligned = panel.join(benchmark, how="inner").dropna(how="all")
    sector_cols = [column for column in panel.columns if column in aligned]
    if len(sector_cols) < 5 or len(aligned) < 504:
        return {"error": "Need at least five sectors and two years of aligned history."}

    # Reindex each source once; compute_sector_forecasts slices by as-of date.
    aligned_sector_data = {}
    for sector in sector_cols:
        item = dict(sector_data[sector])
        item["price"] = item["price"].reindex(aligned.index).ffill(limit=3)
        aligned_sector_data[sector] = item
    aligned_bench = bench_df.reindex(aligned.index).ffill(limit=3)

    start = max(252, SECTOR_ROTATION["min_history_days"])
    forward_days = SECTOR_ROTATION["horizons"][horizon]["forward_days"]
    daily_returns = aligned[sector_cols].pct_change(fill_method=None)
    bench_returns = aligned["Nifty"].pct_change(fill_method=None)
    strategy_returns = pd.Series(0.0, index=aligned.index[start + 1:])
    top1_returns = pd.Series(0.0, index=aligned.index[start + 1:])
    bottom_returns = pd.Series(0.0, index=aligned.index[start + 1:])
    benchmark_returns = bench_returns.reindex(strategy_returns.index).fillna(0)

    records, selection_log = [], []
    previous_top = set()
    turnover_values = []
    regime_forward = {}

    for idx in range(start, len(aligned) - 1, rebalance_days):
        asof = aligned.index[idx]
        forecasts = compute_sector_forecasts(
            aligned_sector_data, aligned_bench, horizon, asof=asof,
            include_live_overlays=False,
        )
        if len(forecasts) < 5:
            continue
        top = [item["sector"] for item in forecasts[:3]]
        top1 = [forecasts[0]["sector"]]
        bottom = [item["sector"] for item in forecasts[-3:]]
        next_idx = min(idx + rebalance_days, len(aligned) - 1)
        period_dates = aligned.index[idx + 1:next_idx + 1]

        current_set = set(top)
        turnover = 1.0 if not previous_top else (
            len(current_set.symmetric_difference(previous_top))
            / max(1, len(current_set) * 2)
        )
        turnover_values.append(turnover)
        previous_top = current_set
        cost = COST_PARAMS["round_trip_cost"] * turnover

        top_period = daily_returns[top].mean(axis=1).reindex(period_dates).fillna(0)
        top1_period = daily_returns[top1].mean(axis=1).reindex(period_dates).fillna(0)
        bottom_period = daily_returns[bottom].mean(axis=1).reindex(period_dates).fillna(0)
        if len(period_dates):
            top_period.iloc[0] -= cost
            top1_period.iloc[0] -= cost
            bottom_period.iloc[0] -= cost
        strategy_returns.loc[period_dates] = top_period
        top1_returns.loc[period_dates] = top1_period
        bottom_returns.loc[period_dates] = bottom_period

        regime = _regime_at(aligned["Nifty"], idx)
        selection_log.append({
            "date": asof.strftime("%Y-%m-%d"), "top": top,
            "top1": top1[0], "bottom": bottom, "regime": regime,
            "turnover": round(turnover, 2),
        })

        fwd_idx = idx + forward_days
        if fwd_idx >= len(aligned):
            continue
        bench_fwd = aligned["Nifty"].iloc[fwd_idx] / aligned["Nifty"].iloc[idx] - 1
        score_values, forward_values = [], []
        forward_by_sector = {}
        for item in forecasts:
            sector = item["sector"]
            start_price = aligned[sector].iloc[idx]
            end_price = aligned[sector].iloc[fwd_idx]
            if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
                continue
            forward_return = end_price / start_price - 1
            forward_excess = forward_return - bench_fwd
            score_values.append(item["forward_rotation"])
            forward_values.append(forward_excess)
            forward_by_sector[sector] = forward_excess
            records.append({
                "date": asof, "sector": sector,
                "score": item["forward_rotation"],
                "seasonality_score": item["seasonality"]["score"],
                "forward_return": forward_return,
                "forward_excess": forward_excess,
                "regime": regime,
            })
        if score_values:
            ic = spearmanr(score_values, forward_values, nan_policy="omit").statistic
            top_values = [forward_by_sector[item["sector"]]
                          for item in forecasts[:3]
                          if item["sector"] in forward_by_sector]
            bottom_values = [forward_by_sector[item["sector"]]
                             for item in forecasts[-3:]
                             if item["sector"] in forward_by_sector]
            if not top_values or not bottom_values:
                continue
            top_excess = float(np.mean(top_values))
            bottom_excess = float(np.mean(bottom_values))
            regime_forward.setdefault(regime, []).append({
                "ic": 0.0 if np.isnan(ic) else float(ic),
                "spread": float(top_excess - bottom_excess),
                "top_hit": float(top_excess > 0),
            })

    if not selection_log:
        return {"error": "No valid walk-forward rebalance dates."}

    curves = {
        "top3": (1 + strategy_returns).cumprod(),
        "top1": (1 + top1_returns).cumprod(),
        "bottom3": (1 + bottom_returns).cumprod(),
        "benchmark": (1 + benchmark_returns).cumprod(),
    }
    metrics = {name: equity_metrics(curve) for name, curve in curves.items()}
    records_df = pd.DataFrame(records)

    date_stats = []
    if not records_df.empty:
        for date, group in records_df.groupby("date"):
            ordered = group.sort_values("score", ascending=False)
            top3 = ordered.head(3)["forward_excess"].mean()
            bottom3 = ordered.tail(3)["forward_excess"].mean()
            top1 = ordered.iloc[0]["forward_excess"]
            ic = spearmanr(
                ordered["score"], ordered["forward_excess"], nan_policy="omit"
            ).statistic
            season_score = ordered["score"] + (
                ordered["seasonality_score"] - 50
            ) * SECTOR_ROTATION["seasonality_weight_cap"] * 2
            season_ic = spearmanr(
                season_score, ordered["forward_excess"], nan_policy="omit"
            ).statistic
            date_stats.append({
                "top_hit": top1 > 0, "top3_hit": top3 > 0,
                "spread": top3 - bottom3,
                "ic": 0 if np.isnan(ic) else ic,
                "season_ic": 0 if np.isnan(season_ic) else season_ic,
            })
    stats_frame = pd.DataFrame(date_stats)
    calibration, brier = _calibration(records_df)
    regime_stats = {}
    for regime, values in regime_forward.items():
        frame = pd.DataFrame(values)
        regime_stats[regime] = {
            "sample": len(frame),
            "rank_ic": round(float(frame["ic"].mean()), 3),
            "spread": round(float(frame["spread"].mean() * 100), 2),
            "top_hit_rate": round(float(frame["top_hit"].mean() * 100), 1),
        }

    return {
        "horizon": horizon,
        "period": {
            "start": strategy_returns.index[0].strftime("%Y-%m-%d"),
            "end": strategy_returns.index[-1].strftime("%Y-%m-%d"),
            "years": round(len(strategy_returns) / 252, 1),
        },
        "metrics": metrics,
        "alpha_cagr": round(
            metrics["top3"].get("cagr", 0)
            - metrics["benchmark"].get("cagr", 0), 1
        ),
        "top1_hit_rate": round(
            float(stats_frame["top_hit"].mean() * 100), 1
        ) if not stats_frame.empty else 0,
        "top3_hit_rate": round(
            float(stats_frame["top3_hit"].mean() * 100), 1
        ) if not stats_frame.empty else 0,
        "rank_ic": round(float(stats_frame["ic"].mean()), 3)
        if not stats_frame.empty else 0,
        "spread": round(float(stats_frame["spread"].mean() * 100), 2)
        if not stats_frame.empty else 0,
        "turnover": round(float(np.mean(turnover_values) * 100), 1),
        "brier": brier,
        "calibration": calibration,
        "regime_stats": regime_stats,
        "ablations": {
            "price_breadth_rank_ic": round(float(stats_frame["ic"].mean()), 3)
            if not stats_frame.empty else 0,
            "plus_seasonality_rank_ic": round(
                float(stats_frame["season_ic"].mean()), 3
            ) if not stats_frame.empty else 0,
            "macro_news": "Not backtested: no point-in-time archive; live weight capped",
        },
        "curves": {name: _curve_points(curve) for name, curve in curves.items()},
        "selection_log": selection_log[-24:],
        "sample": len(records_df),
        "limitations": [
            "Historical ranking excludes current fundamentals, news and live macro overlays.",
            "Official price indices avoid constituent survivorship; synthetic baskets do not.",
            "Price-index returns exclude dividends.",
        ],
    }
