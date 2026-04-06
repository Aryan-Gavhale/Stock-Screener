"""
Backtesting engine.

Simulates the screener on historical data to prove (or disprove) effectiveness.
Uses walk-forward approach:
  1. At a past date, score all stocks using only data available then.
  2. Pick the top N stocks.
  3. Measure their actual forward returns.
  4. Compare to Nifty 50 benchmark.
"""

import pandas as pd
import numpy as np
from technical_analysis import technical_score, relative_strength
from fundamental_analysis import fundamental_score
from scoring_engine import composite
from config import RISK_PROFILES


def _slice_to_date(df: pd.DataFrame, end_idx: int) -> pd.DataFrame:
    """Return dataframe up to a given index position (simulating 'we only have data until here')."""
    if df is None or end_idx <= 30:
        return None
    return df.iloc[:end_idx].copy()


def run_backtest(
    all_data: dict,
    bench_df: pd.DataFrame,
    profile: str = "Moderate",
    lookback_days: int = 126,
    forward_days: int = 63,
    top_n: int = 10,
) -> dict:
    """
    Walk-forward backtest.

    Args:
        all_data: {ticker: {"price": DataFrame, "info": dict}}
        bench_df: Nifty 50 price DataFrame
        profile: risk profile name
        lookback_days: how many days back to simulate the screening point
        forward_days: how many days forward to measure returns
        top_n: pick top N stocks from screening

    Returns:
        dict with backtest results
    """
    if bench_df is None or len(bench_df) < lookback_days + forward_days + 50:
        return {"error": "Not enough benchmark data for backtest"}

    # The screening happens at `screen_idx` (end of lookback period)
    screen_idx = len(bench_df) - forward_days
    if screen_idx < 200:
        return {"error": "Not enough data"}

    # Slice benchmark
    bench_slice = bench_df.iloc[:screen_idx].copy()
    bench_future = bench_df.iloc[screen_idx:screen_idx + forward_days].copy()

    bench_ret = 0
    if len(bench_future) > 0:
        bench_ret = (bench_future["Close"].iloc[-1] / bench_future["Close"].iloc[0] - 1) * 100

    # Score each stock at the screening point
    scored = []
    for ticker, data in all_data.items():
        price = data.get("price")
        info = data.get("info")
        if price is None:
            continue

        # Find matching index — price data index must align
        # We need to slice the stock's price data to the same date window
        if len(price) < lookback_days + forward_days + 50:
            continue

        stock_screen_idx = len(price) - forward_days
        if stock_screen_idx < 100:
            continue

        stock_slice = price.iloc[:stock_screen_idx].copy()
        stock_future = price.iloc[stock_screen_idx:stock_screen_idx + forward_days].copy()

        if len(stock_slice) < 50 or len(stock_future) < 5:
            continue

        # Score using only historical data
        tech = technical_score(stock_slice, bench_slice)
        # Use ticker as fallback name
        name = ticker.replace(".NS", "")
        sector = "Unknown"
        fund = fundamental_score(info, sector)
        comp = composite(tech, fund, profile)

        # Actual forward return
        actual_ret = (stock_future["Close"].iloc[-1] / stock_future["Close"].iloc[0] - 1) * 100

        scored.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "composite": comp,
            "tech_score": tech["score"],
            "fund_score": fund["score"],
            "stage": tech.get("stage", (0, "N/A"))[1],
            "actual_return": round(actual_ret, 2),
        })

    if not scored:
        return {"error": "No stocks could be scored"}

    # Rank and pick top N
    scored.sort(key=lambda x: x["composite"], reverse=True)
    top_picks = scored[:top_n]
    bottom_picks = scored[-top_n:] if len(scored) > top_n else []

    # Portfolio return (equal weight top picks)
    top_returns = [s["actual_return"] for s in top_picks]
    portfolio_return = np.mean(top_returns) if top_returns else 0

    # Win rate
    winners = sum(1 for r in top_returns if r > 0)
    win_rate = (winners / len(top_returns) * 100) if top_returns else 0

    # Bottom picks return for comparison
    bottom_returns = [s["actual_return"] for s in bottom_picks]
    bottom_return = np.mean(bottom_returns) if bottom_returns else 0

    # Calculate max drawdown of top picks
    max_dd = min(top_returns) if top_returns else 0

    # Alpha
    alpha = portfolio_return - bench_ret

    # Sharpe-like ratio (simplified)
    if top_returns and np.std(top_returns) > 0:
        sharpe = portfolio_return / np.std(top_returns)
    else:
        sharpe = 0

    screen_date = bench_df.index[screen_idx].strftime("%Y-%m-%d") if hasattr(bench_df.index[screen_idx], "strftime") else str(screen_idx)
    end_date = bench_df.index[min(screen_idx + forward_days - 1, len(bench_df) - 1)].strftime("%Y-%m-%d") if hasattr(bench_df.index[0], "strftime") else "N/A"

    return {
        "screen_date": screen_date,
        "end_date": end_date,
        "forward_days": forward_days,
        "top_n": top_n,
        "portfolio_return": round(portfolio_return, 2),
        "benchmark_return": round(bench_ret, 2),
        "alpha": round(alpha, 2),
        "win_rate": round(win_rate, 1),
        "sharpe": round(sharpe, 2),
        "max_single_loss": round(max_dd, 2),
        "best_pick_return": round(max(top_returns), 2) if top_returns else 0,
        "worst_pick_return": round(min(top_returns), 2) if top_returns else 0,
        "bottom_picks_return": round(bottom_return, 2),
        "total_scored": len(scored),
        "top_picks": top_picks,
        "bottom_picks": bottom_picks[:5],
    }


def run_multi_period_backtest(all_data, bench_df, profile="Moderate", top_n=10):
    """
    Run backtests across multiple lookback periods to prove consistency.
    """
    results = []
    periods = [
        {"label": "3 months ago → now",   "lookback": 63,  "forward": 63},
        {"label": "6 months ago → 3m ago", "lookback": 126, "forward": 63},
        {"label": "6 months ago → now",    "lookback": 126, "forward": 126},
    ]

    for p in periods:
        r = run_backtest(
            all_data, bench_df, profile,
            lookback_days=p["lookback"],
            forward_days=p["forward"],
            top_n=top_n,
        )
        r["label"] = p["label"]
        results.append(r)

    # Summary stats across all periods
    valid = [r for r in results if "error" not in r]
    if valid:
        avg_return = np.mean([r["portfolio_return"] for r in valid])
        avg_alpha = np.mean([r["alpha"] for r in valid])
        avg_win = np.mean([r["win_rate"] for r in valid])
        consistency = sum(1 for r in valid if r["portfolio_return"] > r["benchmark_return"]) / len(valid) * 100
    else:
        avg_return = avg_alpha = avg_win = consistency = 0

    return {
        "periods": results,
        "summary": {
            "avg_return": round(avg_return, 2),
            "avg_alpha": round(avg_alpha, 2),
            "avg_win_rate": round(avg_win, 1),
            "beat_benchmark_pct": round(consistency, 0),
            "periods_tested": len(valid),
        },
    }
