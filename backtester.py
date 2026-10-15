"""
Backtester (v5) — HONEST, look-ahead-free validation.

Fixes the fatal flaw of the old version (which used TODAY's fundamentals in
the past). Here the historical signal is built ONLY from price data available
up to each point in time, so results are legitimate.

What it does:
  1. Rolling monthly rebalance into the top-N momentum leaders.
  2. Uptrend-only + market-regime filter (go to cash when Nifty < 200-DMA).
  3. Two-stage stop: initial hard stop + ratcheting trailing stop (ride winners).
  4. Realistic round-trip costs on every trade.
  5. Real metrics: CAGR, max drawdown, Sharpe, Sortino, win rate, avg win/loss,
     profit factor, expectancy, exposure — vs a Nifty buy-and-hold benchmark.
  6. A decile test: proves whether high-signal stocks actually beat low-signal
     ones on forward returns (signal monotonicity). This is the real proof.
"""

import numpy as np
import pandas as pd
from config import RISK_PROFILES, COST_PARAMS, REGIME
from risk_engine import equity_metrics

REBAL_DAYS = 21          # ~monthly
TRAIL_PCT = 0.15         # trailing stop distance once in profit
MIN_HISTORY = 210        # need ~200-DMA before trading


# ── Point-in-time momentum signal (price-only → no look-ahead) ───────────────────

def _signal_at(close_row_idx, closes, sma50, sma200):
    """
    closes/sma arrays are 1D numpy for a single stock.
    Returns (score, eligible). Score = blended momentum; eligible = in uptrend.
    """
    i = close_row_idx
    c = closes[i]
    if np.isnan(c) or c <= 0:
        return None, False

    def ret(period):
        j = i - period
        if j < 0 or np.isnan(closes[j]) or closes[j] <= 0:
            return None
        return closes[i] / closes[j] - 1

    r63, r126 = ret(63), ret(126)
    r12_1 = None
    if i >= 252 and not np.isnan(closes[i - 21]) and not np.isnan(closes[i - 252]) and closes[i - 252] > 0:
        r12_1 = closes[i - 21] / closes[i - 252] - 1

    parts, w = 0.0, 0.0
    if r126 is not None: parts += r126 * 0.40; w += 0.40
    if r63 is not None:  parts += r63 * 0.30; w += 0.30
    if r12_1 is not None: parts += r12_1 * 0.30; w += 0.30
    if w == 0:
        return None, False
    score = parts / w

    elig = True
    if not np.isnan(sma200[i]) and sma200[i] > 0:
        elig = elig and c > sma200[i]
    if not np.isnan(sma50[i]) and sma50[i] > 0:
        elig = elig and c > sma50[i]
    return score, elig


# ── Main backtest ────────────────────────────────────────────────────────────────

def run_backtest(all_data: dict, bench_df: pd.DataFrame,
                 profile: str = "Moderate", use_regime: bool = True) -> dict:
    p = RISK_PROFILES.get(profile, RISK_PROFILES["Moderate"])
    top_n = p["max_stocks"]
    init_stop = p["core_stop_pct"]
    cost = COST_PARAMS["round_trip_cost"]

    if bench_df is None or len(bench_df) < MIN_HISTORY + REBAL_DAYS:
        return {"error": "Not enough history. Use a 3Y or 5Y data period for a meaningful backtest."}

    # Align all stocks to benchmark dates (build once via concat to avoid fragmentation)
    dates = bench_df.index
    close_d, high_d, low_d = {}, {}, {}
    for t, d in all_data.items():
        px = d.get("price")
        if px is None or len(px) < MIN_HISTORY:
            continue
        close_d[t] = px["Close"].reindex(dates).ffill(limit=5)
        high_d[t] = px["High"].reindex(dates).ffill(limit=5)
        low_d[t] = px["Low"].reindex(dates).ffill(limit=5)
    close_m = pd.DataFrame(close_d, index=dates)
    high_m = pd.DataFrame(high_d, index=dates)
    low_m = pd.DataFrame(low_d, index=dates)

    tickers = list(close_m.columns)
    if len(tickers) < top_n:
        return {"error": f"Only {len(tickers)} stocks have enough history for a backtest (need ≥{top_n})."}

    # Precompute arrays
    closes = {t: close_m[t].to_numpy() for t in tickers}
    highs = {t: high_m[t].to_numpy() for t in tickers}
    lows = {t: low_m[t].to_numpy() for t in tickers}
    sma50 = {t: close_m[t].rolling(50).mean().to_numpy() for t in tickers}
    sma200 = {t: close_m[t].rolling(200).mean().to_numpy() for t in tickers}

    bench_close = bench_df["Close"].to_numpy()
    bench_sma200 = bench_df["Close"].rolling(REGIME["index_ma"]).mean().to_numpy()

    n = len(dates)
    start = MIN_HISTORY

    # Portfolio state (normalized capital = 1.0, fractional shares)
    cash = 1.0
    holds = {}  # ticker -> {shares, entry, peak}
    equity_curve = []
    trades = []          # closed-trade P&L %
    days_in_cash = 0
    invested_days = 0

    def regime_ok(i):
        if not use_regime:
            return True
        if np.isnan(bench_sma200[i]):
            return True
        return bench_close[i] > bench_sma200[i]

    def port_value(i):
        val = cash
        for t, h in holds.items():
            px = closes[t][i]
            if not np.isnan(px):
                val += h["shares"] * px
        return val

    for i in range(start, n):
        # 1) Update peaks & check stops (intraday low vs stop)
        for t in list(holds.keys()):
            h = holds[t]
            hi = highs[t][i]
            lo = lows[t][i]
            c = closes[t][i]
            if np.isnan(c):
                continue
            if not np.isnan(hi):
                h["peak"] = max(h["peak"], hi)
            stop_level = max(h["entry"] * (1 - init_stop), h["peak"] * (1 - TRAIL_PCT))
            if not np.isnan(lo) and lo <= stop_level:
                # exit at stop (assume stop order fills at stop_level)
                proceeds = h["shares"] * stop_level * (1 - cost / 2)
                cash += proceeds
                pnl = (stop_level / h["entry"] - 1) * 100
                trades.append(pnl)
                del holds[t]

        # 2) Rebalance
        if (i - start) % REBAL_DAYS == 0:
            if not regime_ok(i):
                # risk-off → go fully to cash
                for t in list(holds.keys()):
                    c = closes[t][i]
                    if np.isnan(c):
                        continue
                    cash += holds[t]["shares"] * c * (1 - cost / 2)
                    trades.append((c / holds[t]["entry"] - 1) * 100)
                    del holds[t]
            else:
                # rank eligible stocks by signal
                ranked = []
                for t in tickers:
                    sc, elig = _signal_at(i, closes[t], sma50[t], sma200[t])
                    if sc is not None and elig:
                        ranked.append((sc, t))
                ranked.sort(reverse=True)
                target = set(t for _, t in ranked[:top_n])

                # sell holdings not in target
                for t in list(holds.keys()):
                    if t not in target:
                        c = closes[t][i]
                        if np.isnan(c):
                            continue
                        cash += holds[t]["shares"] * c * (1 - cost / 2)
                        trades.append((c / holds[t]["entry"] - 1) * 100)
                        del holds[t]

                # buy new entrants — equal weight to target size
                buys = [t for t in target if t not in holds]
                if buys:
                    total_val = port_value(i)
                    target_per = total_val / top_n
                    for t in buys:
                        c = closes[t][i]
                        if np.isnan(c) or c <= 0:
                            continue
                        alloc = min(target_per, cash)
                        if alloc <= 0:
                            break
                        shares = (alloc * (1 - cost / 2)) / c
                        cash -= alloc
                        holds[t] = {"shares": shares, "entry": c, "peak": c}

        # 3) Mark to market
        equity_curve.append(port_value(i))
        if holds:
            invested_days += 1
        else:
            days_in_cash += 1

    eq = pd.Series(equity_curve, index=dates[start:n])
    strat = equity_metrics(eq, trades)
    strat["exposure"] = round(invested_days / max(1, (n - start)) * 100, 1)
    strat["days_in_cash_pct"] = round(days_in_cash / max(1, (n - start)) * 100, 1)

    # Benchmark buy & hold
    bench_eq = pd.Series(bench_close[start:n] / bench_close[start], index=dates[start:n])
    bench_m = equity_metrics(bench_eq)

    # Decile validation (does signal predict forward returns?)
    decile = _decile_validation(tickers, closes, sma50, sma200, start, n)

    # Downsample equity curves for charting
    def _sample(series, k=300):
        if len(series) <= k:
            idx = range(len(series))
        else:
            idx = np.linspace(0, len(series) - 1, k).astype(int)
        return [{"date": series.index[j].strftime("%Y-%m-%d"),
                 "value": round(float(series.iloc[j]), 4)} for j in idx]

    years = (n - start) / 252
    return {
        "strategy": strat,
        "benchmark": bench_m,
        "n_trades": len(trades),
        "decile": decile,
        "equity": {"strategy": _sample(eq / eq.iloc[0]),
                   "benchmark": _sample(bench_eq)},
        "period": {
            "start": dates[start].strftime("%Y-%m-%d"),
            "end": dates[n - 1].strftime("%Y-%m-%d"),
            "years": round(years, 1),
        },
        "profile": profile,
        "alpha_cagr": round(strat.get("cagr", 0) - bench_m.get("cagr", 0), 1),
    }


def _decile_validation(tickers, closes, sma50, sma200, start, n, fwd=REBAL_DAYS):
    """
    Across many past dates, bucket stocks by signal into deciles and measure
    average forward return. If the strategy has edge, decile 10 (highest signal)
    should beat decile 1 — ideally monotonically.
    """
    rows = []
    step = REBAL_DAYS
    for i in range(start, n - fwd, step):
        for t in tickers:
            sc, _ = _signal_at(i, closes[t], sma50[t], sma200[t])
            if sc is None:
                continue
            c0, c1 = closes[t][i], closes[t][i + fwd]
            if np.isnan(c0) or np.isnan(c1) or c0 <= 0:
                continue
            rows.append((sc, (c1 / c0 - 1) * 100))
    if len(rows) < 100:
        return {"available": False}

    df = pd.DataFrame(rows, columns=["signal", "fwd"])
    df["decile"] = pd.qcut(df["signal"], 10, labels=False, duplicates="drop")
    grp = df.groupby("decile")["fwd"].mean()
    deciles = [round(float(v), 2) for v in grp.values]
    top, bottom = deciles[-1], deciles[0]
    # monotonic-ish check: top decile beats bottom and top-3 avg beats bottom-3 avg
    mono = (np.mean(deciles[-3:]) > np.mean(deciles[:3])) and (top > bottom)
    return {
        "available": True,
        "deciles": deciles,
        "top_decile_fwd": round(top, 2),
        "bottom_decile_fwd": round(bottom, 2),
        "spread": round(top - bottom, 2),
        "monotonic": bool(mono),
        "sample": len(df),
        "fwd_days": fwd,
    }
