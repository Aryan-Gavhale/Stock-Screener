"""
Risk engine — the module that keeps the portfolio green.

Provides:
  - Market regime detection (risk-on / neutral / risk-off)
  - ATR-based stop-loss, profit target, and trailing-stop levels
  - Volatility-based position sizing (equal risk per trade)
  - Portfolio heat (total open risk) management

Philosophy: every position risks a fixed small % of capital. Losers are
cut fast at the stop; winners are ridden via a ratcheting trailing stop.
This asymmetry is what allows a <50% win rate to still compound at 20%+.
"""

import numpy as np
import pandas as pd
from config import TRADE_PARAMS, REGIME, COST_PARAMS


# ═══════════════════════════════════════════════════════════════════════════════
#  Market regime
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regime(bench_df: pd.DataFrame, breadth_pct: float, crash: dict = None) -> dict:
    """
    Classify the overall market so we can throttle risk.
    Returns dict with regime label, deploy multiplier, and reasons.

    If a Crash Radar result (`crash`) is supplied, it OVERRIDES the base regime
    when crash risk is High/Extreme: deployment is cut hard, stops are tightened,
    and hedge / raise-cash guidance is attached (auto de-risk).
    """
    reasons = []
    above_ma = None
    if bench_df is not None and len(bench_df) >= REGIME["index_ma"]:
        ma = bench_df["Close"].rolling(REGIME["index_ma"]).mean().iloc[-1]
        price = bench_df["Close"].iloc[-1]
        above_ma = bool(price > ma)
        pct = (price / ma - 1) * 100
        reasons.append(f"Nifty {'above' if above_ma else 'below'} {REGIME['index_ma']}-DMA ({pct:+.1f}%)")

        # slope of 200DMA
        if len(bench_df) >= REGIME["index_ma"] + 20:
            ma_prev = bench_df["Close"].rolling(REGIME["index_ma"]).mean().iloc[-20]
            ma_rising = ma > ma_prev
            reasons.append(f"200-DMA {'rising' if ma_rising else 'falling'}")
        else:
            ma_rising = True
    else:
        ma_rising = True
        reasons.append("Insufficient index history for 200-DMA")

    reasons.append(f"Breadth {breadth_pct:.0f}% above 50-DMA")

    # Base decision logic
    bull_breadth = breadth_pct >= REGIME["breadth_bull"]
    bear_breadth = breadth_pct <= REGIME["breadth_bear"]

    below_ma = (above_ma is not None) and (not above_ma)
    if above_ma and ma_rising and not bear_breadth:
        label, deploy, color = "RISK-ON", REGIME["deploy_risk_on"], "green"
        stance = "Full deployment. Take quality breakouts and pullbacks."
    elif below_ma or bear_breadth:
        label, deploy, color = "RISK-OFF", REGIME["deploy_risk_off"], "red"
        stance = "Raise cash. Only A+ setups, half size, tighter stops."
    else:
        label, deploy, color = "NEUTRAL", REGIME["deploy_neutral"], "amber"
        stance = "Selective. Reduce new-position size, favor leaders."

    stop_tighten = 1.0
    hedge = None
    crash_index = None
    if crash:
        crash_index = crash.get("index")
        clabel = crash.get("label")
        reasons.append(f"Crash Radar: {clabel} ({crash_index:.0f}/100)")
        # Auto de-risk: crash risk caps deployment and tightens stops
        deploy = min(deploy, crash.get("deploy", 1.0))
        stop_tighten = crash.get("stop_tighten", 1.0)
        if clabel == "EXTREME":
            label, color = "CRASH-RISK EXTREME", "red"
            stance = crash.get("stance", stance)
            hedge = "Move to cash / consider an index put or reducing beta. Do not open new longs."
        elif clabel == "HIGH":
            label, color = "CRASH-RISK HIGH", "red"
            stance = crash.get("stance", stance)
            hedge = "Trim weakest holdings, raise cash to ~50-70%, tighten stops. Only A+ setups at half size."
        elif clabel == "ELEVATED" and label == "RISK-ON":
            label, color = "CAUTION", "amber"
            stance = crash.get("stance", stance)

    return {
        "label": label, "deploy": round(deploy, 2), "color": color,
        "stance": stance, "reasons": reasons,
        "breadth": round(breadth_pct, 1),
        "stop_tighten": stop_tighten,
        "crash_index": crash_index,
        "hedge": hedge,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Stop / target / trailing levels
# ═══════════════════════════════════════════════════════════════════════════════

def trade_levels(price: float, atr: float, profile_params: dict) -> dict:
    """
    Compute entry, initial stop, target, and R-multiple for a fresh entry.
    Stop is the WIDER of (ATR-based) and (profile % based) so we don't get
    shaken out by normal volatility, but never wider than a hard cap.
    """
    if price is None or price <= 0:
        return {}

    atr = atr if (atr and atr > 0) else price * 0.02
    atr_stop_dist = TRADE_PARAMS["atr_stop_mult"] * atr
    pct_stop_dist = profile_params.get("core_stop_pct", 0.09) * price

    # Use the tighter of the two so risk stays controlled, but floor at 1.5*ATR
    stop_dist = max(min(atr_stop_dist, pct_stop_dist), 1.5 * atr)
    stop_dist = min(stop_dist, 0.15 * price)  # hard cap 15%

    stop = price - stop_dist
    risk_per_share = price - stop
    target = price + TRADE_PARAMS["profit_target_r"] * risk_per_share

    return {
        "entry": round(price, 2),
        "stop": round(stop, 2),
        "stop_pct": round(stop_dist / price * 100, 1),
        "target": round(target, 2),
        "target_pct": round((target / price - 1) * 100, 1),
        "risk_per_share": round(risk_per_share, 2),
        "rr": TRADE_PARAMS["profit_target_r"],
    }


def trailing_stop(entry: float, current: float, high_since_entry: float,
                  atr: float) -> dict:
    """
    Ratcheting trailing stop. Only activates after +trail_trigger_gain.
    Returns the trailing stop level and whether it's active.
    """
    atr = atr if (atr and atr > 0) else current * 0.02
    gain = (current / entry - 1) if entry > 0 else 0
    if gain < TRADE_PARAMS["trail_trigger_gain"]:
        return {"active": False, "trail_stop": None,
                "note": f"Trail activates at +{TRADE_PARAMS['trail_trigger_gain']*100:.0f}% (now {gain*100:+.1f}%)"}

    trail = high_since_entry - TRADE_PARAMS["trail_atr_mult"] * atr
    # Never let trailing stop go below breakeven once we're up big
    trail = max(trail, entry)
    return {"active": True, "trail_stop": round(trail, 2),
            "note": f"Trailing {TRADE_PARAMS['trail_atr_mult']}x ATR below peak ₹{high_since_entry:.0f}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Position sizing (equal-risk / volatility-based)
# ═══════════════════════════════════════════════════════════════════════════════

def position_size(capital: float, price: float, stop: float,
                  profile_params: dict, deploy_mult: float = 1.0) -> dict:
    """
    Size the position so that hitting the stop loses exactly
    `risk_per_trade` (× regime deploy multiplier) of capital.

    This is the core of capital preservation: a ₹20 stock with a ₹2 stop
    and a ₹200 stock with a ₹20 stop risk the SAME rupees.
    """
    if price is None or price <= 0 or stop is None or stop >= price:
        return {"qty": 0, "amount": 0, "weight_pct": 0, "risk_amount": 0}

    risk_frac = profile_params.get("risk_per_trade", 0.01) * deploy_mult
    risk_budget = capital * risk_frac
    risk_per_share = price - stop

    qty_by_risk = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0

    # Cap by max position weight
    max_amt = capital * profile_params.get("max_per_stock", 0.10)
    qty_by_cap = int(max_amt / price) if price > 0 else 0

    qty = max(0, min(qty_by_risk, qty_by_cap))
    amount = qty * price
    actual_risk = qty * risk_per_share

    return {
        "qty": qty,
        "amount": round(amount),
        "weight_pct": round(amount / capital * 100, 1) if capital else 0,
        "risk_amount": round(actual_risk),
        "risk_pct": round(actual_risk / capital * 100, 2) if capital else 0,
        "capped_by": "max_weight" if qty_by_cap < qty_by_risk else "risk_budget",
    }


def portfolio_heat(positions: list, capital: float) -> dict:
    """
    Total open risk across all positions. If a stock is above its stop,
    risk = (entry-stop)*qty; if trailing above entry, risk is locked-in gain
    (negative risk / house money).
    """
    total_risk = 0
    for p in positions:
        entry = p.get("entry", 0)
        stop = p.get("current_stop", p.get("stop", 0))
        qty = p.get("qty", 0)
        total_risk += (entry - stop) * qty
    heat_pct = total_risk / capital * 100 if capital else 0
    return {"open_risk": round(total_risk), "heat_pct": round(heat_pct, 2)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Performance metrics (used by backtester)
# ═══════════════════════════════════════════════════════════════════════════════

def equity_metrics(equity_curve: pd.Series, trades: list = None,
                   periods_per_year: int = 252) -> dict:
    """
    Compute CAGR, max drawdown, Sharpe, Sortino from an equity curve.
    trades: optional list of trade P&L % for win-rate / profit-factor.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return {}

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_days = len(equity_curve)
    years = n_days / periods_per_year
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    # Max drawdown
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1
    max_dd = dd.min()

    # Daily returns
    rets = equity_curve.pct_change().dropna()
    rf_daily = COST_PARAMS["risk_free_rate"] / periods_per_year
    excess = rets - rf_daily
    sharpe = (excess.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else 0
    downside = rets[rets < 0]
    sortino = (excess.mean() / downside.std() * np.sqrt(periods_per_year)) if len(downside) > 1 and downside.std() > 0 else 0

    out = {
        "total_return": round(total_return * 100, 1),
        "cagr": round(cagr * 100, 1),
        "max_drawdown": round(max_dd * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(cagr / abs(max_dd), 2) if max_dd < 0 else 0,
    }

    if trades:
        pnls = [t for t in trades if t is not None]
        if pnls:
            wins = [t for t in pnls if t > 0]
            losses = [t for t in pnls if t <= 0]
            out["num_trades"] = len(pnls)
            out["win_rate"] = round(len(wins) / len(pnls) * 100, 1)
            out["avg_win"] = round(np.mean(wins), 1) if wins else 0
            out["avg_loss"] = round(np.mean(losses), 1) if losses else 0
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            out["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999
            out["expectancy"] = round(np.mean(pnls), 2)
    return out
