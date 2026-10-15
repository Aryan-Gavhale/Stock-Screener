"""
Market Risk module — Crash Radar + Capitulation/Bottom detector.

IMPORTANT (honesty): a crash cannot be predicted with certainty. crash_risk()
is a PROBABILISTIC gauge that reads conditions which have historically preceded
sharp falls (weak trend, breadth breakdown, distribution days, momentum
divergence, VIX spikes, global risk-off, risk news). It will have false alarms
and misses. accumulation_zone() likewise detects capitulation + early
stabilization to time bottom-fishing — it reduces, not eliminates, the risk of
catching a falling knife.

Both return a 0-100 style signal plus the specific reasons firing, so you can
judge the drivers yourself.
"""

import numpy as np
import pandas as pd
import ta

from config import CRASH, ACCUMULATION


# ── small helpers ────────────────────────────────────────────────────────────────

def _rsi(close, n=14):
    try:
        return ta.momentum.RSIIndicator(close, n).rsi()
    except Exception:
        return pd.Series(np.nan, index=close.index)

def _macd_hist(close):
    try:
        m = ta.trend.MACD(close, 26, 12, 9)
        return m.macd(), m.macd_signal(), m.macd_diff()
    except Exception:
        s = pd.Series(np.nan, index=close.index)
        return s, s, s

def _pct_change_over(series, days):
    s = series.dropna()
    if len(s) <= days:
        return None
    return (s.iloc[-1] / s.iloc[-days] - 1) * 100

def _clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))


# ── distribution days (institutional selling) ────────────────────────────────────

def _distribution_days(df, lookback):
    """Count high-volume down days in the index over the lookback window.
    Falls back to price-only if index volume is unreliable (common for ^NSEI)."""
    d = df.tail(lookback + 25).copy()
    if len(d) < 10:
        return 0
    d["ret"] = d["Close"].pct_change()
    vol = d["Volume"].fillna(0)
    use_vol = vol.tail(lookback).sum() > 0
    count = 0
    window = d.tail(lookback)
    if use_vol:
        vsma = d["Volume"].rolling(20).mean()
        for idx in window.index:
            if d.loc[idx, "ret"] < -0.003 and d.loc[idx, "Volume"] >= (vsma.loc[idx] or 0):
                count += 1
    else:
        for idx in window.index:
            if d.loc[idx, "ret"] < -0.007:
                count += 1
    return count


# ── bearish momentum divergence ──────────────────────────────────────────────────

def _divergence(df, lookback=40):
    d = df.tail(lookback).copy()
    if len(d) < 20:
        return False
    rsi = _rsi(d["Close"]).dropna()
    if len(rsi) < 16:
        return False
    half = len(d) // 2
    price_recent, price_prior = d["Close"].iloc[half:], d["Close"].iloc[:half]
    rsi_recent = rsi.iloc[-half:] if len(rsi) >= half else rsi
    rsi_prior = rsi.iloc[:-half] if len(rsi) >= half else rsi
    if len(rsi_prior) == 0 or len(rsi_recent) == 0:
        return False
    price_hh = price_recent.max() > price_prior.max()
    rsi_lh = rsi_recent.max() < rsi_prior.max()
    return bool(price_hh and rsi_lh)


# ═══════════════════════════════════════════════════════════════════════════════
#  CRASH RISK INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def crash_risk(nifty_df, all_data=None, context=None, news=None, breadth_ser=None) -> dict:
    context = context or {}
    news = news or {}
    comp = {}      # component sub-scores 0-100
    signals = []   # human-readable firing reasons
    stats = {}

    # 1) Trend
    trend = 0
    if nifty_df is not None and len(nifty_df) >= 60:
        c = nifty_df["Close"]
        price = c.iloc[-1]
        s50 = c.rolling(50).mean().iloc[-1]
        s200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
        below50 = price < s50
        below200 = (not np.isnan(s200)) and price < s200
        death_cross = (not np.isnan(s200)) and s50 < s200
        if below200: trend += 42; signals.append("Nifty below 200-DMA")
        if below50:  trend += 20; signals.append("Nifty below 50-DMA")
        if death_cross: trend += 26; signals.append("50-DMA below 200-DMA (death cross)")
        if not np.isnan(s200):
            ext = (price / s200 - 1) * 100
            stats["above_200dma_pct"] = round(ext, 1)
            if ext > CRASH["index_extended_above_200"]:
                trend += 15; signals.append(f"Index stretched +{ext:.0f}% above 200-DMA (froth)")
    comp["trend"] = _clamp(trend)

    # 2) Breadth
    breadth = 20
    latest_breadth = None
    if breadth_ser is not None and len(breadth_ser) > 0:
        latest_breadth = float(breadth_ser.iloc[-1])
        stats["breadth"] = round(latest_breadth, 1)
        if latest_breadth >= 60:   breadth = 10
        elif latest_breadth >= 50: breadth = 28
        elif latest_breadth >= 40: breadth = 48
        elif latest_breadth >= 30: breadth = 66
        elif latest_breadth >= 20: breadth = 82
        else:                      breadth = 95
        if latest_breadth < CRASH["breadth_weak"]:
            signals.append(f"Weak breadth — only {latest_breadth:.0f}% above 50-DMA")
        # deterioration
        if len(breadth_ser) > 20:
            prev = float(breadth_ser.iloc[-20])
            if latest_breadth < prev - 12:
                breadth = _clamp(breadth + 8)
                signals.append("Breadth deteriorating fast (last month)")
    comp["breadth"] = _clamp(breadth)

    # 3) Distribution days
    dist = 0
    if nifty_df is not None:
        dd = _distribution_days(nifty_df, CRASH["distribution_lookback"])
        stats["distribution_days"] = dd
        dist = _clamp(dd / (CRASH["distribution_days_warn"] * 2) * 100)
        if dd >= CRASH["distribution_days_warn"]:
            signals.append(f"{dd} distribution days in {CRASH['distribution_lookback']} sessions (institutional selling)")
    comp["distribution"] = dist

    # 4) Divergence
    diverg = 0
    if nifty_df is not None and _divergence(nifty_df):
        diverg = 62; signals.append("Bearish momentum divergence (price up, RSI down)")
    comp["divergence"] = diverg

    # 5) India VIX
    vix = 15
    ivix = context.get("india_vix")
    if ivix is not None and len(ivix) > 5:
        level = ivix["Close"].iloc[-1]
        stats["india_vix"] = round(float(level), 1)
        if level < CRASH["india_vix_elevated"]:  vix = 12
        elif level < CRASH["india_vix_high"]:     vix = 40
        elif level < CRASH["india_vix_extreme"]:  vix = 68
        else:                                     vix = 92
        spike = _pct_change_over(ivix["Close"], 5)
        if spike is not None and spike > CRASH["vix_spike_pct"]:
            vix = _clamp(vix + 20)
            signals.append(f"India VIX spiking (+{spike:.0f}% in a week → {level:.0f})")
        elif level >= CRASH["india_vix_high"]:
            signals.append(f"India VIX elevated ({level:.0f})")
    comp["vix"] = _clamp(vix)

    # 6) Global risk-off
    glob = 0
    sp = context.get("sp500")
    if sp is not None and len(sp) >= 200:
        if sp["Close"].iloc[-1] < sp["Close"].rolling(200).mean().iloc[-1]:
            glob += 42; signals.append("S&P 500 below its 200-DMA (global risk-off)")
    usv = context.get("us_vix")
    if usv is not None and len(usv) > 1:
        if usv["Close"].iloc[-1] > CRASH["us_vix_high"]:
            glob += 28; signals.append(f"US VIX high ({usv['Close'].iloc[-1]:.0f})")
    inr = context.get("usdinr")
    if inr is not None:
        chg = _pct_change_over(inr["Close"], 21)
        if chg is not None:
            stats["usdinr_1m"] = round(chg, 2)
            if chg > CRASH["usdinr_spike_pct"]:
                glob += 16; signals.append(f"Rupee weakening (+{chg:.1f}% USD/INR in a month → FII outflow risk)")
    crude = context.get("crude")
    if crude is not None:
        chg = _pct_change_over(crude["Close"], 21)
        if chg is not None:
            stats["crude_1m"] = round(chg, 1)
            if chg > CRASH["crude_spike_pct"]:
                glob += 14; signals.append(f"Crude spiking (+{chg:.0f}% in a month → inflation risk)")
    comp["global"] = _clamp(glob)

    # 7) News
    news_score = float(news.get("score", 0)) if news else 0
    comp["news"] = _clamp(news_score)
    if news_score >= 40 and news.get("matched"):
        signals.append("Risk-heavy headlines: " + ", ".join(news["matched"][:5]))

    # Blend
    w = CRASH["weights"]
    index = sum(comp[k] * w.get(k, 0) for k in comp)
    index = round(_clamp(index), 1)

    b = CRASH["bands"]
    if index >= b["extreme"]:
        label, color = "EXTREME", "red"
        stance = "Sirens on. Move to cash / hedge. No new longs — protect capital."
        deploy, stop_tighten = 0.0, 0.6
    elif index >= b["high"]:
        label, color = "HIGH", "red"
        stance = "De-risk hard. Trim weak holdings, raise cash, tighten stops. Only A+ setups at half size."
        deploy, stop_tighten = 0.2, 0.7
    elif index >= b["elevated"]:
        label, color = "ELEVATED", "amber"
        stance = "Caution. Reduce new-position size, favor leaders, keep stops honest."
        deploy, stop_tighten = 0.6, 0.85
    else:
        label, color = "LOW", "green"
        stance = "Skies clear. Normal deployment on valid setups."
        deploy, stop_tighten = 1.0, 1.0

    if not signals:
        signals.append("No major crash signals firing")

    return {
        "index": index, "label": label, "color": color, "stance": stance,
        "deploy": deploy, "stop_tighten": stop_tighten,
        "components": {k: round(v, 0) for k, v in comp.items()},
        "signals": signals, "stats": stats,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CAPITULATION / BOTTOM (accumulation zone)
# ═══════════════════════════════════════════════════════════════════════════════

def accumulation_zone(nifty_df, all_data=None, context=None, breadth_ser=None) -> dict:
    context = context or {}
    reasons = []
    sig = {}

    if nifty_df is None or len(nifty_df) < 60:
        return {"status": "No", "drawdown": 0, "reasons": ["Insufficient index data"], "signals": {}}

    c = nifty_df["Close"]
    high_6m = c.tail(126).max()
    price = c.iloc[-1]
    drawdown = (price / high_6m - 1) * 100
    armed = drawdown <= -ACCUMULATION["drawdown_trigger"]

    # RSI oversold and turning up
    rsi = _rsi(c).dropna()
    rsi_turn = False
    if len(rsi) >= 6:
        recent_min = rsi.tail(15).min()
        rsi_turn = recent_min < ACCUMULATION["index_rsi_oversold"] and rsi.iloc[-1] > rsi.iloc[-3]
    sig["rsi_oversold_turning_up"] = rsi_turn
    if rsi_turn: reasons.append(f"Index RSI washed out ({recent_min:.0f}) and turning up")

    # VIX spike then rollover
    vix_roll = False
    ivix = context.get("india_vix")
    if ivix is not None and len(ivix) > 10:
        v = ivix["Close"]
        vmax10 = v.tail(10).max()
        vix_roll = vmax10 >= CRASH["india_vix_high"] and v.iloc[-1] < vmax10 * 0.85
    sig["vix_rollover"] = vix_roll
    if vix_roll: reasons.append("VIX spiked then rolled over (fear peaking)")

    # Breadth washout then improving
    breadth_turn = False
    if breadth_ser is not None and len(breadth_ser) > 5:
        bmin = breadth_ser.tail(20).min()
        breadth_turn = bmin < ACCUMULATION["breadth_washout"] and breadth_ser.iloc[-1] > bmin + 5
    sig["breadth_washout_turn"] = breadth_turn
    if breadth_turn: reasons.append("Breadth washed out (<20%) and starting to recover")

    # MACD bullish cross from below zero
    macd_cross = False
    macd_line, macd_sig, macd_h = _macd_hist(c)
    mh = macd_h.dropna()
    if len(mh) >= 4 and macd_line.dropna().iloc[-1] < 0:
        macd_cross = mh.iloc[-1] > 0 and mh.iloc[-4] < 0
    sig["macd_cross_up"] = macd_cross
    if macd_cross: reasons.append("MACD crossing up from oversold territory")

    # Follow-through day (O'Neil): strong up day on higher volume, a few days after the low
    ft = False
    d = nifty_df.tail(60).copy()
    if len(d) >= 15:
        low_idx = d["Close"].idxmin()
        low_pos = d.index.get_loc(low_idx)
        vsma = d["Volume"].rolling(20).mean()
        use_vol = d["Volume"].tail(30).sum() > 0
        rets = d["Close"].pct_change() * 100
        for pos in range(low_pos + 4, len(d)):
            idx = d.index[pos]
            gain_ok = rets.loc[idx] >= ACCUMULATION["follow_through_gain"]
            vol_ok = (d.loc[idx, "Volume"] >= (vsma.loc[idx] or 0)) if use_vol else True
            if gain_ok and vol_ok:
                ft = True; break
    sig["follow_through_day"] = ft
    if ft: reasons.append("Follow-through day printed (institutional buying returning)")

    bull_count = sum([rsi_turn, vix_roll, breadth_turn, macd_cross, ft])

    if not armed:
        status = "No"
        reasons = [f"Index only {drawdown:+.1f}% from 6-month high — no dip to buy"]
    elif drawdown <= -ACCUMULATION["deep_drawdown"] and bull_count >= 3 and ft:
        status = "Confirmed"
        reasons.insert(0, f"Deep drawdown ({drawdown:.1f}%) + {bull_count} stabilization signals — accumulate quality")
    elif bull_count >= 2:
        status = "Forming"
        reasons.insert(0, f"Drawdown {drawdown:.1f}% with {bull_count} early turn signals — begin staged accumulation")
    else:
        status = "Watching"
        reasons.insert(0, f"Fell {drawdown:.1f}% but no turn yet — wait for stabilization (don't catch the knife)")

    return {"status": status, "drawdown": round(drawdown, 1),
            "bull_signals": bull_count, "reasons": reasons, "signals": sig}
