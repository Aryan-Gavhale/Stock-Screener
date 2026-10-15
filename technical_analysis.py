"""
Technical analysis — MOMENTUM/TREND engine (v5).

Committed to ONE lane: trend-following + momentum with pullback/breakout entries.
We reward STRENGTH (riding the upper Bollinger band, high RS, new highs) and
penalise weakness — the opposite of the old mean-reversion contradiction.

Outputs a trend/momentum score, the current entry SETUP (breakout / pullback /
extended / base / none), ATR for risk sizing, and stability metrics.
All indicators use 3-5 day smoothing to prevent day-to-day score whipsaw.
"""

import pandas as pd
import numpy as np
import ta
from config import TRADE_PARAMS

# ═══════════════════════════════════════════════════════════════════════════════
#  Indicator computation
# ═══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["RSI"]     = ta.momentum.RSIIndicator(c, 14).rsi()
    macd          = ta.trend.MACD(c, 26, 12, 9)
    df["MACD"]    = macd.macd()
    df["MACD_S"]  = macd.macd_signal()
    df["MACD_H"]  = macd.macd_diff()
    df["EMA9"]    = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["EMA21"]   = ta.trend.EMAIndicator(c, 21).ema_indicator()
    df["SMA50"]   = c.rolling(50).mean()
    df["SMA150"]  = c.rolling(150).mean() if len(df) >= 150 else pd.Series(np.nan, index=df.index)
    df["SMA200"]  = c.rolling(200).mean() if len(df) >= 200 else pd.Series(np.nan, index=df.index)
    bb            = ta.volatility.BollingerBands(c, 20, 2)
    df["BB_U"]    = bb.bollinger_hband()
    df["BB_L"]    = bb.bollinger_lband()
    df["BB_P"]    = bb.bollinger_pband()
    if len(df) >= 14:
        adx       = ta.trend.ADXIndicator(h, l, c, 14)
        df["ADX"] = adx.adx()
        df["DI+"] = adx.adx_pos()
        df["DI-"] = adx.adx_neg()
    df["VOL_SMA20"] = v.rolling(20).mean()
    df["VOL_RATIO"] = v / df["VOL_SMA20"]
    df["ATR"]     = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    df["ATR_PCT"] = df["ATR"] / c * 100
    return df

# ═══════════════════════════════════════════════════════════════════════════════
#  Safe helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _v(series, idx=-1):
    try:
        val = series.iloc[idx]
        return None if pd.isna(val) else float(val)
    except Exception:
        return None

def _avg_last_n(series, n=5):
    if series is None:
        return None
    try:
        tail = series.dropna().tail(n)
        return float(tail.mean()) if len(tail) > 0 else None
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage analysis (Weinstein / Minervini) — stabilised
# ═══════════════════════════════════════════════════════════════════════════════

def detect_stage(df: pd.DataFrame) -> tuple:
    c_avg = df["Close"].tail(5).mean()
    s50  = _avg_last_n(df.get("SMA50"), 5)
    s150 = _avg_last_n(df.get("SMA150"), 5)
    s200 = _avg_last_n(df.get("SMA200"), 5)
    if s50 is None:
        return 0, "Insufficient data"

    low_52  = df["Close"].tail(min(252, len(df))).min()
    high_52 = df["Close"].tail(min(252, len(df))).max()
    above_50 = c_avg > s50

    conditions, total = [above_50], 1
    if s150 is not None:
        conditions.append(c_avg > s150); total += 1
    if s200 is not None:
        conditions.append(c_avg > s200); total += 1
        s200_series = df.get("SMA200").dropna()
        if len(s200_series) > 20:
            conditions.append(s200_series.iloc[-1] > s200_series.iloc[-20]); total += 1
    if s150 is not None and s200 is not None:
        conditions.append(s150 > s200); total += 1
    conditions.append(c_avg >= low_52 * 1.20); total += 1
    conditions.append(c_avg >= high_52 * 0.75); total += 1

    ratio = sum(conditions) / total if total else 0.5
    above_200 = (c_avg > s200) if s200 is not None else above_50

    if ratio >= 0.80:
        return 2, "Stage 2 — Uptrend (ideal buy zone)"
    elif ratio >= 0.60 and above_50:
        return 2, "Stage 2 — Early uptrend"
    elif above_50 and not above_200:
        return 1, "Stage 1 — Basing (accumulation)"
    elif not above_50 and above_200:
        return 3, "Stage 3 — Distribution (potential top)"
    elif not above_50 and not above_200:
        return 4, "Stage 4 — Downtrend (avoid)"
    return 2, "Stage 2 — Advancing"

# ═══════════════════════════════════════════════════════════════════════════════
#  Relative Strength vs benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def relative_strength(stock_df, bench_df, lookback: int = 252) -> float:
    n = min(lookback, len(stock_df) - 1, len(bench_df) - 1)
    if n < 20:
        return 1.0
    s_ret = stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-n] - 1
    b_ret = bench_df["Close"].iloc[-1] / bench_df["Close"].iloc[-n] - 1
    if b_ret == 0:
        return 1.0
    return round((1 + s_ret) / (1 + b_ret), 3)

def rs_percentile(rs_value: float, all_rs: list) -> float:
    if not all_rs:
        return 50.0
    rank = sum(1 for x in all_rs if x <= rs_value)
    return round(rank / len(all_rs) * 100, 1)

# ═══════════════════════════════════════════════════════════════════════════════
#  Momentum (12-1 academic + medium-term blend)
# ═══════════════════════════════════════════════════════════════════════════════

def momentum_score(df: pd.DataFrame) -> tuple:
    c = df["Close"]; n = len(c)
    ret = {}
    if n > 5:   ret["1w"]  = round((c.iloc[-1] / c.iloc[-5] - 1) * 100, 2)
    if n > 21:  ret["1m"]  = round((c.iloc[-1] / c.iloc[-21] - 1) * 100, 2)
    if n > 63:  ret["3m"]  = round((c.iloc[-1] / c.iloc[-63] - 1) * 100, 2)
    if n > 126: ret["6m"]  = round((c.iloc[-1] / c.iloc[-126] - 1) * 100, 2)
    if n > 252: ret["12m"] = round((c.iloc[-1] / c.iloc[-252] - 1) * 100, 2)

    # Academic 12-1 momentum: 12m return excluding the most recent month
    mom_12_1 = None
    if n > 252:
        mom_12_1 = (c.iloc[-21] / c.iloc[-252] - 1) * 100

    # Blend: favour medium/long term (stable, predictive), tiny weight on 1w
    parts, wsum = 0.0, 0.0
    for period, w in [("6m", 0.40), ("3m", 0.30), ("1m", 0.10), ("1w", 0.05)]:
        if period in ret:
            parts += ret[period] * w; wsum += w
    if mom_12_1 is not None:
        parts += mom_12_1 * 0.15; wsum += 0.15
    raw = (parts / wsum) if wsum > 0 else 0

    score = max(0, min(100, 50 + raw * (50 / 30)))  # +30% blended → 100

    # Crash guard: sharp recent drop against a positive longer trend
    if ret.get("1m", 0) < -10 and ret.get("6m", 0) > 10:
        score *= 0.75
    return round(score, 1), ret

# ═══════════════════════════════════════════════════════════════════════════════
#  Indicator scores — TREND-FOLLOWING interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def score_rsi(df):
    """Trend context: healthy strength (55-72) is best. Extreme (>82) = extended.
    A pullback dip (45-55) is fine. Weakness (<40) is bad."""
    r = _avg_last_n(df.get("RSI"), 5)
    if r is None:
        return 50, "No RSI"
    if 58 <= r <= 72:  s = 82
    elif 50 <= r < 58: s = 70
    elif 72 < r <= 80: s = 62
    elif 45 <= r < 50: s = 55
    elif r > 80:       s = 42
    elif 40 <= r < 45: s = 40
    else:              s = 22
    return s, f"RSI(5d) {r:.0f}"

def score_macd(df):
    h_series, m_series, s_series = df.get("MACD_H"), df.get("MACD"), df.get("MACD_S")
    if h_series is None or m_series is None:
        return 50, "No MACD"
    h_avg = _avg_last_n(h_series, 5)
    m_now, s_now = _v(m_series), _v(s_series)
    if h_avg is None or m_now is None or s_now is None:
        return 50, "No MACD"
    h_tail = h_series.dropna().tail(5)
    improving = len(h_tail) >= 3 and h_tail.iloc[-1] > h_tail.iloc[0]
    above = m_now > s_now
    m_tail, s_tail = m_series.dropna().tail(5), s_series.dropna().tail(5)
    ml = min(len(m_tail), len(s_tail))
    consistency = (sum(1 for i in range(ml) if m_tail.iloc[-(i+1)] > s_tail.iloc[-(i+1)]) / ml) if ml >= 3 else 0.5
    if above and h_avg > 0 and improving and consistency >= 0.8:
        return 82, "MACD strongly bullish"
    if above and h_avg > 0:
        return 68, "MACD bullish"
    if above:
        return 58, "MACD turning up"
    if not above and h_avg < 0 and consistency <= 0.2:
        return 18, "MACD strongly bearish"
    return 35, "MACD weak"

def score_ma(df):
    """Trend backbone: price above rising, stacked MAs = strongest signal."""
    c_avg = df["Close"].tail(5).mean()
    e21 = _avg_last_n(df.get("EMA21"), 5)
    s50 = _avg_last_n(df.get("SMA50"), 5)
    s150 = _avg_last_n(df.get("SMA150"), 5)
    s200 = _avg_last_n(df.get("SMA200"), 5)
    pts, mx = 0, 0
    for val in [e21, s50, s150, s200]:
        if val is not None:
            mx += 1
            if c_avg > val:
                pts += 1
    # Stacking bonus (proper uptrend structure)
    if e21 and s50:
        mx += 1
        if e21 > s50: pts += 1
    if s50 and s200:
        mx += 1
        if s50 > s200: pts += 1
    if mx <= 1:
        score = 50 + (pts * 25) if mx > 0 else 50
    else:
        score = pts / mx * 100
    return round(score), f"Price(5d) above/stacked {pts}/{mx} MAs"

def score_volume(df):
    vr = _avg_last_n(df.get("VOL_RATIO"), 5)
    if vr is None:
        return 50, "No vol"
    pc = (df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1) if len(df) >= 6 else 0
    up, down = pc > 0.005, pc < -0.005
    if up and vr > 1.3:   return 80, f"Vol(5d) {vr:.1f}x — accumulation"
    if up and vr > 1.0:   return 66, f"Vol(5d) {vr:.1f}x — buying"
    if down and vr > 1.3: return 20, f"Vol(5d) {vr:.1f}x — distribution"
    if down and vr < 0.8: return 52, f"Vol(5d) {vr:.1f}x — quiet pullback"
    return 50, f"Vol(5d) {vr:.1f}x"

def score_bollinger(df):
    """Trend-following: riding the UPPER band = strength (not 'overbought').
    Near/below the LOWER band = weakness."""
    p = _avg_last_n(df.get("BB_P"), 3)
    if p is None:
        return 50, "No BB"
    if p > 1.0:   return 78, f"BB%B {p:.2f} — riding upper band (strong)"
    if p > 0.8:   return 72, f"BB%B {p:.2f} — upper band strength"
    if p > 0.6:   return 62, f"BB%B {p:.2f} — upper half"
    if p > 0.4:   return 50, f"BB%B {p:.2f} — mid"
    if p > 0.2:   return 38, f"BB%B {p:.2f} — lower half (weak)"
    if p >= 0:    return 28, f"BB%B {p:.2f} — near lower band"
    return 18, f"BB%B {p:.2f} — below lower band (breaking down)"

def score_adx(df):
    adx = _avg_last_n(df.get("ADX"), 5)
    dip = _avg_last_n(df.get("DI+"), 5)
    dim = _avg_last_n(df.get("DI-"), 5)
    if adx is None or dip is None or dim is None:
        return 50, "No ADX"
    bull = dip > dim
    if adx > 30 and bull: return 85, f"ADX(5d) {adx:.0f} — strong uptrend"
    if adx > 22 and bull: return 68, f"ADX(5d) {adx:.0f} — uptrend"
    if adx > 30 and not bull: return 18, f"ADX(5d) {adx:.0f} — strong downtrend"
    if adx > 22 and not bull: return 32, f"ADX(5d) {adx:.0f} — downtrend"
    return 50, f"ADX(5d) {adx:.0f} — no trend"

# ═══════════════════════════════════════════════════════════════════════════════
#  Trend consistency
# ═══════════════════════════════════════════════════════════════════════════════

def trend_consistency(df: pd.DataFrame) -> float:
    c = df["Close"]; n = len(c); signals = []
    if n > 5:  signals.append(1 if c.iloc[-1] > c.iloc[-5] else -1)
    if n > 10: signals.append(1 if c.iloc[-1] > c.iloc[-10] else -1)
    if n > 20: signals.append(1 if c.iloc[-1] > c.iloc[-20] else -1)
    e9  = _avg_last_n(df.get("EMA9"), 3)
    e21 = _avg_last_n(df.get("EMA21"), 3)
    s50 = _avg_last_n(df.get("SMA50"), 3)
    if e9 and e21:  signals.append(1 if e9 > e21 else -1)
    if e21 and s50: signals.append(1 if e21 > s50 else -1)
    if not signals:
        return 50
    pos = sum(1 for s in signals if s > 0)
    consistency = abs(pos / len(signals) - 0.5) * 2
    return round(50 + consistency * 50 * (1 if pos > len(signals) / 2 else -1), 1)

# ═══════════════════════════════════════════════════════════════════════════════
#  Entry SETUP detection — the actionable trade trigger
# ═══════════════════════════════════════════════════════════════════════════════

def detect_setup(df: pd.DataFrame, stage_num: int) -> dict:
    """
    Classify the current actionable setup:
      BREAKOUT  — new N-day high with volume, in uptrend
      PULLBACK  — uptrend pulling back to rising EMA21 (low-risk entry)
      EXTENDED  — too far above EMA21, don't chase
      BASE      — consolidating, watch for breakout
      NONE      — not in an uptrend / not actionable
    Returns dict with setup, quality (0-100), and a human note.
    """
    c = df["Close"].iloc[-1]
    e21 = _avg_last_n(df.get("EMA21"), 3)
    s50 = _avg_last_n(df.get("SMA50"), 3)
    atr = _v(df.get("ATR")) or (c * 0.02)
    vr  = _avg_last_n(df.get("VOL_RATIO"), 5) or 1.0
    rsi = _avg_last_n(df.get("RSI"), 3) or 50

    lookback = TRADE_PARAMS["breakout_lookback"]
    if len(df) > lookback:
        prior_high = df["Close"].iloc[-lookback:-1].max()
    else:
        prior_high = df["Close"].max()

    uptrend = (s50 is not None and c > s50) and stage_num in (1, 2)

    # EXTENDED check
    if e21 and atr > 0:
        atr_above = (c - e21) / atr
    else:
        atr_above = 0

    if not uptrend:
        return {"setup": "NONE", "quality": 20,
                "note": "Not in an uptrend — no long setup"}

    if atr_above > TRADE_PARAMS["extended_atr_from_ema"]:
        return {"setup": "EXTENDED", "quality": 35,
                "note": f"{atr_above:.1f} ATR above EMA21 — wait for pullback, don't chase"}

    # BREAKOUT: closing at/above prior N-day high with volume
    if c >= prior_high * 0.995 and vr > 1.1 and stage_num == 2:
        q = 78 + min(12, (vr - 1.1) * 20)
        return {"setup": "BREAKOUT", "quality": round(min(95, q)),
                "note": f"Breaking {lookback}-day high on {vr:.1f}x volume"}

    # PULLBACK: near rising EMA21, RSI cooled into band
    if e21 and abs(c - e21) / atr <= 1.5 and \
       TRADE_PARAMS["pullback_rsi_low"] <= rsi <= TRADE_PARAMS["pullback_rsi_high"]:
        return {"setup": "PULLBACK", "quality": 82,
                "note": f"Pullback to EMA21 (RSI {rsi:.0f}) — low-risk entry in uptrend"}

    # BASE: within 8% of prior high, tight
    if c >= prior_high * 0.92 and stage_num in (1, 2):
        return {"setup": "BASE", "quality": 58,
                "note": f"Basing near highs — watch for breakout above ₹{prior_high:.0f}"}

    return {"setup": "IN-TREND", "quality": 62,
            "note": "In uptrend, no precise trigger — hold/add on dips"}

# ═══════════════════════════════════════════════════════════════════════════════
#  Composite technical (trend/momentum) score
# ═══════════════════════════════════════════════════════════════════════════════

def technical_score(df: pd.DataFrame, bench_df: pd.DataFrame = None) -> dict:
    if df is None or len(df) < 30:
        return {"score": 0, "stage": (0, "N/A"), "trend": "No Data",
                "setup": {"setup": "NONE", "quality": 0, "note": "No data"}}

    df = add_indicators(df.copy())
    stage = detect_stage(df)
    mom_sc, rets = momentum_score(df)
    tc = trend_consistency(df)
    setup = detect_setup(df, stage[0])

    rs = relative_strength(df, bench_df) if bench_df is not None else 1.0

    rsi_s, rsi_r   = score_rsi(df)
    macd_s, macd_r = score_macd(df)
    ma_s, ma_r     = score_ma(df)
    vol_s, vol_r   = score_volume(df)
    bb_s, bb_r     = score_bollinger(df)
    adx_s, adx_r   = score_adx(df)

    # Trend/momentum weighting: MA structure + momentum + ADX dominate
    ind_score = (
        ma_s   * 0.26 +
        mom_sc * 0.22 +
        adx_s  * 0.16 +
        macd_s * 0.12 +
        rsi_s  * 0.10 +
        bb_s   * 0.08 +
        vol_s  * 0.06
    )

    stage_mult = {2: 1.05, 1: 0.97, 3: 0.90, 4: 0.80, 0: 1.0}
    final = ind_score * stage_mult.get(stage[0], 1.0)
    final += (tc - 50) / 50 * 3
    final = min(100, max(0, final))

    daily_ret = df["Close"].pct_change().dropna()
    vol_ann = round(daily_ret.std() * np.sqrt(252) * 100, 1)
    turnover_cr = (
        (df["Close"] * df["Volume"]).replace([np.inf, -np.inf], np.nan)
        .dropna().tail(20).median() / 1e7
    )
    atr = _v(df.get("ATR"))
    atr_pct = _v(df.get("ATR_PCT"))

    price = df["Close"].iloc[-1]
    h52 = df["Close"].tail(252).max()
    l52 = df["Close"].tail(252).min()

    if final >= 70:   trend = "Strong Bullish"
    elif final >= 58: trend = "Bullish"
    elif final >= 42: trend = "Neutral"
    elif final >= 30: trend = "Bearish"
    else:             trend = "Strong Bearish"

    return {
        "score": round(final, 1),
        "stage": stage,
        "trend": trend,
        "setup": setup,
        "rs": rs,
        "rs_pct": 50.0,  # filled at portfolio level
        "momentum": mom_sc,
        "trend_consistency": tc,
        "returns": rets,
        "volatility": vol_ann,
        "avg_turnover_cr": round(float(turnover_cr), 2)
        if not pd.isna(turnover_cr) else 0.0,
        "atr": round(atr, 2) if atr else None,
        "atr_pct": round(atr_pct, 2) if atr_pct else None,
        "price": round(price, 2),
        "high_52w": round(h52, 2),
        "low_52w": round(l52, 2),
        "from_high": round((price / h52 - 1) * 100, 1),
        "from_low": round((price / l52 - 1) * 100, 1),
        "indicators": {
            "Trend/MAs":  {"s": ma_s,   "r": ma_r},
            "Momentum":   {"s": mom_sc, "r": "6m/3m/12-1 blended momentum"},
            "ADX":        {"s": adx_s,  "r": adx_r},
            "MACD":       {"s": macd_s, "r": macd_r},
            "RSI":        {"s": rsi_s,  "r": rsi_r},
            "Bollinger":  {"s": bb_s,   "r": bb_r},
            "Volume":     {"s": vol_s,  "r": vol_r},
        },
        "df": df,
    }
