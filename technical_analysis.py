"""
Advanced technical analysis.

Implements:
  - Trend Stage detection (Weinstein / Minervini method)
  - Relative Strength vs Nifty 50
  - Multi-timeframe momentum
  - Classic indicators: RSI, MACD, MAs, Bollinger, ADX, Volume
  - Composite technical score 0-100
"""

import pandas as pd
import numpy as np
import ta

# ═══════════════════════════════════════════════════════════════════════════════
#  Indicator computation
# ═══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["RSI"]         = ta.momentum.RSIIndicator(c, 14).rsi()
    macd              = ta.trend.MACD(c, 26, 12, 9)
    df["MACD"]        = macd.macd()
    df["MACD_S"]      = macd.macd_signal()
    df["MACD_H"]      = macd.macd_diff()
    df["EMA9"]        = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["EMA21"]       = ta.trend.EMAIndicator(c, 21).ema_indicator()
    df["SMA50"]       = c.rolling(50).mean()
    df["SMA150"]      = c.rolling(150).mean() if len(df) >= 150 else pd.Series(np.nan, index=df.index)
    df["SMA200"]      = c.rolling(200).mean() if len(df) >= 200 else pd.Series(np.nan, index=df.index)
    bb                = ta.volatility.BollingerBands(c, 20, 2)
    df["BB_U"]        = bb.bollinger_hband()
    df["BB_L"]        = bb.bollinger_lband()
    df["BB_P"]        = bb.bollinger_pband()
    if len(df) >= 14:
        adx           = ta.trend.ADXIndicator(h, l, c, 14)
        df["ADX"]     = adx.adx()
        df["DI+"]     = adx.adx_pos()
        df["DI-"]     = adx.adx_neg()
    df["VWAP20"]      = (c * v).rolling(20).sum() / v.rolling(20).sum()
    df["VOL_SMA20"]   = v.rolling(20).mean()
    df["VOL_RATIO"]   = v / df["VOL_SMA20"]
    df["ATR"]         = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    df["ATR_PCT"]     = df["ATR"] / c * 100
    return df

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage analysis (Weinstein / Minervini)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_stage(df: pd.DataFrame) -> tuple[int, str]:
    """
    Returns (stage_number, description).
    Stage 2 = ideal buying zone.
    Works even with limited data by using available MAs.
    """
    c = df["Close"].iloc[-1]

    s50_val  = df.get("SMA50")
    s150_val = df.get("SMA150")
    s200_val = df.get("SMA200")

    s50  = _v(s50_val)  if s50_val is not None else None
    s150 = _v(s150_val) if s150_val is not None else None
    s200 = _v(s200_val) if s200_val is not None else None

    # If we don't even have SMA50, just return neutral
    if s50 is None:
        return 0, "Insufficient data"

    low_52  = df["Close"].tail(min(252, len(df))).min()
    high_52 = df["Close"].tail(min(252, len(df))).max()

    above_50 = c > s50

    # Conditions that require longer MAs — gracefully skip if unavailable
    conditions = [above_50]
    total_checks = 1

    if s150 is not None:
        conditions.append(c > s150)
        total_checks += 1
    if s200 is not None:
        conditions.append(c > s200)
        total_checks += 1
        sma200_series = s200_val.dropna() if s200_val is not None else pd.Series(dtype=float)
        if len(sma200_series) > 20:
            conditions.append(sma200_series.iloc[-1] > sma200_series.iloc[-20])
            total_checks += 1
    if s150 is not None and s200 is not None:
        conditions.append(s150 > s200)
        total_checks += 1

    conditions.append(c >= low_52 * 1.20)
    total_checks += 1
    conditions.append(c >= high_52 * 0.75)
    total_checks += 1

    met = sum(conditions)
    ratio = met / total_checks if total_checks > 0 else 0.5

    above_200 = (c > s200) if s200 is not None else above_50

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
    else:
        return 2, "Stage 2 — Advancing"

# ═══════════════════════════════════════════════════════════════════════════════
#  Relative Strength vs benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def relative_strength(stock_df: pd.DataFrame, bench_df: pd.DataFrame,
                      lookback: int = 252) -> float:
    """
    Mansfield-style RS: stock return vs benchmark return over lookback days.
    Returns a ratio > 1 means outperforming.
    """
    n = min(lookback, len(stock_df) - 1, len(bench_df) - 1)
    if n < 20:
        return 1.0
    s_ret = stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-n] - 1
    b_ret = bench_df["Close"].iloc[-1] / bench_df["Close"].iloc[-n] - 1
    if b_ret == 0:
        return 1.0
    return round((1 + s_ret) / (1 + b_ret), 3)

def rs_percentile(rs_value: float, all_rs: list[float]) -> float:
    """Percentile rank of this stock's RS among all stocks."""
    if not all_rs:
        return 50.0
    rank = sum(1 for x in all_rs if x <= rs_value)
    return round(rank / len(all_rs) * 100, 1)

# ═══════════════════════════════════════════════════════════════════════════════
#  Momentum (multi-timeframe)
# ═══════════════════════════════════════════════════════════════════════════════

def momentum_score(df: pd.DataFrame) -> tuple[float, dict]:
    """
    6-12-1 momentum: strong 6m+12m returns but not extreme 1m crash.
    Returns (score 0-100, details dict).
    """
    c = df["Close"]
    n = len(c)
    ret = {}
    if n > 5:
        ret["1w"] = round((c.iloc[-1] / c.iloc[-5] - 1) * 100, 2)
    if n > 21:
        ret["1m"] = round((c.iloc[-1] / c.iloc[-21] - 1) * 100, 2)
    if n > 63:
        ret["3m"] = round((c.iloc[-1] / c.iloc[-63] - 1) * 100, 2)
    if n > 126:
        ret["6m"] = round((c.iloc[-1] / c.iloc[-126] - 1) * 100, 2)
    if n > 252:
        ret["12m"] = round((c.iloc[-1] / c.iloc[-252] - 1) * 100, 2)

    # Composite momentum: 40% 6m + 30% 3m + 20% 1m + 10% 1w
    parts = []
    if "6m" in ret:
        parts.append(ret["6m"] * 0.40)
    if "3m" in ret:
        parts.append(ret["3m"] * 0.30)
    if "1m" in ret:
        parts.append(ret["1m"] * 0.20)
    if "1w" in ret:
        parts.append(ret["1w"] * 0.10)

    raw = sum(parts) if parts else 0

    # Map raw momentum to 0-100 score
    # +30% composite → 100, 0% → 50, -30% → 0
    score = max(0, min(100, 50 + raw * (50 / 30)))

    # Penalty: if last month is very negative while longer term is positive,
    # that's a momentum crash signal — reduce score
    if ret.get("1m", 0) < -8 and ret.get("6m", 0) > 10:
        score *= 0.7

    return round(score, 1), ret

# ═══════════════════════════════════════════════════════════════════════════════
#  Individual indicator scores
# ═══════════════════════════════════════════════════════════════════════════════

def _v(series, idx=-1):
    """Safe last value."""
    try:
        val = series.iloc[idx]
        return None if pd.isna(val) else val
    except Exception:
        return None


def score_rsi(df):
    r = _v(df.get("RSI"))
    if r is None:
        return 50, "No RSI"
    if r < 30:
        return 85, f"RSI {r:.0f} — Oversold"
    if r < 45:
        return 70, f"RSI {r:.0f} — Nearing oversold"
    if r < 55:
        return 55, f"RSI {r:.0f} — Neutral"
    if r < 70:
        return 40, f"RSI {r:.0f} — Elevated"
    return 15, f"RSI {r:.0f} — Overbought"


def score_macd(df):
    m, s, h = _v(df.get("MACD")), _v(df.get("MACD_S")), _v(df.get("MACD_H"))
    if m is None or s is None:
        return 50, "No MACD"
    prev_h = _v(df.get("MACD_H"), -2)
    expanding = prev_h is not None and h is not None and h > prev_h

    prev_m, prev_s = _v(df.get("MACD"), -2), _v(df.get("MACD_S"), -2)
    bull_cross = prev_m is not None and prev_s is not None and prev_m <= prev_s and m > s

    if bull_cross:
        return 90, "MACD bullish crossover"
    if m > s and h > 0 and expanding:
        return 80, "MACD bullish, histogram expanding"
    if m > s:
        return 65, "MACD above signal"
    if m < s and h < 0:
        return 25, "MACD bearish"
    return 45, "MACD neutral"


def score_ma(df):
    c  = df["Close"].iloc[-1]
    e21 = _v(df.get("EMA21"))
    s50 = _v(df.get("SMA50"))
    s150= _v(df.get("SMA150"))
    s200= _v(df.get("SMA200"))
    pts, mx = 0, 0
    for val in [e21, s50, s150, s200]:
        if val is not None:
            mx += 1
            if c > val:
                pts += 1
    if s50 is not None and s200 is not None:
        mx += 1
        if s50 > s200:
            pts += 1
    # If only 1-2 MAs available, don't punish — scale to a base of 50
    if mx <= 1:
        score = 50 + (pts * 25) if mx > 0 else 50
    else:
        score = (pts / mx * 100) if mx else 50
    return round(score), f"Price above {pts}/{mx} key MAs"


def score_volume(df):
    vr = _v(df.get("VOL_RATIO"))
    pc = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) if len(df) > 1 else 0
    if vr is None:
        return 50, "No vol data"
    up = pc > 0
    if up and vr > 1.5:
        return 85, f"Vol {vr:.1f}x — Strong buying"
    if up and vr > 1.0:
        return 70, f"Vol {vr:.1f}x — Buying"
    if not up and vr > 1.5:
        return 20, f"Vol {vr:.1f}x — Heavy selling"
    if not up and vr < 0.7:
        return 55, f"Vol {vr:.1f}x — Low-vol dip"
    return 50, f"Vol {vr:.1f}x — Normal"


def score_bollinger(df):
    p = _v(df.get("BB_P"))
    if p is None:
        return 50, "No BB"
    if p < 0:
        return 85, "Below lower BB — Oversold"
    if p < 0.2:
        return 72, "Near lower BB"
    if p < 0.5:
        return 58, "Lower half of BB"
    if p < 0.8:
        return 42, "Upper half of BB"
    if p < 1.0:
        return 25, "Near upper BB"
    return 12, "Above upper BB — Overbought"


def score_adx(df):
    adx = _v(df.get("ADX"))
    dip = _v(df.get("DI+"))
    dim = _v(df.get("DI-"))
    if adx is None or dip is None:
        return 50, "No ADX"
    bull = dip > dim
    if adx > 35 and bull:
        return 88, f"ADX {adx:.0f} — Strong uptrend"
    if adx > 25 and bull:
        return 72, f"ADX {adx:.0f} — Uptrend"
    if adx > 25 and not bull:
        return 22, f"ADX {adx:.0f} — Downtrend"
    return 50, f"ADX {adx:.0f} — No trend"

# ═══════════════════════════════════════════════════════════════════════════════
#  Composite technical score
# ═══════════════════════════════════════════════════════════════════════════════

def technical_score(df: pd.DataFrame, bench_df: pd.DataFrame = None) -> dict:
    """Full technical analysis. Returns dict with scores + metadata."""
    if df is None or len(df) < 30:
        return {"score": 0, "stage": (0, "N/A"), "trend": "No Data"}

    df = add_indicators(df.copy())
    stage = detect_stage(df)
    mom_sc, rets = momentum_score(df)

    rs = 1.0
    rs_pct = 50.0
    if bench_df is not None:
        rs = relative_strength(df, bench_df)

    rsi_s,  rsi_r  = score_rsi(df)
    macd_s, macd_r = score_macd(df)
    ma_s,   ma_r   = score_ma(df)
    vol_s,  vol_r  = score_volume(df)
    bb_s,   bb_r   = score_bollinger(df)
    adx_s,  adx_r  = score_adx(df)

    # Weighted composite of indicator scores
    ind_score = (
        rsi_s  * 0.15 + macd_s * 0.15 + ma_s * 0.25 +
        vol_s  * 0.10 + bb_s   * 0.10 + adx_s * 0.10 +
        mom_sc * 0.15
    )

    # Stage bonus / penalty (gentle — don't crush scores)
    stage_mult = {2: 1.08, 1: 0.95, 3: 0.88, 4: 0.75, 0: 1.0}
    final = min(100, ind_score * stage_mult.get(stage[0], 1.0))

    # Volatility
    daily_ret = df["Close"].pct_change().dropna()
    vol_ann = round(daily_ret.std() * np.sqrt(252) * 100, 1)

    price = df["Close"].iloc[-1]
    h52 = df["Close"].tail(252).max()
    l52 = df["Close"].tail(252).min()

    if final >= 72:
        trend = "Strong Bullish"
    elif final >= 58:
        trend = "Bullish"
    elif final >= 42:
        trend = "Neutral"
    elif final >= 28:
        trend = "Bearish"
    else:
        trend = "Strong Bearish"

    return {
        "score": round(final, 1),
        "stage": stage,
        "trend": trend,
        "rs": rs,
        "rs_pct": rs_pct,        # filled later at portfolio level
        "momentum": mom_sc,
        "returns": rets,
        "volatility": vol_ann,
        "price": round(price, 2),
        "high_52w": round(h52, 2),
        "low_52w": round(l52, 2),
        "from_high": round((price / h52 - 1) * 100, 1),
        "from_low": round((price / l52 - 1) * 100, 1),
        "indicators": {
            "RSI":       {"s": rsi_s,  "r": rsi_r},
            "MACD":      {"s": macd_s, "r": macd_r},
            "MAs":       {"s": ma_s,   "r": ma_r},
            "Volume":    {"s": vol_s,  "r": vol_r},
            "Bollinger": {"s": bb_s,   "r": bb_r},
            "ADX":       {"s": adx_s,  "r": adx_r},
        },
        "df": df,  # keep for charting
    }
