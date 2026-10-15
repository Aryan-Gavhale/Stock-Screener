"""
Interpretable sector-rotation forecast engine.

The output is a probability-weighted ranking, not a deterministic prediction.
Price/breadth features form the validated core. Current fundamentals, macro,
news and seasonality are capped overlays and are reported separately.
"""

from __future__ import annotations

from calendar import month_abbr

import numpy as np
import pandas as pd
from scipy import stats

from config import SECTOR_DRIVERS, SECTOR_ROTATION


def _safe_return(series: pd.Series, days: int) -> float:
    values = series.dropna()
    if len(values) <= days or values.iloc[-days - 1] <= 0:
        return 0.0
    return float(values.iloc[-1] / values.iloc[-days - 1] - 1)


def _percentile(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(50.0, index=series.index)
    return series.rank(pct=True, method="average") * 100


def _price_features(
    price: pd.DataFrame, bench: pd.DataFrame, horizon_days: int
) -> dict | None:
    close = price["Close"].dropna()
    bench_close = bench["Close"].dropna()
    if len(close) < max(65, horizon_days + 22) or len(bench_close) < 65:
        return None

    returns = {days: _safe_return(close, days) for days in (21, 63, 126, 252)}
    bench_returns = {
        days: _safe_return(bench_close, days) for days in (21, 63, 126, 252)
    }
    excess = {days: returns[days] - bench_returns[days] for days in returns}

    selected = excess.get(horizon_days, excess[63])
    long_term = excess[126] * 0.55 + excess[252] * 0.45
    momentum = selected * 0.65 + long_term * 0.35
    acceleration = excess[21] - excess[63] / 3

    last = float(close.iloc[-1])
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    trend_votes = [
        last > ma20,
        last > ma50,
        ma20 > ma50,
    ]
    if not np.isnan(ma200):
        trend_votes.extend([last > ma200, ma50 > ma200])
    trend = sum(trend_votes) / len(trend_votes) * 100

    daily = close.pct_change().dropna()
    vol = float(daily.tail(63).std() * np.sqrt(252)) if len(daily) else 0.3
    drawdown = float(last / close.tail(252).max() - 1)
    risk_quality = np.clip(78 - vol * 90 + max(-25, drawdown * 80), 0, 100)

    return {
        "momentum_raw": momentum,
        "acceleration_raw": acceleration,
        "trend": trend,
        "risk_quality": float(risk_quality),
        "return_1m": returns[21] * 100,
        "return_3m": returns[63] * 100,
        "return_6m": returns[126] * 100,
        "return_12m": returns[252] * 100,
        "excess_1m": excess[21] * 100,
        "excess_3m": excess[63] * 100,
        "excess_6m": excess[126] * 100,
        "volatility": vol * 100,
        "drawdown": drawdown * 100,
    }


def _breadth_score(breadth: dict) -> float:
    return float(np.clip(
        breadth.get("above20", 50) * 0.25
        + breadth.get("above50", 50) * 0.30
        + breadth.get("above200", 50) * 0.20
        + breadth.get("new_highs", 0) * 0.10
        + (100 - breadth.get("new_lows", 0)) * 0.05
        + breadth.get("volume_participation", 50) * 0.10,
        0, 100,
    ))


def _context_change(frame: pd.DataFrame | None, days: int) -> float:
    if frame is None:
        return 0.0
    return _safe_return(frame["Close"], days) * 100


def macro_fit_score(sector: str, context: dict, regime: dict) -> tuple[float, list]:
    """Capped directional macro fit with explicit evidence."""
    likes = SECTOR_DRIVERS.get(sector, {}).get("likes", [])
    score, evidence = 50.0, []
    regime_label = (regime or {}).get("label", "")
    risk_on = "RISK-ON" in regime_label
    risk_off = "RISK-OFF" in regime_label or "CRASH" in regime_label
    inr = _context_change((context or {}).get("usdinr"), 21)
    crude = _context_change((context or {}).get("crude"), 21)
    global_return = _context_change((context or {}).get("sp500"), 63)

    def apply(condition, points, text):
        nonlocal score
        if condition:
            score += points
            evidence.append(text)

    apply("risk_on" in likes and risk_on, 14, "Current regime supports cyclicals")
    apply("risk_off" in likes and risk_off, 14, "Defensive fit in risk-off regime")
    apply("inr_weak" in likes and inr > 1, 10, f"Weaker INR ({inr:+.1f}%/1M)")
    apply("crude_up" in likes and crude > 5, 10, f"Crude rising ({crude:+.1f}%/1M)")
    apply(("crude_down" in likes or "gas_down" in likes) and crude < -5,
          10, f"Energy input costs easing ({crude:+.1f}%/1M)")
    apply("commodity_up" in likes and crude > 5, 8, "Commodity-cycle tailwind")
    apply("global_risk_on" in likes and global_return > 2,
          9, f"Global demand proxy positive ({global_return:+.1f}%/3M)")
    if not evidence:
        evidence.append("No strong macro tailwind or headwind")
    return float(np.clip(score, 0, 100)), evidence


def seasonality_signal(
    sector_price: pd.DataFrame,
    bench: pd.DataFrame,
    horizon_days: int,
    asof=None,
) -> dict:
    """
    Training-only month-of-year excess-return estimate with shrinkage.
    `asof` truncation makes this safe to call inside a walk-forward backtest.
    """
    sector = sector_price["Close"]
    benchmark = bench["Close"]
    if asof is not None:
        sector, benchmark = sector.loc[:asof], benchmark.loc[:asof]
    aligned = pd.concat([sector.rename("sector"), benchmark.rename("bench")],
                        axis=1).dropna()
    if len(aligned) < 252 * 3:
        return {"score": 50.0, "expected": 0.0, "pvalue": 1.0,
                "sample": 0, "stable": False, "target_month": None}

    monthly = aligned.resample("ME").last().pct_change(fill_method=None).dropna()
    monthly["excess"] = monthly["sector"] - monthly["bench"]
    months_ahead = max(1, round(horizon_days / 21))
    target_date = aligned.index[-1] + pd.DateOffset(months=months_ahead)
    sample = monthly.loc[monthly.index.month == target_date.month, "excess"].dropna()
    minimum = SECTOR_ROTATION["min_seasonal_years"]
    if len(sample) < minimum:
        return {"score": 50.0, "expected": 0.0, "pvalue": 1.0,
                "sample": len(sample), "stable": False,
                "target_month": month_abbr[target_date.month]}

    mean = float(sample.mean())
    std = float(sample.std(ddof=1))
    shrink = len(sample) / (len(sample) + 8)
    expected = mean * shrink
    if std > 0:
        t_stat = mean / (std / np.sqrt(len(sample)))
        pvalue = float(2 * stats.t.sf(abs(t_stat), df=len(sample) - 1))
    else:
        pvalue = 1.0
    score = float(np.clip(50 + expected * 600, 25, 75))
    return {
        "score": round(score, 1),
        "expected": round(expected * 100, 2),
        "pvalue": pvalue,
        "sample": len(sample),
        "stable": False,  # assigned after cross-sector FDR control
        "target_month": month_abbr[target_date.month],
    }


def _apply_fdr(seasonality: dict, alpha: float = 0.10) -> dict:
    """Benjamini-Hochberg control across all sector seasonality tests."""
    ordered = sorted(
        [(sector, item["pvalue"]) for sector, item in seasonality.items()],
        key=lambda pair: pair[1],
    )
    passed = set()
    total = max(1, len(ordered))
    for rank, (sector, pvalue) in enumerate(ordered, start=1):
        if pvalue <= alpha * rank / total:
            passed.add(sector)
    for sector, item in seasonality.items():
        item["stable"] = sector in passed and item["sample"] >= SECTOR_ROTATION[
            "min_seasonal_years"
        ]
        if not item["stable"]:
            item["score"] = 50.0
            item["expected"] = 0.0
    return seasonality


def _status(current, forward, acceleration):
    if current >= 70 and forward >= 62:
        return "Booming Now"
    if forward >= 64 and acceleration >= 58:
        return "Emerging"
    if current >= 60 and (forward < 48 or acceleration < 35):
        return "Fading"
    if forward >= 56:
        return "Watch"
    if current < 40 and forward < 43:
        return "Avoid"
    return "Neutral"


def compute_sector_forecasts(
    sector_data: dict,
    bench: pd.DataFrame,
    horizon: str,
    context: dict | None = None,
    regime: dict | None = None,
    news_map: dict | None = None,
    asof=None,
    include_live_overlays: bool = True,
    disabled_overlays: set[str] | None = None,
) -> list[dict]:
    """Rank all sectors for one horizon using an interpretable ensemble."""
    config = SECTOR_ROTATION["horizons"][horizon]
    disabled_overlays = disabled_overlays or set()
    horizon_days = config["days"]
    raw = {}
    seasonal = {}
    for sector, item in (sector_data or {}).items():
        price = item.get("price")
        if price is None:
            continue
        price_cut = price.loc[:asof] if asof is not None else price
        bench_cut = bench.loc[:asof] if asof is not None else bench
        features = _price_features(price_cut, bench_cut, horizon_days)
        if features is None:
            continue
        features["breadth_score"] = (
            _breadth_score(item.get("breadth", {}))
            if asof is None else 50.0
        )
        raw[sector] = features
        seasonal[sector] = seasonality_signal(
            price, bench, horizon_days, asof=asof
        )
    if not raw:
        return []

    seasonal = _apply_fdr(seasonal)
    frame = pd.DataFrame(raw).T
    frame["momentum"] = _percentile(frame["momentum_raw"])
    frame["acceleration"] = _percentile(frame["acceleration_raw"])
    weights = config["weights"]
    forecasts = []
    for sector, row in frame.iterrows():
        item = sector_data[sector]
        breadth_score = float(row["breadth_score"])
        core = (
            row["momentum"] * weights["mom"]
            + row["acceleration"] * weights["accel"]
            + breadth_score * weights["breadth"]
            + row["trend"] * weights["trend"]
            + row["risk_quality"] * weights["risk"]
        )
        current = (
            row["momentum"] * 0.45 + row["trend"] * 0.25
            + breadth_score * 0.20 + row["risk_quality"] * 0.10
        )

        fundamentals = item.get("fundamentals", {})
        fundamental_score = (
            fundamentals.get("growth_breadth", 50) * 0.60
            + fundamentals.get("quality_breadth", 50) * 0.40
        )
        macro_score, macro_evidence = macro_fit_score(
            sector, context or {}, regime or {}
        )
        news = (news_map or {}).get(sector, {})
        catalyst_score = float(news.get("score", 50))
        seasonal_item = seasonal[sector]

        forward = float(core)
        if include_live_overlays:
            if "fundamentals" not in disabled_overlays:
                forward += (fundamental_score - 50) / 50 * (
                    SECTOR_ROTATION["live_fundamental_weight_cap"] * 100
                )
            if "macro" not in disabled_overlays:
                forward += (macro_score - 50) / 50 * (
                    SECTOR_ROTATION["macro_weight_cap"] * 100
                )
            if "news" not in disabled_overlays:
                forward += (catalyst_score - 50) / 50 * (
                    SECTOR_ROTATION["catalyst_weight_cap"] * 100
                )
            if "seasonality" not in disabled_overlays:
                forward += (seasonal_item["score"] - 50) / 50 * (
                    SECTOR_ROTATION["seasonality_weight_cap"] * 100
                )
        forward = float(np.clip(forward, 0, 100))

        components = [
            float(row["momentum"]), float(row["acceleration"]),
            breadth_score, float(row["trend"]), float(row["risk_quality"]),
        ]
        agreement = max(0.0, 100 - float(np.std(components)) * 2)
        history = min(100.0, len(item["price"]) / (252 * 5) * 100)
        coverage = min(
            100.0,
            item.get("coverage_members", 0)
            / max(1, len(item.get("members", []))) * 100,
        ) if item.get("members") else (100.0 if item["source"] == "Official" else 50.0)
        confidence = float(np.clip(
            agreement * 0.45 + history * 0.35 + coverage * 0.20, 0, 100
        ))
        status = _status(current, forward, row["acceleration"])

        reasons = [
            f"{horizon} relative-momentum rank {row['momentum']:.0f}/100",
            f"Momentum acceleration rank {row['acceleration']:.0f}/100",
            f"Breadth participation {breadth_score:.0f}/100",
        ]
        if seasonal_item["stable"] and "seasonality" not in disabled_overlays:
            reasons.append(
                f"Validated {seasonal_item['target_month']} seasonality "
                f"{seasonal_item['expected']:+.1f}% excess"
            )
        reasons.extend(macro_evidence[:2])
        if news.get("positive"):
            reasons.append("Positive catalysts: " + ", ".join(news["positive"][:3]))
        risks = []
        if row["drawdown"] < -15:
            risks.append(f"Sector remains {row['drawdown']:.0f}% below its high")
        if breadth_score < 40:
            risks.append("Participation is narrow/weak")
        if row["acceleration"] < 35:
            risks.append("Momentum is decelerating")
        if item["source"] == "Synthetic":
            risks.append("Synthetic basket has current-membership survivorship bias")
        if "seasonality" in disabled_overlays:
            risks.append("Seasonality was zero-weighted after failing the out-of-sample ablation")

        forecasts.append({
            "sector": sector,
            "status": status,
            "current_strength": round(float(current), 1),
            "forward_rotation": round(forward, 1),
            "confidence": round(confidence, 1),
            "probability": round(np.clip(50 + (forward - 50) * 0.65, 25, 75), 1),
            "momentum": round(float(row["momentum"]), 1),
            "acceleration": round(float(row["acceleration"]), 1),
            "breadth": round(breadth_score, 1),
            "trend": round(float(row["trend"]), 1),
            "risk_quality": round(float(row["risk_quality"]), 1),
            "fundamental_breadth": round(float(fundamental_score), 1),
            "macro_fit": round(macro_score, 1),
            "catalyst": round(catalyst_score, 1),
            "seasonality": seasonal_item,
            "disabled_overlays": sorted(disabled_overlays),
            "source": item["source"],
            "members": item.get("coverage_members", 0),
            "reasons": reasons,
            "risks": risks or ["No major sector-specific risk signal"],
            "returns": {
                "1M": round(float(row["return_1m"]), 1),
                "3M": round(float(row["return_3m"]), 1),
                "6M": round(float(row["return_6m"]), 1),
                "12M": round(float(row["return_12m"]), 1),
            },
        })
    return sorted(
        forecasts,
        key=lambda item: (item["forward_rotation"], item["confidence"]),
        reverse=True,
    )
