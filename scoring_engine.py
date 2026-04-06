"""
Multi-factor scoring engine.

Combines technical, fundamental, momentum, and quality signals
into a final composite score with risk assessment, position sizing,
and holding period recommendation.
"""

import numpy as np
from config import RISK_PROFILES

# ═══════════════════════════════════════════════════════════════════════════════
#  Composite score
# ═══════════════════════════════════════════════════════════════════════════════

def composite(tech: dict, fund: dict, profile: str = "Moderate") -> float:
    p = RISK_PROFILES.get(profile, RISK_PROFILES["Moderate"])
    ts = tech.get("score", 0)
    fs = fund.get("score", 0)
    ms = tech.get("momentum", 50)

    # Quality: average of fundamental quality sub-scores
    fb = fund.get("breakdown", {})
    qs = fb.get("Quality", {}).get("s", 50)

    raw = (
        ts * p["tech_w"] +
        fs * p["fund_w"] +
        ms * p["momentum_w"] +
        qs * p["quality_w"]
    )

    # Stage multiplier — gentle nudge, don't crush scores
    stage_num = tech.get("stage", (0,))[0] if isinstance(tech.get("stage"), tuple) else 0
    stage_mult = {2: 1.06, 1: 0.96, 3: 0.90, 4: 0.78, 0: 1.0}
    raw *= stage_mult.get(stage_num, 1.0)

    # RS bonus: top 25% RS gets small boost
    rs_pct = tech.get("rs_pct", 50)
    if rs_pct >= 80:
        raw *= 1.05
    elif rs_pct >= 60:
        raw *= 1.02

    return round(min(100, max(0, raw)), 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Risk assessment
# ═══════════════════════════════════════════════════════════════════════════════

def assess_risk(tech: dict, info: dict) -> dict:
    score = 50
    factors = []

    vol = tech.get("volatility", 25)
    if vol > 45:
        score += 18; factors.append(f"Very high volatility ({vol:.0f}%)")
    elif vol > 35:
        score += 10; factors.append(f"High volatility ({vol:.0f}%)")
    elif vol < 15:
        score -= 10; factors.append(f"Low volatility ({vol:.0f}%)")

    beta = (info or {}).get("beta", 1.0) or 1.0
    if beta > 1.5:
        score += 12; factors.append(f"High beta {beta:.2f}")
    elif beta < 0.7:
        score -= 8; factors.append(f"Defensive beta {beta:.2f}")

    de = (info or {}).get("de")
    if de is not None:
        dr = de / 100 if de > 10 else de
        if dr > 2:
            score += 12; factors.append(f"High leverage D/E {dr:.2f}")
        elif dr < 0.3:
            score -= 5; factors.append("Low debt")

    mcap = (info or {}).get("market_cap", 0) or 0
    if mcap / 1e7 < 5000:
        score += 10; factors.append("Small cap risk")
    elif mcap / 1e7 > 50000:
        score -= 5; factors.append("Large cap stability")

    from_high = tech.get("from_high", -10)
    if from_high > -3:
        score += 8; factors.append("Near 52W high — limited upside")

    stage = tech.get("stage", (0,))[0] if isinstance(tech.get("stage"), tuple) else 0
    if stage == 4:
        score += 15; factors.append("Stage 4 downtrend")
    elif stage == 3:
        score += 8; factors.append("Stage 3 distribution")

    score = max(0, min(100, score))
    level = "High" if score >= 65 else ("Medium" if score >= 40 else "Low")
    return {"score": score, "level": level, "factors": factors}


# ═══════════════════════════════════════════════════════════════════════════════
#  Holding period
# ═══════════════════════════════════════════════════════════════════════════════

def holding_period(tech_s, fund_s, stage, risk_level):
    strong_t = tech_s >= 65
    strong_f = fund_s >= 65
    st = stage[0] if isinstance(stage, tuple) else 0

    if strong_t and strong_f and st == 2:
        return "Long Term (6-24 mo)", "Strong on all fronts in Stage 2 uptrend", "Trail 18% SL from peak, review quarterly"
    if strong_f and not strong_t:
        return "Medium Term (1-6 mo)", "Fundamentals strong, technicals lagging — accumulate dips", "Buy in 2-3 tranches, exit if fundamentals deteriorate"
    if strong_t and not strong_f:
        return "Short Term (1-4 wk)", "Momentum play — technicals strong but fundamentals weak", "Strict 8% SL, book profits at 15-20%"
    if st == 2 and fund_s >= 50:
        return "Medium Term (2-8 wk)", "Trend following in Stage 2", "Trail 12% SL"
    if strong_f and st in (1, 3):
        return "Medium-Long (3-12 mo)", "Deep value — waiting for re-rating", "Accumulate slowly, patience required"
    return "Wait", "No strong conviction from either side", "Wait for better entry or improved signals"


# ═══════════════════════════════════════════════════════════════════════════════
#  Position sizing
# ═══════════════════════════════════════════════════════════════════════════════

def position_size(comp_score, risk_level, profile, portfolio_val):
    p = RISK_PROFILES.get(profile, RISK_PROFILES["Moderate"])
    mx = p["max_per_stock"]

    if   comp_score >= 80: base = mx
    elif comp_score >= 70: base = mx * 0.85
    elif comp_score >= 60: base = mx * 0.70
    elif comp_score >= 50: base = mx * 0.50
    else:                   base = mx * 0.25

    risk_mult = {"Low": 1.15, "Medium": 1.0, "High": 0.55}
    alloc = min(mx, base * risk_mult.get(risk_level, 1.0))
    return {"pct": round(alloc * 100, 2), "amount": round(portfolio_val * alloc)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Full recommendation for one stock
# ═══════════════════════════════════════════════════════════════════════════════

def recommend(ticker, name, sector, cap, tech, fund, info, profile, portfolio_val):
    comp = composite(tech, fund, profile)
    risk = assess_risk(tech, info)
    hp, hp_reason, exit_strat = holding_period(
        tech["score"], fund["score"], tech.get("stage", (0,)), risk["level"]
    )
    pos = position_size(comp, risk["level"], profile, portfolio_val)

    if   comp >= 72 and risk["level"] != "High": action = "STRONG BUY"
    elif comp >= 62:                              action = "BUY"
    elif comp >= 52 and risk["level"] != "High":  action = "ACCUMULATE"
    elif comp >= 42:                              action = "HOLD"
    elif comp >= 32:                              action = "UNDERWEIGHT"
    else:                                         action = "AVOID"

    return {
        "ticker": ticker, "name": name, "sector": sector, "cap": cap,
        "action": action, "composite": comp,
        "tech_score": tech["score"], "fund_score": fund["score"],
        "momentum": tech.get("momentum", 50),
        "stage": tech.get("stage", (0, "N/A")),
        "trend": tech.get("trend", "—"),
        "rs": tech.get("rs", 1.0), "rs_pct": tech.get("rs_pct", 50),
        "risk_level": risk["level"], "risk_score": risk["score"],
        "risk_factors": risk["factors"],
        "holding": hp, "hold_reason": hp_reason, "exit": exit_strat,
        "alloc_pct": pos["pct"], "alloc_amt": pos["amount"],
        "price": tech.get("price", 0),
        "from_high": tech.get("from_high", 0),
        "from_low": tech.get("from_low", 0),
        "volatility": tech.get("volatility", 0),
        "returns": tech.get("returns", {}),
        "indicators": tech.get("indicators", {}),
        "fund_breakdown": fund.get("breakdown", {}),
        "fund_notes": fund.get("notes", []),
        "metrics": fund.get("metrics", {}),
    }


def rank(recs):
    return sorted(recs, key=lambda r: r["composite"], reverse=True)
