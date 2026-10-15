"""
Scoring engine (v5) — return-oriented, setup-driven.

Composite blends: Trend/Momentum + Quality + Growth + Value (GARP),
then nudges for entry-setup quality, relative strength, earnings acceleration,
and penalises red flags. Classifies each name into a Core or Burst bucket and
produces an ACTIONABLE recommendation tied to a real entry setup — with
ATR-based stop, target, and risk-sized position from risk_engine.
"""

import numpy as np
from config import RISK_PROFILES, BURST_CRITERIA, DIP_CRITERIA, SECTOR_ROTATION
from risk_engine import trade_levels, position_size, trailing_stop


# ═══════════════════════════════════════════════════════════════════════════════
#  Quality-on-sale DIP candidate (post-crash accumulation)
# ═══════════════════════════════════════════════════════════════════════════════

def _last(series, i=-1):
    try:
        v = series.dropna().iloc[i]
        return float(v)
    except Exception:
        return None


def dip_candidate(tech: dict, fund: dict, info: dict) -> dict:
    """
    Flag a stock that has fallen hard but whose fundamentals are intact AND
    which is starting to stabilise (not a falling knife). Returns a staggered
    tranche accumulation plan and a stop below the capitulation low.

    This is a deliberately MEAN-REVERSION screen (distinct from the momentum
    lane) — best used when the market accumulation_zone is Forming/Confirmed.
    """
    dc = DIP_CRITERIA
    out = {"is_dip": False, "dip_score": 0, "signals": [], "tranches": [],
           "fell_pct": tech.get("from_high", 0)}

    df = tech.get("df")
    price = tech.get("price", 0)
    if df is None or price <= 0:
        return out

    fell = tech.get("from_high", 0)
    fell_hard = fell <= dc["min_fall_from_high"]

    # Quality intact?
    de = (info or {}).get("de")
    if isinstance(de, (int, float)) and de > 10:
        de = de / 100
    quality_ok = (
        fund.get("quality_s", 0) >= dc["min_quality"]
        and (not isinstance(de, (int, float)) or de <= dc["max_de"])
        and len(fund.get("red_flags", [])) < 2
        and "Loss-making (negative earnings)" not in fund.get("red_flags", [])
    )

    # Stabilisation signals (avoid catching a falling knife)
    sigs = []
    c = df["Close"]
    cap_low = float(c.tail(60).min())

    rsi = df.get("RSI")
    if rsi is not None:
        rsi_now = _last(rsi)
        rsi_min = float(rsi.dropna().tail(15).min()) if len(rsi.dropna()) else None
        rsi_prev = _last(rsi, -3)
        if rsi_now and rsi_min is not None and rsi_prev and rsi_min < 40 and rsi_now > rsi_prev:
            sigs.append("RSI turning up from oversold")

    mh = df.get("MACD_H")
    if mh is not None and len(mh.dropna()) >= 5:
        if _last(mh) is not None and _last(mh, -5) is not None and _last(mh) > _last(mh, -5):
            sigs.append("MACD histogram improving")

    e21 = _last(df.get("EMA21")) if df.get("EMA21") is not None else None
    if e21 and price > e21:
        sigs.append("Reclaimed 21-EMA")

    low20 = float(c.tail(20).min())
    if price > low20 * 1.01:
        sigs.append("Holding above recent swing low")

    volr = df.get("VOL_RATIO")
    if volr is not None:
        vr = _last(volr)
        pc = (price / c.iloc[-5] - 1) if len(c) >= 6 else 0
        if vr and ((pc > 0 and vr > 1.1) or (pc < 0 and vr < 0.9)):
            sigs.append("Volume pattern supportive (buying > selling)")

    stabilizing = len(sigs) >= 2
    out["signals"] = sigs
    out["cap_low"] = round(cap_low, 2)

    if not (fell_hard and quality_ok and stabilizing):
        # still expose why not, but not a dip buy
        out["quality_ok"] = quality_ok
        return out

    # Dip attractiveness score
    depth_pts = min(25, (abs(fell) - abs(dc["min_fall_from_high"])) * 0.6 + 10)
    score = (fund.get("quality_s", 50) * 0.34
             + fund.get("value_s", 50) * 0.20
             + min(30, len(sigs) * 9)
             + depth_pts)
    eg = (info or {}).get("earn_growth")
    if isinstance(eg, (int, float)) and eg > 0:
        score += 5
    score = round(min(100, score), 1)

    # Staggered tranche plan: start now, add lower toward the capitulation low
    n = dc["tranches"]
    stop = round(cap_low * (1 - dc["stop_below_low_pct"] / 100), 2)
    levels = []
    lo = max(cap_low, price * 0.88)
    for k in range(n):
        lvl = price - (price - lo) * (k / (n - 1)) if n > 1 else price
        levels.append(round(lvl, 2))
    out.update({
        "is_dip": True, "dip_score": score, "quality_ok": True,
        "stop": stop, "tranches": levels, "cap_low": round(cap_low, 2),
        "note": f"Fell {fell:.0f}% from high; quality intact; {len(sigs)} stabilisation signals. "
                f"Buy in {n} tranches, stop ₹{stop:.0f} (below capitulation low).",
    })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Composite
# ═══════════════════════════════════════════════════════════════════════════════

def composite(tech: dict, fund: dict, profile: str = "Moderate") -> float:
    p = RISK_PROFILES.get(profile, RISK_PROFILES["Moderate"])
    ts = tech.get("score", 0)
    qs = fund.get("quality_s", 50)
    gs = fund.get("growth_s", 50)
    vs = fund.get("value_s", 50)

    raw = (ts * p["trend_w"] + qs * p["quality_w"] +
           gs * p["growth_w"] + vs * p["value_w"])

    # Entry-setup quality nudge (max ±3)
    setup_q = tech.get("setup", {}).get("quality", 50)
    raw += (setup_q - 50) / 50 * 3

    # Relative strength bonus (capped)
    rs_pct = tech.get("rs_pct", 50)
    if rs_pct >= 80:   raw *= 1.03
    elif rs_pct >= 65: raw *= 1.015

    # Earnings acceleration bonus
    if fund.get("earnings_accel"):
        raw += 2

    # Stage multiplier — gentle
    stage_num = tech.get("stage", (0,))[0] if isinstance(tech.get("stage"), tuple) else 0
    raw *= {2: 1.04, 1: 0.98, 3: 0.93, 4: 0.85, 0: 1.0}.get(stage_num, 1.0)

    # Trend consistency (max ±1)
    raw += (tech.get("trend_consistency", 50) - 50) / 50 * 1.0

    # Red-flag penalty (−4 each, cap −12)
    raw -= min(12, 4 * len(fund.get("red_flags", [])))

    return round(min(100, max(0, raw)), 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Bucket classification: Core (compounder) vs Burst (multibagger hunt)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_bucket(tech: dict, fund: dict, info: dict) -> str:
    bc = BURST_CRITERIA
    mcap_cr = fund.get("mcap_cr", 0) or 0
    if mcap_cr == 0 or mcap_cr > bc["max_mcap_cr"]:
        return "Core"

    checks = 0
    rg = (info or {}).get("rev_growth")
    eg = (info or {}).get("earn_growth")
    de = (info or {}).get("de")
    if isinstance(de, (int, float)) and de > 10:
        de = de / 100
    if isinstance(rg, (int, float)) and rg >= bc["min_rev_growth"]: checks += 1
    if isinstance(eg, (int, float)) and eg >= bc["min_earn_growth"]: checks += 1
    if tech.get("rs_pct", 0) >= bc["min_rs_pct"]: checks += 1
    if tech.get("from_high", -100) >= bc["max_from_high"]: checks += 1
    if isinstance(de, (int, float)) and de <= bc["max_de"]: checks += 1
    if fund.get("earnings_accel"): checks += 1

    return "Burst" if checks >= 4 else "Core"


# ═══════════════════════════════════════════════════════════════════════════════
#  Risk assessment
# ═══════════════════════════════════════════════════════════════════════════════

def assess_risk(tech: dict, info: dict) -> dict:
    score, factors = 50, []
    vol = tech.get("volatility", 25)
    if vol > 45:   score += 18; factors.append(f"Very high volatility ({vol:.0f}%)")
    elif vol > 35: score += 10; factors.append(f"High volatility ({vol:.0f}%)")
    elif vol < 15: score -= 10; factors.append(f"Low volatility ({vol:.0f}%)")

    beta = (info or {}).get("beta", 1.0) or 1.0
    if isinstance(beta, str):
        try: beta = float(beta)
        except: beta = 1.0
    if beta > 1.5:   score += 12; factors.append(f"High beta {beta:.2f}")
    elif beta < 0.7: score -= 8;  factors.append(f"Defensive beta {beta:.2f}")

    de = (info or {}).get("de")
    if isinstance(de, str):
        try: de = float(de)
        except: de = None
    if de is not None:
        dr = de / 100 if de > 10 else de
        if dr > 2:    score += 12; factors.append(f"High leverage D/E {dr:.2f}")
        elif dr < 0.3: score -= 5; factors.append("Low debt")

    stage = tech.get("stage", (0,))[0] if isinstance(tech.get("stage"), tuple) else 0
    if stage == 4:   score += 15; factors.append("Stage 4 downtrend")
    elif stage == 3: score += 8;  factors.append("Stage 3 distribution")

    for f in tech.get("_extra_flags", []):
        factors.append(f)

    score = max(0, min(100, score))
    level = "High" if score >= 65 else ("Medium" if score >= 40 else "Low")
    return {"score": score, "level": level, "factors": factors}


# ═══════════════════════════════════════════════════════════════════════════════
#  Holding & exit guidance (text)
# ═══════════════════════════════════════════════════════════════════════════════

def holding_guidance(tech, fund, bucket):
    stage = tech.get("stage", (0,))[0] if isinstance(tech.get("stage"), tuple) else 0
    setup = tech.get("setup", {}).get("setup", "NONE")
    if bucket == "Burst":
        return ("Swing / position (weeks-months)",
                "Small speculative position — earnings-driven re-rating play",
                "Honor stop strictly; trail hard once +15%. Book partial into spikes.")
    if stage == 2 and setup in ("BREAKOUT", "PULLBACK", "IN-TREND", "BASE"):
        return ("Position / long term (months+)",
                "Confirmed Stage-2 leader — ride the trend",
                "Trail below rising 50-DMA / 3x ATR. Exit on Stage-4 break.")
    if fund.get("score", 50) >= 60:
        return ("Medium term (1-6 mo)",
                "Quality name — accumulate on strength/pullbacks",
                "Stop under recent swing low; reassess on trend break.")
    return ("Wait for setup", "No clean entry trigger yet",
            "Add to watchlist; act on breakout or pullback to support.")


def exit_signal(position: dict, tech: dict) -> dict:
    """
    For a HELD position, decide HOLD / TRIM / EXIT.
    position: {entry, qty, stop (initial), high_since_entry}
    tech: fresh technical_score() output.
    """
    price = tech.get("price", 0)
    entry = position.get("entry", 0)
    init_stop = position.get("stop", 0)
    high_since = max(position.get("high_since_entry", entry), price)
    atr = tech.get("atr") or price * 0.02
    stage = tech.get("stage", (0,))[0] if isinstance(tech.get("stage"), tuple) else 0

    trail = trailing_stop(entry, price, high_since, atr)
    current_stop = init_stop
    if trail["active"] and trail["trail_stop"]:
        current_stop = max(init_stop, trail["trail_stop"])

    gain = (price / entry - 1) * 100 if entry else 0

    # Hard exits
    if price <= current_stop:
        return {"action": "EXIT", "reason": f"Stop hit (₹{current_stop:.0f})",
                "current_stop": round(current_stop, 2), "gain_pct": round(gain, 1),
                "high_since": round(high_since, 2)}
    if stage == 4:
        return {"action": "EXIT", "reason": "Trend broke — Stage 4 downtrend",
                "current_stop": round(current_stop, 2), "gain_pct": round(gain, 1),
                "high_since": round(high_since, 2)}

    # Warnings / trims
    s50 = None
    df = tech.get("df")
    if df is not None and "SMA50" in df.columns:
        try: s50 = df["SMA50"].dropna().iloc[-1]
        except: s50 = None
    if s50 and price < s50 and stage == 3:
        return {"action": "TRIM", "reason": "Lost 50-DMA in distribution — reduce",
                "current_stop": round(current_stop, 2), "gain_pct": round(gain, 1),
                "high_since": round(high_since, 2)}

    note = trail["note"] if trail["active"] else "Hold — trend intact"
    return {"action": "HOLD", "reason": note,
            "current_stop": round(current_stop, 2), "gain_pct": round(gain, 1),
            "high_since": round(high_since, 2), "trailing": trail["active"]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Full recommendation
# ═══════════════════════════════════════════════════════════════════════════════

def recommend(ticker, name, sector, cap, tech, fund, info,
              profile, capital, deploy_mult=1.0, stop_tighten=1.0,
              sector_signal: dict = None):
    p = RISK_PROFILES.get(profile, RISK_PROFILES["Moderate"])
    comp = composite(tech, fund, profile)
    sector_tilt = 0.0
    if sector_signal and sector_signal.get("tilt_enabled", True):
        sector_tilt = (
            (sector_signal.get("forward_rotation", 50) - 50) / 50
            * SECTOR_ROTATION["sector_tilt_cap"] * 100
            * sector_signal.get("confidence", 50) / 100
        )
        comp = round(min(100, max(0, comp + sector_tilt)), 1)
    risk = assess_risk(tech, info)
    bucket = classify_bucket(tech, fund, info)
    setup = tech.get("setup", {"setup": "NONE", "quality": 0, "note": ""})
    stage_num = tech.get("stage", (0,))[0] if isinstance(tech.get("stage"), tuple) else 0

    price = tech.get("price", 0)
    atr = tech.get("atr")
    # In high crash-risk regimes, tighten stops (stop_tighten < 1 shrinks the stop distance)
    p_eff = dict(p)
    p_eff["core_stop_pct"] = p["core_stop_pct"] * stop_tighten
    levels = trade_levels(price, atr, p_eff)
    pos = position_size(capital, price, levels.get("stop"), p, deploy_mult) if levels else {}

    # ── Action logic: return-oriented + setup-driven ────────────────────────────
    severe_flags = len(fund.get("red_flags", [])) >= 2
    actionable = setup["setup"] in ("BREAKOUT", "PULLBACK")
    in_trend = setup["setup"] in ("BREAKOUT", "PULLBACK", "IN-TREND", "BASE")

    if stage_num == 4 or comp < 35 or severe_flags:
        action = "AVOID"
    elif comp >= 66 and actionable and risk["level"] != "High" and not fund.get("red_flags"):
        action = "STRONG BUY"
    elif comp >= 56 and in_trend:
        action = "BUY"
    elif comp >= 50 and setup["setup"] in ("EXTENDED", "BASE"):
        action = "WATCH"
    elif comp >= 46:
        action = "HOLD"
    else:
        action = "AVOID"

    hold, hold_reason, exit_strat = holding_guidance(tech, fund, bucket)
    dip = dip_candidate(tech, fund, info)

    return {
        "ticker": ticker, "name": name, "sector": sector, "cap": cap,
        "bucket": bucket, "action": action, "composite": comp,
        "sector_tilt": round(sector_tilt, 1),
        "sector_status": (sector_signal or {}).get("status", "Unrated"),
        "sector_forward": (sector_signal or {}).get("forward_rotation", 50),
        "sector_confidence": (sector_signal or {}).get("confidence", 0),
        "is_dip": dip.get("is_dip", False), "dip_score": dip.get("dip_score", 0),
        "dip_stop": dip.get("stop"), "dip_tranches": dip.get("tranches", []),
        "dip_signals": dip.get("signals", []), "dip_note": dip.get("note", ""),
        "dip_cap_low": dip.get("cap_low"),
        "tech_score": tech.get("score", 0), "fund_score": fund.get("score", 0),
        "quality_s": fund.get("quality_s", 50), "growth_s": fund.get("growth_s", 50),
        "value_s": fund.get("value_s", 50),
        "momentum": tech.get("momentum", 50),
        "trend_consistency": tech.get("trend_consistency", 50),
        "stage": tech.get("stage", (0, "N/A")),
        "trend": tech.get("trend", "—"),
        "setup": setup["setup"], "setup_quality": setup["quality"], "setup_note": setup["note"],
        "rs": tech.get("rs", 1.0), "rs_pct": tech.get("rs_pct", 50),
        "earnings_accel": fund.get("earnings_accel", False),
        "red_flags": fund.get("red_flags", []),
        "risk_level": risk["level"], "risk_score": risk["score"], "risk_factors": risk["factors"],
        # trade plan
        "entry": levels.get("entry", price), "stop": levels.get("stop"),
        "stop_pct": levels.get("stop_pct"), "target": levels.get("target"),
        "target_pct": levels.get("target_pct"), "rr": levels.get("rr"),
        "qty": pos.get("qty", 0), "alloc_amt": pos.get("amount", 0),
        "alloc_pct": pos.get("weight_pct", 0), "risk_amt": pos.get("risk_amount", 0),
        "risk_pct": pos.get("risk_pct", 0),
        # guidance
        "holding": hold, "hold_reason": hold_reason, "exit": exit_strat,
        # display
        "price": price, "from_high": tech.get("from_high", 0), "from_low": tech.get("from_low", 0),
        "volatility": tech.get("volatility", 0), "atr_pct": tech.get("atr_pct"),
        "avg_turnover_cr": tech.get("avg_turnover_cr", 0),
        "returns": tech.get("returns", {}), "indicators": tech.get("indicators", {}),
        "fund_breakdown": fund.get("breakdown", {}), "fund_notes": fund.get("notes", []),
        "metrics": fund.get("metrics", {}),
    }


def rank(recs):
    return sorted(recs, key=lambda r: r["composite"], reverse=True)
