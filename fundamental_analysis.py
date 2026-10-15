"""
Fundamental analysis — GARP + earnings-acceleration (v5).

For a momentum/growth strategy we care about, in order:
  1. GROWTH & earnings acceleration  — the igniter of re-ratings / "bursts"
  2. QUALITY (durability)            — ROE, margins, low debt, cash generation
  3. VALUE (GARP)                    — reasonable price FOR the growth, not deep value

Deep-value cheapness is intentionally NOT heavily rewarded (in a momentum book
it's usually a value trap). Returns separate pillar scores the composite blends.
"""

import numpy as np
from config import SECTOR_PE, SECTOR_ROE


def _g(info, key, default=None):
    """Safe get: handles None / NaN / string values from Yahoo."""
    v = (info or {}).get(key, default)
    if v is None:
        return default
    if isinstance(v, str):
        try:
            v = float(v)
        except (ValueError, TypeError):
            return default
    try:
        if np.isnan(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


# ── Growth & earnings acceleration ───────────────────────────────────────────────

def _growth(info):
    scores, notes = [], []
    accel = False

    rg = _g(info, "rev_growth")
    if rg is not None:
        rp = rg * 100
        if   rp > 25:  s = 95
        elif rp > 15:  s = 80
        elif rp > 8:   s = 62
        elif rp > 0:   s = 42
        elif rp > -10: s = 22
        else:          s = 8
        scores.append(s); notes.append(f"Rev growth {rp:.1f}%")

    eg = _g(info, "earn_growth")
    if eg is not None:
        ep = eg * 100
        if   ep > 40:  s = 97
        elif ep > 25:  s = 88
        elif ep > 15:  s = 74
        elif ep > 5:   s = 58
        elif ep > 0:   s = 42
        elif ep > -15: s = 20
        else:          s = 6
        scores.append(s); notes.append(f"Earnings growth {ep:.1f}%")

    # Acceleration proxy: market expects higher earnings (forward PE < trailing PE)
    tpe, fpe = _g(info, "pe"), _g(info, "forward_pe")
    if tpe and fpe and tpe > 0 and fpe > 0:
        if fpe < tpe * 0.75:
            scores.append(90); notes.append("Fwd P/E << trailing — strong earnings acceleration"); accel = True
        elif fpe < tpe:
            scores.append(70); notes.append("Fwd P/E < trailing — earnings accelerating"); accel = True
        else:
            scores.append(35); notes.append("Fwd P/E ≥ trailing — earnings slowing")

    return (float(np.mean(scores)) if scores else 50), notes, accel


# ── Quality / durability ─────────────────────────────────────────────────────────

def _quality(info, sector):
    scores, notes = [], []

    roe = _g(info, "roe")
    sroe = SECTOR_ROE.get(sector, 15)
    if roe is not None:
        rp = roe * 100
        if   rp > sroe * 1.5: s = 92
        elif rp > sroe:       s = 76
        elif rp > sroe * 0.7: s = 55
        elif rp > 0:          s = 30
        else:                 s = 10
        scores.append(s); notes.append(f"ROE {rp:.1f}% (sector {sroe}%)")

    pm = _g(info, "profit_margin")
    if pm is not None:
        pp = pm * 100
        s = 90 if pp > 25 else 72 if pp > 15 else 52 if pp > 8 else 30 if pp > 0 else 8
        scores.append(s); notes.append(f"Net margin {pp:.1f}%")

    om = _g(info, "op_margin")
    if om is not None:
        op = om * 100
        s = 88 if op > 30 else 70 if op > 20 else 50 if op > 10 else 28 if op > 0 else 8
        scores.append(s)

    roa = _g(info, "roa")
    if roa is not None:
        rap = roa * 100
        s = 88 if rap > 15 else 68 if rap > 8 else 48 if rap > 4 else 28 if rap > 0 else 8
        scores.append(s)

    de = _g(info, "de")
    if de is not None:
        dr = de / 100 if de > 10 else de
        if   dr < 0.1: s = 92; notes.append("Near debt-free")
        elif dr < 0.3: s = 82
        elif dr < 0.5: s = 68
        elif dr < 1.0: s = 52
        elif dr < 1.5: s = 35
        elif dr < 2.5: s = 20
        else:          s = 8; notes.append(f"High debt D/E {dr:.2f}")
        scores.append(s)

    fcf, rev = _g(info, "fcf"), _g(info, "revenue")
    if fcf is not None:
        if fcf > 0 and rev and rev > 0:
            fm = fcf / rev * 100
            s = 90 if fm > 15 else 72 if fm > 8 else 55 if fm > 3 else 42
            scores.append(s)
        elif fcf > 0:
            scores.append(60)
        else:
            scores.append(20); notes.append("Negative FCF")

    return (float(np.mean(scores)) if scores else 50), notes


# ── Value (GARP) ───────────────────────────────────────────────────────────────────

def _value(info, sector, growth_pct):
    """Reward reasonable price FOR the growth. Don't over-reward dirt-cheap."""
    scores, notes = [], []

    peg = _g(info, "peg")
    if peg is not None and peg > 0:
        if   peg < 1.0: s = 90
        elif peg < 1.5: s = 74
        elif peg < 2.0: s = 55
        elif peg < 3.0: s = 35
        else:           s = 18
        scores.append(s); notes.append(f"PEG {peg:.2f}")

    pe = _g(info, "pe")
    spe = SECTOR_PE.get(sector, 25)
    if pe is not None and pe > 0:
        ratio = pe / spe
        # GARP: near/slightly-below sector is ideal; ultra-cheap only mild (trap risk)
        if   ratio < 0.4:  s = 62
        elif ratio < 0.7:  s = 82
        elif ratio < 1.0:  s = 72
        elif ratio < 1.3:  s = 52
        elif ratio < 1.7:  s = 34
        else:              s = 16
        scores.append(s); notes.append(f"P/E {pe:.1f} vs sector {spe}")
    elif pe is not None and pe < 0:
        scores.append(12); notes.append("Negative earnings")

    pb = _g(info, "pb")
    if pb is not None and pb > 0:
        s = 80 if pb < 2 else 60 if pb < 4 else 40 if pb < 8 else 18
        scores.append(s); notes.append(f"P/B {pb:.2f}")

    return (float(np.mean(scores)) if scores else 50), notes


# ── Red flags ──────────────────────────────────────────────────────────────────────

def _red_flags(info):
    flags = []
    pe = _g(info, "pe")
    if pe is not None and pe < 0:
        flags.append("Loss-making (negative earnings)")
    de = _g(info, "de")
    if de is not None:
        dr = de / 100 if de > 10 else de
        if dr > 2.5:
            flags.append(f"Very high debt (D/E {dr:.1f})")
    rg = _g(info, "rev_growth")
    if rg is not None and rg < -0.1:
        flags.append("Revenue declining")
    fcf = _g(info, "fcf")
    if fcf is not None and fcf < 0:
        flags.append("Burning cash (negative FCF)")
    return flags


# ═══════════════════════════════════════════════════════════════════════════════
#  Composite
# ═══════════════════════════════════════════════════════════════════════════════

def fundamental_score(info: dict, sector: str) -> dict:
    if info is None:
        return {"score": 50, "quality_s": 50, "growth_s": 50, "value_s": 50,
                "quality": "No Data", "breakdown": {}, "notes": [],
                "earnings_accel": False, "red_flags": [], "metrics": {}}

    gro_s, gro_n, accel = _growth(info)
    qua_s, qua_n = _quality(info, sector)
    val_s, val_n = _value(info, sector, gro_s)
    flags = _red_flags(info)

    # Overall fundamental blend (growth-tilted for this strategy)
    total = gro_s * 0.40 + qua_s * 0.35 + val_s * 0.25

    if total >= 72:   q = "Excellent"
    elif total >= 58: q = "Good"
    elif total >= 42: q = "Average"
    elif total >= 28: q = "Weak"
    else:             q = "Poor"

    mcap = _g(info, "market_cap", 0)
    mcap_cr = mcap / 1e7 if mcap else 0

    return {
        "score": round(total, 1),
        "quality_s": round(qua_s, 1),
        "growth_s": round(gro_s, 1),
        "value_s": round(val_s, 1),
        "quality": q,
        "earnings_accel": accel,
        "red_flags": flags,
        "breakdown": {
            "Growth":    {"s": round(gro_s, 1), "n": gro_n},
            "Quality":   {"s": round(qua_s, 1), "n": qua_n},
            "Value(GARP)": {"s": round(val_s, 1), "n": val_n},
        },
        "notes": gro_n + qua_n + val_n,
        "metrics": {
            "P/E":     _g(info, "pe"),
            "Fwd P/E": _g(info, "forward_pe"),
            "PEG":     _g(info, "peg"),
            "P/B":     _g(info, "pb"),
            "ROE":     f"{_g(info,'roe',0)*100:.1f}%" if _g(info,"roe") else "—",
            "D/E":     _g(info, "de"),
            "Rev Gr":  f"{_g(info,'rev_growth',0)*100:.1f}%" if _g(info,"rev_growth") else "—",
            "Earn Gr": f"{_g(info,'earn_growth',0)*100:.1f}%" if _g(info,"earn_growth") else "—",
            "MCap Cr": f"₹{mcap_cr:,.0f}",
        },
        "mcap_cr": round(mcap_cr),
    }
