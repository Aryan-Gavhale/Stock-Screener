"""
Advanced fundamental analysis.

Implements:
  - Valuation score (P/E vs sector, P/B, PEG)
  - Quality score (ROE, margins, FCF, debt)
  - Growth score (revenue, earnings, forward vs trailing)
  - Financial health (D/E, current ratio, cash)
  - CAN SLIM inspired composite
"""

import numpy as np
from config import SECTOR_PE, SECTOR_ROE


def _g(info, key, default=None):
    """Safe get that handles None / NaN."""
    v = (info or {}).get(key, default)
    if v is None:
        return default
    try:
        if np.isnan(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


# ── Valuation ──────────────────────────────────────────────────────────────────

def _valuation(info, sector):
    scores, notes = [], []
    pe = _g(info, "pe")
    spe = SECTOR_PE.get(sector, 25)
    if pe is not None and pe > 0:
        ratio = pe / spe
        if   ratio < 0.5:  s = 95
        elif ratio < 0.75: s = 78
        elif ratio < 1.0:  s = 62
        elif ratio < 1.25: s = 45
        elif ratio < 1.5:  s = 30
        else:               s = 15
        scores.append(s)
        notes.append(f"P/E {pe:.1f} vs sector {spe}")
    elif pe is not None and pe < 0:
        scores.append(8)
        notes.append("Negative earnings")

    pb = _g(info, "pb")
    if pb is not None and pb > 0:
        if   pb < 1:   s = 88
        elif pb < 2:   s = 72
        elif pb < 4:   s = 52
        elif pb < 8:   s = 32
        else:           s = 12
        scores.append(s)
        notes.append(f"P/B {pb:.2f}")

    peg = _g(info, "peg")
    if peg is not None and peg > 0:
        if   peg < 0.5: s = 92
        elif peg < 1.0: s = 78
        elif peg < 1.5: s = 58
        elif peg < 2.0: s = 38
        else:            s = 18
        scores.append(s)
        notes.append(f"PEG {peg:.2f}")

    return (float(np.mean(scores)) if scores else 50), notes


# ── Quality ────────────────────────────────────────────────────────────────────

def _quality(info, sector):
    scores, notes = [], []
    roe = _g(info, "roe")
    sroe = SECTOR_ROE.get(sector, 15)
    if roe is not None:
        rp = roe * 100
        if   rp > sroe * 1.5: s = 92
        elif rp > sroe:        s = 75
        elif rp > sroe * 0.7:  s = 55
        elif rp > 0:           s = 30
        else:                   s = 10
        scores.append(s)
        notes.append(f"ROE {rp:.1f}% (sector avg {sroe}%)")

    pm = _g(info, "profit_margin")
    if pm is not None:
        pp = pm * 100
        if   pp > 25: s = 90
        elif pp > 15: s = 72
        elif pp > 8:  s = 52
        elif pp > 0:  s = 30
        else:          s = 8
        scores.append(s)
        notes.append(f"Margin {pp:.1f}%")

    om = _g(info, "op_margin")
    if om is not None:
        op = om * 100
        if   op > 30: s = 88
        elif op > 20: s = 70
        elif op > 10: s = 50
        elif op > 0:  s = 28
        else:          s = 8
        scores.append(s)

    roa = _g(info, "roa")
    if roa is not None:
        rap = roa * 100
        if   rap > 15: s = 88
        elif rap > 8:  s = 68
        elif rap > 4:  s = 48
        elif rap > 0:  s = 28
        else:           s = 8
        scores.append(s)

    return (float(np.mean(scores)) if scores else 50), notes


# ── Growth ─────────────────────────────────────────────────────────────────────

def _growth(info):
    scores, notes = [], []
    rg = _g(info, "rev_growth")
    if rg is not None:
        rp = rg * 100
        if   rp > 25:  s = 92
        elif rp > 15:  s = 78
        elif rp > 8:   s = 62
        elif rp > 0:   s = 42
        elif rp > -10: s = 22
        else:           s = 8
        scores.append(s)
        notes.append(f"Rev growth {rp:.1f}%")

    eg = _g(info, "earn_growth")
    if eg is not None:
        ep = eg * 100
        if   ep > 30:  s = 95
        elif ep > 15:  s = 78
        elif ep > 5:   s = 62
        elif ep > 0:   s = 42
        elif ep > -15: s = 22
        else:           s = 8
        scores.append(s)
        notes.append(f"Earnings growth {ep:.1f}%")

    tpe = _g(info, "pe")
    fpe = _g(info, "forward_pe")
    if tpe and fpe and tpe > 0 and fpe > 0:
        if fpe < tpe * 0.8:
            scores.append(82)
            notes.append("Forward PE much lower — growth expected")
        elif fpe < tpe:
            scores.append(68)
            notes.append("Forward PE lower — growth expected")
        else:
            scores.append(35)
            notes.append("Forward PE higher — slowdown expected")

    return (float(np.mean(scores)) if scores else 50), notes


# ── Financial health ───────────────────────────────────────────────────────────

def _health(info):
    scores, notes = [], []
    de = _g(info, "de")
    if de is not None:
        dr = de / 100 if de > 10 else de
        if   dr < 0.1: s = 95; notes.append("Near debt-free")
        elif dr < 0.3: s = 82
        elif dr < 0.5: s = 68
        elif dr < 1.0: s = 52
        elif dr < 1.5: s = 35
        elif dr < 2.5: s = 20
        else:           s = 8;  notes.append(f"D/E {dr:.2f} — Very high debt")
        scores.append(s)

    cr = _g(info, "current_ratio")
    if cr is not None:
        if   cr > 2.0: s = 82
        elif cr > 1.5: s = 88
        elif cr > 1.0: s = 62
        elif cr > 0.7: s = 32
        else:           s = 12
        scores.append(s)

    cash = _g(info, "total_cash")
    debt = _g(info, "total_debt")
    if cash and debt and debt > 0:
        ratio = cash / debt
        if ratio > 1:     scores.append(88)
        elif ratio > 0.5: scores.append(68)
        elif ratio > 0.2: scores.append(45)
        else:              scores.append(22)

    fcf = _g(info, "fcf")
    rev = _g(info, "revenue")
    if fcf is not None:
        if fcf > 0 and rev and rev > 0:
            fm = fcf / rev * 100
            if   fm > 15: scores.append(90)
            elif fm > 8:  scores.append(72)
            elif fm > 3:  scores.append(55)
            else:          scores.append(40)
        elif fcf > 0:
            scores.append(62)
        else:
            scores.append(18)
            notes.append("Negative free cash flow")

    dy = _g(info, "div_yield", 0)
    if dy and dy > 0.02:
        notes.append(f"Div yield {dy*100:.2f}%")

    return (float(np.mean(scores)) if scores else 50), notes


# ═══════════════════════════════════════════════════════════════════════════════
#  Composite fundamental score
# ═══════════════════════════════════════════════════════════════════════════════

def fundamental_score(info: dict, sector: str) -> dict:
    if info is None:
        return {"score": 0, "quality": "No Data", "breakdown": {}, "notes": []}

    val_s, val_n = _valuation(info, sector)
    qua_s, qua_n = _quality(info, sector)
    gro_s, gro_n = _growth(info)
    hea_s, hea_n = _health(info)

    total = val_s * 0.25 + qua_s * 0.25 + gro_s * 0.25 + hea_s * 0.25

    if total >= 72:   q = "Excellent"
    elif total >= 58: q = "Good"
    elif total >= 42: q = "Average"
    elif total >= 28: q = "Weak"
    else:              q = "Poor"

    mcap = _g(info, "market_cap", 0)
    mcap_cr = mcap / 1e7 if mcap else 0

    return {
        "score": round(total, 1),
        "quality": q,
        "breakdown": {
            "Valuation":  {"s": round(val_s, 1), "n": val_n},
            "Quality":    {"s": round(qua_s, 1), "n": qua_n},
            "Growth":     {"s": round(gro_s, 1), "n": gro_n},
            "Health":     {"s": round(hea_s, 1), "n": hea_n},
        },
        "notes": val_n + qua_n + gro_n + hea_n,
        "metrics": {
            "P/E":       _g(info, "pe"),
            "P/B":       _g(info, "pb"),
            "ROE":       f"{_g(info,'roe',0)*100:.1f}%" if _g(info,"roe") else "—",
            "D/E":       _g(info, "de"),
            "Margin":    f"{_g(info,'profit_margin',0)*100:.1f}%" if _g(info,"profit_margin") else "—",
            "Rev Gr":    f"{_g(info,'rev_growth',0)*100:.1f}%" if _g(info,"rev_growth") else "—",
            "Div":       f"{_g(info,'div_yield',0)*100:.2f}%",
            "MCap Cr":   f"₹{mcap_cr:,.0f}",
        },
    }
