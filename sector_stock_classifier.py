"""
Within-sector stock leadership.

At most one Hero is emitted, and only when strict trend, liquidity, setup and
sector-confidence gates pass. Otherwise the UI must say "No confirmed Hero".
"""

from __future__ import annotations

import numpy as np

from config import SECTOR_ROTATION


PSU_BANK_TICKERS = {
    "SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS",
    "INDIANB.NS", "BANKINDIA.NS", "MAHABANK.NS", "CENTRALBK.NS",
    "UCOBANK.NS", "IOB.NS", "PSB.NS",
}


def _sector_recs(
    recs: list[dict], sector: str, official_members: list[str] | None = None
) -> list[dict]:
    if official_members:
        member_set = set(official_members)
        return [rec for rec in recs if rec.get("ticker") in member_set]
    if sector == "PSU Banks":
        return [rec for rec in recs if rec.get("ticker") in PSU_BANK_TICKERS]
    return [rec for rec in recs if rec.get("sector") == sector]


def _percentile(values: list[float], value: float) -> float:
    if not values:
        return 50.0
    return sum(v <= value for v in values) / len(values) * 100


def _stage_num(rec: dict) -> int:
    stage = rec.get("stage", (0, ""))
    return int(stage[0]) if isinstance(stage, tuple) else 0


def _avoid_reasons(rec: dict, within_rs: float) -> list[str]:
    reasons = []
    stage = _stage_num(rec)
    if stage in (3, 4):
        reasons.append(f"Stage {stage} trend")
    if rec.get("action") == "AVOID":
        reasons.append("Fails stock-level buy criteria")
    if within_rs < 30:
        reasons.append("Bottom-third relative strength within sector")
    if rec.get("avg_turnover_cr", 0) < SECTOR_ROTATION["min_turnover_cr"]:
        reasons.append("Insufficient traded-value liquidity")
    if len(rec.get("red_flags", [])) >= 2:
        reasons.append("Multiple fundamental red flags")
    if rec.get("growth_s", 50) < 30:
        reasons.append("Weak/negative growth")
    if rec.get("tech_score", 50) < 35:
        reasons.append("Technical structure is weak")
    return reasons


def classify_sector_stocks(
    recs: list[dict], sector_forecast: dict,
    official_members: list[str] | None = None,
) -> dict:
    """Return Hero, Contenders and Avoid stocks for one sector."""
    sector = sector_forecast["sector"]
    members = _sector_recs(recs, sector, official_members)
    if not members:
        return {"sector": sector, "hero": None, "contenders": [], "avoid": [],
                "coverage": 0}

    rs_values = [float(rec.get("rs", 1.0)) for rec in members]
    accelerations = []
    for rec in members:
        returns = rec.get("returns", {})
        accelerations.append(
            float(returns.get("1m", 0)) - float(returns.get("3m", 0)) / 3
        )

    enriched = []
    for rec, acceleration in zip(members, accelerations):
        item = dict(rec)
        within_rs = _percentile(rs_values, float(rec.get("rs", 1.0)))
        accel_pct = _percentile(accelerations, acceleration)
        setup_score = float(rec.get("setup_quality", 50))
        leadership = (
            float(rec.get("composite", 0)) * 0.28
            + within_rs * 0.20
            + accel_pct * 0.10
            + float(rec.get("tech_score", 0)) * 0.12
            + float(rec.get("growth_s", 50)) * 0.10
            + float(rec.get("quality_s", 50)) * 0.10
            + setup_score * 0.10
        )
        reasons = [
            f"Sector-relative strength {within_rs:.0f}th percentile",
            f"Stock composite {rec.get('composite', 0):.0f}/100",
            f"{rec.get('setup', 'NONE')} setup quality {setup_score:.0f}/100",
        ]
        if rec.get("earnings_accel"):
            reasons.append("Earnings acceleration is positive")
        if rec.get("avg_turnover_cr", 0) >= SECTOR_ROTATION["min_turnover_cr"]:
            reasons.append(
                f"Liquid: ₹{rec.get('avg_turnover_cr', 0):.1f} Cr median daily turnover"
            )
        avoid_reasons = _avoid_reasons(rec, within_rs)
        item.update({
            "sector_rs_pct": round(within_rs, 1),
            "sector_accel_pct": round(accel_pct, 1),
            "leadership_score": round(float(np.clip(leadership, 0, 100)), 1),
            "leadership_reasons": reasons,
            "avoid_reasons": avoid_reasons,
        })
        enriched.append(item)

    valid = sorted(
        [item for item in enriched if not item["avoid_reasons"]],
        key=lambda item: item["leadership_score"],
        reverse=True,
    )
    avoided = sorted(
        [item for item in enriched if item["avoid_reasons"]],
        key=lambda item: item["leadership_score"],
    )

    hero = None
    if valid:
        best = valid[0]
        hero_ok = (
            best["leadership_score"] >= SECTOR_ROTATION["hero_min_score"]
            and best["sector_rs_pct"] >= SECTOR_ROTATION["hero_min_sector_rs"]
            and _stage_num(best) == 2
            and best.get("setup") in ("BREAKOUT", "PULLBACK", "BASE", "IN-TREND")
            and sector_forecast.get("forward_rotation", 0) >= 56
            and sector_forecast.get("confidence", 0) >= 55
            and not best.get("red_flags")
        )
        if hero_ok:
            hero = best
            valid = valid[1:]

    return {
        "sector": sector,
        "hero": hero,
        "contenders": valid[:5],
        "avoid": avoided[:8],
        "coverage": len(members),
    }


def classify_all_sectors(
    recs: list[dict], forecasts: list[dict], sector_data: dict | None = None
) -> dict:
    return {
        forecast["sector"]: classify_sector_stocks(
            recs, forecast,
            ((sector_data or {}).get(forecast["sector"]) or {}).get("members"),
        )
        for forecast in forecasts
    }
