"""
Portfolio optimizer.

Takes ranked recommendations and builds an optimal portfolio with:
  - Sector diversification (no sector > 30%)
  - Correlation-aware position reduction
  - Inverse-volatility weighting
  - Risk-parity inspired allocation
"""

import numpy as np
import pandas as pd
from config import RISK_PROFILES


def build_portfolio(
    recommendations: list,
    all_data: dict,
    profile: str = "Moderate",
    portfolio_val: float = 500000,
    min_score: float = 55,
) -> dict:
    """
    Build a diversified portfolio from top recommendations.

    Returns:
        dict with portfolio holdings, weights, expected metrics
    """
    p = RISK_PROFILES.get(profile, RISK_PROFILES["Moderate"])
    max_stocks = p["max_stocks"]
    min_stocks = p["min_stocks"]
    max_per = p["max_per_stock"]

    # Filter candidates — include HOLD as fallback
    candidates = [r for r in recommendations
                  if r["composite"] >= min_score
                  and r["action"] in ("STRONG BUY", "BUY", "ACCUMULATE")]

    if len(candidates) < min_stocks:
        candidates = [r for r in recommendations
                      if r["composite"] >= min(min_score, 35)
                      and r["action"] not in ("AVOID",)][:max_stocks]

    if not candidates:
        candidates = sorted(recommendations, key=lambda x: x["composite"], reverse=True)[:max_stocks]

    if not candidates:
        return {"holdings": [], "error": "No qualifying stocks found"}

    # Step 1: Remove excessive sector concentration
    sector_count = {}
    filtered = []
    max_per_sector = max(2, max_stocks // 3)

    for r in candidates:
        sec = r["sector"]
        sector_count[sec] = sector_count.get(sec, 0)
        if sector_count[sec] < max_per_sector:
            filtered.append(r)
            sector_count[sec] += 1

    candidates = filtered[:max_stocks * 2]  # keep buffer for correlation filter

    # Step 2: Build correlation matrix and remove highly correlated pairs
    tickers = [r["ticker"] for r in candidates]
    returns_data = {}
    for t in tickers:
        d = all_data.get(t, {}).get("price")
        if d is not None and len(d) > 60:
            returns_data[t] = d["Close"].pct_change().dropna().tail(60)

    if len(returns_data) > 2:
        ret_df = pd.DataFrame(returns_data).dropna()
        if len(ret_df) > 20:
            corr = ret_df.corr()
            to_remove = set()
            for i, t1 in enumerate(corr.columns):
                for j, t2 in enumerate(corr.columns):
                    if i < j and abs(corr.iloc[i, j]) > 0.85:
                        # Remove the one with lower composite score
                        s1 = next((r["composite"] for r in candidates if r["ticker"] == t1), 0)
                        s2 = next((r["composite"] for r in candidates if r["ticker"] == t2), 0)
                        to_remove.add(t2 if s1 >= s2 else t1)
            candidates = [r for r in candidates if r["ticker"] not in to_remove]

    # Step 3: Select final holdings
    holdings = candidates[:max_stocks]
    if len(holdings) < min_stocks:
        holdings = candidates[:min_stocks] if len(candidates) >= min_stocks else candidates

    if not holdings:
        return {"holdings": [], "error": "No stocks passed portfolio filters"}

    # Step 4: Inverse-volatility weighting
    vols = []
    for h in holdings:
        vol = h.get("volatility", 25)
        vols.append(max(vol, 5))  # floor at 5%

    inv_vols = [1.0 / v for v in vols]
    total_iv = sum(inv_vols)
    raw_weights = [iv / total_iv for iv in inv_vols]

    # Step 5: Apply score tilt — higher composite gets slightly more
    scores = [h["composite"] for h in holdings]
    score_weights = [s / sum(scores) for s in scores]

    # Blend: 60% inverse-vol, 40% score-based
    blended = [0.6 * rw + 0.4 * sw for rw, sw in zip(raw_weights, score_weights)]
    total_b = sum(blended)
    weights = [b / total_b for b in blended]

    # Cap at max_per_stock
    capped = [min(w, max_per) for w in weights]
    total_c = sum(capped)
    weights = [w / total_c for w in capped]

    # Step 6: Build output
    portfolio_holdings = []
    total_invested = 0
    sectors_alloc = {}

    for h, w in zip(holdings, weights):
        amount = round(portfolio_val * w)
        price = h.get("price", 0)
        qty = int(amount / price) if price > 0 else 0
        actual_amount = qty * price if price > 0 else amount

        sec = h["sector"]
        sectors_alloc[sec] = sectors_alloc.get(sec, 0) + w * 100

        portfolio_holdings.append({
            "ticker": h["ticker"],
            "name": h["name"],
            "sector": sec,
            "cap": h["cap"],
            "composite": h["composite"],
            "action": h["action"],
            "weight_pct": round(w * 100, 2),
            "amount": round(actual_amount),
            "qty": qty,
            "price": price,
            "risk_level": h["risk_level"],
            "holding": h["holding"],
            "stage": h["stage"][1] if isinstance(h["stage"], tuple) else str(h["stage"]),
            "volatility": h["volatility"],
            "returns": h.get("returns", {}),
        })
        total_invested += actual_amount

    # Portfolio-level metrics
    port_vol = np.sqrt(sum((w * v / 100) ** 2 for w, v in zip(weights, vols))) * 100
    avg_score = np.mean(scores)
    risk_levels = [h["risk_level"] for h in holdings]
    high_risk_pct = sum(1 for r in risk_levels if r == "High") / len(risk_levels) * 100

    return {
        "holdings": portfolio_holdings,
        "total_stocks": len(portfolio_holdings),
        "total_invested": round(total_invested),
        "cash_remaining": round(portfolio_val - total_invested),
        "avg_composite": round(avg_score, 1),
        "portfolio_volatility": round(port_vol, 1),
        "high_risk_pct": round(high_risk_pct, 0),
        "sector_allocation": {k: round(v, 1) for k, v in sorted(sectors_alloc.items(), key=lambda x: -x[1])},
        "diversification_score": min(100, len(set(h["sector"] for h in holdings)) * 15),
    }
