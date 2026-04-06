"""
Configuration for StockPulse India v3.
Scoring weights, sector benchmarks, risk profiles.
Stock universe is now fetched dynamically from NSE via stock_universe.py.
"""

SECTOR_PE = {
    "Banking & Finance": 16, "IT": 28, "Oil & Gas": 11, "FMCG": 50,
    "Pharma": 30, "Healthcare": 40, "Automobile": 24, "Metals": 10,
    "Cement": 28, "Power": 14, "Telecom": 35, "Capital Goods": 35,
    "Chemicals": 28, "Consumer Durables": 48, "Real Estate": 22,
    "Defence": 35, "Infrastructure": 22, "Diversified": 20,
    "Services": 22, "Retail": 60, "Media": 18, "Textiles": 18,
    "Logistics": 22, "Agri": 16, "Paper": 14, "Other": 22,
}

SECTOR_ROE = {
    "Banking & Finance": 14, "IT": 28, "Oil & Gas": 14, "FMCG": 30,
    "Pharma": 16, "Healthcare": 14, "Automobile": 16, "Metals": 12,
    "Cement": 12, "Power": 12, "Telecom": 8, "Capital Goods": 16,
    "Chemicals": 16, "Consumer Durables": 20, "Real Estate": 8,
    "Defence": 22, "Infrastructure": 12, "Diversified": 12,
    "Services": 14, "Retail": 15, "Media": 18, "Textiles": 10,
    "Logistics": 14, "Agri": 12, "Paper": 10, "Other": 12,
}

RISK_PROFILES = {
    "Conservative": {
        "desc": "Capital protection first. Blue-chip focus.",
        "tech_w": 0.25, "fund_w": 0.40, "momentum_w": 0.10, "quality_w": 0.25,
        "max_per_stock": 0.08, "min_stocks": 8, "max_stocks": 15,
        "preferred_caps": ["Large Cap"],
    },
    "Moderate": {
        "desc": "Balanced growth and safety.",
        "tech_w": 0.30, "fund_w": 0.30, "momentum_w": 0.20, "quality_w": 0.20,
        "max_per_stock": 0.07, "min_stocks": 6, "max_stocks": 12,
        "preferred_caps": ["Large Cap", "Mid Cap"],
    },
    "Aggressive": {
        "desc": "Maximum growth. Higher risk tolerance.",
        "tech_w": 0.30, "fund_w": 0.20, "momentum_w": 0.30, "quality_w": 0.20,
        "max_per_stock": 0.06, "min_stocks": 5, "max_stocks": 10,
        "preferred_caps": ["Large Cap", "Mid Cap", "Small Cap"],
    },
}

ANALYSIS_PERIODS = {
    "6 Months": "6mo",
    "1 Year":   "1y",
    "2 Years":  "2y",
}
