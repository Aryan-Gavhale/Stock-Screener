"""
StockPulse India v5 — Personal Alpha Engine
Configuration: scoring weights, sector benchmarks, risk profiles,
regime thresholds, burst criteria, and trade-management parameters.

Design goal: generate ~18-22%/yr with asymmetric upside ("bursts")
via a momentum + quality strategy with strict risk management.
Stock universe is fetched dynamically from NSE (stock_universe.py).
"""

# ── Sector valuation / quality benchmarks ───────────────────────────────────────
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

# ── Strategy profiles ────────────────────────────────────────────────────────────
# Pillar weights: Trend/Momentum is the primary return driver.
# tqgv = trend/momentum, quality, growth(earnings), value(GARP)
RISK_PROFILES = {
    "Conservative": {
        "desc": "Large-cap quality compounder. Target 14-18%/yr, lower drawdown.",
        "trend_w": 0.30, "quality_w": 0.30, "growth_w": 0.20, "value_w": 0.20,
        "max_per_stock": 0.10, "min_stocks": 8, "max_stocks": 15,
        "preferred_caps": ["Large Cap"],
        "risk_per_trade": 0.007,      # 0.7% of capital risked per position
        "core_stop_pct": 0.10,        # baseline stop distance
        "burst_alloc": 0.0,           # no speculative burst bets
        "max_portfolio_heat": 0.05,
    },
    "Moderate": {
        "desc": "Balanced momentum + quality. Target ~18-22%/yr, drawdowns ~20-25%.",
        "trend_w": 0.40, "quality_w": 0.25, "growth_w": 0.20, "value_w": 0.15,
        "max_per_stock": 0.10, "min_stocks": 6, "max_stocks": 12,
        "preferred_caps": ["Large Cap", "Mid Cap", "Small Cap"],
        "risk_per_trade": 0.010,      # risk 1% of capital per position
        "core_stop_pct": 0.09,
        "burst_alloc": 0.20,          # up to 20% of book in small burst bets
        "max_portfolio_heat": 0.07,
    },
    "Aggressive": {
        "desc": "Momentum-led, burst hunting. Target 25-35%/yr, drawdowns 30-40%.",
        "trend_w": 0.48, "quality_w": 0.17, "growth_w": 0.25, "value_w": 0.10,
        "max_per_stock": 0.12, "min_stocks": 5, "max_stocks": 10,
        "preferred_caps": ["Large Cap", "Mid Cap", "Small Cap"],
        "risk_per_trade": 0.0125,
        "core_stop_pct": 0.11,
        "burst_alloc": 0.35,
        "max_portfolio_heat": 0.09,
    },
}

# ── Trade management (applies across profiles, tuned per profile above) ──────────
TRADE_PARAMS = {
    "atr_stop_mult": 2.5,          # initial stop = entry - 2.5 * ATR (or profile %, wider)
    "trail_trigger_gain": 0.15,    # start trailing once +15%
    "trail_atr_mult": 3.0,         # trailing stop = high - 3 * ATR
    "profit_target_r": 3.0,        # primary target = 3R (3x initial risk)
    "pullback_rsi_low": 40,        # ideal pullback entry RSI band
    "pullback_rsi_high": 58,
    "extended_atr_from_ema": 4.0,  # >4 ATR above EMA21 = extended, don't chase
    "breakout_lookback": 50,       # breakout = new high over N sessions
}

# ── Market regime filter (protects capital in downtrends) ────────────────────────
REGIME = {
    "index_ma": 200,               # Nifty above/below 200-DMA
    "breadth_bull": 55,            # % of stocks above 50-DMA for "risk-on"
    "breadth_bear": 40,            # below this = "risk-off"
    # deployment multipliers applied to position sizes
    "deploy_risk_on": 1.0,
    "deploy_neutral": 0.6,
    "deploy_risk_off": 0.3,
}

# ── "Burst" / multibagger candidate criteria (asymmetric upside bucket) ──────────
BURST_CRITERIA = {
    "max_mcap_cr": 25000,          # small/mid only
    "min_rev_growth": 0.18,        # >18% revenue growth
    "min_earn_growth": 0.20,       # >20% earnings growth (accelerating ideal)
    "min_rs_pct": 70,              # top 30% relative strength
    "max_from_high": -18,          # within 18% of 52w high (strength, not falling knife)
    "min_vol_ratio": 1.15,         # accumulation
    "max_de": 1.0,                 # manageable debt
}

# ── Costs for realistic backtesting ──────────────────────────────────────────────
COST_PARAMS = {
    "round_trip_cost": 0.004,      # ~0.4% brokerage+STT+slippage per round trip
    "stcg_tax": 0.15,              # short-term capital gains (held < 1yr)
    "risk_free_rate": 0.065,       # for Sharpe
}

ANALYSIS_PERIODS = {
    "6 Months": "6mo",
    "1 Year":   "1y",
    "2 Years":  "2y",
    "3 Years":  "3y",
    "5 Years":  "5y",
    "10 Years": "10y",
}

# ── Sector rotation intelligence ────────────────────────────────────────────────
# Yahoo symbols validated against the corresponding Nifty price indices.
# Sectors without a reliable public symbol are built as equal-weight Nifty 500
# baskets in sector_data.py and clearly marked "Synthetic".
SECTOR_INDICES = {
    "Automobile": "^CNXAUTO",
    "Banking & Finance": "^NSEBANK",
    "IT": "^CNXIT",
    "FMCG": "^CNXFMCG",
    "Metals": "^CNXMETAL",
    "Pharma": "^CNXPHARMA",
    "Real Estate": "^CNXREALTY",
    "Media": "^CNXMEDIA",
    "Oil & Gas": "^CNXENERGY",
    "PSU Banks": "^CNXPSUBANK",
    "Consumer Durables": "^CNXCONSUM",
    "Infrastructure": "^CNXINFRA",
}

SECTOR_CONSTITUENT_FILES = {
    "Automobile": "ind_niftyautolist.csv",
    "Banking & Finance": "ind_niftybanklist.csv",
    "IT": "ind_niftyitlist.csv",
    "FMCG": "ind_niftyfmcglist.csv",
    "Metals": "ind_niftymetallist.csv",
    "Pharma": "ind_niftypharmalist.csv",
    "Healthcare": "ind_niftyhealthcarelist.csv",
    "Real Estate": "ind_niftyrealtylist.csv",
    "Media": "ind_niftymedialist.csv",
    "Oil & Gas": "ind_niftyenergylist.csv",
    "PSU Banks": "ind_niftypsubanklist.csv",
    "Consumer Durables": "ind_niftyconsumerdurableslist.csv",
    "Infrastructure": "ind_niftyinfralist.csv",
}

SECTOR_ROTATION = {
    "horizons": {
        "1M Tactical": {"days": 21, "forward_days": 21,
                        "weights": {"mom": 0.34, "accel": 0.24, "breadth": 0.22,
                                    "trend": 0.12, "risk": 0.08}},
        "3M Swing": {"days": 63, "forward_days": 63,
                     "weights": {"mom": 0.38, "accel": 0.18, "breadth": 0.20,
                                 "trend": 0.14, "risk": 0.10}},
        "6M Positional": {"days": 126, "forward_days": 126,
                          "weights": {"mom": 0.42, "accel": 0.12, "breadth": 0.18,
                                      "trend": 0.18, "risk": 0.10}},
    },
    "default_horizon": "3M Swing",
    "seasonality_weight_cap": 0.07,
    "live_fundamental_weight_cap": 0.08,
    "macro_weight_cap": 0.08,
    "catalyst_weight_cap": 0.05,
    "min_sector_members": 4,
    "min_history_days": 252,
    "min_seasonal_years": 6,
    "top_sectors": 3,
    "sector_tilt_cap": 0.04,       # max +4 score points on an individual stock
    "hero_min_score": 66,
    "hero_min_sector_rs": 70,
    "min_turnover_cr": 1.0,        # 20-day median daily traded value
}

# Directional fit only; these are capped context signals, never standalone calls.
SECTOR_DRIVERS = {
    "Banking & Finance": {"style": "cyclical", "likes": ["risk_on", "rates_easing"]},
    "PSU Banks": {"style": "cyclical", "likes": ["risk_on", "rates_easing"]},
    "Automobile": {"style": "cyclical", "likes": ["risk_on", "rates_easing", "crude_down"]},
    "Real Estate": {"style": "cyclical", "likes": ["risk_on", "rates_easing"]},
    "Capital Goods": {"style": "cyclical", "likes": ["risk_on", "capex"]},
    "Infrastructure": {"style": "cyclical", "likes": ["risk_on", "capex"]},
    "Metals": {"style": "cyclical", "likes": ["risk_on", "commodity_up"]},
    "Oil & Gas": {"style": "inflation", "likes": ["crude_up"]},
    "IT": {"style": "export", "likes": ["inr_weak", "global_risk_on"]},
    "Pharma": {"style": "defensive_export", "likes": ["risk_off", "inr_weak"]},
    "Healthcare": {"style": "defensive", "likes": ["risk_off"]},
    "FMCG": {"style": "defensive", "likes": ["risk_off", "crude_down"]},
    "Power": {"style": "defensive", "likes": ["risk_off", "capex"]},
    "Chemicals": {"style": "export", "likes": ["global_risk_on", "crude_down"]},
    "Fertilizers & Agrochemicals": {
        "style": "agri", "likes": ["monsoon", "rural_demand", "gas_down"]
    },
}

SECTOR_NEWS_TERMS = {
    "positive": ["upgrade", "demand rises", "record order", "policy support",
                 "subsidy", "rate cut", "export growth", "profit jumps",
                 "strong monsoon", "capacity expansion", "approval"],
    "negative": ["downgrade", "demand falls", "margin pressure", "ban", "penalty",
                 "rate hike", "profit falls", "weak monsoon", "default",
                 "regulatory action", "price control"],
}

# ── Market context symbols (cross-asset crash radar) ─────────────────────────────
MARKET_CONTEXT = {
    "india_vix": "^INDIAVIX",
    "sp500":     "^GSPC",
    "us_vix":    "^VIX",
    "usdinr":    "USDINR=X",
    "crude":     "BZ=F",
}

# ── Crash Radar thresholds ───────────────────────────────────────────────────────
CRASH = {
    "india_vix_elevated": 16,
    "india_vix_high": 20,
    "india_vix_extreme": 26,
    "vix_spike_pct": 20,             # 5-day India VIX jump % = fear spike
    "us_vix_high": 25,
    "distribution_lookback": 25,     # sessions to count institutional selling
    "distribution_days_warn": 4,     # >=4 distribution days = warning
    "index_extended_above_200": 15,  # index >15% above 200-DMA = froth/mean-revert risk
    "usdinr_spike_pct": 2.0,         # 1-month rupee depreciation % (FII-outflow proxy)
    "crude_spike_pct": 15,           # 1-month crude jump % (imported-inflation risk)
    "breadth_weak": 40,
    "breadth_washout": 20,
    "bands": {"elevated": 30, "high": 55, "extreme": 75},
    # blend weights for the 0-100 index
    "weights": {"trend": 0.22, "breadth": 0.18, "distribution": 0.14,
                "divergence": 0.10, "vix": 0.16, "global": 0.12, "news": 0.08},
}

CRASH_NEWS_KEYWORDS = {
    "high": ["crash", "collapse", "plunge", "selloff", "sell-off", "meltdown",
             "panic", "default", "crisis", "war", "recession", "circuit breaker", "bloodbath"],
    "med":  ["rate hike", "inflation", "fed ", "rbi", "tariff", "slump", "tumble",
             "correction", "fear", "downgrade", "surging yields", "geopolitical", "slowdown", "fii selling"],
}

# ── Dip / accumulation (post-crash bottom-fishing) ───────────────────────────────
ACCUMULATION = {
    "drawdown_trigger": 8,          # index down >8% from 6m high to arm the detector
    "deep_drawdown": 12,
    "index_rsi_oversold": 35,
    "breadth_washout": 20,          # % above 50-DMA capitulation low
    "follow_through_gain": 1.2,     # % up day for O'Neil follow-through
    "follow_through_vol": 1.0,      # volume >= 20d avg
}

DIP_CRITERIA = {
    "min_fall_from_high": -25,      # fell at least 25% from 52-week high
    "deep_fall": -40,
    "max_rsi": 48,                  # washed out, not extended
    "min_quality": 55,              # fundamentals must be intact (quality_s)
    "max_de": 1.2,                  # not over-levered
    "tranches": 3,                  # stagger the accumulation
    "stop_below_low_pct": 8,        # stop 8% below capitulation low
}
