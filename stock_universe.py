"""
Dynamic stock universe fetcher.
Pulls Nifty 50 / 100 / 500 constituent lists directly from NSE India,
maps industries to sectors, and classifies market cap using
actual market cap data when available, falling back to index position.
"""

import streamlit as st
from curl_cffi.requests import Session
import io
import csv
import json
from datetime import datetime
from pathlib import Path

NSE_BASE = "https://archives.nseindia.com/content/indices"
NSE_MIRRORS = (
    "https://nsearchives.nseindia.com/content/indices",
    "https://www.niftyindices.com/IndexConstituent",
    NSE_BASE,
)
CACHE_DIR = Path(__file__).resolve().parent / "data" / "index_cache"
MEMBERSHIP_DIR = Path(__file__).resolve().parent / "data" / "sector_membership"

# The official Nifty 500 CSV only labels these as "Chemicals", so a dedicated
# symbol taxonomy is required to avoid losing the fertilizer/agri-input theme.
FERTILIZER_AGRI_TICKERS = {
    "BAYERCROP", "CHAMBLFERT", "COROMANDEL", "DEEPAKFERT", "FACT",
    "PARADEEP", "PIIND", "SUMICHEM", "UPL",
}

INDEX_FILES = {
    "Nifty 50":  "ind_nifty50list.csv",
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty 500": "ind_nifty500list.csv",
}

MIN_CONSTITUENTS = {
    "Nifty 50": 45,
    "Nifty 100": 90,
    "Nifty 500": 450,
}

INDUSTRY_TO_SECTOR = {
    "Financial Services":         "Banking & Finance",
    "Banks":                      "Banking & Finance",
    "Insurance":                  "Banking & Finance",
    "Information Technology":     "IT",
    "IT - Software":              "IT",
    "IT - Services":              "IT",
    "Software":                   "IT",
    "Oil Gas & Consumable Fuels": "Oil & Gas",
    "Oil & Gas":                  "Oil & Gas",
    "Petroleum Products":         "Oil & Gas",
    "Gas Transmission/Marketing": "Oil & Gas",
    "Fast Moving Consumer Goods": "FMCG",
    "FMCG":                       "FMCG",
    "Food Products":              "FMCG",
    "Beverages":                  "FMCG",
    "Personal Products":          "FMCG",
    "Tobacco Products":           "FMCG",
    "Household Products":         "FMCG",
    "Pharmaceuticals":            "Pharma",
    "Pharmaceuticals & Biotechnology": "Pharma",
    "Healthcare":                 "Healthcare",
    "Healthcare Services":        "Healthcare",
    "Hospitals":                  "Healthcare",
    "Automobile and Auto Components": "Automobile",
    "Automobile":                 "Automobile",
    "Auto Components":            "Automobile",
    "Auto Components & Equipments": "Automobile",
    "Metals & Mining":            "Metals",
    "Metals":                     "Metals",
    "Steel":                      "Metals",
    "Iron & Steel":               "Metals",
    "Aluminium":                  "Metals",
    "Zinc":                       "Metals",
    "Copper":                     "Metals",
    "Mining & Mineral Products":  "Metals",
    "Construction":               "Infrastructure",
    "Construction Materials":     "Infrastructure",
    "Cement & Cement Products":   "Cement",
    "Cement":                     "Cement",
    "Power":                      "Power",
    "Electrical Equipment":       "Power",
    "Telecommunication":          "Telecom",
    "Telecom - Services":         "Telecom",
    "Capital Goods":              "Capital Goods",
    "Industrial Manufacturing":   "Capital Goods",
    "Aerospace & Defense":        "Defence",
    "Defence":                    "Defence",
    "Chemicals":                  "Chemicals",
    "Fertilizers & Agrochemicals":"Chemicals",
    "Pesticides & Agrochemicals": "Chemicals",
    "Specialty Chemicals":        "Chemicals",
    "Consumer Durables":          "Consumer Durables",
    "Consumer Electronics":       "Consumer Durables",
    "Household Appliances":       "Consumer Durables",
    "Realty":                     "Real Estate",
    "Real Estate":                "Real Estate",
    "Media Entertainment & Publication": "Media",
    "Media":                      "Media",
    "Entertainment":              "Media",
    "Diversified":                "Diversified",
    "Conglomerates":              "Diversified",
    "Services":                   "Services",
    "Retailing":                  "Retail",
    "Textiles":                   "Textiles",
    "Textile Products":           "Textiles",
    "Paper & Paper Products":     "Paper",
    "Forest Materials":           "Paper",
    "Transport Services":         "Logistics",
    "Transport Infrastructure":   "Logistics",
    "Logistics":                  "Logistics",
    "Agricultural Food & other Products": "Agri",
    "Sugar":                      "Agri",
    "Edible Oil":                 "Agri",
    "Fertilizers":                "Chemicals",
    "Gems Jewellery And Watches": "Consumer Durables",
    "Leisure Services":           "Services",
    "Trading":                    "Diversified",
    "Commercial Services & Supplies": "Services",
}


def _map_sector(industry: str) -> str:
    if not industry:
        return "Other"
    industry_lower = industry.lower()
    if any(term in industry_lower for term in
           ("fertilizer", "agrochemical", "pesticide")):
        return "Fertilizers & Agrochemicals"
    for key, sector in INDUSTRY_TO_SECTOR.items():
        if key.lower() in industry_lower or industry_lower in key.lower():
            return sector
    return industry


def _classify_cap_by_position(index_name: str, position: int) -> str:
    """Position-based classification within Nifty 500."""
    if index_name in ("Nifty 50", "Nifty 100"):
        return "Large Cap"
    if position <= 100:
        return "Large Cap"
    elif position <= 250:
        return "Mid Cap"
    else:
        return "Small Cap"


def _parse_constituents(csv_text: str, index_name: str) -> dict:
    """Parse and validate an official Nifty constituent CSV."""
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    stocks = {}
    for i, row in enumerate(reader):
        symbol = row.get("Symbol", "").strip()
        name = row.get("Company Name", "").strip()
        industry = row.get("Industry", "").strip()
        if not symbol:
            continue

        ticker = f"{symbol}.NS"
        sector = (
            "Fertilizers & Agrochemicals"
            if symbol in FERTILIZER_AGRI_TICKERS else _map_sector(industry)
        )
        cap = _classify_cap_by_position(index_name, i + 1)
        stocks[ticker] = (name, sector, cap, industry)

    return stocks if len(stocks) >= MIN_CONSTITUENTS[index_name] else {}


def _save_index_cache(filename: str, csv_text: str):
    """Persist the last-known-good official list for upstream outages."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / filename
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(csv_text, encoding="utf-8")
        tmp_path.replace(cache_path)
    except OSError:
        pass


def _load_index_cache(filename: str, index_name: str) -> dict:
    try:
        csv_text = (CACHE_DIR / filename).read_text(encoding="utf-8")
        return _parse_constituents(csv_text, index_name)
    except OSError:
        return {}


def _parse_symbol_list(csv_text: str) -> set[str]:
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    symbols = {
        f"{row.get('Symbol', '').strip()}.NS"
        for row in reader if row.get("Symbol", "").strip()
    }
    return symbols if len(symbols) >= 5 else set()


def _save_membership_snapshot(index_name: str, stocks: dict):
    """Keep one current-universe snapshot per month for future PIT validation."""
    try:
        MEMBERSHIP_DIR.mkdir(parents=True, exist_ok=True)
        month = datetime.now().strftime("%Y-%m")
        path = MEMBERSHIP_DIR / f"{index_name.lower().replace(' ', '-')}-{month}.json"
        if path.exists():
            return
        payload = {
            "index": index_name,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "stocks": {
                ticker: {"name": values[0], "sector": values[1],
                         "cap": values[2], "industry": values[3]}
                for ticker, values in stocks.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def classify_cap_by_mcap(mcap_inr: float) -> str:
    """
    SEBI-compliant market cap classification.
    Large Cap: top 100 by mcap (roughly > ₹40,000 Cr)
    Mid Cap: 101-250 (roughly ₹10,000-40,000 Cr)
    Small Cap: 251+ (< ₹10,000 Cr)
    """
    if mcap_inr is None or mcap_inr <= 0:
        return "Unknown"
    mcap_cr = mcap_inr / 1e7
    if mcap_cr >= 40000:
        return "Large Cap"
    elif mcap_cr >= 10000:
        return "Mid Cap"
    else:
        return "Small Cap"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_constituents(index_name: str = "Nifty 500") -> dict:
    """
    Fetch index constituents from resilient official NSE/Nifty mirrors.
    Falls back to the last-known-good local copy during upstream outages.
    Returns {ticker: (name, sector, cap_category, industry)}
    """
    filename = INDEX_FILES.get(index_name)
    if not filename:
        return {}

    session = Session(verify=False, impersonate="chrome")
    for base_url in NSE_MIRRORS:
        try:
            url = f"{base_url}/{filename}"
            response = session.get(
                url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}
            )
            if response.status_code != 200:
                continue
            stocks = _parse_constituents(response.text, index_name)
            if stocks:
                _save_index_cache(filename, response.text)
                _save_membership_snapshot(index_name, stocks)
                return stocks
        except Exception:
            continue

    stocks = _load_index_cache(filename, index_name)
    if stocks:
        _save_membership_snapshot(index_name, stocks)
    return stocks


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sector_constituents() -> dict[str, set[str]]:
    """Fetch current official Nifty sector memberships for precise stock tiers."""
    from config import SECTOR_CONSTITUENT_FILES

    session = Session(verify=False, impersonate="chrome")
    result = {}
    for sector, filename in SECTOR_CONSTITUENT_FILES.items():
        symbols = set()
        for base_url in NSE_MIRRORS:
            try:
                response = session.get(
                    f"{base_url}/{filename}", timeout=20,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if response.status_code != 200:
                    continue
                symbols = _parse_symbol_list(response.text)
                if symbols:
                    _save_index_cache(filename, response.text)
                    break
            except Exception:
                continue
        if not symbols:
            try:
                symbols = _parse_symbol_list(
                    (CACHE_DIR / filename).read_text(encoding="utf-8")
                )
            except OSError:
                symbols = set()
        if symbols:
            result[sector] = symbols
    return result


def reclassify_caps(stocks: dict, all_data: dict) -> dict:
    """
    Re-classify market caps using actual market cap data from Yahoo Finance.
    Called after data fetching to get accurate cap categories.
    """
    updated = {}
    for ticker, (name, sector, pos_cap, industry) in stocks.items():
        mcap = None
        d = all_data.get(ticker)
        if d and d.get("info"):
            mcap = d["info"].get("market_cap")
        real_cap = classify_cap_by_mcap(mcap) if mcap else pos_cap
        updated[ticker] = (name, sector, real_cap, industry)
    return updated


def get_all_sectors(stocks: dict) -> list:
    return sorted(set(v[1] for v in stocks.values()))


def get_all_caps(stocks: dict) -> list:
    return sorted(set(v[2] for v in stocks.values()))


def get_sector_members(stocks: dict, sector: str) -> dict:
    """Return current members of one normalized sector/sub-industry basket."""
    return {ticker: values for ticker, values in stocks.items()
            if values[1] == sector}


def filter_stocks(stocks: dict, sectors=None, caps=None) -> dict:
    out = {}
    for ticker, (name, sector, cap, ind) in stocks.items():
        if caps and cap not in caps:
            continue
        if sectors and sector not in sectors:
            continue
        out[ticker] = (name, sector, cap, ind)
    return out


def has_cap_category(stocks: dict, cap: str) -> bool:
    return any(v[2] == cap for v in stocks.values())
