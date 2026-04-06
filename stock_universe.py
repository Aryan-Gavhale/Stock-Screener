"""
Dynamic stock universe fetcher.
Pulls Nifty 50 / 100 / 500 constituent lists directly from NSE India,
maps industries to sectors, and classifies market cap.
"""

import streamlit as st
from curl_cffi.requests import Session
import io
import csv

NSE_BASE = "https://archives.nseindia.com/content/indices"

INDEX_FILES = {
    "Nifty 50":  "ind_nifty50list.csv",
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty 500": "ind_nifty500list.csv",
}

# Map NSE industry names to broader sectors for analysis
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
    for key, sector in INDUSTRY_TO_SECTOR.items():
        if key.lower() in industry.lower() or industry.lower() in key.lower():
            return sector
    return industry  # Use raw industry if no mapping


def _classify_cap(index_name: str, position: int) -> str:
    """Classify market cap based on index membership and position."""
    if index_name == "Nifty 50":
        return "Large Cap"
    if index_name == "Nifty 100":
        return "Large Cap"
    if position <= 100:
        return "Large Cap"
    elif position <= 250:
        return "Mid Cap"
    else:
        return "Small Cap"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_constituents(index_name: str = "Nifty 500") -> dict:
    """
    Fetch index constituents from NSE India.
    Returns {ticker: (name, sector, cap_category, industry)}
    """
    filename = INDEX_FILES.get(index_name)
    if not filename:
        return {}

    try:
        s = Session(verify=False, impersonate="chrome")
        url = f"{NSE_BASE}/{filename}"
        r = s.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return {}

        reader = csv.DictReader(io.StringIO(r.text))
        stocks = {}
        for i, row in enumerate(reader):
            symbol = row.get("Symbol", "").strip()
            name = row.get("Company Name", "").strip()
            industry = row.get("Industry", "").strip()
            if not symbol:
                continue

            ticker = f"{symbol}.NS"
            sector = _map_sector(industry)
            cap = _classify_cap(index_name, i + 1)
            stocks[ticker] = (name, sector, cap, industry)

        return stocks
    except Exception:
        return {}


def get_all_sectors(stocks: dict) -> list:
    """Get sorted unique sector list from stock universe."""
    return sorted(set(v[1] for v in stocks.values()))


def filter_stocks(stocks: dict, sectors=None, caps=None) -> dict:
    """Filter stocks by sector and cap category."""
    out = {}
    for ticker, (name, sector, cap, ind) in stocks.items():
        if caps and cap not in caps:
            continue
        if sectors and sector not in sectors:
            continue
        out[ticker] = (name, sector, cap, ind)
    return out
