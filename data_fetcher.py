"""
Robust data fetching layer using direct Yahoo Finance API.
Uses curl_cffi with SSL verification disabled (for corporate networks)
and proper crumb-based authentication for fundamentals.
"""

import pandas as pd
import numpy as np
import requests as stdlib_requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import time

from curl_cffi.requests import Session as CurlSession

NIFTY_TICKER = "^NSEI"

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
QUOTE_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
COOKIE_URL = "https://fc.yahoo.com"

PERIOD_MAP = {
    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730,
    "3y": 1095, "5y": 1825, "10y": 3650,
}

# ── Session & Crumb management ────────────────────────────────────────────────

class YahooSession:
    """Manages a curl_cffi session with cookie + crumb for Yahoo Finance."""

    def __init__(self):
        self._session = CurlSession(verify=False, impersonate="chrome")
        self._crumb = None

    def _ensure_crumb(self):
        if self._crumb:
            return
        try:
            self._session.get(COOKIE_URL, timeout=10, allow_redirects=True)
            r = self._session.get(CRUMB_URL, timeout=10)
            if r.status_code == 200 and r.text:
                self._crumb = r.text
        except Exception:
            pass

    @property
    def session(self):
        return self._session

    @property
    def crumb(self):
        self._ensure_crumb()
        return self._crumb

    def get(self, url, **kwargs):
        return self._session.get(url, **kwargs)


@st.cache_resource
def _get_yahoo_session():
    return YahooSession()

# ── Price data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """Fetch OHLCV data from Yahoo Finance chart API (no auth needed)."""
    try:
        ys = _get_yahoo_session()
        days = PERIOD_MAP.get(period, 365)
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=days)).timestamp())

        r = ys.get(CHART_URL.format(ticker=ticker),
                   params={"period1": start, "period2": end, "interval": "1d"},
                   timeout=15)
        if r.status_code != 200:
            return None

        result = r.json().get("chart", {}).get("result")
        if not result:
            return None
        result = result[0]

        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        if not timestamps or not quote.get("close"):
            return None

        df = pd.DataFrame({
            "Open":   quote.get("open", []),
            "High":   quote.get("high", []),
            "Low":    quote.get("low", []),
            "Close":  quote.get("close", []),
            "Volume": quote.get("volume", []),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))
        df.index = df.index.tz_convert(None)
        df.index.name = "Date"
        df = df.dropna(subset=["Close"])
        return df if len(df) >= 20 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nifty(period: str = "2y") -> pd.DataFrame | None:
    return fetch_price(NIFTY_TICKER, period)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_context() -> dict:
    """
    Cross-asset context for the Crash Radar: India VIX, S&P 500, US VIX,
    USD/INR, crude. Each may be None if Yahoo doesn't return it — the radar
    degrades gracefully to whatever is available.
    """
    from config import MARKET_CONTEXT
    out = {}
    for name, sym in MARKET_CONTEXT.items():
        try:
            out[name] = fetch_price(sym, "2y")
        except Exception:
            out[name] = None
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_indices(period: str = "10y") -> dict:
    """Fetch all configured official Nifty sector price indices in parallel."""
    from config import SECTOR_INDICES

    out = {}
    workers = min(6, len(SECTOR_INDICES))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_price, symbol, period): (sector, symbol)
            for sector, symbol in SECTOR_INDICES.items()
        }
        for future in as_completed(futures):
            sector, symbol = futures[future]
            try:
                price = future.result()
            except Exception:
                price = None
            if price is not None:
                out[sector] = {"price": price, "symbol": symbol, "source": "Official"}
    return out

# ── Fundamentals ───────────────────────────────────────────────────────────────

def _raw(d, key):
    """Extract raw value from Yahoo's {raw: x, fmt: y} format."""
    v = d.get(key, {})
    if isinstance(v, dict):
        return v.get("raw")
    return v


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker: str) -> dict | None:
    """Fetch fundamental data using quoteSummary API with crumb auth."""
    try:
        ys = _get_yahoo_session()
        crumb = ys.crumb
        params = {
            "modules": "defaultKeyStatistics,financialData,summaryDetail",
        }
        if crumb:
            params["crumb"] = crumb

        r = ys.get(QUOTE_SUMMARY_URL.format(ticker=ticker), params=params, timeout=15)
        if r.status_code != 200:
            return None

        qr = r.json().get("quoteSummary", {}).get("result")
        if not qr:
            return None

        ks = qr[0].get("defaultKeyStatistics", {})
        fd = qr[0].get("financialData", {})
        sd = qr[0].get("summaryDetail", {})

        return {
            "market_cap":    _raw(sd, "marketCap"),
            "pe":            _raw(sd, "trailingPE"),
            "forward_pe":    _raw(sd, "forwardPE") or _raw(ks, "forwardPE"),
            "pb":            _raw(ks, "priceToBook"),
            "peg":           _raw(ks, "pegRatio"),
            "div_yield":     _raw(sd, "dividendYield") or 0,
            "roe":           _raw(fd, "returnOnEquity"),
            "roa":           _raw(fd, "returnOnAssets"),
            "profit_margin": _raw(fd, "profitMargins"),
            "op_margin":     _raw(fd, "operatingMargins"),
            "rev_growth":    _raw(fd, "revenueGrowth"),
            "earn_growth":   _raw(fd, "earningsGrowth"),
            "de":            _raw(fd, "debtToEquity"),
            "current_ratio": _raw(fd, "currentRatio"),
            "beta":          _raw(ks, "beta") or 1.0,
            "eps":           _raw(sd, "trailingEps") or _raw(ks, "trailingEps"),
            "forward_eps":   _raw(ks, "forwardEps"),
            "fcf":           _raw(fd, "freeCashflow"),
            "revenue":       _raw(fd, "totalRevenue"),
            "total_debt":    _raw(fd, "totalDebt"),
            "total_cash":    _raw(fd, "totalCash"),
            "price":         _raw(fd, "currentPrice"),
            "52w_high":      _raw(sd, "fiftyTwoWeekHigh"),
            "52w_low":       _raw(sd, "fiftyTwoWeekLow"),
            "50d_avg":       _raw(sd, "fiftyDayAverage"),
            "200d_avg":      _raw(sd, "twoHundredDayAverage"),
            "rec":           fd.get("recommendationKey"),
        }
    except Exception:
        return None

# ── Bulk fetch ─────────────────────────────────────────────────────────────────

def fetch_all(tickers: list, period: str = "2y", progress_cb=None) -> dict:
    results = {}
    total = len(tickers)

    def _one(t):
        price = fetch_price(t, period)
        info = fetch_info(t)
        return t, price, info

    workers = min(6, max(3, total // 50))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, t): t for t in tickers}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                t, price, info = f.result()
                if price is not None:
                    results[t] = {"price": price, "info": info}
            except Exception:
                pass
            if progress_cb:
                pct = 0.05 + (done / total) * 0.80
                progress_cb(pct, f"Fetched {done}/{total} · {len(results)} valid")

    return results

# ── News ───────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(query: str = "indian stock market NSE") -> list:
    try:
        import urllib3
        urllib3.disable_warnings()
        url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
        r = stdlib_requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8, verify=False)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        return [
            {"title": it.findtext("title", ""),
             "link":  it.findtext("link", ""),
             "date":  it.findtext("pubDate", "")}
            for it in list(root.findall(".//item"))[:12]
        ]
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_risk_news() -> dict:
    """
    Scan headlines for crash-related keywords and return a 0-100 news-risk
    score plus the matched terms. Low-confidence signal by nature — weighted
    lightly in the Crash Radar.
    """
    from config import CRASH_NEWS_KEYWORDS
    q = ("India stock market crash OR selloff OR correction OR RBI rate OR Fed OR "
         "inflation OR recession OR geopolitical OR FII selling")
    headlines = fetch_news(q)
    hi = CRASH_NEWS_KEYWORDS["high"]
    md = CRASH_NEWS_KEYWORDS["med"]
    score, matched = 0, []
    for a in headlines:
        title = (a.get("title") or "").lower()
        for k in hi:
            if k in title:
                score += 12; matched.append(k)
        for k in md:
            if k in title:
                score += 5; matched.append(k)
    return {"score": min(100, score), "matched": sorted(set(matched)), "headlines": headlines}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_sector_news(sector: str) -> dict:
    """Small, transparent catalyst score for a sector's current headlines."""
    from config import SECTOR_NEWS_TERMS

    query_name = {
        "Banking & Finance": "Indian banks financial stocks",
        "Fertilizers & Agrochemicals": "India fertilizer agrochemical stocks monsoon subsidy",
        "Oil & Gas": "India oil gas energy stocks crude",
        "Real Estate": "India real estate stocks interest rates",
    }.get(sector, f"India {sector} sector stocks")
    headlines = fetch_news(query_name)
    score = 50
    positive, negative = [], []
    for article in headlines:
        title = (article.get("title") or "").lower()
        for term in SECTOR_NEWS_TERMS["positive"]:
            if term in title:
                score += 6
                positive.append(term)
        for term in SECTOR_NEWS_TERMS["negative"]:
            if term in title:
                score -= 7
                negative.append(term)
    return {
        "score": max(0, min(100, score)),
        "positive": sorted(set(positive)),
        "negative": sorted(set(negative)),
        "headlines": headlines,
        "fetched_at": datetime.now().isoformat(timespec="minutes"),
    }


def fetch_all_sector_news(sectors: list[str]) -> dict:
    """Fetch sector catalyst scans concurrently."""
    out = {}
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(sectors)))) as pool:
        futures = {pool.submit(fetch_sector_news, sector): sector
                   for sector in sectors}
        for future in as_completed(futures):
            sector = futures[future]
            try:
                out[sector] = future.result()
            except Exception:
                out[sector] = {"score": 50, "positive": [], "negative": [],
                               "headlines": [], "fetched_at": None}
    return out

# ── Market breadth ─────────────────────────────────────────────────────────────

def market_breadth(all_data: dict) -> dict:
    above, total = 0, 0
    for _, d in (all_data or {}).items():
        df = d.get("price")
        if df is None or len(df) < 50:
            continue
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        total += 1
        if df["Close"].iloc[-1] > sma50:
            above += 1
    pct = (above / total * 100) if total else 50
    label = "Bullish" if pct >= 65 else ("Neutral" if pct >= 45 else "Bearish")
    return {"above_50sma_pct": round(pct, 1), "label": label, "above": above, "total": total}


def breadth_series(all_data: dict, lookback: int = 300) -> pd.Series | None:
    """
    Time series of % of stocks above their 50-DMA. Used by the Crash Radar to
    detect breadth deterioration / index-up-breadth-down divergence and by the
    Dip detector to spot a breadth washout + turn.
    """
    closes = {}
    for t, d in (all_data or {}).items():
        px = d.get("price")
        if px is None or len(px) < 60:
            continue
        closes[t] = px["Close"]
    if len(closes) < 10:
        return None
    cm = pd.DataFrame(closes)
    sma = cm.rolling(50).mean()
    above = (cm > sma)
    valid = cm.notna()
    pct = (above & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan) * 100
    pct = pct.dropna()
    return pct.tail(lookback) if len(pct) > 0 else None
