"""
Sector data foundation.

Official Nifty sector indices are preferred for return histories. Sectors and
sub-industries without a reliable public index (notably Fertilizers &
Agrochemicals) use a clearly-labelled equal-weight synthetic basket built from
the currently scanned Nifty 500 members.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SECTOR_ROTATION


def _member_frames(all_data: dict, members: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for ticker in members:
        item = (all_data or {}).get(ticker)
        price = item.get("price") if item else None
        if price is not None and len(price) >= 30:
            frames[ticker] = price
    return frames


def build_synthetic_index(
    all_data: dict, members: list[str], min_members: int | None = None
) -> pd.DataFrame | None:
    """Build an equal-weight daily-rebalanced sector basket from stock returns."""
    min_members = min_members or SECTOR_ROTATION["min_sector_members"]
    frames = _member_frames(all_data, members)
    if len(frames) < min_members:
        return None

    closes = pd.concat(
        {ticker: frame["Close"] for ticker, frame in frames.items()}, axis=1
    ).sort_index()
    returns = closes.pct_change(fill_method=None)
    valid_count = returns.notna().sum(axis=1)
    basket_return = returns.mean(axis=1, skipna=True).where(
        valid_count >= min_members
    )
    basket_close = (1 + basket_return.fillna(0)).cumprod() * 100
    first_valid = basket_return.first_valid_index()
    if first_valid is None:
        return None
    basket_close = basket_close.loc[first_valid:]

    volumes = pd.concat(
        {ticker: frame["Volume"] for ticker, frame in frames.items()}, axis=1
    ).reindex(basket_close.index).sum(axis=1, min_count=1).fillna(0)
    return pd.DataFrame(
        {
            "Open": basket_close,
            "High": basket_close,
            "Low": basket_close,
            "Close": basket_close,
            "Volume": volumes,
        },
        index=basket_close.index,
    )


def sector_breadth(
    all_data: dict, members: list[str], lookback: int = 300
) -> dict:
    """Current and historical participation metrics for one sector."""
    frames = _member_frames(all_data, members)
    if not frames:
        return {
            "members": 0, "above20": 50.0, "above50": 50.0, "above200": 50.0,
            "new_highs": 0.0, "new_lows": 0.0, "breadth50": None,
            "volume_participation": 50.0,
        }

    closes = pd.concat(
        {ticker: frame["Close"] for ticker, frame in frames.items()}, axis=1
    ).sort_index()
    valid = closes.notna()

    def _above(window: int) -> pd.Series:
        ma = closes.rolling(window).mean()
        return ((closes > ma) & valid).sum(axis=1) / valid.sum(axis=1).replace(
            0, np.nan
        ) * 100

    above20, above50, above200 = _above(20), _above(50), _above(200)
    high252 = closes.rolling(252, min_periods=60).max()
    low252 = closes.rolling(252, min_periods=60).min()
    new_highs = ((closes >= high252 * 0.98) & valid).sum(axis=1)
    new_lows = ((closes <= low252 * 1.02) & valid).sum(axis=1)
    denominator = valid.sum(axis=1).replace(0, np.nan)

    volume_votes = []
    for frame in frames.values():
        if len(frame) < 25:
            continue
        ratio = frame["Volume"] / frame["Volume"].rolling(20).mean()
        up = frame["Close"].pct_change(5) > 0
        vote = ((ratio > 1.05) & up).astype(float)
        volume_votes.append(vote.rename(str(len(volume_votes))))
    volume_participation = 50.0
    if volume_votes:
        vote_frame = pd.concat(volume_votes, axis=1)
        latest_votes = vote_frame.ffill().iloc[-1].dropna()
        if len(latest_votes):
            volume_participation = float(latest_votes.mean() * 100)

    def _latest(series: pd.Series, default=50.0):
        series = series.dropna()
        return round(float(series.iloc[-1]), 1) if len(series) else default

    return {
        "members": len(frames),
        "above20": _latest(above20),
        "above50": _latest(above50),
        "above200": _latest(above200),
        "new_highs": _latest(new_highs / denominator * 100, 0.0),
        "new_lows": _latest(new_lows / denominator * 100, 0.0),
        "breadth50": above50.dropna().tail(lookback),
        "volume_participation": round(volume_participation, 1),
    }


def live_fundamental_breadth(all_data: dict, members: list[str]) -> dict:
    """Live-only growth/quality breadth. This must never enter historical tests."""
    infos = []
    for ticker in members:
        info = ((all_data or {}).get(ticker) or {}).get("info")
        if info:
            infos.append(info)
    if not infos:
        return {"coverage": 0, "growth_breadth": 50.0, "quality_breadth": 50.0}

    growth_votes, quality_votes = [], []
    for info in infos:
        rev, earn = info.get("rev_growth"), info.get("earn_growth")
        roe, de = info.get("roe"), info.get("de")
        if isinstance(rev, (int, float)) or isinstance(earn, (int, float)):
            growth_votes.append(
                float((rev or 0) > 0.08 or (earn or 0) > 0.10)
            )
        if isinstance(de, (int, float)) and de > 10:
            de = de / 100
        if isinstance(roe, (int, float)) or isinstance(de, (int, float)):
            quality_votes.append(
                float((roe or 0) > 0.12 and (de is None or de < 1.2))
            )
    return {
        "coverage": len(infos),
        "growth_breadth": round(
            np.mean(growth_votes) * 100 if growth_votes else 50.0, 1
        ),
        "quality_breadth": round(
            np.mean(quality_votes) * 100 if quality_votes else 50.0, 1
        ),
    }


def build_sector_dataset(
    all_data: dict,
    stock_map: dict,
    official_indices: dict,
    bench_df: pd.DataFrame | None = None,
    official_memberships: dict | None = None,
) -> dict:
    """
    Combine official index histories and synthetic Nifty 500 sector baskets.

    Returns {sector: {price, source, symbol, members, breadth, fundamentals}}.
    """
    groups: dict[str, list[str]] = {}
    for ticker, values in (stock_map or {}).items():
        sector = values[1]
        groups.setdefault(sector, []).append(ticker)

    # Preserve official sectors even when a filtered scan has no constituents.
    sectors = sorted(
        set(groups) | set(official_indices or {}) | set(official_memberships or {})
    )
    result = {}
    for sector in sectors:
        official_members = (official_memberships or {}).get(sector)
        members = (
            sorted(ticker for ticker in official_members if ticker in stock_map)
            if official_members else groups.get(sector, [])
        )
        official = (official_indices or {}).get(sector)
        synthetic = build_synthetic_index(all_data, members)
        if official and official.get("price") is not None:
            price = official["price"]
            source = "Official"
            symbol = official.get("symbol")
        elif synthetic is not None:
            price = synthetic
            source = "Synthetic"
            symbol = None
        else:
            continue

        result[sector] = {
            "price": price,
            "source": source,
            "symbol": symbol,
            "members": members,
            "coverage_members": len(_member_frames(all_data, members)),
            "breadth": sector_breadth(all_data, members),
            "fundamentals": live_fundamental_breadth(all_data, members),
        }
    return result


def sector_close_panel(sector_data: dict) -> pd.DataFrame:
    """Aligned sector close panel for ranking/backtesting."""
    series = {
        sector: item["price"]["Close"]
        for sector, item in (sector_data or {}).items()
        if item.get("price") is not None
    }
    return pd.DataFrame(series).sort_index() if series else pd.DataFrame()
