"""
Personal positions ledger + live exit monitoring.

Since you actively manage, this is where the money is actually made or saved.
You record what you bought (ticker, entry price, qty, date). On each scan the
app re-checks every holding against fresh price data and tells you:
  HOLD  — trend intact
  TRIM  — losing key support / distribution
  EXIT  — stop hit or trend broken (Stage 4)
plus the current ratcheting stop level so you always know your line in the sand.

Ledger persists to positions.json next to this file.
"""

import json
import os
from datetime import datetime

from data_fetcher import fetch_price

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions.json")


def load_positions() -> list:
    if not os.path.exists(LEDGER_PATH):
        return []
    try:
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_positions(positions: list):
    try:
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(positions, f, indent=2, default=str)
    except Exception:
        pass


def add_position(ticker: str, name: str, entry: float, qty: int,
                 stop: float, date: str = None) -> list:
    positions = load_positions()
    positions.append({
        "ticker": ticker if ticker.endswith(".NS") else f"{ticker}.NS",
        "name": name,
        "entry": float(entry),
        "qty": int(qty),
        "stop": float(stop),
        "high_since_entry": float(entry),
        "date": date or datetime.now().strftime("%Y-%m-%d"),
    })
    save_positions(positions)
    return positions


def remove_position(index: int) -> list:
    positions = load_positions()
    if 0 <= index < len(positions):
        positions.pop(index)
        save_positions(positions)
    return positions


def update_high(ticker: str, price: float):
    """Ratchet the recorded high (used for trailing stop)."""
    positions = load_positions()
    changed = False
    for p in positions:
        if p["ticker"] == ticker and price > p.get("high_since_entry", 0):
            p["high_since_entry"] = float(price)
            changed = True
    if changed:
        save_positions(positions)


def monitor_positions(positions: list, data: dict, exit_signal_fn,
                      technical_score_fn, bench) -> list:
    """
    For each held position, fetch fresh technicals from `data` and evaluate the
    exit signal. Returns enriched rows for display. Persists updated highs/stops.
    """
    rows = []
    changed = False
    for p in positions:
        t = p["ticker"]
        d = (data or {}).get(t)
        price_df = d.get("price") if d else None

        # My Positions is independent of the latest screener universe. Fetch
        # any missing holding directly, and upgrade short scan histories so
        # stage/MA/exit calculations always have enough data.
        if price_df is None or len(price_df) < 200:
            fallback_price = fetch_price(t, "2y")
            if fallback_price is not None:
                d = {"price": fallback_price}

        if not d or d.get("price") is None:
            rows.append({**p, "action": "NO DATA", "reason": "Live price temporarily unavailable",
                         "current": None, "gain_pct": None, "current_stop": p.get("stop")})
            continue

        tech = technical_score_fn(d["price"], bench)
        # keep high ratcheted
        cur = tech.get("price", p["entry"])
        if cur > p.get("high_since_entry", 0):
            p["high_since_entry"] = float(cur); changed = True

        sig = exit_signal_fn(p, tech)
        # persist updated stop
        if sig.get("current_stop") and sig["current_stop"] != p.get("stop"):
            p["current_stop"] = sig["current_stop"]; changed = True

        invested = p["entry"] * p["qty"]
        cur_val = cur * p["qty"]
        rows.append({
            **p,
            "current": round(cur, 2),
            "current_stop": sig.get("current_stop"),
            "gain_pct": sig.get("gain_pct"),
            "pnl_amt": round(cur_val - invested),
            "action": sig.get("action"),
            "reason": sig.get("reason"),
            "stage": tech.get("stage", (0, "?"))[1] if isinstance(tech.get("stage"), tuple) else "?",
            "trailing": sig.get("trailing", False),
        })

    if changed:
        save_positions(positions)
    return rows


def portfolio_summary(rows: list) -> dict:
    invested = sum(r["entry"] * r["qty"] for r in rows if r.get("qty"))
    current = sum((r.get("current") or r["entry"]) * r["qty"] for r in rows if r.get("qty"))
    exits = sum(1 for r in rows if r.get("action") == "EXIT")
    trims = sum(1 for r in rows if r.get("action") == "TRIM")
    return {
        "invested": round(invested),
        "current": round(current),
        "pnl": round(current - invested),
        "pnl_pct": round((current / invested - 1) * 100, 1) if invested else 0,
        "n": len(rows),
        "exits": exits,
        "trims": trims,
    }
