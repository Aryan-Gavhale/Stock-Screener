"""Deterministic tests for sector rotation's point-in-time guarantees."""

import copy
import unittest

import numpy as np
import pandas as pd

from sector_data import build_synthetic_index
from sector_rotation import compute_sector_forecasts, seasonality_signal


def _price(seed: int, drift: float, periods=900):
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.01, periods)
    close = 100 * np.cumprod(1 + returns)
    index = pd.bdate_range("2020-01-01", periods=periods)
    return pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995,
        "Close": close, "Volume": np.full(periods, 1_000_000),
    }, index=index)


class SectorRotationTests(unittest.TestCase):
    def setUp(self):
        self.bench = _price(1, 0.0002)
        self.dataset = {
            "Alpha": {
                "price": _price(2, 0.0005), "source": "Official",
                "members": [], "coverage_members": 0, "breadth": {},
                "fundamentals": {},
            },
            "Beta": {
                "price": _price(3, 0.0001), "source": "Official",
                "members": [], "coverage_members": 0, "breadth": {},
                "fundamentals": {},
            },
            "Gamma": {
                "price": _price(4, -0.0001), "source": "Official",
                "members": [], "coverage_members": 0, "breadth": {},
                "fundamentals": {},
            },
            "Delta": {
                "price": _price(5, 0.0003), "source": "Official",
                "members": [], "coverage_members": 0, "breadth": {},
                "fundamentals": {},
            },
            "Epsilon": {
                "price": _price(6, 0.00025), "source": "Official",
                "members": [], "coverage_members": 0, "breadth": {},
                "fundamentals": {},
            },
        }

    def test_future_mutation_does_not_change_asof_forecast(self):
        asof = self.bench.index[650]
        before = compute_sector_forecasts(
            self.dataset, self.bench, "3M Swing", asof=asof,
            include_live_overlays=False,
        )
        mutated = copy.deepcopy(self.dataset)
        for item in mutated.values():
            item["price"].loc[item["price"].index > asof, "Close"] *= 5
        after = compute_sector_forecasts(
            mutated, self.bench, "3M Swing", asof=asof,
            include_live_overlays=False,
        )
        before_scores = {x["sector"]: x["forward_rotation"] for x in before}
        after_scores = {x["sector"]: x["forward_rotation"] for x in after}
        self.assertEqual(before_scores, after_scores)

    def test_short_history_disables_seasonality(self):
        result = seasonality_signal(
            self.dataset["Alpha"]["price"].tail(300),
            self.bench.tail(300), 63,
        )
        self.assertFalse(result["stable"])
        self.assertEqual(result["score"], 50.0)

    def test_synthetic_index_requires_minimum_members(self):
        data = {
            f"S{i}.NS": {"price": _price(10 + i, 0.0002), "info": {}}
            for i in range(4)
        }
        result = build_synthetic_index(data, list(data), min_members=4)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 500)


if __name__ == "__main__":
    unittest.main()
