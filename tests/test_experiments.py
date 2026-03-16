import math
import pytest

from experiments.strategies import (
    PassthroughStrategy, MomentumFilterStrategy, MeanReversionStrategy,
    MultiTimeframeStrategy, VolatilityScaledStrategy, EnsembleStrategy,
    ALL_STRATEGIES, StrategyConfig,
)
from experiments.runner import run_experiments, ExperimentSuite, ExperimentResult
from simulator.data_feed import Bar


def _make_bars(n: int, base: float = 100.0) -> list[Bar]:
    bars = []
    for i in range(n):
        p = base + 2 * math.sin(i * 0.15) + 0.5 * math.sin(i * 0.6)
        bars.append(Bar(
            timestamp=1000 + i * 3600,
            open=p, high=p + 1.5, low=p - 1.5, close=p, volume=100,
        ))
    return bars


class TestPassthrough:
    def test_returns_same_prices(self):
        bars = _make_bars(48)
        bp = [99.0] * 48
        sp = [101.0] * 48
        bp_out, sp_out = PassthroughStrategy().transform_signals(bars, bp, sp)
        assert bp_out == bp
        assert sp_out == sp

    def test_run_produces_result(self):
        bars = _make_bars(48)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        result = PassthroughStrategy().run(bars, bp, sp)
        assert result.symbol == "passthrough"


class TestMomentumFilter:
    def test_disables_buy_in_downtrend(self):
        # create downtrend
        bars = [Bar(1000 + i * 3600, 100 - i, 101 - i, 99 - i, 100 - i, 100) for i in range(24)]
        bp = [90.0] * 24
        sp = [110.0] * 24
        strat = MomentumFilterStrategy(fast_period=4, slow_period=12)
        bp_out, _ = strat.transform_signals(bars, bp, sp)
        # some buys should be disabled
        disabled = sum(1 for b in bp_out if b == 0)
        assert disabled > 0

    def test_keeps_buy_in_uptrend(self):
        bars = [Bar(1000 + i * 3600, 100 + i, 101 + i, 99 + i, 100 + i, 100) for i in range(24)]
        bp = [90.0] * 24
        sp = [110.0] * 24
        strat = MomentumFilterStrategy(fast_period=4, slow_period=12)
        bp_out, _ = strat.transform_signals(bars, bp, sp)
        active = sum(1 for b in bp_out if b > 0)
        assert active >= 12  # most should remain active


class TestMeanReversion:
    def test_widens_spread_in_high_vol(self):
        # high vol bars
        bars = []
        for i in range(48):
            p = 100 + 5 * math.sin(i * 0.8)
            bars.append(Bar(1000 + i * 3600, p, p + 3, p - 3, p, 100))
        bp = [99.0] * 48
        sp = [101.0] * 48
        strat = MeanReversionStrategy(lookback=24, vol_mult=2.0)
        bp_out, sp_out = strat.transform_signals(bars, bp, sp)
        # buy should be lower, sell should be higher in high vol
        assert bp_out[-1] < bp[-1]
        assert sp_out[-1] > sp[-1]


class TestMultiTimeframe:
    def test_filters_double_bearish(self):
        bars = [Bar(1000 + i * 3600, 100 - i * 0.5, 101 - i * 0.5, 99 - i * 0.5, 100 - i * 0.5, 100)
                for i in range(48)]
        bp = [80.0] * 48
        sp = [120.0] * 48
        strat = MultiTimeframeStrategy(short_period=6, long_period=24)
        bp_out, _ = strat.transform_signals(bars, bp, sp)
        disabled = sum(1 for b in bp_out if b == 0)
        assert disabled > 0


class TestVolatilityScaled:
    def test_scales_spread(self):
        bars = _make_bars(72)
        bp = [b.close * 0.99 for b in bars]
        sp = [b.close * 1.01 for b in bars]
        strat = VolatilityScaledStrategy(lookback=48, base_vol=0.01)
        bp_out, sp_out = strat.transform_signals(bars, bp, sp)
        assert len(bp_out) == len(bp)
        # spreads should be different from original
        diffs = sum(1 for i in range(48, len(bp)) if abs(bp_out[i] - bp[i]) > 0.01)
        assert diffs > 0


class TestEnsemble:
    def test_averages_strategies(self):
        bars = _make_bars(48)
        bp = [b.close * 0.99 for b in bars]
        sp = [b.close * 1.01 for b in bars]
        strat = EnsembleStrategy()
        bp_out, sp_out = strat.transform_signals(bars, bp, sp)
        assert len(bp_out) == len(bp)

    def test_requires_majority(self):
        bars = _make_bars(48)
        bp = [b.close * 0.99 for b in bars]
        sp = [b.close * 1.01 for b in bars]
        strat = EnsembleStrategy()
        bp_out, _ = strat.transform_signals(bars, bp, sp)
        # should have some active buys
        active = sum(1 for b in bp_out if b > 0)
        assert active > 0


class TestExperimentRunner:
    def test_run_all_strategies(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        suite = run_experiments(
            bars, bp, sp,
            periods=[7],
            bar_margins=[5.0],
        )
        assert len(suite.results) > 0
        assert len(suite.results) >= len(ALL_STRATEGIES)

    def test_best_by_combined(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        suite = run_experiments(bars, bp, sp, periods=[7], bar_margins=[5.0])
        best = suite.best_by_combined_score(top_n=3)
        assert len(best) <= 3

    def test_table_output(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        suite = run_experiments(bars, bp, sp, periods=[7], bar_margins=[5.0])
        table = suite.to_table()
        assert "strategy" in table
        assert "ret%" in table

    def test_insufficient_data_skips(self):
        bars = _make_bars(24)
        bp = [b.close * 0.99 for b in bars]
        sp = [b.close * 1.01 for b in bars]
        suite = run_experiments(bars, bp, sp, periods=[7, 30])
        assert len(suite.results) == 0
