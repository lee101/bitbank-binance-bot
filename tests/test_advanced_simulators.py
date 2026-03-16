import math
import pytest

from simulator.monte_carlo import MonteCarloSimulator, MonteCarloResult
from simulator.portfolio_simulator import PortfolioSimulator, PortfolioResult
from simulator.walk_forward import WalkForwardValidator, WalkForwardResult
from simulator.data_feed import Bar


def _make_bars(n: int, base: float = 100.0, amp: float = 1.5) -> list[Bar]:
    bars = []
    for i in range(n):
        p = base + amp * math.sin(i * 0.2)
        bars.append(Bar(
            timestamp=1000 + i * 3600,
            open=p, high=p + amp, low=p - amp, close=p, volume=100,
        ))
    return bars


class TestMonteCarlo:
    def test_basic_run(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        mc = MonteCarloSimulator(bars, bp, sp)
        result = mc.run(n_simulations=50, seed=42)
        assert result.n_simulations == 50
        assert len(result.returns) == 50

    def test_prob_positive(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        mc = MonteCarloSimulator(bars, bp, sp)
        result = mc.run(n_simulations=100, seed=42)
        assert 0 <= result.prob_positive <= 1

    def test_confidence_interval(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        mc = MonteCarloSimulator(bars, bp, sp)
        result = mc.run(n_simulations=100, seed=42)
        assert result.percentile_5 <= result.median_return <= result.percentile_95

    def test_summary_string(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        mc = MonteCarloSimulator(bars, bp, sp)
        result = mc.run(n_simulations=20, seed=42)
        s = result.summary()
        assert "Monte Carlo" in s
        assert "mean return" in s

    def test_deterministic_with_seed(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        mc1 = MonteCarloSimulator(bars, bp, sp)
        mc2 = MonteCarloSimulator(bars, bp, sp)
        r1 = mc1.run(n_simulations=20, seed=123)
        r2 = mc2.run(n_simulations=20, seed=123)
        assert r1.returns == pytest.approx(r2.returns, rel=1e-6)


class TestPortfolioSimulator:
    def test_single_pair(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        ps = PortfolioSimulator(initial_capital=1000)
        result = ps.simulate(
            {"BTC": bars}, {"BTC": bp}, {"BTC": sp}, {"BTC": 2.0}
        )
        assert len(result.equity_curve) > 0
        assert "BTC" in result.per_pair_returns

    def test_multi_pair(self):
        bars1 = _make_bars(7 * 24, base=100)
        bars2 = _make_bars(7 * 24, base=50)
        ps = PortfolioSimulator(initial_capital=1000, max_pair_allocation=0.5)
        result = ps.simulate(
            {"BTC": bars1, "ETH": bars2},
            {"BTC": [b.close * 0.985 for b in bars1], "ETH": [b.close * 0.985 for b in bars2]},
            {"BTC": [b.close * 1.015 for b in bars1], "ETH": [b.close * 1.015 for b in bars2]},
            {"BTC": 3.0, "ETH": 2.0},
        )
        assert "BTC" in result.per_pair_returns
        assert "ETH" in result.per_pair_returns
        assert len(result.equity_curve) > 0

    def test_correlation_computation(self):
        bars1 = _make_bars(48, base=100)
        bars2 = _make_bars(48, base=50)
        ps = PortfolioSimulator()
        corr = ps.compute_correlations({"A": bars1, "B": bars2})
        assert "A" in corr
        assert "B" in corr["A"]
        assert -1.0 <= corr["A"]["B"] <= 1.0
        assert corr["A"]["A"] == pytest.approx(1.0)

    def test_capital_allocation_by_pnl(self):
        ps = PortfolioSimulator(initial_capital=1000, max_pair_allocation=0.5)
        allocs = ps.allocate_capital(["A", "B"], {"A": 5.0, "B": 1.0})
        assert allocs["A"] > allocs["B"]  # A has higher pnl

    def test_drawdown_tracked(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        ps = PortfolioSimulator(initial_capital=1000)
        result = ps.simulate({"BTC": bars}, {"BTC": bp}, {"BTC": sp})
        assert result.max_drawdown_pct >= 0

    def test_empty_pairs(self):
        ps = PortfolioSimulator(initial_capital=1000)
        result = ps.simulate({}, {}, {})
        assert result.total_return_pct == 0
        assert result.equity_curve == [1000]


class TestWalkForward:
    def test_basic_walk_forward(self):
        bars = _make_bars(60 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        wf = WalkForwardValidator(bars, bp, sp)
        result = wf.run(
            train_hours=14 * 24,
            test_hours=7 * 24,
            step_hours=7 * 24,
            opt_maxiter=3,
            opt_popsize=3,
        )
        assert len(result.windows) > 0
        assert 0 <= result.consistency_ratio <= 1

    def test_summary_output(self):
        bars = _make_bars(60 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        wf = WalkForwardValidator(bars, bp, sp)
        result = wf.run(train_hours=14 * 24, test_hours=7 * 24, opt_maxiter=2, opt_popsize=3)
        s = result.summary()
        assert "Walk-Forward" in s
        assert "consistency" in s

    def test_insufficient_data(self):
        bars = _make_bars(48)
        bp = [b.close * 0.99 for b in bars]
        sp = [b.close * 1.01 for b in bars]
        wf = WalkForwardValidator(bars, bp, sp)
        result = wf.run(train_hours=30 * 24, test_hours=7 * 24)
        assert len(result.windows) == 0
        assert result.consistency_ratio == 0

    def test_windows_non_overlapping_test(self):
        bars = _make_bars(60 * 24)
        bp = [b.close * 0.985 for b in bars]
        sp = [b.close * 1.015 for b in bars]
        wf = WalkForwardValidator(bars, bp, sp)
        result = wf.run(train_hours=14 * 24, test_hours=7 * 24, step_hours=7 * 24, opt_maxiter=2, opt_popsize=3)
        for i in range(1, len(result.windows)):
            assert result.windows[i].test_start >= result.windows[i - 1].test_start
