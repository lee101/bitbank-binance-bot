import pytest

from simulator.market_simulator import MarketSimulator, SimResult
from simulator.data_feed import Bar


def _make_bars(prices: list[tuple[float, float, float, float]], start_ts: float = 1000.0) -> list[Bar]:
    """Create bars from (open, high, low, close) tuples."""
    return [
        Bar(timestamp=start_ts + i * 3600, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(prices)
    ]


class TestCanExecuteAtBar:
    def test_buy_executes_when_low_reaches_target(self):
        sim = MarketSimulator(bar_margin_bp=5.0)
        bar = Bar(timestamp=0, open=100, high=101, low=99, close=100, volume=100)
        assert sim.can_execute_at_bar(99.5, bar, "buy")

    def test_buy_fails_when_low_too_high(self):
        sim = MarketSimulator(bar_margin_bp=5.0)
        bar = Bar(timestamp=0, open=100, high=101, low=99.5, close=100, volume=100)
        assert not sim.can_execute_at_bar(99.0, bar, "buy")

    def test_sell_executes_when_high_reaches_target(self):
        sim = MarketSimulator(bar_margin_bp=5.0)
        bar = Bar(timestamp=0, open=100, high=101, low=99, close=100, volume=100)
        assert sim.can_execute_at_bar(100.5, bar, "sell")

    def test_sell_fails_when_high_too_low(self):
        sim = MarketSimulator(bar_margin_bp=5.0)
        bar = Bar(timestamp=0, open=100, high=100.2, low=99, close=100, volume=100)
        assert not sim.can_execute_at_bar(100.5, bar, "sell")

    def test_bar_margin_gives_tolerance(self):
        sim = MarketSimulator(bar_margin_bp=50.0)  # 50bp = 0.5%
        bar = Bar(timestamp=0, open=100, high=100, low=100, close=100, volume=100)
        # target buy at 99.6, bar low=100, margin=0.5 -> 100 <= 99.6 + 0.498 = 100.098
        assert sim.can_execute_at_bar(99.6, bar, "buy")

    def test_zero_margin_strict(self):
        sim = MarketSimulator(bar_margin_bp=0.0)
        bar = Bar(timestamp=0, open=100, high=100, low=100, close=100, volume=100)
        assert sim.can_execute_at_bar(100.0, bar, "buy")
        assert not sim.can_execute_at_bar(99.9, bar, "buy")


class TestSimulatePair:
    def test_profitable_round_trip(self):
        sim = MarketSimulator(fee_rate=0.001, bar_margin_bp=5.0, initial_capital=1000.0)
        bars = _make_bars([
            (100, 100.5, 99.5, 100),   # buy at 99.5
            (100, 101.5, 99.8, 101),    # sell at 101
        ])
        result = sim.simulate_pair("TEST", bars, [99.5, 99.5], [101.0, 101.0])
        assert result.net_pnl > 0
        assert result.wins == 1
        assert result.num_round_trips >= 1

    def test_losing_trade(self):
        sim = MarketSimulator(fee_rate=0.001, bar_margin_bp=5.0, initial_capital=1000.0)
        bars = _make_bars([
            (100, 100.5, 99.5, 100),    # buy at 100.2
            (100, 100.3, 99.0, 99.5),   # sell at 100.1 (below entry after fees)
        ])
        result = sim.simulate_pair("TEST", bars, [100.2, 100.2], [100.1, 100.1])
        # sell price < buy price -> should not trade (filtered by assertion in strategy)
        # but simulator handles it - positions close at loss
        assert result.total_return_pct <= 0 or result.num_trades == 0

    def test_no_execution_outside_range(self):
        sim = MarketSimulator(fee_rate=0.001, bar_margin_bp=5.0, initial_capital=1000.0)
        bars = _make_bars([
            (100, 100.1, 99.9, 100),
            (100, 100.1, 99.9, 100),
        ])
        # buy at 98, sell at 102 - neither reachable
        result = sim.simulate_pair("TEST", bars, [98.0, 98.0], [102.0, 102.0])
        assert result.num_trades == 0
        assert result.total_return_pct == pytest.approx(0.0)

    def test_same_bar_round_trip(self):
        sim = MarketSimulator(fee_rate=0.001, bar_margin_bp=5.0, initial_capital=1000.0)
        # wide range bar that hits both buy and sell
        bars = _make_bars([
            (100, 102, 98, 100),
        ])
        result = sim.simulate_pair("TEST", bars, [99.0], [101.0])
        assert result.num_round_trips >= 1
        assert result.num_trades >= 2  # buy + sell

    def test_forced_close_at_end(self):
        sim = MarketSimulator(fee_rate=0.001, bar_margin_bp=5.0, initial_capital=1000.0)
        bars = _make_bars([
            (100, 100.5, 99.5, 100),  # buy executes
            (100, 100.2, 99.8, 100),  # sell never hits
        ])
        result = sim.simulate_pair("TEST", bars, [99.5, 99.5], [105.0, 105.0])
        # should force close at bar close price
        assert result.num_trades >= 2  # buy + forced sell

    def test_fees_deducted(self):
        sim = MarketSimulator(fee_rate=0.01, bar_margin_bp=5.0, initial_capital=1000.0)
        bars = _make_bars([
            (100, 101, 99, 100),
            (100, 102, 99, 101),
        ])
        result = sim.simulate_pair("TEST", bars, [99.5, 99.5], [101.5, 101.5])
        assert result.total_fees > 0
        assert result.gross_pnl > result.net_pnl or result.num_trades == 0

    def test_equity_curve_length(self):
        sim = MarketSimulator(fee_rate=0.001, bar_margin_bp=5.0, initial_capital=1000.0)
        bars = _make_bars([(100, 101, 99, 100)] * 10)
        result = sim.simulate_pair("TEST", bars, [99.5] * 10, [101.0] * 10)
        assert len(result.equity_curve) == len(bars) + 1  # initial + each bar

    def test_max_drawdown_calculated(self):
        sim = MarketSimulator(fee_rate=0.001, bar_margin_bp=50.0, initial_capital=1000.0)
        bars = _make_bars([
            (100, 101, 99, 100),
            (99, 100, 95, 96),    # drop while holding
            (96, 102, 95, 101),   # recovery and sell
        ])
        result = sim.simulate_pair("TEST", bars, [99.5, 99.5, 99.5], [101.0, 101.0, 101.0])
        assert result.max_drawdown_pct >= 0

    def test_daily_prices_with_hourly_bars(self):
        sim = MarketSimulator(fee_rate=0.001, bar_margin_bp=5.0, initial_capital=1000.0)
        # 48 hourly bars = 2 days
        bars = _make_bars([(100, 101, 99, 100)] * 48)
        # 2 daily buy/sell prices
        result = sim.simulate_pair("TEST", bars, [99.5, 99.5], [101.0, 101.0])
        assert result.period_days == 2
        assert result.num_trades > 0


class TestSimResult:
    def test_win_rate(self):
        r = SimResult(symbol="T", period_days=7, bar_margin_bp=5, total_return_pct=1,
                      gross_pnl=10, net_pnl=8, total_fees=2, num_trades=10,
                      num_round_trips=5, wins=3, losses=2, max_drawdown_pct=5)
        assert r.win_rate == pytest.approx(0.6)

    def test_avg_trade_pnl(self):
        r = SimResult(symbol="T", period_days=7, bar_margin_bp=5, total_return_pct=1,
                      gross_pnl=10, net_pnl=8, total_fees=2, num_trades=10,
                      num_round_trips=4, wins=3, losses=1, max_drawdown_pct=5)
        assert r.avg_trade_pnl == pytest.approx(2.0)

    def test_zero_round_trips(self):
        r = SimResult(symbol="T", period_days=7, bar_margin_bp=5, total_return_pct=0,
                      gross_pnl=0, net_pnl=0, total_fees=0, num_trades=0,
                      num_round_trips=0, wins=0, losses=0, max_drawdown_pct=0)
        assert r.win_rate == 0
        assert r.avg_trade_pnl == 0


class TestBarMarginComparison:
    def test_wider_margin_more_fills(self):
        bars = _make_bars([(100, 100.2, 99.8, 100)] * 24)
        sim_tight = MarketSimulator(fee_rate=0.001, bar_margin_bp=1.0, initial_capital=1000.0)
        sim_wide = MarketSimulator(fee_rate=0.001, bar_margin_bp=50.0, initial_capital=1000.0)
        r_tight = sim_tight.simulate_pair("T", bars, [99.5] * 24, [100.5] * 24)
        r_wide = sim_wide.simulate_pair("T", bars, [99.5] * 24, [100.5] * 24)
        assert r_wide.num_trades >= r_tight.num_trades
