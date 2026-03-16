import pytest

from config import Config
from simulator.backtest import Backtester, BacktestSuite
from simulator.data_feed import Bar, DataFeed
from simulator.market_simulator import SimResult


def _make_bars(n: int, base_price: float = 100.0) -> list[Bar]:
    bars = []
    for i in range(n):
        p = base_price + (i % 5) * 0.5 - 1.0
        bars.append(Bar(
            timestamp=1000.0 + i * 3600,
            open=p, high=p + 1.0, low=p - 1.0, close=p, volume=100.0,
        ))
    return bars


class TestBacktester:
    def test_run_7d(self):
        cfg = Config()
        cfg.capital = 1000.0
        cfg.fee_rate = 0.001
        bt = Backtester(config=cfg)
        bars = _make_bars(7 * 24)
        buy_prices = [99.5] * len(bars)
        sell_prices = [101.0] * len(bars)
        suite = bt.run("TEST", buy_prices, sell_prices, bars=bars, periods=[7], bar_margins=[5.0])
        assert len(suite.results) == 1
        key = list(suite.results.keys())[0]
        assert "7d" in key
        assert "5" in key

    def test_run_multiple_periods(self):
        cfg = Config()
        cfg.capital = 1000.0
        cfg.fee_rate = 0.001
        bt = Backtester(config=cfg)
        bars = _make_bars(60 * 24)
        buy_prices = [99.5] * len(bars)
        sell_prices = [101.0] * len(bars)
        suite = bt.run("TEST", buy_prices, sell_prices, bars=bars, periods=[7, 30, 60], bar_margins=[5.0])
        assert len(suite.results) == 3

    def test_run_multiple_margins(self):
        cfg = Config()
        cfg.capital = 1000.0
        cfg.fee_rate = 0.001
        bt = Backtester(config=cfg)
        bars = _make_bars(7 * 24)
        buy_prices = [99.5] * len(bars)
        sell_prices = [101.0] * len(bars)
        suite = bt.run("TEST", buy_prices, sell_prices, bars=bars, periods=[7], bar_margins=[5.0, 15.0])
        assert len(suite.results) == 2

    def test_insufficient_data_skips(self):
        cfg = Config()
        cfg.capital = 1000.0
        bt = Backtester(config=cfg)
        bars = _make_bars(24)  # only 1 day
        buy_prices = [99.5] * len(bars)
        sell_prices = [101.0] * len(bars)
        suite = bt.run("TEST", buy_prices, sell_prices, bars=bars, periods=[7, 30])
        assert len(suite.results) == 0

    def test_summary(self):
        cfg = Config()
        cfg.capital = 1000.0
        cfg.fee_rate = 0.001
        bt = Backtester(config=cfg)
        bars = _make_bars(7 * 24)
        buy_prices = [99.5] * len(bars)
        sell_prices = [101.0] * len(bars)
        suite = bt.run("TEST", buy_prices, sell_prices, bars=bars, periods=[7], bar_margins=[5.0])
        rows = suite.summary()
        assert len(rows) >= 1
        assert "return_pct" in rows[0]
        assert "win_rate" in rows[0]


class TestBacktestSuiteViability:
    def test_viable_strategy(self):
        suite = BacktestSuite()
        suite.results["test"] = [SimResult(
            symbol="T", period_days=30, bar_margin_bp=5, total_return_pct=5.0,
            gross_pnl=50, net_pnl=45, total_fees=5, num_trades=20,
            num_round_trips=10, wins=6, losses=4, max_drawdown_pct=10,
        )]
        assert suite.is_strategy_viable()

    def test_not_viable_low_win_rate(self):
        suite = BacktestSuite()
        suite.results["test"] = [SimResult(
            symbol="T", period_days=30, bar_margin_bp=5, total_return_pct=5.0,
            gross_pnl=50, net_pnl=45, total_fees=5, num_trades=20,
            num_round_trips=10, wins=2, losses=8, max_drawdown_pct=10,
        )]
        assert not suite.is_strategy_viable()

    def test_not_viable_high_drawdown(self):
        suite = BacktestSuite()
        suite.results["test"] = [SimResult(
            symbol="T", period_days=30, bar_margin_bp=5, total_return_pct=5.0,
            gross_pnl=50, net_pnl=45, total_fees=5, num_trades=20,
            num_round_trips=10, wins=6, losses=4, max_drawdown_pct=25,
        )]
        assert not suite.is_strategy_viable()

    def test_not_viable_negative_return_30d(self):
        suite = BacktestSuite()
        suite.results["test"] = [SimResult(
            symbol="T", period_days=30, bar_margin_bp=5, total_return_pct=-2.0,
            gross_pnl=-10, net_pnl=-20, total_fees=10, num_trades=20,
            num_round_trips=10, wins=6, losses=4, max_drawdown_pct=10,
        )]
        assert not suite.is_strategy_viable()
