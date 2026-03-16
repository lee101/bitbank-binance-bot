import time

import pytest

from bitbank_client import Signal
from config import Config
from strategy import WorkStealingStrategy


def _make_signal(pair="BTC_USDT", buy=100.0, sell=101.0, pnl_7d=2.0, conf=0.8) -> Signal:
    return Signal(
        pair=pair, buy_price=buy, sell_price=sell,
        confidence=conf, signal_type="buy",
        pnl_7d_pct=pnl_7d, trades_7d=10, win_rate_7d=0.6,
    )


def _config(**overrides) -> Config:
    c = Config()
    c.capital = overrides.get("capital", 1000.0)
    c.fee_rate = overrides.get("fee_rate", 0.001)
    c.entry_tolerance_bp = overrides.get("entry_tolerance_bp", 5)
    c.min_pnl_7d_pct = overrides.get("min_pnl_7d_pct", 0.0)
    c.max_position_pct = overrides.get("max_position_pct", 0.25)
    c.cooldown_seconds = overrides.get("cooldown_seconds", 0)
    return c


class TestFilterTradable:
    def test_positive_pnl_passes(self):
        s = WorkStealingStrategy(config=_config())
        signals = [_make_signal(pnl_7d=1.0)]
        assert len(s.filter_tradable(signals)) == 1

    def test_negative_pnl_filtered(self):
        s = WorkStealingStrategy(config=_config(min_pnl_7d_pct=0.0))
        signals = [_make_signal(pnl_7d=-1.0)]
        assert len(s.filter_tradable(signals)) == 0

    def test_zero_buy_price_filtered(self):
        s = WorkStealingStrategy(config=_config())
        signals = [_make_signal(buy=0.0)]
        assert len(s.filter_tradable(signals)) == 0

    def test_spread_too_small_filtered(self):
        s = WorkStealingStrategy(config=_config(fee_rate=0.001))
        # spread = (100.1 - 100) / 100 = 0.001, round trip fees = 0.002
        signals = [_make_signal(buy=100.0, sell=100.1)]
        assert len(s.filter_tradable(signals)) == 0

    def test_valid_spread_passes(self):
        s = WorkStealingStrategy(config=_config(fee_rate=0.001))
        # spread = (101 - 100) / 100 = 0.01 > 0.002
        signals = [_make_signal(buy=100.0, sell=101.0)]
        assert len(s.filter_tradable(signals)) == 1


class TestEvaluateSignals:
    def test_buy_within_tolerance(self):
        s = WorkStealingStrategy(config=_config(entry_tolerance_bp=5))
        signals = [_make_signal(buy=100.0, sell=101.0)]
        # price at 100.03 = 3bp above buy target
        actions = s.evaluate_signals(signals, {"BTCUSDT": 100.03})
        assert len(actions) == 1
        assert actions[0]["action"] == "buy"

    def test_buy_outside_tolerance(self):
        s = WorkStealingStrategy(config=_config(entry_tolerance_bp=5))
        signals = [_make_signal(buy=100.0, sell=101.0)]
        # price at 101.0 = 100bp above buy target
        actions = s.evaluate_signals(signals, {"BTCUSDT": 101.0})
        assert len(actions) == 0

    def test_sell_when_holding(self):
        s = WorkStealingStrategy(config=_config(entry_tolerance_bp=5))
        s.positions["BTCUSDT"] = 1.0
        s.entry_prices["BTCUSDT"] = 100.0
        signals = [_make_signal(buy=100.0, sell=101.0)]
        # price near sell target
        actions = s.evaluate_signals(signals, {"BTCUSDT": 101.0})
        assert len(actions) == 1
        assert actions[0]["action"] == "sell"

    def test_no_double_buy(self):
        s = WorkStealingStrategy(config=_config(entry_tolerance_bp=5))
        s.positions["BTCUSDT"] = 1.0
        s.entry_prices["BTCUSDT"] = 100.0
        signals = [_make_signal(buy=100.0, sell=101.0)]
        # price near buy but already holding -> should try sell not buy
        actions = s.evaluate_signals(signals, {"BTCUSDT": 100.0})
        buy_actions = [a for a in actions if a["action"] == "buy"]
        assert len(buy_actions) == 0

    def test_capital_allocation(self):
        cfg = _config(capital=1000.0, max_position_pct=0.25)
        s = WorkStealingStrategy(config=cfg)
        signals = [_make_signal(buy=100.0, sell=101.0)]
        actions = s.evaluate_signals(signals, {"BTCUSDT": 100.0})
        assert len(actions) == 1
        assert actions[0]["notional"] <= 250.0

    def test_sorted_by_pnl(self):
        s = WorkStealingStrategy(config=_config(entry_tolerance_bp=100))
        signals = [
            _make_signal("BTC_USDT", buy=100, sell=102, pnl_7d=1.0),
            _make_signal("ETH_USDT", buy=50, sell=51, pnl_7d=5.0),
        ]
        actions = s.evaluate_signals(signals, {"BTCUSDT": 100.0, "ETHUSDT": 50.0})
        # ETH has higher 7d pnl, should be first
        assert actions[0]["symbol"] == "ETHUSDT"


class TestExecuteTrades:
    def test_buy_updates_state(self):
        s = WorkStealingStrategy(config=_config())
        s.execute_buy("BTCUSDT", 1.0, 100.0, 2.0)
        assert s.positions["BTCUSDT"] == 1.0
        assert s.entry_prices["BTCUSDT"] == 100.0
        assert s.cash < 1000.0

    def test_sell_calculates_pnl(self):
        s = WorkStealingStrategy(config=_config(fee_rate=0.001))
        s.execute_buy("BTCUSDT", 1.0, 100.0, 2.0)
        s.execute_sell("BTCUSDT", 1.0, 105.0)
        assert "BTCUSDT" not in s.positions
        sells = [t for t in s.trade_log if t.side == "SELL"]
        assert len(sells) == 1
        assert sells[0].pnl_pct > 0

    def test_portfolio_value(self):
        s = WorkStealingStrategy(config=_config(capital=1000.0))
        s.execute_buy("BTCUSDT", 1.0, 100.0, 2.0)
        pv = s.portfolio_value({"BTCUSDT": 110.0})
        # started with 1000, spent ~100.1 (with fee), hold 1 BTC at 110
        assert pv > 1000.0


class TestWorkStealing:
    def test_steal_from_worst(self):
        cfg = _config(capital=100.0, max_position_pct=0.5)
        s = WorkStealingStrategy(config=cfg)
        s.execute_buy("BADUSDT", 0.5, 100.0, 0.5)
        # set current_price far from limit to make distance_pct large
        s.active_orders["BADUSDT"].current_price = 101.0  # 1% away

        initial_cash = s.cash
        stolen = s._attempt_steal("GOODUSDT", 5.0, time.time())
        assert stolen
        assert s.cash > initial_cash

    def test_protected_orders_not_stolen(self):
        cfg = _config(capital=100.0, max_position_pct=0.5, entry_tolerance_bp=5)
        cfg.protection_bp = 10
        s = WorkStealingStrategy(config=cfg)
        s.execute_buy("CLOSEUSDT", 0.5, 100.0, 1.0)
        # price very close to limit -> within protection
        s.active_orders["CLOSEUSDT"].current_price = 100.005  # ~0.5bp

        stolen = s._attempt_steal("NEWUSDT", 5.0, time.time())
        assert not stolen

    def test_fight_detection(self):
        cfg = _config(capital=1000.0, max_position_pct=0.1)
        s = WorkStealingStrategy(config=cfg)
        now = time.time()
        s.steal_history = [(now - i, f"A{i}", f"B{i}") for i in range(5)]
        s.execute_buy("TARGETUSDT", 0.5, 100.0, 0.5)
        s.active_orders["TARGETUSDT"].current_price = 105.0  # far away

        stolen = s._attempt_steal("NEWUSDT", 10.0, now)
        assert not stolen


class TestSignalProperties:
    def test_binance_symbol(self):
        sig = _make_signal("BTC_USDT")
        assert sig.binance_symbol == "BTCUSDT"

    def test_spread_pct(self):
        sig = _make_signal(buy=100.0, sell=101.0)
        assert abs(sig.spread_pct - 0.01) < 1e-6
