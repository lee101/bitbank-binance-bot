import pytest

from bitbank_client import Signal


class TestSignal:
    def test_binance_symbol_conversion(self):
        s = Signal(pair="BTC_USDT", buy_price=100, sell_price=101, confidence=0.8,
                   signal_type="buy", pnl_7d_pct=1.0, trades_7d=5, win_rate_7d=0.6)
        assert s.binance_symbol == "BTCUSDT"

    def test_spread_pct(self):
        s = Signal(pair="ETH_USDT", buy_price=2000, sell_price=2020, confidence=0.8,
                   signal_type="buy", pnl_7d_pct=1.0, trades_7d=5, win_rate_7d=0.6)
        assert s.spread_pct == pytest.approx(0.01)

    def test_spread_pct_zero_buy(self):
        s = Signal(pair="X_USDT", buy_price=0, sell_price=100, confidence=0.8,
                   signal_type="buy", pnl_7d_pct=1.0, trades_7d=5, win_rate_7d=0.6)
        assert s.spread_pct == 0.0
