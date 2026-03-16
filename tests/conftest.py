import pytest
from bitbank_client import Signal
from config import Config
from simulator.data_feed import Bar


@pytest.fixture
def cfg():
    c = Config()
    c.capital = 1000.0
    c.fee_rate = 0.001
    c.entry_tolerance_bp = 5
    c.min_pnl_7d_pct = 0.0
    c.max_position_pct = 0.25
    c.cooldown_seconds = 0
    c.dry_run = True
    return c


def make_signal(pair="BTC_USDT", buy=100.0, sell=101.0, pnl_7d=2.0, conf=0.8) -> Signal:
    return Signal(
        pair=pair, buy_price=buy, sell_price=sell,
        confidence=conf, signal_type="buy",
        pnl_7d_pct=pnl_7d, trades_7d=10, win_rate_7d=0.6,
    )


def make_bars(prices, start_ts=1000.0):
    """Create bars from (open, high, low, close) tuples."""
    return [
        Bar(timestamp=start_ts + i * 3600, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(prices)
    ]


def make_flat_bars(n, base_price=100.0):
    bars = []
    for i in range(n):
        p = base_price + (i % 5) * 0.5 - 1.0
        bars.append(Bar(
            timestamp=1000.0 + i * 3600,
            open=p, high=p + 1.0, low=p - 1.0, close=p, volume=100.0,
        ))
    return bars
