"""Strategy variants built on top of bitbank forecasts."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from simulator.data_feed import Bar
from simulator.market_simulator import MarketSimulator, SimResult

log = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    fee_rate: float = 0.001
    bar_margin_bp: float = 5.0
    initial_capital: float = 1000.0


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def transform_signals(
        self,
        bars: list[Bar],
        buy_prices: list[float],
        sell_prices: list[float],
    ) -> tuple[list[float], list[float]]:
        """Transform raw bitbank signals into strategy-specific buy/sell prices."""
        ...

    def run(
        self,
        bars: list[Bar],
        buy_prices: list[float],
        sell_prices: list[float],
        config: StrategyConfig | None = None,
    ) -> SimResult:
        config = config or StrategyConfig()
        bp, sp = self.transform_signals(bars, buy_prices, sell_prices)
        sim = MarketSimulator(config.fee_rate, config.bar_margin_bp, config.initial_capital)
        return sim.simulate_pair(self.name, bars, bp, sp)


class PassthroughStrategy(BaseStrategy):
    """Use bitbank signals as-is. Baseline."""
    name = "passthrough"

    def transform_signals(self, bars, buy_prices, sell_prices):
        return buy_prices, sell_prices


class MomentumFilterStrategy(BaseStrategy):
    """Only trade when short-term momentum aligns with signal direction.
    Uses 4-bar and 12-bar moving averages on closes.
    """
    name = "momentum_filter"

    def __init__(self, fast_period: int = 4, slow_period: int = 12):
        self.fast = fast_period
        self.slow = slow_period

    def transform_signals(self, bars, buy_prices, sell_prices):
        closes = [b.close for b in bars]
        n = len(closes)
        bp_out = list(buy_prices)
        sp_out = list(sell_prices)

        for i in range(self.slow, n):
            fast_ma = np.mean(closes[i - self.fast:i])
            slow_ma = np.mean(closes[i - self.slow:i])
            # only buy when fast > slow (uptrend)
            if fast_ma < slow_ma:
                idx = i if len(buy_prices) == n else min(i // 24, len(buy_prices) - 1)
                bp_out[idx] = 0.0  # disable buy
        return bp_out, sp_out


class MeanReversionStrategy(BaseStrategy):
    """Widen buy/sell targets based on recent volatility.
    Buy lower and sell higher when volatility is high.
    """
    name = "mean_reversion"

    def __init__(self, lookback: int = 24, vol_mult: float = 1.5):
        self.lookback = lookback
        self.vol_mult = vol_mult

    def transform_signals(self, bars, buy_prices, sell_prices):
        closes = [b.close for b in bars]
        n = len(closes)
        per_bar = len(buy_prices) == n
        bp_out = list(buy_prices)
        sp_out = list(sell_prices)

        for i in range(self.lookback, n):
            window = closes[i - self.lookback:i]
            vol = np.std(window) / np.mean(window) if np.mean(window) > 0 else 0
            adjustment = vol * self.vol_mult
            idx = i if per_bar else min(i // 24, len(buy_prices) - 1)
            if idx < len(bp_out):
                bp_out[idx] = buy_prices[idx] * (1 - adjustment)
                sp_out[idx] = sell_prices[idx] * (1 + adjustment)
        return bp_out, sp_out


class MultiTimeframeStrategy(BaseStrategy):
    """Confirm signals across multiple timeframes.
    Only trades when hourly and 4h trends agree.
    """
    name = "multi_timeframe"

    def __init__(self, short_period: int = 6, long_period: int = 24):
        self.short = short_period
        self.long = long_period

    def transform_signals(self, bars, buy_prices, sell_prices):
        closes = [b.close for b in bars]
        n = len(closes)
        per_bar = len(buy_prices) == n
        bp_out = list(buy_prices)
        sp_out = list(sell_prices)

        for i in range(self.long, n):
            short_ret = (closes[i - 1] - closes[i - self.short]) / closes[i - self.short] if closes[i - self.short] > 0 else 0
            long_ret = (closes[i - 1] - closes[i - self.long]) / closes[i - self.long] if closes[i - self.long] > 0 else 0
            idx = i if per_bar else min(i // 24, len(buy_prices) - 1)
            # disable buy if both timeframes bearish
            if short_ret < 0 and long_ret < 0 and idx < len(bp_out):
                bp_out[idx] = 0.0
            # disable sell if both timeframes bullish (let it ride)
            if short_ret > 0.005 and long_ret > 0.005 and idx < len(sp_out):
                sp_out[idx] = sp_out[idx] * 1.5  # raise sell target
        return bp_out, sp_out


class VolatilityScaledStrategy(BaseStrategy):
    """Scale position entry/exit based on realized volatility.
    Tighten spread in low vol, widen in high vol.
    """
    name = "vol_scaled"

    def __init__(self, lookback: int = 48, base_vol: float = 0.01):
        self.lookback = lookback
        self.base_vol = base_vol

    def transform_signals(self, bars, buy_prices, sell_prices):
        closes = [b.close for b in bars]
        n = len(closes)
        per_bar = len(buy_prices) == n
        bp_out = list(buy_prices)
        sp_out = list(sell_prices)

        for i in range(self.lookback, n):
            returns = []
            for j in range(i - self.lookback + 1, i + 1):
                if closes[j - 1] > 0:
                    returns.append((closes[j] - closes[j - 1]) / closes[j - 1])
            vol = np.std(returns) if returns else self.base_vol
            scale = vol / self.base_vol if self.base_vol > 0 else 1.0
            scale = max(0.5, min(scale, 3.0))  # clamp

            idx = i if per_bar else min(i // 24, len(buy_prices) - 1)
            if idx < len(bp_out):
                mid = (buy_prices[idx] + sell_prices[idx]) / 2
                half_spread = (sell_prices[idx] - buy_prices[idx]) / 2
                bp_out[idx] = mid - half_spread * scale
                sp_out[idx] = mid + half_spread * scale
        return bp_out, sp_out


class EnsembleStrategy(BaseStrategy):
    """Combine multiple strategies by averaging their buy/sell prices.
    Only trades when majority of strategies agree on direction.
    """
    name = "ensemble"

    def __init__(self, strategies: list[BaseStrategy] | None = None):
        self.strategies = strategies or [
            PassthroughStrategy(),
            MomentumFilterStrategy(),
            MeanReversionStrategy(),
            VolatilityScaledStrategy(),
        ]

    def transform_signals(self, bars, buy_prices, sell_prices):
        all_buys = []
        all_sells = []
        for strat in self.strategies:
            bp, sp = strat.transform_signals(bars, buy_prices, sell_prices)
            all_buys.append(bp)
            all_sells.append(sp)

        n = len(buy_prices)
        bp_out = []
        sp_out = []
        for i in range(n):
            buys = [b[i] for b in all_buys if i < len(b)]
            sells = [s[i] for s in all_sells if i < len(s)]
            # only trade if majority have non-zero buy
            active = sum(1 for b in buys if b > 0)
            if active >= len(self.strategies) / 2:
                valid_buys = [b for b in buys if b > 0]
                bp_out.append(np.mean(valid_buys) if valid_buys else 0.0)
                sp_out.append(np.mean(sells))
            else:
                bp_out.append(0.0)
                sp_out.append(sells[0] if sells else 0.0)
        return bp_out, sp_out


ALL_STRATEGIES: list[BaseStrategy] = [
    PassthroughStrategy(),
    MomentumFilterStrategy(),
    MomentumFilterStrategy(fast_period=6, slow_period=24),
    MeanReversionStrategy(),
    MeanReversionStrategy(lookback=48, vol_mult=2.0),
    MultiTimeframeStrategy(),
    VolatilityScaledStrategy(),
    VolatilityScaledStrategy(lookback=72, base_vol=0.015),
    EnsembleStrategy(),
]
