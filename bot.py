"""Main bot loop - fetches signals from bitbank, executes on Binance."""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from binance_client import BinanceClient
from bitbank_client import BitbankClient
from config import Config
from simulator.data_feed import Bar, DataFeed
from strategy import WorkStealingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("bot")

STATE_DIR = Path("state")
STATE_FILE = STATE_DIR / "bot_state.json"
TRADE_LOG = STATE_DIR / "trades.jsonl"


def save_state(strategy: WorkStealingStrategy):
    STATE_DIR.mkdir(exist_ok=True)
    state = {
        "positions": strategy.positions,
        "entry_prices": strategy.entry_prices,
        "cash": strategy.cash,
        "capital_used": strategy.capital_used,
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_state(strategy: WorkStealingStrategy):
    if not STATE_FILE.exists():
        return
    state = json.loads(STATE_FILE.read_text())
    strategy.positions = state.get("positions", {})
    strategy.entry_prices = state.get("entry_prices", {})
    strategy.cash = state.get("cash", strategy.config.capital)
    strategy.capital_used = state.get("capital_used", 0.0)
    log.info("restored state: cash=%.2f positions=%d", strategy.cash, len(strategy.positions))


def log_trade(record: dict):
    STATE_DIR.mkdir(exist_ok=True)
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def collect_and_save_data(binance: BinanceClient, symbols: list[str], feed: DataFeed):
    for sym in symbols:
        klines = binance.get_klines(sym, interval="1h", limit=24)
        if not klines:
            continue
        bars = [
            Bar(
                timestamp=k["open_time"] / 1000,
                open=k["open"], high=k["high"],
                low=k["low"], close=k["close"],
                volume=k["volume"],
            )
            for k in klines
        ]
        feed.save_bars(sym, bars, granularity="hourly")


async def run_loop(config: Config):
    bitbank = BitbankClient(config.bitbank_base_url, config.bitbank_api_key)
    binance = BinanceClient(
        config.binance_api_key, config.binance_api_secret,
        config.binance_base_url, dry_run=config.dry_run,
    )
    feed = DataFeed()
    strategy = WorkStealingStrategy(config=config)
    load_state(strategy)

    log.info("bot started capital=%.2f dry_run=%s tolerance=%dbp",
             config.capital, config.dry_run, config.entry_tolerance_bp)

    while True:
        try:
            signals = await bitbank.fetch_signals()
            log.info("fetched %d signals", len(signals))

            tradable = strategy.filter_tradable(signals)
            log.info("tradable: %d pairs", len(tradable))

            # get current prices for all tradable symbols
            symbols = [s.binance_symbol for s in tradable]
            current_prices = {}
            for sym in symbols:
                p = binance.get_price(sym)
                if p is not None:
                    current_prices[sym] = p

            actions = strategy.evaluate_signals(tradable, current_prices)

            for action in actions:
                sym = action["symbol"]
                sig = action["signal"]

                if action["action"] == "buy":
                    result = binance.place_limit_buy(sym, action["qty"], action["price"])
                    if result.filled or result.order_id:
                        strategy.execute_buy(sym, action["qty"], action["price"], sig.pnl_7d_pct)
                        log.info("BUY %s qty=%.6f @ %.4f reason=%s",
                                 sym, action["qty"], action["price"], action["reason"])
                        log_trade({"event": "buy", "symbol": sym, "qty": action["qty"],
                                   "price": action["price"], "time": time.time()})

                elif action["action"] == "sell":
                    result = binance.place_limit_sell(sym, action["qty"], action["price"])
                    if result.filled or result.order_id:
                        strategy.execute_sell(sym, action["qty"], action["price"])
                        log.info("SELL %s qty=%.6f @ %.4f reason=%s",
                                 sym, action["qty"], action["price"], action["reason"])
                        log_trade({"event": "sell", "symbol": sym, "qty": action["qty"],
                                   "price": action["price"], "time": time.time()})

            save_state(strategy)

            # collect data for future backtesting
            collect_and_save_data(binance, symbols, feed)

            summary = strategy.summary(current_prices)
            log.info("portfolio=%.2f pnl=%.2f%% positions=%d trades=%d wr=%.1f%%",
                     summary["portfolio_value"], summary["total_pnl_pct"],
                     summary["positions"], summary["total_trades"],
                     summary["win_rate"] * 100)

        except Exception as e:
            log.error("loop error: %s", e, exc_info=True)

        await asyncio.sleep(config.poll_interval_seconds)


def main():
    config = Config()
    if not config.bitbank_api_key and not config.bitbank_base_url:
        log.error("set BITBANK_BASE_URL in .env")
        sys.exit(1)
    asyncio.run(run_loop(config))


if __name__ == "__main__":
    main()
