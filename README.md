# bitbank-binance-bot

A Python trading bot that follows [bitbank.nz](https://bitbank.nz) signals and executes trades on Binance. Uses a work-stealing strategy that watches all coin pairs simultaneously, entering positions opportunistically when price comes within 5 basis points of the predicted buy price -- but only for pairs showing positive 7-day PnL.

## How It Works

1. **Signal Fetch**: Polls bitbank.nz API for trading signals (buy/sell prices, 7d PnL, confidence)
2. **Filter**: Only considers pairs with positive 7-day reported PnL and sufficient spread to cover fees
3. **Work-Stealing Entry**: Watches all tradable pairs. When current price drops within 5bp of the predicted buy price, it enters. If capital is fully allocated, it can "steal" capital from orders that are furthest from execution and have lower expected PnL
4. **Exit**: Sells when price reaches within 5bp of the predicted sell target
5. **Re-evaluation**: On each new position entry, re-selects the best pairs by 7d PnL

## Setup

### 1. Clone

```bash
git clone git@github.com:lee101/bitbank-binance-bot.git
cd bitbank-binance-bot
```

### 2. Python Environment

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure API Keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

#### Bitbank API Key

1. Go to [bitbank.nz](https://bitbank.nz)
2. Create an account or log in
3. Navigate to the API page to generate your API key
4. Set `BITBANK_API_KEY` in `.env`

The bot uses the `/api/trading-bot/public-signals-latest` endpoint which returns buy/sell prices and 7-day PnL for each trading pair.

#### Binance API Key

##### Non-US Users (binance.com)

1. Log in to [binance.com](https://www.binance.com)
2. Go to **Account** -> **API Management**
3. Create a new API key
4. Enable **Spot Trading** permissions (do NOT enable withdrawals)
5. Optionally restrict to your IP address
6. Set in `.env`:
   ```
   BINANCE_API_KEY=your_key
   BINANCE_API_SECRET=your_secret
   BINANCE_BASE_URL=https://api.binance.com
   ```

##### US Users (binance.us)

1. Log in to [binance.us](https://www.binance.us)
2. Go to **API Management**
3. Create a new API key with **Spot Trading** enabled
4. Set in `.env`:
   ```
   BINANCE_API_KEY=your_key
   BINANCE_API_SECRET=your_secret
   BINANCE_BASE_URL=https://api.binance.us
   ```

Note: Binance.US has a more limited set of trading pairs. The bot will skip pairs not available on your exchange.

### 4. Configure Trading Parameters

In `.env`:

```bash
# Starting capital in USDT
TRADING_CAPITAL=1000.0

# Only trade pairs with positive 7d PnL (set higher for more selective)
MIN_PNL_7D_PCT=0.0

# Entry tolerance -- how close price must be to buy target (in basis points)
# 5bp = 0.05%. Lower = more selective, higher = more fills
ENTRY_TOLERANCE_BP=5

# IMPORTANT: Start with dry run to verify behavior
DRY_RUN=true
```

## Running

### Dry Run (recommended first)

```bash
DRY_RUN=true python bot.py
```

This logs what trades would execute without placing real orders. Watch the output to verify the strategy behaves as expected.

### Live Trading

```bash
DRY_RUN=false python bot.py
```

### Running Tests

```bash
pytest -v
```

### Backtesting

To backtest the strategy against historical data:

```python
from config import Config
from simulator import Backtester, DataFeed

config = Config()
bt = Backtester(config=config)

# Load your data (CSV with timestamp, open, high, low, close, volume)
feed = DataFeed()
bars = feed.load_hourly("BTCUSDT")

# Define buy/sell prices (one per bar, or one per day for daily signals)
buy_prices = [bar.close * 0.998 for bar in bars]  # example: 0.2% below close
sell_prices = [bar.close * 1.003 for bar in bars]  # example: 0.3% above close

# Run across 7d, 30d, 60d with 5bp and 15bp bar margins
suite = bt.run("BTCUSDT", buy_prices, sell_prices, bars=bars)

for row in suite.summary():
    print(f"{row['key']}: return={row['return_pct']:.2f}% wr={row['win_rate']:.1%} dd={row['max_dd_pct']:.1f}%")

# Check if strategy meets viability thresholds
print(f"Viable: {suite.is_strategy_viable()}")
```

The simulator uses hourly bar granularity even for daily trading strategies, allowing it to detect intraday bounces that cause multiple executions within a single day. Bar margins (5bp/15bp) model the uncertainty of whether a limit order would actually fill at a given price level.

### Historical Data

The bot automatically saves hourly kline data from Binance to `data/hourly/` for future backtesting. You can also symlink training data from other projects:

```bash
# If you have the stock repo with training data
ln -s ~/code/stock/trainingdata data/daily
ln -s ~/code/stock/trainingdatahourly data/hourly_stock
```

## Architecture

```
bot.py                  Main loop: fetch signals -> evaluate -> execute -> save
strategy.py             Work-stealing strategy with capital allocation
bitbank_client.py       Bitbank.nz API client (signals + coins)
binance_client.py       Binance trading client (US + non-US)
config.py               Configuration from environment
simulator/
  market_simulator.py   Core simulator with bar margin execution modeling
  backtest.py           Multi-period, multi-margin backtester
  data_feed.py          Historical data loading and saving
tests/
  test_strategy.py      Strategy logic, work stealing, filtering
  test_simulator.py     Execution modeling, PnL math, bar margins
  test_backtest.py      Backtester periods, viability checks
  test_bitbank_client.py Signal parsing
```

## Strategy Details

### Work Stealing

When capital is fully allocated across positions, a new high-conviction signal can "steal" capital from the worst-performing active order. Orders are ranked by:
1. Distance from execution (furthest first)
2. Expected PnL (lowest first)

Orders within 1bp of execution are protected from stealing. A fight detector prevents excessive stealing (max 5 steals per 10 minutes).

### Execution Realism

The market simulator accounts for:
- **Bar margins**: A 5bp buffer means the price must come within 5bp of the target to count as a fill. This prevents overly optimistic backtests where limit orders fill at exact prices
- **Round-trip fees**: Configurable per-side fee rate (default 10bp for Binance spot)
- **Intraday bounces**: Hourly simulation detects when price passes through both buy and sell levels within the same bar, correctly modeling multiple executions per day
- **Forced closes**: Open positions are marked to market at period end

## License

MIT
