from unittest.mock import MagicMock, patch

import pytest

from binance_client import BinanceClient, OrderResult


@pytest.fixture
def dry_client():
    with patch("binance_client.BinanceSDK"):
        return BinanceClient("key", "secret", dry_run=True)


@pytest.fixture
def live_client():
    with patch("binance_client.BinanceSDK") as mock_sdk:
        client = BinanceClient("key", "secret", dry_run=False)
        client._sdk_mock = mock_sdk.return_value
        return client


class TestDryRun:
    def test_buy_returns_filled(self, dry_client):
        r = dry_client.place_limit_buy("BTCUSDT", 1.0, 50000.0)
        assert r.filled
        assert r.order_id == "dry"
        assert r.side == "BUY"

    def test_sell_returns_filled(self, dry_client):
        r = dry_client.place_limit_sell("BTCUSDT", 1.0, 50000.0)
        assert r.filled
        assert r.order_id == "dry"
        assert r.side == "SELL"

    def test_cancel_returns_zero(self, dry_client):
        assert dry_client.cancel_open_orders("BTCUSDT") == 0


class TestQtyRounding:
    def test_rounds_to_step_size(self, dry_client):
        dry_client._symbol_info["BTCUSDT"] = {
            "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"}]
        }
        assert dry_client._round_qty("BTCUSDT", 1.23456) == 1.234

    def test_zero_qty_rejected(self, dry_client):
        r = dry_client.place_limit_buy("BTCUSDT", 0.0, 50000.0)
        assert r.error == "qty too small"
        assert not r.filled

    def test_fallback_step_size(self, dry_client):
        qty = dry_client._round_qty("UNKNOWN", 1.23456789)
        assert qty == pytest.approx(1.23456, abs=0.00001)


class TestPriceFetch:
    def test_get_price(self, live_client):
        live_client._sdk_mock.get_symbol_ticker.return_value = {"price": "42000.50"}
        assert live_client.get_price("BTCUSDT") == 42000.50

    def test_get_price_error_returns_none(self, live_client):
        from binance.exceptions import BinanceAPIException
        live_client._sdk_mock.get_symbol_ticker.side_effect = BinanceAPIException(MagicMock(), 400, "bad")
        assert live_client.get_price("BADUSDT") is None


class TestLiveOrders:
    def test_buy_order_success(self, live_client):
        live_client._sdk_mock.order_limit_buy.return_value = {
            "orderId": 12345, "status": "FILLED"
        }
        r = live_client.place_limit_buy("BTCUSDT", 0.5, 50000.0)
        assert r.order_id == "12345"
        assert r.filled

    def test_sell_order_partial(self, live_client):
        live_client._sdk_mock.order_limit_sell.return_value = {
            "orderId": 67890, "status": "PARTIALLY_FILLED"
        }
        r = live_client.place_limit_sell("BTCUSDT", 0.5, 51000.0)
        assert r.order_id == "67890"
        assert not r.filled

    def test_buy_order_api_error(self, live_client):
        from binance.exceptions import BinanceAPIException
        live_client._sdk_mock.order_limit_buy.side_effect = BinanceAPIException(MagicMock(), 400, "insufficient balance")
        r = live_client.place_limit_buy("BTCUSDT", 1.0, 50000.0)
        assert r.error
        assert not r.filled


class TestTLDDetection:
    def test_us_url(self):
        with patch("binance_client.BinanceSDK") as mock_sdk:
            c = BinanceClient("k", "s", base_url="https://api.binance.us", dry_run=True)
            mock_sdk.assert_called_once_with("k", "s", tld="us")

    def test_com_url(self):
        with patch("binance_client.BinanceSDK") as mock_sdk:
            c = BinanceClient("k", "s", base_url="https://api.binance.com", dry_run=True)
            mock_sdk.assert_called_once_with("k", "s", tld="com")


class TestGetBalances:
    def test_filters_zero_balances(self, live_client):
        live_client._sdk_mock.get_account.return_value = {
            "balances": [
                {"asset": "BTC", "free": "0.5"},
                {"asset": "ETH", "free": "0.0"},
                {"asset": "USDT", "free": "1000.0"},
            ]
        }
        balances = live_client.get_balances()
        assert "BTC" in balances
        assert "USDT" in balances
        assert "ETH" not in balances


class TestGetKlines:
    def test_parses_kline_data(self, live_client):
        live_client._sdk_mock.get_klines.return_value = [
            [1000, "100", "101", "99", "100.5", "500", 2000],
        ]
        klines = live_client.get_klines("BTCUSDT", "1h", 1)
        assert len(klines) == 1
        assert klines[0]["open"] == 100.0
        assert klines[0]["high"] == 101.0
        assert klines[0]["close"] == 100.5
