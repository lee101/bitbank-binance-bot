import json

import httpx
import pytest

from bitbank_client import BitbankClient, Signal


@pytest.fixture
def mock_signals_response():
    return {
        "signals": [
            {
                "currency_pair": "BTC_USDT",
                "buy_price": "42000.0",
                "sell_price": "42500.0",
                "confidence": "0.85",
                "signal_type": "buy",
                "pnl_7d_pct": "3.5",
                "trades_7d": "15",
                "win_rate_7d": "0.7",
            },
            {
                "currency_pair": "ETH_USDT",
                "buy_price": "2200.0",
                "sell_price": "2240.0",
                "confidence": "0.6",
                "signal_type": "buy",
                "pnl_7d_pct": "1.2",
                "trades_7d": "8",
                "win_rate_7d": "0.5",
            },
        ]
    }


class TestFetchSignals:
    @pytest.mark.asyncio
    async def test_parses_signals(self, mock_signals_response):
        client = BitbankClient("https://fake.test")
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=mock_signals_response)
        )
        client._client = httpx.AsyncClient(transport=transport)
        signals = await client.fetch_signals()
        assert len(signals) == 2
        assert signals[0].pair == "BTC_USDT"
        assert signals[0].buy_price == 42000.0
        assert signals[0].pnl_7d_pct == 3.5
        await client.close()

    @pytest.mark.asyncio
    async def test_handles_list_response(self):
        raw = [
            {
                "currency_pair": "SOL_USDT",
                "buy_price": "100",
                "sell_price": "102",
                "pnl_7d_pct": "2.0",
                "trades_7d": "5",
                "win_rate_7d": "0.6",
            }
        ]
        client = BitbankClient("https://fake.test")
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=raw)
        )
        client._client = httpx.AsyncClient(transport=transport)
        signals = await client.fetch_signals()
        assert len(signals) == 1
        assert signals[0].binance_symbol == "SOLUSDT"
        await client.close()

    @pytest.mark.asyncio
    async def test_skips_malformed(self):
        data = {"signals": [
            {"currency_pair": "BTC_USDT"},  # missing buy_price
            {"currency_pair": "ETH_USDT", "buy_price": "100", "sell_price": "101",
             "pnl_7d_pct": "1", "trades_7d": "5", "win_rate_7d": "0.5"},
        ]}
        client = BitbankClient("https://fake.test")
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=data)
        )
        client._client = httpx.AsyncClient(transport=transport)
        signals = await client.fetch_signals()
        assert len(signals) == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_api_key_header(self):
        def check_header(req):
            assert req.headers.get("X-API-Key") == "testkey"
            return httpx.Response(200, json={"signals": []})

        client = BitbankClient("https://fake.test", api_key="testkey")
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(check_header))
        await client.fetch_signals()
        await client.close()

    @pytest.mark.asyncio
    async def test_no_header_without_key(self):
        def check_no_header(req):
            assert "X-API-Key" not in req.headers
            return httpx.Response(200, json={"signals": []})

        client = BitbankClient("https://fake.test")
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(check_no_header))
        await client.fetch_signals()
        await client.close()
