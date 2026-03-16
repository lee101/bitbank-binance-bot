import json
from pathlib import Path

import pytest

from config import Config
from strategy import WorkStealingStrategy


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("bot.STATE_DIR", tmp_path)
    monkeypatch.setattr("bot.STATE_FILE", tmp_path / "bot_state.json")
    monkeypatch.setattr("bot.TRADE_LOG", tmp_path / "trades.jsonl")
    return tmp_path


class TestSaveLoadState:
    def test_round_trip(self, state_dir):
        from bot import save_state, load_state
        cfg = Config()
        cfg.capital = 500.0
        s = WorkStealingStrategy(config=cfg)
        s.positions = {"BTCUSDT": 0.5}
        s.entry_prices = {"BTCUSDT": 40000.0}
        s.cash = 300.0
        s.capital_used = 200.0

        save_state(s)
        assert (state_dir / "bot_state.json").exists()

        s2 = WorkStealingStrategy(config=cfg)
        load_state(s2)
        assert s2.positions == {"BTCUSDT": 0.5}
        assert s2.entry_prices == {"BTCUSDT": 40000.0}
        assert s2.cash == 300.0
        assert s2.capital_used == 200.0

    def test_load_missing_file(self, state_dir):
        from bot import load_state
        cfg = Config()
        s = WorkStealingStrategy(config=cfg)
        load_state(s)
        assert s.cash == cfg.capital

    def test_save_creates_dir(self, tmp_path, monkeypatch):
        from bot import save_state
        nested = tmp_path / "sub" / "state"
        monkeypatch.setattr("bot.STATE_DIR", nested)
        monkeypatch.setattr("bot.STATE_FILE", nested / "bot_state.json")
        cfg = Config()
        s = WorkStealingStrategy(config=cfg)
        save_state(s)
        assert (nested / "bot_state.json").exists()


class TestLogTrade:
    def test_appends_jsonl(self, state_dir):
        from bot import log_trade
        log_trade({"event": "buy", "symbol": "BTCUSDT", "qty": 1.0})
        log_trade({"event": "sell", "symbol": "BTCUSDT", "qty": 1.0})
        lines = (state_dir / "trades.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "buy"
        assert json.loads(lines[1])["event"] == "sell"


class TestMainValidation:
    def test_requires_api_key_for_live(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("BINANCE_API_KEY", "")
        monkeypatch.setenv("BINANCE_API_SECRET", "")
        from bot import main
        with pytest.raises(SystemExit):
            main()

    def test_dry_run_no_key_ok(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("BINANCE_API_KEY", "")

        import asyncio
        from unittest.mock import patch, AsyncMock

        with patch("bot.run_loop", new_callable=AsyncMock) as mock_loop:
            mock_loop.side_effect = KeyboardInterrupt
            try:
                from bot import main
                main()
            except (KeyboardInterrupt, SystemExit):
                pass
