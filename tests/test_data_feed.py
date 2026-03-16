import csv
from pathlib import Path

import pytest

from simulator.data_feed import Bar, DataFeed


@pytest.fixture
def feed(tmp_path):
    return DataFeed(data_dir=tmp_path)


def _write_csv(path, rows, header=("timestamp", "open", "high", "low", "close", "volume")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


class TestSaveLoadRoundTrip:
    def test_save_and_load(self, feed, tmp_path):
        bars = [
            Bar(timestamp=1000, open=100, high=101, low=99, close=100.5, volume=50),
            Bar(timestamp=4600, open=100.5, high=102, low=100, close=101, volume=60),
        ]
        feed.save_bars("BTCUSDT", bars, granularity="hourly")
        loaded = feed.load_hourly("BTCUSDT")
        assert len(loaded) == 2
        assert loaded[0].open == 100.0
        assert loaded[1].close == 101.0

    def test_save_creates_subdirectory(self, feed, tmp_path):
        feed.save_bars("ETHUSDT", [Bar(0, 1, 2, 0.5, 1.5, 10)], granularity="hourly")
        assert (tmp_path / "hourly" / "ETHUSDT.csv").exists()


class TestColumnMapping:
    def test_alternate_column_names(self, feed, tmp_path):
        path = tmp_path / "TESTUSDT.csv"
        _write_csv(path, [(1000, 50, 55, 45, 52, 100)],
                   header=("Date", "Open", "High", "Low", "Close", "Vol"))
        bars = feed._load_csv(path)
        assert len(bars) == 1
        assert bars[0].high == 55.0
        assert bars[0].volume == 100.0

    def test_datetime_string_timestamp(self, feed, tmp_path):
        path = tmp_path / "DTTEST.csv"
        _write_csv(path, [("2024-01-01 00:00:00", 100, 101, 99, 100, 50)],
                   header=("datetime", "open", "high", "low", "close", "volume"))
        bars = feed._load_csv(path)
        assert bars[0].timestamp > 0


class TestFindFile:
    def test_finds_direct_match(self, feed, tmp_path):
        path = tmp_path / "BTCUSDT.csv"
        _write_csv(path, [(1000, 100, 101, 99, 100, 50)])
        result = feed._find_file("BTCUSDT", "hourly")
        assert result == path

    def test_finds_in_subdir(self, feed, tmp_path):
        subdir = tmp_path / "hourly"
        subdir.mkdir()
        path = subdir / "BTCUSDT.csv"
        _write_csv(path, [(1000, 100, 101, 99, 100, 50)])
        result = feed._find_file("BTCUSDT", "hourly")
        assert result == path

    def test_strips_underscores(self, feed, tmp_path):
        path = tmp_path / "BTCUSDT.csv"
        _write_csv(path, [(1000, 100, 101, 99, 100, 50)])
        result = feed._find_file("BTC_USDT", "hourly")
        assert result == path

    def test_missing_raises(self, feed):
        with pytest.raises(FileNotFoundError):
            feed._find_file("NOSUCHPAIR", "hourly")


class TestBarsToDataframe:
    def test_converts(self, feed):
        bars = [Bar(1000, 100, 101, 99, 100, 50)]
        df = feed.bars_to_dataframe(bars)
        assert len(df) == 1
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
