import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
STOCK_TRAINING_DATA = Path(os.getenv("STOCK_TRAINING_DATA", str(Path.home() / "code" / "stock" / "trainingdata")))
STOCK_TRAINING_HOURLY = Path(os.getenv("STOCK_TRAINING_HOURLY", str(Path.home() / "code" / "stock" / "trainingdatahourly")))


@dataclass
class Bar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFeed:
    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def load_daily(self, symbol: str) -> list[Bar]:
        return self._load_csv(self._find_file(symbol, "daily"))

    def load_hourly(self, symbol: str) -> list[Bar]:
        return self._load_csv(self._find_file(symbol, "hourly"))

    def _find_file(self, symbol: str, granularity: str) -> Path:
        clean = symbol.replace("/", "").replace("_", "")
        candidates = [
            self.data_dir / f"{clean}.csv",
            self.data_dir / f"{symbol}.csv",
            self.data_dir / granularity / f"{clean}.csv",
        ]
        if granularity == "daily":
            candidates.append(STOCK_TRAINING_DATA / f"{clean}.csv")
        elif granularity == "hourly":
            candidates.append(STOCK_TRAINING_HOURLY / f"{clean}.csv")
            for subdir in STOCK_TRAINING_HOURLY.iterdir() if STOCK_TRAINING_HOURLY.exists() else []:
                if subdir.is_dir():
                    candidates.append(subdir / f"{clean}.csv")
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"no data for {symbol} ({granularity}), searched: {[str(c) for c in candidates]}")

    def _load_csv(self, path: Path) -> list[Bar]:
        df = pd.read_csv(path)
        col_map = {}
        for col in df.columns:
            lc = col.lower().strip()
            if lc in ("timestamp", "date", "time", "open_time", "datetime"):
                col_map["timestamp"] = col
            elif lc == "open":
                col_map["open"] = col
            elif lc == "high":
                col_map["high"] = col
            elif lc == "low":
                col_map["low"] = col
            elif lc == "close":
                col_map["close"] = col
            elif lc in ("volume", "vol"):
                col_map["volume"] = col

        bars = []
        for _, row in df.iterrows():
            ts_raw = row.get(col_map.get("timestamp", ""), 0)
            if isinstance(ts_raw, str):
                try:
                    ts_raw = pd.Timestamp(ts_raw).timestamp()
                except Exception:
                    ts_raw = 0
            bars.append(Bar(
                timestamp=float(ts_raw),
                open=float(row.get(col_map.get("open", ""), 0)),
                high=float(row.get(col_map.get("high", ""), 0)),
                low=float(row.get(col_map.get("low", ""), 0)),
                close=float(row.get(col_map.get("close", ""), 0)),
                volume=float(row.get(col_map.get("volume", ""), 0)),
            ))
        return bars

    def save_bars(self, symbol: str, bars: list[Bar], granularity: str = "hourly"):
        out_dir = self.data_dir / granularity
        out_dir.mkdir(parents=True, exist_ok=True)
        clean = symbol.replace("/", "").replace("_", "")
        path = out_dir / f"{clean}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for b in bars:
                writer.writerow([b.timestamp, b.open, b.high, b.low, b.close, b.volume])
        log.info("saved %d bars to %s", len(bars), path)

    def bars_to_dataframe(self, bars: list[Bar]) -> pd.DataFrame:
        return pd.DataFrame([
            {"timestamp": b.timestamp, "open": b.open, "high": b.high,
             "low": b.low, "close": b.close, "volume": b.volume}
            for b in bars
        ])
