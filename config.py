import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    bitbank_api_key: str = field(default_factory=lambda: os.getenv("BITBANK_API_KEY", ""))
    bitbank_base_url: str = field(default_factory=lambda: os.getenv("BITBANK_BASE_URL", "https://bitbank.nz"))

    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    binance_base_url: str = field(default_factory=lambda: os.getenv("BINANCE_BASE_URL", "https://api.binance.com"))

    capital: float = field(default_factory=lambda: float(os.getenv("TRADING_CAPITAL", "1000.0")))
    min_pnl_7d_pct: float = field(default_factory=lambda: float(os.getenv("MIN_PNL_7D_PCT", "0.0")))
    entry_tolerance_bp: float = field(default_factory=lambda: float(os.getenv("ENTRY_TOLERANCE_BP", "5")))
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "true").lower() == "true")

    poll_interval_seconds: int = 60
    fee_rate: float = 0.001  # 10bp per side (Binance spot maker/taker)
    protection_bp: float = 1.0  # orders within 1bp of execution are protected from stealing
    cooldown_seconds: int = 120
    max_position_pct: float = 0.25  # max 25% capital per position

    @property
    def entry_tolerance_pct(self) -> float:
        return self.entry_tolerance_bp / 10000.0

    @property
    def is_binance_us(self) -> bool:
        return "binance.us" in self.binance_base_url
