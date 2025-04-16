from datetime import datetime
from dataclasses import dataclass
@dataclass
class Position:
    """Represents an open trading position"""
    is_active: bool
    token: str
    entry_blocktime: int
    position_size: float
    token_amount: float = 0.0
    entry_price: float = 0.0
    tx_fees: float = 0.0
    tx_sig: str = ""
    wallet_copy_trading: str = ""
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        return (current_price - self.entry_price) * self.token_amount 