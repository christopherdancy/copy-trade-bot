from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, List
from core.position_tracker import Position
from utils.logger import TradingLogger
import pandas as pd
from asyncio import Lock

class RiskManager:
    def __init__(self,
                 initial_capital: float,
                 risk_params: Dict,
                 logger: Optional[TradingLogger] = None):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_params = risk_params
        self.logger = logger or TradingLogger("risk_manager")
        
        self.daily_pnl: float = 0
        self.last_reset = datetime.now()
        self.position_lock = Lock()  # Keeping this for capital and PnL updates
    
    async def can_enter_position(self, token: str, current_pending_positions: int) -> Tuple[bool, float]:
        """Check if we can enter a new position based on current positions and risk parameters"""
        # Check max positions limit
        if current_pending_positions >= self.risk_params['max_positions']:
            self.logger.debug(f"Max positions reached: {current_pending_positions}")
            return False, 0.0
        
        # Use fixed position size
        position_size = self.risk_params['max_position_size']

        # Check if we have enough capital
        if position_size > self.current_capital:
            self.logger.debug(f"Insufficient capital: {self.current_capital:.4f} SOL")
            return False, 0.0
        
        return True, position_size
    
    def check_stop_loss(self, entry_price: float, current_price: float) -> bool:
        """Check if position has hit stop loss"""
        # Calculate percentage drop from entry
        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct <= -self.risk_params['stop_loss_pct']:
            return True
        return False

    async def update_capital_after_entry(self, total_sol_change: float):
        """Update capital after a position entry"""
        async with self.position_lock:
            self.current_capital -= float(total_sol_change)
    
    async def update_capital_after_exit(self, sol_received: float):
        """Update capital after a position exit"""
        async with self.position_lock:
            self.current_capital += sol_received
    
    def update_capital(self, new_capital: float):
        """Update the current capital amount directly"""
        self.current_capital = new_capital
        self.initial_capital = new_capital  # Update initial capital reference as well
        self.logger.info(f"Capital updated to {new_capital:.4f} SOL")
    
    def reset_daily_pnl(self):
        """Reset daily PnL counter"""
        now = datetime.now()
        if (now - self.last_reset).days >= 1:
            self.daily_pnl = 0
            self.last_reset = now 