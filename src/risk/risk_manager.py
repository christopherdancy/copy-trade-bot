from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional
from .position import Position
from utils.config import RiskParameters
from utils.logger import TradingLogger
import pandas as pd
from asyncio import Lock

class RiskManager:
    def __init__(self,
                 initial_capital: float,
                 risk_params: Optional[RiskParameters] = None,
                 logger: Optional[TradingLogger] = None):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_params = risk_params or RiskParameters()
        self.logger = logger or TradingLogger("risk_manager")
        
        self.positions: Dict[str, Position] = {}
        self.daily_pnl: float = 0
        self.last_reset = datetime.now()
        self.position_lock = Lock()  # Using asyncio.Lock instead of threading.Lock
        # Add trade history tracking
        self.trade_history: Dict[str, datetime] = {}
    
    async def can_enter_position(self, token: str) -> Tuple[bool, float]:
        """Check if we can enter a new position and reserve the slot if possible"""
        async with self.position_lock:
            # Check cooldown period
            # if token in self.trade_history:
            #     # time_since_last_trade = datetime.now(timezone.utc) - self.trade_history[token]
            #     # cooldown_minutes = self.risk_params.trade_cooldown_minutes
            #     # if time_since_last_trade < timedelta(minutes=cooldown_minutes):
            #     self.logger.debug(f"Token {token} in cooldown period.")
            #     return False, 0.0
            
            # Single source of truth for position checks
            if token in self.positions:
                self.logger.debug(f"Token {token} already in positions")
                return False, 0.0
            
            if len(self.positions) >= self.risk_params.max_positions:
                self.logger.debug(f"Max positions reached: {len(self.positions)}")
                return False, 0.0
            
            # Use fixed position size
            position_size = self.risk_params.max_position_size

            # Check if we have enough capital
            if position_size > self.current_capital:
                self.logger.debug(f"Insufficient capital: {self.current_capital:.4f} SOL")
                return False, 0.0
            
            # # Check daily loss limit
            # if self.daily_pnl < -(self.initial_capital * self.risk_params.max_daily_loss_pct):
            #     self.logger.debug(f"Daily loss limit reached: {self.daily_pnl:.4f} SOL")
            #     return False, 0.0
            
            # Reserve the position slot with a placeholder
            self.positions[token] = Position(
                is_active=False,  # Mark as inactive until confirmed
                token=token,
                entry_blocktime=0,
                position_size=position_size,
                token_amount=0.0,
                entry_price=0.0,
                tx_fees=0.0,
                tx_sig="",
                wallet_copy_trading="",
            )
            
            return True, position_size
    
    async def add_position(
        self, 
        token: str, 
        blocktime: int, 
        sol_trade_amount: float, 
        token_trade_amount: float, 
        execution_price: float, 
        tx_fees: float, 
        tx_sig: str,
        total_sol_change: float,
        wallet_copy_trading: str
    ):
        """Update the reserved position with actual entry details"""
        async with self.position_lock:
            try:
                if token not in self.positions:
                    self.logger.warning(f"No reserved position found for {token}")
                    return
            
                if self.positions[token].is_active:
                    self.logger.warning(f"Position already active for {token}")
                    return

                # Update trade history with exit timestamp
                # self.trade_history[token] = datetime.now(timezone.utc)

                # Update the reserved position with actual entry details
                self.positions[token] = Position(
                    is_active=True,
                    token=token,
                    entry_blocktime=blocktime,
                    position_size=float(sol_trade_amount),
                    token_amount=float(token_trade_amount),
                    entry_price=float(execution_price),
                    tx_fees=float(tx_fees),
                    tx_sig=str(tx_sig),
                    wallet_copy_trading=str(wallet_copy_trading)
                )
                self.current_capital -= float(total_sol_change)
                self.logger.info(f"Added position: {token}, Position Size: {sol_trade_amount}, Token Amount: {token_trade_amount}, Tx Fees: {tx_fees}, current capital: {self.current_capital}")
            except Exception as e:
                self.logger.error(f"RiskManager - Error entering position: {token}, {e}")
    
    async def exit_position(self, token: str, price: float, sol_received: float) -> float:
        """Exit a position and return PnL"""
        async with self.position_lock:
            try:
                if token not in self.positions:
                    self.logger.warning(f"No position found for {token}")
                    return 0.0
            
                position = self.positions[token]
                if not position.is_active:
                    self.logger.warning(f"Position not active for {token}")
                    return 0.0

                pnl = position.calculate_pnl(price)
                
                # Update capital and daily PnL
                self.current_capital += sol_received
                self.daily_pnl += pnl
                
                
                # Remove position
                del self.positions[token]
                self.logger.info(f"Exited position: {token}, PnL: {pnl}, Current Capital: {self.current_capital}")
                return pnl
            except Exception as e:
                self.logger.error(f"RiskManager - Error exiting position: {token}, {e}")
                return 0.0
    
    def check_stop_loss(self, token: str, current_price: float) -> bool:
        """Check if position has hit stop loss"""
        position = self.positions.get(token)
        if not position:
            return False

        # Calculate percentage drop from entry
        pnl_pct = (current_price - position.entry_price) / position.entry_price

        if pnl_pct <= -self.risk_params.stop_loss_pct:
            return True
        return False
    
    def reset_daily_pnl(self):
        """Reset daily PnL counter"""
        now = datetime.now()
        if (now - self.last_reset).days >= 1:
            self.daily_pnl = 0
            self.last_reset = now 
    
    def has_position(self, token: str) -> bool:
        """Check if we have an active position for the given token"""
        if token not in self.positions:
            return False
            
        position = self.positions[token]
        
        return position.is_active

    def get_position_info(self, token: str) -> Optional[Position]:
        """Get position information if it exists"""
        return self.positions.get(token)

    def get_current_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.positions.copy()

    async def remove_reserved_position(self, token: str):
        """Remove a reserved position if the entry fails"""
        async with self.position_lock:
            if token in self.positions and not self.positions[token].is_active:
                del self.positions[token]