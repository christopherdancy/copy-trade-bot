from typing import List, Dict, Optional, Union
import logging
from decimal import Decimal
from .base_strategy import BaseStrategy
from dataclasses import dataclass, field
from data.pump_data_feed import TradeEvent
from datetime import datetime, timedelta
import pandas as pd
from core.position_tracker import Position
import csv

@dataclass
class Signal:
    is_valid: bool
    wallet_address: str = None
    transaction_hash: str = None

class CopyTradingStrategy(BaseStrategy):
    def __init__(self, 
                 tracked_wallets: List[str],
                 take_profit_pct: float = 0.15,  # 15% take profit
                 logger: logging.Logger = None):
        super().__init__()
        self.take_profit_pct = take_profit_pct
        self.logger = logger
        
        # Load tracked wallets from CSV
        self.tracked_wallets = tracked_wallets

    def generate_signal(self, trade_event: TradeEvent) -> Signal:
        """
        Generate a signal based on a trade event from a tracked wallet
        """
        try:
            # Check if the trade is from a tracked wallet
            wallet_address = trade_event.user.lower()
            
            # Add debug logging to trace wallet address checking
            is_tracked = wallet_address in [w.lower() for w in self.tracked_wallets]
            
            if not is_tracked:
                return Signal(is_valid=False)
                
            return Signal(
                is_valid=True,
                wallet_address=wallet_address,
            )
            
        except Exception as e:
            self.logger.error(f"Strategy - Error in generate_signal: {str(e)}")
            return Signal(is_valid=False)

    def check_exit(self, token: str, trade_wallet: str, wallet_followed: str, entry_price: float, current_price: float) -> Signal:
        """Exit strategy based on time, take profit, and stop loss"""
        try:  
            should_exit = False
            
            # Take profit
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct >= self.take_profit_pct:
                should_exit = True
            
            # Followed wallet
            wallet_followed = wallet_followed.lower()
            trade_wallet = trade_wallet.lower()
            
            if trade_wallet == wallet_followed:
                should_exit = True
                
            return Signal(
                is_valid=should_exit,
                wallet_address=wallet_followed,
            )
        except Exception as e:
            self.logger.error(f"Strategy - Error in check_exit: {str(e)}")
            return Signal(is_valid=False) 