from typing import List, Dict, Optional, Union
import logging
from decimal import Decimal
from .base_strategy import BaseStrategy
from core.aggregator import Candle
from dataclasses import dataclass, field
from data.pump_data_feed import TradeEvent
from datetime import datetime, timedelta
import pandas as pd
from risk.position import Position
import csv

@dataclass
class Signal:
    is_valid: bool
    price: float
    volume: float
    wallet_address: str = None
    transaction_hash: str = None

class CopyTradingStrategy(BaseStrategy):
    def __init__(self, 
                 tracked_wallets: List[str],
                 min_trade_amount: float = 0.1,  # Minimum SOL amount to consider
                 max_time_in_trade: int = 30,    # Minutes to stay in trade
                 take_profit_pct: float = 0.15,  # 15% take profit
                 logger: logging.Logger = None):
        super().__init__()
        # self.min_trade_amount = min_trade_amount
        # self.max_time_in_trade = max_time_in_trade
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
            
            if wallet_address not in self.tracked_wallets:
                return Signal(is_valid=False, price=0, volume=0)
            
            # Get trade details
            price = float(trade_event.sol_amount) / float(trade_event.token_amount)
            volume = float(trade_event.sol_amount)
            
            # All conditions met - generate valid signal
            return Signal(
                is_valid=True,
                price=price,
                volume=volume,
                wallet_address=wallet_address,
            )
            
        except Exception as e:
            self.logger.error(f"Strategy - Error in generate_signal: {str(e)}")
            return Signal(is_valid=False, price=0, volume=0)

    def check_exit(self, token: str, trade_event: TradeEvent, position: Position, timestamp: datetime, current_price: float) -> Signal:
        """Exit strategy based on time, take profit, and stop loss"""
        try:  
            # # Calculate time elapsed since position entry
            # time_elapsed = timestamp - position.entry_time
            # minutes_elapsed = time_elapsed.total_seconds() / 60
            price = current_price
            volume = float(trade_event.sol_amount)
            wallet_address = trade_event.user.lower()
            pnl_pct = (price - position.entry_price) / position.entry_price
            
            should_exit = False

            # # Exit if we've been in the trade for too long
            # if minutes_elapsed >= self.max_time_in_trade:
            #     should_exit = True

            # Take profit
            if pnl_pct >= self.take_profit_pct:
                should_exit = True
            
            if wallet_address == position.wallet_copy_trading:
                should_exit = True
                
            
            return Signal(
                is_valid=should_exit,
                price=price,
                volume=volume,
                wallet_address=position.wallet_copy_trading,
            )
        except Exception as e:
            self.logger.error(f"Strategy - Error in check_exit: {str(e)}")
            return Signal(is_valid=False, price=0, volume=None) 