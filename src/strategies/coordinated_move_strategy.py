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

@dataclass
class Signal:
    is_valid: bool
    price: float
    volume: float
    trigger_move: float 
    confirmation_move: float
    base_candle: Candle | None
    trigger_candle: Candle | None
    confirmation_candle: Candle | None

class CoordinatedMoveStrategy(BaseStrategy):
    def __init__(self, 
                 min_price: float = 1.25e-7,     # 25k mcap
                 max_price: float = 2.5e-7,      # 50k mcap
                 min_price_move: float = 0.20,   # 20% minimum price move
                 max_price_move: float = 0.40,   # 40% maximum price move
                 max_confirmation_move: float = 0.19,   # 20% maximum confirmation move
                 max_volume_threshold: float = 30,
                 max_time_in_trade: int = 5,
                 take_profit_pct: float = 0.10,
                 lookback: int = 3,              # Need three candles now
                 logger: logging.Logger = None):
        super().__init__()
        self.min_price = min_price
        self.max_price = max_price
        self.min_price_move = min_price_move
        self.max_price_move = max_price_move
        self.max_confirmation_move = max_confirmation_move
        self.max_volume_threshold = max_volume_threshold
        self.max_time_in_trade = max_time_in_trade
        self.take_profit_pct = take_profit_pct
        self.early_window_minutes = 2
        
        self.lookback = lookback
        self.logger = logger

    def generate_signal(self, candles: List[Candle]) -> Signal:
        """
        Analyze price movement with confirmation candle
        """
        try:
            if len(candles) < self.lookback:
                return Signal(is_valid=False, price=0, volume=0)
                
            # Get the three candles
            base_candle = candles[-3]
            trigger_candle = candles[-2]
            confirmation_candle = candles[-1]

            # Validate prices are not zero
            if float(base_candle.close) <= 0 or float(trigger_candle.close) <= 0 or float(confirmation_candle.close) <= 0:
                return Signal(is_valid=False, price=0, volume=0, trigger_move=0, confirmation_move=0, base_candle=None, trigger_candle=None, confirmation_candle=None)
            
            # Calculate key metrics
            trigger_price = float(trigger_candle.close)
            base_price = float(base_candle.close)
            current_price = float(confirmation_candle.close)
            
            # Calculate moves
            trigger_move = (trigger_price - base_price) / base_price
            confirmation_move = (current_price - trigger_price) / trigger_price
            
            # Check price range
            if not (self.min_price <= trigger_price <= self.max_price):
                return Signal(is_valid=False, price=current_price, volume=trigger_candle.volume, trigger_move=trigger_move, confirmation_move=confirmation_move, base_candle=base_candle, trigger_candle=trigger_candle, confirmation_candle=confirmation_candle)
                
            # Check trigger move range
            if not (self.min_price_move <= trigger_move <= self.max_price_move):
                return Signal(is_valid=False, price=current_price, volume=trigger_candle.volume, trigger_move=trigger_move, confirmation_move=confirmation_move, base_candle=base_candle, trigger_candle=trigger_candle, confirmation_candle=confirmation_candle)
                
            # Check volume threshold
            if float(trigger_candle.volume) > self.max_volume_threshold:
                return Signal(is_valid=False, price=current_price, volume=trigger_candle.volume, trigger_move=trigger_move, confirmation_move=confirmation_move, base_candle=base_candle, trigger_candle=trigger_candle, confirmation_candle=confirmation_candle)
                
            # Check confirmation candle is positive
            if confirmation_move < .07 or confirmation_move > self.max_confirmation_move:
                return Signal(is_valid=False, price=current_price, volume=trigger_candle.volume, trigger_move=trigger_move, confirmation_move=confirmation_move, base_candle=base_candle, trigger_candle=trigger_candle, confirmation_candle=confirmation_candle)
            # Unique_buyer_filter
            if base_candle.unique_buyers >= 9:
                return Signal(is_valid=False, price=current_price, volume=trigger_candle.volume, trigger_move=trigger_move, confirmation_move=confirmation_move, base_candle=base_candle, trigger_candle=trigger_candle, confirmation_candle=confirmation_candle)

            # All conditions met
            return Signal(
                is_valid=True,
                price=current_price,
                volume=float(trigger_candle.volume),
                trigger_move=trigger_move,
                confirmation_move=confirmation_move,
                base_candle=base_candle,
                trigger_candle=trigger_candle,
                confirmation_candle=confirmation_candle
            )
        except Exception as e:
            self.logger.error(f"candles: {candles}")
            self.logger.error(f"Strategy - Error in generate_signal: {str(e)}")
            return Signal(is_valid=False, price=0, volume=0)

    def check_exit(self, token: str, current_price: float, position: Position, timestamp: datetime) -> Signal:
        """Exit strategy focused on trailing stops at different profit levels"""
        try:  
            # Calculate time elapsed since position entry
            time_elapsed = timestamp - position.entry_time
            minutes_elapsed = time_elapsed.total_seconds() / 60

            pnl = (current_price - position.entry_price) / position.entry_price
            
            should_exit = False

            # Exit if we've been in the trade for too long
            if minutes_elapsed >= self.max_time_in_trade:
                should_exit = True

            # If we EVER reached these profit levels, protect them
            if pnl >= self.take_profit_pct: 
                should_exit = True
            
            return Signal(
                is_valid=should_exit,
                price=current_price,
                volume=None,
                trigger_move=None,
                confirmation_move=None,
                base_candle=None,
                trigger_candle=None,
                confirmation_candle=None
            )
        except Exception as e:
            self.logger.error(f"Strategy - Error in check_exit: {str(e)}")
            return Signal(is_valid=False, price=0, volume=None)