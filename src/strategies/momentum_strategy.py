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
import logging


@dataclass
class Signal:
    is_valid: bool
    price: float
    volume: float

class PureMomentumStrategy(BaseStrategy):
    def __init__(self,
                 min_price: float = 1.25e-7,     # 25k mcap
                 max_price: float = 2.5e-7,      # 50k mcap 
                 min_volume_threshold: float = 0.001,    
                 lookback: int = 3,
                 max_time_in_trade: int = 5,
                 take_profit_pct: float = 0.2,
                 logger: logging.Logger = None):
        super().__init__()
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume_threshold = min_volume_threshold
        self.max_time_in_trade = max_time_in_trade
        self.lookback = lookback
        self.logger = logger

    def generate_signal(self, candles: List[Candle]) -> Signal:
        """
        Analyze newly completed candle against previous candle for signal
        Returns True if entry conditions are met
        """
        try:
            if len(candles) < self.lookback: 
                return Signal(is_valid=False, price=0, volume=0)
                
            # Get the last two candles
            prev_candle = candles[-2]
            curr_candle = candles[-1]
            
            # Skip if current candle has no volume (empty minute)
            if curr_candle.volume == 0:
                return Signal(is_valid=False, price=0, volume=0)
                
            # Calculate key metrics
            current_price = float(curr_candle.close)
            total_volume = float(curr_candle.volume)

            if not (self.min_price <= current_price <= self.max_price):
                return Signal(is_valid=False, price=current_price, volume=total_volume)
            
            # Check all conditions
            is_valid = (
                self.analyze_volume_pattern(candles)
            )
                
            return Signal(
                is_valid=is_valid,
                price=current_price,
                volume=total_volume,
            )

        except Exception as e:
            self.logger.error(f"candles: {candles}")
            self.logger.error(f"Strategy - Error in generate_signal: {str(e)}")
            return Signal(is_valid=False, price=0, volume=0)

    def analyze_volume_pattern(self, candles: List[Candle]) -> bool:
        """
        Check if volume is stable and trending across multiple candles
        """
            
        volumes = [float(c.volume) for c in candles[-self.lookback:]]
        prices = [float(c.close) for c in candles[-self.lookback:]]

        
        # 1. Check Base Volume Threshold
        current_volume = volumes[-1]
        if current_volume < self.min_volume_threshold:
            return False
        
        # 3. Progressive Growth
        # Add protection for zero volumes
        growth_rates = []
        for i in range(len(volumes)-1):
            if volumes[i] == 0:  # Protect against division by zero
                growth_rates.append(0)
            else:
                growth_rates.append((volumes[i+1] - volumes[i])/volumes[i])

        price_changes = []
        for i in range(len(prices)-1):
            price_changes.append((prices[i+1] - prices[i])/prices[i])

        # Only check growth if we have valid rates
        # TODO: Add a price direction check... volume spread analysis + wyckoff method
        # TODO: Check for extreme growth spikes > 1000%
        is_growing = (len(growth_rates) >= 2 and  # Need at least 2 rates to compare
                    all(rate > 0 for rate in growth_rates) and  # All must be positive
                    all(growth_rates[i+1] >= growth_rates[i]
                        for i in range(len(growth_rates)-1)) and 
                    all(change > 0 for change in price_changes))
    
        return (
            is_growing            # Stable or growing
        )

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
            )
        except Exception as e:
            self.logger.error(f"Strategy - Error in check_exit: {str(e)}")
            return Signal(is_valid=False, price=0, volume=None)