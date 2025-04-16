from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta
from data.pump_data_feed import TradeEvent
from typing import Optional, List, Tuple
import decimal

@dataclass
class Candle:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    buy_volume: Decimal
    sell_volume: Decimal
    timestamp: datetime
    trade_count: int
    largest_buyer_volume: Decimal = Decimal(0)  # Volume from largest buyer (as percentage)
    unique_buyers: int = 0  # Number of unique buyers
    max_trade_amount: Decimal = Decimal(0)  # Largest single trade
    avg_trade_amount: Decimal = Decimal(0)  # Average trade size


class TokenAggregator:
    def __init__(self, mint: str, timeframe_seconds: int, max_candles: int = 100):
        self.mint = mint
        self.timeframe_seconds = timeframe_seconds
        self.max_candles = max_candles
        self.candles: List[Candle] = []
        self.current_candle: Optional[Candle] = None
        self.last_processed_candle_time: Optional[datetime] = None
        self.processed_txs: Set[str] = set()
        # Add tracking for buyer volumes
        self.current_buyers: Dict[str, Decimal] = defaultdict(Decimal)

    def get_candle_time(self, timestamp: datetime) -> datetime:
        """Normalize timestamp to candle start time"""
        total_seconds = int(timestamp.timestamp())
        normalized_seconds = total_seconds - (total_seconds % self.timeframe_seconds)
        return datetime.fromtimestamp(normalized_seconds)

    def process_trade(self, trade: TradeEvent) -> bool:
        """Returns True if new candle created"""
        if trade.signature in self.processed_txs:
            return False    
        
        self.processed_txs.add(trade.signature)
        candle_time = self.get_candle_time(trade.timestamp)
        
        # Add safety check for token_amount
        if trade.token_amount == 0:
            return False
        
        price = trade.sol_amount / trade.token_amount

        # Create new candle if needed
        if not self.current_candle or candle_time > self.current_candle.timestamp:
            if self.current_candle:
                # Finalize metrics for the closing candle
                self._finalize_candle_metrics()
                self.candles.append(self.current_candle)
                while len(self.candles) > self.max_candles:
                    self.candles.pop(0)
            
            # Reset buyer tracking for new candle
            self.current_buyers.clear()
            
            self.current_candle = Candle(
                open=price,
                high=price,
                low=price,
                close=price,
                volume=trade.sol_amount,
                buy_volume=trade.sol_amount if trade.is_buy else Decimal(0),
                sell_volume=trade.sol_amount if not trade.is_buy else Decimal(0),
                timestamp=candle_time,
                trade_count=1,
                max_trade_amount=trade.sol_amount,
                avg_trade_amount=trade.sol_amount
            )
            
            # Initialize first trade metrics
            if trade.is_buy:
                self.current_buyers[trade.user] += trade.sol_amount
            
            return True
            
        # Update current candle
        self.current_candle.high = max(self.current_candle.high, price)
        self.current_candle.low = min(self.current_candle.low, price)
        self.current_candle.close = price
        self.current_candle.volume += trade.sol_amount
        self.current_candle.trade_count += 1
        self.current_candle.max_trade_amount = max(self.current_candle.max_trade_amount, trade.sol_amount)
        self.current_candle.avg_trade_amount = self.current_candle.volume / self.current_candle.trade_count
        
        if trade.is_buy:
            self.current_candle.buy_volume += trade.sol_amount
            self.current_buyers[trade.user] += trade.sol_amount
            # Update largest buyer and unique buyers count
            self.current_candle.largest_buyer_volume = max(self.current_buyers.values())
            self.current_candle.unique_buyers = len(self.current_buyers)
        else:
            self.current_candle.sell_volume += trade.sol_amount
            
        return False

    def _finalize_candle_metrics(self):
        """Finalize metrics for the current candle before closing"""
        if self.current_candle and self.current_candle.volume > 0:
            # Calculate percentage of volume from largest buyer
            if self.current_candle.buy_volume > 0:
                largest_buyer_pct = (self.current_candle.largest_buyer_volume / 
                                   self.current_candle.buy_volume * 100)
            else:
                largest_buyer_pct = Decimal(0)
            
            # Store the percentage in the largest_buyer_volume field
            self.current_candle.largest_buyer_volume = largest_buyer_pct

    def _get_missing_periods(self, last_time: datetime, current_time: datetime) -> List[datetime]:
        """Get list of missing period timestamps between last and current"""
        missing = []
        next_time = last_time + timedelta(seconds=self.timeframe_seconds)
        
        while next_time < current_time:
            missing.append(next_time)
            next_time += timedelta(seconds=self.timeframe_seconds)
            
        return missing

class MarketAggregator:
    def __init__(self, timeframe_seconds: int, cleanup_minutes: int = 30):
        self.timeframe_seconds = timeframe_seconds  
        self.tokens: Dict[str, TokenAggregator] = {}
        self.cleanup_minutes = cleanup_minutes
        self.current_time: Optional[datetime] = None
        # Add trade tracking
        self.trade_stats = {
            'total_trades_received': 0,
            'trades_processed': 0,
            'trades_skipped': 0,
            'trades_by_token': defaultdict(int)
        }
        
    # def cleanup_inactive_tokens(self, current_time: datetime):
    #     """Remove tokens that haven't been updated recently"""
    #     inactive_tokens = [
    #         mint for mint, token in self.tokens.items()
    #         if (current_time - token.last_processed_candle_time).total_seconds() / 60 > self.cleanup_minutes
    #     ]
        
    #     for mint in inactive_tokens:
    #         del self.tokens[mint]
            
    def process_trade(self, trade: TradeEvent) -> Tuple[TokenAggregator, bool]:
        self.current_time = trade.timestamp
        self.trade_stats['total_trades_received'] += 1
        
        # Periodically cleanup inactive tokens
        # self.cleanup_inactive_tokens(self.current_time)
        
        # Get or create token aggregator
        if trade.mint not in self.tokens:
            self.tokens[trade.mint] = TokenAggregator(
                mint=trade.mint,
                timeframe_seconds=self.timeframe_seconds
            )
        
        token = self.tokens[trade.mint]
        token.last_processed_timestamp = self.current_time
        
        # Track before processing
        if trade.signature in token.processed_txs:
            self.trade_stats['trades_skipped'] += 1
            new_candle = False
        else:
            self.trade_stats['trades_processed'] += 1
            self.trade_stats['trades_by_token'][trade.mint] += 1
            new_candle = token.process_trade(trade)
        
        return token, new_candle

    def get_processing_stats(self) -> dict:
        return {
            'total_received': self.trade_stats['total_trades_received'],
            'processed': self.trade_stats['trades_processed'],
            'skipped': self.trade_stats['trades_skipped'],
            'tokens': dict(self.trade_stats['trades_by_token']),
            'active_tokens': len(self.tokens)
        }