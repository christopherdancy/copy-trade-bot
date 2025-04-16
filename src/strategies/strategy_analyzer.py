from typing import List, Dict, Optional, Union
from decimal import Decimal
from core.aggregator import Candle
from dataclasses import dataclass, field
from data.pump_data_feed import TradeEvent
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from strategies.coordinated_move_strategy import Signal

@dataclass
class CandleEvent:
    time: datetime
    price: Decimal
    volume: Decimal
    volume_growth: Optional[Decimal]
    price_growth: Optional[Decimal]
    pnl: Decimal  # From entry

def analyze_candle_sequence(candle_events: List[dict], entry_price: Decimal) -> List[CandleEvent]:
    """Analyze full sequence of candles with growth rates"""
    sequence = []
    
    for i, event in enumerate(candle_events):
        # Calculate growth rates from previous candle
        volume_growth = None
        price_growth = None
        
        if i > 0:
            prev_event = candle_events[i-1]
            
            # Safe volume growth calculation
            if prev_event['volume'] > 0:  # Prevent division by zero
                volume_growth = (event['volume'] - prev_event['volume']) / prev_event['volume']
            else:
                volume_growth = Decimal('0')  # or None if you prefer
                
            # Safe price growth calculation
            if prev_event['price'] > 0:  # Prevent division by zero
                price_growth = (event['price'] - prev_event['price']) / prev_event['price']
            else:
                price_growth = Decimal('0')  # or None if you prefer
        
        # Safe PnL calculation
        if entry_price > 0:
            pnl = (event['price'] - entry_price) / entry_price
        else:
            pnl = Decimal('0')
        
        sequence.append(CandleEvent(
            time=event['time'],
            price=event['price'],
            volume=event['volume'],
            volume_growth=volume_growth,
            price_growth=price_growth,
            pnl=pnl
        ))
    
    return sequence

@dataclass
class TradeEventAnalysis:
    time: datetime
    price: Decimal
    pnl: Decimal  # From entry

def analyze_trade_sequence(trade_events: List[dict], entry_price: Decimal) -> List[TradeEventAnalysis]:
    """Analyze sequence of individual trades with PnL tracking"""
    sequence = []
    
    for trade in trade_events:
        # Safe PnL calculation
        if entry_price > 0:
            pnl = (trade['price'] - entry_price) / entry_price
        else:
            pnl = Decimal('0')
        
        sequence.append(TradeEventAnalysis(
            time=trade['time'],
            price=trade['price'],
            pnl=pnl
        ))
    
    return sequence

@dataclass
class Position:
    token: str
    entry_price: Decimal
    entry_time: datetime
    
    # Raw data for analysis
    candles: List[Candle] = field(default_factory=list)
    trades: List[TradeEvent] = field(default_factory=list)

    # Exit information
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None

    def set_exit(self, price: Decimal, time: datetime) -> None:
        """Record exit details but continue collecting data"""
        self.exit_price = price
        
        # Ensure naive timestamps
        if time.tzinfo is not None:
            time = time.replace(tzinfo=None)
        
        self.exit_time = time

    def add_candle(self, candle: Candle) -> None:
        """Add a new candle to position history if it's after entry time"""
        # Convert both times to UTC
        candle_time = pd.Timestamp(candle.timestamp).tz_localize('UTC')
        entry_time = pd.Timestamp(self.entry_time).tz_localize('UTC')
        
        # Adjust candle time by +5 hours
        candle_time = candle_time + pd.Timedelta(hours=5)
        
        if candle_time >= entry_time:
            self.candles.append(candle)

    def add_trade(self, trade: TradeEvent) -> None:
        """Add a new trade to position history if it's after entry time"""
        trade_time = trade.timestamp
        if trade_time.tzinfo is not None:
            trade_time = trade_time.replace(tzinfo=None)
        # print(f"Trade object: {trade}")
        # print(f"Trade attributes: {dir(trade)}")
        if trade.mint == self.token and trade_time >= self.entry_time:
            self.trades.append(trade)

@dataclass
class TradeAnalysis:
    max_profit: Decimal
    max_profit_time: datetime
    max_drawdown: Decimal
    max_drawdown_time: datetime
    optimal_exit_time: datetime
    profit_loss: Decimal

@dataclass
class CandleAnalysis:
    max_profit: Decimal
    max_profit_time: datetime
    max_drawdown: Decimal
    max_drawdown_time: datetime
    optimal_exit_time: datetime
    profit_loss: Decimal
    max_volume: Decimal
    max_volume_time: datetime

@dataclass
class PositionAnalysis:
    token: str
    entry_time: datetime
    entry_price: Decimal
    exit_time: Optional[datetime]
    exit_price: Optional[Decimal]

    trade_analysis: TradeAnalysis
    candle_analysis: CandleAnalysis

    @staticmethod
    def analyze_price_series(events: List[dict], entry_time: datetime, entry_price: Decimal, exit_price: Optional[Decimal], is_candle: bool = False) -> Union[TradeAnalysis, CandleAnalysis]:
        max_profit = Decimal('0')
        max_profit_time = entry_time
        max_drawdown = Decimal('0')
        max_drawdown_time = entry_time
        running_high = entry_price
        optimal_exit_time = entry_time

        max_volume = Decimal('0')
        max_volume_time = entry_time
        volume_profile = []
        
        for event in events:
            event_time = event['time']
            
            if event_time < entry_time:
                continue
                
            current_price = event['price']
            pnl = (current_price - entry_price) / entry_price

            # Track volume patterns
            if is_candle:
                current_volume = event['volume']
                volume_profile.append((event_time, current_volume))
            
                if current_volume > max_volume:
                    max_volume = current_volume
                    max_volume_time = event_time
                
            if pnl > max_profit:
                max_profit = pnl
                max_profit_time = event_time
                running_high = current_price
            
            drawdown = (current_price - running_high) / running_high
            if drawdown < max_drawdown:
                max_drawdown = drawdown
                max_drawdown_time = event_time
            
            if pnl > Decimal('0.1') or drawdown < Decimal('-0.05'):
                optimal_exit_time = event_time
        
        profit_loss = (exit_price - entry_price) / entry_price if exit_price else Decimal('0')
        
        if is_candle:
            return CandleAnalysis(
                max_profit=max_profit,
                max_profit_time=max_profit_time,
                max_drawdown=max_drawdown,
                max_drawdown_time=max_drawdown_time,
                optimal_exit_time=optimal_exit_time,
                profit_loss=profit_loss,
                max_volume=max_volume,
                max_volume_time=max_volume_time,
            )
        else:
            return TradeAnalysis(
                max_profit=max_profit,
                max_profit_time=max_profit_time,
                max_drawdown=max_drawdown,
                max_drawdown_time=max_drawdown_time,
                optimal_exit_time=optimal_exit_time,
                profit_loss=profit_loss
            )
    
    @staticmethod
    def from_position(position: Position) -> 'PositionAnalysis':
        entry_price = Decimal(str(position.entry_price))
        exit_price = Decimal(str(position.exit_price)) if position.exit_price else None

        # Create trade events with consistent timezone
        trade_events = [
            {
                'time':  trade.timestamp.replace(tzinfo=None) if trade.timestamp.tzinfo else trade.timestamp,
                'price': Decimal(str(trade.price)),
            }
            for trade in sorted(position.trades, key=lambda x: x.timestamp)
            if trade.timestamp >= position.entry_time
        ]

        candle_events = [
            {
                'time': candle.timestamp.replace(tzinfo=None) if candle.timestamp.tzinfo else candle.timestamp,
                'price': Decimal(str(candle.close)),
                'volume': Decimal(str(candle.volume)),
            }
            for candle in position.candles
        ]
        
        trade_analysis = PositionAnalysis.analyze_price_series(
            trade_events, 
            position.entry_time,
            entry_price,
            exit_price,
            is_candle=False
        )

        candle_analysis = PositionAnalysis.analyze_price_series(
            candle_events,
            position.entry_time,
            entry_price,
            exit_price,
            is_candle=True
        )

        return PositionAnalysis(
            token=position.token,
            entry_time=position.entry_time,
            entry_price=entry_price,
            exit_time=position.exit_time,
            exit_price=exit_price,
            trade_analysis=trade_analysis,
            candle_analysis=candle_analysis
        )

@dataclass
class TradeLifecycle:
    """Represents a minute-by-minute snapshot of a trade"""
    token: str
    trade_number: int
    minute: int  # Minutes from entry
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    pnl_from_entry: Decimal
    drawdown_from_high: Decimal
    new_high: bool
    final_pnl: Decimal
    max_pnl: Decimal  # Track max PnL reached

class StrategyAnalyzer():
    def __init__(self):
        self.positions: Dict[str, Position] = {} 

    def record_position_entry(
        self,
        token: str,
        signal: Signal,
        timestamp: datetime
    ) -> Optional[Position]:
        """
        Create position with naive timestamp
        """
        # Ensure naive timestamp
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)
        
        position = Position(
            token=token,
            entry_price=Decimal(str(signal.price)),
            entry_time=timestamp,
        )
        self.positions[token] = position
        return position
    
    def record_position_exit(
        self,
        token: str,
        price: float,
        timestamp: datetime
    ) -> Optional[Position]:
        """
        Exit position using naive timestamp
        """
        # Ensure naive timestamp
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)
        
        position = self.positions[token]
        position.set_exit(Decimal(str(price)), timestamp)

    def update_position_data(self, token: str, candle: Optional[Candle] = None, 
                           trade: Optional[TradeEvent] = None) -> None:
        """Update all active positions with new data"""
        if token not in self.positions:
            return
        
        position = self.positions.get(token)

        if candle:
            position.add_candle(candle)
                
        if trade:
            position.add_trade(trade)

    def format_analysis_for_csv(self, summary_stats: dict) -> dict:
        """Format the nested summary stats into a flat CSV structure"""
        flat_data = {}
        
        # Flatten the summary statistics
        for category, stats in summary_stats.items():
            for metric, value in stats.items():
                flat_data[f'{category.lower().replace(" ", "_")}_{metric}'] = value
        
        return flat_data
        
    def analyze_fixed_targets(self, filename: str, 
                             stop_losses=[0.01, 0.02],
                             take_profits=[0.10, 0.20]) -> None:
        """
        Analyze trades using fixed stop-loss and take-profit levels
        """
        results = []
        
        for sl in stop_losses:
            for tp in take_profits:
                trades_analysis = []
                
                for token, position in self.positions.items():
                    entry_price = position.entry_price
                    sl_price = entry_price * (Decimal('1') - Decimal(str(sl)))
                    tp_price = entry_price * (Decimal('1') + Decimal(str(tp)))
                    
                    result = {
                        'token': token,
                        'entry_time': position.entry_time,
                        'hit_tp': False,
                        'hit_sl': False,
                        'time_to_exit': None
                    }
                    
                    # Step through each trade
                    for trade in sorted(position.trades, key=lambda x: x.timestamp):
                        trade_time = trade.timestamp
                        if trade_time.tzinfo is not None:
                            trade_time = trade_time.replace(tzinfo=None)
                        
                        if trade_time < position.entry_time:
                            continue
                            
                        current_price = Decimal(str(trade.price))
                        minutes_from_entry = (trade_time - position.entry_time).total_seconds() / 60
                        
                        # Check if TP hit first
                        if current_price >= tp_price:
                            result['hit_tp'] = True
                            result['time_to_exit'] = minutes_from_entry
                            break
                        
                        # Check if SL hit first
                        if current_price <= sl_price:
                            result['hit_sl'] = True
                            result['time_to_exit'] = minutes_from_entry
                            break
                    
                    trades_analysis.append(result)
                
                # Calculate statistics
                total_trades = len(trades_analysis)
                tp_hits = sum(1 for t in trades_analysis if t['hit_tp'])
                sl_hits = sum(1 for t in trades_analysis if t['hit_sl'])
                no_hits = total_trades - tp_hits - sl_hits
                
                win_rate = tp_hits / total_trades if total_trades > 0 else 0
                avg_time_to_tp = sum(t['time_to_exit'] for t in trades_analysis if t['hit_tp']) / tp_hits if tp_hits > 0 else 0
                avg_time_to_sl = sum(t['time_to_exit'] for t in trades_analysis if t['hit_sl']) / sl_hits if sl_hits > 0 else 0
                
                # Calculate expected value per trade
                expected_value = (win_rate * tp) - ((1 - win_rate) * sl)
                
                results.append({
                    'stop_loss': sl,
                    'take_profit': tp,
                    'risk_reward': tp/sl,
                    'total_trades': total_trades,
                    'tp_hits': tp_hits,
                    'sl_hits': sl_hits,
                    'no_trigger': no_hits,
                    'win_rate': win_rate,
                    'avg_time_to_tp': avg_time_to_tp,
                    'avg_time_to_sl': avg_time_to_sl,
                    'expected_value': expected_value
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"positions/{filename}_fixed_targets_summary.csv", index=False)
        
        return results_df
        