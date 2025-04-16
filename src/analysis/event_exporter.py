from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import List
from core.events import SignalEvent, EntryEvent, ExitEvent

class EventExporter:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def export_run_data(self, run_id: str, test_name: str):
        """Export all data for a specific run"""
        db = DatabaseService(run_id)
        
        # Get signals
        signals_df = db.get_run_signals(run_id)
        if not signals_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            signals_df.to_csv(self.results_dir / f"signals_{test_name}_{timestamp}.csv", index=False)
        
        # Get positions/trades
        trades_df = db.get_run_trades(run_id)
        if not trades_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trades_df.to_csv(self.results_dir / f"trades_{test_name}_{timestamp}.csv", index=False)
        
    def export_signals(self, signals: List[SignalEvent], test_name: str):
        """Export detailed signal data"""
        signals_data = [
            {
                "timestamp": s.timestamp,
                "mint": s.mint,
                "price": float(s.price),
                "volume": float(s.volume),
                "trigger_move": float(s.trigger_move),
                "confirmation_move": float(s.confirmation_move),
                "signal_type": s.signal_type,
            }
            for s in signals
        ]
        
        df = pd.DataFrame(signals_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(self.results_dir / f"signals/signals_{test_name}_{timestamp}.csv", index=False)
        
    def export_trades(self, entries: List[EntryEvent], exits: List[ExitEvent], test_name: str):
        """Export detailed trade data with entry and exit pairs"""
        trades_data = []
        
        for entry in entries:
            # Find corresponding exit if it exists
            exit_event = next(
                (e for e in exits if e.entry_reference == entry),
                None
            )
            
            trade_data = {
                "entry_time": entry.timestamp,
                "mint": entry.mint,
                "entry_price": float(entry.price),
                "position_size": float(entry.size),
                "entry_reason": entry.reason,
                "entry_volume": float(entry.signal_reference.volume),
                "entry_trigger_move": float(entry.signal_reference.trigger_move),
                "entry_confirmation_move": float(entry.signal_reference.confirmation_move),
                "exit_time": exit_event.timestamp if exit_event else None,
                "exit_price": float(exit_event.price) if exit_event else None,
                "exit_reason": exit_event.reason if exit_event else None,
                "pnl": float(exit_event.pnl) if exit_event else None,
                "hold_time_minutes": exit_event.hold_time_minutes if exit_event else None
            }
            trades_data.append(trade_data)
            
        df = pd.DataFrame(trades_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(self.results_dir / f"trades/trades_{test_name}_{timestamp}.csv", index=False) 