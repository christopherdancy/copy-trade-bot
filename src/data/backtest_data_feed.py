from datetime import datetime
import pandas as pd
from decimal import Decimal
from data.pump_data_feed import TradeEvent
import asyncio
from typing import Callable, List

class BacktestDataFeed:
    def __init__(self, csv_path: str):
        self.trades_df = pd.read_csv(csv_path)
        self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
        self.callbacks: List[Callable] = []
        self.is_running = False
        
    async def start(self):
        self.is_running = True
        
        for _, row in self.trades_df.iterrows():
            if not self.is_running:
                break
                
            trade = TradeEvent(
                mint=row['mint'],
                sol_amount=Decimal(str(row['sol_amount'])),
                token_amount=Decimal(str(row['token_amount'])),
                is_buy=row['is_buy'],
                user=row['user'],
                timestamp=row['timestamp'],
                virtual_sol_reserves=Decimal(str(row['virtual_sol_reserves'])),
                virtual_token_reserves=Decimal(str(row['virtual_token_reserves'])),
                real_sol_reserves=Decimal(str(row['real_sol_reserves'])),
                real_token_reserves=Decimal(str(row['real_token_reserves'])),
                signature=row['signature']
            )
            
            # Process callbacks
            for callback in self.callbacks:
                await callback(trade)  # Changed back to await since we're using async callbacks
            
            await asyncio.sleep(0.000001)
            
    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)
            
    async def stop(self):
        self.is_running = False 