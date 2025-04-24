from datetime import datetime
import pandas as pd
from decimal import Decimal
from data.pump_data_feed_enhanced import TradeEvent
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

            # Convert values to proper types
            try:
                sol_amount = Decimal(str(row['sol_amount']))
                token_amount = Decimal(str(row['token_amount']))
                v_sol_reserves = Decimal(str(row['virtual_sol_reserves']))
                v_token_reserves = Decimal(str(row['virtual_token_reserves']))
                r_sol_reserves = Decimal(str(row.get('real_sol_reserves', v_sol_reserves)))
                r_token_reserves = Decimal(str(row.get('real_token_reserves', v_token_reserves)))
                signature = str(row.get('signature', 'backtest_tx'))
                # Get blocktime as unix timestamp
                blocktime = int(row.get('blocktime', row['timestamp'].timestamp()))
                total_sol_change = float(sol_amount)
                
                # Calculate price directly
                price = sol_amount / token_amount if token_amount > 0 else Decimal(0)
                
                # Create TradeEvent with all required fields
                trade = TradeEvent(
                    mint=str(row['mint']),
                    sol_amount=sol_amount,
                    token_amount=token_amount,
                    is_buy=bool(row['is_buy']),
                    user=str(row['user']),
                    virtual_sol_reserves=v_sol_reserves,
                    virtual_token_reserves=v_token_reserves,
                    real_sol_reserves=r_sol_reserves,
                    real_token_reserves=r_token_reserves,
                    signature=signature,
                    blocktime=blocktime,
                    total_sol_change=total_sol_change
                )
                
                # Explicitly set price (even though TradeEvent calculates it internally)
                trade.price = price
                
                # Process callbacks concurrently
                tasks = []
                for callback in self.callbacks:
                    tasks.append(asyncio.create_task(callback(trade)))
                
                # Wait for a tiny amount of time to allow tasks to be scheduled
                await asyncio.sleep(0.00001)
                
            except Exception as e:
                print(f"Error processing backtest trade row: {e}")
            
    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)
            
    async def stop(self):
        self.is_running = False 