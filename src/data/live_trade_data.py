import asyncio
import pandas as pd
from datetime import datetime, timezone
from data.pump_data_feed import PumpDataFeed, TradeEvent

class PumpDataCollector:
    def __init__(self, output_path: str = "collected_trades_8_hours.csv"):
        self.trades = []
        self.output_path = output_path
        self.feed = PumpDataFeed()
        
    async def trade_callback(self, trade: TradeEvent):
        """Callback to receive trades from feed"""
        self.trades.append(trade)  # Store the entire TradeEvent object
        
    async def collect(self, duration_minutes: int):
        """Collect trades for specified duration"""
        print(f"Starting trade collection for {duration_minutes} minutes...")
        
        # Add callback to feed
        self.feed.add_callback(self.trade_callback)
        
        # Start feed
        feed_task = asyncio.create_task(self.feed.start())
        
        # Wait for specified duration
        await asyncio.sleep(duration_minutes * 60)
        
        # Stop feed
        await self.feed.stop()
        
        # Save collected trades
        if self.trades:
            df = pd.DataFrame(self.trades)
            df.to_csv(self.output_path, index=False)
            print(f"Saved {len(self.trades)} trades to {self.output_path}")
        else:
            print("No trades collected")

# Usage example
async def main():
    print("Starting data collection...")
    collector = PumpDataCollector()
    await collector.collect(duration_minutes=480)
    print("Data collection completed.")

if __name__ == "__main__":
    asyncio.run(main()) 