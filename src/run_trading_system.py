import asyncio
import signal
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from dataclasses import dataclass
from core.copy_trading_system import TradingSystem
from utils.config import RiskParameters
import pandas as pd
from pathlib import Path
import statistics 
import logging
from utils.logger import TradingLogger
import sys
import csv
from risk.position_tracker import PositionTracker

@dataclass
class Config:
    name: str
    initial_capital: float
    strategy_params: Dict
    risk_params: Dict
    timeframe_seconds: int

class GracefulExit(SystemExit):
    pass

def load_tracked_wallets(file_path: str) -> List[str]:
    """Load tracked wallet addresses from CSV file"""
    wallets = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                address = row.get('wallet_address', '').lower()
                if address:
                    wallets.append(address)
        print(f"Loaded {len(wallets)} tracked wallets")
        return wallets
    except Exception as e:
        print(f"Error loading tracked wallets: {str(e)}")
        return []

class InitTradingSystem:
    def __init__(self, logger: TradingLogger = None):
        self.trading_bot = None
        self.logger = logger
        self._shutdown_requested = False
        self._shutdown_event = asyncio.Event()
        
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}. Starting graceful shutdown...")
        self._shutdown_requested = True
        # Set the event in the event loop
        loop = asyncio.get_running_loop()
        loop.create_task(self._trigger_shutdown())

    async def _trigger_shutdown(self):
        """Helper method to trigger shutdown"""
        self.logger.info("Graceful shutdown initiated")
        await self.shutdown()
        
    async def run_trading_system(self, config: Config) -> None:
        """Run trading system with graceful shutdown"""
        try:
            tracked_wallets = load_tracked_wallets('data/tracked_wallets.csv')
            
            # Set up strategy parameters with tracked wallets
            strategy_params = {
                'tracked_wallets': tracked_wallets,
                'min_trade_amount': config.strategy_params.get('min_trade_amount', 0.1),
                'max_time_in_trade': config.strategy_params.get('max_time_in_trade', 30),
                'take_profit_pct': config.strategy_params.get('take_profit_pct', 0.15),
            }
            
            risk_params = {
                'max_position_size': config.risk_params['max_position_size'],
                'stop_loss_pct': config.risk_params['stop_loss_pct'],
                'max_positions': config.risk_params['max_positions'],
                'max_hold_time_minutes': config.risk_params['max_hold_time_minutes'],
                'max_daily_loss_pct': config.risk_params['max_daily_loss_pct']
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            position_tracker = PositionTracker(csv_path=f"results/trades/live_{config.name}_{timestamp}.csv")  
            
            self.trading_bot = TradingSystem(
                initial_capital=config.initial_capital,
                dry_run=False,  # Set to False for live trading
                backtest_mode=False,
                backtest_data_path="backtest_data/2025_03_27.csv",
                strategy_params=strategy_params,
                risk_params=risk_params,
                run_name=config.name,
                logger=self.logger,
                position_tracker=position_tracker
            )

            await self.trading_bot.start()
            self.logger.info("Trading system started successfully")
            
            # Keep the system running until shutdown signal
            while self.trading_bot.is_running and not self._shutdown_requested:
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    self.logger.info("Received cancellation request")
                    self._shutdown_requested = True
                    break
            
            if self._shutdown_requested:
                self.logger.info("Shutdown requested, initiating shutdown sequence")
                await self.shutdown()
                
        except Exception as e:
            self.logger.error(f"Error in trading system: {e}")
            await self.shutdown()
            raise
        finally:
            if self.trading_bot and self.trading_bot.is_running:
                self.logger.info("Final cleanup in finally block")
                await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown the trading system"""
        if self.trading_bot and self.trading_bot.is_running:
            self.logger.info("Shutting down trading system...")
            try:
                # Set a timeout for the shutdown process
                shutdown_timeout = 3  # 5 minutes timeout
                try:
                    await asyncio.wait_for(self.trading_bot.stop(), timeout=shutdown_timeout)
                    self.logger.info("Trading system stopped successfully")
                except asyncio.TimeoutError:
                    self.logger.error(f"Shutdown timed out after {shutdown_timeout} seconds")
                    self.trading_bot.is_running = False
                    
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
                self.trading_bot.is_running = False
            finally:
                self.trading_bot.is_running = False
                sys.exit(0)  # Force exit after cleanup

async def main():
    # Configure logging
    logger = TradingLogger("trading_system", console_output=False)

    baseline_risk_params = {
        'max_position_size': 0.02,
        'max_positions': 5,
        'max_hold_time_minutes': 5,
        'max_daily_loss_pct': 0.5,
        'stop_loss_pct': 0.02
    }

    # Copy trading strategy parameters
    baseline_strategy_params = {
        'min_trade_amount': 0.1,
        'max_time_in_trade': 30,
        'take_profit_pct': 0.07,
    }
    
    config = Config(
        name=f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        initial_capital=.7015,
        strategy_params=baseline_strategy_params,
        risk_params=baseline_risk_params,
        timeframe_seconds=60
    )
    
    init_system = InitTradingSystem(logger)
    
    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, init_system.handle_shutdown)
    
    try:
        logger.info("Starting trading system...")
        await init_system.run_trading_system(config)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 