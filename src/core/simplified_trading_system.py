from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List
from data.pump_data_feed import PumpDataFeed
from core.aggregator import MarketAggregator
from strategies.coordinated_move_strategy import CoordinatedMoveStrategy, Signal
# from strategies.momentum_strategy import PureMomentumStrategy, Signal
from risk.risk_manager import RiskManager
from utils.logger import TradingLogger
import pandas as pd
import asyncio
from dataclasses import dataclass
from utils.config import RiskParameters
from execution.live_run_executor import LiveRunExecutor
from solders.keypair import Keypair
import os
from pathlib import Path
from execution.dry_run_executor import DryRunExecutor
from solders.pubkey import Pubkey
from data.backtest_data_feed import BacktestDataFeed
from core.events import SignalEvent, EntryEvent, ExitEvent
from risk.position import Position
from asyncio import Lock
from risk.monitoring import HeartbeatMonitor
from risk.position_tracker import PositionTracker
from db.service import DatabaseService
from dotenv import load_dotenv
import json
import base58

# Load environment variables
load_dotenv()

# TODO: Deal with system running issues
# TODO: log system progress for most important metrics

class TradingSystem:
    def __init__(self, 
                 initial_capital: float = 10.0, 
                 dry_run: bool = True,
                 backtest_mode: bool = True,
                 backtest_data_path: str = None,
                 strategy_params: Dict = None,
                 risk_params: Dict = None,
                 timeframe_seconds: int = 60,
                 run_name: str = None,
                 logger: TradingLogger = None,
                 position_tracker: PositionTracker = None):

        # Initalize Trading System
        self.logger = logger or TradingLogger("trading_system", console_output=False)
        self.is_running = False
        self.dry_run = dry_run
        self.backtest_mode = backtest_mode
        self.timeframe_seconds = timeframe_seconds
        self.is_accepting_new_trades = True
        
        # Initialize data feed based on mode
        if self.backtest_mode:
            if not backtest_data_path:
                raise ValueError("backtest_data_path must be provided in backtest mode")
            self.data_feed = BacktestDataFeed(backtest_data_path)
        else:
            self.data_feed = PumpDataFeed(debug=True, logger=self.logger)

        # Initialize event lists
        self.position_locks = {} 

        # Load wallet for transactions
        if self.dry_run:
            self.executor = DryRunExecutor(initial_capital)
        else:
            secret_key = bytes(base58.b58decode(os.getenv('PRIVATE_KEY')))
            wallet = Keypair.from_bytes(secret_key)
            connection_string = os.getenv('RPC_URL')
            if not connection_string:
                raise ValueError("RPC_URL not found in environment variables")
            self.executor = LiveRunExecutor(wallet, connection_string, self.logger)
        
        # Initialize strategy and risk management with provided or default parameters
        self.market_aggregator = MarketAggregator(timeframe_seconds=self.timeframe_seconds)
        self.setup_strategy(strategy_params)
        self.setup_risk_management(initial_capital, risk_params)
        self.position_tracker = position_tracker or PositionTracker()

        # Log system initialization
        self.logger.info(f"Trading System Initializing: Mode={'Backtest' if backtest_mode else 'Live'}, "
                        f"Dry Run={dry_run}, Initial Capital={initial_capital}")
        self.logger.info(f"Strategy Parameters: {strategy_params}")
        self.logger.info(f"Risk Parameters: {risk_params}")
        self.logger.info(f"Backtest Mode: {self.backtest_mode} Dry Run: {self.dry_run}")

    def setup_strategy(self, params: Dict = None):
        """Initialize strategy with provided or default parameters"""
        default_params = {
            'min_price': 1.25e-7,
            'max_price': 2.5e-7,
            'min_price_move': 0.20,
            'max_price_move': 0.40,
            'max_confirmation_move': 0.19,
            'max_volume_threshold': 40,
            'max_time_in_trade': 5,
            'take_profit_pct': 0.20,
            'lookback': 3
        }
        
        # momentum strategy
        # default_params = {
        #     'min_price': 1.25e-7,
        #     'max_price': 2.5e-7,
        #     'min_volume_threshold': 0.001,
        #     'lookback': 3
        # }
        # Use provided params or defaults
        strategy_params = {**default_params, **(params or {})}
        
        self.strategy = CoordinatedMoveStrategy(
            min_price=strategy_params['min_price'],
            max_price=strategy_params['max_price'],
            min_price_move=strategy_params['min_price_move'],
            max_price_move=strategy_params['max_price_move'],
            max_confirmation_move=strategy_params['max_confirmation_move'],
            max_volume_threshold=strategy_params['max_volume_threshold'],
            max_time_in_trade=strategy_params['max_time_in_trade'],
            take_profit_pct=strategy_params['take_profit_pct'],
            lookback=strategy_params['lookback'],
            logger=self.logger
        )

    def setup_risk_management(self, initial_capital: float, params: Dict = None):
        """Initialize risk management with provided or default parameters"""
        default_params = {
            'max_position_size': 0.1,
            'stop_loss_pct': 1,
            'max_positions': 10000,
            'max_hold_time_minutes': 10000,
            'max_daily_loss_pct': 1
        }
        
        # Use provided params or defaults
        risk_params = {**default_params, **(params or {})}
        
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            risk_params=RiskParameters(**risk_params),
            logger=self.logger
        )

    async def start(self):
        """Start the trading system"""
        try:
            self.is_running = True
            
            # Log system start
            self.logger.critical("Trading System Starting")

            async def on_trade(trade):
                if self.is_running:
                    await self.process_trade(trade)

            self.data_feed.add_callback(on_trade)
            await self.data_feed.start()

        except Exception as e:
            self.logger.critical(f"Failed to start trading system: {str(e)}")
            self.is_running = False
            raise

    async def stop(self):
        """Stop the trading system"""
        self.logger.critical("Initiating trading system shutdown")

        # Stop accepting new trades
        if not self.dry_run:
            self.is_accepting_new_trades = False
            open_positions = self.risk_manager.get_current_positions()
            if open_positions:
                self.logger.warning(f"Waiting for {len(open_positions)} positions to close naturally")
                # Keep checking positions until they're all closed
                while open_positions and self.is_running:
                    self.logger.info(f"Still waiting on {len(open_positions)} positions to close...")
                    
                    # List current positions
                    for mint, position in open_positions.items():
                        self.logger.info(f"Open position: {mint}, Entry time: {position.entry_time}")
                    
                    # Wait before checking again
                    await asyncio.sleep(60)  # Check every minute
                    open_positions = self.risk_manager.get_current_positions()

            self.logger.info("No open positions remaining - proceeding with full shutdown")

        # Shutdown Services
        self.logger.info("Shutting down services...")
        self.is_running = False
        await self.data_feed.stop()
        
        self.position_tracker.analyze_performance()
        
        # Log Shutdown
        self.logger.info("Trading system stopped successfully")

    # Each part of the system should be logged and tracked seperatly
    async def process_trade(self, trade):
        """Process a single trade"""
        if not self.is_running:
            return

        try:
            if trade.mint not in self.position_locks:
                self.position_locks[trade.mint] = Lock()

            # Process trade and get token aggregator
            token, new_candle = self.market_aggregator.process_trade(trade)  
                
            async with self.position_locks[trade.mint]:
                # Exit Logic
                if self.risk_manager.has_position(trade.mint):
                    self.logger.debug(f"Checking exit conditions for {trade.mint}")
                    await self._handle_exit_checks(token, trade)
                    return      
                
                else:
                    if not new_candle or len(token.candles) < self.strategy.lookback:
                        return

                # Entry logic
                await self._handle_entry_checks(token, trade)

        except Exception as e:
            self.logger.error(f"Error processing trade in trading system: {str(e)}")

    async def _handle_exit_checks(self, token, trade):
        """Handle exit checks for existing position"""
        try:
            current_price = float(str(trade.price))
            position = self.risk_manager.get_position_info(trade.mint)
            if not position or not position.is_active:
                return
        
            # Check exit conditions 
            exit_signal = self.strategy.check_exit(
                token.mint,
                current_price,
                position,
                trade.timestamp
            )
            stop_loss_hit = self.risk_manager.check_stop_loss(
                token.mint,
                current_price
            )

            # Execute exit if conditions met
            if (exit_signal and exit_signal.is_valid) or stop_loss_hit:
                exit_reason = "Strategy Exit" if exit_signal.is_valid else "Stop Loss"
                self.logger.info(f"Exit Triggered: Mint={trade.mint}, "
                               f"Reason={exit_reason}, Price={current_price}")

                success, tx_details = await self.execute_trade(
                    mint=trade.mint,
                    price=current_price,
                    is_buy=False,
                    amount=position.token_amount,
                    timestamp=trade.timestamp
                )
                
                if success:
                    # Update Position Data
                    pnl = await self.risk_manager.exit_position(
                        trade.mint,
                        tx_details['execution_price'],
                        tx_details['total_sol_change']
                    )
                    exit_type = "Strategy" if (exit_signal and exit_signal.is_valid) else "Stop Loss"
                    self.logger.info(f"Exit Successful: Mint={trade.mint}, "
                                   f"PnL={pnl}, exit reason={exit_reason} tx_details={tx_details}")

        except Exception as e:
            self.logger.error(f"Exit check error: Mint={trade.mint}, Error={str(e)}")

    async def _handle_entry_checks(self, token, trade):
        """Handle entry checks for new position"""
        if not self.is_accepting_new_trades:
            self.logger.info("System is in shutdown mode - no new positions allowed")
            return
        
        try:
            # Generate entry signal
            entry_signal = self.strategy.generate_signal(token.candles)
            if entry_signal and entry_signal.is_valid:
                self.logger.info(f"Entry Signal Generated: Mint={trade.mint}, "
                            f"Price={entry_signal.price}, Volume={entry_signal.volume}")

                # Risk Manager Entry Check
                can_enter, position_size = await self.risk_manager.can_enter_position(trade.mint)
                if can_enter and position_size > 0:
                    self.logger.info(f"Attempting Entry: Mint={trade.mint}, "
                                f"Size={position_size}, Price={trade.price}")
                    
                    # Execute Entry
                    success, tx_details = await self.execute_trade(
                        mint=trade.mint,
                        price=float(trade.price),
                        is_buy=True,
                        amount=position_size,
                        timestamp=trade.timestamp,
                        signal=entry_signal
                    )

                    if success:
                        self.logger.info(f"Entry Successful: Mint={trade.mint}, "
                                    f"Details={tx_details}")
                        await self.risk_manager.add_position(
                            trade.mint,
                            trade.timestamp,
                            tx_details['sol_trade_amount'],
                            tx_details['token_trade_amount'],
                            tx_details['execution_price'],
                            tx_details['total_fees'],
                            tx_details['tx_sig'],
                            tx_details['total_sol_change']
                        )

                    else:
                        await self.risk_manager.remove_reserved_position(trade.mint)
        except Exception as e:
            self.logger.error(f"Entry check error: Mint={trade.mint}, Error={str(e)}")
            await self.risk_manager.remove_reserved_position(token.mint)
            raise

    async def execute_trade(self, mint: str, price: float, is_buy: bool, amount: float, timestamp: datetime, signal: Signal = None) -> tuple[bool, Dict]:
        """
        Execute a trade (entry or exit)
        Args:
            mint: Token mint address
            price: Current price
            is_buy: True for entry/buy, False for exit/sell
            amount: Amount to trade (in SOL for buy, in tokens for sell)
        Returns: 
            tuple[bool, Dict]: (success, execution_details)
        """
        trade_type = "buy" if is_buy else "sell"
        try:
            tx_details = None
            if self.dry_run:
                # Simulate trade execution in dry run mode
                self.logger.info(f"Executing DRY RUN {trade_type} transaction: {mint}, {amount} {'SOL' if is_buy else 'tokens'}")
                success, tx_details = (
                    self.executor.buy_token(
                        mint=Pubkey.from_string(mint),
                        amount_sol=amount,
                        price=price
                    ) if is_buy else
                    self.executor.sell_token(
                        mint=mint,
                        token_amount=amount,
                        price=price
                    )
                )
            else:
                self.logger.info(f"Executing {trade_type} transaction: {mint}, {amount} {'SOL' if is_buy else 'tokens'}")
                tx_details = await (
                    self.executor.buy_token(
                        mint=Pubkey.from_string(mint),
                        amount_sol=amount
                    ) if is_buy else
                    self.executor.sell_token(
                        mint=Pubkey.from_string(mint),
                        token_amount=amount
                    )
                )
                
            if tx_details:
                if not is_buy:
                    self.logger.info(f"Exit transaction successful: {tx_details}")
                    self.position_tracker.record_exit(mint, tx_details, timestamp)
                else:
                    self.logger.info(f"Entry transaction successful: {tx_details}")
                    self.position_tracker.record_entry(mint, tx_details, timestamp, signal)
                return True, tx_details
            else:
                self.logger.error("Transaction failed")
                return False, {}
                    
        except Exception as e:
            self.logger.error(f"Error executing {trade_type} transaction for {mint}: {str(e)}")
            return False, {}
                