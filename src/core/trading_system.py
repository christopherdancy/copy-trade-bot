from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List
from data.pump_data_feed_enhanced import PumpDataFeed, TradeEvent
from strategies.copy_trading_strategy import CopyTradingStrategy
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
from asyncio import Lock
from risk.monitoring import HeartbeatMonitor
from core.position_tracker import PositionTracker, Position
from db.service import DatabaseService
from dotenv import load_dotenv
import json
import base58

# Load environment variables
load_dotenv()

# TODO: Deal with system running issues
# TODO: log system progress for most important metrics
# TODO: Test actual live and determine if we can batch by block, then we to track latency metrics

class TradingSystem:
    def __init__(self, 
                 initial_capital: float = 10.0, 
                 dry_run: bool = True,
                 backtest_mode: bool = True,
                 backtest_data_path: str = None,
                 strategy_params: Dict = None,
                 risk_params: Dict = None,
                 logger: TradingLogger = None):

        # Initialize Trading System
        self.logger = logger or TradingLogger("trading_system", console_output=False)
        self.is_running = False
        self.dry_run = dry_run
        self.backtest_mode = backtest_mode
        self.is_accepting_new_trades = True
        
        # Initialize data feed based on mode
        if self.backtest_mode:
            if not backtest_data_path:
                raise ValueError("backtest_data_path must be provided in backtest mode")
            self.data_feed = BacktestDataFeed(backtest_data_path)
        else:
            self.data_feed = PumpDataFeed(debug=True, logger=self.logger)

        # Initialize position locks
        self.position_locks = {} 

        # Load wallet for transactions
        if self.dry_run:
            self.executor = DryRunExecutor(initial_capital)
            self.wallet = "7K3yK4K8cKGNZddgMMsMQuGtSt9Q3S4VTQimW3Rkf8QB"
        else:
            secret_key = bytes(base58.b58decode(os.getenv('PRIVATE_KEY')))
            self.wallet = Keypair.from_bytes(secret_key)
            connection_string = os.getenv('RPC_URL')
            if not connection_string:
                raise ValueError("RPC_URL not found in environment variables")
            self.executor = LiveRunExecutor(self.wallet, connection_string, self.logger)
        
        # Initialize strategy and risk management with provided or default parameters
        self.setup_strategy(strategy_params)
        self.setup_risk_management(initial_capital, risk_params)
        self.position_tracker = PositionTracker(
            csv_path="data/trades/live_run.csv",
            logger=self.logger
        )

        # Log system initialization
        self.logger.info(f"Trading System Initializing: Mode={'Backtest' if backtest_mode else 'Live'}, "
                        f"Dry Run={dry_run}, Initial Capital={initial_capital}")
        self.logger.info(f"Strategy Parameters: {strategy_params}")
        self.logger.info(f"Risk Parameters: {risk_params}")
        self.logger.info(f"Backtest Mode: {self.backtest_mode} Dry Run: {self.dry_run}")

    def setup_strategy(self, params: Dict = None):
        """Initialize strategy with provided or default parameters"""
        default_params = {
            'tracked_wallets': [],  # List of wallet addresses to track
            'take_profit_pct': 0.15
        }
        
        # Use provided params or defaults
        strategy_params = {**default_params, **(params or {})}
        
        # Ensure tracked_wallets are all lowercase for case-insensitive comparison
        tracked_wallets = [w.lower() for w in strategy_params.get('tracked_wallets', [])]
        
        self.strategy = CopyTradingStrategy(
            tracked_wallets=tracked_wallets,
            take_profit_pct=strategy_params.get('take_profit_pct'),
            logger=self.logger
        )
        
        # Log tracked wallets
        if tracked_wallets:
            self.logger.info(f"Tracking wallets: {tracked_wallets}")
        else:
            self.logger.warning("No tracked wallets configured!")

    def setup_risk_management(self, initial_capital: float, params: Dict = None):
        """Initialize risk management with provided or default parameters"""
        default_params = {
            'max_position_size': 0.1,
            'stop_loss_pct': .01,
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
                    asyncio.create_task(self.process_trade(trade))

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
            open_positions = await self.position_tracker.get_all_positions()
            if open_positions:
                self.logger.warning(f"Waiting for {len(open_positions)} positions to close naturally")
                # Keep checking positions until they're all closed
                while open_positions and self.is_running:
                    self.logger.info(f"Still waiting on {len(open_positions)} positions to close...")
                    
                    # List current positions
                    for mint, position in open_positions.items():
                        self.logger.info(f"Open position: {mint}, Entry time: {position.entry_blocktime}")
                    
                    # Wait before checking again
                    await asyncio.sleep(60)  # Check every minute
                    open_positions = await self.position_tracker.get_all_positions()

            self.logger.info("No open positions remaining - proceeding with full shutdown")

        # Shutdown Services
        self.logger.info("Shutting down services...")
        self.is_running = False
        await self.data_feed.stop()
        
        # Log Shutdown
        self.logger.info("Trading system stopped successfully")

    
    async def process_trade(self, trade):
        """Process a single trade"""
        if not self.is_running:
            return

        try:
            wallet_address = trade.user.lower()
            our_wallet_address = self.wallet.lower() if self.dry_run else str(self.wallet.pubkey()).lower()
            
            # Ensure we have a lock for this token
            if trade.mint not in self.position_locks:
                self.position_locks[trade.mint] = Lock()
            
            # Use the lock for ALL processing of this token, including our own transactions
            async with self.position_locks[trade.mint]:
                # Track our own transactions directly through WebSocket feed
                if wallet_address == our_wallet_address:
                    await self._wallet_tracking(trade)
                    return

                # Exit Logic - check for exit conditions if we have a position in this token
                # This applies to both buys and sells from other wallets, as price changes can trigger stop loss/take profit
                position = await self.position_tracker.get_position(trade.mint)
                if position:
                    # Perform exit checks
                    should_exit = await self._handle_exit_checks(trade, position)

                    if should_exit:
                        await self.execute_trade(trade, position.token_amount, False)
                    
                    return
                
                # If no position and no pending entry, check if we should enter
                elif trade.is_buy:
                    # Check if this wallet is in our tracked list (case-insensitive)
                    is_tracked = wallet_address in [w.lower() for w in self.strategy.tracked_wallets]
                    
                    if is_tracked:
                        self.logger.info(f"Processing entry for tracked wallet: {trade.user}")
                        # Perform entry checks
                        should_enter, position_size = await self._handle_entry_checks(trade)
                        
                        if should_enter:
                            await self.execute_trade(trade, position_size, True)
                    
                    return

        except Exception as e:
            self.logger.error(f"Error processing trade in trading system: {str(e)}")

    async def _wallet_tracking(self, trade):
        """Track our own transactions directly through WebSocket feed"""
        try:
            self.logger.info(f"Processing tx for {trade.mint}: "
                f"price={trade.price}, "
                f"signature={trade.signature}, "
                f"sol_amount={trade.sol_amount}, "
                f"token_amount={trade.token_amount},"
                f"user={trade.user}")
                
            if trade.is_buy:
                # Directly confirm entry with trade details from WebSocket
                if not await self.position_tracker.has_pending_entry(trade.mint):
                    self.logger.info(f"Skipping entry confirmation for {trade.mint} because it's not pending")
                    return
                await self.position_tracker.confirm_entry(trade)
                await self.risk_manager.update_capital_after_entry(trade.total_sol_change)
            else:
                # Directly confirm exit with trade details from WebSocket
                if not await self.position_tracker.has_pending_exit(trade.mint):
                    self.logger.info(f"Skipping exit confirmation for {trade.mint} because it's not pending")
                    return
                await self.position_tracker.confirm_exit(trade)
                await self.risk_manager.update_capital_after_exit(trade.total_sol_change)
        except Exception as e:
            self.logger.error(f"Error processing wallet tracking: {str(e)}")
    
    async def _handle_exit_checks(self, trade, position):
        """Handle exit checks for existing position"""
        try:    
            self.logger.info(f"Checking exit conditions for {trade.mint}: "
                f"price={trade.price}, "
                f"signature={trade.signature}, "
                f"sol_amount={trade.sol_amount}, "
                f"token_amount={trade.token_amount}")

            if await self.position_tracker.has_pending_exit(trade.mint):
                self.logger.info(f"Skipping exit check for {trade.mint} because it's already pending")
                return   

            # Skip trades that happened before our entry
            if not self.backtest_mode and position.entry_blocktime >= trade.blocktime:
                self.logger.info(f"Skipping exit check for {trade.mint} because it's before or at the entry blocktime")
                return

            exit_signal = self.strategy.check_exit(
                trade.mint,
                trade.user,
                position.wallet_followed,
                position.entry_price,
                float(trade.price),
            )
            stop_loss_hit = self.risk_manager.check_stop_loss(position.entry_price, float(trade.price))

            # Execute exit if conditions met
            if (exit_signal and exit_signal.is_valid) or stop_loss_hit:
                exit_reason = "Strategy Exit" if exit_signal.is_valid else "Stop Loss"

                return True

        except Exception as e:
            self.logger.error(f"Exit check error: Mint={trade.mint}, Error={str(e)}")
            
    async def _handle_entry_checks(self, trade):
        """Handle entry checks for new position"""
        if not self.is_accepting_new_trades:
            self.logger.info("System is in shutdown mode - no new positions allowed")
            return False, 0.0
        
        try:
            self.logger.info(f"Checking entry conditions for {trade.user}: "
                f"Mint={trade.mint}, "
                f"price={trade.price}, "
                f"signature={trade.signature}, "
                f"sol_amount={trade.sol_amount}, "
                f"token_amount={trade.token_amount}")
            
            # Clean up any stale pending entries that might have timed out
            # Using a shorter timeout of 15 seconds for cleanup during entry processing
            await self.position_tracker.clear_stale_transactions(timeout_minutes=0.2)
            
            # Check if we already have a pending entry for this token
            if await self.position_tracker.has_pending_entry(trade.mint):
                self.logger.info(f"Already have a pending entry for {trade.mint}, skipping")
                return False, 0.0

            # Generate entry signal directly from trade
            entry_signal = self.strategy.generate_signal(trade)
            
            if not (entry_signal and entry_signal.is_valid):
                return False, 0.0
            
            # Get all positions and pending entries
            current_positions = await self.position_tracker.get_all_positions()
            pending_entries = await self.position_tracker.get_pending_entries()
            
            # Check total position count (active + pending)
            total_positions = len(current_positions) + len(pending_entries)
            
            # Risk Manager Entry Check with pending allocations considered
            can_enter, position_size = await self.risk_manager.can_enter_position(
                trade.mint, 
                total_positions
            )
            
            if not (can_enter and position_size > 0):
                return False, 0.0
                
            # Return success and position size
            return True, position_size
                
        except Exception as e:
            self.logger.error(f"Entry check error: Mint={trade.mint}, Error={str(e)}")
            return False, 0.0

    async def execute_trade(self, trade: TradeEvent, amount: float = None, is_buy: bool = True) -> None:
        """
        Execute a trade (entry or exit)
        Args:
            trade: Trade event with details
            position_size: Position size for entry (if applicable)
        Returns: 
            None - all tracking is handled through position_tracker
        """
        try:
            if self.dry_run:
                if is_buy:
                    await self.position_tracker.add_pending_entry(trade.mint, trade.user)
                else:
                    await self.position_tracker.add_pending_exit(trade.mint, trade.user)
                self.logger.info(f"Dry run - Transaction for {trade.mint} confirmed")
                return

            if is_buy:
                await self.executor.buy_token(
                    mint=Pubkey.from_string(trade.mint),
                    amount_sol=amount,
                    price=trade.price
                )
                await self.position_tracker.add_pending_entry(trade.mint, trade.user)
                self.logger.info(f"Buy transaction for {trade.mint} submitted")
            else:
                await self.executor.sell_token(
                    mint=Pubkey.from_string(trade.mint),
                    token_amount=amount
                )
                await self.position_tracker.add_pending_exit(trade.mint, trade.user)
                self.logger.info(f"Sell transaction for {trade.mint} submitted")
                    
        except Exception as e:
            self.logger.error(f"Error executing transaction for {trade.mint}: {str(e)}")
                