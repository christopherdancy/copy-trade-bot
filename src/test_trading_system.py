import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from dataclasses import dataclass
from core.copy_trading_system import TradingSystem
from strategies.copy_trading_strategy import CopyTradingStrategy
from risk.risk_manager import RiskManager
from utils.config import RiskParameters
import pandas as pd
from pathlib import Path
import csv
from risk.position_tracker import PositionTracker
import os

@dataclass
class TestConfig:
    name: str
    initial_capital: float
    strategy_params: Dict
    risk_params: Dict

@dataclass
class TestResult:
    # Basic Info
    config_name: str
    
    # Trade Stats
    total_trades: int
    
    # Performance Metrics
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_trade_pnl: float
    avg_hold_time: float
    total_fees: float
    
    # Risk Analysis
    risk_analysis: Dict
    wallet_analysis: Dict

    # Strategy Parameters (for comparison)
    strategy_params: Dict
    risk_params: Dict

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

class TradingSystemTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def run_test(self, config: TestConfig) -> TestResult:
        """Run a single test with specified configuration"""
        system = None
        try:
            # Load tracked wallets directly
            tracked_wallets = load_tracked_wallets('data/tracked_wallets.csv')
            
            # Set up strategy parameters with tracked wallets
            strategy_params = {
                'tracked_wallets': tracked_wallets,
                'min_trade_amount': config.strategy_params['min_trade_amount'],
                'max_time_in_trade': config.strategy_params['max_time_in_trade'],
                'take_profit_pct': config.strategy_params['take_profit_pct'],
            }
            
            risk_params = {
                'max_position_size': config.risk_params['max_position_size'],
                'stop_loss_pct': config.risk_params['stop_loss_pct'],
                'max_positions': config.risk_params['max_positions'],
                'max_hold_time_minutes': config.risk_params['max_hold_time_minutes'],
                'max_daily_loss_pct': config.risk_params['max_daily_loss_pct']
            }
            
            # Create position tracker for this test
            position_tracker = PositionTracker(csv_path=f"results/trades/test_{config.name}_{self.timestamp}.csv")
            
            system = TradingSystem(
                initial_capital=config.initial_capital,
                dry_run=True,
                backtest_mode=True,
                backtest_data_path="backtest_data/2025_04_09.csv",
                strategy_params=strategy_params,
                risk_params=risk_params,
                position_tracker=position_tracker
            )
            
            # Start the trading system
            await system.start()
            
            # Get performance analysis from position tracker
            analysis = position_tracker.analyze_performance()
            
            result = TestResult(
                # Basic Info
                config_name=config.name,
                
                # Trade Stats
                total_trades=analysis['overall']['total_trades'],
                
                # Performance Metrics
                total_pnl=analysis['overall']['total_pnl'],
                win_rate=analysis['overall']['winning_trades'] / analysis['overall']['total_trades'] if analysis['overall']['total_trades'] > 0 else 0,
                avg_win=analysis['overall']['average_pnl'] if analysis['overall']['winning_trades'] > 0 else 0,
                avg_loss=analysis['overall']['average_pnl'] if analysis['overall']['losing_trades'] > 0 else 0,
                avg_trade_pnl=analysis['overall']['average_pnl'],
                avg_hold_time=analysis['overall']['average_hold_time'],
                total_fees=analysis['overall']['total_fees_paid'],
                
                # Analysis Components
                risk_analysis=analysis['risk_analysis'],
                wallet_analysis=analysis['wallet_analysis'],
                # Parameters
                strategy_params=config.strategy_params,
                risk_params=config.risk_params
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"Error during test {config.name}: {e}")
            raise
        finally:
            # Ensure system is properly stopped even if an error occurs
            if system and system.is_running:
                try:
                    await system.stop()
                    print(f"Successfully stopped trading system for test: {config.name}")
                except Exception as e:
                    print(f"Error stopping trading system: {e}")

    def export_results(self):
        """Export summary results comparing different configurations"""
        results_data = []
        
        for r in self.results:
            # Get wallet details directly from wallet_analysis field
            wallet_details = {}
            if hasattr(r, 'wallet_analysis'):
                wallet_details = r.wallet_analysis.get('wallet_details', {})
            
            result_dict = {
                # Basic Info
                "config_name": r.config_name,
                
                # Trade Stats
                "total_trades": r.total_trades,
                
                # Performance Metrics
                "total_pnl": r.total_pnl,
                "win_rate": r.win_rate,
                "avg_win": r.avg_win,
                "avg_loss": r.avg_loss,
                "avg_trade_pnl": r.avg_trade_pnl,
                "avg_hold_time": r.avg_hold_time,
                "total_fees": r.total_fees,
                
                # Risk Analysis
                "profit_factor": r.risk_analysis.get('profit_factor', 0),
                "avg_win_loss_ratio": r.risk_analysis.get('avg_win_loss_ratio', 0),
                
                # Wallet Performance - detailed information
                "wallet_details": wallet_details,
                
                # Strategy Parameters
                "strategy_params": r.strategy_params,
                "risk_params": r.risk_params
            }
            results_data.append(result_dict)
        
        # Convert to DataFrame for CSV export
        df = pd.DataFrame(results_data)
        
        # Handle nested dictionaries for CSV export
        for col in ['wallet_details', 'strategy_params', 'risk_params']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x))
        
        # Create directory if it doesn't exist
        os.makedirs(self.results_dir / "config", exist_ok=True)
        
        csv_path = self.results_dir / f"config/config_comparison_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)
        

async def main():
    # Define baseline parameters
    baseline_risk_params = {
        'max_position_size': 0.07,
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

    test_configs = [
        TestConfig(
            name="baseline_3_17_27",
            initial_capital=.715,
            strategy_params={**baseline_strategy_params},
            risk_params={**baseline_risk_params}
        ),
    ]
    
    # Run tests
    tester = TradingSystemTester()
    for config in test_configs:
        print(f"\nStarting test: {config.name}")
        print(f"Strategy Params: {config.strategy_params}")
        print(f"Risk Params: {config.risk_params}")
        result = await tester.run_test(config)
        tester.export_results()
        print(f"Test completed: {config.name}")
        print(f"PnL: {result.total_pnl:.3f} SOL")
        print(f"Win Rate: {result.win_rate:.2%}")
    

if __name__ == "__main__":
    asyncio.run(main()) 