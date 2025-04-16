from typing import List, Dict, Optional, Union
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from strategies.coordinated_move_strategy import Signal
from core.aggregator import Candle

@dataclass
class CandleMetrics:
    largest_buyer_pct: float
    unique_buyers: int
    max_trade_amount: float
    avg_trade_amount: float
    volume: float
    buy_volume: float
    sell_volume: float

@dataclass
class TradeRecord:
    token: str
    entry_time: datetime
    entry_price: int  # In lamports
    entry_token_amount: int  # Raw token amount
    entry_sol_amount: int  # In lamports
    entry_network_fee: int  # In lamports
    entry_pump_fee: int  # In lamports
    entry_token_account_fee: int  # In lamports
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[int] = None  # In lamports
    exit_token_amount: Optional[int] = None  # Raw token amount
    exit_sol_amount: Optional[int] = None  # In lamports
    exit_network_fee: Optional[int] = None  # In lamports
    exit_pump_fee: Optional[int] = None  # In lamports
    exit_token_account_refund: Optional[int] = None  # In lamports
    hold_time: Optional[float] = None  # In minutes
    gross_pnl: Optional[float] = None  # In SOL
    net_pnl: Optional[float] = None  # In SOL
    gross_roi: Optional[float] = None  # Percentage
    net_roi: Optional[float] = None  # Percentage
    wallet_address: Optional[str] = None

class PositionTracker:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.current_positions: Dict[str, TradeRecord] = {}
        self.initialize_csv()

    def initialize_csv(self):
        """Create CSV with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            columns = [
                'token', 'entry_time', 'entry_price', 'entry_token_amount', 
                'entry_sol_amount', 'entry_network_fee', 'entry_pump_fee',
                'entry_token_account_fee',
                'exit_time', 'exit_price', 'exit_token_amount', 
                'exit_sol_amount', 'exit_network_fee', 'exit_pump_fee', 'exit_token_account_refund',
                'hold_time', 'gross_pnl', 'net_pnl', 'gross_roi', 'net_roi',
                'wallet_address'
            ]
            pd.DataFrame(columns=columns).to_csv(self.csv_path, index=False)

    def record_entry(self, token: str, tx_result: dict, timestamp: datetime, signal: Signal) -> None:
        """Record entry position with candle metrics"""

        trade = TradeRecord(
            token=token,
            entry_time=timestamp,
            entry_price=abs(tx_result['execution_price']),
            entry_sol_amount=tx_result['sol_trade_amount'],
            entry_token_amount=tx_result['token_trade_amount'],
            entry_network_fee=tx_result['network_fee'],
            entry_pump_fee=tx_result['pump_fee'],
            entry_token_account_fee=tx_result['token_account_fee'],
            wallet_address=signal.wallet_address
        )
        self.current_positions[token] = trade
        
    def record_exit(self, token: str, tx_result: dict, timestamp: datetime) -> None:
        """Record exit and save completed trade to CSV"""
        try:
            if token not in self.current_positions:
                print(f"No entry found for token {token}")
                return

            trade = self.current_positions[token]
            exit_time = timestamp
            
            # Calculate hold time in minutes
            hold_time = (exit_time - trade.entry_time).total_seconds() / 60
            
            # Calculate PnL in SOL
            gross_pnl = tx_result['sol_trade_amount'] - trade.entry_sol_amount
            gross_roi = (gross_pnl / trade.entry_sol_amount) * 100 if trade.entry_sol_amount != 0 else 0
            
            
            net_entry_cost = (trade.entry_sol_amount + trade.entry_network_fee + trade.entry_pump_fee + trade.entry_token_account_fee)
            net_exit_value = (tx_result['sol_trade_amount'] - tx_result['network_fee'] - tx_result['pump_fee'] + tx_result['token_account_refund'])
            net_pnl = net_exit_value - net_entry_cost
            # Calculate ROI
            net_roi = (net_pnl / net_entry_cost) * 100 if net_entry_cost != 0 else 0

            # Update trade record
            trade.exit_time = exit_time
            trade.exit_price = tx_result['execution_price']
            trade.exit_token_amount = tx_result['token_trade_amount']
            trade.exit_sol_amount = tx_result['sol_trade_amount']
            trade.exit_network_fee = tx_result['network_fee']
            trade.exit_pump_fee = tx_result['pump_fee']
            trade.exit_token_account_refund = tx_result['token_account_refund']
            trade.hold_time = hold_time
            trade.gross_pnl = gross_pnl
            trade.gross_roi = gross_roi
            trade.net_pnl = net_pnl
            trade.net_roi = net_roi

            # Save to CSV
            trade_dict = vars(trade)
            df = pd.DataFrame([trade_dict])
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
            # Remove from current positions
            del self.current_positions[token]
        except Exception as e:
            print(f"Error recording exit: {e}")

    def analyze_performance(self) -> dict:
        """Analyze trading performance with detailed statistical analysis"""
        df = pd.read_csv(self.csv_path)
        if len(df) == 0:
            return {"message": "No trades recorded yet"}

        # Split into winning and losing trades
        winners = df[df['gross_roi'] > 0]
        losers = df[df['gross_roi'] <= 0]

        def analyze_metric_distribution(metric_name: str, winners_data: pd.Series, losers_data: pd.Series) -> dict:
            """Analyze distribution statistics for a single metric with enhanced data cleaning"""
            # Convert to float and clean the data more thoroughly
            def clean_series(series):
                return (pd.to_numeric(series, errors='coerce')
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna())

            winners_data = clean_series(winners_data)
            losers_data = clean_series(losers_data)
            combined_data = pd.concat([winners_data, losers_data])
            
            # Safe statistical calculations
            def safe_stat(series, stat_func, default=0):
                try:
                    if len(series) > 0:
                        result = stat_func(series)
                        return float(result) if pd.notnull(result) else default
                    return default
                except:
                    return default

            # Only calculate percentiles if we have enough data
            if len(combined_data) >= 10:  # Arbitrary minimum size for meaningful percentiles
                percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
                threshold_values = np.percentile(combined_data, percentiles)
                winners_above_threshold = [
                    safe_stat(winners_data > threshold, np.mean, 0) * 100 
                    for threshold in threshold_values
                ]
            else:
                percentiles = []
                threshold_values = []
                winners_above_threshold = []

            return {
                "winners": {
                    "median": safe_stat(winners_data, np.median),
                    "q1": safe_stat(winners_data, lambda x: np.percentile(x, 25)),
                    "q3": safe_stat(winners_data, lambda x: np.percentile(x, 75)),
                    "min": safe_stat(winners_data, np.min),
                    "max": safe_stat(winners_data, np.max),
                    "std": safe_stat(winners_data, np.std),
                    "count": len(winners_data)
                },
                "losers": {
                    "median": safe_stat(losers_data, np.median),
                    "q1": safe_stat(losers_data, lambda x: np.percentile(x, 25)),
                    "q3": safe_stat(losers_data, lambda x: np.percentile(x, 75)),
                    "min": safe_stat(losers_data, np.min),
                    "max": safe_stat(losers_data, np.max),
                    "std": safe_stat(losers_data, np.std),
                    "count": len(losers_data)
                },
                "threshold_analysis": {
                    "percentiles": percentiles,
                    "threshold_values": threshold_values.tolist() if len(threshold_values) > 0 else [],
                    "winners_pct_above_threshold": winners_above_threshold
                }
            }

        def analyze_candle_type(prefix: str) -> dict:
            """Analyze all metrics for a specific candle type"""
            metrics = {
                f"{prefix}_largest_buyer_pct": "Largest Buyer %",
                f"{prefix}_unique_buyers": "Unique Buyers",
                f"{prefix}_volume": "Total Volume",
                f"{prefix}_buy_volume": "Buy Volume",
                f"{prefix}_sell_volume": "Sell Volume",
                f"{prefix}_avg_trade": "Average Trade Size",
                f"{prefix}_max_trade": "Maximum Trade Size",
            }
            
            results = {}
            for col, label in metrics.items():
                results[label] = analyze_metric_distribution(
                    col,
                    winners[col],
                    losers[col]
                )
                
            # Add buy/sell ratio analysis
            buy_sell_ratio_winners = winners[f"{prefix}_buy_volume"] / winners[f"{prefix}_sell_volume"]
            buy_sell_ratio_losers = losers[f"{prefix}_buy_volume"] / losers[f"{prefix}_sell_volume"]
            results["Buy/Sell Ratio"] = analyze_metric_distribution(
                f"{prefix}_buy_sell_ratio",
                buy_sell_ratio_winners,
                buy_sell_ratio_losers
            )
            
            return results

        # Analyze progression between candles
        def analyze_volume_progression():
            """Analyze how volume changes between candles"""
            def calc_progression(df, col_suffix):
                """Calculate progression ratios between candles safely"""
                # Initialize Series with float dtype explicitly
                base_to_trigger = pd.Series(0.0, index=df.index, dtype=float)
                trigger_to_conf = pd.Series(0.0, index=df.index, dtype=float)
                
                try:
                    # Handle division by zero and invalid values
                    base_values = df[f"base_{col_suffix}"].astype(float).replace(0, np.nan)
                    trigger_values = df[f"trigger_{col_suffix}"].astype(float).replace(0, np.nan)
                    conf_values = df[f"confirmation_{col_suffix}"].astype(float).replace(0, np.nan)
                    
                    # Only calculate where we have valid values
                    valid_b2t = (base_values > 0) & (trigger_values > 0)
                    valid_t2c = (trigger_values > 0) & (conf_values > 0)
                    
                    # Calculate progressions where valid
                    if valid_b2t.any():
                        base_to_trigger.loc[valid_b2t] = (
                            (trigger_values[valid_b2t] / base_values[valid_b2t]) - 1
                        ).astype(float)
                        
                    if valid_t2c.any():
                        trigger_to_conf.loc[valid_t2c] = (
                            (conf_values[valid_t2c] / trigger_values[valid_t2c]) - 1
                        ).astype(float)
                    
                    # Clean any remaining invalid values
                    base_to_trigger = base_to_trigger.replace([np.inf, -np.inf], np.nan)
                    trigger_to_conf = trigger_to_conf.replace([np.inf, -np.inf], np.nan)
                    
                except Exception as e:
                    print(f"Warning: Error in progression calculation for {col_suffix}: {e}")
                
                return base_to_trigger, trigger_to_conf

            metrics = {
                "volume": "Volume Progression",
                "buy_volume": "Buy Volume Progression",
                "sell_volume": "Sell Volume Progression"
            }
            
            progression_analysis = {}
            for suffix, label in metrics.items():
                try:
                    winners_b2t, winners_t2c = calc_progression(winners, suffix)
                    losers_b2t, losers_t2c = calc_progression(losers, suffix)
                    
                    progression_analysis[label] = {
                        "base_to_trigger": analyze_metric_distribution(
                            f"{suffix}_base_to_trigger",
                            winners_b2t.replace([np.inf, -np.inf], np.nan).dropna(),
                            losers_b2t.replace([np.inf, -np.inf], np.nan).dropna()
                        ),
                        "trigger_to_confirmation": analyze_metric_distribution(
                            f"{suffix}_trigger_to_confirmation",
                            winners_t2c.replace([np.inf, -np.inf], np.nan).dropna(),
                            losers_t2c.replace([np.inf, -np.inf], np.nan).dropna()
                        )
                    }
                except Exception as e:
                    print(f"Warning: Error calculating progression for {suffix}: {e}")
                    progression_analysis[label] = {
                        "base_to_trigger": {
                            "winners": {"median": 0, "q1": 0, "q3": 0, "min": 0, "max": 0, "std": 0, "count": 0},
                            "losers": {"median": 0, "q1": 0, "q3": 0, "min": 0, "max": 0, "std": 0, "count": 0},
                            "threshold_analysis": {"percentiles": [], "threshold_values": [], "winners_pct_above_threshold": []}
                        },
                        "trigger_to_confirmation": {
                            "winners": {"median": 0, "q1": 0, "q3": 0, "min": 0, "max": 0, "std": 0, "count": 0},
                            "losers": {"median": 0, "q1": 0, "q3": 0, "min": 0, "max": 0, "std": 0, "count": 0},
                            "threshold_analysis": {"percentiles": [], "threshold_values": [], "winners_pct_above_threshold": []}
                        }
                    }
            
            return progression_analysis

        # Add move analysis
        # move_analysis = {
        #     "trigger_move": analyze_metric_distribution(
        #         "trigger_move",
        #         winners['trigger_move'],
        #         losers['trigger_move']
        #     ),
        #     "confirmation_move": analyze_metric_distribution(
        #         "confirmation_move",
        #         winners['confirmation_move'],
        #         losers['confirmation_move']
        #     )
        # }

        # Get wallet analysis
        wallet_analysis = self.analyze_wallet_performance()

        analysis = {
            "overall": {
                "total_trades": len(df),
                "winning_trades": len(winners),
                "losing_trades": len(losers),
                "win_rate": len(winners) / len(df) * 100 if len(df) > 0 else 0,
                "average_hold_time": df['hold_time'].mean(),
                "total_pnl": df['gross_pnl'].sum(),
                "average_pnl": df['gross_pnl'].mean(),
                "median_roi": df['gross_roi'].median(),
                "roi_std": df['gross_roi'].std(),
                "average_roi": df['gross_roi'].mean(),
                "total_fees_paid": (
                    df['entry_network_fee'].sum() + 
                    df['entry_pump_fee'].sum() + 
                    df['exit_network_fee'].sum() + 
                    df['exit_pump_fee'].sum() +
                    df['entry_token_account_fee'].sum()
                ) / 1e9,
            },
            # "candle_analysis": {
            #     "base": analyze_candle_type("base"),
            #     "trigger": analyze_candle_type("trigger"),
            #     "confirmation": analyze_candle_type("confirmation")
            # },
            # "progression_analysis": analyze_volume_progression(),
            "risk_analysis": {
                "profit_factor": abs(winners['gross_pnl'].sum() / losers['gross_pnl'].sum()) 
                               if len(losers) > 0 and losers['gross_pnl'].sum() != 0 else float('inf'),
                "avg_win_loss_ratio": abs(winners['gross_pnl'].mean() / losers['gross_pnl'].mean()) 
                                    if len(losers) > 0 and losers['gross_pnl'].mean() != 0 else float('inf')
            },
            "wallet_analysis": wallet_analysis
            # "correlation_analysis": self.analyze_correlations(df),
            # "move_analysis": move_analysis  # Add the move analysis
        }
        
        return analysis

    def parse_trades_from_log(self, log_file_path: str) -> pd.DataFrame:
        """Parse trades from a log file and return them as a DataFrame"""
        trades = []
        current_trade = {}
        
        with open(log_file_path, 'r') as file:
            for line in file:
                try:
                    if 'Entry Signal Generated' in line:
                        # Reset current trade
                        current_trade = {}
                        # Extract timestamp and token
                        parts = line.split(' - ')
                        timestamp = parts[0]
                        mint = parts[3].split('Mint=')[1].split(',')[0]
                        price = float(parts[3].split('Price=')[1].split(',')[0])
                        current_trade['entry_time'] = timestamp
                        current_trade['token'] = mint
                        current_trade['entry_price'] = price
                    
                    elif 'Entry Successful' in line and current_trade:
                        # Extract entry details
                        parts = line.split(' - ')
                        details_str = parts[3].split('Details=')[1]
                        if details_str.endswith('...(line too long; chars omitted)'):
                            details_str = details_str.split('...')[0] + '}'
                        details = eval(details_str)
                        
                        # Safely get values with defaults
                        current_trade['entry_sol_amount'] = details.get('sol_trade_amount', 0)
                        current_trade['entry_token_amount'] = details.get('token_trade_amount', 0)
                        current_trade['entry_network_fee'] = details.get('network_fee', 0)
                        current_trade['entry_pump_fee'] = details.get('pump_fee', 0)
                        current_trade['entry_token_account_fee'] = details.get('token_account_fee', 0)
                    
                    elif 'Exit Successful' in line and current_trade:
                        # Extract exit details
                        parts = line.split(' - ')
                        timestamp = parts[0]
                        exit_details = parts[3].split('tx_details=')[1]
                        if exit_details.endswith('...(line too long; chars omitted)'):
                            exit_details = exit_details.split('...')[0] + '}'
                        details = eval(exit_details)
                        
                        # Safely get values with defaults
                        current_trade['exit_time'] = timestamp
                        current_trade['exit_price'] = details.get('execution_price', 0)
                        current_trade['exit_sol_amount'] = details.get('sol_trade_amount', 0)
                        current_trade['exit_token_amount'] = details.get('token_trade_amount', 0)
                        current_trade['exit_network_fee'] = details.get('network_fee', 0)
                        current_trade['exit_pump_fee'] = details.get('pump_fee', 0)
                        current_trade['exit_token_account_refund'] = details.get('token_account_refund', 0)
                        
                        # Calculate metrics only if we have all required data
                        if all(k in current_trade for k in ['entry_time', 'exit_time', 'entry_sol_amount']):
                            # Calculate PnL and other metrics
                            entry_time = pd.to_datetime(current_trade['entry_time'])
                            exit_time = pd.to_datetime(current_trade['exit_time'])
                            current_trade['hold_time'] = (exit_time - entry_time).total_seconds() / 60
                            
                            gross_pnl = current_trade['entry_sol_amount'] - current_trade['exit_sol_amount']
                            current_trade['gross_pnl'] = gross_pnl
                            
                            if current_trade['entry_sol_amount'] != 0:
                                current_trade['gross_roi'] = (gross_pnl / current_trade['entry_sol_amount']) * 100
                            else:
                                current_trade['gross_roi'] = 0
                            
                            net_entry_cost = (current_trade['entry_sol_amount'] + 
                                            current_trade['entry_network_fee'] + 
                                            current_trade['entry_pump_fee'] +
                                            current_trade['entry_token_account_fee'])
                            net_exit_value = (current_trade['exit_sol_amount'] - 
                                            current_trade['exit_network_fee'] - 
                                            current_trade['exit_pump_fee'] +
                                            current_trade['exit_token_account_refund'])
                            net_pnl = net_exit_value - net_entry_cost
                            current_trade['net_pnl'] = net_pnl
                            
                            if net_entry_cost != 0:
                                current_trade['net_roi'] = (net_pnl / net_entry_cost) * 100
                            else:
                                current_trade['net_roi'] = 0
                            
                            # Add completed trade to list
                            trades.append(current_trade.copy())
                            current_trade = {}
                
                except Exception as e:
                    print(f"Warning: Error processing line: {e}")
                    continue
        
        # Convert to DataFrame
        if trades:
            df = pd.DataFrame(trades)
            return df
        else:
            return pd.DataFrame(columns=[
                'token', 'entry_time', 'entry_price', 'entry_token_amount', 
                'entry_sol_amount', 'entry_network_fee', 'entry_pump_fee', 
                'entry_token_account_fee',
                'exit_time', 'exit_price', 'exit_token_amount', 
                'exit_sol_amount', 'exit_network_fee', 'exit_pump_fee', 
                'exit_token_account_refund',
                'hold_time', 'gross_pnl', 'net_pnl', 'gross_roi', 'net_roi'
            ])

    def analyze_correlations(self, df: pd.DataFrame) -> dict:
        """Analyze correlations between metrics and trading outcomes"""
        # Clean the data by removing NaN values
        df = df.fillna(0)  # or df.dropna() depending on your preference
        
        # Select relevant columns for correlation
        metric_columns = [col for col in df.columns 
                         if any(col.startswith(prefix) for prefix in ['base_', 'trigger_', 'confirmation_'])
                         and not col.endswith(('_time', '_price'))]
        
        # Add performance metrics for correlation
        target_columns = ['gross_roi', 'net_roi', 'hold_time']
        correlation_df = df[metric_columns + target_columns]
        
        # Calculate correlations only if we have data
        if len(correlation_df) > 0:
            correlation_matrix = correlation_df.corr()
            
            # Analyze correlations with performance metrics
            performance_correlations = {}
            for target in target_columns:
                correlations = correlation_matrix[target].sort_values(ascending=False)
                performance_correlations[target] = {
                    "strongest_positive": correlations[correlations > 0].drop(target, errors='ignore').head(5).to_dict(),
                    "strongest_negative": correlations[correlations < 0].head(5).to_dict()
                }
            
            # Find highly correlated metric pairs (excluding performance metrics)
            metric_correlations = correlation_matrix.loc[metric_columns, metric_columns]
            high_correlations = []
            for i in range(len(metric_columns)):
                for j in range(i+1, len(metric_columns)):
                    corr = metric_correlations.iloc[i, j]
                    if abs(corr) > 0.7:  # Threshold for strong correlation
                        high_correlations.append({
                            "metric1": metric_columns[i],
                            "metric2": metric_columns[j],
                            "correlation": corr
                        })
            
            # Analyze candle progression correlations
            candle_progression = {}
            for metric in ['volume', 'buy_volume', 'sell_volume', 'unique_buyers', 'largest_buyer_pct']:
                base_trigger_corr = correlation_df[f'base_{metric}'].corr(correlation_df[f'trigger_{metric}'])
                trigger_conf_corr = correlation_df[f'trigger_{metric}'].corr(correlation_df[f'confirmation_{metric}'])
                candle_progression[metric] = {
                    "base_to_trigger": base_trigger_corr,
                    "trigger_to_confirmation": trigger_conf_corr
                }
            
            return {
                "performance_correlations": performance_correlations,
                "high_metric_correlations": sorted(high_correlations, key=lambda x: abs(x['correlation']), reverse=True),
                "candle_progression_correlations": candle_progression
            }
        else:
            return {
                "performance_correlations": {},
                "high_metric_correlations": [],
                "candle_progression_correlations": {}
            }

    def analyze_wallet_performance(self) -> dict:
        """Analyze performance by wallet address to evaluate which wallets to follow"""
        df = pd.read_csv(self.csv_path)
        if len(df) == 0:
            return {"message": "No trades recorded yet"}
        
        # Filter out rows with missing wallet_address
        df = df[df['wallet_address'].notna()]
        
        # Group by wallet address
        wallet_groups = df.groupby('wallet_address')
        
        wallet_performance = {}
        
        for wallet, trades in wallet_groups:
            # Skip if wallet is empty string or None
            if not wallet or pd.isna(wallet):
                continue
            
            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = len(trades[trades['gross_roi'] > 0])
            losing_trades = total_trades - winning_trades
            
            # Calculate win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate PnL metrics
            total_pnl = trades['gross_pnl'].sum()
            avg_pnl = trades['gross_pnl'].mean()
            median_roi = trades['gross_roi'].median()
            avg_roi = trades['gross_roi'].mean()
            
            # Calculate risk metrics
            winners = trades[trades['gross_roi'] > 0]
            losers = trades[trades['gross_roi'] <= 0]
            
            profit_factor = abs(winners['gross_pnl'].sum() / losers['gross_pnl'].sum()) if len(losers) > 0 and losers['gross_pnl'].sum() != 0 else float('inf')
            avg_win = winners['gross_pnl'].mean() if len(winners) > 0 else 0
            avg_loss = losers['gross_pnl'].mean() if len(losers) > 0 else 0
            avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Calculate time metrics
            avg_hold_time = trades['hold_time'].mean()
            
            # Calculate fee metrics
            total_fees = (
                trades['entry_network_fee'].sum() + 
                trades['entry_pump_fee'].sum() + 
                trades['exit_network_fee'].sum() + 
                trades['exit_pump_fee'].sum() +
                trades['entry_token_account_fee'].sum()
            ) / 1e9  # Convert from lamports to SOL
            
            # Calculate expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
            # Store wallet performance data
            wallet_performance[wallet] = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate * 100,  # Convert to percentage
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "median_roi": median_roi,
                "avg_roi": avg_roi,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "avg_win_loss_ratio": avg_win_loss_ratio,
                "avg_hold_time": avg_hold_time,
                "total_fees": total_fees,
                "expectancy": expectancy,
                "last_trade_time": trades['exit_time'].max()
            }
        
        # Sort wallets by total PnL
        sorted_wallets = sorted(
            wallet_performance.items(), 
            key=lambda x: x[1]['total_pnl'], 
            reverse=True
        )
        
        # Create detailed wallet summary
        wallet_summary = {}
        for wallet, stats in sorted_wallets:
            wallet_summary[wallet] = {
                "rank": sorted_wallets.index((wallet, stats)) + 1,
                "total_trades": stats["total_trades"],
                "win_rate": f"{stats['win_rate']:.1f}%",
                "total_pnl": f"{stats['total_pnl']:.4f} SOL",
                "avg_roi": f"{stats['avg_roi']*100:.1f}%",
                "profit_factor": f"{stats['profit_factor']:.2f}",
                "expectancy": f"{stats['expectancy']:.4f} SOL/trade"
            }
        
        return {
            "total_wallets": len(wallet_performance),
            "wallet_details": wallet_summary,
            # Keep these for backward compatibility
            "best_performing_wallets": [w[0] for w in sorted_wallets[:5]],
            "worst_performing_wallets": [w[0] for w in sorted_wallets[-5:] if len(sorted_wallets) >= 5]
        }

def main():
    """Command line interface for parsing trade logs"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse trading logs and generate CSV report')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--output', '-o', type=str, 
                       default='results/trades/parsed_trades.csv',
                       help='Output CSV file path (default: results/trades/parsed_trades.csv)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Parse trades
    tracker = PositionTracker()
    try:
        trades_df = tracker.parse_trades_from_log(args.log_file)
        
        # Save to CSV
        trades_df.to_csv(args.output, index=False)
        
        # Print summary
        print(f"\nProcessed {len(trades_df)} trades")
        print(f"Results saved to: {args.output}")
            
    except Exception as e:
        print(f"Error processing log file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())