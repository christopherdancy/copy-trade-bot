import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from decimal import Decimal
from typing import Optional

class TradeVisualizer:
    def __init__(self, position_file: str, trades_file: str):
        """Initialize with both position analysis and trades data"""
        self.position_df = pd.read_csv(position_file)
        self.trades_df = pd.read_csv(trades_file)
        
        # Convert timestamps to datetime
        self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
        self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
        
        # Process numeric columns
        decimal_columns = ['entry_price', 'exit_price', 'pnl']
        for col in decimal_columns:
            self.trades_df[col] = self.trades_df[col].astype(float)

    def plot_trade(self, trade_num: int) -> None:
        """Create detailed visualization combining position and trade data"""
        position_data = self.position_df[self.position_df['trade_number'] == trade_num].copy()
        
        # Get corresponding trade data
        trade_data = self.trades_df[self.trades_df['mint'] == position_data['token'].iloc[0]].copy()
        
        if position_data.empty or trade_data.empty:
            print(f"No data found for trade {trade_num}")
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])
        fig.suptitle(f'Trade {trade_num} Analysis - {trade_data["mint"].iloc[0]}', fontsize=14)

        # Add trade info
        entry_time = trade_data['entry_time'].iloc[0]
        exit_time = trade_data['exit_time'].iloc[0]
        entry_price = trade_data['entry_price'].iloc[0]
        exit_price = trade_data['exit_price'].iloc[0]
        final_pnl = trade_data['pnl'].iloc[0] * 100
        hold_time = trade_data['hold_time_minutes'].iloc[0]
        
        # Plot position data with actual entry/exit points
        self._plot_candlesticks(ax1, position_data, entry_price, exit_price, 
                              entry_time, exit_time)
        self._plot_pnl_drawdown(ax2, position_data, final_pnl, hold_time)

        # Add trade summary
        summary = (f"Entry: ${entry_price:.8f}\n"
                  f"Exit: ${exit_price:.8f}\n"
                  f"PnL: {final_pnl:.2f}%\n"
                  f"Hold Time: {hold_time:.1f}min\n"
                  f"Exit Reason: {trade_data['exit_reason'].iloc[0]}")
        
        fig.text(0.02, 0.02, summary, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"trade_plots/trade_{trade_num}.png")
        plt.close()

    def _plot_candlesticks(self, ax, trade_data: pd.DataFrame, entry_price: float, exit_price: float, entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> None:
        """Enhanced candlestick visualization with entry and exit points"""
        for idx, row in trade_data.iterrows():
            # Candlestick body
            color = 'g' if row['close'] >= row['open'] else 'r'
            bottom = min(row['open'], row['close'])
            height = abs(row['close'] - row['open'])
            
            # Plot candle body and wicks
            ax.bar(row['minute'], height, bottom=bottom, color=color, width=0.6)
            ax.vlines(row['minute'], row['low'], row['high'], color=color)

            # Mark new highs
            if row['new_high']:
                ax.scatter(row['minute'], row['high'], marker='^', 
                         color='g', s=100, zorder=5)

        # Mark entry point
        ax.scatter(entry_time.minute, entry_price, color='blue', s=200, marker='o', 
                  label='Entry', zorder=5, linewidth=2)

        # Mark exit point
        ax.scatter(exit_time.minute, exit_price, color='red', s=200, marker='x', 
                  label='Exit', zorder=5, linewidth=2)

        # Add reference lines and labels
        ax.axhline(y=entry_price, color='gray', linestyle='--', alpha=0.5, label='Entry Price')
        ax.axhline(y=exit_price, color='gray', linestyle='--', alpha=0.5, label='Exit Price')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Price')
        ax.set_title('Price Action')
        ax.legend()

    def _plot_pnl_drawdown(self, ax, trade_data: pd.DataFrame, final_pnl: float, hold_time: float) -> None:
        """Enhanced PnL and drawdown visualization with exit point"""
        # Plot PnL line
        ax.plot(trade_data['minute'], trade_data['pnl_from_entry'] * 100,
                'b-', label='PnL %', linewidth=2, marker='o')
        
        # Plot drawdown line
        ax.plot(trade_data['minute'], trade_data['drawdown_from_high'] * 100,
                'r--', label='Drawdown %', linewidth=2, marker='o')

        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=5, color='g', linestyle='--', alpha=0.3, label='5% Profit')
        ax.axhline(y=10, color='g', linestyle='--', alpha=0.3, label='10% Profit')
        ax.axhline(y=-5, color='r', linestyle='--', alpha=0.3, label='5% Loss')

        # Mark exit point (last data point)
        final_minute = trade_data['minute'].iloc[-1]
        ax.scatter(final_minute, final_pnl, color='red', s=200, marker='x', 
                  label='Exit', zorder=5, linewidth=2)
        
        # Add exit annotation
        ax.annotate(f'Exit: {final_pnl:.1f}%\nHold: {hold_time:.1f}min',
                   (final_minute, final_pnl),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))

        # Add early trade window marker (first 2 minutes)
        ax.axvspan(0, 2, color='yellow', alpha=0.1, label='Early Trade Window')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Percentage')
        ax.set_xlabel('Minutes from Entry')
        ax.set_title('Profit/Loss and Drawdown')
        ax.legend()

    def analyze_all_trades(self, save_dir: str = "trade_plots") -> None:
        """Generate visualizations for all trades"""
        for trade_num in self.position_df['trade_number'].unique():
            print(f"Analyzing trade {trade_num}...")
            self.plot_trade(trade_num)

    def analyze_early_moves(self, minutes_threshold: int = 2) -> None:
        """Analyze trades with significant early moves within specified minutes"""
        early_move_trades = []

        # Group by trade number
        for trade_num in self.position_df['trade_number'].unique():
            trade_data = self.position_df[self.position_df['trade_number'] == trade_num].copy()
            
            # Get early trade data (first N minutes)
            early_data = trade_data[trade_data['minute'] <= minutes_threshold]
            if early_data.empty:
                continue

            # Calculate metrics
            early_max_pnl = early_data['pnl'].max() * 100
            early_min_pnl = early_data['pnl'].min() * 100
            final_pnl = trade_data['pnl'].iloc[-1] * 100
            max_pnl = trade_data['pnl'].max() * 100
            
            early_move_trades.append({
                'trade_number': trade_num,
                'token': trade_data['token'].iloc[0],
                f'{minutes_threshold}m_max_pnl': early_max_pnl,
                f'{minutes_threshold}m_min_pnl': early_min_pnl,
                'final_pnl': final_pnl,
                'overall_max_pnl': max_pnl,
                'max_reached_early': abs(early_max_pnl) > abs(max_pnl * 0.8),  # 80% of max reached early
                'duration': trade_data['minute'].max()
            })

        # Convert to DataFrame and sort by early max PnL
        early_moves_df = pd.DataFrame(early_move_trades)
        early_moves_df = early_moves_df.sort_values(f'{minutes_threshold}m_max_pnl', ascending=False)

        # Print analysis
        print(f"\nTrades Analysis - First {minutes_threshold} Minutes")
        print("=" * 80)
        
        # Positive early moves
        print("\nTop Positive Early Moves:")
        positive_moves = early_moves_df[early_moves_df[f'{minutes_threshold}m_max_pnl'] > 5]  # 5% threshold
        for _, trade in positive_moves.iterrows():
            print(f"\nTrade #{int(trade['trade_number'])} - {trade['token']}")
            print(f"  Early Max: {trade[f'{minutes_threshold}m_max_pnl']:.2f}%")
            print(f"  Final PnL: {trade['final_pnl']:.2f}%")
            print(f"  Overall Max: {trade['overall_max_pnl']:.2f}%")
            print(f"  Duration: {trade['duration']} minutes")
            if trade['max_reached_early']:
                print("  * Peak reached in early minutes")

        # Negative early moves
        print("\nWorst Early Dumps:")
        negative_moves = early_moves_df[early_moves_df[f'{minutes_threshold}m_min_pnl'] < -5]  # -5% threshold
        for _, trade in negative_moves.iterrows():
            print(f"\nTrade #{int(trade['trade_number'])} - {trade['token']}")
            print(f"  Early Min: {trade[f'{minutes_threshold}m_min_pnl']:.2f}%")
            print(f"  Final PnL: {trade['final_pnl']:.2f}%")
            print(f"  Overall Max: {trade['overall_max_pnl']:.2f}%")
            print(f"  Duration: {trade['duration']} minutes")

        # Save to CSV
        early_moves_df.to_csv(f"positions/early_moves_{minutes_threshold}min_analysis.csv", index=False)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total trades analyzed: {len(early_moves_df)}")
        print(f"Trades with >5% gain in first {minutes_threshold} mins: {len(positive_moves)}")
        print(f"Trades with >5% loss in first {minutes_threshold} mins: {len(negative_moves)}")
        print(f"Trades reaching 80% of max PnL in first {minutes_threshold} mins: "
              f"{len(early_moves_df[early_moves_df['max_reached_early']])}")

if __name__ == "__main__":
    visualizer = TradeVisualizer(
        "positions/position_analysis_20250206_150950_high_potential_trades.csv",
        "results/trades_take_profit_10pct_trailing_stop_8pct_20250206_150947.csv"
    )
    
    # Get all unique trade numbers
    trade_numbers = visualizer.position_df['trade_number'].unique()
    print(f"Found {len(trade_numbers)} trades to analyze")
    
    # Analyze each trade
    for trade_num in trade_numbers:
        print(f"Analyzing trade {trade_num}...")
        visualizer.plot_trade(trade_num)
        
    print("Analysis complete! Check the trade_plots directory for all visualizations.")
    
    # Analyze early moves (default 2 minutes)
    # visualizer.analyze_early_moves()
    
    # You can also specify a different threshold
    # visualizer.analyze_early_moves(minutes_threshold=1)  # Analysis for first minute only 