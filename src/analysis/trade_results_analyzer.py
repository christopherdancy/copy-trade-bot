import pandas as pd
from datetime import datetime
from typing import Dict, List
import numpy as np

class TradeResultsAnalyzer:
    def __init__(self, trades_file: str):
        """Initialize with path to trades CSV file"""
        self.df = pd.read_csv(trades_file)
        self.df['entry_time'] = pd.to_datetime(self.df['entry_time'])
        self.df['exit_time'] = pd.to_datetime(self.df['exit_time'])
        
        # Calculate basic metrics
        self.df['pnl'] = pd.to_numeric(self.df['pnl'])
        self.df['entry_hour'] = self.df['entry_time'].dt.hour
        self.df['entry_minute'] = self.df['entry_time'].dt.minute
        self.df['entry_day'] = self.df['entry_time'].dt.dayofweek
        
        # Add time segments
        self.define_time_segments()
        
    def define_time_segments(self):
        """Define trading session time segments"""
        hour = self.df['entry_hour']
        
        # Define trading sessions
        conditions = [
            (hour >= 0) & (hour < 4),    # Late Night/Early Morning UTC
            (hour >= 4) & (hour < 8),    # Asian Session
            (hour >= 8) & (hour < 12),   # European Session Open
            (hour >= 12) & (hour < 16),  # European/US Overlap
            (hour >= 16) & (hour < 20),  # US Session
            (hour >= 20) & (hour < 24)   # US Close/Asian Open
        ]
        
        segments = [
            'late_night',
            'asian_session',
            'european_open',
            'euro_us_overlap',
            'us_session',
            'us_close_asian_open'
        ]
        
        self.df['time_segment'] = np.select(conditions, segments, default='unknown')
    
    def analyze_time_segments(self) -> Dict:
        """Analyze performance by time segment"""
        segments_analysis = {}
        
        for segment in self.df['time_segment'].unique():
            segment_data = self.df[self.df['time_segment'] == segment]
            winning_trades = segment_data[segment_data['pnl'] > 0]
            
            segments_analysis[segment] = {
                'trade_count': len(segment_data),
                'win_rate': len(winning_trades) / len(segment_data) if len(segment_data) > 0 else 0,
                'avg_pnl': segment_data['pnl'].mean(),
                'median_pnl': segment_data['pnl'].median(),
                'avg_volume': segment_data['entry_volume'].mean(),
                'avg_trigger_move': segment_data['entry_trigger_move'].mean(),
                'avg_confirmation_move': segment_data['entry_confirmation_move'].mean(),
                'avg_hold_time': segment_data['hold_time_minutes'].mean(),
                'big_losses': len(segment_data[segment_data['pnl'] <= -0.05]),
                'profit_factor': abs(
                    winning_trades['pnl'].sum() / 
                    segment_data[segment_data['pnl'] <= 0]['pnl'].sum()
                ) if len(segment_data[segment_data['pnl'] <= 0]) > 0 else float('inf')
            }
        
        return segments_analysis

    def analyze_trades(self) -> Dict:
        """Analyze trade results and return summary statistics"""
        BIG_LOSS_THRESHOLD = -0.05  # 5% loss threshold
        
        # Split trades into categories
        winning_trades = self.df[self.df['pnl'] > 0]
        losing_trades = self.df[self.df['pnl'] <= 0]
        big_losses = self.df[self.df['pnl'] <= BIG_LOSS_THRESHOLD]
        
        # Time analysis
        time_analysis = {
            'hourly_performance': {
                'best_hours': self.df.groupby('entry_hour')['pnl'].mean().nlargest(3).to_dict(),
                'worst_hours': self.df.groupby('entry_hour')['pnl'].mean().nsmallest(3).to_dict(),
                'highest_win_rate_hours': (self.df[self.df['pnl'] > 0].groupby('entry_hour').size() / 
                                         self.df.groupby('entry_hour').size()).nlargest(3).to_dict(),
                'most_active_hours': self.df['entry_hour'].value_counts().head(3).to_dict(),
            },
            'day_of_week_performance': {
                'best_days': self.df.groupby('entry_day')['pnl'].mean().nlargest(3).to_dict(),
                'worst_days': self.df.groupby('entry_day')['pnl'].mean().nsmallest(3).to_dict(),
                'highest_win_rate_days': (self.df[self.df['pnl'] > 0].groupby('entry_day').size() / 
                                        self.df.groupby('entry_day').size()).nlargest(3).to_dict()
            },
            'session_performance': self.analyze_time_segments()
        }
        
        # Add best/worst performing segments
        best_segment = max(time_analysis['session_performance'].items(), 
                         key=lambda x: x[1]['avg_pnl'])
        worst_segment = min(time_analysis['session_performance'].items(), 
                          key=lambda x: x[1]['avg_pnl'])
        
        time_analysis['best_performing_segment'] = {
            'segment': best_segment[0],
            'avg_pnl': best_segment[1]['avg_pnl']
        }
        
        time_analysis['worst_performing_segment'] = {
            'segment': worst_segment[0],
            'avg_pnl': worst_segment[1]['avg_pnl']
        }
        
        # Correlation analysis
        correlation_analysis = {
            'volume_vs_pnl': self.df['entry_volume'].corr(self.df['pnl']),
            'hold_time_vs_pnl': self.df['hold_time_minutes'].corr(self.df['pnl']),
            'entry_price_vs_pnl': self.df['entry_price'].corr(self.df['pnl']),
            'volume_vs_hold_time': self.df['entry_volume'].corr(self.df['hold_time_minutes']),
            'trigger_move_vs_pnl': self.df['entry_trigger_move'].corr(self.df['pnl']),
            'confirmation_move_vs_pnl': self.df['entry_confirmation_move'].corr(self.df['pnl'])
        }
        
        # Entry price analysis for big losses
        big_loss_price_analysis = {
            'price_percentiles': {
                'p10': big_losses['entry_price'].quantile(0.1),
                'p25': big_losses['entry_price'].quantile(0.25),
                'p50': big_losses['entry_price'].quantile(0.5),
                'p75': big_losses['entry_price'].quantile(0.75),
                'p90': big_losses['entry_price'].quantile(0.9)
            },
            'avg_entry_price': big_losses['entry_price'].mean(),
            'std_entry_price': big_losses['entry_price'].std(),
            'price_range': {
                'min': big_losses['entry_price'].min(),
                'max': big_losses['entry_price'].max()
            }
        }
        
        summary_stats = {
            'Overall_Statistics': {
                'total_trades': len(self.df),
                'win_rate': len(winning_trades) / len(self.df) if len(self.df) > 0 else 0,
                'avg_profit': self.df['pnl'].mean() if not self.df.empty else 0,
                'median_profit': self.df['pnl'].median() if not self.df.empty else 0,
                'largest_win': self.df['pnl'].max() if not self.df.empty else 0,
                'largest_loss': self.df['pnl'].min() if not self.df.empty else 0,
                'avg_hold_time': self.df['hold_time_minutes'].mean() if not self.df.empty else 0,
                'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty else 0
            },
            'Time_Analysis': time_analysis,
            'Correlation_Analysis': correlation_analysis,
            'Big_Loss_Analysis': {
                'count': len(big_losses),
                'price_analysis': big_loss_price_analysis,
                'avg_volume': big_losses['entry_volume'].mean() if not big_losses.empty else 0,
                'avg_trigger_move': big_losses['entry_trigger_move'].mean() if not big_losses.empty else 0,
                'avg_confirmation_move': big_losses['entry_confirmation_move'].mean() if not big_losses.empty else 0,
                'avg_hold_time': big_losses['hold_time_minutes'].mean() if not big_losses.empty else 0,
                'most_common_hours': big_losses['entry_hour'].value_counts().head(3).to_dict(),
                'most_common_days': big_losses['entry_day'].value_counts().head(3).to_dict()
            }
        }
        
        return summary_stats

    def export_analysis(self, output_file: str):
        """Export analysis results to CSV"""
        summary_stats = self.analyze_trades()
        
        # Flatten the nested dictionary for CSV export
        flat_data = {}
        for category, stats in summary_stats.items():
            for metric, value in stats.items():
                flat_data[f'{category.lower()}_{metric}'] = value
        
        # Convert to DataFrame and export
        summary_df = pd.DataFrame([flat_data])
        summary_df.to_csv(output_file, index=False)

def main():
    # Example usage
    analyzer = TradeResultsAnalyzer("results/trades_max_price_215k_stop_loss_2pct_20250210_012943.csv")
    stats = analyzer.analyze_trades()
    
    # Print summary
    for category, metrics in stats.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
            
    # Export to CSV
    analyzer.export_analysis("results/trade_analysis_summary3.csv")

if __name__ == "__main__":
    main() 