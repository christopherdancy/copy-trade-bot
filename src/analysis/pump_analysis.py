import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os

def analyze_pump_patterns(df: pd.DataFrame):
    # Basic statistics
    print("\n=== BASIC STATISTICS ===")
    print(f"Total number of trades: {len(df)}")
    print(f"Total number of unique tokens: {df['token_mint'].nunique()}")
    print(f"Total SOL volume: {df['sol_amount'].sum():.2f}")
    
    # Analyze pump patterns
    print("\n=== PUMP ANALYSIS ===")
    token_stats = df.groupby('token_mint').agg({
        'sol_amount': ['sum', 'count', 'mean'],
        'tx_type': lambda x: (x == 'buy').mean()  # Buy ratio
    }).round(3)
    
    token_stats.columns = ['total_volume', 'num_trades', 'avg_trade_size', 'buy_ratio']
    token_stats = token_stats.sort_values('total_volume', ascending=False)
    
    print("\nTop 10 Tokens by Volume:")
    print(token_stats.head(10))
    
    # Time-based analysis
    print("\n=== TIME PATTERNS ===")
    df['minute'] = pd.to_datetime(df['block_time']).dt.floor('1min')
    volume_by_minute = df.groupby('minute').agg({
        'sol_amount': 'sum',
        'token_mint': 'nunique',
        'tx_type': 'count'
    }).rename(columns={
        'sol_amount': 'volume',
        'token_mint': 'unique_tokens',
        'tx_type': 'num_trades'
    })
    
    print("\nHigh Activity Minutes (top 5):")
    print(volume_by_minute.sort_values('volume', ascending=False).head())
    
    # Buy pressure analysis
    print("\n=== BUY PRESSURE ANALYSIS ===")
    buy_pressure = df[df['tx_type'] == 'buy'].groupby('token_mint')['sol_amount'].sum()
    sell_pressure = df[df['tx_type'] == 'sell'].groupby('token_mint')['sol_amount'].sum()
    pressure_ratio = (buy_pressure / (buy_pressure + sell_pressure)).fillna(0)
    
    print("\nTokens with Highest Buy Pressure (top 5):")
    print(pressure_ratio.sort_values(ascending=False).head())
    
    # Add pump pattern detection
    print("\n=== PUMP PATTERN DETECTION ===")
    
    def detect_pump_patterns(group):
        if len(group) < 10:  # Minimum trades to analyze
            return pd.Series({'pump_score': 0, 'volume_acceleration': 0})
            
        # Calculate metrics
        volume_acceleration = group['sol_amount'].pct_change().mean()
        buy_pressure = (group['tx_type'] == 'buy').rolling(5).mean()
        price_impact = group['sol_amount'].rolling(5).mean()
        
        # Combined pump score
        pump_score = (
            volume_acceleration * 0.4 +  # Volume growth
            buy_pressure.mean() * 0.4 +  # Sustained buy pressure
            (price_impact.max() / price_impact.mean()) * 0.2  # Price impact
        )
        
        return pd.Series({
            'pump_score': pump_score,
            'volume_acceleration': volume_acceleration
        })
    
    # Calculate pump metrics for each token
    pump_metrics = df.sort_values('block_time').groupby('token_mint').apply(detect_pump_patterns)
    
    print("\nTop Potential Pump Tokens:")
    print(pump_metrics.sort_values('pump_score', ascending=False).head(10))
    
    # Analyze trade size distribution
    print("\n=== TRADE SIZE DISTRIBUTION ===")
    percentiles = df['sol_amount'].describe(percentiles=[.25, .5, .75, .9, .95, .99])
    print("\nTrade Size Percentiles:")
    print(percentiles)
    
    return token_stats, volume_by_minute, pressure_ratio, pump_metrics

def analyze_token_patterns(df: pd.DataFrame):
    # Group tokens by volume tiers
    token_volumes = df.groupby('token_mint')['sol_amount'].sum().sort_values(ascending=False)
    
    # Create volume tiers
    volume_tiers = {
        'High (>20k SOL)': token_volumes[token_volumes > 20000],
        'Medium (5k-20k SOL)': token_volumes[(token_volumes > 5000) & (token_volumes <= 20000)],
        'Low (1k-5k SOL)': token_volumes[(token_volumes > 1000) & (token_volumes <= 5000)],
        'Micro (<1k SOL)': token_volumes[token_volumes <= 1000]
    }
    
    print("\n=== VOLUME TIER ANALYSIS ===")
    for tier, tokens in volume_tiers.items():
        print(f"\n{tier}:")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Total volume: {tokens.sum():,.2f} SOL")
        if len(tokens) > 0:
            print("Sample tokens:")
            print(tokens.head(3))
    
    # Analyze time patterns for top tokens
    print("\n=== TOP TOKEN TIME PATTERNS ===")
    top_tokens = token_volumes.head(10).index
    
    for token in top_tokens:
        token_data = df[df['token_mint'] == token].copy()
        token_data['minute'] = pd.to_datetime(token_data['block_time']).dt.floor('1min')
        
        # Get volume pattern
        volume_pattern = token_data.groupby('minute')['sol_amount'].sum()
        
        # Calculate key metrics
        total_duration = (token_data['block_time'].max() - token_data['block_time'].min()).total_seconds() / 60
        active_minutes = len(volume_pattern)
        volume_concentration = volume_pattern.max() / volume_pattern.sum()
        
        print(f"\nToken: {token}")
        print(f"Total Volume: {token_volumes[token]:,.2f} SOL")
        print(f"Duration: {total_duration:.1f} minutes")
        print(f"Active Minutes: {active_minutes}")
        print(f"Volume Concentration: {volume_concentration:.1%}")
        print(f"Peak Minute Volume: {volume_pattern.max():,.2f} SOL")
    
    return volume_tiers

if __name__ == "__main__":
    # Get the correct path regardless of where the script is run from
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'trades_20250102_205818.csv')
    
    # First, read the CSV without any type conversion
    data = pd.read_csv(data_path)
    
    # Clean the data
    # Convert sol_amount to float, skipping the header row if it exists
    if 'sol_amount' in data.columns[2]:  # Check if we have headers
        data = pd.read_csv(data_path, skiprows=1)
    
    # Now convert types
    data.columns = ['block_time', 'tx_type', 'sol_amount', 'token_mint', 'signature']
    data['sol_amount'] = pd.to_numeric(data['sol_amount'])
    data['block_time'] = pd.to_datetime(data['block_time'])
    
    token_stats, volume_by_minute, pressure_ratio, pump_metrics = analyze_pump_patterns(data) 
    
    # Add this to your main analysis
    volume_tiers = analyze_token_patterns(data) 