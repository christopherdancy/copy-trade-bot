from dataclasses import dataclass
from typing import Dict, Any
import yaml
import os

@dataclass
class StrategyConfig:
    min_volume_threshold: float = 0.01
    max_price_threshold: float = 0.00001
    lookback_periods: int = 1

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_positions: int = 100                # Maximum number of concurrent positions
    max_position_size: float = 0.2        # Maximum size for any single position
    max_hold_time_minutes: int = 45       # Maximum time to hold a position
    stop_loss_pct: float = 0.15          # Stop loss percentage
    max_daily_loss_pct: float = 0.1      # Maximum drawdown allowed
    min_liquidity_ratio: float = 2.0     # Minimum liquidity required for entry
    position_sizing_method: str = 'fixed' # How to calculate position sizes

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.strategy = StrategyConfig()
        self.risk = RiskParameters()
        
        if os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        if 'strategy' in config_data:
            self.strategy = StrategyConfig(**config_data['strategy'])
        if 'risk' in config_data:
            self.risk = RiskParameters(**config_data['risk']) 