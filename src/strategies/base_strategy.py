from typing import Dict
import pandas as pd

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Generate entry signals for tokens"""
        raise NotImplementedError
        
    def check_exits(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Generate exit signals for current positions"""
        raise NotImplementedError 