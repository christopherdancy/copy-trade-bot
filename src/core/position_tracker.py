from dataclasses import dataclass
import asyncio
from datetime import datetime
from typing import Dict, Optional, List, Any
import pandas as pd
import os
from data.pump_data_feed_enhanced import TradeEvent
@dataclass
class Position:
    """Represents an active trading position"""
    mint: str
    entry_price: float
    token_amount: float
    entry_time: datetime
    entry_blocktime: int
    sol_amount: float
    tx_sig: str
    entry_fees: float
    wallet_followed: str
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        return (current_price - self.entry_price) * self.token_amount

@dataclass
class PendingEntry:
    """Represents a pending entry transaction"""
    mint: str
    wallet_followed: str
    created_at: datetime
    tx_sig: Optional[str] = None

@dataclass
class PendingExit:
    """Represents a pending exit transaction"""
    mint: str
    wallet_followed: str
    created_at: datetime

class PositionTracker:
    def __init__(self, csv_path: str, logger: Any):
        """Initialize the position tracker with a CSV file path"""
        # Create timestamped filename for the CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract base directory from csv_path
        base_dir = os.path.dirname(csv_path)
        
        # Get filename without extension
        filename = os.path.basename(csv_path)
        name, ext = os.path.splitext(filename)
        
        # Create new filename with timestamp
        self.csv_path = os.path.join(base_dir, f"{name}_{timestamp}{ext}")
        
        self.logger = logger
        
        # Core state tracking
        self.positions = {}
        self.pending_entries = {}
        self.pending_exits = {}  # mint -> tx_sig
        
        # Transaction tracking
        self.pending_tx_by_signature = {}  # signature -> details
        
        # Locks for thread safety
        self.position_lock = asyncio.Lock()
        
        # Initialize CSV file with headers
        self._initialize_csv()
        
    def _initialize_csv(self):
        """Create CSV with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            columns = [
                'wallet_followed', 'mint', 'entry_time', 'entry_price', 'entry_token_amount', 
                'entry_sol_amount', 'entry_fees', 'exit_time', 'exit_price', 'exit_token_amount', 
                'exit_sol_amount', 'exit_fees', 'hold_time',
                'gross_pnl', 'net_pnl', 'gross_roi', 'net_roi',
                'entry_tx_sig', 'exit_tx_sig'
            ]
            pd.DataFrame(columns=columns).to_csv(self.csv_path, index=False)
    
    # Position management methods
    
    async def has_position(self, mint: str) -> bool:
        """Check if we have an active position for this token"""
        async with self.position_lock:
            return mint in self.positions
    
    async def get_position(self, mint: str) -> Optional[Position]:
        """Get position details if exists"""
        async with self.position_lock:
            return self.positions.get(mint)
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all active positions"""
        async with self.position_lock:
            return self.positions.copy()
    
    # Pending entry management
    async def has_pending_entry(self, mint: str) -> bool:
        """Check if we have a pending entry for this token"""
        async with self.position_lock:
            return mint in self.pending_entries
    
    async def add_pending_entry(self, mint: str, wallet_followed: str) -> None:
        """Mark a token as having a pending entry"""
        async with self.position_lock:
            self.pending_entries[mint] = PendingEntry(
                mint=mint,
                wallet_followed=wallet_followed,
                created_at=datetime.now()
            )
            self.logger.info(f"Added pending entry for {mint} from wallet {wallet_followed}")

    async def get_pending_entries(self) -> Dict[str, PendingEntry]:
        """Get all pending entries"""
        async with self.position_lock:
            return self.pending_entries.copy()

    # Pending exit management
    async def has_pending_exit(self, mint: str) -> bool:
        """Check if we have a pending exit for this token"""
        async with self.position_lock:
            return mint in self.pending_exits

    async def add_pending_exit(self, mint: str, wallet_followed: str) -> None:
        """Add a pending exit transaction"""
        async with self.position_lock:
            self.pending_exits[mint] = PendingExit(
                mint=mint,
                wallet_followed=wallet_followed,
                created_at=datetime.now()
            )
            self.logger.info(f"Added pending exit for {mint}")
    
    # Position creation and exit
    async def confirm_entry(self, trade: TradeEvent) -> None:
        """Confirm an entry from transaction details"""
        async with self.position_lock:
            # Get wallet from pending entry if exists
            wallet_followed = None
            if trade.mint in self.pending_entries:
                wallet_followed = self.pending_entries[trade.mint].wallet_followed
                # Clear the pending entry
                del self.pending_entries[trade.mint]
            
            # Create position
            self.positions[trade.mint] = Position(
                mint=trade.mint,
                entry_price=float(trade.price),
                sol_amount=float(trade.sol_amount),
                token_amount=float(trade.token_amount),
                entry_time=datetime.fromtimestamp(trade.blocktime),
                entry_blocktime=trade.blocktime,
                entry_fees=float(trade.total_sol_change) - float(trade.sol_amount),
                tx_sig=trade.signature,
                wallet_followed=wallet_followed or '',
            )
            
            self.logger.info(f"Confirmed position for {trade.mint} at {trade.price}")
    
    async def confirm_exit(self, trade: TradeEvent) -> None:
        """Confirm an exit and record trade completion"""
        async with self.position_lock:
            if trade.mint not in self.positions:
                self.logger.warning(f"Tried to confirm exit for non-existent position: {trade.mint}")
                return
            
            # Get current position
            position = self.positions[trade.mint]
            
            # Calculate metrics
            exit_price = float(trade.price)
            exit_time = datetime.fromtimestamp(trade.blocktime)
            hold_time = (exit_time - position.entry_time).total_seconds() / 60.0
            
            gross_pnl = (exit_price - position.entry_price) * position.token_amount
            gross_roi = (gross_pnl / position.sol_amount) * 100 if position.sol_amount != 0 else 0
            
            # Calculate net metrics including fees
            total_entry_cost = position.sol_amount + position.entry_fees
            
            exit_sol_amount = float(trade.sol_amount)
            exit_fees = float(trade.sol_amount) - float(trade.total_sol_change)
            total_exit_value = float(trade.total_sol_change)
            
            net_pnl = total_exit_value - total_entry_cost
            net_roi = (net_pnl / total_entry_cost) * 100 if total_entry_cost != 0 else 0
            
            # Record to CSV
            trade_data = {
                'wallet_followed': position.wallet_followed,
                'mint': position.mint,
                'entry_time': position.entry_time,
                'entry_price': position.entry_price,
                'entry_token_amount': position.token_amount,
                'entry_sol_amount': position.sol_amount,
                'entry_fees': position.entry_fees,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'exit_token_amount': float(trade.token_amount),
                'exit_sol_amount': exit_sol_amount,
                'exit_fees': exit_fees,
                'hold_time': hold_time,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'gross_roi': gross_roi,
                'net_roi': net_roi,
                'entry_tx_sig': position.tx_sig,
                'exit_tx_sig': trade.signature
            }
            
            # Append to CSV
            df = pd.DataFrame([trade_data])
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
            
            # Clear from pending exits if present
            if trade.mint in self.pending_exits:
                del self.pending_exits[trade.mint]
            
            # Remove position
            del self.positions[trade.mint]
            
            self.logger.info(f"Confirmed exit for {trade.mint} at {exit_price}, PnL: {gross_pnl}, ROI: {gross_roi:.2f}%")
    
    # Transaction tracking
    
    async def get_transaction_type(self, tx_sig: str) -> Optional[str]:
        """Get the type of a pending transaction by signature"""
        async with self.position_lock:
            if tx_sig in self.pending_tx_by_signature:
                return self.pending_tx_by_signature[tx_sig].get('type')
            return None
    
    async def get_transaction_details(self, tx_sig: str) -> Optional[Dict]:
        """Get details of a pending transaction by signature"""
        async with self.position_lock:
            return self.pending_tx_by_signature.get(tx_sig)
    
    async def process_transaction_confirmation(self, tx_sig: str, tx_details: Dict) -> None:
        """Process a transaction confirmation from WebSocket"""
        async with self.position_lock:
            if tx_sig not in self.pending_tx_by_signature:
                self.logger.warning(f"Received confirmation for unknown transaction: {tx_sig}")
                return
            
            tx_info = self.pending_tx_by_signature[tx_sig]
            mint = tx_info.get('mint')
            tx_type = tx_info.get('type')
            
            if tx_type == 'entry':
                await self.confirm_entry(mint, tx_details)
            elif tx_type == 'exit':
                await self.confirm_exit(mint, tx_details)
    
    # Timeout handling
    
    async def clear_stale_transactions(self, timeout_minutes: float = 5.0) -> None:
        """Clear pending transactions that haven't confirmed within timeout
        
        Args:
            timeout_minutes: Number of minutes to consider a transaction stale (can be fractional)
        """
        async with self.position_lock:
            now = datetime.now()
            timeout_seconds = timeout_minutes * 60
            stale_entries = []
            stale_exits = []
            
            # Check pending entries
            for mint, entry in self.pending_entries.items():
                elapsed_seconds = (now - entry.created_at).total_seconds()
                if elapsed_seconds > timeout_seconds:
                    stale_entries.append(mint)
                    self.logger.warning(f"Stale pending entry for {mint} after {elapsed_seconds:.1f} seconds")
            
            # Check pending exits
            for mint, exit_info in self.pending_exits.items():
                elapsed_seconds = (now - exit_info.created_at).total_seconds()
                if elapsed_seconds > timeout_seconds:
                    stale_exits.append(mint)
                    self.logger.warning(f"Stale pending exit for {mint} after {elapsed_seconds:.1f} seconds")
            
            # Clear stale entries and exits
            for mint in stale_entries:
                del self.pending_entries[mint]
            
            for mint in stale_exits:
                del self.pending_exits[mint]