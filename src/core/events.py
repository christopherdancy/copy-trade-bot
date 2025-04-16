from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

@dataclass
class SignalEvent:
    timestamp: datetime
    mint: str
    price: Decimal
    volume: Decimal
    trigger_move: Decimal
    confirmation_move: Decimal
    signal_type: str  # "ENTRY" or "EXIT"

@dataclass
class EntryEvent:
    timestamp: datetime
    mint: str
    price: Decimal
    size: Decimal
    token_amount: Decimal
    tx_fees: Decimal
    reason: str
    signal_reference: SignalEvent

@dataclass
class ExitEvent:
    timestamp: datetime
    mint: str
    price: Decimal
    size: Decimal
    token_amount: Decimal
    reason: str
    pnl: Decimal
    hold_time_minutes: float
    entry_reference: EntryEvent 