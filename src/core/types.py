from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from solders.pubkey import Pubkey
from construct import Struct, Int64ul, Flag

@dataclass
class MarketData:
    mint: str
    price: Decimal
    virtual_price: Decimal
    real_price: Decimal
    is_buy: bool
    timestamp: datetime 

@dataclass
class PumpToken:
    mint: Pubkey
    bonding_curve: Pubkey
    associated_bonding_curve: Pubkey
    
@dataclass
class TradeResult:
    success: bool
    signature: str = None
    error: str = None

@dataclass
class BondingCurveAccount:
    virtual_token_reserves: int
    virtual_sol_reserves: int
    real_token_reserves: int
    real_sol_reserves: int
    token_total_supply: int
    complete: bool

    @classmethod
    def from_buffer(cls, data: bytes):
        # Skip 8-byte discriminator
        CURVE_STRUCT = Struct(
            "virtual_token_reserves" / Int64ul,
            "virtual_sol_reserves" / Int64ul,
            "real_token_reserves" / Int64ul,
            "real_sol_reserves" / Int64ul,
            "token_total_supply" / Int64ul,
            "complete" / Flag
        )
        parsed = CURVE_STRUCT.parse(data[8:])
        return cls(**parsed)