from dataclasses import dataclass
from time import time
from typing import Optional, Dict, Tuple
import asyncio
import struct
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solana.transaction import Transaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from spl.token.instructions import get_associated_token_address, create_associated_token_account
from construct import Struct, Int64ul, Flag
from .constants import PUMP_PROGRAM
from logging import Logger

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
        """Parse account data into BondingCurveAccount"""
        try:
            # Skip 8-byte discriminator
            account_data = data[8:]
            
            # Define the struct format
            ACCOUNT_STRUCT = Struct(
                "virtual_token_reserves" / Int64ul,
                "virtual_sol_reserves" / Int64ul,
                "real_token_reserves" / Int64ul,
                "real_sol_reserves" / Int64ul,
                "token_total_supply" / Int64ul,
                "complete" / Flag
            )
            
            # Parse the data
            parsed = ACCOUNT_STRUCT.parse(account_data)
            
            # Convert to dictionary and create instance
            return cls(
                virtual_token_reserves=parsed.virtual_token_reserves,
                virtual_sol_reserves=parsed.virtual_sol_reserves,
                real_token_reserves=parsed.real_token_reserves,
                real_sol_reserves=parsed.real_sol_reserves,
                token_total_supply=parsed.token_total_supply,
                complete=parsed.complete
            )
            
        except Exception as e:
            print(f"Error parsing bonding curve account: {e}")
            print(f"Raw data length: {len(data)}")
            print(f"Raw data (hex): {data.hex()}")
            raise

@dataclass
class CachedBondingCurve:
    data: BondingCurveAccount
    timestamp: float

# TODO: Try / catch errors in fetch_bonding_curve
class BondingCurveExecutor:
    def __init__(self, wallet: Keypair, client: AsyncClient, logger: Logger):
        self.wallet = wallet
        self.client = client
        self.cache: Dict[str, CachedBondingCurve] = {}
        self.last_fetch_time: float = 0
        self.CACHE_DURATION = 600  # 300 seconds
        self.FETCH_COOLDOWN = 2   # 2 seconds
        self.logger = logger
        
    def get_bonding_curve_pda(self, mint: Pubkey, program_id: Pubkey) -> Pubkey:
        """Derive the bonding curve PDA for a given mint"""
        try:
            # Create seeds exactly like JavaScript
            BONDING_CURVE_SEED = b"bonding-curve"  # Already bytes
            seeds = [
                BONDING_CURVE_SEED,
                bytes(mint)
            ]
            
            # Get PDA (first address)
            pda, _ = Pubkey.find_program_address(seeds, program_id)
            return pda
            
        except Exception as e:
            print(f"Error deriving PDA: {e}")
            raise
        
    def get_associated_bonding_curve(
        self,
        mint: Pubkey,
        bonding_curve_pda: Pubkey
    ) -> Pubkey:
        """Get associated token account for bonding curve"""
        return get_associated_token_address(
            bonding_curve_pda,
            mint
        )

    async def fetch_bonding_curve(self, mint: Pubkey) -> Optional[BondingCurveAccount]:
        """Fetch bonding curve data with caching and rate limiting"""
        try:
            mint_str = str(mint)
            now = time()
            
            # Check cache
            if mint_str in self.cache:
                cached = self.cache[mint_str]
                if now - cached.timestamp < self.CACHE_DURATION:
                    return cached.data
            
            # Rate limiting
            time_since_last = now - self.last_fetch_time
            if time_since_last < self.FETCH_COOLDOWN:
                await asyncio.sleep(self.FETCH_COOLDOWN - time_since_last)
            
            self.last_fetch_time = now
            
            # Get bonding curve PDA
            bonding_curve_pda = self.get_bonding_curve_pda(mint, PUMP_PROGRAM)
            self.logger.debug(f"Derived bonding curve PDA: {bonding_curve_pda} for mint {mint}")
            
            # Fetch with retries
            retries = 3
            delay = 1.0
            
            while retries > 0:
                try:
                    account_info = await self.client.get_account_info(bonding_curve_pda)
                    
                    if not account_info.value:
                        self.logger.warning(f"Bonding curve account not found for mint {mint}")
                        retries -= 1
                        if retries > 0:
                            self.logger.debug(f"Retrying fetch ({retries} attempts left)...")
                            await asyncio.sleep(delay)
                            delay *= 2
                        continue
                    
                    if not account_info.value.data:
                        self.logger.warning(f"Bonding curve account has no data for mint {mint}")
                        retries -= 1
                        if retries > 0:
                            await asyncio.sleep(delay)
                            delay *= 2
                        continue
                    
                    try:
                        account = BondingCurveAccount.from_buffer(account_info.value.data)
                        
                        # Check if the bonding curve is complete
                        if account.complete:
                            self.logger.warning(f"Bonding curve is marked as complete for mint {mint}")
                            return None
                        
                        # Cache the result
                        # self.cache[mint_str] = CachedBondingCurve(
                        #     data=account,
                        #     timestamp=now
                        # )
                        
                        self.logger.debug(f"Successfully fetched bonding curve for {mint}: " 
                                         f"virtual_token={account.virtual_token_reserves}, "
                                         f"virtual_sol={account.virtual_sol_reserves}, "
                                         f"real_token={account.real_token_reserves}, "
                                         f"real_sol={account.real_sol_reserves}, "
                                         f"complete={account.complete}")
                        
                        return account
                        
                    except Exception as parse_error:
                        self.logger.error(f"Error parsing bonding curve data for mint {mint}: {str(parse_error)}")
                        retries -= 1
                        if retries > 0:
                            await asyncio.sleep(delay)
                            delay *= 2
                            
                except Exception as e:
                    if "429" in str(e):  # Rate limit error
                        self.logger.warning(f"Rate limit hit when fetching bonding curve for {mint}")
                        retries -= 1
                        if retries > 0:
                            await asyncio.sleep(delay)
                            delay *= 2
                    else:
                        self.logger.error(f"Error fetching bonding curve for mint {mint}: {str(e)}")
                        raise
            
            self.logger.warning(f"Failed to fetch valid bonding curve for mint {mint} after {3-retries} attempts")
            return None
            
        except Exception as e:
            self.logger.error(f"Unexpected error fetching bonding curve for mint {mint}: {str(e)}")
            return None

    def calculate_buy_price(self, sol_amount: int, curve: BondingCurveAccount, slippage_basis_points: int = 300) -> Tuple[int, int]:
        """Calculate token amount from SOL input with slippage
        
        Args:
            sol_amount: Amount of SOL in lamports
            curve: Bonding curve account data
            slippage_basis_points: Slippage tolerance in basis points (1bp = 0.01%)
            
        Returns:
            Tuple of (expected_token_amount, minimum_token_amount)
        """
        virtual_sol = curve.virtual_sol_reserves
        virtual_token = curve.virtual_token_reserves
        
        k = virtual_sol * virtual_token
        new_virtual_sol = virtual_sol + sol_amount
        new_virtual_token = k // new_virtual_sol
        
        expected_token_amount = virtual_token - new_virtual_token
        
        # Apply slippage to minimum tokens received
        slippage_adjustment = (expected_token_amount * slippage_basis_points) // 10000
        minimum_token_amount = expected_token_amount - slippage_adjustment
        
        return expected_token_amount, minimum_token_amount

    # TODO: Play with slippage
    def calculate_sell_price(self, token_amount: int, curve: BondingCurveAccount, slippage_basis_points: int = 300) -> Tuple[int, int]:
        """Calculate SOL output for sell with slippage
        
        Args:
            token_amount: Amount of tokens to sell
            curve: Bonding curve account data
            slippage_basis_points: Slippage tolerance in basis points (1bp = 0.01%)
            
        Returns:
            Tuple of (expected_sol_amount, minimum_sol_amount)
        """
        virtual_sol = curve.virtual_sol_reserves
        virtual_token = curve.virtual_token_reserves
        
        k = virtual_sol * virtual_token
        new_virtual_token = virtual_token + token_amount
        new_virtual_sol = k // new_virtual_token
        
        expected_sol_amount = virtual_sol - new_virtual_sol
        
        # Apply slippage to minimum SOL received
        slippage_adjustment = (expected_sol_amount * slippage_basis_points) // 10000
        minimum_sol_amount = expected_sol_amount - slippage_adjustment
        
        return expected_sol_amount, minimum_sol_amount