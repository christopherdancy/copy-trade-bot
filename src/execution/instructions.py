from dataclasses import dataclass
from time import time
from typing import Optional, Dict, Tuple, Set
import asyncio
import struct
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solana.transaction import Transaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts, TokenAccountOpts
from spl.token.instructions import get_associated_token_address, create_associated_token_account, close_account, CloseAccountParams
from spl.token.constants import TOKEN_PROGRAM_ID
from .constants import PUMP_PROGRAM, PUMP_GLOBAL, PUMP_FEE, PUMP_EVENT_AUTHORITY, SYSTEM_PROGRAM, SYSTEM_TOKEN_PROGRAM, SYSTEM_RENT, ASSOCIATED_TOKEN_PROGRAM_ID, COMPUTE_BUDGET_ID


class Instructions:
    def __init__(self, wallet: Keypair, client: AsyncClient):
        self.wallet = wallet
        self.client = client
        self.created_atas = set()  # Track ATAs we've created or verified

    async def get_or_create_ata(self, mint: Pubkey) -> Tuple[Pubkey, Optional[Instruction]]:
        """Get or create user's associated token account using in-memory tracking"""
        mint_str = str(mint)
        ata = get_associated_token_address(self.wallet.pubkey(), mint)
        
        # Check our in-memory tracking first
        if mint_str in self.created_atas:
            # We've already created or verified this ATA
            return ata, None
        
        # If not in our tracking, assume we need to create it
        # We'll add it to our tracking regardless of whether creation succeeds
        self.created_atas.add(mint_str)
        
        create_ata_ix = create_associated_token_account(
            payer=self.wallet.pubkey(),
            owner=self.wallet.pubkey(),
            mint=mint
        )
        return ata, create_ata_ix

    async def get_and_close_ata(self, mint: Pubkey) -> Tuple[Pubkey, Instruction]:
        """Get associated token account and create instruction to close it
        
        This method is specifically for selling tokens where we want to
        close the ATA after selling all tokens to reclaim rent.
        
        Args:
            mint: The token mint address
            
        Returns:
            Tuple of (token_account, close_instruction)
        """
        mint_str = str(mint)
        ata = get_associated_token_address(self.wallet.pubkey(), mint)
        
        # Create close instruction using close_account with CloseAccountParams
        params = CloseAccountParams(
            account=ata,
            dest=self.wallet.pubkey(),
            owner=self.wallet.pubkey(),
            program_id=TOKEN_PROGRAM_ID
        )
        
        close_ix = close_account(params)
        
        # Remove from our tracking if it's there
        if mint_str in self.created_atas:
            self.created_atas.remove(mint_str)
            
        return ata, close_ix

    def create_compute_budget_instructions(
        self,
        priority_fee: int = 350_000,
        compute_unit_limit: int = 150_000
    ) -> Tuple[Instruction, Instruction]:
        """Create compute budget instructions for transaction priority
        
        Args:
            priority_fee: Priority fee in microlamports
            compute_unit_limit: Compute unit limit (default 200,000)
            
        Returns:
            Tuple of (compute_limit_ix, compute_price_ix)
        """
        
        # 1. Set Compute Unit Limit (instruction ID: 2)
        compute_limit_ix = Instruction(
            program_id=COMPUTE_BUDGET_ID,
            data=bytes([2]) + compute_unit_limit.to_bytes(4, "little"),
            accounts=[]
        )
        
        # 2. Set Compute Unit Price (instruction ID: 3)
        compute_price_ix = Instruction(
            program_id=COMPUTE_BUDGET_ID,
            data=bytes([3]) + priority_fee.to_bytes(8, "little"),
            accounts=[]
        )
        
        return compute_limit_ix, compute_price_ix

    def create_buy_instruction(
        self,
        mint: Pubkey,
        bonding_curve: Pubkey,
        associated_bonding_curve: Pubkey,
        user_ata: Pubkey,
        token_amount: int,
        max_sol_amount: int
    ) -> Instruction:
        """Create the buy instruction with token amount and max SOL"""
        accounts = [
            AccountMeta(pubkey=PUMP_GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=user_ata, is_signer=False, is_writable=True),
            AccountMeta(pubkey=self.wallet.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSTEM_TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSTEM_RENT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_PROGRAM, is_signer=False, is_writable=False),
        ]

        discriminator = struct.pack("<Q", 16927863322537952870)
        data = discriminator + struct.pack("<Q", token_amount) + struct.pack("<Q", max_sol_amount)
        
        return Instruction(PUMP_PROGRAM, data, accounts)

    def create_sell_instruction(
        self,
        mint: Pubkey,
        bonding_curve: Pubkey,
        associated_bonding_curve: Pubkey,
        user_ata: Pubkey,
        token_amount: int,
        min_sol_output: int
    ) -> Instruction:
        """Create the sell instruction according to the Pump IDL"""
        accounts = [
            AccountMeta(pubkey=PUMP_GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=user_ata, is_signer=False, is_writable=True),
            AccountMeta(pubkey=self.wallet.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=ASSOCIATED_TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSTEM_TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_PROGRAM, is_signer=False, is_writable=False),
        ]

        # Use the correct discriminator
        discriminator = bytes.fromhex("33e685a4017f83ad")
        # Pack instruction data
        data = discriminator + struct.pack("<Q", token_amount) + struct.pack("<Q", min_sol_output)
        return Instruction(PUMP_PROGRAM, data, accounts)