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
from solana.rpc.commitment import Confirmed, Processed
from solana.rpc.types import TxOpts, TokenAccountOpts
from spl.token.instructions import get_associated_token_address, create_associated_token_account
from construct import Struct, Int64ul, Flag
from decimal import Decimal
from spl.token.constants import TOKEN_PROGRAM_ID
from solders.signature import Signature
from .instructions import Instructions
from .bonding_curve import BondingCurveExecutor
from .constants import PUMP_PROGRAM, PUMP_FEE
import aiohttp  
import statistics
from logging import Logger

class TransactionError(Exception):
    """Raised when transaction-related operations fail"""
    pass

class NetworkError(Exception):
    """Raised when network-related operations fail"""
    pass

class InsufficientFundsError(Exception):
    """Raised when wallet lacks sufficient funds"""
    pass

class LiveRunExecutor:
    def __init__(self, wallet: Keypair, rpc_url: str, logger: Logger):
        self.wallet = wallet
        self.client = AsyncClient(rpc_url)
        self.instructions = Instructions(wallet, self.client)
        self.bonding_curve_executor = BondingCurveExecutor(wallet, self.client, logger)
        self.logger = logger
        self.compute_limit_ix, self.compute_price_ix = self.instructions.create_compute_budget_instructions()

    # TODO: Need to reapply slippage tolerance to ensure execution
    async def buy_token(
        self, 
        mint: Pubkey, 
        amount_sol: float,
        price: float,
        slippage_percent: float = 10.0
    ):
        """Execute a buy transaction with derived accounts - pure fire and forget approach"""
        try:
            self.logger.info(f"Building buy transaction")
            
            # Get PDAs and associated accounts
            bonding_curve_pda = self.bonding_curve_executor.get_bonding_curve_pda(mint, PUMP_PROGRAM)
            associated_bonding_curve = self.bonding_curve_executor.get_associated_bonding_curve(
                mint, bonding_curve_pda
            )

            # Create transaction and set fee payer
            transaction = Transaction()
            transaction.fee_payer = self.wallet.pubkey()
            
            # Create compute budget instructions manually
            transaction.add(self.compute_limit_ix)
            transaction.add(self.compute_price_ix)
            
            # Get or create user's token account
            user_ata, user_ata_ix = await self.instructions.get_or_create_ata(mint)
            if user_ata_ix:
                transaction.add(user_ata_ix)
            
            # Calculate token amount directly from the observed price
            amount_sol_lamports = int(amount_sol * 1_000_000_000)
            
            if price <= 0:
                raise ValueError("Invalid price: must be greater than zero")
                
            # Calculate expected token amount based on price
            expected_token_amount = int((amount_sol / price) * 1_000_000)  # Convert to token decimals (6)
            
            # Apply slippage tolerance - using percentage for clarity
            slippage_factor = 1 - (slippage_percent / 100)
            minimum_token_amount = int(expected_token_amount * slippage_factor)
            
            buy_ix = self.instructions.create_buy_instruction(
                mint=mint,
                bonding_curve=bonding_curve_pda,
                associated_bonding_curve=associated_bonding_curve,
                user_ata=user_ata,
                token_amount=minimum_token_amount,
                max_sol_amount=amount_sol_lamports
            )
            transaction.add(buy_ix)

            # Submit transaction in background task - don't even wait for signature
            asyncio.create_task(self._submit_transaction(transaction, mint, "buy"))
            
            # Return immediately - we'll get all updates through WebSocket
            return True
            
        except ValueError as ve:
            self.logger.error(f"Error building buy transaction: {str(ve)}")
            return False
        except Exception as e:
            self.logger.error(f"Error in buy transaction creation: {str(e)}")
            return False
            
    async def sell_token(
        self,
        mint: Pubkey,
        token_amount: float,
        slippage_percent: float = 100,
        close_account: bool = True
    ):
        """Execute a sell transaction - pure fire and forget approach"""
        try:
            self.logger.info(f"Building sell transaction")
            
            # Get PDAs and associated accounts
            bonding_curve_pda = self.bonding_curve_executor.get_bonding_curve_pda(mint, PUMP_PROGRAM)
            associated_bonding_curve = self.bonding_curve_executor.get_associated_bonding_curve(
                mint, bonding_curve_pda
            )
            
            # Create transaction
            transaction = Transaction()
            transaction.fee_payer = self.wallet.pubkey()

            # Add compute budget instructions
            transaction.add(self.compute_limit_ix)
            transaction.add(self.compute_price_ix)
            
            # Get ATA and create close instruction
            user_ata, close_ix = await self.instructions.get_and_close_ata(mint)
                
            # Convert to raw amount (assuming 6 decimals)
            token_amount_decimals = int(token_amount * 1_000_000)

            # Create and add sell instruction
            sell_ix = self.instructions.create_sell_instruction(
                mint=mint,
                bonding_curve=bonding_curve_pda,
                associated_bonding_curve=associated_bonding_curve,
                user_ata=user_ata,
                token_amount=token_amount_decimals,
                min_sol_output=0
            )
            transaction.add(sell_ix)
            
            # Add close instruction if requested
            if close_account:
                transaction.add(close_ix)
            
            # Submit transaction in background task - don't even wait for signature
            asyncio.create_task(self._submit_transaction(transaction, mint, "sell"))
            
            # Return immediately - we'll get all updates through WebSocket
            return True
            
        except Exception as e:
            self.logger.error(f"Error in sell_token: {str(e)}")
            return False

    async def _submit_transaction(self, transaction, mint, tx_type):
        """Background task for transaction submission"""
        try:
            self.logger.info(f"Submitting {tx_type} transaction for {mint}")
            tx = await self.client.send_transaction(
                transaction,
                self.wallet,
                opts=TxOpts(
                    skip_preflight=True,
                    preflight_commitment=Processed,
                    max_retries=0
                )
            )
            self.logger.info(f"{tx_type.capitalize()} transaction for {mint} sent: {tx.value}")
        except Exception as e:
            self.logger.error(f"Error submitting {tx_type} transaction for {mint}: {str(e)}")

    # TODO: Scalping Strategy Optimization Path
    # 
    # Future improvements to consider:
    # 1. "Assume and Verify Later" approach:
    #    - Track transactions asynchronously without waiting for confirmation
    #    - Periodically reconcile actual vs expected balances
    #    - Make decisions based on estimated outcomes, corrected as real data arrives
    #
    # 2. Dynamic SOL allocation based on bonding curve liquidity:
    #    - Calculate optimal SOL amount to minimize slippage
    #    - Scale down for less liquid tokens to avoid excessive price impact
    #
    # 3. Implement a transaction tracker service:
    #    - Record all transaction signatures
    #    - Use background tasks to eventually collect all transaction details
    #    - Build models to improve parameter selection based on historical performance
    #
    # 4. Balance-based verification:
    #    - Before selling, verify token ownership
    #    - Skip transactions if prerequisites aren't met
    #
    # These changes would prioritize throughput and opportunity capture 
    # while still maintaining data for analysis and strategy improvement.
    async def send_and_confirm_transaction(self, transaction: Transaction, isBuy: bool = True, isTest: bool = True) -> Optional[dict]:
        """
        Send a transaction with a "try once" approach - optimized for speed over guaranteed confirmation
        
        Args:
            transaction: The transaction to send
            isBuy: Whether this is a buy transaction (for testing)
            isTest: Whether to use test signatures
            
        Returns:
            Optional[dict]: Transaction details if successful, None if failed
        """
        try:
            if isTest:
                test_signatures = {
                    "buy": "2VDPC8mnFLG4Wa1Ji3cAH3ooLiCv96KmNTov4sTS2XqvMLHGKHBHUj3P9LRP3BS2Z5sf31Y1mhx3GMNVeoC9h6D2",
                    "sell": "3hVQanMpEPpQpCjwWWGT1utjr9E3HNfixQy4AWpUhTBAjTmgQQ8GHNvBMBPPBjGVjQKCpWfc8pinMVv6c9T4zsF3"
                }
                sig = test_signatures.get("buy", test_signatures["buy"]) if isBuy else test_signatures.get("sell", test_signatures["sell"])
                self.logger.info(f"Using test signature: {sig}")
                return await self.get_transaction_details(Signature.from_string(sig))
            
            # Send transaction - single attempt
            try:
                self.logger.info(f"Sending transaction")
                tx = await self.client.send_transaction(
                    transaction,
                    self.wallet,
                    opts=TxOpts(
                        skip_preflight=True,
                        preflight_commitment=Processed,  # Use Processed instead of Confirmed
                        max_retries=0  # No retries at RPC level
                    )
                )
                self.logger.info(f"Transaction sent successfully: {tx.value}")
            except Exception as e:
                raise TransactionError(f"Failed to send transaction: {str(e)}") from e
            
            # Quick confirmation with timeout
            try:
                # Wait for transaction to be processed with timeout
                self.logger.info(f"Waiting for transaction to be processed")
                await asyncio.wait_for(
                    self.client.confirm_transaction(tx.value, commitment=Processed),
                    timeout=3  # 500ms max wait - slightly generous timeout
                )
                self.logger.info(f"Transaction confirmed successfully")
                
                # Get transaction details with timeout
                try:
                    tx_details = await asyncio.wait_for(
                        self.get_transaction_details(tx.value),
                        timeout=0.5  # 500ms max wait for details
                    )
                    
                    # Verify transaction succeeded
                    if tx_details and tx_details.get("execution_price", 0) > 0:
                        self.logger.info("Successfully retrieved transaction details")
                        return tx_details
                    else:
                        self.logger.warning("Transaction details invalid or execution price is zero")
                        return None
                        
                except asyncio.TimeoutError:
                    self.logger.warning("Timed out waiting for transaction details")
                    return None
                except Exception as e:
                    self.logger.error(f"Error getting transaction details: {str(e)}")
                    return None
                    
            except asyncio.TimeoutError:
                self.logger.warning("Timed out waiting for transaction confirmation")
                return None
            except Exception as e:
                self.logger.error(f"Error confirming transaction: {str(e)}")
                return None

        except Exception as e:
            raise  # Re-raise the exception after logging

    async def get_transaction_details(self, signature: Signature) -> Optional[dict]:
        """Get transaction details from the network"""
        try:
            tx_details = await self.client.get_transaction(
                signature,
                commitment=Confirmed
            )
            
            transaction = tx_details.value.transaction
            meta = transaction.meta
            account_keys = transaction.transaction.message.account_keys
            
            # Get the blocktime (timestamp) of the transaction
            blocktime = tx_details.value.block_time
            
            # Determine transaction type from logs
            tx_type = "unknown"
            for log in meta.log_messages:
                if "Instruction: Buy" in log:
                    tx_type = "buy"
                elif "Instruction: Sell" in log:
                    tx_type = "sell"

            # Get wallet token balance changes
            our_wallet = str(self.wallet.pubkey())
            our_token_change = 0.0
            
            # First find the bonding curve token account by looking at token movements
            bonding_curve_token_account_index = None
            bonding_curve_owner_index = None

            if meta.pre_token_balances and meta.post_token_balances:
                # Find our token account first
                our_token_account_index = next(
                    (b.account_index for b in meta.pre_token_balances + meta.post_token_balances 
                     if b.owner and str(b.owner) == our_wallet),
                    None
                )
                
                if our_token_account_index is not None:
                    # Get our token balance change
                    our_pre_balance = next(
                        (float(b.ui_token_amount.ui_amount_string) 
                         for b in meta.pre_token_balances 
                         if b.account_index == our_token_account_index),
                        0.0
                    )
                    
                    our_post_balance = next(
                        (float(b.ui_token_amount.ui_amount_string)
                         for b in meta.post_token_balances 
                         if b.account_index == our_token_account_index),
                        0.0
                    )
                    
                    our_token_change = abs(our_post_balance - our_pre_balance)
                    
                    # Now find the bonding curve token account - it's the one with opposite token movement
                    for pre_balance in meta.pre_token_balances:
                        if pre_balance.account_index != our_token_account_index:
                            post_balance = next(
                                (b for b in meta.post_token_balances 
                                 if b.account_index == pre_balance.account_index),
                                None
                            )
                            
                            if post_balance:
                                pre_amount = float(pre_balance.ui_token_amount.ui_amount_string)
                                post_amount = float(post_balance.ui_token_amount.ui_amount_string)
                                
                                # In a buy, bonding curve sends tokens (decreases)
                                # In a sell, bonding curve receives tokens (increases)
                                if (tx_type == "buy" and post_amount < pre_amount) or \
                                   (tx_type == "sell" and post_amount > pre_amount):
                                    bonding_curve_token_account_index = pre_balance.account_index
                                    
                                    # Now find the owner of this token account in the account keys
                                    if pre_balance.owner:
                                        bonding_curve_owner_index = next(
                                            (i for i, key in enumerate(account_keys) 
                                             if str(key) == str(pre_balance.owner)),
                                            None
                                        )
                                    break

            # Now that we have the bonding curve owner index, we can directly track its SOL changes
            bonding_curve_sol_change = 0.0
            if bonding_curve_owner_index is not None:
                if tx_type == "buy":
                    # In a buy, bonding curve receives SOL
                    bonding_curve_sol_change = (meta.post_balances[bonding_curve_owner_index] - 
                                               meta.pre_balances[bonding_curve_owner_index]) / 1e9
                else:  # sell
                    # In a sell, bonding curve sends SOL
                    bonding_curve_sol_change = (meta.pre_balances[bonding_curve_owner_index] - 
                                               meta.post_balances[bonding_curve_owner_index]) / 1e9

            # Use this as our sol_trade_amount
            sol_trade_amount = bonding_curve_sol_change if bonding_curve_sol_change > 0 else 0

            # Get wallet SOL balance changes
            our_account_index = next(
                (i for i, key in enumerate(account_keys) 
                if str(key) == our_wallet),
                None
            )
            
            sol_change = 0.0
            if our_account_index is not None:
                if tx_type == "buy":
                    sol_change = abs(meta.pre_balances[our_account_index] - meta.post_balances[our_account_index]) / 1e9
                else:  # sell
                    sol_change = (meta.post_balances[our_account_index] - meta.pre_balances[our_account_index]) / 1e9

            # Get pump fee
            fee_account = str(PUMP_FEE)
            fee_account_index = next(
                (i for i, key in enumerate(account_keys) 
                if str(key) == fee_account),
                None
            )
            pump_fee_amount = 0.0
            if fee_account_index is not None:
                pump_fee_amount = abs(
                    (meta.post_balances[fee_account_index] - meta.pre_balances[fee_account_index]) / 1e9
                )

            # Get network fee
            network_fee = meta.fee / 1e9
            
            # Check for token account creation
            token_account_creation_fee = 0.0
            for log in meta.log_messages:
                if "Initialize the associated token account" in log:
                    token_account_creation_fee = 0.00203928  # Standard rent exemption for token accounts
                    break
            
            # For sells, assume account is closed and include the refund value
            # You can use this in your tracking system
            token_account_refund = 0.0
            if tx_type == "sell":
                token_account_refund = 0.00203928  # Standard rent exemption amount
            
            # Calculate effective price
            execution_price = abs(sol_trade_amount / our_token_change) if our_token_change != 0 else 0

            return {
                "tx_type": tx_type,
                "execution_price": execution_price,
                "sol_trade_amount": sol_trade_amount,
                "token_trade_amount": our_token_change,
                "total_sol_change": sol_change,
                "pump_fee": pump_fee_amount,
                "network_fee": network_fee,
                "token_account_fee": token_account_creation_fee,
                "token_account_refund": token_account_refund,
                "total_fees": pump_fee_amount + network_fee + token_account_creation_fee,
                "tx_sig": str(signature),
                "blocktime": blocktime  # Add the transaction blocktime
            }
        except Exception as e:
            self.logger.error(f"Transaction details failed: {str(e)}")
            return None

    async def close(self):
        await self.client.close() 

    async def get_sol_balance(self) -> float:
        """Get SOL balance in wallet"""
        try:
            response = await self.client.get_balance(self.wallet.pubkey())
            return response.value / 1_000_000_000  # Convert lamports to SOL
        except Exception as e:
            self.logger.error(f"Failed to get SOL balance: {e}")
            return 0.0

    async def get_token_balance(self, mint: Pubkey) -> float:
        """Get token balance for a specific mint"""
        try:
            # Get associated token account
            ata = get_associated_token_address(self.wallet.pubkey(), mint)
            
            # Get token balance
            try:
                response = await self.client.get_token_account_balance(ata)
                if response and hasattr(response, 'value'):
                    return float(response.value.amount) / 1_000_000  # Assuming 6 decimals
            except Exception as e:
                if "could not find account" in str(e):
                    # Account doesn't exist yet, which means balance is 0
                    return 0.0
                raise  # Re-raise other exceptions
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Failed to get token balance: {e}")
            return 0.0 

    async def get_recent_priority_fee(self) -> int:
        """Get recent priority fees from Helius"""
        try:
            # Format the request exactly like Jupiter
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getRecentPrioritizationFees",
                "params": [[str(self.wallet.pubkey())]]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.client._provider.endpoint_uri,
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    result = await response.json()
                    
                    if "result" in result and len(result["result"]) > 0:
                        # Get median fee like Jupiter does
                        fees = [fee["prioritizationFee"] for fee in result["result"]]
                        median_fee = statistics.median(fees)
                        
                        # If median is 0, use minimum fee
                        if median_fee == 0:
                            return 1_000  # 1000 microlamports as minimum
                        
                        # Add 10% like Jupiter
                        recommended_fee = int(median_fee * 1.1)
                        
                        return max(recommended_fee, 1_000)  # Never go below minimum
                    
                    return 1_000  # Minimum fee
                    
        except Exception as e:
            # self.logger.error(f"Error getting priority fees: {e}")
            return 1_000  # Minimum fee


    # TODO: Implement balance check for live runs after live testing
    async def check_balances(
        self,
        required_sol: float = 0,
        required_tokens: float = 0,
        mint: Optional[Pubkey] = None
    ) -> Tuple[bool, str]:
        """
        Check if wallet has sufficient balances for a transaction
        Returns (is_sufficient, error_message)
        """
        try:
            current_sol = await self.get_sol_balance()
            
            # Always check minimum SOL for fees
            min_sol = max(required_sol, 0.001)  # At least enough for fees
            if current_sol < min_sol:
                return False, f"Insufficient SOL. Required: {min_sol}, Available: {current_sol}"
                
            # Check token balance if required
            if required_tokens > 0 and mint:
                current_tokens = await self.get_token_balance(mint)
                if current_tokens < required_tokens:
                    return False, f"Insufficient tokens. Required: {required_tokens}, Available: {current_tokens}"
                    
            return True, ""
            
        except Exception as e:
            return False, f"Balance check failed: {str(e)}"