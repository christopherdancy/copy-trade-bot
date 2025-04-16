from execution.pump_executor import PumpExecutor
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
import base58
import asyncio
import json
import os
from pathlib import Path

async def load_wallet():
    """Load wallet from Solana CLI config"""
    config_path = Path(os.path.expanduser("~/.config/solana/id.json"))
    
    try:
        with open(config_path, 'r') as f:
            secret_key = json.load(f)  # This should load as array of numbers
            return Keypair.from_bytes(bytes(secret_key))
    except Exception as e:
        print(f"Error loading wallet: {e}")
        return None

async def test_buy():
    try:
        # Load wallet directly from Solana config
        wallet = await load_wallet()
        if not wallet:
            return
            
        print(f"Wallet loaded successfully!")
        print(f"Public key: {wallet.pubkey()}")
        
        # Initialize client and executor
        client = AsyncClient("https://devnet.helius-rpc.com/?api-key=cd782283-4629-4f2e-9966-12e10b7134b9")
        executor = PumpExecutor(
            wallet=wallet,
            rpc_url="https://devnet.helius-rpc.com/?api-key=cd782283-4629-4f2e-9966-12e10b7134b9"
        )
        
        # Check balance
        balance = await client.get_balance(wallet.pubkey())
        print(f"Wallet balance: {balance.value / 1_000_000_000} SOL")
        
        if balance.value == 0:
            print("Warning: Wallet has no SOL!")
            return
            
        # Example: Test with a specific token
        # Replace with actual pump token mint address
        mint = Pubkey.from_string("BqVk4yHwkB9gboCczGy2PYHd2pM3U27UpAJVDSjqcAqf")
        
        # Try to buy 0.1 SOL worth with 2% slippage
        tx_sig = await executor.buy_token(
            mint=mint,
            amount_sol=0.1,
            slippage_basis_points=200
        )
        
        if tx_sig:
            print(f"Transaction successful!")
            print(f"Signature: https://explorer.solana.com/tx/{tx_sig}")
        else:
            print("Transaction failed")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'executor' in locals():
            await executor.close()
        if 'client' in locals():
            await client.close()

async def test_pda_derivation():
    # Create executor instance
    wallet = Keypair()  # Dummy wallet for testing
    executor = PumpExecutor(wallet, "https://devnet.helius-rpc.com/?api-key=cd782283-4629-4f2e-9966-12e10b7134b9")
    
    # Test mint address (replace with a known pump token mint)
    mint = Pubkey.from_string("CKgsHAPgTukPW7QJZ7XbSoi9cD3pB2ogehTiNUTjWxmA")
    program_id = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
    
    # Get PDA
    pda = executor.get_bonding_curve_pda(mint, program_id)
    print(f"Mint: {mint}")
    print(f"Derived PDA: {pda}")
    
    await executor.close()

if __name__ == "__main__":
    asyncio.run(test_buy()) 