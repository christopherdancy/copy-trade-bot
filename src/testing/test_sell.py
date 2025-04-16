from execution.pump_executor import PumpExecutor
from solders.keypair import Keypair
from solders.pubkey import Pubkey
import asyncio
import json
import os
from pathlib import Path

async def load_wallet():
    """Load wallet from Solana CLI config"""
    config_path = Path(os.path.expanduser("~/.config/solana/id.json"))
    with open(config_path, 'r') as f:
        secret_key = json.load(f)
        return Keypair.from_bytes(bytes(secret_key))

async def test_sell():
    try:
        # Load wallet
        wallet = await load_wallet()
        print(f"Loaded wallet: {wallet.pubkey()}")
        
        # Initialize executor
        executor = PumpExecutor(
            wallet=wallet,
            rpc_url="https://devnet.helius-rpc.com/?api-key=cd782283-4629-4f2e-9966-12e10b7134b9"
        )
        
        # Replace with your token mint
        mint = Pubkey.from_string("BqVk4yHwkB9gboCczGy2PYHd2pM3U27UpAJVDSjqcAqf")

        # Sell 1000 tokens with 2% slippage
        tx_sig = await executor.sell_token(
            mint=mint,
            token_amount=1000,  # Amount in UI units
            slippage_basis_points=200  # 2%
        )
        
        if tx_sig:
            print(f"Sell successful!")
            print(f"Transaction: https://explorer.solana.com/tx/{tx_sig}")
        else:
            print("Sell failed")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'executor' in locals():
            await executor.close()

if __name__ == "__main__":
    asyncio.run(test_sell()) 