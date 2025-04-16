import base58
from solders.keypair import Keypair
from pathlib import Path
import json
import os
import asyncio
from solana.rpc.async_api import AsyncClient

async def check_wallet():
    # Default Solana config directory
    config_dir = os.path.expanduser("~/.config/solana")
    wallet_path = Path(config_dir) / "id.json"
    
    try:
        if not wallet_path.exists():
            print(f"No wallet found at: {wallet_path}")
            return
            
        # Read the wallet file
        with open(wallet_path, 'r') as f:
            content = f.read().strip()
            
        try:
            # Try parsing as JSON first
            secret = json.loads(content)
            if isinstance(secret, list):
                wallet = Keypair.from_bytes(bytes(secret))
            else:
                # Try as base58 string
                wallet = Keypair.from_bytes(base58.b58decode(content))
        except json.JSONDecodeError:
            # Try as raw base58 string
            wallet = Keypair.from_bytes(base58.b58decode(content))
        
        print("\nWallet Info:")
        print(f"Path: {wallet_path}")
        print(f"Public Key: {wallet.pubkey()}")
        print(f"\nPrivate Key (base58): {base58.b58encode(bytes(wallet.secret())).decode()}")
        print("\nCopy the private key above and use it in test_buy.py")
        
        # Check balance
        client = AsyncClient("https://devnet.helius-rpc.com/?api-key=cd782283-4629-4f2e-9966-12e10b7134b9")
        balance = await client.get_balance(wallet.pubkey())
        print(f"\nBalance: {balance.value / 1_000_000_000} SOL")
        await client.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_wallet()) 