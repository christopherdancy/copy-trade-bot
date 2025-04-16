from execution.wallet import WalletManager
import asyncio
from solana.rpc.async_api import AsyncClient

async def test_wallet():
    # Create wallet manager
    wallet_manager = WalletManager()
    
    # Create new wallet
    wallet = wallet_manager.create_new_wallet()
    
    # Save wallet
    wallet_manager.save_wallet(wallet, "pump-test-wallet.json")
    
    # Print wallet info
    print(f"\nWallet created:")
    print(f"Address: {wallet.pubkey()}")
    
    # Check balance
    client = AsyncClient("https://devnet.helius-rpc.com/?api-key=cd782283-4629-4f2e-9966-12e10b7134b9")
    balance = await client.get_balance(wallet.pubkey())
    print(f"Balance: {balance.value / 1_000_000_000} SOL")
    await client.close()
    
    return wallet

if __name__ == "__main__":
    wallet = asyncio.run(test_wallet()) 