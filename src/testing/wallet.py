import base58
from solders.keypair import Keypair
from pathlib import Path
import json
import os

class WalletManager:
    def __init__(self, config_dir: str = "~/.config/solana"):
        self.config_dir = os.path.expanduser(config_dir)
        
    def create_new_wallet(self) -> Keypair:
        """Create a new random wallet"""
        return Keypair()
        
    def save_wallet(self, keypair: Keypair, filename: str = "test-wallet.json"):
        """Save wallet to JSON file"""
        path = Path(self.config_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert bytes to list for JSON serialization
        secret = list(keypair.secret())
        
        with open(path, 'w') as f:
            json.dump(secret, f)
            
        print(f"Wallet saved to: {path}")
        print(f"Public key: {keypair.pubkey()}")
        print(f"Private key (base58): {base58.b58encode(bytes(keypair.secret())).decode()}")
        
    def load_wallet(self, filename: str = "test-wallet.json") -> Keypair:
        """Load wallet from JSON file"""
        path = Path(self.config_dir) / filename
        
        with open(path, 'r') as f:
            secret = json.load(f)
            
        return Keypair.from_bytes(bytes(secret)) 