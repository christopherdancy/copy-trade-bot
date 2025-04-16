import json
import websockets
import asyncio
import base64
import base58
from decimal import Decimal
from datetime import datetime
from typing import List, Callable, Optional
from dataclasses import dataclass
import struct
from dotenv import load_dotenv
import os
import websockets.exceptions
from logging import Logger

# Load environment variables
load_dotenv()

@dataclass
class TradeEvent:
    mint: str
    sol_amount: Decimal
    token_amount: Decimal
    is_buy: bool
    user: str
    timestamp: datetime
    virtual_sol_reserves: Decimal
    virtual_token_reserves: Decimal
    real_sol_reserves: Decimal
    real_token_reserves: Decimal
    signature: str
    blocktime: Optional[int] = None

    @property
    def price(self) -> Decimal:
        """Calculate price as SOL/token ratio"""
        return self.sol_amount / self.token_amount if self.token_amount != 0 else Decimal(0)

    @property
    def virtual_price(self) -> Decimal:
        """Calculate virtual price from reserves"""
        return (self.virtual_sol_reserves / self.virtual_token_reserves 
                if self.virtual_token_reserves != 0 else Decimal(0))

    @property
    def real_price(self) -> Decimal:
        """Calculate real price from reserves"""
        return (self.real_sol_reserves / self.real_token_reserves 
                if self.real_token_reserves != 0 else Decimal(0))

class PumpDataFeed:
    def __init__(self, logger: Logger, debug: bool = True):
        self.ws_url = os.getenv('WS_URL')
        self.program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        self.callbacks: List[Callable] = []
        self.ws = None
        self.logger = logger
        
        # Simple metrics tracking
        self.connection_status = {
            'last_disconnect_time': None,
            'disconnect_code': None,
            'reconnect_success': False,
            'time_to_reconnect': 0.0
        }
        
        self.message_health = {
            'last_message_time': None,
            'messages_received': 0,
            'processing_errors': 0
        }

    async def connect(self):
        while True:
            try:
                connect_start = datetime.now()
                self.logger.info(f"Attempting to connect to WebSocket at {self.ws_url}")
                self.ws = await websockets.connect(self.ws_url)
                
                # Track successful connection
                self.connection_status['reconnect_success'] = True
                self.connection_status['time_to_reconnect'] = (datetime.now() - connect_start).total_seconds()
                
                # Subscribe to program logs
                subscribe_message = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "logsSubscribe",
                    "params": [
                        {"mentions": [self.program_id]},
                        {"commitment": "confirmed"}
                    ]
                }
                
                await self.ws.send(json.dumps(subscribe_message))
                self.logger.info("Subscription message sent successfully")
                
                while True:
                    try:
                        msg = await self.ws.recv()
                        self.message_health['last_message_time'] = datetime.now()
                        self.message_health['messages_received'] += 1
                        await self.process_message(msg)
                        
                    except websockets.exceptions.ConnectionClosed as e:
                        self.connection_status['last_disconnect_time'] = datetime.now()
                        self.connection_status['disconnect_code'] = e.code
                        self.connection_status['reconnect_success'] = False
                        
                        self.logger.error(
                            f"WebSocket disconnected. Code: {e.code}, "
                            f"Last message: {self.message_health['last_message_time']}"
                        )
                        break
                        
            except Exception as e:
                self.logger.error(f"Connection error: {str(e)}")
                await asyncio.sleep(5)
                continue

    def add_callback(self, callback: Callable):
        """Add a callback to receive market data"""
        self.callbacks.append(callback)

    async def start(self):
        """Start the data feed"""
        await self.connect()

    async def stop(self):
        """Async method to stop the feed"""
        if self.ws:
            await self.ws.close()

    async def process_message(self, msg: str):
        try:
            data = json.loads(msg)
            if "params" not in data:
                return

            # Debug print
            # if self.debug:
            #     print(f"Received message: {msg}")

            result = data["params"].get("result", {}).get("value", {})
            if not result or "logs" not in result:
                return

            logs = result["logs"]
            signature = result.get("signature", "")

            # Look for successful trade instructions
            if not any("Program log: Instruction: Sell" in log or "Program log: Instruction: Buy" in log for log in logs):
                return

            # Find the program data log
            for log in logs:
                if "Program data:" in log:
                    try:
                        encoded_data = log.split("Program data: ")[1]
                        decoded_data = base64.b64decode(encoded_data)
                        
                        # Validate decoded data length
                        expected_length = 8 + 32 + 16 + 1 + 32 + 40
                        if len(decoded_data) != expected_length:
                            # self._debug(f"Skipping invalid data length: {len(decoded_data)}")
                            continue
                        
                        # Skip 8 byte discriminator
                        offset = 8
                        
                        # Parse mint (32 bytes)
                        mint_bytes = decoded_data[offset:offset+32]
                        mint = base58.b58encode(mint_bytes).decode('utf-8')
                        offset += 32
                        
                        # Parse amounts (16 bytes)
                        try:
                            sol_amount, token_amount = struct.unpack("<QQ", decoded_data[offset:offset+16])
                            if sol_amount == 0 or token_amount == 0:
                                continue
                        except struct.error:
                            # print("Failed to parse amounts")
                            continue
                        offset += 16
                        
                        # Parse is_buy (1 byte)
                        is_buy = bool(decoded_data[offset])
                        offset += 1
                        
                        # Parse user (32 bytes)
                        user_bytes = decoded_data[offset:offset+32]
                        user = base58.b58encode(user_bytes).decode('utf-8')
                        offset += 32
                        
                        # Parse timestamp and reserves (40 bytes)
                        try:
                            timestamp, v_sol, v_token, r_sol, r_token = struct.unpack("<QQQQQ", decoded_data[offset:offset+40])
                            # Validate timestamp
                            if timestamp > 2**32:  # Basic sanity check
                                continue
                        except struct.error:
                            # print("Failed to parse reserves")
                            continue
                        
                        # Convert to decimals before division
                        sol_decimal = Decimal(sol_amount) / Decimal(1_000_000_000)
                        token_decimal = Decimal(token_amount) / Decimal(1_000_000)
                        
                        if token_decimal == 0:
                            continue
                            
                        price = sol_decimal / token_decimal

                        trade_event = TradeEvent(
                            mint=mint,
                            sol_amount=sol_decimal,
                            token_amount=token_decimal,
                            is_buy=is_buy,
                            user=user,
                            timestamp=datetime.fromtimestamp(timestamp),
                            virtual_sol_reserves=Decimal(v_sol) / Decimal(1_000_000_000),
                            virtual_token_reserves=Decimal(v_token) / Decimal(1_000_000),
                            real_sol_reserves=Decimal(r_sol) / Decimal(1_000_000_000),
                            real_token_reserves=Decimal(r_token) / Decimal(1_000_000),
                            signature=signature,
                            blocktime=timestamp  # Add blocktime to the trade event
                        )
                        
                        # Call callbacks only if we successfully parsed everything
                        for callback in self.callbacks:
                            # if self.debug:
                                # print(f"Calling callback with trade event: {trade_event.mint}")
                            await callback(trade_event)
                            
                    except Exception as e:
                        # self._debug(f"Error processing Trade in pump data feed: {str(e)}")
                        continue

        except Exception as e:
            self.message_health['processing_errors'] += 1
            self.logger.error(f"Error processing message: {str(e)}")