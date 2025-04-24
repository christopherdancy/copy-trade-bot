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
    virtual_sol_reserves: Decimal
    virtual_token_reserves: Decimal
    real_sol_reserves: Decimal
    real_token_reserves: Decimal
    signature: str
    blocktime: int
    # Essential transaction details
    price: Decimal = None
    total_sol_change: float = 0.0  # Total SOL including fees
    
    def __init__(self, 
                 mint: str, 
                 sol_amount: Decimal, 
                 token_amount: Decimal, 
                 is_buy: bool, 
                 user: str, 
                 virtual_sol_reserves: Decimal,
                 virtual_token_reserves: Decimal,
                 real_sol_reserves: Decimal,
                 real_token_reserves: Decimal,
                 signature: str,
                 blocktime: int,
                 total_sol_change: float):
        self.mint = mint
        self.sol_amount = sol_amount
        self.token_amount = token_amount
        self.is_buy = is_buy
        self.user = user
        self.virtual_sol_reserves = virtual_sol_reserves
        self.virtual_token_reserves = virtual_token_reserves
        self.real_sol_reserves = real_sol_reserves
        self.real_token_reserves = real_token_reserves
        self.signature = signature
        self.blocktime = blocktime
        
        self.price = sol_amount / token_amount if token_amount > 0 else Decimal(0)
        self.total_sol_change = total_sol_change
        
    def as_dict(self) -> dict:
        """Convert to dictionary format compatible with tx_details"""
        return {
            "tx_type": "buy" if self.is_buy else "sell",
            "execution_price": float(self.price),
            "sol_trade_amount": float(self.sol_amount),
            "token_trade_amount": float(self.token_amount),
            "total_sol_change": self.total_sol_change,
            "tx_sig": self.signature,
            "blocktime": self.blocktime
        }

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
                self.logger.info(f"Attempting to connect to Laserstream WebSocket at {self.ws_url}")
                self.ws = await websockets.connect(self.ws_url)
                
                # Track successful connection
                self.connection_status['reconnect_success'] = True
                self.connection_status['time_to_reconnect'] = (datetime.now() - connect_start).total_seconds()
                
                # Use transactionSubscribe specifically for the pump program
                subscribe_message = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "transactionSubscribe",
                    "params": [
                        {
                            "vote": False,
                            "failed": False,
                            "accountInclude": [self.program_id],  # Focus only on our pump program
                            "accountExclude": [],
                            "accountRequired": []  # Could set this to [self.program_id] to be even more specific
                        },
                        {
                            "commitment": "processed",  # Changed to "processed" for lower latency
                            "encoding": "jsonParsed",
                            "transactionDetails": "full",
                            "showRewards": False,
                            "maxSupportedTransactionVersion": 0
                        }
                    ]
                }
                
                await self.ws.send(json.dumps(subscribe_message))
                self.logger.info(f"Enhanced transactionSubscribe sent successfully for pump program: {self.program_id}")
                
                # Setup ping to keep connection alive
                # ping_task = asyncio.create_task(self._keep_alive())
                
                msg_counter = 0
                last_log_time = datetime.now()
                
                try:
                    while True:
                        msg = await self.ws.recv()
                        current_time = datetime.now()
                        self.message_health['last_message_time'] = current_time
                        self.message_health['messages_received'] += 1
                        msg_counter += 1
                            
                        asyncio.create_task(self.process_message(msg))
                except websockets.exceptions.ConnectionClosed as e:
                    self.connection_status['last_disconnect_time'] = datetime.now()
                    self.connection_status['disconnect_code'] = e.code
                    self.connection_status['reconnect_success'] = False
                    
                    self.logger.error(
                        f"WebSocket disconnected. Code: {e.code}, "
                        f"Last message: {self.message_health['last_message_time']}"
                    )
                    
                    # Cancel the ping task
                    ping_task.cancel()
                    
                    # Wait a bit before reconnecting
                    await asyncio.sleep(5)
                        
            except Exception as e:
                self.logger.error(f"Connection error: {str(e)}")
                await asyncio.sleep(5)
                continue
                
    async def _keep_alive(self):
        """Send pings periodically to keep the connection alive"""
        while True:
            try:
                await asyncio.sleep(30)  # Send ping every 30 seconds
                ping_msg = {"jsonrpc": "2.0", "id": 999, "method": "ping"}
                await self.ws.send(json.dumps(ping_msg))
                self.logger.debug("Ping sent to server")
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                return
            except Exception as e:
                self.logger.error(f"Ping error: {str(e)}")
                return  # Stop the ping task if there's an error

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
            # Add timestamp for message receipt
            receipt_time = datetime.now()
            
            data = json.loads(msg)
            
            # Check for ping response
            if "result" in data and data.get("id") == 999:
                self.logger.debug("Received ping response from server")
                return
                
            # For debugging only - log entire message structure
            # if self.message_health['messages_received'] <= 2:
            #     self.logger.info(f"Full message structure: {json.dumps(data, indent=2)}")
            
            # For Enhanced WebSocket, the transaction will be in the result field for the first message
            # and in the params.result field for subscription messages
            transaction = None
            
            if "result" in data and isinstance(data["result"], dict) and "transaction" in data["result"]:
                transaction = data["result"]["transaction"]
            elif "params" in data and "result" in data["params"]:
                result = data["params"]["result"]
                if isinstance(result, dict) and "transaction" in result:
                    transaction = result["transaction"]
            
            if not transaction:
                return
                
            # Extract signature
            signature = result.get("signature", "unknown")
            
            # First check if our program is involved
            found_program = False
            program_idx = None
            
            # Check if transaction has the transaction and meta fields
            if "transaction" in transaction and "meta" in transaction:
                tx_data = transaction["transaction"]
                tx_meta = transaction["meta"]
                
                # Check instructions to see if our program is called
                if "message" in tx_data:
                    message = tx_data["message"]
                    account_keys = message.get("accountKeys", [])
                    instructions = message.get("instructions", [])
                    
                    # Find our program in account keys
                    for i, account in enumerate(account_keys):
                        if account.get("pubkey") == self.program_id:
                            found_program = True
                            program_idx = i
                            break
                    
                    # Also check directly in instructions
                    for instr in instructions:
                        if instr.get("programId") == self.program_id:
                            found_program = True
                            break
                            
                # If program not found directly, check inner instructions
                if not found_program and "innerInstructions" in tx_meta:
                    for inner in tx_meta["innerInstructions"]:
                        for instr in inner.get("instructions", []):
                            if instr.get("programId") == self.program_id:
                                found_program = True
                                break
                        if found_program:
                            break
                
                # If program found, look for trade data in log messages
                if found_program and "logMessages" in tx_meta:
                    # Determine if this is a buy or sell instruction
                    is_buy = None
                    program_data = None
                    
                    # First pass - identify if this is a Buy or Sell instruction
                    for log in tx_meta["logMessages"]:
                        if "Program log: Instruction: Buy" in log:
                            is_buy = True
                            break
                        elif "Program log: Instruction: Sell" in log:
                            is_buy = False
                            break
                    
                    # If this is not a buy or sell instruction, skip processing
                    if is_buy is None:
                        return
                    
                    # Second pass - find the program data for this buy/sell instruction
                    for log in tx_meta["logMessages"]:
                        if "Program data:" in log:
                            # Found the binary data containing trade details
                            program_data = log.split("Program data: ")[1]
                            break
                    
                    if not program_data:
                        self.logger.debug("Buy/Sell instruction found but no program data available")
                        return
                        
                    try:
                        # Decode the base64 data
                        decoded_data = base64.b64decode(program_data)
                        
                        # Validate length before proceeding
                        expected_length = 8 + 32 + 16 + 1 + 32 + 40  # 129 bytes total
                        if len(decoded_data) < expected_length:
                            # self.logger.debug(f"Program data too short: {len(decoded_data)} bytes, expected {expected_length}")
                            return
                        
                        # Skip the first 8 bytes (discriminator)
                        offset = 8
                        
                        # Parse mint (32 bytes)
                        mint_bytes = decoded_data[offset:offset+32]
                        mint = base58.b58encode(mint_bytes).decode('utf-8')
                        offset += 32
                        
                        # Parse amounts (16 bytes)
                        try:
                            sol_amount, token_amount = struct.unpack("<QQ", decoded_data[offset:offset+16])
                            
                            # Basic validation
                            if sol_amount == 0 and token_amount == 0:
                                self.logger.debug("Both sol and token amounts are zero, skipping")
                                return
                        except struct.error as e:
                            self.logger.debug(f"Failed to parse amounts: {str(e)}")
                            return
                            
                        offset += 16
                        
                        # Parse is_buy (1 byte) - we already know this from the logs
                        offset += 1  # Skip the byte since we already know is_buy
                        
                        # Parse user (32 bytes)
                        user_bytes = decoded_data[offset:offset+32]
                        user = base58.b58encode(user_bytes).decode('utf-8')
                        offset += 32
                        
                        # Parse timestamp and reserves (40 bytes)
                        try:
                            timestamp, v_sol, v_token, r_sol, r_token = struct.unpack("<QQQQQ", decoded_data[offset:offset+40])
                            
                            # Validate timestamp (between 2020 and 2050)
                            min_timestamp = 1577836800  # Jan 1, 2020
                            max_timestamp = 2524608000  # Jan 1, 2050
                            if timestamp < min_timestamp or timestamp > max_timestamp:
                                # self.logger.debug(f"Timestamp out of range: {timestamp}")
                                return
                        except struct.error as e:
                            self.logger.debug(f"Failed to parse timestamp and reserves: {str(e)}")
                            return
                        
                        # Convert to decimals with correct decimal places
                        sol_decimal = Decimal(sol_amount) / Decimal(1_000_000_000)  # 9 decimal places for SOL
                        token_decimal = Decimal(token_amount) / Decimal(1_000_000)  # 6 decimal places for tokens
                        
                        if token_decimal == 0:
                            self.logger.debug("Token amount is zero, skipping")
                            return
                            
                        # Create timestamp from unix timestamp
                        try:
                            tx_timestamp = datetime.fromtimestamp(timestamp)
                        except (ValueError, OverflowError, OSError) as e:
                            self.logger.debug(f"Invalid timestamp conversion: {timestamp}, error: {str(e)}")
                            return
                        
                        # Look for the user account first, then fall back to sol_amount
                        if "accountKeys" in tx_data["message"] and "preBalances" in tx_meta and "postBalances" in tx_meta:
                            account_keys = tx_data["message"]["accountKeys"]
                            user_account_idx = None
                            user_change = None
                            
                            # First, try to find the user account specifically
                            for i, account in enumerate(account_keys):
                                account_key = account.get("pubkey", account) if isinstance(account, dict) else account
                                if str(account_key) == str(user):
                                    user_account_idx = i
                                    pre_balance = tx_meta["preBalances"][i]
                                    post_balance = tx_meta["postBalances"][i]
                                    user_change = (post_balance - pre_balance) / 1e9
                                    break
                            
                            # If we found the user account with a significant change, use that
                            if user_account_idx is not None and abs(user_change) > 0.0001:
                                if is_buy and user_change < 0:
                                    # For buys, user change will be negative (SOL spent)
                                    total_sol_change = abs(user_change)
                                elif not is_buy and user_change > 0:
                                    # For sells, user change will be positive (SOL received)
                                    total_sol_change = user_change
                                else:
                                    # User change exists but doesn't match expected direction
                                    # Fall back to sol_amount
                                    total_sol_change = float(sol_decimal)
                            else:
                                # User account not found or change too small
                                # Fall back to sol_amount
                                total_sol_change = float(sol_decimal)
                        else:
                            # Fall back to sol_amount
                            total_sol_change = float(sol_decimal)
                        
                        # Create TradeEvent with accurate total_sol_change
                        trade_event = TradeEvent(
                            mint=mint,
                            sol_amount=sol_decimal,
                            token_amount=token_decimal,
                            is_buy=is_buy,
                            user=user,
                            virtual_sol_reserves=Decimal(v_sol) / Decimal(1_000_000_000),
                            virtual_token_reserves=Decimal(v_token) / Decimal(1_000_000),
                            real_sol_reserves=Decimal(r_sol) / Decimal(1_000_000_000),
                            real_token_reserves=Decimal(r_token) / Decimal(1_000_000),
                            signature=signature,
                            blocktime=timestamp,
                            total_sol_change=total_sol_change
                        )
                        
                        # Calculate processing time
                        parsing_time = datetime.now()
                        parsing_latency_ms = (parsing_time - receipt_time).total_seconds() * 1000
                        
                        # Call callbacks
                        for callback in self.callbacks:
                            asyncio.create_task(callback(trade_event))
                        
                        # Calculate total time
                        total_time = datetime.now()
                        total_latency_ms = (total_time - receipt_time).total_seconds() * 1000

                    except Exception as e:
                        self.logger.error(f"Error parsing program data: {str(e)}")

        except Exception as e:
            self.message_health['processing_errors'] += 1
            self.logger.error(f"Error processing message: {str(e)}")
            # Log the raw message in case of error
            try:
                self.logger.error(f"Raw message that caused error: {msg[:500]}...")
            except:
                self.logger.error("Could not log raw message that caused error")