import asyncio
import json
import time
import tkinter as tk
from tkinter import ttk
import websockets
from DataGenerator import SimulatedDataGenerator
from ui import CryptoTradingSimulator

# Add support for offline mode using simulated data
class OfflineCryptoTradingSimulator(CryptoTradingSimulator):
    """Extended simulator that can work offline with simulated data"""
    
    def __init__(self, root):
        super().__init__(root)
        # Create simulated data generator
        self.data_generator = SimulatedDataGenerator()
        # Add UI toggle for offline mode
        self.offline_mode = tk.BooleanVar(value=False)
        self.add_offline_toggle()
    
    def add_offline_toggle(self):
        """Add offline mode toggle to UI"""
        # Find the button frame
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for frame in child.winfo_children():
                    if isinstance(frame, ttk.Frame):
                        for subframe in frame.winfo_children():
                            if isinstance(subframe, ttk.LabelFrame) and subframe.winfo_children():
                                for widget in subframe.winfo_children():
                                    if isinstance(widget, ttk.Frame):
                                        # This is probably our button frame
                                        offline_check = ttk.Checkbutton(
                                            widget, 
                                            text="Offline Mode", 
                                            variable=self.offline_mode,
                                            command=self.toggle_offline_mode
                                        )
                                        offline_check.pack(side=tk.LEFT, padx=5)
                                        return
    
    def toggle_offline_mode(self):
        """Handle toggle between online and offline mode"""
        # First stop any running simulation
        self.stop_simulation()
        time.sleep(0.5)  # Wait for threads to clean up
        
        # Start the appropriate mode
        self.start_simulation()
    
    async def connect_websocket(self):
        """Connect to websocket and handle incoming data with reconnection logic"""
        max_retries = 5  # Maximum number of reconnection attempts
        retry_delay = 5  # Seconds between retries
        
        attempt = 0
        while self.running and attempt < max_retries:
            try:
                self.update_status("Connecting to market data...")
                self.log_message(f"Connecting to websocket (attempt {attempt + 1}/{max_retries})...")
                exchange = self.exchange_var.get().lower()
                asset = self.asset_var.get()
                
                url = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{asset}"
                
                async with websockets.connect(url) as ws:
                    self.websocket = ws
                    self.connected = True
                    attempt = 0  # Reset attempt counter on successful connection
                    self.update_status(f"Connected to {exchange} - {asset}")
                    self.log_message(f"✅ Connected to {url}")
                    
                    while self.running:
                        try:
                            response = await ws.recv()
                            data = json.loads(response)
                            
                            # Extract data
                            timestamp = data.get("timestamp", "N/A")
                            exchange = data.get("exchange", "OKX")
                            symbol = data.get("symbol", "BTC-USDT-SWAP")
                            bids = data.get("bids", [])
                            asks = data.get("asks", [])
                            
                            # Store data
                            self.orderbook_data = {
                                "timestamp": timestamp,
                                "exchange": exchange,
                                "symbol": symbol,
                                "bids": bids,
                                "asks": asks
                            }
                            
                            # Update UI and trigger simulation immediately
                            self.update_orderbook(bids, asks)
                            
                        except json.JSONDecodeError:
                            self.log_message(f"⚠️ Invalid JSON response")
                            await asyncio.sleep(0.1)
                        except websockets.exceptions.ConnectionClosed as cc:
                            self.log_message(f"⚠️ Connection closed: {cc}, attempting to reconnect...")
                            self.connected = False
                            break  # Break inner loop to attempt reconnection
                        except Exception as e:
                            self.log_message(f"❌ Error processing data: {str(e)}")
                            await asyncio.sleep(0.1)
                    
                    if not self.running:
                        break  # Exit if we're shutting down
                    
            except Exception as e:
                self.log_message(f"❌ Connection error: {str(e)}")
                self.update_status(f"Connection error: {str(e)}")
                self.connected = False
                
            # Only attempt reconnection if running and not manually disconnected
            if self.running:
                attempt += 1
                if attempt < max_retries:
                    self.log_message(f"⏳ Waiting {retry_delay} seconds before reconnection attempt {attempt + 1}/{max_retries}...")
                    await asyncio.sleep(retry_delay)
                else:
                    self.log_message("❌ Maximum reconnection attempts reached")
                    self.update_status("Connection failed - click Connect to retry")
                    self.running = False  # Stop trying to reconnect
                    # Queue the disconnect button state update
                    self.queue_ui_task(lambda: self.connect_button.config(state=tk.NORMAL))
                    self.queue_ui_task(lambda: self.disconnect_button.config(state=tk.DISABLED))
        
        self.log_message("Disconnected from websocket")
        self.update_status("Disconnected")
        self.connected = False
