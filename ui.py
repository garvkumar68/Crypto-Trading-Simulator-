import asyncio
import json
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import websockets
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import csv
from queue import Queue
from collections import deque
from CryptoCalculations import CryptoCalculations

class CryptoTradingSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Crypto Trading Simulator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.orderbook_data = {}
        self.simulation_results = []
        self.websocket = None
        self.running = False
        self.connected = False
        
        # Message queue for thread-safe UI updates
        self.ui_queue = Queue()
        
        # Throttling for UI updates
        self.last_ui_update = 0
        self.ui_update_interval = 0.1  # seconds
        
        # Throttling for simulations
        self.last_simulation = 0
        self.simulation_interval = 1.0  # seconds
        
        # Cache for orderbook updates to avoid redundant calculations
        self.orderbook_cache = {
            "bids": [],
            "asks": [],
            "mid_price": 0,
            "last_update": 0
        }
        
        # Data buffers for performance optimization
        self.slippage_data = deque(maxlen=100)  # Store only the last 100 points
        
        # Default values
        self.exchange_var = tk.StringVar(value="OKX")
        self.asset_var = tk.StringVar(value="BTC-USDT-SWAP")
        self.order_type_var = tk.StringVar(value="market")
        self.order_side_var = tk.StringVar(value="buy")
        self.quantity_var = tk.StringVar(value="100")
        self.volatility_var = tk.StringVar(value="medium")
        self.fee_tier_var = tk.StringVar(value="0.1%")
        
        # Create UI elements
        self.create_ui()
        
        # Setup periodic UI updater
        self.setup_periodic_updater()
        
        # Start simulation on launch
        self.start_simulation()
    
    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (inputs)
        left_frame = ttk.LabelFrame(main_frame, text="Input Parameters")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Right panel (outputs)
        right_frame = ttk.LabelFrame(main_frame, text="Processed Output Values")
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Bottom frame for orderbook and logs
        bottom_frame = ttk.LabelFrame(main_frame, text="Market Data & Logs")
        bottom_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        # Create input parameters
        self.create_input_panel(left_frame)
        
        # Create output parameters
        self.create_output_panel(right_frame)
        
        # Create orderbook and logs section
        self.create_orderbook_panel(bottom_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_input_panel(self, parent):
        # Exchange selection
        ttk.Label(parent, text="Exchange:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        exchange_combo = ttk.Combobox(parent, textvariable=self.exchange_var, state="readonly")
        exchange_combo['values'] = ('OKX', 'Binance', 'Coinbase', 'Kraken')
        exchange_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        exchange_combo.bind("<<ComboboxSelected>>", self.on_parameter_change)
        
        # Asset selection
        ttk.Label(parent, text="Spot Asset:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        asset_combo = ttk.Combobox(parent, textvariable=self.asset_var)
        asset_combo['values'] = ('BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP')
        asset_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        asset_combo.bind("<<ComboboxSelected>>", self.on_parameter_change)
        
        # Order type
        ttk.Label(parent, text="Order Type:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        order_type_combo = ttk.Combobox(parent, textvariable=self.order_type_var, state="readonly")
        order_type_combo['values'] = ('market', 'limit')
        order_type_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        order_type_combo.bind("<<ComboboxSelected>>", self.on_parameter_change)
        
        # Order side
        ttk.Label(parent, text="Order Side:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        order_side_combo = ttk.Combobox(parent, textvariable=self.order_side_var, state="readonly")
        order_side_combo['values'] = ('buy', 'sell')
        order_side_combo.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        order_side_combo.bind("<<ComboboxSelected>>", self.on_parameter_change)
        
        # Quantity
        ttk.Label(parent, text="Quantity (USD):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        quantity_entry = ttk.Entry(parent, textvariable=self.quantity_var)
        quantity_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        quantity_entry.bind("<FocusOut>", self.on_parameter_change)
        quantity_entry.bind("<Return>", self.on_parameter_change)
        
        # Volatility
        ttk.Label(parent, text="Volatility:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        volatility_combo = ttk.Combobox(parent, textvariable=self.volatility_var, state="readonly")
        volatility_combo['values'] = ('low', 'medium', 'high')
        volatility_combo.grid(row=5, column=1, padx=5, pady=5, sticky="ew")
        volatility_combo.bind("<<ComboboxSelected>>", self.on_parameter_change)
        
        # Fee tier
        ttk.Label(parent, text="Fee Tier:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        fee_tier_combo = ttk.Combobox(parent, textvariable=self.fee_tier_var, state="readonly")
        fee_tier_combo['values'] = ('0.1%', '0.08%', '0.06%', '0.04%', '0.02%')
        fee_tier_combo.grid(row=6, column=1, padx=5, pady=5, sticky="ew")
        fee_tier_combo.bind("<<ComboboxSelected>>", self.on_parameter_change)
        
        # Action buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=7, column=0, columnspan=2, padx=5, pady=15, sticky="ew")
        
        self.connect_button = ttk.Button(button_frame, text="Connect", command=self.start_simulation)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_button = ttk.Button(button_frame, text="Disconnect", command=self.stop_simulation)
        self.disconnect_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Slippage vs Order Size plot button
        self.plot_button = ttk.Button(button_frame, text="Plot Slippage", command=self.plot_slippage)
        self.plot_button.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        parent.columnconfigure(1, weight=1)
    
    def create_output_panel(self, parent):
        # Create a frame for simulation results
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Matplotlib figure for slippage vs order size
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize the plot with empty data
        self.line, = self.ax.plot([], [], 'r-', alpha=0.5)
        self.scatter = self.ax.scatter([], [], alpha=0.7)
        self.regression_line, = self.ax.plot([], [], 'b--', alpha=0.8)
        self.ax.set_xlabel('Order Size (USD)')
        self.ax.set_ylabel('Slippage (%)')
        self.ax.set_title('Slippage vs Order Size')
        self.ax.grid(True, alpha=0.3)
        self.ax.text(0.5, 0.5, "No simulation data yet", ha='center', va='center', transform=self.ax.transAxes)
        self.plot_canvas.draw()
    
    def create_orderbook_panel(self, parent):
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Orderbook tab
        orderbook_frame = ttk.Frame(notebook)
        notebook.add(orderbook_frame, text="Order Book")
        
        # Log tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Logs")
        
        # Orderbook display
        orderbook_display = ttk.Frame(orderbook_frame)
        orderbook_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Ask side
        ask_frame = ttk.LabelFrame(orderbook_display, text="Ask")
        ask_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ask_text = scrolledtext.ScrolledText(ask_frame, wrap=tk.WORD, height=10, width=30)
        self.ask_text.pack(fill=tk.BOTH, expand=True)
        
        # Bid side
        bid_frame = ttk.LabelFrame(orderbook_display, text="Bid")
        bid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.bid_text = scrolledtext.ScrolledText(bid_frame, wrap=tk.WORD, height=10, width=30)
        self.bid_text.pack(fill=tk.BOTH, expand=True)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_periodic_updater(self):
        """Set up a periodic task to process UI updates from the queue"""
        def process_queue():
            # Process all items in queue
            try:
                while not self.ui_queue.empty():
                    task, args = self.ui_queue.get_nowait()
                    try:
                        task(*args)
                    except Exception as e:
                        print(f"Error executing UI task: {e}")
                    finally:
                        self.ui_queue.task_done()
            except Exception as e:
                print(f"Error processing UI queue: {e}")
            
            # Schedule next check
            self.root.after(10, process_queue)
        
        # Start the periodic check
        self.root.after(10, process_queue)
    
    def queue_ui_task(self, task, *args):
        """Add a task to the UI update queue"""
        self.ui_queue.put((task, args))
    
    def on_parameter_change(self, event=None):
        """Handle parameter changes and trigger simulation if needed"""
        # Trigger a new simulation if we have orderbook data
        current_time = time.time()
        if (self.orderbook_cache["bids"] and self.orderbook_cache["asks"] and 
            current_time - self.last_simulation >= self.simulation_interval):
            self.last_simulation = current_time
            # Queue the simulation to avoid blocking UI
            self.queue_ui_task(
                self.simulate_market_order, 
                self.orderbook_cache["bids"], 
                self.orderbook_cache["asks"], 
                self.orderbook_cache["mid_price"]
            )
            
    def update_plot_data(self):
        """Update the plot data with USD amounts on x-axis, better scaling for large values"""
        if not self.simulation_results:
            return
            
        # Get USD amounts from simulation results
        sizes_usd = [result.get('Order Size', 0) for result in self.simulation_results]
        slippages = [float(result.get('Slippage (%)').strip('%')) 
                    if isinstance(result.get('Slippage (%)'), str) 
                    else result.get('Slippage (%)', 0) 
                    for result in self.simulation_results]
        
        # Clear the axes
        self.ax.clear()
        
        # Plot the data
        self.ax.scatter(sizes_usd, slippages, alpha=0.7, color='blue')
        self.ax.plot(sizes_usd, slippages, 'r-', alpha=0.5)
        
        # Add regression line if we have enough data
        if len(sizes_usd) > 2 and len(set(sizes_usd)) > 1 and len(set(slippages)) > 1:
            try:
                z = np.polyfit(sizes_usd, slippages, 1)
                p = np.poly1d(z)
                self.ax.plot(sizes_usd, p(sizes_usd), "b--", alpha=0.8)
            except:
                pass
        
        # Set labels and title
        self.ax.set_xlabel('Order Size (USD)')
        self.ax.set_ylabel('Slippage (%)')
        self.ax.set_title('Slippage vs Order Size (USD)')
        self.ax.grid(True, alpha=0.3)
        
        # Set intelligent axis scaling
        if sizes_usd and slippages:
            max_usd = max(sizes_usd)
            
            # Determine scaling based on order size magnitude
            if max_usd > 100000:  # For very large orders (100k+ USD)
                # Scale x-axis in 100k increments
                x_min = 0
                x_max = max(100000, max_usd * 1.1)  # Add 10% buffer
                self.ax.set_xlim([x_min, x_max])
                
                # Use scientific notation or "k" notation for x-axis
                if max_usd > 1000000:
                    formatter = plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M')
                else:
                    formatter = plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k')
                self.ax.xaxis.set_major_formatter(formatter)
                
                # Adjust y-axis to focus on typical slippage range
                y_min = 0
                y_max = max(0.1, max(slippages) * 1.5)  # Cap at 1.5x max slippage
                self.ax.set_ylim([y_min, y_max])
                
            else:  # Normal scaling for smaller orders
                # Standard buffer calculation
                x_buffer = max(1000, (max(sizes_usd) - min(sizes_usd)) * 0.2) if len(sizes_usd) > 1 else max(1000, sizes_usd[0] * 0.2)
                y_buffer = max(0.01, (max(slippages) - min(slippages)) * 0.2) if len(slippages) > 1 else max(0.01, abs(slippages[0] * 0.2))
                
                x_min = max(0, min(sizes_usd) - x_buffer)
                x_max = max(sizes_usd) + x_buffer
                y_min = max(0, min(slippages) - y_buffer)
                y_max = max(slippages) + y_buffer
                
                self.ax.set_xlim([x_min, x_max])
                self.ax.set_ylim([y_min, y_max])
        
        # Add legend
        self.ax.legend(['Data Points', 'Actual Trend', 'Regression Line'], 
                    loc='upper left', framealpha=0.5)
        
        # Redraw canvas
        self.plot_canvas.draw_idle()

    def log_message(self, message):
        """Add message to log with timestamp"""
        def _log():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        
        self.queue_ui_task(_log)
        
    def update_orderbook(self, bids, asks):
        """Update the orderbook display and trigger simulation immediately"""
        def _update():
            self.orderbook_cache = {
            "bids": bids,
            "asks": asks,
            "mid_price": (float(bids[0][0]) + float(asks[0][0])) / 2 if bids and asks else 0,
            "last_update": time.time()
            }
            # Clear current data
            self.bid_text.delete(1.0, tk.END)
            self.ask_text.delete(1.0, tk.END)
            
            # Format and display bids
            self.bid_text.insert(tk.END, "PRICE\t\tQUANTITY\n")
            self.bid_text.insert(tk.END, "-" * 30 + "\n")
            for price, qty in bids[:10]:  # Show top 10 levels
                self.bid_text.insert(tk.END, f"{price}\t\t{qty}\n")
            
            # Format and display asks
            self.ask_text.insert(tk.END, "PRICE\t\tQUANTITY\n")
            self.ask_text.insert(tk.END, "-" * 30 + "\n")
            for price, qty in asks[:10]:  # Show top 10 levels
                self.ask_text.insert(tk.END, f"{price}\t\t{qty}\n")
        
        # Update immediately without throttling when we have new data
        self.queue_ui_task(_update)
        
        # Calculate mid price
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            
            # Trigger simulation immediately with new data
            self.queue_ui_task(
                self.simulate_market_order, 
                bids, 
                asks, 
                mid_price
            )

    def update_simulation_results(self, results):
        """Update the simulation results display"""
        def _update():
            self.results_text.delete(1.0, tk.END)
            
            # Add header
            self.results_text.insert(tk.END, f"Simulation Results\n")
            self.results_text.insert(tk.END, f"{'=' * 50}\n\n")
            
            # Order details
            quantity = self.quantity_var.get()
            side = self.order_side_var.get().upper()
            self.results_text.insert(tk.END, f"Order: {side} {quantity} USD\n")
            self.results_text.insert(tk.END, f"Asset: {self.asset_var.get()}\n")
            self.results_text.insert(tk.END, f"Exchange: {self.exchange_var.get()}\n\n")
            
            # Results
            for key, value in results.items():
                self.results_text.insert(tk.END, f"{key}: {value}\n")
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.results_text.insert(tk.END, f"\nGenerated at: {timestamp}\n")
        
        # Store results for plotting
        order_size = float(self.quantity_var.get())
        result_with_size = {**results, 'Order Size': order_size}
        self.simulation_results.append(result_with_size)
        
        self.queue_ui_task(_update)
        self.queue_ui_task(self.update_plot_data)
    
    def update_status(self, message):
        """Update status bar message"""
        self.queue_ui_task(lambda: self.status_var.set(message))
        
    def simulate_market_order(self, bids, asks, mid_price):
        """Simulate a market order execution with the current orderbook using models"""
        try:
            # Get parameters from UI
            order_side = self.order_side_var.get()
            usd_amount = float(self.quantity_var.get())
            volatility = self.volatility_var.get()
            exchange = self.exchange_var.get()
            fee_tier = self.fee_tier_var.get()
            asset = self.asset_var.get().split('-')[0]
            
            # Select book side based on order side
            book_side = asks if order_side == "buy" else bids
            if order_side == "sell":
                book_side = [(float(price), float(qty)) for price, qty in reversed(book_side)]
            else:
                book_side = [(float(price), float(qty)) for price, qty in book_side]
            
            # Implement the simulation logic
            total_cost = 0.0
            total_qty = 0.0
            remaining_usd = usd_amount
            
            for price, qty in book_side:
                order_cost = price * qty
                
                if order_cost <= remaining_usd:
                    total_cost += order_cost
                    total_qty += qty
                    remaining_usd -= order_cost
                else:
                    partial_qty = remaining_usd / price
                    total_cost += remaining_usd
                    total_qty += partial_qty
                    break
            
            if total_qty == 0:
                self.log_message(f"⚠️ Market too shallow to execute ${usd_amount:,} order.")
                return None
            
            # Calculate average execution price
            avg_exec_price = total_cost / total_qty
            
            # Calculate metrics using models
            # 1. Slippage - use regression model
            slippage_pct = CryptoCalculations.predict_slippage_with_regression(
                self.simulation_results, usd_amount, book_side, self.log_message
            )
            slippage_usd = round(mid_price * slippage_pct / 100, 4)
            
            # 2. Maker/Taker proportion - use logistic regression
            maker_prop, taker_prop = CryptoCalculations.calculate_maker_taker_proportion(
                usd_amount, volatility, exchange, asset
            )
            
            # 3. Fees - use rule-based model
            fee = CryptoCalculations.calculate_fees(
                usd_amount, maker_prop, fee_tier, exchange
            )
            
            # 4. Market Impact - use Almgren-Chriss model
            market_impact = CryptoCalculations.calculate_market_impact(
                usd_amount, mid_price, book_side, order_side, volatility, asset
            )
            
            # 5. Measure internal latency
            latency = CryptoCalculations.measure_internal_latency(exchange, asset)
            
            # 6. Calculate net cost
            net_cost = round(slippage_usd * total_qty + fee + (market_impact/100 * usd_amount), 4)
            
            # Prepare results
            results = {
                "Executed Amount": f"{round(total_qty, 6)} {self.asset_var.get().split('-')[0]}",
                "Avg Execution Price": f"${round(avg_exec_price, 2)}",
                "Slippage (USD)": f"${slippage_usd}",
                "Slippage (%)": f"{slippage_pct}%",
                "Fees (USD)": f"${fee}",
                "Market Impact (%)": f"{market_impact}%",
                "Net Cost (USD)": f"${net_cost}",
                "Maker/Taker": f"{round(maker_prop*100, 2)}% / {round(taker_prop*100, 2)}%",
                "Internal Latency": f"{latency} ms"
            }
            
            # Update UI with results
            self.update_simulation_results(results)
            
            # Log simulation
            self.log_message(f"Simulated {order_side.upper()} order: ${usd_amount:,} at avg price ${round(avg_exec_price, 2)}")
            
            return results
            
        except Exception as e:
            self.log_message(f"❌ Simulation error: {str(e)}")
            return None

    # Add these imports at the top of your file
    # import numpy as np
    # import time
    # from sklearn.linear_model import LinearRegression
    # Try to import QuantileRegressor, but don't fail if not available
    # try:
    #     from sklearn.linear_model import QuantileRegressor
    # except ImportError:
    #     pass

    async def connect_websocket(self):
        """Connect to websocket and handle incoming data"""
        try:
            self.update_status("Connecting to market data...")
            self.log_message(f"Connecting to websocket...")
            exchange = self.exchange_var.get().lower()
            asset = self.asset_var.get()
            
            url = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{asset}"
            
            async with websockets.connect(url) as ws:
                self.websocket = ws
                self.connected = True
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
                    except websockets.exceptions.ConnectionClosed:
                        self.log_message(f"⚠️ Connection closed, attempting to reconnect...")
                        self.connected = False
                        break
                    except Exception as e:
                        self.log_message(f"❌ Error processing data: {str(e)}")
                        await asyncio.sleep(0.1)
                
                self.log_message("Disconnected from websocket")
                self.update_status("Disconnected")
                self.connected = False
                
        except Exception as e:
            self.log_message(f"❌ Connection error: {str(e)}")
            self.update_status(f"Connection error: {str(e)}")
            self.connected = False
    
    def start_websocket_loop(self):
        """Start the websocket loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect_websocket())
        loop.close()
    
    def start_simulation(self):
        """Start the simulation"""
        if not self.running:
            self.running = True
            self.update_status("Starting simulation...")
            self.log_message("Starting simulation...")
            # Update button states
            self.connect_button.config(state=tk.DISABLED)
            self.disconnect_button.config(state=tk.NORMAL)
            self.thread = threading.Thread(target=self.start_websocket_loop)
            self.thread.daemon = True
            self.thread.start()  
             
    def stop_simulation(self):
        """Stop the simulation and clean up"""
        if self.running:
            self.running = False
            self.connected = False
            self.update_status("Stopping simulation...")
            self.log_message("Stopping simulation...")
            
            # Close websocket if it exists
            if self.websocket:
                try:
                    asyncio.get_event_loop().run_until_complete(self.websocket.close())
                except:
                    pass
                finally:
                    self.websocket = None
            
            # Update button states
            self.connect_button.config(state=tk.NORMAL)
            self.disconnect_button.config(state=tk.DISABLED)

    
    def save_results(self):
        """Save simulation results to CSV file"""
        if not self.simulation_results:
            messagebox.showinfo("No Data", "No simulation results to save")
            return
        
        try:
            # Create a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_sim_results_{timestamp}.csv"
            
            # Ensure all dictionaries have the same keys
            all_keys = set()
            for result in self.simulation_results:
                all_keys.update(result.keys())
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(all_keys))
                writer.writeheader()
                for result in self.simulation_results:
                    # Ensure all keys are present in each row
                    row = {key: result.get(key, '') for key in all_keys}
                    writer.writerow(row)
            
            self.log_message(f"✅ Results saved to {filename}")
            self.update_status(f"Results saved to {filename}")
            messagebox.showinfo("Save Complete", f"Results saved to {filename}")
            
        except Exception as e:
            self.log_message(f"❌ Error saving results: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save results: {str(e)}")
                
    def plot_slippage(self):
        """Create slippage plot with USD amounts and better large-value handling"""
        if not self.simulation_results:
            messagebox.showinfo("No Data", "No simulation results to plot")
            return
        
        try:
            # Get USD amounts
            sizes_usd = [result.get('Order Size', 0) for result in self.simulation_results]
            slippages = [float(result.get('Slippage (%)').strip('%')) 
                        if isinstance(result.get('Slippage (%)'), str) 
                        else result.get('Slippage (%)', 0) 
                        for result in self.simulation_results]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(sizes_usd, slippages, alpha=0.7, color='blue')
            ax.plot(sizes_usd, slippages, 'r-', alpha=0.5)
            
            if len(sizes_usd) > 2:
                z = np.polyfit(sizes_usd, slippages, 1)
                p = np.poly1d(z)
                ax.plot(sizes_usd, p(sizes_usd), "b--", alpha=0.8)
            
            ax.set_xlabel('Order Size (USD)')
            ax.set_ylabel('Slippage (%)')
            ax.set_title('Slippage vs Order Size (USD)')
            ax.grid(True, alpha=0.3)
            
            # Intelligent scaling
            if sizes_usd and slippages:
                max_usd = max(sizes_usd)
                
                if max_usd > 100000:  # Large orders
                    x_min = 0
                    x_max = max(100000, max_usd * 1.1)
                    ax.set_xlim([x_min, x_max])
                    
                    if max_usd > 1000000:
                        formatter = plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M')
                    else:
                        formatter = plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k')
                    ax.xaxis.set_major_formatter(formatter)
                    
                    y_min = 0
                    y_max = max(0.1, max(slippages) * 1.5)
                    ax.set_ylim([y_min, y_max])
                else:  # Normal orders
                    x_buffer = max(1000, (max(sizes_usd) - min(sizes_usd)) * 0.2) if len(sizes_usd) > 1 else max(1000, sizes_usd[0] * 0.2)
                    y_buffer = max(0.01, (max(slippages) - min(slippages)) * 0.2) if len(slippages) > 1 else max(0.01, abs(slippages[0] * 0.2))
                    
                    x_min = max(0, min(sizes_usd) - x_buffer)
                    x_max = max(sizes_usd) + x_buffer
                    y_min = max(0, min(slippages) - y_buffer)
                    y_max = max(slippages) + y_buffer
                    
                    ax.set_xlim([x_min, x_max])
                    ax.set_ylim([y_min, y_max])
            
            ax.legend(['Data Points', 'Actual Trend', 'Regression Line'], 
                    loc='upper left', framealpha=0.5)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.log_message(f"❌ Error creating plot: {str(e)}")
            messagebox.showerror("Plot Error", f"Failed to create plot: {str(e)}")