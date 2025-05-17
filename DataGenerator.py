import time
import numpy as np

# Add simulation data generator if no real data is available
class SimulatedDataGenerator:
    """Generates simulated market data when real data isn't available"""
    
    def __init__(self, base_price=30000, volatility=0.001, levels=20):
        self.base_price = base_price
        self.volatility = volatility
        self.levels = levels
        self.last_price = base_price
        self.last_timestamp = int(time.time() * 1000)
    
    def get_orderbook(self):
        """Generate a simulated orderbook"""
        # Generate some price drift
        drift = np.random.normal(0, self.volatility * self.last_price)
        self.last_price = max(0.1, self.last_price + drift)
        
        # Generate bid and ask prices
        spread = self.last_price * 0.0002  # 0.02% spread
        best_bid = self.last_price - spread/2
        best_ask = self.last_price + spread/2
        
        # Generate levels
        bids = []
        asks = []
        
        # Generate bids (descending order)
        for i in range(self.levels):
            price = best_bid * (1 - 0.0001 * i)
            size = np.random.exponential(1) * 2 + 0.5  # Random size
            bids.append([round(price, 2), round(size, 6)])
        
        # Generate asks (ascending order)
        for i in range(self.levels):
            price = best_ask * (1 + 0.0001 * i)
            size = np.random.exponential(1) * 2 + 0.5  # Random size
            asks.append([round(price, 2), round(size, 6)])
        
        # Create timestamp
        self.last_timestamp = int(time.time() * 1000)
        
        return {
            "timestamp": self.last_timestamp,
            "exchange": "OKX",
            "symbol": "BTC-USDT-SWAP",
            "bids": bids,
            "asks": asks
        }