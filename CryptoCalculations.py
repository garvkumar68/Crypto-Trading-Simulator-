#CryptoCalculations.py
import numpy as np
import time
from sklearn.linear_model import LinearRegression
try:
    from sklearn.linear_model import QuantileRegressor
except ImportError:
    pass

class CryptoCalculations:
    @staticmethod
    def predict_slippage_with_regression(simulation_results, order_size, book_side, log_message=None):
        """
        Calculate expected slippage using linear or quantile regression
        
        Args:
            simulation_results: List of previous simulation results
            order_size: Size of order in USD
            book_side: List of tuples (price, quantity) from orderbook
            log_message: Optional logging function
            
        Returns:
            Estimated slippage percentage
        """
        try:
            # If we have enough simulation results, use historical data for regression
            if len(simulation_results) > 5:
                # Extract historical data
                sizes = np.array([result.get('Order Size', 0) for result in simulation_results])
                slippages = np.array([float(result.get('Slippage (%)').strip('%')) 
                                    if isinstance(result.get('Slippage (%)'), str) 
                                    else result.get('Slippage (%)', 0) 
                                    for result in simulation_results])
                
                # Reshape for scikit-learn compatibility
                sizes = sizes.reshape(-1, 1)
                
                # Use quantile regression for more robust estimates
                # We'll predict the 75th percentile for a conservative estimate
                from sklearn.linear_model import LinearRegression
                
                # First try linear regression as fallback
                model = LinearRegression()
                model.fit(sizes, slippages)
                predicted_slippage = model.predict([[order_size]])[0]
                
                # Only attempt quantile regression if data looks suitable
                if len(set(sizes.flatten())) > 3 and len(set(slippages)) > 3:
                    try:
                        from sklearn.linear_model import QuantileRegressor
                        # Use more conservative parameters for large orders
                        alpha = 0.5 if order_size < 1e6 else 1.0
                        model = QuantileRegressor(quantile=0.75, alpha=alpha, solver='highs')
                        model.fit(sizes, slippages)
                        predicted_slippage = model.predict([[order_size]])[0]
                    except:
                        # If quantile regression fails, use the linear regression result
                        pass
                
                # Add a safety margin for very large orders
                if order_size > 1e6:
                    predicted_slippage *= 1.5
            else:
                # Not enough historical data, use theoretical model based on orderbook
                predicted_slippage = CryptoCalculations.calculate_theoretical_slippage(order_size, book_side)
                
            # Ensure slippage is at least slightly positive
            return max(0.001, predicted_slippage)
        except Exception as e:
            if log_message:
                log_message(f"⚠️ Slippage regression error: {str(e)}")
            # Fall back to a simple model scaled by order size
            base_slip = 0.01 * (order_size / 10000)
            return min(0.5, base_slip)  # Cap at 0.5% to prevent extreme values
        
    @staticmethod
    def calculate_theoretical_slippage(order_size, book_side):
        """
        Calculate theoretical slippage based on orderbook liquidity
        
        Args:
            order_size: Size of order in USD
            book_side: List of tuples (price, quantity) from orderbook
            
        Returns:
            Estimated slippage percentage
        """
        # Copy and ensure data types are float
        book = [(float(price), float(qty)) for price, qty in book_side]
        
        # Calculate price impact by walking the book
        executed_qty = 0
        notional_executed = 0
        initial_price = float(book[0][0])
        
        for price, qty in book:
            notional_qty = price * qty
            if executed_qty + qty >= order_size / price:
                # This level will partially fill
                remaining = order_size / price - executed_qty
                notional_executed += price * remaining
                executed_qty += remaining
                break
            else:
                # This level will be fully consumed
                executed_qty += qty
                notional_executed += notional_qty
        
        # If we couldn't fill the order with available liquidity
        if executed_qty == 0:
            return 0.5  # Default 0.5% slippage
        
        # Calculate average execution price
        avg_price = notional_executed / executed_qty
        
        # Calculate slippage
        slippage_pct = abs(avg_price - initial_price) / initial_price * 100
        
        return slippage_pct
    
    @staticmethod
    def calculate_fees(order_size, maker_prop, fee_tier, exchange):
        """
        Calculate expected fees using a rule-based model
        
        Args:
            order_size: Size of order in USD
            maker_prop: Proportion of order expected to be maker
            fee_tier: String representing fee tier (e.g., "0.1%")
            exchange: Exchange name (e.g., "binance")
            
        Returns:
            Expected fee in USD
        """
        # Get fee tier
        fee_str = fee_tier
        fee_tier = float(fee_str.strip('%')) / 100
        
        # Calculate maker and taker fees
        # Typically taker fees are higher than maker fees
        maker_fee_rate = fee_tier * 0.8  # 20% discount for maker orders
        taker_fee_rate = fee_tier * 1.2  # 20% premium for taker orders
        
        # Calculate fees
        maker_fee = order_size * maker_prop * maker_fee_rate
        taker_fee = order_size * (1 - maker_prop) * taker_fee_rate
        
        # Get exchange-specific adjustments
        exchange = exchange.lower()
        exchange_factor = 1.0
        if exchange == "binance":
            exchange_factor = 0.95  # Binance tends to have lower fees
        elif exchange == "coinbase":
            exchange_factor = 1.1  # Coinbase tends to have higher fees
            
        total_fee = (maker_fee + taker_fee) * exchange_factor
        
        # Large orders might get special treatment (volume discounts)
        if order_size > 500000:  # $500k+
            total_fee *= 0.85  # 15% volume discount
        elif order_size > 100000:  # $100k+
            total_fee *= 0.9  # 10% volume discount
            
        return round(total_fee, 4)
    
    @staticmethod
    def calculate_market_impact(order_size, mid_price, book_side, order_side, volatility, asset):
        """
        Calculate expected market impact using simplified Almgren-Chriss model
        
        Args:
            order_size: Size of order in USD
            mid_price: Current mid price
            book_side: List of tuples (price, quantity) from orderbook
            order_side: "buy" or "sell"
            volatility: "low", "medium", or "high"
            asset: Asset symbol (e.g., "BTC")
            
        Returns:
            Expected market impact percentage
        """
        # Constants for Almgren-Chriss model
        # These would normally be calibrated from market data
        # Simplified implementation based on the paper
        
        # Asset-specific parameters
        if asset == "BTC":
            sigma = 0.02  # 2% daily volatility for BTC
        elif asset == "ETH":
            sigma = 0.025  # 2.5% daily volatility for ETH
        else:
            sigma = 0.03  # 3% for other assets
            
        # Adjust for selected volatility
        if volatility == "low":
            sigma *= 0.7
        elif volatility == "high":
            sigma *= 1.5
            
        # Market depth parameter - derived from orderbook
        try:
            # Calculate liquidity within 0.5% of mid price
            price_range = mid_price * 0.005
            liquidity = sum(qty for price, qty in book_side 
                          if abs(float(price) - mid_price) < price_range)
            
            # Convert to market depth parameter
            market_depth = liquidity * mid_price / 1e6  # Normalize to millions
        except:
            # Default value if calculation fails
            market_depth = 5.0  # Default market depth parameter
            
        # Temporary market impact parameter (temporary price change per unit of order size)
        # Higher values mean lower liquidity
        temporary_impact = sigma / (market_depth ** 0.5) * 0.2
        
        # Permanent market impact parameter (lasting price change)
        permanent_impact = temporary_impact * 0.3  # Permanent impact is typically fraction of temporary
        
        # Calculate market impact percentage
        # Simplified Almgren-Chriss formula: MI = permanent * X + temporary * X * V
        # where X is normalized order size and V is execution speed factor
        
        # Normalize order size relative to market depth
        normalized_size = order_size / (market_depth * 1e6)
        
        # Execution speed factor - faster execution has more impact
        # Using a medium execution speed for simulation
        execution_speed = 1.0
        
        # Calculate total market impact
        impact_pct = (permanent_impact * normalized_size + 
                     temporary_impact * normalized_size * execution_speed) * 100
                     
        # Ensure we have a minimum impact
        impact_pct = max(0.001, impact_pct)
        
        return round(impact_pct, 4)
    
    @staticmethod
    def calculate_maker_taker_proportion(order_size, volatility, exchange, asset):
        """
        Calculate maker/taker proportion using logistic regression
        
        Args:
            order_size: Size of order in USD
            volatility: "low", "medium", or "high"
            exchange: Exchange name (e.g., "binance")
            asset: Asset symbol (e.g., "BTC")
            
        Returns:
            Tuple (maker_proportion, taker_proportion)
        """
        # Base taker probability based on order size using logistic function
        # p = 1 / (1 + e^(-k * (x - x0)))
        # where k is steepness, x is order size, x0 is midpoint
        
        # Parameters for logistic function
        k = 0.00001  # Steepness
        x0 = 50000   # Order size at which 50% is taker
        
        # Adjust parameters based on volatility
        if volatility == "low":
            k *= 0.7
            x0 *= 1.2
        elif volatility == "high":
            k *= 1.3
            x0 *= 0.8
            
        # Adjust for exchange-specific factors
        if exchange == "binance":
            k *= 0.9  # Binance has higher liquidity
        elif exchange == "coinbase":
            k *= 1.1  # Coinbase has lower liquidity
            
        # Calculate base taker probability
        base_taker_prob = 1 / (1 + np.exp(-k * (order_size - x0)))
        
        # Adjust for asset liquidity
        liquidity_factor = 1.0
        if asset == "BTC":
            liquidity_factor = 0.9  # BTC is more liquid
        elif asset == "SOL":
            liquidity_factor = 1.1  # SOL is less liquid
            
        # Final taker probability
        taker_prob = min(0.99, base_taker_prob * liquidity_factor)
        maker_prob = 1 - taker_prob
        
        return (maker_prob, taker_prob)
    
    @staticmethod
    def measure_internal_latency(exchange, asset):
        """
        Measure the internal latency of processing a tick
        
        Args:
            exchange: Exchange name (e.g., "binance")
            asset: Asset symbol (e.g., "BTC")
            
        Returns:
            Latency in milliseconds
        """
        # Start timer
        start_time = time.time()
        
        # Simple processing simulation
        array_size = 1000 * (1 + (exchange == "binance") * 0.2)
        temp_array = np.random.random((int(array_size), int(array_size)))
        _ = np.matmul(temp_array, temp_array.T)
        
        # Calculate time difference
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Add realistic noise
        latency += np.random.gamma(1.5, 0.3)
        
        return round(latency, 3)