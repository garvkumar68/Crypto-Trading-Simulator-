# Crypto-Trading-Simulator

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

A real-time cryptocurrency trading simulator with:
- Live orderbook data via WebSocket
- Advanced cost modeling (slippage, fees, market impact)
- Interactive Tkinter GUI
- Offline backtesting mode

![Screenshot](docs/screenshot.png) *Example: Slippage analysis dashboard*

## Features

✅ **Real-time Market Data**  
- Connects to OKX/Binance-style WebSocket APIs
- Processes L2 orderbook updates at 100+ msg/sec

📊 **Advanced Trading Models**  
- Hybrid slippage estimation (Quantile Regression + Orderbook Walk)
- Almgren-Chriss market impact modeling
- Dynamic fee calculation with volume discounts

💻 **Interactive GUI**  
- Live orderbook visualization
- Slippage vs. order size plotting
- Export results to CSV

🔌 **Offline Mode**  
- Synthetic data generation for backtesting
- Configurable volatility parameters

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crypto-trading-simulator.git
   cd crypto-trading-simulator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Live Mode
```bash
python app.py
```
- Configure exchange/asset in UI
- Adjust order parameters in real-time


### Key Controls
| Button          | Function                          |
|-----------------|-----------------------------------|
| Connect         | Start WebSocket connection        |
| Plot Slippage   | Generate slippage analysis graph  |
| Save Results    | Export to CSV                     |

## File Structure
```
.
├── app.py                      # Main application entry point
├── CryptoCalculations.py       # Core trading models
├── DataGenerator.py            # Synthetic orderbook generator
├── OfflineCryptoTrading.py     # Offline mode implementation
├── ui.py                       # Tkinter GUI implementation
├── requirements.txt            # Dependencies
└── docs/
    └── screenshot.png          # Application screenshot
```

## Models Implemented

### 1. Slippage Estimation
```python
if len(history) > 5:
    model = QuantileRegressor(quantile=0.75)  # Conservative estimate
else:
    slippage = walk_orderbook(order_size, bids)  # Theoretical fallback
```

### 2. Fee Calculation
```python
# Volume discounts
if order_size > 500000:
    fee *= 0.85  # 15% discount
```

### 3. Market Impact (Almgren-Chriss)
```python
impact = (permanent_impact * X) + (temporary_impact * X * V)
```

