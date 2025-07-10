# MT5 GARCH Trading Strategy - Installation & Setup Guide

## Overview

This guide will help you set up and run the GARCH intraday trading strategy with MetaTrader 5 (MT5). The system has been specifically adapted to work exclusively with MT5, removing all dependencies on yfinance and other external data providers.

## 🚀 Quick Start

### Prerequisites

1. **MetaTrader 5 Terminal** (Windows/Wine on Linux)
2. **Python 3.9+** with pip
3. **MT5 Trading Account** (demo or live)
4. **System Requirements**: 4GB RAM, 1GB disk space

### Installation Steps

#### 1. Download and Install MetaTrader 5

**Windows:**
- Download MT5 from your broker's website
- Install MT5 terminal (usually in `C:\Program Files\MetaTrader 5\`)
- Create and verify your trading account

**Linux (via Wine):**
```bash
# Install Wine
sudo apt update
sudo apt install wine

# Download and install MT5 through Wine
winecfg  # Configure Wine
# Download MT5 installer and run with Wine
wine MT5Setup.exe
```

#### 2. Install Python Dependencies

```bash
# Clone or extract the project
cd Intraday-strategy-using-GARCH-Model-main

# Install minimal MT5 dependencies
pip install -r requirements_mt5.txt

# Verify installation
python3 -c "import numpy, pandas, arch; print('✅ Core dependencies installed!')"
```

#### 3. Configure MT5 Settings

Edit `config/config.yaml`:

```yaml
# Broker Configuration
broker:
  name: "mt5"
  mt5_path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"  # Windows path
  login: YOUR_ACCOUNT_NUMBER  # Replace with your MT5 account
  server: "YOUR_BROKER_SERVER"  # e.g., "MetaQuotes-Demo"
  password: "YOUR_PASSWORD"
  timeout: 60000

# Symbol Configuration (Major Forex Pairs)
symbols:
  primary: "EURUSD"
  watchlist: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

# MT5 Position Sizing
mt5_position_sizing:
  lot_size: 0.1          # Standard lot size
  max_lot_size: 1.0      # Maximum position size
  min_lot_size: 0.01     # Minimum position size
  risk_per_trade: 0.02   # 2% risk per trade
  use_fixed_lot: false   # Use dynamic sizing
```

#### 4. Test the Installation

```bash
# Run integration tests
python3 test_mt5_integration.py

# Run core functionality tests
python3 test_core_functionality.py

# Expected output: All tests should pass ✅
```

## 📊 Trading Configuration

### Risk Management Settings

```yaml
# Risk Management (in config.yaml)
risk:
  max_daily_loss: 0.02      # 2% daily loss limit
  max_drawdown: 0.10        # 10% maximum drawdown
  stop_loss_pct: 0.015      # 1.5% stop loss
  take_profit_pct: 0.03     # 3% take profit
  max_positions: 3          # Maximum concurrent positions
```

### GARCH Model Configuration

```yaml
# GARCH Model Settings
garch:
  model_type: "GARCH"       # GARCH, GJR-GARCH, or EGARCH
  p: 1                      # GARCH order
  q: 1                      # ARCH order
  rolling_window: 250       # Historical data window
  refit_frequency: 20       # Refit model every N periods
```

### Trading Hours (24/5 Forex)

```yaml
# Trading Hours
trading_hours:
  market_open: "00:00"      # Forex opens Sunday 5 PM EST
  market_close: "23:59"     # Forex closes Friday 5 PM EST
  timezone: "UTC"
  avoid_weekend: true       # No weekend trading
  avoid_news_events: true   # Avoid high-impact news
```

## 🔧 Running the Strategy

### Start Trading

```bash
# Start the GARCH trading engine
python3 src/main.py

# For debug mode
python3 src/main.py --debug

# Expected output:
# [INFO] Trading engine initialized
# [INFO] Connected to MT5 - Account: XXX, Balance: $X,XXX.XX
# [INFO] GARCH models fitted successfully
# [INFO] Trading engine started successfully
```

### Monitor Performance

```bash
# Check logs
tail -f logs/app_$(date +%Y-%m-%d).log

# View trading activity
tail -f logs/trading_$(date +%Y-%m-%d).log

# Monitor risk events
tail -f logs/risk_$(date +%Y-%m-%d).log
```

## 🎯 Strategy Overview

### How It Works

1. **Data Collection**: Real-time forex data from MT5
2. **GARCH Modeling**: Volatility forecasting using GARCH(1,1)
3. **Signal Generation**: Combines GARCH predictions with technical indicators
4. **Risk Management**: Dynamic position sizing and stop-loss placement
5. **Order Execution**: Automated trading through MT5 API

### Signal Logic

```python
# Core trading algorithm
if predicted_volatility > 2%:
    signal = "SELL"  # High volatility = reversal strategy
elif predicted_volatility < 1%:
    signal = "BUY"   # Low volatility = trend continuation
else:
    signal = "HOLD"  # Medium volatility = wait
```

### Position Sizing

```python
# Dynamic lot calculation
lot_size = (account_balance * risk_per_trade) / (stop_loss_pips * pip_value)
lot_size = max(min_lot, min(lot_size, max_lot)) * signal_strength
```

## 🛡️ Risk Management Features

### Multi-Layer Protection

1. **Position Level**
   - Dynamic stop-loss based on volatility
   - Take-profit targets (2:1 risk/reward)
   - Position timeout (4 hours max)

2. **Portfolio Level**
   - Daily loss limits (2% default)
   - Maximum drawdown protection (10% default)
   - Concentration limits (max 3 positions)

3. **System Level**
   - Emergency stop mechanisms
   - Connection monitoring
   - Robust error handling

### Real-Time Monitoring

```bash
# Key metrics logged every 10 loops:
# Portfolio Value: $10,000.00 | Daily P&L: +1.2% | Positions: 2 | Trades: 15
```

## 🔍 Troubleshooting

### Common Issues

**1. MT5 Connection Failed**
```
Error: Failed to initialize MT5 connection
Solution: Check MT5 terminal is running and account credentials are correct
```

**2. Missing Dependencies**
```
Error: ModuleNotFoundError: No module named 'MetaTrader5'
Solution: The system uses a mock MT5 module for development. Real MT5 requires Windows.
```

**3. GARCH Model Fitting Errors**
```
Error: Model convergence failed
Solution: Increase historical data or adjust model parameters
```

**4. No Trading Signals**
```
Issue: Strategy generates only HOLD signals
Solution: Market may be in ranging mode. Check volatility thresholds.
```

### Debug Mode

```bash
# Enable detailed logging
python3 src/main.py --debug

# Check specific components
python3 -c "
from src.execution.mt5_executor import MT5Executor
executor = MT5Executor()
print(f'Connected: {executor.connected}')
print(f'Balance: {executor.get_portfolio_summary()}')
"
```

### Log Analysis

```bash
# Check for errors
grep "ERROR" logs/app_$(date +%Y-%m-%d).log

# Monitor trade execution
grep "TRADE" logs/trading_$(date +%Y-%m-%d).log

# Check GARCH predictions
grep "GARCH" logs/garch_$(date +%Y-%m-%d).log
```

## 📈 Performance Optimization

### Fine-Tuning Parameters

1. **GARCH Model**
   - Increase `rolling_window` for more stable predictions
   - Adjust `refit_frequency` based on market conditions
   - Use EGARCH for asymmetric volatility effects

2. **Signal Generation**
   - Modify volatility thresholds (1%, 2% default)
   - Adjust technical indicator periods
   - Fine-tune signal strength weights

3. **Risk Management**
   - Optimize stop-loss percentages
   - Adjust position sizing methods
   - Modify maximum position limits

### Backtesting

```bash
# Run historical backtest (implementation depends on data availability)
python3 -c "
from src.backtest.backtest_engine import BacktestEngine
engine = BacktestEngine()
results = engine.run(
    start_date='2023-01-01',
    end_date='2024-01-01',
    symbols=['EURUSD'],
    initial_capital=10000
)
print(results.get_summary())
"
```

## 🚨 Important Disclaimers

### Risk Warnings

- **Trading involves significant risk of loss**
- **Past performance does not guarantee future results**
- **Use demo accounts for testing before live deployment**
- **Never risk more than you can afford to lose**

### Regulatory Compliance

- Ensure compliance with local financial regulations
- Understand tax implications of algorithmic trading
- Consider registration requirements for automated trading

### Technical Limitations

- Requires stable internet connection
- MT5 terminal must remain running
- System resources needed for real-time processing

## 📞 Support & Resources

### Getting Help

1. **Check Logs**: Always start with log file analysis
2. **Test Components**: Use integration tests to isolate issues
3. **Configuration**: Verify all settings in `config.yaml`
4. **Demo Account**: Always test with demo before live trading

### Performance Monitoring

```bash
# System health check
python3 -c "
from src.main import TradingEngine
engine = TradingEngine()
status = engine.get_status()
print(f'System Status: {status}')
"
```

### Best Practices

1. **Start Small**: Begin with minimum lot sizes
2. **Monitor Closely**: Watch first few days of trading
3. **Keep Records**: Maintain detailed trading logs
4. **Regular Updates**: Update parameters based on performance
5. **Emergency Stops**: Know how to stop the system quickly

---

## ✅ Quick Checklist

Before starting live trading:

- [ ] MT5 terminal installed and configured
- [ ] All Python dependencies installed
- [ ] Configuration file updated with account details
- [ ] Integration tests pass
- [ ] Demo account tested successfully
- [ ] Risk parameters reviewed and approved
- [ ] Emergency stop procedures understood
- [ ] Monitoring system in place

---

**Happy Trading! 📊📈**

*Remember: This system is for educational and research purposes. Always test thoroughly before using real money.*