# GARCH Intraday Trading Strategy

A fully functional, production-ready algorithmic trading bot that uses GARCH models for volatility forecasting to execute profitable intraday trading strategies.

## 🚀 Features

### Core Functionality
- **GARCH Model Integration**: Advanced volatility forecasting using GARCH, GJR-GARCH, and EGARCH models
- **Real-time Market Data**: Live market data streaming via Alpaca API
- **Automated Trading**: Fully automated signal generation and order execution
- **Risk Management**: Comprehensive risk controls with position sizing and stop-loss mechanisms
- **Backtesting**: Historical strategy validation and performance analysis

### Technical Highlights
- **Production-Ready**: Robust error handling, logging, and monitoring
- **Scalable Architecture**: Modular design supporting multiple symbols and strategies
- **Real-time Processing**: Sub-second latency for signal generation and execution
- **Advanced Analytics**: Technical indicators, correlation analysis, and performance metrics
- **Future-Proof**: Modern Python architecture with async support

## 📊 Strategy Overview

The strategy combines GARCH volatility forecasting with technical analysis to identify profitable intraday trading opportunities:

1. **Volatility Forecasting**: GARCH models predict next-period volatility
2. **Signal Generation**: Combines volatility predictions with technical indicators
3. **Risk Assessment**: Multi-layer risk management and position sizing
4. **Order Execution**: Automated order management via Alpaca API
5. **Performance Monitoring**: Real-time performance tracking and alerting

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Alpaca trading account (paper or live)
- Required Python packages (see requirements.txt)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Intraday-strategy-using-GARCH-Model-main
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp config/.env.example config/.env
   # Edit .env file with your API credentials
   ```

5. **Configure trading parameters**
   ```bash
   # Edit config/config.yaml for strategy parameters
   ```

## ⚙️ Configuration

### API Credentials
Add your Alpaca API credentials to `config/.env`:
```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
```

### Strategy Parameters
Edit `config/config.yaml` to customize:
- Trading symbols and watchlist
- GARCH model parameters
- Risk management limits
- Position sizing methods
- Trading hours and timeframes

### Key Configuration Options

```yaml
# Environment
environment: "paper"  # or "live"

# Strategy
strategy:
  volatility_target: 0.15
  max_position_size: 0.10
  position_sizing_method: "volatility_target"

# Risk Management
risk:
  max_daily_loss: 0.02
  max_drawdown: 0.10
  stop_loss_pct: 0.015
  take_profit_pct: 0.03

# GARCH Model
garch:
  model_type: "GARCH"
  p: 1
  q: 1
  rolling_window: 250
```

## 🚀 Usage

### Running the Trading Bot

1. **Start the trading engine**
   ```bash
   python src/main.py
   ```

2. **Monitor performance**
   - Check logs in `logs/` directory
   - View real-time metrics in console output
   - Monitor positions via Alpaca dashboard

### Command Line Options

```bash
# Run with custom config
python src/main.py --config config/custom_config.yaml

# Run in paper trading mode
python src/main.py --paper

# Run with debug logging
python src/main.py --debug
```

## 📈 Performance Analysis

### Backtesting

Run historical backtests to validate strategy performance:

```python
from src.backtest.backtest_engine import BacktestEngine

# Create backtest engine
backtest = BacktestEngine()

# Run backtest
results = backtest.run(
    start_date="2023-01-01",
    end_date="2024-01-01",
    symbols=["SPY", "QQQ"],
    initial_capital=100000
)

# View results
print(results.get_summary())
```

### Key Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Volatility**: Strategy volatility vs benchmark
- **Alpha/Beta**: Risk-adjusted performance metrics

## 🔧 Architecture

### Core Components

1. **Data Pipeline** (`src/data/`)
   - Real-time market data streaming
   - Historical data management
   - Data validation and cleaning

2. **GARCH Models** (`src/models/`)
   - Volatility forecasting
   - Model fitting and validation
   - Rolling window predictions

3. **Trading Strategy** (`src/strategy/`)
   - Signal generation
   - Technical indicators
   - Strategy logic

4. **Execution Engine** (`src/execution/`)
   - Order management
   - Risk controls
   - Position tracking

5. **Risk Management** (`src/execution/risk_manager.py`)
   - Position sizing
   - Stop-loss/take-profit
   - Drawdown protection

### System Flow

```
Market Data → GARCH Model → Signal Generation → Risk Check → Order Execution → Position Management
```

## 🛡️ Risk Management

### Multi-Layer Risk Controls

1. **Position Level**
   - Dynamic stop-loss based on volatility
   - Take-profit targets
   - Position timeout

2. **Portfolio Level**
   - Daily loss limits
   - Maximum drawdown protection
   - Concentration limits

3. **System Level**
   - Emergency stop mechanisms
   - Connection monitoring
   - Error handling

### Risk Metrics Monitoring

- **Value at Risk (VaR)**: 95% and 99% confidence intervals
- **Portfolio Volatility**: Real-time volatility tracking
- **Correlation Analysis**: Position correlation monitoring
- **Liquidity Risk**: Market impact assessment

## 📊 Monitoring & Alerting

### Real-time Monitoring

- **Trading Dashboard**: Live performance metrics
- **Risk Alerts**: Automated risk event notifications
- **System Health**: Component status monitoring
- **Performance Tracking**: Real-time P&L and metrics

### Logging System

- **Structured Logging**: JSON-formatted logs for analysis
- **Log Rotation**: Automated log file management
- **Error Tracking**: Comprehensive error logging
- **Audit Trail**: Complete trading activity history

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/unit/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Performance Tests
```bash
python -m pytest tests/performance/
```

## 📚 Documentation

### API Documentation
- **Strategy API**: Custom strategy development
- **Risk API**: Risk management functions
- **Data API**: Market data integration

### Model Documentation
- **GARCH Implementation**: Technical details
- **Signal Generation**: Strategy logic
- **Risk Calculations**: Risk metric formulas

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Issues**
   - Check API credentials
   - Verify network connectivity
   - Check Alpaca service status

2. **Model Fitting Errors**
   - Ensure sufficient historical data
   - Check data quality
   - Validate model parameters

3. **Performance Issues**
   - Monitor system resources
   - Check data feed latency
   - Optimize model parameters

### Debug Mode

Enable debug logging for detailed troubleshooting:
```bash
python src/main.py --debug
```

## 🚨 Important Disclaimers

### Risk Warning
- **Trading involves significant risk of loss**
- **Past performance does not guarantee future results**
- **Use paper trading for testing before live deployment**
- **Never risk more than you can afford to lose**

### Regulatory Compliance
- Ensure compliance with local financial regulations
- Understand tax implications of algorithmic trading
- Consider registration requirements for automated trading

## 📞 Support

### Getting Help
- Review documentation and examples
- Check existing issues and solutions
- Join community discussions

### Contributing
- Fork the repository
- Create feature branches
- Submit pull requests with tests
- Follow coding standards

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Alpaca API**: Market data and execution services
- **ARCH Library**: GARCH model implementation
- **Python Community**: Open source libraries and tools

---

**⚠️ IMPORTANT**: This software is for educational and research purposes. Always test thoroughly in paper trading before using real money. Trading involves substantial risk of loss.

## 🔗 Quick Links

- [Installation Guide](#installation)
- [Configuration Reference](#configuration)
- [API Documentation](docs/api.md)
- [Strategy Development](docs/strategy.md)
- [Risk Management](docs/risk.md)
- [Performance Analysis](docs/performance.md)

---

*Built with ❤️ for algorithmic trading enthusiasts*