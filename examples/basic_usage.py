#!/usr/bin/env python3
"""
Basic Usage Example for GARCH Intraday Trading Strategy

This script demonstrates how to use the trading system components
individually for research and testing purposes.
"""

import sys
import os

# Add the project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Also add src directory
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Import core components
from data.market_data import market_data_manager
from models.garch_model import garch_manager
from strategy.garch_strategy import strategy_manager
from execution.alpaca_executor import alpaca_executor
from execution.risk_manager import risk_manager
from utils.config import config
from utils.logger import log_info, log_error


def example_1_data_retrieval():
    """Example: Retrieve and analyze market data"""
    
    print("=== Example 1: Market Data Retrieval ===")
    
    # Get historical data for SPY
    symbol = "SPY"
    df = market_data_manager.get_historical_data(symbol, timeframe='1Min', days_back=30)
    
    if not df.empty:
        print(f"Retrieved {len(df)} data points for {symbol}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"Average volume: {df['volume'].mean():.0f}")
        
        # Calculate basic statistics
        returns = market_data_manager.calculate_returns(df)
        print(f"Daily return statistics:")
        print(f"  Mean: {returns.mean():.4f}")
        print(f"  Std: {returns.std():.4f}")
        print(f"  Min: {returns.min():.4f}")
        print(f"  Max: {returns.max():.4f}")
    else:
        print("No data retrieved")
    
    print()


def example_2_garch_modeling():
    """Example: Fit GARCH model and generate predictions"""
    
    print("=== Example 2: GARCH Model Fitting ===")
    
    # Get market data
    symbol = "SPY"
    df = market_data_manager.get_historical_data(symbol, timeframe='1Min', days_back=60)
    
    if df.empty:
        print("No data available for GARCH modeling")
        return
    
    # Calculate returns
    returns = market_data_manager.calculate_returns(df)
    
    # Get GARCH model
    garch_model = garch_manager.get_model(symbol)
    
    # Fit model
    print(f"Fitting GARCH model for {symbol}...")
    success = garch_model.fit(returns)
    
    if success:
        print("Model fitted successfully!")
        
        # Generate prediction
        prediction = garch_model.predict()
        
        if prediction:
            print(f"Volatility forecast: {prediction.predicted_volatility:.4f}")
            print(f"Confidence interval: {prediction.confidence_interval}")
            print(f"Model AIC: {prediction.aic:.2f}")
            print(f"Model BIC: {prediction.bic:.2f}")
            
            # Calculate prediction premium
            current_price = df['close'].iloc[-1]
            premium = garch_model.calculate_prediction_premium(
                current_price, prediction.predicted_volatility
            )
            print(f"Prediction premium: {premium:.4f}")
        else:
            print("Failed to generate prediction")
    else:
        print("Failed to fit GARCH model")
    
    print()


def example_3_signal_generation():
    """Example: Generate trading signals"""
    
    print("=== Example 3: Signal Generation ===")
    
    # Get strategy for SPY
    symbol = "SPY"
    strategy = strategy_manager.get_strategy(symbol)
    
    # Get current market data
    df = market_data_manager.get_historical_data(symbol, timeframe='1Min', days_back=30)
    
    if df.empty:
        print("No data available for signal generation")
        return
    
    # Update strategy with historical data
    print(f"Updating strategy with {len(df)} data points...")
    
    from data.market_data import MarketDataPoint
    
    for idx, row in df.iterrows():
        data_point = MarketDataPoint(
            symbol=symbol,
            timestamp=idx,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timeframe='1Min'
        )
        strategy.update_market_data(data_point)
    
    # Generate signal
    current_price = df['close'].iloc[-1]
    signal = strategy.generate_signal(current_price)
    
    if signal:
        print(f"Generated signal for {symbol}:")
        print(f"  Signal type: {signal.signal_type.value}")
        print(f"  Strength: {signal.strength:.3f}")
        print(f"  Confidence: {signal.confidence:.3f}")
        print(f"  Predicted volatility: {signal.predicted_volatility:.4f}")
        print(f"  Prediction premium: {signal.prediction_premium:.4f}")
        print(f"  Reasoning: {signal.reasoning}")
    else:
        print("No signal generated")
    
    print()


def example_4_risk_analysis():
    """Example: Risk analysis and position sizing"""
    
    print("=== Example 4: Risk Analysis ===")
    
    # Mock portfolio data
    mock_positions = {
        'SPY': {
            'quantity': 100,
            'market_value': 45000,
            'cost_basis': 44000,
            'unrealized_pnl': 1000,
            'unrealized_pnl_pct': 0.0227,
            'current_price': 450.0,
            'entry_time': datetime.now() - timedelta(hours=2)
        }
    }
    
    # Update risk manager with portfolio value
    portfolio_value = 100000
    risk_manager.update_portfolio_value(portfolio_value)
    
    # Calculate risk metrics
    from execution.risk_manager import PositionInfo
    
    positions = {
        symbol: PositionInfo(
            symbol=symbol,
            quantity=pos['quantity'],
            market_value=pos['market_value'],
            cost_basis=pos['cost_basis'],
            unrealized_pnl=pos['unrealized_pnl'],
            unrealized_pnl_pct=pos['unrealized_pnl_pct'],
            current_price=pos['current_price'],
            entry_time=pos['entry_time']
        )
        for symbol, pos in mock_positions.items()
    }
    
    risk_metrics = risk_manager.calculate_risk_metrics(portfolio_value, positions)
    
    print("Risk Metrics:")
    print(f"  Total exposure: ${risk_metrics.total_exposure:.2f}")
    print(f"  Max position size: ${risk_metrics.max_position_size:.2f}")
    print(f"  Concentration ratio: {risk_metrics.concentration_ratio:.3f}")
    print(f"  Daily P&L: {risk_metrics.daily_pnl:.3f}")
    print(f"  Max drawdown: {risk_metrics.max_drawdown:.3f}")
    print(f"  Portfolio volatility: {risk_metrics.portfolio_volatility:.3f}")
    print(f"  Sharpe ratio: {risk_metrics.sharpe_ratio:.3f}")
    print(f"  Risk level: {risk_metrics.risk_level.value}")
    
    print()


def example_5_portfolio_summary():
    """Example: Get portfolio summary from Alpaca"""
    
    print("=== Example 5: Portfolio Summary ===")
    
    try:
        # Get portfolio summary
        portfolio = alpaca_executor.get_portfolio_summary()
        
        print("Portfolio Summary:")
        print(f"  Account value: ${portfolio['account_value']:.2f}")
        print(f"  Buying power: ${portfolio['buying_power']:.2f}")
        print(f"  Positions count: {portfolio['positions_count']}")
        print(f"  Active orders: {portfolio['active_orders']}")
        print(f"  Total unrealized P&L: ${portfolio['total_unrealized_pnl']:.2f}")
        print(f"  Paper trading: {portfolio['is_paper_trading']}")
        
        # Show positions
        if portfolio['positions']:
            print("\nCurrent Positions:")
            for symbol, position in portfolio['positions'].items():
                print(f"  {symbol}: {position['quantity']} shares, "
                      f"${position['market_value']:.2f} value, "
                      f"${position['unrealized_pnl']:.2f} P&L")
        else:
            print("\nNo current positions")
        
    except Exception as e:
        print(f"Error getting portfolio summary: {e}")
        print("Make sure you have valid Alpaca API credentials configured")
    
    print()


def example_6_strategy_performance():
    """Example: Get strategy performance metrics"""
    
    print("=== Example 6: Strategy Performance ===")
    
    # Get performance summary for all strategies
    performance = strategy_manager.get_performance_summary()
    
    for symbol, metrics in performance.items():
        print(f"Strategy Performance for {symbol}:")
        print(f"  Total signals: {metrics['total_signals']}")
        print(f"  Successful signals: {metrics.get('successful_signals', 0)}")
        print(f"  Success rate: {metrics.get('success_rate', 0.0):.2%}")
        print(f"  Average strength: {metrics.get('average_strength', 0.0):.3f}")
        print(f"  Average confidence: {metrics.get('average_confidence', 0.0):.3f}")
        print(f"  Signal distribution: {metrics.get('signal_distribution', {})}")
        print()


def run_all_examples():
    """Run all examples in sequence"""
    
    print("🚀 GARCH Intraday Trading Strategy - Usage Examples")
    print("=" * 60)
    print()
    
    try:
        # Run examples
        example_1_data_retrieval()
        example_2_garch_modeling()
        example_3_signal_generation()
        example_4_risk_analysis()
        example_5_portfolio_summary()
        example_6_strategy_performance()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run all examples
    run_all_examples()