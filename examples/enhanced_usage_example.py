#!/usr/bin/env python3
"""
Enhanced GARCH Trading Strategy Usage Example

This script demonstrates how to use the enhanced trading system with:
- ML ensemble models (GARCH + LSTM + XGBoost)
- Dynamic correlation-based risk management
- Real-time monitoring dashboard
- Advanced performance analytics
"""

import sys
import os
import asyncio
import threading
import time
from datetime import datetime, timedelta

# Add the project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

import pandas as pd
import numpy as np

# Import enhanced components
from models.ensemble_model import ensemble_manager
from models.lstm_volatility_model import lstm_manager
from models.xgboost_signal_model import xgboost_manager
from strategy.enhanced_garch_strategy import enhanced_strategy_manager
from execution.enhanced_risk_manager import enhanced_risk_manager
from execution.alpaca_executor import alpaca_executor
from monitoring.dashboard import start_monitoring_dashboard
from data.market_data import market_data_manager
from utils.config import config
from utils.logger import log_info, log_error


def generate_synthetic_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate synthetic market data for demonstration"""
    
    # Generate realistic market data using random walk with drift
    np.random.seed(42)  # For reproducible results
    
    # Parameters for realistic market simulation
    initial_price = 100.0
    drift = 0.0001  # Small positive drift
    volatility = 0.02  # 2% daily volatility
    
    # Generate minute-by-minute data
    minutes_per_day = 6.5 * 60  # 6.5 trading hours
    total_minutes = int(days * minutes_per_day)
    
    # Generate returns
    returns = np.random.normal(drift, volatility, total_minutes)
    
    # Add some autocorrelation to make it more realistic
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
    
    # Generate prices
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove initial price
    
    # Generate OHLC data
    data = []
    timestamps = []
    
    start_date = datetime.now() - timedelta(days=days)
    
    for i in range(total_minutes):
        timestamp = start_date + timedelta(minutes=i)
        
        # Skip weekends
        if timestamp.weekday() >= 5:
            continue
        
        # Skip non-trading hours (before 9:30 AM or after 4:00 PM)
        if timestamp.hour < 9 or (timestamp.hour == 9 and timestamp.minute < 30) or timestamp.hour >= 16:
            continue
        
        price = prices[i]
        
        # Generate OHLC with some realistic variation
        high = price * (1 + abs(np.random.normal(0, 0.001)))
        low = price * (1 - abs(np.random.normal(0, 0.001)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        
        # Ensure OHLC relationships are correct
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Generate volume (higher volume on larger price moves)
        base_volume = 100000
        volume_multiplier = 1 + abs(returns[i]) * 10
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
        timestamps.append(timestamp)
    
    df = pd.DataFrame(data, index=timestamps)
    return df


def example_1_ensemble_model_training():
    """Example 1: Train ensemble models"""
    
    print("=== Example 1: Enhanced Ensemble Model Training ===")
    
    # Generate synthetic data for multiple symbols
    symbols = ["SPY", "QQQ", "IWM"]
    data_dict = {}
    
    for symbol in symbols:
        print(f"Generating synthetic data for {symbol}...")
        data_dict[symbol] = generate_synthetic_market_data(symbol, days=90)
    
    # Train ensemble models for all symbols
    print("Training ensemble models...")
    fit_results = enhanced_strategy_manager.fit_all_ensemble_models(data_dict)
    
    for symbol, results in fit_results.items():
        print(f"\nModel training results for {symbol}:")
        for model_type, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {model_type.upper()}: {status}")
    
    print("\n" + "="*60)


def example_2_enhanced_signal_generation():
    """Example 2: Generate enhanced trading signals"""
    
    print("=== Example 2: Enhanced Signal Generation ===")
    
    # Use SPY as example
    symbol = "SPY"
    strategy = enhanced_strategy_manager.get_strategy(symbol)
    
    # Generate recent market data
    print(f"Generating market data for {symbol}...")
    recent_data = generate_synthetic_market_data(symbol, days=30)
    
    # Update strategy with market data
    print("Updating strategy with market data...")
    from data.market_data import MarketDataPoint
    
    for timestamp, row in recent_data.iterrows():
        data_point = MarketDataPoint(
            symbol=symbol,
            timestamp=timestamp,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timeframe='1Min'
        )
        strategy.update_market_data(data_point)
    
    # Generate enhanced signal
    current_price = recent_data['close'].iloc[-1]
    print(f"Current price for {symbol}: ${current_price:.2f}")
    
    enhanced_signal = strategy.generate_enhanced_signal(current_price)
    
    if enhanced_signal:
        print(f"\n📊 Enhanced Signal Generated for {symbol}:")
        print(f"  Signal Type: {enhanced_signal.signal_type.value}")
        print(f"  Signal Strength: {enhanced_signal.strength:.3f}")
        print(f"  Confidence: {enhanced_signal.confidence:.3f}")
        print(f"  Quality Score: {enhanced_signal.signal_quality_score:.3f}")
        print(f"  Model Agreement: {enhanced_signal.model_agreement:.3f}")
        print(f"  Predicted Volatility: {enhanced_signal.predicted_volatility:.4f}")
        print(f"  Market Regime: {enhanced_signal.regime_context['current_regime']}")
        print(f"  Risk-Adjusted Strength: {enhanced_signal.risk_adjusted_strength:.3f}")
        print(f"  Reasoning: {enhanced_signal.reasoning}")
        
        if enhanced_signal.ensemble_signal:
            print(f"\n🔬 Ensemble Model Breakdown:")
            ens = enhanced_signal.ensemble_signal
            if ens.garch_prediction:
                print(f"  GARCH Volatility: {ens.garch_prediction.predicted_volatility:.4f}")
            if ens.lstm_prediction:
                print(f"  LSTM Volatility: {ens.lstm_prediction.predicted_volatility:.4f}")
            if ens.xgboost_signal:
                print(f"  XGBoost Signal: {ens.xgboost_signal.signal_type} "
                     f"(prob: {ens.xgboost_signal.signal_probability:.3f})")
    else:
        print("❌ No enhanced signal generated")
    
    print("\n" + "="*60)


def example_3_advanced_risk_management():
    """Example 3: Advanced risk management with correlation analysis"""
    
    print("=== Example 3: Advanced Risk Management ===")
    
    # Simulate portfolio positions
    mock_positions = {
        'SPY': type('PositionInfo', (), {
            'symbol': 'SPY',
            'quantity': 100,
            'market_value': 45000,
            'cost_basis': 44000,
            'unrealized_pnl': 1000,
            'unrealized_pnl_pct': 0.0227,
            'current_price': 450.0,
            'entry_time': datetime.now() - timedelta(hours=2)
        })(),
        'QQQ': type('PositionInfo', (), {
            'symbol': 'QQQ',
            'quantity': 150,
            'market_value': 52500,
            'cost_basis': 51000,
            'unrealized_pnl': 1500,
            'unrealized_pnl_pct': 0.0294,
            'current_price': 350.0,
            'entry_time': datetime.now() - timedelta(hours=1)
        })()
    }
    
    portfolio_value = 150000
    
    # Calculate enhanced risk metrics
    print("Calculating enhanced risk metrics...")
    risk_metrics = enhanced_risk_manager.calculate_enhanced_risk_metrics(
        portfolio_value, mock_positions
    )
    
    print(f"\n📈 Enhanced Risk Metrics:")
    print(f"  Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Total Exposure: ${risk_metrics.total_exposure:,.2f}")
    print(f"  Risk Level: {risk_metrics.risk_level.value}")
    print(f"  Daily P&L: {risk_metrics.daily_pnl:.3%}")
    print(f"  Max Drawdown: {risk_metrics.max_drawdown:.3%}")
    print(f"  Portfolio Volatility: {risk_metrics.portfolio_volatility:.3%}")
    print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.3f}")
    
    print(f"\n🔗 Correlation Analysis:")
    print(f"  Max Correlation: {risk_metrics.correlation_metrics.max_correlation:.3f}")
    print(f"  Diversification Ratio: {risk_metrics.correlation_metrics.diversification_ratio:.3f}")
    print(f"  Effective Positions: {risk_metrics.correlation_metrics.effective_positions:.1f}")
    print(f"  Concentration Risk: {risk_metrics.correlation_metrics.concentration_risk:.3f}")
    
    print(f"\n⚡ Stress Test Results:")
    for scenario, result in risk_metrics.stress_test_results.items():
        print(f"  {scenario.replace('_', ' ').title()}: {result:.2%}")
    
    print(f"\n💧 Liquidity Metrics:")
    for metric, value in risk_metrics.liquidity_metrics.items():
        if isinstance(value, float):
            print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    # Test dynamic position sizing
    print(f"\n🎯 Dynamic Position Sizing Example:")
    
    # Create a mock trading signal
    from strategy.garch_strategy import TradingSignal, SignalType
    mock_signal = TradingSignal(
        symbol="TSLA",
        signal_type=SignalType.BUY,
        strength=0.75,
        confidence=0.80,
        timestamp=datetime.now(),
        price=250.0,
        predicted_volatility=0.035,
        prediction_premium=0.025,
        technical_indicators={},
        reasoning="High conviction buy signal"
    )
    
    optimal_size = enhanced_risk_manager.calculate_dynamic_position_size(
        mock_signal, mock_positions, portfolio_value
    )
    
    print(f"  Signal: {mock_signal.symbol} {mock_signal.signal_type.value}")
    print(f"  Signal Strength: {mock_signal.strength:.3f}")
    print(f"  Predicted Volatility: {mock_signal.predicted_volatility:.3f}")
    print(f"  Optimal Position Size: {optimal_size:.0f} shares")
    print(f"  Position Value: ${optimal_size * mock_signal.price:,.2f}")
    print(f"  Portfolio Allocation: {(optimal_size * mock_signal.price) / portfolio_value:.2%}")
    
    print("\n" + "="*60)


def example_4_performance_analytics():
    """Example 4: Performance analytics and reporting"""
    
    print("=== Example 4: Performance Analytics ===")
    
    # Get enhanced performance summary
    performance_summary = enhanced_strategy_manager.get_enhanced_performance_summary()
    
    print("📊 Strategy Performance Summary:")
    for symbol, metrics in performance_summary.items():
        print(f"\n  {symbol}:")
        print(f"    Total Signals: {metrics['total_signals']}")
        print(f"    Success Rate: {metrics['success_rate']:.2%}")
        print(f"    Current Regime: {metrics['current_regime']}")
        
        if 'average_signal_quality' in metrics:
            print(f"    Avg Signal Quality: {metrics['average_signal_quality']:.3f}")
            print(f"    Avg Model Agreement: {metrics['average_model_agreement']:.3f}")
        
        if 'regime_performance' in metrics:
            print(f"    Regime Performance:")
            for regime, stats in metrics['regime_performance'].items():
                print(f"      {regime}: {stats['success_rate']:.2%} ({stats['signals']} signals)")
    
    print("\n" + "="*60)


def example_5_monitoring_dashboard():
    """Example 5: Real-time monitoring dashboard"""
    
    print("=== Example 5: Real-time Monitoring Dashboard ===")
    print("Starting monitoring dashboard on http://localhost:8050")
    print("Note: This would normally run the web dashboard")
    print("To actually run the dashboard, execute:")
    print("  python src/monitoring/dashboard.py")
    print("\nDashboard features include:")
    print("  📈 Real-time portfolio performance")
    print("  ⚠️  Risk metrics and alerts")
    print("  🔍 Signal analysis and history")
    print("  🤖 Model performance comparison")
    print("  ⚙️  System health monitoring")
    
    print("\n" + "="*60)


def run_comprehensive_example():
    """Run comprehensive example of enhanced system"""
    
    print("🚀 Enhanced GARCH Trading Strategy - Comprehensive Demo")
    print("=" * 70)
    print()
    
    try:
        # Run all examples
        example_1_ensemble_model_training()
        example_2_enhanced_signal_generation()
        example_3_advanced_risk_management()
        example_4_performance_analytics()
        example_5_monitoring_dashboard()
        
        print("✅ All examples completed successfully!")
        print("\n🎯 Key Enhancements Demonstrated:")
        print("   • ML Ensemble Models (GARCH + LSTM + XGBoost)")
        print("   • Dynamic Correlation-Based Risk Management")
        print("   • Market Regime Detection and Adaptation")
        print("   • Advanced Signal Quality Scoring")
        print("   • Real-time Performance Monitoring")
        print("   • Comprehensive Risk Analytics")
        
        print("\n📋 Next Steps:")
        print("   1. Configure API credentials for live data")
        print("   2. Adjust risk parameters in config.yaml")
        print("   3. Run paper trading to validate performance")
        print("   4. Deploy with monitoring dashboard")
        print("   5. Monitor and optimize model performance")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_example()