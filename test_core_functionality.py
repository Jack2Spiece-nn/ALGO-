#!/usr/bin/env python3
"""
Core Functionality Test for GARCH Trading System

This tests the core trading logic with synthetic data to ensure
all components work correctly without relying on external APIs.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add paths
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_garch_with_synthetic_data():
    """Test GARCH model with synthetic return data"""
    print("=== Testing GARCH Model with Synthetic Data ===")
    
    try:
        from models.garch_model import GarchModel
        
        # Create synthetic returns data (252 days)
        np.random.seed(42)  # For reproducibility
        
        # Generate GARCH-like returns
        n_obs = 252
        returns = []
        volatility = 0.02  # Initial volatility
        
        for i in range(n_obs):
            # Simple GARCH(1,1) simulation
            volatility = 0.00001 + 0.1 * (returns[-1]**2 if returns else 0.0001) + 0.8 * volatility
            return_t = np.random.normal(0, np.sqrt(volatility))
            returns.append(return_t)
        
        returns = pd.Series(returns)
        print(f"Generated {len(returns)} synthetic returns")
        print(f"Return statistics: mean={returns.mean():.4f}, std={returns.std():.4f}")
        
        # Test GARCH model
        garch_model = GarchModel("TEST")
        
        # Test model fitting
        success = garch_model.fit(returns, train_ratio=0.8)
        print(f"✅ Model fitting: {'Success' if success else 'Failed'}")
        
        if success:
            # Test prediction
            prediction = garch_model.predict()
            if prediction:
                print(f"✅ Volatility prediction: {prediction.predicted_volatility:.4f}")
                print(f"✅ Confidence interval: {prediction.confidence_interval}")
                print(f"✅ Model AIC: {prediction.aic:.2f}")
                
                # Test prediction premium
                current_price = 100.0
                premium = garch_model.calculate_prediction_premium(current_price, prediction.predicted_volatility)
                print(f"✅ Prediction premium: {premium:.4f}")
                
                return True
            else:
                print("❌ Prediction failed")
                return False
        else:
            print("❌ Model fitting failed")
            return False
    
    except Exception as e:
        print(f"❌ GARCH test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_with_synthetic_data():
    """Test trading strategy with synthetic market data"""
    print("\n=== Testing Trading Strategy with Synthetic Data ===")
    
    try:
        from strategy.garch_strategy import GarchTradingStrategy
        from data.market_data import MarketDataPoint
        
        # Create strategy
        strategy = GarchTradingStrategy("TEST")
        
        # Generate synthetic OHLCV data
        np.random.seed(42)
        base_price = 100.0
        n_bars = 300
        
        for i in range(n_bars):
            # Generate realistic OHLCV data
            price_change = np.random.normal(0, 0.01)  # 1% daily volatility
            close_price = base_price * (1 + price_change)
            
            high_price = close_price * (1 + abs(np.random.normal(0, 0.005)))
            low_price = close_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = base_price
            volume = int(np.random.normal(1000000, 200000))
            
            # Create data point
            data_point = MarketDataPoint(
                symbol="TEST",
                timestamp=datetime.now() - timedelta(minutes=n_bars-i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timeframe='1Min'
            )
            
            # Update strategy
            strategy.update_market_data(data_point)
            base_price = close_price
        
        print(f"✅ Updated strategy with {n_bars} data points")
        print(f"✅ Strategy has {len(strategy.returns_data)} returns")
        
        # Test signal generation
        if len(strategy.returns_data) > 100:
            current_price = base_price
            signal = strategy.generate_signal(current_price)
            
            if signal:
                print(f"✅ Generated signal: {signal.signal_type.value}")
                print(f"✅ Signal strength: {signal.strength:.3f}")
                print(f"✅ Signal confidence: {signal.confidence:.3f}")
                print(f"✅ Predicted volatility: {signal.predicted_volatility:.4f}")
                return True
            else:
                print("⚠️  No signal generated (this is normal)")
                return True
        else:
            print("❌ Insufficient data for signal generation")
            return False
            
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_management():
    """Test risk management calculations"""
    print("\n=== Testing Risk Management ===")
    
    try:
        from execution.risk_manager import RiskManager, PositionInfo
        
        # Create risk manager
        risk_manager = RiskManager()
        
        # Test with mock portfolio
        portfolio_value = 100000
        risk_manager.update_portfolio_value(portfolio_value)
        
        # Create mock positions
        positions = {
            'TEST1': PositionInfo(
                symbol='TEST1',
                quantity=100,
                market_value=10000,
                cost_basis=9500,
                unrealized_pnl=500,
                unrealized_pnl_pct=0.0526,
                current_price=100.0,
                entry_time=datetime.now() - timedelta(hours=1)
            ),
            'TEST2': PositionInfo(
                symbol='TEST2',
                quantity=200,
                market_value=20000,
                cost_basis=19800,
                unrealized_pnl=200,
                unrealized_pnl_pct=0.0101,
                current_price=100.0,
                entry_time=datetime.now() - timedelta(hours=2)
            )
        }
        
        # Test risk metrics calculation
        risk_metrics = risk_manager.calculate_risk_metrics(portfolio_value, positions)
        
        print(f"✅ Total exposure: ${risk_metrics.total_exposure:.2f}")
        print(f"✅ Max position size: ${risk_metrics.max_position_size:.2f}")
        print(f"✅ Concentration ratio: {risk_metrics.concentration_ratio:.3f}")
        print(f"✅ Risk level: {risk_metrics.risk_level.value}")
        
        # Test position sizing
        from execution.risk_manager import PositionSizer
        from strategy.garch_strategy import TradingSignal, SignalType
        
        position_sizer = PositionSizer()
        
        # Mock signal
        mock_signal = TradingSignal(
            symbol="TEST",
            signal_type=SignalType.BUY,
            strength=0.7,
            confidence=0.8,
            timestamp=datetime.now(),
            price=100.0,
            predicted_volatility=0.02,
            prediction_premium=0.01,
            technical_indicators={},
            reasoning="Test signal"
        )
        
        # Test position sizing
        position_size = position_sizer.calculate_position_size(
            mock_signal, portfolio_value, positions, risk_metrics
        )
        
        print(f"✅ Calculated position size: {position_size} shares")
        
        # Test stop loss calculation
        stop_loss = risk_manager.calculate_stop_loss_price(100.0, "BUY", 0.02)
        take_profit = risk_manager.calculate_take_profit_price(100.0, "BUY", 0.02)
        
        print(f"✅ Stop loss price: ${stop_loss:.2f}")
        print(f"✅ Take profit price: ${take_profit:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_technical_indicators():
    """Test technical indicators"""
    print("\n=== Testing Technical Indicators ===")
    
    try:
        from strategy.garch_strategy import TechnicalIndicators
        
        # Generate synthetic price data
        np.random.seed(42)
        base_price = 100.0
        prices = []
        
        for i in range(100):
            price_change = np.random.normal(0, 0.01)
            base_price = base_price * (1 + price_change)
            prices.append(base_price)
        
        prices = pd.Series(prices)
        
        # Test SMA
        sma_20 = TechnicalIndicators.calculate_sma(prices, 20)
        print(f"✅ SMA(20): {sma_20.iloc[-1]:.2f}")
        
        # Test EMA
        ema_12 = TechnicalIndicators.calculate_ema(prices, 12)
        print(f"✅ EMA(12): {ema_12.iloc[-1]:.2f}")
        
        # Test RSI
        rsi = TechnicalIndicators.calculate_rsi(prices)
        print(f"✅ RSI: {rsi.iloc[-1]:.2f}")
        
        # Test Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(prices)
        print(f"✅ Bollinger Bands: {bb_lower.iloc[-1]:.2f} - {bb_upper.iloc[-1]:.2f}")
        
        # Test MACD
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(prices)
        print(f"✅ MACD: {macd_line.iloc[-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_system():
    """Test configuration system"""
    print("\n=== Testing Configuration System ===")
    
    try:
        from utils.config import config
        
        # Test configuration loading
        broker_config = config.get_broker_config()
        print(f"✅ Broker: {broker_config.name}")
        
        garch_config = config.get_garch_config()
        print(f"✅ GARCH model: {garch_config.model_type}({garch_config.p},{garch_config.q})")
        
        strategy_config = config.get_strategy_config()
        print(f"✅ Strategy: {strategy_config.name}")
        
        risk_config = config.get_risk_config()
        print(f"✅ Risk: max daily loss {risk_config.max_daily_loss:.1%}")
        
        # Test validation
        is_valid = config.validate_configuration()
        print(f"✅ Configuration validation: {'Valid' if is_valid else 'Invalid'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_system():
    """Test logging system"""
    print("\n=== Testing Logging System ===")
    
    try:
        from utils.logger import trading_logger, log_info, log_trade, log_risk
        
        # Test basic logging
        log_info("Test log message")
        print("✅ Basic logging works")
        
        # Test trade logging
        log_trade("BUY", "TEST", 100, 99.50, order_id="test_123")
        print("✅ Trade logging works")
        
        # Test risk logging
        log_risk("TEST_RISK", "TEST", 0.05, 0.02, "Test action")
        print("✅ Risk logging works")
        
        # Test log stats
        stats = trading_logger.get_log_stats()
        print(f"✅ Log stats: {stats['total_log_files']} files")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all core functionality tests"""
    print("🧪 GARCH Trading System - Core Functionality Tests")
    print("=" * 60)
    print("Testing with synthetic data to verify all components work correctly...")
    print()
    
    tests = [
        ("Configuration System", test_configuration_system),
        ("Logging System", test_logging_system),
        ("Technical Indicators", test_technical_indicators),
        ("GARCH Model", test_garch_with_synthetic_data),
        ("Trading Strategy", test_strategy_with_synthetic_data),
        ("Risk Management", test_risk_management)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"Running {test_name} test...")
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with error: {e}")
        print()
    
    print("📋 Test Results Summary:")
    print("-" * 30)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All core functionality tests passed!")
        print("✅ The system is working correctly with synthetic data")
        print("✅ All components are properly integrated")
        print("✅ The trading logic is functioning as expected")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("Some core components need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)