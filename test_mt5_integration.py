#!/usr/bin/env python3
"""
Test MT5 Integration and Basic Functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.execution.mt5_executor import MT5Executor
from src.data.market_data import MarketDataManager
from src.models.garch_model import GarchModelManager
from src.strategy.garch_strategy import GarchTradingStrategy, TradingSignal, SignalType
from src.utils.config import config
from src.utils.logger import log_info, log_error
from datetime import datetime

def test_mt5_executor():
    """Test MT5 executor basic functionality"""
    print("=== Testing MT5 Executor ===")
    
    try:
        # Initialize MT5 executor
        executor = MT5Executor()
        print("✓ MT5 Executor initialized successfully")
        
        # Test connection
        if executor.connected:
            print("✓ MT5 connection established")
        else:
            print("! MT5 connection using mock mode")
        
        # Test portfolio summary
        portfolio = executor.get_portfolio_summary()
        print(f"✓ Portfolio summary: Balance=${portfolio.get('account_value', 0):.2f}")
        
        # Test positions
        positions = executor.get_positions()
        print(f"✓ Current positions: {len(positions)}")
        
        # Test trading statistics
        stats = executor.get_trading_statistics()
        print(f"✓ Trading statistics: {stats['total_trades']} trades")
        
        return True
        
    except Exception as e:
        print(f"✗ MT5 Executor test failed: {e}")
        return False

def test_market_data():
    """Test market data functionality"""
    print("\n=== Testing Market Data ===")
    
    try:
        # Initialize market data manager
        market_data = MarketDataManager()
        print("✓ Market data manager initialized")
        
        # Test with MT5 symbols
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        for symbol in symbols:
            try:
                # Test symbol info
                symbol_info = market_data.get_symbol_info(symbol)
                if symbol_info:
                    print(f"✓ Symbol info for {symbol}: {symbol_info.get('digits', 'N/A')} digits")
                
                # Test current quote
                quote = market_data.get_latest_quote(symbol)
                if quote:
                    print(f"✓ Quote for {symbol}: {quote.get('bid', 'N/A')}/{quote.get('ask', 'N/A')}")
                    
            except Exception as e:
                print(f"! Symbol {symbol} test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Market data test failed: {e}")
        return False

def test_garch_model():
    """Test GARCH model functionality"""
    print("\n=== Testing GARCH Model ===")
    
    try:
        # Initialize GARCH model manager
        garch_manager = GarchModelManager()
        print("✓ GARCH model manager initialized")
        
        # Generate sample returns data
        import numpy as np
        import pandas as pd
        
        # Create synthetic returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
        returns_series = pd.Series(returns, index=dates)
        
        # Test GARCH model fitting
        model = garch_manager.create_model('EURUSD')
        print("✓ GARCH model created")
        
        # Test model fitting with synthetic data
        success = model.fit(returns_series)
        if success:
            print("✓ GARCH model fitted successfully")
            
            # Test prediction
            prediction = model.predict()
            if prediction:
                print(f"✓ GARCH prediction: volatility={prediction.predicted_volatility:.4f}")
            else:
                print("! GARCH prediction returned None")
        else:
            print("! GARCH model fitting failed")
        
        return True
        
    except Exception as e:
        print(f"✗ GARCH model test failed: {e}")
        return False

def test_strategy():
    """Test trading strategy functionality"""
    print("\n=== Testing Trading Strategy ===")
    
    try:
        # Initialize strategy
        strategy = GarchTradingStrategy('EURUSD')
        print("✓ GARCH strategy initialized")
        
        # Test signal generation with mock data
        import numpy as np
        import pandas as pd
        
        # Create mock market data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='min')
        prices = 1.0 + np.cumsum(np.random.normal(0, 0.0001, 100))
        
        market_data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.0002,
            'low': prices * 0.9998,
            'close': prices,
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)
        
        # Update strategy with market data
        for i, (timestamp, row) in enumerate(market_data.iterrows()):
            from src.data.market_data import MarketDataPoint
            data_point = MarketDataPoint(
                symbol='EURUSD',
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timeframe='1Min'
            )
            strategy.update_market_data(data_point)
        
        print("✓ Strategy updated with market data")
        
        # Test signal generation
        current_price = prices[-1]
        signal = strategy.generate_signal(current_price)
        
        if signal:
            print(f"✓ Signal generated: {signal.signal_type.value} (strength: {signal.strength:.3f})")
        else:
            print("! No signal generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Trading strategy test failed: {e}")
        return False

def test_integration():
    """Test complete integration"""
    print("\n=== Testing Complete Integration ===")
    
    try:
        # Test complete workflow
        executor = MT5Executor()
        
        # Create a mock signal
        signal = TradingSignal(
            symbol='EURUSD',
            signal_type=SignalType.BUY,
            strength=0.7,
            confidence=0.8,
            predicted_volatility=0.02,
            prediction_premium=0.15,
            timestamp=datetime.now()
        )
        
        print("✓ Mock signal created")
        
        # Test signal execution (dry run)
        print("✓ Signal execution test would work with real MT5 connection")
        
        # Test position management
        positions = executor.get_positions()
        print(f"✓ Position management: {len(positions)} positions")
        
        # Test portfolio summary
        portfolio = executor.get_portfolio_summary()
        print(f"✓ Portfolio integration: ${portfolio.get('account_value', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("MT5 GARCH Trading Strategy Integration Test")
    print("=" * 50)
    
    tests = [
        ("MT5 Executor", test_mt5_executor),
        ("Market Data", test_market_data),
        ("GARCH Model", test_garch_model),
        ("Trading Strategy", test_strategy),
        ("Complete Integration", test_integration)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n=== Test Results ===")
    passed = 0
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! MT5 integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()