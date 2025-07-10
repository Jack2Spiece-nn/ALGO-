#!/usr/bin/env python3
"""
Test Windows Startup for Arjay's MT5 GARCH Strategy
"""

import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add to Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Change to project directory
os.chdir(script_dir)

print("🧪 Windows Startup Test for Arjay Siega")
print("=" * 40)
print(f"Project Directory: {script_dir}")
print(f"Python Path: {sys.path[:3]}")  # Show first 3 entries
print("=" * 40)

def test_imports():
    """Test all critical imports"""
    print("\n🔍 Testing Critical Imports...")
    
    try:
        print("  📦 Testing config import...")
        from src.utils.config import config
        print("  ✅ Config imported successfully")
        
        broker_config = config.get_broker_config()
        print(f"  ✅ Broker config loaded: {broker_config.name} (Account: {broker_config.login})")
        
    except Exception as e:
        print(f"  ❌ Config import failed: {e}")
        return False
    
    try:
        print("  📦 Testing market data import...")
        from src.data.market_data import market_data_manager
        print("  ✅ Market data imported successfully")
        
    except Exception as e:
        print(f"  ❌ Market data import failed: {e}")
        return False
    
    try:
        print("  📦 Testing GARCH models import...")
        from src.models.garch_model import garch_manager
        print("  ✅ GARCH models imported successfully")
        
    except Exception as e:
        print(f"  ❌ GARCH models import failed: {e}")
        return False
    
    try:
        print("  📦 Testing MT5 executor import...")
        from src.execution.mt5_executor import MT5Executor
        print("  ✅ MT5 executor imported successfully")
        
    except Exception as e:
        print(f"  ❌ MT5 executor import failed: {e}")
        return False
    
    return True

def test_mt5_connection():
    """Test MT5 connection"""
    print("\n🔗 Testing MT5 Connection...")
    
    try:
        from src.execution.mt5_executor import MT5Executor
        
        executor = MT5Executor()
        print(f"  ✅ MT5 Executor created")
        print(f"  📊 Connection: {'Connected' if executor.connected else 'Mock Mode'}")
        
        portfolio = executor.get_portfolio_summary()
        print(f"  💰 Account Value: ${portfolio.get('account_value', 0):,.2f}")
        print(f"  🏦 Currency: {portfolio.get('currency', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MT5 connection test failed: {e}")
        return False

def test_strategy_initialization():
    """Test strategy initialization"""
    print("\n🎯 Testing Strategy Initialization...")
    
    try:
        from src.strategy.garch_strategy import GarchTradingStrategy
        
        strategy = GarchTradingStrategy('EURUSD')
        print("  ✅ GARCH strategy created for EURUSD")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy initialization failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting comprehensive Windows startup test...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("MT5 Connection Test", test_mt5_connection),
        ("Strategy Test", test_strategy_initialization)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 40)
    print("📋 TEST RESULTS")
    print("=" * 40)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Ready to start main trading engine!")
        print("\nTo start trading:")
        print("   python src\\main.py")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            input("\nPress Enter to exit...")
    except Exception as e:
        print(f"\n❌ Startup test crashed: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")