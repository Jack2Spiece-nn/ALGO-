#!/usr/bin/env python3
"""
Test MT5 Configuration for Arjay Siega's Account
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import config
from src.execution.mt5_executor import MT5Executor

def test_mt5_config():
    """Test MT5 configuration is loaded correctly"""
    print("🔧 Testing MT5 Configuration for Arjay Siega")
    print("=" * 50)
    
    try:
        # Load broker configuration
        broker_config = config.get_broker_config()
        
        print("✅ Configuration loaded successfully")
        print(f"   Account Name: {broker_config.account_name}")
        print(f"   Account Type: {broker_config.account_type}")
        print(f"   Login: {broker_config.login}")
        print(f"   Server: {broker_config.server}")
        print(f"   MT5 Path: {broker_config.mt5_path}")
        print(f"   Password: {'*' * len(broker_config.password or '')}")
        print(f"   Investor: {'*' * len(broker_config.investor or '')}")
        
        # Verify required fields
        required_fields = {
            'login': broker_config.login,
            'server': broker_config.server,
            'password': broker_config.password,
            'mt5_path': broker_config.mt5_path
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        else:
            print("✅ All required fields present")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_mt5_executor_with_config():
    """Test MT5 executor with the configured account"""
    print("\n🚀 Testing MT5 Executor with Account Configuration")
    print("=" * 50)
    
    try:
        # Initialize MT5 executor with the configuration
        executor = MT5Executor()
        
        print("✅ MT5 Executor initialized")
        print(f"   Connection Status: {'Connected' if executor.connected else 'Mock Mode'}")
        
        # Get portfolio summary
        portfolio = executor.get_portfolio_summary()
        print(f"   Account Value: ${portfolio.get('account_value', 0):,.2f}")
        print(f"   Currency: {portfolio.get('currency', 'N/A')}")
        print(f"   Leverage: {portfolio.get('leverage', 'N/A')}")
        
        # Test position management settings
        print(f"   Lot Size: {executor.lot_size}")
        print(f"   Max Lot Size: {executor.max_lot_size}")
        print(f"   Risk Per Trade: {executor.risk_per_trade * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ MT5 Executor test failed: {e}")
        return False

def test_symbol_configuration():
    """Test symbol configuration"""
    print("\n📊 Testing Symbol Configuration")
    print("=" * 30)
    
    try:
        symbols_config = config.get_symbols()
        
        print(f"✅ Primary Symbol: {symbols_config.get('primary', 'N/A')}")
        print(f"✅ Watchlist: {symbols_config.get('watchlist', [])}")
        
        # Test data configuration
        data_config = config.get_data_config()
        print(f"✅ Timeframes: {data_config.get('timeframes', [])}")
        print(f"✅ Lookback Days: {data_config.get('lookback_days', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Symbol configuration test failed: {e}")
        return False

def test_risk_configuration():
    """Test risk management configuration"""
    print("\n🛡️ Testing Risk Management Configuration")
    print("=" * 40)
    
    try:
        risk_config = config.get_risk_config()
        
        print(f"✅ Max Daily Loss: {risk_config.max_daily_loss * 100:.1f}%")
        print(f"✅ Max Drawdown: {risk_config.max_drawdown * 100:.1f}%")
        print(f"✅ Stop Loss: {risk_config.stop_loss_pct * 100:.1f}%")
        print(f"✅ Take Profit: {risk_config.take_profit_pct * 100:.1f}%")
        print(f"✅ Max Positions: {risk_config.max_positions}")
        
        # Test MT5 position sizing
        mt5_sizing = config.get('mt5_position_sizing', {})
        print(f"✅ MT5 Lot Size: {mt5_sizing.get('lot_size', 'N/A')}")
        print(f"✅ MT5 Risk Per Trade: {mt5_sizing.get('risk_per_trade', 0) * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk configuration test failed: {e}")
        return False

def main():
    """Run all configuration tests"""
    print("🔧 MT5 GARCH Trading Strategy - Arjay Siega Account Test")
    print("=" * 60)
    
    tests = [
        ("MT5 Configuration", test_mt5_config),
        ("MT5 Executor", test_mt5_executor_with_config),
        ("Symbol Configuration", test_symbol_configuration),
        ("Risk Configuration", test_risk_configuration)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your MT5 account is properly configured.")
        print("🚀 Ready to run the GARCH trading strategy!")
        print("\nTo start trading:")
        print("   python3 src/main.py")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration.")
    
    print("\n📝 Account Details Configured:")
    print(f"   Name: Arjay Siega")
    print(f"   Type: Forex Hedged USD")
    print(f"   Server: MetaQuotes-Demo")
    print(f"   Login: 94435704")
    print(f"   Status: Demo Account")

if __name__ == "__main__":
    main()