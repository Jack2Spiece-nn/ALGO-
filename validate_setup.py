#!/usr/bin/env python3
"""
Setup Validation Script for GARCH Intraday Trading Strategy

This script validates that all dependencies are installed correctly
and the system is properly configured.
"""

import sys
import os
import importlib
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version >= (3, 9):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Required: Python 3.9+")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'arch',
        'yfinance',
        'alpaca_trade_api',
        'matplotlib',
        'seaborn',
        'python-dotenv',
        'pydantic',
        'yaml',
        'loguru',
        'asyncio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'alpaca_trade_api':
                importlib.import_module('alpaca_trade_api')
            elif package == 'python-dotenv':
                importlib.import_module('dotenv')
            elif package == 'yaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """Check project structure"""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        'src',
        'src/data',
        'src/models',
        'src/strategy',
        'src/execution',
        'src/utils',
        'config',
        'logs',
        'data',
        'tests'
    ]
    
    required_files = [
        'src/__init__.py',
        'src/main.py',
        'src/utils/config.py',
        'src/utils/logger.py',
        'src/data/market_data.py',
        'src/models/garch_model.py',
        'src/strategy/garch_strategy.py',
        'src/execution/alpaca_executor.py',
        'src/execution/risk_manager.py',
        'config/config.yaml',
        'config/.env.example',
        'requirements.txt',
        'README.md'
    ]
    
    missing_items = []
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ - OK")
        else:
            print(f"❌ {dir_path}/ - Missing")
            missing_items.append(dir_path)
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - OK")
        else:
            print(f"❌ {file_path} - Missing")
            missing_items.append(file_path)
    
    return len(missing_items) == 0

def check_configuration():
    """Check configuration files"""
    print("\n⚙️  Checking configuration...")
    
    try:
        from utils.config import config
        
        # Check if config loads
        print("✅ Configuration file loads - OK")
        
        # Check essential config sections
        essential_sections = ['broker', 'garch', 'strategy', 'risk', 'trading_hours']
        
        for section in essential_sections:
            if config.get(section):
                print(f"✅ Config section '{section}' - OK")
            else:
                print(f"❌ Config section '{section}' - Missing")
        
        # Check API credentials
        credentials = config.get_api_credentials()
        if credentials.get('alpaca_api_key') and credentials.get('alpaca_secret_key'):
            print("✅ API credentials configured - OK")
        else:
            print("⚠️  API credentials not configured - Please set in config/.env")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_core_modules():
    """Check if core modules can be imported"""
    print("\n🔧 Checking core modules...")
    
    modules_to_check = [
        ('utils.config', 'Configuration manager'),
        ('utils.logger', 'Logging system'),
        ('data.market_data', 'Market data pipeline'),
        ('models.garch_model', 'GARCH model'),
        ('strategy.garch_strategy', 'Trading strategy'),
        ('execution.alpaca_executor', 'Order execution'),
        ('execution.risk_manager', 'Risk management')
    ]
    
    all_ok = True
    
    for module_name, description in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"✅ {description} - OK")
        except Exception as e:
            print(f"❌ {description} - Error: {e}")
            all_ok = False
    
    return all_ok

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test configuration
        from utils.config import config
        broker_config = config.get_broker_config()
        print(f"✅ Broker config: {broker_config.name}")
        
        # Test logging
        from utils.logger import log_info
        log_info("Setup validation test")
        print("✅ Logging system working")
        
        # Test market data (without API call)
        from data.market_data import MarketDataManager
        market_manager = MarketDataManager()
        print("✅ Market data manager initialized")
        
        # Test GARCH model
        from models.garch_model import GarchModel
        garch_model = GarchModel("SPY")
        print("✅ GARCH model initialized")
        
        # Test strategy
        from strategy.garch_strategy import GarchTradingStrategy
        strategy = GarchTradingStrategy("SPY")
        print("✅ Trading strategy initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def run_validation():
    """Run complete validation"""
    print("🔍 GARCH Intraday Trading Strategy - Setup Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Configuration", check_configuration),
        ("Core Modules", check_core_modules),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n📋 Validation Summary:")
    print("-" * 30)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your setup is ready for trading.")
        print("\nNext steps:")
        print("1. Configure your API credentials in config/.env")
        print("2. Review and adjust config/config.yaml")
        print("3. Run: python src/main.py")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("Need help? Check the README.md file or documentation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)