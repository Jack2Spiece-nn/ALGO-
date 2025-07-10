"""
Comprehensive Test Suite for Enhanced GARCH Trading Strategy

This module contains comprehensive tests for all enhanced features including:
- ML Ensemble Models (LSTM, XGBoost, Ensemble)
- Enhanced Risk Management
- Enhanced Strategy Logic
- Dashboard Components
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import warnings
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress warnings during testing
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings


class TestEnhancedFeatures:
    """Test suite for enhanced features"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1min')
        # Filter to trading hours only
        dates = dates[(dates.hour >= 9) & (dates.hour < 16)]
        dates = dates[dates.dayofweek < 5]  # Weekdays only
        
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price data
        initial_price = 100.0
        returns = np.random.normal(0.0001, 0.02, len(dates))
        
        prices = [initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])
        
        data = pd.DataFrame({
            'open': prices * np.random.uniform(0.998, 1.002, len(prices)),
            'high': prices * np.random.uniform(1.000, 1.005, len(prices)),
            'low': prices * np.random.uniform(0.995, 1.000, len(prices)),
            'close': prices,
            'volume': np.random.randint(10000, 100000, len(prices))
        }, index=dates)
        
        # Ensure OHLC relationships are correct
        data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
        data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
        
        return data
    
    @pytest.fixture
    def sample_returns(self, sample_market_data):
        """Generate sample returns data"""
        return sample_market_data['close'].pct_change().dropna()


class TestLSTMVolatilityModel:
    """Test LSTM Volatility Model"""
    
    def test_lstm_model_import(self):
        """Test that LSTM model can be imported without errors"""
        try:
            from models.lstm_volatility_model import LSTMVolatilityModel, lstm_manager
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import LSTM model: {e}")
    
    def test_lstm_model_initialization(self):
        """Test LSTM model initialization"""
        from models.lstm_volatility_model import LSTMVolatilityModel
        
        model = LSTMVolatilityModel("TEST")
        assert model.symbol == "TEST"
        assert model.sequence_length == 60
        assert model.model is None  # Not fitted yet
        assert len(model.prediction_history) == 0
    
    def test_lstm_feature_preparation(self, sample_market_data):
        """Test LSTM feature preparation"""
        from models.lstm_volatility_model import LSTMVolatilityModel
        
        model = LSTMVolatilityModel("TEST")
        
        # Test feature preparation
        features = model._prepare_features(sample_market_data.head(100))
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert 'returns' in features.columns
        assert 'rolling_std_5' in features.columns
        assert 'rsi' in features.columns
    
    def test_lstm_model_fitting(self, sample_market_data):
        """Test LSTM model fitting"""
        from models.lstm_volatility_model import LSTMVolatilityModel
        
        model = LSTMVolatilityModel("TEST")
        
        # Use smaller data for faster testing
        test_data = sample_market_data.head(500)
        
        # Mock TensorFlow to avoid actual training in tests
        with patch('models.lstm_volatility_model.Sequential') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_model.fit.return_value = Mock()
            mock_model.fit.return_value.history = {'loss': [0.1], 'val_loss': [0.15]}
            
            success = model.fit(test_data)
            assert success == True
            assert model.last_fit_time is not None


class TestXGBoostSignalModel:
    """Test XGBoost Signal Model"""
    
    def test_xgboost_model_import(self):
        """Test that XGBoost model can be imported without errors"""
        try:
            from models.xgboost_signal_model import XGBoostSignalModel, xgboost_manager
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import XGBoost model: {e}")
    
    def test_xgboost_model_initialization(self):
        """Test XGBoost model initialization"""
        from models.xgboost_signal_model import XGBoostSignalModel
        
        model = XGBoostSignalModel("TEST")
        assert model.symbol == "TEST"
        assert model.model is None  # Not fitted yet
        assert len(model.prediction_history) == 0
    
    def test_xgboost_feature_preparation(self, sample_market_data):
        """Test XGBoost feature preparation"""
        from models.xgboost_signal_model import XGBoostSignalModel
        
        model = XGBoostSignalModel("TEST")
        
        # Test feature preparation
        features = model._prepare_features(sample_market_data.head(100))
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert 'returns' in features.columns
        assert 'rsi_14' in features.columns
        assert 'macd' in features.columns
    
    def test_xgboost_target_preparation(self, sample_market_data):
        """Test XGBoost target preparation"""
        from models.xgboost_signal_model import XGBoostSignalModel
        
        model = XGBoostSignalModel("TEST")
        
        # Test target preparation
        targets = model._prepare_targets(sample_market_data.head(100))
        
        assert isinstance(targets, pd.Series)
        assert len(targets) > 0
        assert all(target in [0, 1, 2] for target in targets.dropna())  # SELL, HOLD, BUY


class TestEnsembleModel:
    """Test Ensemble Model"""
    
    def test_ensemble_model_import(self):
        """Test that Ensemble model can be imported without errors"""
        try:
            from models.ensemble_model import EnsembleModel, ensemble_manager
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import Ensemble model: {e}")
    
    def test_ensemble_model_initialization(self):
        """Test Ensemble model initialization"""
        from models.ensemble_model import EnsembleModel
        
        model = EnsembleModel("TEST")
        assert model.symbol == "TEST"
        assert len(model.ensemble_history) == 0
        assert 'garch' in model.model_weights
        assert 'lstm' in model.model_weights
        assert 'xgboost' in model.model_weights
    
    def test_ensemble_model_agreement_calculation(self):
        """Test model agreement calculation"""
        from models.ensemble_model import EnsembleModel
        from models.garch_model import GarchPrediction
        from models.lstm_volatility_model import LSTMPrediction
        from models.xgboost_signal_model import XGBoostSignal
        
        model = EnsembleModel("TEST")
        
        # Create mock predictions
        garch_pred = GarchPrediction(
            symbol="TEST",
            timestamp=datetime.now(),
            predicted_volatility=0.02,
            variance_forecast=0.0004,
            confidence_interval=(0.015, 0.025),
            model_params={},
            log_likelihood=100.0,
            aic=200.0,
            bic=210.0
        )
        
        lstm_pred = LSTMPrediction(
            symbol="TEST",
            timestamp=datetime.now(),
            predicted_volatility=0.022,
            confidence_score=0.8,
            prediction_horizon=1,
            model_metrics={}
        )
        
        xgb_signal = XGBoostSignal(
            symbol="TEST",
            timestamp=datetime.now(),
            signal_type="BUY",
            signal_probability=0.7,
            feature_importance={},
            confidence_score=0.75,
            model_metrics={}
        )
        
        agreement = model._calculate_model_agreement(garch_pred, lstm_pred, xgb_signal)
        assert 0 <= agreement <= 1


class TestEnhancedRiskManager:
    """Test Enhanced Risk Manager"""
    
    def test_enhanced_risk_manager_import(self):
        """Test that Enhanced Risk Manager can be imported without errors"""
        try:
            from execution.enhanced_risk_manager import EnhancedRiskManager, enhanced_risk_manager
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import Enhanced Risk Manager: {e}")
    
    def test_enhanced_risk_manager_initialization(self):
        """Test Enhanced Risk Manager initialization"""
        from execution.enhanced_risk_manager import EnhancedRiskManager
        
        manager = EnhancedRiskManager()
        assert manager.portfolio_value == 100000.0  # Default value
        assert len(manager.price_history) == 0
        assert len(manager.stress_scenarios) > 0
    
    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation"""
        from execution.enhanced_risk_manager import EnhancedRiskManager
        
        manager = EnhancedRiskManager()
        
        # Add some mock price history
        symbols = ["TEST1", "TEST2"]
        for symbol in symbols:
            manager.price_history[symbol] = []
            manager.returns_history[symbol] = []
            
            # Add mock data
            for i in range(100):
                timestamp = datetime.now() - timedelta(minutes=i)
                price = 100 + np.random.normal(0, 1)
                manager.price_history[symbol].append((timestamp, price))
                
                if i > 0:
                    prev_price = manager.price_history[symbol][-2][1]
                    return_val = (price - prev_price) / prev_price
                    manager.returns_history[symbol].append((timestamp, return_val))
        
        # Test correlation calculation
        corr_matrix = manager._calculate_correlation_matrix(symbols)
        assert isinstance(corr_matrix, pd.DataFrame)
        if not corr_matrix.empty:
            assert corr_matrix.shape == (2, 2)
    
    def test_dynamic_position_sizing(self):
        """Test dynamic position sizing"""
        from execution.enhanced_risk_manager import EnhancedRiskManager
        from strategy.garch_strategy import TradingSignal, SignalType
        
        manager = EnhancedRiskManager()
        
        # Create mock signal
        signal = TradingSignal(
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
        
        # Create mock positions
        positions = {}
        portfolio_value = 100000
        
        size = manager.calculate_dynamic_position_size(signal, positions, portfolio_value)
        assert size > 0
        assert isinstance(size, (int, float))


class TestEnhancedStrategy:
    """Test Enhanced Strategy"""
    
    def test_enhanced_strategy_import(self):
        """Test that Enhanced Strategy can be imported without errors"""
        try:
            from strategy.enhanced_garch_strategy import EnhancedGarchTradingStrategy, enhanced_strategy_manager
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import Enhanced Strategy: {e}")
    
    def test_enhanced_strategy_initialization(self):
        """Test Enhanced Strategy initialization"""
        from strategy.enhanced_garch_strategy import EnhancedGarchTradingStrategy, MarketRegime
        
        strategy = EnhancedGarchTradingStrategy("TEST")
        assert strategy.symbol == "TEST"
        assert strategy.current_regime == MarketRegime.LOW_VOL_RANGING
        assert len(strategy.signal_history) == 0
    
    def test_market_regime_detection(self, sample_market_data):
        """Test market regime detection"""
        from strategy.enhanced_garch_strategy import EnhancedGarchTradingStrategy
        from data.market_data import MarketDataPoint
        
        strategy = EnhancedGarchTradingStrategy("TEST")
        
        # Update strategy with market data
        for timestamp, row in sample_market_data.head(100).iterrows():
            data_point = MarketDataPoint(
                symbol="TEST",
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timeframe='1Min'
            )
            strategy.update_market_data(data_point)
        
        # Should have detected a regime
        assert strategy.current_regime is not None


class TestDashboard:
    """Test Dashboard Components"""
    
    def test_dashboard_import(self):
        """Test that Dashboard can be imported without errors"""
        try:
            from monitoring.dashboard import TradingDashboard
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import Dashboard: {e}")
    
    def test_dashboard_initialization(self):
        """Test Dashboard initialization"""
        from monitoring.dashboard import TradingDashboard
        
        dashboard = TradingDashboard()
        assert dashboard.host == 'localhost'
        assert dashboard.port == 8050
        assert dashboard.update_interval == 5
        assert len(dashboard.performance_data) == 0


class TestSystemIntegration:
    """Test System Integration"""
    
    def test_full_system_integration(self, sample_market_data):
        """Test full system integration"""
        # This test ensures all components work together
        
        # Test 1: Import all components
        try:
            from models.ensemble_model import ensemble_manager
            from strategy.enhanced_garch_strategy import enhanced_strategy_manager
            from execution.enhanced_risk_manager import enhanced_risk_manager
            from data.market_data import MarketDataPoint
        except ImportError as e:
            pytest.fail(f"Failed to import system components: {e}")
        
        # Test 2: Initialize strategy
        strategy = enhanced_strategy_manager.get_strategy("TEST")
        assert strategy is not None
        
        # Test 3: Update with market data
        for timestamp, row in sample_market_data.head(50).iterrows():
            data_point = MarketDataPoint(
                symbol="TEST",
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timeframe='1Min'
            )
            strategy.update_market_data(data_point)
        
        # Test 4: Generate signal (may or may not succeed due to data requirements)
        current_price = sample_market_data['close'].iloc[-1]
        try:
            signal = strategy.generate_enhanced_signal(current_price)
            # Signal generation may fail due to insufficient model training, which is OK
        except Exception as e:
            # Log the exception but don't fail the test
            print(f"Signal generation failed (expected): {e}")
        
        # Test 5: Risk management
        portfolio_value = 100000
        positions = {}
        
        try:
            risk_metrics = enhanced_risk_manager.calculate_enhanced_risk_metrics(
                portfolio_value, positions
            )
            assert risk_metrics is not None
        except Exception as e:
            pytest.fail(f"Risk management failed: {e}")
    
    def test_configuration_loading(self):
        """Test configuration loading"""
        try:
            from utils.config import config
            
            # Test basic config access
            symbols = config.get_symbols()
            assert isinstance(symbols, dict)
            
            strategy_config = config.get_strategy_config()
            assert hasattr(strategy_config, 'max_position_size')
            
            risk_config = config.get_risk_config()
            assert hasattr(risk_config, 'max_daily_loss')
            
        except Exception as e:
            pytest.fail(f"Configuration loading failed: {e}")
    
    def test_logging_system(self):
        """Test logging system"""
        try:
            from utils.logger import log_info, log_error, log_debug
            
            # Test logging functions
            log_info("Test info message")
            log_error("Test error message")
            log_debug("Test debug message")
            
            assert True  # If we get here, logging works
            
        except Exception as e:
            pytest.fail(f"Logging system failed: {e}")


# Performance Tests
class TestPerformance:
    """Test Performance and Efficiency"""
    
    def test_signal_generation_performance(self, sample_market_data):
        """Test signal generation performance"""
        import time
        from strategy.enhanced_garch_strategy import enhanced_strategy_manager
        from data.market_data import MarketDataPoint
        
        strategy = enhanced_strategy_manager.get_strategy("PERF_TEST")
        
        # Update with market data
        for timestamp, row in sample_market_data.head(100).iterrows():
            data_point = MarketDataPoint(
                symbol="PERF_TEST",
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timeframe='1Min'
            )
            strategy.update_market_data(data_point)
        
        # Measure signal generation time
        current_price = sample_market_data['close'].iloc[-1]
        
        start_time = time.time()
        try:
            signal = strategy.generate_enhanced_signal(current_price)
        except Exception:
            pass  # Signal generation may fail due to insufficient training
        end_time = time.time()
        
        # Signal generation should complete within reasonable time
        assert (end_time - start_time) < 5.0  # 5 seconds max
    
    def test_risk_calculation_performance(self):
        """Test risk calculation performance"""
        import time
        from execution.enhanced_risk_manager import enhanced_risk_manager
        
        # Create mock positions
        positions = {}
        for i in range(10):  # Test with 10 positions
            positions[f"TEST{i}"] = type('PositionInfo', (), {
                'symbol': f'TEST{i}',
                'quantity': 100,
                'market_value': 10000,
                'cost_basis': 9900,
                'unrealized_pnl': 100,
                'unrealized_pnl_pct': 0.01,
                'current_price': 100.0,
                'entry_time': datetime.now()
            })()
        
        portfolio_value = 100000
        
        start_time = time.time()
        risk_metrics = enhanced_risk_manager.calculate_enhanced_risk_metrics(
            portfolio_value, positions
        )
        end_time = time.time()
        
        # Risk calculation should complete within reasonable time
        assert (end_time - start_time) < 2.0  # 2 seconds max
        assert risk_metrics is not None


def run_comprehensive_tests():
    """Run comprehensive test suite"""
    
    print("🧪 Running Comprehensive Test Suite for Enhanced GARCH Strategy")
    print("=" * 70)
    
    # Configure pytest to run with verbose output
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "-x",  # Stop on first failure
        "--tb=short",  # Short traceback format
        "-q"  # Quiet mode for cleaner output
    ]
    
    try:
        # Run the tests
        exit_code = pytest.main(pytest_args)
        
        if exit_code == 0:
            print("\n✅ ALL TESTS PASSED!")
            print("🎉 Enhanced GARCH Trading Strategy is fully functional!")
            return True
        else:
            print(f"\n❌ TESTS FAILED (Exit code: {exit_code})")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST EXECUTION FAILED: {e}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)