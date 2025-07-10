"""
Advanced logging system for GARCH Intraday Trading Strategy
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
try:
    from src.utils.config import config
except ImportError:
    try:
        from utils.config import config
    except ImportError:
        # Create a basic config fallback
        class BasicLogConfig:
            def get(self, key, default=None):
                configs = {
                    'monitoring.log_level': 'INFO',
                    'development.debug': False
                }
                return configs.get(key, default)
        
        config = BasicLogConfig()


class TradingLogger:
    """
    Enhanced logging system for trading operations
    """
    
    def __init__(self, name: str = "GarchTrading"):
        self.name = name
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Remove default handler
        logger.remove()
        
        # Setup structured logging
        self._setup_handlers()
        
        # Create specialized loggers
        self._setup_specialized_loggers()
    
    def _setup_handlers(self):
        """Setup logging handlers for different log levels and destinations"""
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level=config.get('monitoring.log_level', 'INFO'),
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # General application log
        logger.add(
            self.log_dir / "app_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="30 days",
            compression="zip",
            level="DEBUG"
        )
        
        # Error-only log
        logger.add(
            self.log_dir / "errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
            rotation="1 day",
            retention="90 days",
            compression="zip",
            level="ERROR",
            backtrace=True,
            diagnose=True
        )
    
    def _setup_specialized_loggers(self):
        """Setup specialized loggers for different components"""
        
        # Trading operations log
        self.trading_log_path = self.log_dir / "trading_{time:YYYY-MM-DD}.log"
        
        # GARCH model log
        self.garch_log_path = self.log_dir / "garch_{time:YYYY-MM-DD}.log"
        
        # Risk management log
        self.risk_log_path = self.log_dir / "risk_{time:YYYY-MM-DD}.log"
        
        # Performance log
        self.performance_log_path = self.log_dir / "performance_{time:YYYY-MM-DD}.log"
    
    def log_trading_operation(self, operation_type: str, symbol: str, 
                             quantity: float, price: float, 
                             additional_data: Optional[Dict] = None):
        """Log trading operations with structured data"""
        
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'additional_data': additional_data or {}
        }
        
        logger.add(
            self.trading_log_path,
            format="{message}",
            rotation="1 day",
            retention="1 year",
            level="INFO"
        )
        
        logger.info(f"TRADE: {json.dumps(trade_data)}")
    
    def log_garch_prediction(self, symbol: str, predicted_volatility: float,
                           confidence_interval: tuple, model_params: Dict):
        """Log GARCH model predictions"""
        
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'predicted_volatility': predicted_volatility,
            'confidence_interval': confidence_interval,
            'model_params': model_params
        }
        
        logger.add(
            self.garch_log_path,
            format="{message}",
            rotation="1 day",
            retention="1 year",
            level="INFO"
        )
        
        logger.info(f"GARCH: {json.dumps(prediction_data)}")
    
    def log_risk_event(self, event_type: str, symbol: str, 
                      current_value: float, threshold: float,
                      action_taken: str):
        """Log risk management events"""
        
        risk_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'symbol': symbol,
            'current_value': current_value,
            'threshold': threshold,
            'action_taken': action_taken
        }
        
        logger.add(
            self.risk_log_path,
            format="{message}",
            rotation="1 day",
            retention="1 year",
            level="WARNING"
        )
        
        logger.warning(f"RISK: {json.dumps(risk_data)}")
    
    def log_performance_metric(self, metric_name: str, value: float,
                             benchmark: Optional[float] = None):
        """Log performance metrics"""
        
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'value': value,
            'benchmark': benchmark
        }
        
        logger.add(
            self.performance_log_path,
            format="{message}",
            rotation="1 day",
            retention="1 year",
            level="INFO"
        )
        
        logger.info(f"PERFORMANCE: {json.dumps(performance_data)}")
    
    def log_system_status(self, status: str, details: Dict):
        """Log system status updates"""
        
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'details': details
        }
        
        logger.info(f"SYSTEM: {json.dumps(status_data)}")
    
    def log_error_with_context(self, error: Exception, context: Dict):
        """Log errors with additional context"""
        
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }
        
        logger.error(f"ERROR: {json.dumps(error_data)}")
    
    def log_backtest_result(self, strategy_name: str, period: str,
                          returns: float, sharpe_ratio: float,
                          max_drawdown: float, win_rate: float):
        """Log backtesting results"""
        
        backtest_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy_name': strategy_name,
            'period': period,
            'returns': returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
        
        logger.info(f"BACKTEST: {json.dumps(backtest_data)}")
    
    def create_session_log(self, session_id: str):
        """Create a session-specific log file"""
        
        session_log_path = self.log_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger.add(
            session_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
            level="DEBUG",
            enqueue=True
        )
        
        return session_log_path
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        
        log_files = list(self.log_dir.glob("*.log"))
        
        stats = {
            'total_log_files': len(log_files),
            'log_directory': str(self.log_dir),
            'log_files': [str(f.name) for f in log_files],
            'total_size_mb': sum(f.stat().st_size for f in log_files) / (1024 * 1024)
        }
        
        return stats
    
    @staticmethod
    def get_logger(name: str = "GarchTrading") -> 'TradingLogger':
        """Get or create a logger instance"""
        return TradingLogger(name)


# Global logger instance
trading_logger = TradingLogger()

# Convenience functions
def log_info(message: str, **kwargs):
    """Log info message"""
    logger.info(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Log warning message"""
    logger.warning(message, **kwargs)

def log_error(message: str, **kwargs):
    """Log error message"""
    logger.error(message, **kwargs)

def log_debug(message: str, **kwargs):
    """Log debug message"""
    logger.debug(message, **kwargs)

def log_trade(operation_type: str, symbol: str, quantity: float, price: float, **kwargs):
    """Log trading operation"""
    trading_logger.log_trading_operation(operation_type, symbol, quantity, price, kwargs)

def log_garch(symbol: str, predicted_volatility: float, confidence_interval: tuple, model_params: Dict):
    """Log GARCH prediction"""
    trading_logger.log_garch_prediction(symbol, predicted_volatility, confidence_interval, model_params)

def log_risk(event_type: str, symbol: str, current_value: float, threshold: float, action_taken: str):
    """Log risk event"""
    trading_logger.log_risk_event(event_type, symbol, current_value, threshold, action_taken)

def log_performance(metric_name: str, value: float, benchmark: Optional[float] = None):
    """Log performance metric"""
    trading_logger.log_performance_metric(metric_name, value, benchmark)