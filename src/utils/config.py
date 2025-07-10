"""
Configuration management for GARCH Intraday Trading Strategy
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, validator
from dataclasses import dataclass


@dataclass
class BrokerConfig:
    name: str
    paper_base_url: Optional[str] = None
    live_base_url: Optional[str] = None
    data_feed: Optional[str] = None
    # MT5-specific parameters
    mt5_path: Optional[str] = None
    login: Optional[int] = None
    server: Optional[str] = None
    password: Optional[str] = None
    investor: Optional[str] = None
    timeout: Optional[int] = None
    account_name: Optional[str] = None
    account_type: Optional[str] = None


@dataclass
class GarchConfig:
    model_type: str
    p: int
    q: int
    mean_model: str
    distribution: str
    rescale: bool
    forecast_horizon: int
    rolling_window: int
    refit_frequency: int


@dataclass
class StrategyConfig:
    name: str
    position_sizing_method: str
    volatility_target: float
    max_position_size: float
    signal_threshold: float
    hold_period_minutes: int


@dataclass
class RiskConfig:
    max_daily_loss: float
    max_drawdown: float
    stop_loss_pct: float
    take_profit_pct: float
    position_timeout_minutes: int
    max_positions: int


@dataclass
class TradingHoursConfig:
    market_open: str
    market_close: str
    timezone: str
    exclude_first_minutes: int
    exclude_last_minutes: int


class ConfigManager:
    """
    Centralized configuration management for the trading system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config_data = self._load_config()
        self._load_environment_variables()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        current_dir = Path(__file__).parent.parent.parent
        return str(current_dir / "config" / "config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _load_environment_variables(self):
        """Load environment variables from .env file"""
        env_path = Path(self.config_path).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'broker.name')"""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_broker_config(self) -> BrokerConfig:
        """Get broker configuration"""
        broker_data = self.config_data.get('broker', {})
        return BrokerConfig(**broker_data)
    
    def get_garch_config(self) -> GarchConfig:
        """Get GARCH model configuration"""
        garch_data = self.config_data.get('garch', {})
        return GarchConfig(**garch_data)
    
    def get_strategy_config(self) -> StrategyConfig:
        """Get strategy configuration"""
        strategy_data = self.config_data.get('strategy', {})
        return StrategyConfig(**strategy_data)
    
    def get_risk_config(self) -> RiskConfig:
        """Get risk management configuration"""
        risk_data = self.config_data.get('risk', {})
        return RiskConfig(**risk_data)
    
    def get_trading_hours_config(self) -> TradingHoursConfig:
        """Get trading hours configuration"""
        hours_data = self.config_data.get('trading_hours', {})
        return TradingHoursConfig(**hours_data)
    
    def get_api_credentials(self) -> Dict[str, str]:
        """Get API credentials from environment variables"""
        return {
            'alpaca_api_key': os.getenv('ALPACA_API_KEY'),
            'alpaca_secret_key': os.getenv('ALPACA_SECRET_KEY'),
            'polygon_api_key': os.getenv('POLYGON_API_KEY'),
            'quandl_api_key': os.getenv('QUANDL_API_KEY')
        }
    
    def get_symbols(self) -> Dict[str, Any]:
        """Get trading symbols configuration"""
        return self.config_data.get('symbols', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config_data.get('data', {})
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration"""
        return self.config_data.get('backtesting', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.config_data.get('monitoring', {})
    
    def is_live_trading(self) -> bool:
        """Check if running in live trading mode"""
        return self.config_data.get('environment', 'paper') == 'live'
    
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode"""
        return self.config_data.get('environment', 'paper') == 'paper'
    
    def validate_configuration(self) -> bool:
        """Validate configuration completeness and correctness"""
        required_sections = ['broker', 'garch', 'strategy', 'risk', 'trading_hours']
        
        for section in required_sections:
            if section not in self.config_data:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate API credentials for live trading
        if self.is_live_trading():
            credentials = self.get_api_credentials()
            if not credentials['alpaca_api_key'] or not credentials['alpaca_secret_key']:
                raise ValueError("Alpaca API credentials required for live trading")
        
        # Validate risk parameters
        risk_config = self.get_risk_config()
        if risk_config.max_daily_loss <= 0 or risk_config.max_daily_loss > 0.2:
            raise ValueError("max_daily_loss must be between 0 and 0.2 (20%)")
        
        if risk_config.max_drawdown <= 0 or risk_config.max_drawdown > 0.5:
            raise ValueError("max_drawdown must be between 0 and 0.5 (50%)")
        
        return True
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration value"""
        keys = key.split('.')
        config_ref = self.config_data
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config_data, file, default_flow_style=False)


# Global configuration instance
config = ConfigManager()