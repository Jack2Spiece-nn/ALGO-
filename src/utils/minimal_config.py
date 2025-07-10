"""
Minimal configuration for testing without full config system
"""

class MinimalConfig:
    """Minimal config class for testing"""
    
    def get_symbols(self):
        """Return basic symbol configuration"""
        return {
            'SPY': {'allocation': 0.4},
            'QQQ': {'allocation': 0.3},
            'IWM': {'allocation': 0.3}
        }
    
    def get_strategy_config(self):
        """Return basic strategy configuration"""
        class StrategyConfig:
            max_position_size = 1000
            min_signal_strength = 0.6
            lookback_period = 60
        
        return StrategyConfig()
    
    def get_risk_config(self):
        """Return basic risk configuration"""
        class RiskConfig:
            max_daily_loss = 0.02
            max_portfolio_exposure = 0.9
            var_confidence = 0.95
        
        return RiskConfig()
    
    def get(self, key, default=None):
        """Get configuration value"""
        config_map = {
            'monitoring.log_level': 'INFO',
            'symbols': self.get_symbols(),
            'strategy': self.get_strategy_config(),
            'risk': self.get_risk_config()
        }
        return config_map.get(key, default)

# Global minimal config instance
config = MinimalConfig()