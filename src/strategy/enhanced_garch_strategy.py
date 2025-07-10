"""
Enhanced GARCH Trading Strategy with ML Ensemble Integration

This module extends the original GARCH strategy with:
- Ensemble model predictions (GARCH + LSTM + XGBoost)
- Advanced signal processing
- Enhanced performance tracking
- Dynamic strategy adaptation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from src.models.ensemble_model import ensemble_manager, EnsembleSignal
from src.models.garch_model import garch_manager
from src.models.lstm_volatility_model import lstm_manager
from src.models.xgboost_signal_model import xgboost_manager
from src.data.market_data import MarketDataManager, MarketDataPoint, market_data_manager
from src.strategy.garch_strategy import SignalType, TradingSignal, TechnicalIndicators
from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug


@dataclass
class EnhancedTradingSignal(TradingSignal):
    """Enhanced trading signal with ensemble predictions"""
    ensemble_signal: Optional[EnsembleSignal]
    model_agreement: float
    signal_quality_score: float
    regime_context: Dict[str, Any]
    risk_adjusted_strength: float
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'ensemble_signal': self.ensemble_signal.to_dict() if self.ensemble_signal else None,
            'model_agreement': self.model_agreement,
            'signal_quality_score': self.signal_quality_score,
            'regime_context': self.regime_context,
            'risk_adjusted_strength': self.risk_adjusted_strength
        })
        return base_dict


class MarketRegime(Enum):
    """Market regime classifications"""
    LOW_VOL_TRENDING = "low_vol_trending"
    HIGH_VOL_TRENDING = "high_vol_trending"
    LOW_VOL_RANGING = "low_vol_ranging"
    HIGH_VOL_RANGING = "high_vol_ranging"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class EnhancedGarchTradingStrategy:
    """
    Enhanced GARCH trading strategy with ML ensemble integration
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.strategy_config = config.get_strategy_config()
        self.risk_config = config.get_risk_config()
        
        # Model managers
        self.ensemble_model = ensemble_manager.get_model(symbol)
        self.garch_model = garch_manager.get_model(symbol)
        self.lstm_model = lstm_manager.get_model(symbol)
        self.xgboost_model = xgboost_manager.get_model(symbol)
        
        # Market data
        self.market_data = market_data_manager
        self.technical_indicators = TechnicalIndicators()
        
        # Strategy state
        self.current_regime = MarketRegime.LOW_VOL_RANGING
        self.regime_history = []
        self.signal_history = []
        self.performance_metrics = {}
        
        # Enhanced configuration
        self.regime_detection_window = 50
        self.signal_quality_threshold = 0.6
        self.regime_adaptation_enabled = True
        
        # Data storage
        self.price_data = pd.DataFrame()
        self.returns_data = pd.Series()
        self.volume_data = pd.Series()
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.regime_performance = {regime: {'signals': 0, 'success': 0} for regime in MarketRegime}
        
        log_info(f"Enhanced GARCH strategy initialized for {symbol}")
    
    def update_market_data(self, data_point: MarketDataPoint):
        """Update strategy with new market data"""
        
        # Add to price data
        new_row = pd.DataFrame({
            'timestamp': [data_point.timestamp],
            'open': [data_point.open],
            'high': [data_point.high],
            'low': [data_point.low],
            'close': [data_point.close],
            'volume': [data_point.volume]
        })
        
        new_row.set_index('timestamp', inplace=True)
        self.price_data = pd.concat([self.price_data, new_row])
        
        # Keep only recent data (last 2000 points for enhanced analysis)
        if len(self.price_data) > 2000:
            self.price_data = self.price_data.tail(2000)
        
        # Update returns and volume data
        if len(self.price_data) > 1:
            self.returns_data = self.price_data['close'].pct_change().dropna()
            self.volume_data = self.price_data['volume']
        
        # Update regime detection
        if len(self.price_data) >= self.regime_detection_window:
            self._detect_market_regime()
    
    def _detect_market_regime(self):
        """Detect current market regime"""
        try:
            if len(self.returns_data) < self.regime_detection_window:
                return
            
            # Get recent data
            recent_returns = self.returns_data.tail(self.regime_detection_window)
            recent_prices = self.price_data['close'].tail(self.regime_detection_window)
            recent_volume = self.volume_data.tail(self.regime_detection_window)
            
            # Calculate regime indicators
            volatility = recent_returns.std() * np.sqrt(252)  # Annualized
            avg_volume = recent_volume.mean()
            volume_std = recent_volume.std()
            
            # Trend strength
            prices_array = recent_prices.values
            trend_slope = np.polyfit(range(len(prices_array)), prices_array, 1)[0]
            trend_strength = abs(trend_slope) / np.mean(prices_array)
            
            # Volume regime
            volume_regime = recent_volume.iloc[-1] / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum
            momentum_5d = (recent_prices.iloc[-1] / recent_prices.iloc[-5] - 1) if len(recent_prices) >= 5 else 0
            momentum_20d = (recent_prices.iloc[-1] / recent_prices.iloc[-20] - 1) if len(recent_prices) >= 20 else 0
            
            # Classify regime
            vol_threshold = 0.20  # 20% annual volatility threshold
            trend_threshold = 0.001  # 0.1% daily trend threshold
            crisis_threshold = 0.40  # 40% annual volatility for crisis
            
            if volatility > crisis_threshold:
                new_regime = MarketRegime.CRISIS
            elif volatility > vol_threshold:
                if trend_strength > trend_threshold:
                    new_regime = MarketRegime.HIGH_VOL_TRENDING
                else:
                    new_regime = MarketRegime.HIGH_VOL_RANGING
            else:
                if trend_strength > trend_threshold:
                    new_regime = MarketRegime.LOW_VOL_TRENDING
                else:
                    new_regime = MarketRegime.LOW_VOL_RANGING
            
            # Special case for recovery (after crisis with improving momentum)
            if (self.current_regime == MarketRegime.CRISIS and 
                volatility < crisis_threshold and 
                momentum_5d > 0 and momentum_20d > 0):
                new_regime = MarketRegime.RECOVERY
            
            # Update regime if changed
            if new_regime != self.current_regime:
                self.regime_history.append({
                    'timestamp': datetime.now(),
                    'old_regime': self.current_regime,
                    'new_regime': new_regime,
                    'volatility': volatility,
                    'trend_strength': trend_strength,
                    'volume_regime': volume_regime
                })
                
                self.current_regime = new_regime
                log_info(f"Regime change detected for {self.symbol}: {new_regime.value}")
            
        except Exception as e:
            log_error(f"Error detecting market regime: {e}")
    
    def _calculate_signal_quality_score(self, ensemble_signal: EnsembleSignal) -> float:
        """Calculate signal quality score"""
        try:
            quality_score = 0.0
            
            # Model agreement component (30%)
            quality_score += ensemble_signal.model_agreement * 0.3
            
            # Signal strength component (25%)
            quality_score += ensemble_signal.signal_strength * 0.25
            
            # Confidence component (20%)
            quality_score += ensemble_signal.confidence * 0.2
            
            # Regime consistency component (15%)
            regime_consistency = self._calculate_regime_consistency(ensemble_signal)
            quality_score += regime_consistency * 0.15
            
            # Volatility context component (10%)
            vol_context = min(1.0, ensemble_signal.predicted_volatility / 0.05)  # Normalize to 5% volatility
            vol_score = 1.0 - abs(vol_context - 0.5) * 2  # Prefer moderate volatility
            quality_score += vol_score * 0.1
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            log_error(f"Error calculating signal quality score: {e}")
            return 0.5
    
    def _calculate_regime_consistency(self, ensemble_signal: EnsembleSignal) -> float:
        """Calculate how consistent the signal is with current regime"""
        try:
            if self.current_regime == MarketRegime.LOW_VOL_TRENDING:
                # Favor trend-following signals in low vol trending markets
                if ensemble_signal.signal_type == 'BUY' and ensemble_signal.predicted_volatility < 0.02:
                    return 0.9
                elif ensemble_signal.signal_type == 'SELL' and ensemble_signal.predicted_volatility < 0.02:
                    return 0.7
                else:
                    return 0.5
            
            elif self.current_regime == MarketRegime.HIGH_VOL_TRENDING:
                # Be more cautious in high volatility trending markets
                if ensemble_signal.signal_strength > 0.7:
                    return 0.8
                else:
                    return 0.3
            
            elif self.current_regime == MarketRegime.LOW_VOL_RANGING:
                # Favor mean reversion in ranging markets
                if ensemble_signal.signal_type == 'HOLD':
                    return 0.8
                else:
                    return 0.6
            
            elif self.current_regime == MarketRegime.HIGH_VOL_RANGING:
                # Very cautious in high vol ranging markets
                if ensemble_signal.signal_strength > 0.8:
                    return 0.6
                else:
                    return 0.2
            
            elif self.current_regime == MarketRegime.CRISIS:
                # Extremely conservative in crisis
                if ensemble_signal.signal_type == 'HOLD':
                    return 0.9
                else:
                    return 0.1
            
            elif self.current_regime == MarketRegime.RECOVERY:
                # Favor buy signals in recovery
                if ensemble_signal.signal_type == 'BUY' and ensemble_signal.signal_strength > 0.6:
                    return 0.9
                else:
                    return 0.5
            
            return 0.5
            
        except Exception as e:
            log_error(f"Error calculating regime consistency: {e}")
            return 0.5
    
    def _adjust_signal_for_regime(self, signal_type: str, signal_strength: float) -> Tuple[str, float]:
        """Adjust signal based on current market regime"""
        
        if not self.regime_adaptation_enabled:
            return signal_type, signal_strength
        
        try:
            if self.current_regime == MarketRegime.CRISIS:
                # Force conservative approach in crisis
                return 'HOLD', 0.1
            
            elif self.current_regime == MarketRegime.HIGH_VOL_RANGING:
                # Reduce signal strength in high vol ranging
                adjusted_strength = signal_strength * 0.5
                if adjusted_strength < 0.3:
                    return 'HOLD', adjusted_strength
                return signal_type, adjusted_strength
            
            elif self.current_regime == MarketRegime.RECOVERY:
                # Boost buy signals in recovery
                if signal_type == 'BUY':
                    adjusted_strength = min(1.0, signal_strength * 1.2)
                    return signal_type, adjusted_strength
                elif signal_type == 'SELL':
                    adjusted_strength = signal_strength * 0.7
                    return signal_type, adjusted_strength
            
            elif self.current_regime == MarketRegime.LOW_VOL_TRENDING:
                # Boost trend signals in low vol trending
                if signal_type != 'HOLD':
                    adjusted_strength = min(1.0, signal_strength * 1.1)
                    return signal_type, adjusted_strength
            
            return signal_type, signal_strength
            
        except Exception as e:
            log_error(f"Error adjusting signal for regime: {e}")
            return signal_type, signal_strength
    
    def generate_enhanced_signal(self, current_price: float) -> Optional[EnhancedTradingSignal]:
        """Generate enhanced trading signal using ensemble model"""
        
        if len(self.price_data) < 100:
            log_debug(f"Insufficient data for enhanced signal generation: {len(self.price_data)} points")
            return None
        
        try:
            # Get ensemble prediction
            ensemble_signal = self.ensemble_model.predict(self.price_data, current_price)
            
            if not ensemble_signal:
                log_debug(f"No ensemble signal generated for {self.symbol}")
                return None
            
            # Calculate signal quality score
            signal_quality = self._calculate_signal_quality_score(ensemble_signal)
            
            # Check quality threshold
            if signal_quality < self.signal_quality_threshold:
                log_debug(f"Signal quality too low for {self.symbol}: {signal_quality:.3f}")
                return None
            
            # Adjust signal for current regime
            adjusted_signal_type, adjusted_strength = self._adjust_signal_for_regime(
                ensemble_signal.signal_type, ensemble_signal.signal_strength
            )
            
            # Risk-adjusted signal strength
            risk_adjustment = 1.0
            if ensemble_signal.predicted_volatility > 0.03:  # High volatility
                risk_adjustment = 0.03 / ensemble_signal.predicted_volatility
            
            risk_adjusted_strength = adjusted_strength * risk_adjustment
            
            # Calculate technical indicators for context
            technical_indicators = self._calculate_technical_indicators(current_price)
            
            # Create regime context
            regime_context = {
                'current_regime': self.current_regime.value,
                'regime_consistency': self._calculate_regime_consistency(ensemble_signal),
                'volatility_level': 'high' if ensemble_signal.predicted_volatility > 0.02 else 'low',
                'trend_direction': self._get_trend_direction(),
                'volume_regime': self._get_volume_regime()
            }
            
            # Convert ensemble signal type to SignalType enum
            signal_type_map = {'BUY': SignalType.BUY, 'SELL': SignalType.SELL, 'HOLD': SignalType.HOLD}
            signal_type_enum = signal_type_map.get(adjusted_signal_type, SignalType.HOLD)
            
            # Create enhanced trading signal
            enhanced_signal = EnhancedTradingSignal(
                symbol=self.symbol,
                signal_type=signal_type_enum,
                strength=risk_adjusted_strength,
                confidence=ensemble_signal.confidence,
                timestamp=datetime.now(),
                price=current_price,
                predicted_volatility=ensemble_signal.predicted_volatility,
                prediction_premium=ensemble_signal.prediction_premium,
                technical_indicators=technical_indicators,
                reasoning=f"Ensemble: {ensemble_signal.reasoning} | Regime: {self.current_regime.value}",
                ensemble_signal=ensemble_signal,
                model_agreement=ensemble_signal.model_agreement,
                signal_quality_score=signal_quality,
                regime_context=regime_context,
                risk_adjusted_strength=risk_adjusted_strength
            )
            
            # Store signal
            self.signal_history.append(enhanced_signal)
            self.total_signals += 1
            
            # Update regime performance tracking
            self.regime_performance[self.current_regime]['signals'] += 1
            
            log_info(f"Enhanced signal for {self.symbol}: {adjusted_signal_type} "
                    f"(strength: {risk_adjusted_strength:.3f}, quality: {signal_quality:.3f}, "
                    f"regime: {self.current_regime.value})")
            
            return enhanced_signal
            
        except Exception as e:
            log_error(f"Error generating enhanced signal for {self.symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, current_price: float) -> Dict[str, Any]:
        """Calculate technical indicators for enhanced context"""
        if len(self.price_data) < 50:
            return {}
        
        try:
            prices = self.price_data['close']
            high = self.price_data['high']
            low = self.price_data['low']
            volume = self.price_data['volume']
            
            # Enhanced indicators set
            indicators = {}
            
            # Trend indicators
            indicators['sma_20'] = prices.rolling(20).mean().iloc[-1]
            indicators['sma_50'] = prices.rolling(50).mean().iloc[-1]
            indicators['ema_12'] = prices.ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = prices.ewm(span=26).mean().iloc[-1]
            
            # Momentum indicators
            indicators['rsi'] = self.technical_indicators.calculate_rsi(prices).iloc[-1]
            macd, signal_line, histogram = self.technical_indicators.calculate_macd(prices)
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = histogram.iloc[-1]
            
            # Volatility indicators
            bb_upper, bb_middle, bb_lower = self.technical_indicators.calculate_bollinger_bands(prices)
            indicators['bb_upper'] = bb_upper.iloc[-1]
            indicators['bb_middle'] = bb_middle.iloc[-1]
            indicators['bb_lower'] = bb_lower.iloc[-1]
            indicators['bb_width'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = volume.rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma']
            
            # Advanced indicators
            atr = self.technical_indicators.calculate_atr(high, low, prices)
            indicators['atr'] = atr.iloc[-1]
            
            stoch_k, stoch_d = self.technical_indicators.calculate_stochastic(high, low, prices)
            indicators['stoch_k'] = stoch_k.iloc[-1]
            indicators['stoch_d'] = stoch_d.iloc[-1]
            
            return indicators
            
        except Exception as e:
            log_error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _get_trend_direction(self) -> str:
        """Get current trend direction"""
        if len(self.price_data) < 20:
            return 'neutral'
        
        try:
            prices = self.price_data['close'].tail(20)
            slope = np.polyfit(range(len(prices)), prices, 1)[0]
            
            if slope > 0.01:
                return 'up'
            elif slope < -0.01:
                return 'down'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _get_volume_regime(self) -> str:
        """Get current volume regime"""
        if len(self.volume_data) < 20:
            return 'normal'
        
        try:
            recent_volume = self.volume_data.tail(20)
            avg_volume = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]
            
            ratio = current_volume / avg_volume
            
            if ratio > 1.5:
                return 'high'
            elif ratio < 0.5:
                return 'low'
            else:
                return 'normal'
                
        except Exception:
            return 'normal'
    
    def should_enter_position(self, signal: EnhancedTradingSignal) -> bool:
        """Enhanced position entry logic"""
        
        # Basic checks from original strategy
        if signal.signal_type == SignalType.HOLD:
            return False
        
        if signal.signal_quality_score < self.signal_quality_threshold:
            return False
        
        if signal.confidence < 0.5:
            return False
        
        # Regime-specific entry conditions
        if self.current_regime == MarketRegime.CRISIS:
            return False  # No new positions in crisis
        
        if self.current_regime == MarketRegime.HIGH_VOL_RANGING:
            if signal.strength < 0.8:  # Higher threshold in volatile ranging markets
                return False
        
        # Model agreement check
        if signal.model_agreement < 0.6:
            return False
        
        # Risk-adjusted strength check
        if signal.risk_adjusted_strength < 0.4:
            return False
        
        return True
    
    def update_signal_performance(self, signal: EnhancedTradingSignal, outcome: float):
        """Update signal performance tracking"""
        try:
            # Determine if signal was successful
            success = False
            if signal.signal_type == SignalType.BUY and outcome > 0:
                success = True
            elif signal.signal_type == SignalType.SELL and outcome < 0:
                success = True
            elif signal.signal_type == SignalType.HOLD and abs(outcome) < 0.005:
                success = True
            
            if success:
                self.successful_signals += 1
                
                # Update regime performance
                regime_at_signal = signal.regime_context.get('current_regime')
                if regime_at_signal:
                    regime_enum = MarketRegime(regime_at_signal)
                    self.regime_performance[regime_enum]['success'] += 1
            
            # Update ensemble model performance
            self.ensemble_model.update_model_performance(outcome)
            
        except Exception as e:
            log_error(f"Error updating signal performance: {e}")
    
    def get_enhanced_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        base_summary = {
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'success_rate': self.successful_signals / self.total_signals if self.total_signals > 0 else 0,
            'current_regime': self.current_regime.value,
            'regime_changes': len(self.regime_history)
        }
        
        # Regime performance breakdown
        regime_performance = {}
        for regime, stats in self.regime_performance.items():
            if stats['signals'] > 0:
                regime_performance[regime.value] = {
                    'signals': stats['signals'],
                    'success': stats['success'],
                    'success_rate': stats['success'] / stats['signals']
                }
        
        base_summary['regime_performance'] = regime_performance
        
        # Signal quality statistics
        if self.signal_history:
            recent_signals = self.signal_history[-50:]  # Last 50 signals
            avg_quality = np.mean([s.signal_quality_score for s in recent_signals])
            avg_agreement = np.mean([s.model_agreement for s in recent_signals])
            
            base_summary.update({
                'average_signal_quality': avg_quality,
                'average_model_agreement': avg_agreement,
                'last_signal_time': self.signal_history[-1].timestamp if self.signal_history else None
            })
        
        return base_summary


class EnhancedStrategyManager:
    """Manager for enhanced trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.active_symbols = config.get_symbols().get('watchlist', [])
        
        # Initialize enhanced strategies
        for symbol in self.active_symbols:
            self.strategies[symbol] = EnhancedGarchTradingStrategy(symbol)
        
        log_info(f"Enhanced strategy manager initialized with {len(self.strategies)} strategies")
    
    def get_strategy(self, symbol: str) -> EnhancedGarchTradingStrategy:
        """Get enhanced strategy for symbol"""
        if symbol not in self.strategies:
            self.strategies[symbol] = EnhancedGarchTradingStrategy(symbol)
        
        return self.strategies[symbol]
    
    def generate_all_signals(self, current_prices: Dict[str, float]) -> Dict[str, Optional[EnhancedTradingSignal]]:
        """Generate enhanced signals for all strategies"""
        signals = {}
        
        for symbol, price in current_prices.items():
            if symbol in self.strategies:
                signal = self.strategies[symbol].generate_enhanced_signal(price)
                signals[symbol] = signal
        
        return signals
    
    def fit_all_ensemble_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, bool]]:
        """Fit all ensemble models for all strategies"""
        results = {}
        
        for symbol, strategy in self.strategies.items():
            if symbol in data_dict:
                results[symbol] = strategy.ensemble_model.fit_all_models(data_dict[symbol])
        
        return results
    
    def get_enhanced_performance_summary(self) -> Dict[str, Any]:
        """Get enhanced performance summary for all strategies"""
        summary = {}
        
        for symbol, strategy in self.strategies.items():
            summary[symbol] = strategy.get_enhanced_performance_summary()
        
        return summary


# Global enhanced strategy manager instance
enhanced_strategy_manager = EnhancedStrategyManager()