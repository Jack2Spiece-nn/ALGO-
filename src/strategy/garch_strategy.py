"""
GARCH-based Intraday Trading Strategy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from src.models.garch_model import GarchModel, GarchPrediction, garch_manager
from src.data.market_data import MarketDataManager, MarketDataPoint, market_data_manager
from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    symbol: str
    signal_type: SignalType
    strength: float  # Signal strength (0-1)
    confidence: float  # Confidence level (0-1)
    timestamp: datetime
    price: float
    predicted_volatility: float
    prediction_premium: float
    technical_indicators: Dict[str, Any]
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'price': self.price,
            'predicted_volatility': self.predicted_volatility,
            'prediction_premium': self.prediction_premium,
            'technical_indicators': self.technical_indicators,
            'reasoning': self.reasoning
        }


class TechnicalIndicators:
    """Technical indicators for signal enhancement"""
    
    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, 
                                 std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                     window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def calculate_volume_profile(prices: pd.Series, volume: pd.Series,
                                bins: int = 20) -> Dict[str, float]:
        """Volume Profile Analysis"""
        price_range = prices.max() - prices.min()
        price_bins = np.linspace(prices.min(), prices.max(), bins + 1)
        
        volume_profile = {}
        for i in range(len(price_bins) - 1):
            mask = (prices >= price_bins[i]) & (prices < price_bins[i + 1])
            volume_profile[f"price_{price_bins[i]:.2f}"] = volume[mask].sum()
        
        # Find Point of Control (POC) - price level with highest volume
        poc_price = max(volume_profile.keys(), key=lambda x: volume_profile[x])
        
        return {
            'poc_price': float(poc_price.split('_')[1]),
            'volume_profile': volume_profile
        }


class GarchTradingStrategy:
    """
    GARCH-based intraday trading strategy
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.strategy_config = config.get_strategy_config()
        self.risk_config = config.get_risk_config()
        
        # Initialize components
        self.garch_model = garch_manager.get_model(symbol)
        self.market_data = market_data_manager
        self.technical_indicators = TechnicalIndicators()
        
        # Strategy state
        self.last_signal = None
        self.signal_history = []
        self.current_position = None
        self.position_entry_time = None
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.failed_signals = 0
        
        # Data storage
        self.price_data = pd.DataFrame()
        self.returns_data = pd.Series()
        
        log_info(f"GARCH trading strategy initialized for {symbol}")
    
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
        
        # Keep only recent data (last 1000 points)
        if len(self.price_data) > 1000:
            self.price_data = self.price_data.tail(1000)
        
        # Update returns data
        if len(self.price_data) > 1:
            self.returns_data = self.price_data['close'].pct_change().dropna()
    
    def _calculate_technical_indicators(self, current_price: float) -> Dict[str, Any]:
        """Calculate technical indicators for signal generation"""
        
        if len(self.price_data) < 50:
            return {}
        
        try:
            prices = self.price_data['close']
            high = self.price_data['high']
            low = self.price_data['low']
            volume = self.price_data['volume']
            
            # Moving averages
            sma_20 = self.technical_indicators.calculate_sma(prices, 20)
            sma_50 = self.technical_indicators.calculate_sma(prices, 50)
            ema_12 = self.technical_indicators.calculate_ema(prices, 12)
            ema_26 = self.technical_indicators.calculate_ema(prices, 26)
            
            # RSI
            rsi = self.technical_indicators.calculate_rsi(prices)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.technical_indicators.calculate_bollinger_bands(prices)
            
            # MACD
            macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(prices)
            
            # Stochastic
            stoch_k, stoch_d = self.technical_indicators.calculate_stochastic(high, low, prices)
            
            # ATR
            atr = self.technical_indicators.calculate_atr(high, low, prices)
            
            # Volume analysis
            volume_profile = self.technical_indicators.calculate_volume_profile(prices, volume)
            
            # Current values
            indicators = {
                'sma_20': sma_20.iloc[-1] if not sma_20.empty else None,
                'sma_50': sma_50.iloc[-1] if not sma_50.empty else None,
                'ema_12': ema_12.iloc[-1] if not ema_12.empty else None,
                'ema_26': ema_26.iloc[-1] if not ema_26.empty else None,
                'rsi': rsi.iloc[-1] if not rsi.empty else None,
                'bb_upper': bb_upper.iloc[-1] if not bb_upper.empty else None,
                'bb_middle': bb_middle.iloc[-1] if not bb_middle.empty else None,
                'bb_lower': bb_lower.iloc[-1] if not bb_lower.empty else None,
                'macd_line': macd_line.iloc[-1] if not macd_line.empty else None,
                'signal_line': signal_line.iloc[-1] if not signal_line.empty else None,
                'histogram': histogram.iloc[-1] if not histogram.empty else None,
                'stoch_k': stoch_k.iloc[-1] if not stoch_k.empty else None,
                'stoch_d': stoch_d.iloc[-1] if not stoch_d.empty else None,
                'atr': atr.iloc[-1] if not atr.empty else None,
                'volume_profile': volume_profile,
                'current_price': current_price
            }
            
            # Price position relative to indicators
            indicators['price_vs_sma20'] = (current_price - indicators['sma_20']) / indicators['sma_20'] if indicators['sma_20'] else 0
            indicators['price_vs_sma50'] = (current_price - indicators['sma_50']) / indicators['sma_50'] if indicators['sma_50'] else 0
            indicators['bb_position'] = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']) if indicators['bb_upper'] and indicators['bb_lower'] else 0.5
            
            return indicators
            
        except Exception as e:
            log_error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _generate_garch_signal(self, garch_prediction: GarchPrediction,
                              current_price: float) -> Tuple[SignalType, float, str]:
        """Generate signal based on GARCH prediction"""
        
        # Calculate prediction premium
        prediction_premium = self.garch_model.calculate_prediction_premium(
            current_price, garch_prediction.predicted_volatility
        )
        
        # Signal logic based on volatility regime
        signal_threshold = self.strategy_config.signal_threshold
        
        reasoning = []
        
        # High volatility signals
        if garch_prediction.predicted_volatility > 0.02:  # 2% threshold
            # High volatility - potential reversal signal
            if prediction_premium > signal_threshold:
                reasoning.append(f"High volatility predicted ({garch_prediction.predicted_volatility:.4f})")
                reasoning.append(f"Prediction premium: {prediction_premium:.4f}")
                
                # Check if volatility is increasing (momentum)
                if len(self.garch_model.prediction_history) > 1:
                    prev_vol = self.garch_model.prediction_history[-2].predicted_volatility
                    if garch_prediction.predicted_volatility > prev_vol * 1.1:
                        reasoning.append("Volatility momentum increasing")
                        return SignalType.SELL, min(prediction_premium * 2, 1.0), " | ".join(reasoning)
                
                return SignalType.SELL, prediction_premium, " | ".join(reasoning)
        
        # Low volatility signals
        elif garch_prediction.predicted_volatility < 0.01:  # 1% threshold
            # Low volatility - potential trend continuation
            if prediction_premium < signal_threshold / 2:
                reasoning.append(f"Low volatility predicted ({garch_prediction.predicted_volatility:.4f})")
                reasoning.append(f"Trend continuation signal")
                
                return SignalType.BUY, 0.5, " | ".join(reasoning)
        
        # Medium volatility - hold
        reasoning.append(f"Medium volatility ({garch_prediction.predicted_volatility:.4f})")
        return SignalType.HOLD, 0.1, " | ".join(reasoning)
    
    def _generate_technical_signal(self, indicators: Dict[str, Any]) -> Tuple[SignalType, float, str]:
        """Generate signal based on technical indicators"""
        
        if not indicators:
            return SignalType.HOLD, 0.0, "No technical indicators available"
        
        signal_strength = 0.0
        reasoning = []
        
        # RSI signals
        if indicators.get('rsi'):
            rsi = indicators['rsi']
            if rsi < 30:
                signal_strength += 0.3
                reasoning.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                signal_strength -= 0.3
                reasoning.append(f"RSI overbought ({rsi:.1f})")
        
        # Moving average signals
        if indicators.get('price_vs_sma20'):
            ma_signal = indicators['price_vs_sma20']
            if ma_signal > 0.02:  # 2% above SMA
                signal_strength += 0.2
                reasoning.append(f"Price above SMA20 ({ma_signal:.3f})")
            elif ma_signal < -0.02:  # 2% below SMA
                signal_strength -= 0.2
                reasoning.append(f"Price below SMA20 ({ma_signal:.3f})")
        
        # Bollinger Bands
        if indicators.get('bb_position'):
            bb_pos = indicators['bb_position']
            if bb_pos < 0.2:  # Near lower band
                signal_strength += 0.25
                reasoning.append(f"Price near lower BB ({bb_pos:.3f})")
            elif bb_pos > 0.8:  # Near upper band
                signal_strength -= 0.25
                reasoning.append(f"Price near upper BB ({bb_pos:.3f})")
        
        # MACD signals
        if indicators.get('macd_line') and indicators.get('signal_line'):
            macd_diff = indicators['macd_line'] - indicators['signal_line']
            if macd_diff > 0:
                signal_strength += 0.15
                reasoning.append("MACD bullish")
            else:
                signal_strength -= 0.15
                reasoning.append("MACD bearish")
        
        # Stochastic signals
        if indicators.get('stoch_k') and indicators.get('stoch_d'):
            stoch_k = indicators['stoch_k']
            stoch_d = indicators['stoch_d']
            
            if stoch_k < 20 and stoch_d < 20:
                signal_strength += 0.2
                reasoning.append(f"Stochastic oversold ({stoch_k:.1f}, {stoch_d:.1f})")
            elif stoch_k > 80 and stoch_d > 80:
                signal_strength -= 0.2
                reasoning.append(f"Stochastic overbought ({stoch_k:.1f}, {stoch_d:.1f})")
        
        # Determine signal type
        if signal_strength > 0.3:
            return SignalType.BUY, min(signal_strength, 1.0), " | ".join(reasoning)
        elif signal_strength < -0.3:
            return SignalType.SELL, min(abs(signal_strength), 1.0), " | ".join(reasoning)
        else:
            return SignalType.HOLD, abs(signal_strength), " | ".join(reasoning) if reasoning else "No clear technical signal"
    
    def generate_signal(self, current_price: float) -> Optional[TradingSignal]:
        """Generate comprehensive trading signal"""
        
        # Check if we have enough data
        if len(self.returns_data) < 100:
            log_debug(f"Insufficient data for signal generation: {len(self.returns_data)} points")
            return None
        
        try:
            # Refit GARCH model if needed
            if self.garch_model.needs_refit():
                log_info(f"Refitting GARCH model for {self.symbol}")
                success = self.garch_model.fit(self.returns_data)
                if not success:
                    log_error(f"Failed to fit GARCH model for {self.symbol}")
                    return None
            
            # Get GARCH prediction
            garch_prediction = self.garch_model.predict()
            if not garch_prediction:
                log_error(f"Failed to generate GARCH prediction for {self.symbol}")
                return None
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(current_price)
            
            # Generate GARCH-based signal
            garch_signal, garch_strength, garch_reasoning = self._generate_garch_signal(
                garch_prediction, current_price
            )
            
            # Generate technical signal
            tech_signal, tech_strength, tech_reasoning = self._generate_technical_signal(
                technical_indicators
            )
            
            # Combine signals
            combined_signal, combined_strength, combined_confidence = self._combine_signals(
                garch_signal, garch_strength,
                tech_signal, tech_strength
            )
            
            # Check signal threshold
            if combined_strength < self.strategy_config.signal_threshold:
                combined_signal = SignalType.HOLD
                combined_strength = 0.0
            
            # Calculate prediction premium
            prediction_premium = self.garch_model.calculate_prediction_premium(
                current_price, garch_prediction.predicted_volatility
            )
            
            # Create trading signal
            trading_signal = TradingSignal(
                symbol=self.symbol,
                signal_type=combined_signal,
                strength=combined_strength,
                confidence=combined_confidence,
                timestamp=datetime.now(),
                price=current_price,
                predicted_volatility=garch_prediction.predicted_volatility,
                prediction_premium=prediction_premium,
                technical_indicators=technical_indicators,
                reasoning=f"GARCH: {garch_reasoning} | Technical: {tech_reasoning}"
            )
            
            # Store signal
            self.signal_history.append(trading_signal)
            self.last_signal = trading_signal
            self.total_signals += 1
            
            log_info(f"Generated signal for {self.symbol}: {combined_signal.value} (strength: {combined_strength:.3f})")
            
            return trading_signal
            
        except Exception as e:
            log_error(f"Error generating signal for {self.symbol}: {e}")
            return None
    
    def _combine_signals(self, garch_signal: SignalType, garch_strength: float,
                        tech_signal: SignalType, tech_strength: float) -> Tuple[SignalType, float, float]:
        """Combine GARCH and technical signals"""
        
        # Weight configuration
        garch_weight = 0.6  # GARCH gets higher weight
        tech_weight = 0.4
        
        # Convert signals to numeric values
        signal_values = {
            SignalType.BUY: 1.0,
            SignalType.HOLD: 0.0,
            SignalType.SELL: -1.0
        }
        
        # Calculate weighted signal
        garch_value = signal_values[garch_signal] * garch_strength
        tech_value = signal_values[tech_signal] * tech_strength
        
        combined_value = (garch_value * garch_weight) + (tech_value * tech_weight)
        combined_strength = abs(combined_value)
        
        # Determine final signal
        if combined_value > 0.2:
            final_signal = SignalType.BUY
        elif combined_value < -0.2:
            final_signal = SignalType.SELL
        else:
            final_signal = SignalType.HOLD
        
        # Calculate confidence based on signal agreement
        if garch_signal == tech_signal:
            confidence = min(0.8 + (combined_strength * 0.2), 1.0)
        else:
            confidence = max(0.3, 0.6 - (abs(garch_strength - tech_strength) * 0.3))
        
        return final_signal, combined_strength, confidence
    
    def should_enter_position(self, signal: TradingSignal) -> bool:
        """Check if position should be entered"""
        
        # Don't enter if already in position
        if self.current_position is not None:
            return False
        
        # Check signal strength threshold
        if signal.strength < self.strategy_config.signal_threshold:
            return False
        
        # Check confidence threshold
        if signal.confidence < 0.5:
            return False
        
        # Check if signal is actionable
        if signal.signal_type == SignalType.HOLD:
            return False
        
        # Check market hours
        trading_hours = config.get_trading_hours_config()
        current_time = datetime.now().time()
        
        # Convert time strings to time objects
        from datetime import time
        market_open = time.fromisoformat(trading_hours.market_open)
        market_close = time.fromisoformat(trading_hours.market_close)
        
        # Exclude first and last minutes
        exclude_first = timedelta(minutes=trading_hours.exclude_first_minutes)
        exclude_last = timedelta(minutes=trading_hours.exclude_last_minutes)
        
        adjusted_open = (datetime.combine(datetime.today(), market_open) + exclude_first).time()
        adjusted_close = (datetime.combine(datetime.today(), market_close) - exclude_last).time()
        
        if not (adjusted_open <= current_time <= adjusted_close):
            return False
        
        return True
    
    def should_exit_position(self, current_price: float) -> bool:
        """Check if position should be exited"""
        
        if self.current_position is None:
            return False
        
        # Check hold period timeout
        if self.position_entry_time:
            hold_duration = datetime.now() - self.position_entry_time
            max_hold = timedelta(minutes=self.strategy_config.hold_period_minutes)
            
            if hold_duration > max_hold:
                return True
        
        # Check if near market close
        trading_hours = config.get_trading_hours_config()
        current_time = datetime.now().time()
        
        from datetime import time
        market_close = time.fromisoformat(trading_hours.market_close)
        exclude_last = timedelta(minutes=trading_hours.exclude_last_minutes)
        
        adjusted_close = (datetime.combine(datetime.today(), market_close) - exclude_last).time()
        
        if current_time >= adjusted_close:
            return True
        
        return False
    
    def update_position(self, position_info: Dict[str, Any]):
        """Update current position information"""
        self.current_position = position_info
        
        if position_info is None:
            self.position_entry_time = None
        else:
            self.position_entry_time = position_info.get('entry_time', datetime.now())
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        
        if self.total_signals == 0:
            return {
                'total_signals': 0,
                'success_rate': 0.0,
                'average_strength': 0.0,
                'average_confidence': 0.0
            }
        
        # Calculate metrics
        success_rate = self.successful_signals / self.total_signals
        
        strengths = [s.strength for s in self.signal_history]
        confidences = [s.confidence for s in self.signal_history]
        
        avg_strength = np.mean(strengths) if strengths else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Signal type distribution
        signal_types = [s.signal_type.value for s in self.signal_history]
        signal_distribution = pd.Series(signal_types).value_counts().to_dict()
        
        return {
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'failed_signals': self.failed_signals,
            'success_rate': success_rate,
            'average_strength': avg_strength,
            'average_confidence': avg_confidence,
            'signal_distribution': signal_distribution,
            'last_signal_time': self.last_signal.timestamp if self.last_signal else None,
            'current_position': self.current_position
        }
    
    def reset_strategy(self):
        """Reset strategy state"""
        self.signal_history = []
        self.last_signal = None
        self.current_position = None
        self.position_entry_time = None
        self.total_signals = 0
        self.successful_signals = 0
        self.failed_signals = 0
        
        log_info(f"Strategy reset for {self.symbol}")


class StrategyManager:
    """
    Manager for multiple trading strategies
    """
    
    def __init__(self):
        self.strategies = {}
        self.active_symbols = config.get_symbols().get('watchlist', [])
        
        # Initialize strategies for all symbols
        for symbol in self.active_symbols:
            self.strategies[symbol] = GarchTradingStrategy(symbol)
        
        log_info(f"Strategy manager initialized with {len(self.strategies)} strategies")
    
    def get_strategy(self, symbol: str) -> GarchTradingStrategy:
        """Get strategy for symbol"""
        if symbol not in self.strategies:
            self.strategies[symbol] = GarchTradingStrategy(symbol)
        
        return self.strategies[symbol]
    
    def update_all_strategies(self, market_data: Dict[str, MarketDataPoint]):
        """Update all strategies with new market data"""
        for symbol, data_point in market_data.items():
            if symbol in self.strategies:
                self.strategies[symbol].update_market_data(data_point)
    
    def generate_all_signals(self, current_prices: Dict[str, float]) -> Dict[str, Optional[TradingSignal]]:
        """Generate signals for all strategies"""
        signals = {}
        
        for symbol, price in current_prices.items():
            if symbol in self.strategies:
                signal = self.strategies[symbol].generate_signal(price)
                signals[symbol] = signal
        
        return signals
    
    def get_actionable_signals(self, current_prices: Dict[str, float]) -> Dict[str, TradingSignal]:
        """Get signals that should result in trading actions"""
        actionable_signals = {}
        
        all_signals = self.generate_all_signals(current_prices)
        
        for symbol, signal in all_signals.items():
            if signal and self.strategies[symbol].should_enter_position(signal):
                actionable_signals[symbol] = signal
        
        return actionable_signals
    
    def get_exit_signals(self, current_prices: Dict[str, float]) -> List[str]:
        """Get symbols that should exit positions"""
        exit_signals = []
        
        for symbol, price in current_prices.items():
            if symbol in self.strategies:
                if self.strategies[symbol].should_exit_position(price):
                    exit_signals.append(symbol)
        
        return exit_signals
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies"""
        summary = {}
        
        for symbol, strategy in self.strategies.items():
            summary[symbol] = strategy.get_strategy_performance()
        
        return summary


# Global strategy manager instance
strategy_manager = StrategyManager()