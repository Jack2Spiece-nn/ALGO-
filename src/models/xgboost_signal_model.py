"""
XGBoost Feature-Based Signal Generation Model

This module implements an XGBoost-based model for generating trading signals
based on engineered features from market data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path

from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug


@dataclass
class XGBoostSignal:
    """XGBoost model signal result"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    signal_probability: float
    feature_importance: Dict[str, float]
    confidence_score: float
    model_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'signal_type': self.signal_type,
            'signal_probability': self.signal_probability,
            'feature_importance': self.feature_importance,
            'confidence_score': self.confidence_score,
            'model_metrics': self.model_metrics
        }


class XGBoostSignalModel:
    """
    XGBoost-based trading signal generation model
    
    This model uses engineered features to predict optimal trading signals
    using gradient boosting.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_config = config.get_strategy_config()
        
        # Model parameters
        self.lookback_window = 100  # Number of periods to look back
        self.prediction_horizon = 5  # Periods ahead to predict
        self.min_signal_strength = 0.6  # Minimum probability for signal
        
        # XGBoost parameters
        self.xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # BUY, SELL, HOLD
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Model objects
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Training data
        self.feature_names = []
        self.training_features = None
        self.training_targets = None
        
        # Performance tracking
        self.last_fit_time = None
        self.model_metrics = {}
        self.feature_importance = {}
        
        # Prediction history
        self.prediction_history = []
        
        # Model persistence
        self.model_dir = Path("models/xgboost")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        log_info(f"XGBoost signal model initialized for {symbol}")
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare comprehensive features for XGBoost training
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        features = data.copy()
        
        # Price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['price_change'] = (features['close'] - features['open']) / features['open']
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_open_ratio'] = features['close'] / features['open']
        features['volume_price_trend'] = features['volume'] * features['returns']
        
        # Moving averages and ratios
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = features['close'].rolling(window).mean()
            features[f'ema_{window}'] = features['close'].ewm(span=window).mean()
            features[f'price_sma_ratio_{window}'] = features['close'] / features[f'sma_{window}']
            features[f'volume_sma_{window}'] = features['volume'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = features['volume'] / features[f'volume_sma_{window}']
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'parkinson_vol_{window}'] = np.sqrt(
                (np.log(features['high'] / features['low']) ** 2).rolling(window).mean() / (4 * np.log(2))
            )
        
        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(features['close'], 14)
        features['rsi_7'] = self._calculate_rsi(features['close'], 7)
        
        # MACD
        macd, macd_signal, macd_histogram = self._calculate_macd(features['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(features['close'])
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (features['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic oscillator
        stoch_k, stoch_d = self._calculate_stochastic(features['high'], features['low'], features['close'])
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        # Williams %R
        features['williams_r'] = self._calculate_williams_r(features['high'], features['low'], features['close'])
        
        # Average True Range
        features['atr'] = self._calculate_atr(features['high'], features['low'], features['close'])
        
        # Commodity Channel Index
        features['cci'] = self._calculate_cci(features['high'], features['low'], features['close'])
        
        # Money Flow Index
        features['mfi'] = self._calculate_mfi(features['high'], features['low'], features['close'], features['volume'])
        
        # Rate of Change
        for period in [1, 3, 5, 10]:
            features[f'roc_{period}'] = features['close'].pct_change(period)
        
        # Momentum indicators
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = features['close'] / features['close'].shift(period)
        
        # Volume indicators
        features['ad_line'] = self._calculate_ad_line(features['high'], features['low'], features['close'], features['volume'])
        features['obv'] = self._calculate_obv(features['close'], features['volume'])
        
        # Time-based features
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        features['is_market_open'] = (features.index.hour >= 9) & (features.index.hour < 16)
        features['is_end_of_day'] = (features.index.hour >= 15) & (features.index.hour < 16)
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_change_lag_{lag}'] = features['volume'].pct_change().shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi_14'].shift(lag)
        
        # Statistical features
        for window in [10, 20]:
            features[f'returns_skew_{window}'] = features['returns'].rolling(window).skew()
            features[f'returns_kurt_{window}'] = features['returns'].rolling(window).kurt()
            features[f'returns_quantile_25_{window}'] = features['returns'].rolling(window).quantile(0.25)
            features[f'returns_quantile_75_{window}'] = features['returns'].rolling(window).quantile(0.75)
        
        # Drop original OHLCV columns
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        features = features.drop([col for col in original_cols if col in features.columns], axis=1)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci
    
    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return mfi
    
    def _calculate_ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        
        return ad_line
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = volume.copy()
        obv[close < close.shift()] *= -1
        obv = obv.cumsum()
        
        return obv
    
    def _prepare_targets(self, data: pd.DataFrame) -> pd.Series:
        """
        Prepare target labels for classification
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with target labels (0=SELL, 1=HOLD, 2=BUY)
        """
        returns = data['close'].pct_change()
        
        # Calculate forward returns
        forward_returns = returns.shift(-self.prediction_horizon)
        
        # Define thresholds for signal generation
        buy_threshold = 0.01  # 1% gain threshold
        sell_threshold = -0.01  # 1% loss threshold
        
        # Create labels
        labels = pd.Series(1, index=forward_returns.index)  # Default to HOLD
        labels[forward_returns > buy_threshold] = 2  # BUY
        labels[forward_returns < sell_threshold] = 0  # SELL
        
        return labels
    
    def fit(self, data: pd.DataFrame) -> bool:
        """
        Fit XGBoost model to market data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            bool: True if fitting successful
        """
        try:
            log_info(f"Starting XGBoost model training for {self.symbol}")
            
            # Prepare features and targets
            features_df = self._prepare_features(data)
            targets = self._prepare_targets(data)
            
            # Remove NaN values
            valid_mask = ~(features_df.isna().any(axis=1) | targets.isna())
            features_df = features_df[valid_mask]
            targets = targets[valid_mask]
            
            if len(features_df) < 200:
                log_error(f"Insufficient data for XGBoost training: {len(features_df)} observations")
                return False
            
            # Store feature names
            self.feature_names = list(features_df.columns)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_df)
            features_scaled = pd.DataFrame(features_scaled, columns=self.feature_names, index=features_df.index)
            
            # Store training data
            self.training_features = features_scaled
            self.training_targets = targets
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(**self.xgb_params)
            
            # Fit model
            self.model.fit(features_scaled, targets)
            
            self.last_fit_time = datetime.now()
            
            # Calculate model metrics
            self._calculate_model_metrics()
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
            # Save model
            self._save_model()
            
            log_info(f"XGBoost model trained successfully for {self.symbol}")
            return True
            
        except Exception as e:
            log_error(f"Error training XGBoost model for {self.symbol}: {e}")
            return False
    
    def _calculate_model_metrics(self):
        """Calculate model performance metrics"""
        if self.model is None or self.training_features is None:
            return
        
        try:
            # Split data for validation
            split_idx = int(len(self.training_features) * 0.8)
            X_train = self.training_features.iloc[:split_idx]
            X_val = self.training_features.iloc[split_idx:]
            y_train = self.training_targets.iloc[:split_idx]
            y_val = self.training_targets.iloc[split_idx:]
            
            # Predict on validation set
            val_predictions = self.model.predict(X_val)
            val_probabilities = self.model.predict_proba(X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, val_predictions)
            precision = precision_score(y_val, val_predictions, average='weighted', zero_division=0)
            recall = recall_score(y_val, val_predictions, average='weighted', zero_division=0)
            
            # Calculate class distribution
            class_distribution = y_train.value_counts(normalize=True).to_dict()
            
            # Store metrics
            self.model_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'class_distribution': class_distribution,
                'feature_count': len(self.feature_names)
            }
            
            log_info(f"XGBoost model metrics calculated for {self.symbol}: Accuracy={accuracy:.4f}, Precision={precision:.4f}")
            
        except Exception as e:
            log_error(f"Error calculating XGBoost model metrics: {e}")
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        if self.model is None:
            return
        
        try:
            # Get feature importance
            importance_scores = self.model.feature_importances_
            
            # Create importance dictionary
            self.feature_importance = dict(zip(self.feature_names, importance_scores))
            
            # Sort by importance
            self.feature_importance = dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            log_debug(f"Feature importance calculated for {self.symbol}")
            
        except Exception as e:
            log_error(f"Error calculating feature importance: {e}")
    
    def predict(self, data: pd.DataFrame) -> Optional[XGBoostSignal]:
        """
        Generate trading signal prediction
        
        Args:
            data: Recent market data
            
        Returns:
            XGBoostSignal object or None
        """
        if self.model is None:
            log_error(f"XGBoost model not fitted for {self.symbol}")
            return None
        
        try:
            # Prepare features
            features_df = self._prepare_features(data)
            
            if len(features_df) < self.lookback_window:
                log_error(f"Insufficient data for prediction: {len(features_df)} < {self.lookback_window}")
                return None
            
            # Take the last observation
            last_features = features_df.iloc[-1:].copy()
            
            # Scale features
            last_features_scaled = self.scaler.transform(last_features)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(last_features_scaled)[0]
            prediction_class = self.model.predict(last_features_scaled)[0]
            
            # Map prediction to signal type
            signal_mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal_type = signal_mapping[prediction_class]
            signal_probability = prediction_proba[prediction_class]
            
            # Calculate confidence score
            confidence = max(prediction_proba) - np.median(prediction_proba)
            
            # Only generate signal if probability is high enough
            if signal_probability < self.min_signal_strength and signal_type != 'HOLD':
                signal_type = 'HOLD'
                signal_probability = prediction_proba[1]  # HOLD probability
            
            # Get top features for this prediction
            top_features = dict(list(self.feature_importance.items())[:10])
            
            # Create signal object
            xgb_signal = XGBoostSignal(
                symbol=self.symbol,
                timestamp=datetime.now(),
                signal_type=signal_type,
                signal_probability=signal_probability,
                feature_importance=top_features,
                confidence_score=confidence,
                model_metrics=self.model_metrics
            )
            
            # Store prediction
            self.prediction_history.append(xgb_signal)
            
            log_debug(f"XGBoost signal for {self.symbol}: {signal_type} (prob: {signal_probability:.4f})")
            
            return xgb_signal
            
        except Exception as e:
            log_error(f"Error generating XGBoost signal for {self.symbol}: {e}")
            return None
    
    def _save_model(self):
        """Save model and scalers"""
        try:
            # Save XGBoost model
            model_path = self.model_dir / f"{self.symbol}_xgboost_model.pkl"
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = self.model_dir / f"{self.symbol}_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            
            # Save model metadata
            metadata = {
                'symbol': self.symbol,
                'feature_names': self.feature_names,
                'last_fit_time': self.last_fit_time,
                'model_metrics': self.model_metrics,
                'feature_importance': self.feature_importance,
                'xgb_params': self.xgb_params
            }
            
            metadata_path = self.model_dir / f"{self.symbol}_xgboost_metadata.pkl"
            joblib.dump(metadata, metadata_path)
            
            log_info(f"XGBoost model saved for {self.symbol}")
            
        except Exception as e:
            log_error(f"Error saving XGBoost model: {e}")
    
    def load_model(self) -> bool:
        """Load saved model and scalers"""
        try:
            # Load XGBoost model
            model_path = self.model_dir / f"{self.symbol}_xgboost_model.pkl"
            if not model_path.exists():
                return False
            
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = self.model_dir / f"{self.symbol}_scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = self.model_dir / f"{self.symbol}_xgboost_metadata.pkl"
            metadata = joblib.load(metadata_path)
            
            self.feature_names = metadata['feature_names']
            self.last_fit_time = metadata['last_fit_time']
            self.model_metrics = metadata['model_metrics']
            self.feature_importance = metadata['feature_importance']
            
            log_info(f"XGBoost model loaded for {self.symbol}")
            return True
            
        except Exception as e:
            log_error(f"Error loading XGBoost model: {e}")
            return False
    
    def needs_refit(self) -> bool:
        """Check if model needs refitting"""
        if self.model is None or self.last_fit_time is None:
            return True
        
        # Refit every 3 days
        days_since_fit = (datetime.now() - self.last_fit_time).days
        return days_since_fit >= 3
    
    def get_model_summary(self) -> str:
        """Get model summary"""
        if self.model is None:
            return "XGBoost model not fitted"
        
        summary = []
        summary.append(f"XGBoost Model for {self.symbol}")
        summary.append(f"Features: {len(self.feature_names)}")
        summary.append(f"Last Fit: {self.last_fit_time}")
        summary.append(f"Metrics: {self.model_metrics}")
        summary.append(f"Top Features: {dict(list(self.feature_importance.items())[:5])}")
        
        return "\n".join(summary)


class XGBoostModelManager:
    """Manager for XGBoost signal models"""
    
    def __init__(self):
        self.models = {}
        log_info("XGBoost model manager initialized")
    
    def get_model(self, symbol: str) -> XGBoostSignalModel:
        """Get or create XGBoost model for symbol"""
        if symbol not in self.models:
            self.models[symbol] = XGBoostSignalModel(symbol)
            
            # Try to load existing model
            if not self.models[symbol].load_model():
                log_debug(f"No existing XGBoost model found for {symbol}")
        
        return self.models[symbol]
    
    def fit_all_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Fit all XGBoost models"""
        results = {}
        
        for symbol, data in data_dict.items():
            model = self.get_model(symbol)
            success = model.fit(data)
            results[symbol] = success
        
        return results
    
    def predict_all(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Optional[XGBoostSignal]]:
        """Generate signals for all symbols"""
        signals = {}
        
        for symbol, data in data_dict.items():
            model = self.get_model(symbol)
            signal = model.predict(data)
            signals[symbol] = signal
        
        return signals
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        performance = {}
        
        for symbol, model in self.models.items():
            if model.model_metrics:
                performance[symbol] = model.model_metrics
        
        return performance


# Global XGBoost model manager instance
xgboost_manager = XGBoostModelManager()