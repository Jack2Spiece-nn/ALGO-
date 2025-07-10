"""
LSTM Neural Network for Volatility Prediction

This module implements an LSTM-based volatility forecasting model that complements
the existing GARCH models for ensemble predictions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug


@dataclass
class LSTMPrediction:
    """LSTM model prediction result"""
    symbol: str
    timestamp: datetime
    predicted_volatility: float
    confidence_score: float
    prediction_horizon: int
    model_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'predicted_volatility': self.predicted_volatility,
            'confidence_score': self.confidence_score,
            'prediction_horizon': self.prediction_horizon,
            'model_metrics': self.model_metrics
        }


class LSTMVolatilityModel:
    """
    LSTM-based volatility prediction model
    
    This model uses past price movements, volume, and technical indicators
    to predict future volatility using deep learning.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_config = config.get_garch_config()  # We'll extend this config
        
        # Model parameters
        self.sequence_length = 60  # Use 60 periods for prediction
        self.features_dim = 0  # Will be set based on features
        self.batch_size = 32
        self.epochs = 100
        self.validation_split = 0.2
        
        # Model objects
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_target = MinMaxScaler()
        
        # Training data
        self.training_features = None
        self.training_targets = None
        self.validation_features = None
        self.validation_targets = None
        
        # Performance tracking
        self.training_history = None
        self.last_fit_time = None
        self.model_metrics = {}
        
        # Prediction history
        self.prediction_history = []
        
        # Model persistence
        self.model_dir = Path("models/lstm")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        log_info(f"LSTM volatility model initialized for {symbol}")
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for LSTM training
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        features = data.copy()
        
        # Price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_open_ratio'] = features['close'] / features['open']
        
        # Volatility features
        features['rolling_std_5'] = features['returns'].rolling(5).std()
        features['rolling_std_10'] = features['returns'].rolling(10).std()
        features['rolling_std_20'] = features['returns'].rolling(20).std()
        features['parkinson_vol'] = np.sqrt(
            np.log(features['high'] / features['low']) ** 2 / (4 * np.log(2))
        )
        
        # Volume features
        features['volume_change'] = features['volume'].pct_change()
        features['volume_ma_5'] = features['volume'].rolling(5).mean()
        features['volume_ma_20'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma_20']
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(features['close'])
        features['macd'], features['macd_signal'] = self._calculate_macd(features['close'])
        features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(features['close'])
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['close']
        
        # Time-based features
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume_change'].shift(lag)
        
        # Drop unnecessary columns
        features = features.drop(['open', 'high', 'low', 'close', 'volume'], axis=1)
        
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
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band
    
    def _prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Prepare target variable (future volatility)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with target volatility
        """
        returns = data['close'].pct_change()
        
        # Calculate forward-looking volatility (next period realized volatility)
        target_vol = returns.rolling(window=5).std().shift(-5)
        
        return target_vol
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            features: Feature matrix
            targets: Target vector
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input sequences
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')  # Output volatility (0-1 range)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def fit(self, data: pd.DataFrame) -> bool:
        """
        Fit LSTM model to market data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            bool: True if fitting successful
        """
        try:
            log_info(f"Starting LSTM model training for {self.symbol}")
            
            # Prepare features and targets
            features_df = self._prepare_features(data)
            targets = self._prepare_target(data)
            
            # Remove NaN values
            valid_mask = ~(features_df.isna().any(axis=1) | targets.isna())
            features_df = features_df[valid_mask]
            targets = targets[valid_mask]
            
            if len(features_df) < self.sequence_length + 100:
                log_error(f"Insufficient data for LSTM training: {len(features_df)} observations")
                return False
            
            # Scale features and targets
            features_scaled = self.scaler_features.fit_transform(features_df)
            targets_scaled = self.scaler_target.fit_transform(targets.values.reshape(-1, 1)).flatten()
            
            # Create sequences
            X, y = self._create_sequences(features_scaled, targets_scaled)
            
            # Store dimensions
            self.features_dim = X.shape[2]
            
            # Split data
            split_idx = int(len(X) * (1 - self.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Store training data
            self.training_features = X_train
            self.training_targets = y_train
            self.validation_features = X_val
            self.validation_targets = y_val
            
            # Build model
            self.model = self._build_model((self.sequence_length, self.features_dim))
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
            
            # Train model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                history = self.model.fit(
                    X_train, y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
            
            self.training_history = history.history
            self.last_fit_time = datetime.now()
            
            # Calculate model metrics
            self._calculate_model_metrics()
            
            # Save model
            self._save_model()
            
            log_info(f"LSTM model trained successfully for {self.symbol}")
            return True
            
        except Exception as e:
            log_error(f"Error training LSTM model for {self.symbol}: {e}")
            return False
    
    def _calculate_model_metrics(self):
        """Calculate model performance metrics"""
        if self.model is None or self.validation_features is None:
            return
        
        try:
            # Predict on validation set
            val_predictions = self.model.predict(self.validation_features, verbose=0)
            val_predictions = val_predictions.flatten()
            
            # Calculate metrics
            mse = mean_squared_error(self.validation_targets, val_predictions)
            mae = mean_absolute_error(self.validation_targets, val_predictions)
            rmse = np.sqrt(mse)
            
            # Calculate correlation
            correlation = np.corrcoef(self.validation_targets, val_predictions)[0, 1]
            
            # Store metrics
            self.model_metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'correlation': correlation,
                'training_loss': self.training_history['loss'][-1],
                'validation_loss': self.training_history['val_loss'][-1],
                'training_samples': len(self.training_features),
                'validation_samples': len(self.validation_features)
            }
            
            log_info(f"LSTM model metrics calculated for {self.symbol}: RMSE={rmse:.4f}, Correlation={correlation:.4f}")
            
        except Exception as e:
            log_error(f"Error calculating LSTM model metrics: {e}")
    
    def predict(self, data: pd.DataFrame, horizon: int = 1) -> Optional[LSTMPrediction]:
        """
        Generate volatility prediction
        
        Args:
            data: Recent market data
            horizon: Prediction horizon
            
        Returns:
            LSTMPrediction object or None
        """
        if self.model is None:
            log_error(f"LSTM model not fitted for {self.symbol}")
            return None
        
        try:
            # Prepare features
            features_df = self._prepare_features(data)
            
            # Take the last sequence
            features_scaled = self.scaler_features.transform(features_df)
            
            if len(features_scaled) < self.sequence_length:
                log_error(f"Insufficient data for prediction: {len(features_scaled)} < {self.sequence_length}")
                return None
            
            # Get last sequence
            last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, self.features_dim)
            
            # Make prediction
            prediction_scaled = self.model.predict(last_sequence, verbose=0)[0, 0]
            
            # Inverse transform prediction
            prediction = self.scaler_target.inverse_transform([[prediction_scaled]])[0, 0]
            
            # Calculate confidence score (based on model performance)
            confidence = max(0.1, min(0.9, self.model_metrics.get('correlation', 0.5)))
            
            # Create prediction object
            lstm_prediction = LSTMPrediction(
                symbol=self.symbol,
                timestamp=datetime.now(),
                predicted_volatility=prediction,
                confidence_score=confidence,
                prediction_horizon=horizon,
                model_metrics=self.model_metrics
            )
            
            # Store prediction
            self.prediction_history.append(lstm_prediction)
            
            log_debug(f"LSTM prediction for {self.symbol}: {prediction:.4f}")
            
            return lstm_prediction
            
        except Exception as e:
            log_error(f"Error generating LSTM prediction for {self.symbol}: {e}")
            return None
    
    def _save_model(self):
        """Save model and scalers"""
        try:
            # Save Keras model
            model_path = self.model_dir / f"{self.symbol}_lstm_model.h5"
            self.model.save(str(model_path))
            
            # Save scalers
            scaler_features_path = self.model_dir / f"{self.symbol}_features_scaler.pkl"
            scaler_target_path = self.model_dir / f"{self.symbol}_target_scaler.pkl"
            
            joblib.dump(self.scaler_features, scaler_features_path)
            joblib.dump(self.scaler_target, scaler_target_path)
            
            # Save model metadata
            metadata = {
                'symbol': self.symbol,
                'sequence_length': self.sequence_length,
                'features_dim': self.features_dim,
                'last_fit_time': self.last_fit_time,
                'model_metrics': self.model_metrics
            }
            
            metadata_path = self.model_dir / f"{self.symbol}_lstm_metadata.pkl"
            joblib.dump(metadata, metadata_path)
            
            log_info(f"LSTM model saved for {self.symbol}")
            
        except Exception as e:
            log_error(f"Error saving LSTM model: {e}")
    
    def load_model(self) -> bool:
        """Load saved model and scalers"""
        try:
            # Load Keras model
            model_path = self.model_dir / f"{self.symbol}_lstm_model.h5"
            if not model_path.exists():
                return False
            
            self.model = tf.keras.models.load_model(str(model_path))
            
            # Load scalers
            scaler_features_path = self.model_dir / f"{self.symbol}_features_scaler.pkl"
            scaler_target_path = self.model_dir / f"{self.symbol}_target_scaler.pkl"
            
            self.scaler_features = joblib.load(scaler_features_path)
            self.scaler_target = joblib.load(scaler_target_path)
            
            # Load metadata
            metadata_path = self.model_dir / f"{self.symbol}_lstm_metadata.pkl"
            metadata = joblib.load(metadata_path)
            
            self.sequence_length = metadata['sequence_length']
            self.features_dim = metadata['features_dim']
            self.last_fit_time = metadata['last_fit_time']
            self.model_metrics = metadata['model_metrics']
            
            log_info(f"LSTM model loaded for {self.symbol}")
            return True
            
        except Exception as e:
            log_error(f"Error loading LSTM model: {e}")
            return False
    
    def needs_refit(self) -> bool:
        """Check if model needs refitting"""
        if self.model is None or self.last_fit_time is None:
            return True
        
        # Refit every 7 days
        days_since_fit = (datetime.now() - self.last_fit_time).days
        return days_since_fit >= 7
    
    def get_model_summary(self) -> str:
        """Get model summary"""
        if self.model is None:
            return "LSTM model not fitted"
        
        summary = []
        summary.append(f"LSTM Model for {self.symbol}")
        summary.append(f"Architecture: {self.model.count_params()} parameters")
        summary.append(f"Sequence Length: {self.sequence_length}")
        summary.append(f"Features: {self.features_dim}")
        summary.append(f"Last Fit: {self.last_fit_time}")
        summary.append(f"Metrics: {self.model_metrics}")
        
        return "\n".join(summary)


class LSTMModelManager:
    """Manager for LSTM volatility models"""
    
    def __init__(self):
        self.models = {}
        log_info("LSTM model manager initialized")
    
    def get_model(self, symbol: str) -> LSTMVolatilityModel:
        """Get or create LSTM model for symbol"""
        if symbol not in self.models:
            self.models[symbol] = LSTMVolatilityModel(symbol)
            
            # Try to load existing model
            if not self.models[symbol].load_model():
                log_debug(f"No existing LSTM model found for {symbol}")
        
        return self.models[symbol]
    
    def fit_all_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Fit all LSTM models"""
        results = {}
        
        for symbol, data in data_dict.items():
            model = self.get_model(symbol)
            success = model.fit(data)
            results[symbol] = success
        
        return results
    
    def predict_all(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Optional[LSTMPrediction]]:
        """Generate predictions for all symbols"""
        predictions = {}
        
        for symbol, data in data_dict.items():
            model = self.get_model(symbol)
            prediction = model.predict(data)
            predictions[symbol] = prediction
        
        return predictions
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        performance = {}
        
        for symbol, model in self.models.items():
            if model.model_metrics:
                performance[symbol] = model.model_metrics
        
        return performance


# Global LSTM model manager instance
lstm_manager = LSTMModelManager()