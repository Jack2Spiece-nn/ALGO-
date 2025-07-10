"""
Ensemble Model for Combining GARCH, LSTM, and XGBoost Predictions

This module implements an advanced ensemble system that combines predictions
from multiple models to generate superior trading signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
from enum import Enum
import joblib
from pathlib import Path

from src.models.garch_model import GarchModel, GarchPrediction, garch_manager
from src.models.lstm_volatility_model import LSTMVolatilityModel, LSTMPrediction, lstm_manager
from src.models.xgboost_signal_model import XGBoostSignalModel, XGBoostSignal, xgboost_manager
from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug


class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    ADAPTIVE_WEIGHTS = "adaptive_weights"


@dataclass
class EnsembleSignal:
    """Ensemble model signal result"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    signal_strength: float  # 0-1
    confidence: float  # 0-1
    predicted_volatility: float
    prediction_premium: float
    
    # Individual model contributions
    garch_prediction: Optional[GarchPrediction]
    lstm_prediction: Optional[LSTMPrediction]
    xgboost_signal: Optional[XGBoostSignal]
    
    # Ensemble metadata
    ensemble_weights: Dict[str, float]
    model_agreement: float  # How much models agree (0-1)
    ensemble_method: str
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'signal_type': self.signal_type,
            'signal_strength': self.signal_strength,
            'confidence': self.confidence,
            'predicted_volatility': self.predicted_volatility,
            'prediction_premium': self.prediction_premium,
            'garch_prediction': self.garch_prediction.to_dict() if self.garch_prediction else None,
            'lstm_prediction': self.lstm_prediction.to_dict() if self.lstm_prediction else None,
            'xgboost_signal': self.xgboost_signal.to_dict() if self.xgboost_signal else None,
            'ensemble_weights': self.ensemble_weights,
            'model_agreement': self.model_agreement,
            'ensemble_method': self.ensemble_method,
            'reasoning': self.reasoning
        }


class EnsembleModel:
    """
    Advanced ensemble model combining GARCH, LSTM, and XGBoost predictions
    
    This model uses multiple combination strategies to generate robust trading signals
    and volatility predictions.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_config = config.get_strategy_config()
        
        # Individual models
        self.garch_model = garch_manager.get_model(symbol)
        self.lstm_model = lstm_manager.get_model(symbol)
        self.xgboost_model = xgboost_manager.get_model(symbol)
        
        # Ensemble configuration
        self.ensemble_method = EnsembleMethod.ADAPTIVE_WEIGHTS
        self.min_models_required = 2  # Minimum models needed for ensemble
        
        # Model weights (adaptive)
        self.model_weights = {
            'garch': 0.4,
            'lstm': 0.3,
            'xgboost': 0.3
        }
        
        # Performance tracking for adaptive weights
        self.model_performance_history = {
            'garch': [],
            'lstm': [],
            'xgboost': []
        }
        
        # Ensemble performance tracking
        self.ensemble_history = []
        self.last_update_time = None
        
        # Model persistence
        self.model_dir = Path("models/ensemble")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        log_info(f"Ensemble model initialized for {symbol}")
    
    def _calculate_model_agreement(self, garch_pred: Optional[GarchPrediction],
                                 lstm_pred: Optional[LSTMPrediction],
                                 xgb_signal: Optional[XGBoostSignal]) -> float:
        """
        Calculate agreement between models
        
        Returns:
            Agreement score (0-1)
        """
        predictions = []
        
        # Convert predictions to standardized format
        if garch_pred:
            # Convert GARCH volatility prediction to signal strength
            vol_signal = min(max(garch_pred.predicted_volatility * 10, 0), 1)
            predictions.append(vol_signal)
        
        if lstm_pred:
            # LSTM volatility prediction
            vol_signal = min(max(lstm_pred.predicted_volatility * 10, 0), 1)
            predictions.append(vol_signal)
        
        if xgb_signal:
            # XGBoost signal probability
            if xgb_signal.signal_type == 'BUY':
                predictions.append(xgb_signal.signal_probability)
            elif xgb_signal.signal_type == 'SELL':
                predictions.append(1 - xgb_signal.signal_probability)
            else:  # HOLD
                predictions.append(0.5)
        
        if len(predictions) < 2:
            return 0.5  # Neutral agreement if insufficient predictions
        
        # Calculate agreement as inverse of standard deviation
        agreement = 1 - (np.std(predictions) / np.mean(predictions) if np.mean(predictions) > 0 else 1)
        return max(0, min(1, agreement))
    
    def _adaptive_weight_update(self, model_performance: Dict[str, float]):
        """
        Update model weights based on recent performance
        
        Args:
            model_performance: Performance scores for each model
        """
        try:
            # Update performance history
            for model, performance in model_performance.items():
                if model in self.model_performance_history:
                    self.model_performance_history[model].append(performance)
                    # Keep only last 50 performance scores
                    if len(self.model_performance_history[model]) > 50:
                        self.model_performance_history[model] = self.model_performance_history[model][-50:]
            
            # Calculate average performance
            avg_performance = {}
            for model, history in self.model_performance_history.items():
                if history:
                    avg_performance[model] = np.mean(history[-10:])  # Use last 10 scores
                else:
                    avg_performance[model] = 0.5  # Default performance
            
            # Calculate new weights based on performance
            total_performance = sum(avg_performance.values())
            if total_performance > 0:
                for model in self.model_weights:
                    if model in avg_performance:
                        self.model_weights[model] = avg_performance[model] / total_performance
            
            # Apply minimum weight constraints
            min_weight = 0.1
            for model in self.model_weights:
                self.model_weights[model] = max(min_weight, self.model_weights[model])
            
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            for model in self.model_weights:
                self.model_weights[model] /= total_weight
            
            log_debug(f"Updated ensemble weights for {self.symbol}: {self.model_weights}")
            
        except Exception as e:
            log_error(f"Error updating adaptive weights: {e}")
    
    def _combine_volatility_predictions(self, garch_pred: Optional[GarchPrediction],
                                      lstm_pred: Optional[LSTMPrediction]) -> Tuple[float, float]:
        """
        Combine volatility predictions from GARCH and LSTM
        
        Returns:
            Tuple of (combined_volatility, confidence)
        """
        predictions = []
        weights = []
        confidences = []
        
        if garch_pred:
            predictions.append(garch_pred.predicted_volatility)
            weights.append(self.model_weights['garch'])
            # GARCH confidence based on model metrics
            garch_confidence = 0.8  # Default high confidence for GARCH
            confidences.append(garch_confidence)
        
        if lstm_pred:
            predictions.append(lstm_pred.predicted_volatility)
            weights.append(self.model_weights['lstm'])
            confidences.append(lstm_pred.confidence_score)
        
        if not predictions:
            return 0.02, 0.1  # Default volatility and low confidence
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        combined_volatility = np.average(predictions, weights=weights)
        combined_confidence = np.average(confidences, weights=weights)
        
        return combined_volatility, combined_confidence
    
    def _combine_signals(self, garch_pred: Optional[GarchPrediction],
                        lstm_pred: Optional[LSTMPrediction],
                        xgb_signal: Optional[XGBoostSignal],
                        current_price: float) -> Tuple[str, float, str]:
        """
        Combine signals from all models
        
        Returns:
            Tuple of (signal_type, signal_strength, reasoning)
        """
        signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        reasoning_parts = []
        
        # GARCH contribution
        if garch_pred:
            garch_weight = self.model_weights['garch']
            
            # High volatility typically suggests caution (SELL/HOLD)
            if garch_pred.predicted_volatility > 0.02:
                signal_scores['SELL'] += garch_weight * 0.6
                signal_scores['HOLD'] += garch_weight * 0.4
                reasoning_parts.append(f"GARCH: High vol ({garch_pred.predicted_volatility:.4f}) suggests caution")
            else:
                signal_scores['BUY'] += garch_weight * 0.7
                signal_scores['HOLD'] += garch_weight * 0.3
                reasoning_parts.append(f"GARCH: Low vol ({garch_pred.predicted_volatility:.4f}) suggests opportunity")
        
        # LSTM contribution
        if lstm_pred:
            lstm_weight = self.model_weights['lstm']
            
            # Similar logic to GARCH for volatility prediction
            if lstm_pred.predicted_volatility > 0.02:
                signal_scores['SELL'] += lstm_weight * 0.6
                signal_scores['HOLD'] += lstm_weight * 0.4
                reasoning_parts.append(f"LSTM: High vol predicted ({lstm_pred.predicted_volatility:.4f})")
            else:
                signal_scores['BUY'] += lstm_weight * 0.7
                signal_scores['HOLD'] += lstm_weight * 0.3
                reasoning_parts.append(f"LSTM: Low vol predicted ({lstm_pred.predicted_volatility:.4f})")
        
        # XGBoost contribution
        if xgb_signal:
            xgb_weight = self.model_weights['xgboost']
            
            if xgb_signal.signal_type == 'BUY':
                signal_scores['BUY'] += xgb_weight * xgb_signal.signal_probability
                reasoning_parts.append(f"XGBoost: {xgb_signal.signal_type} ({xgb_signal.signal_probability:.3f})")
            elif xgb_signal.signal_type == 'SELL':
                signal_scores['SELL'] += xgb_weight * xgb_signal.signal_probability
                reasoning_parts.append(f"XGBoost: {xgb_signal.signal_type} ({xgb_signal.signal_probability:.3f})")
            else:
                signal_scores['HOLD'] += xgb_weight * xgb_signal.signal_probability
                reasoning_parts.append(f"XGBoost: {xgb_signal.signal_type} ({xgb_signal.signal_probability:.3f})")
        
        # Determine final signal
        max_signal = max(signal_scores.keys(), key=lambda x: signal_scores[x])
        signal_strength = signal_scores[max_signal]
        
        # Apply minimum threshold
        min_threshold = 0.3
        if signal_strength < min_threshold:
            max_signal = 'HOLD'
            signal_strength = signal_scores['HOLD']
        
        reasoning = " | ".join(reasoning_parts)
        
        return max_signal, signal_strength, reasoning
    
    def predict(self, data: pd.DataFrame, current_price: float) -> Optional[EnsembleSignal]:
        """
        Generate ensemble prediction
        
        Args:
            data: Market data for prediction
            current_price: Current asset price
            
        Returns:
            EnsembleSignal object or None
        """
        try:
            log_debug(f"Generating ensemble prediction for {self.symbol}")
            
            # Get predictions from individual models
            garch_pred = None
            lstm_pred = None
            xgb_signal = None
            
            # GARCH prediction
            try:
                if self.garch_model.fitted_model is not None:
                    garch_pred = self.garch_model.predict()
                elif len(data) >= 100:
                    # Try to fit if we have enough data
                    returns = data['close'].pct_change().dropna()
                    if self.garch_model.fit(returns):
                        garch_pred = self.garch_model.predict()
            except Exception as e:
                log_debug(f"GARCH prediction failed: {e}")
            
            # LSTM prediction
            try:
                if self.lstm_model.model is not None:
                    lstm_pred = self.lstm_model.predict(data)
                elif len(data) >= 200:
                    # Try to fit if we have enough data
                    if self.lstm_model.fit(data):
                        lstm_pred = self.lstm_model.predict(data)
            except Exception as e:
                log_debug(f"LSTM prediction failed: {e}")
            
            # XGBoost signal
            try:
                if self.xgboost_model.model is not None:
                    xgb_signal = self.xgboost_model.predict(data)
                elif len(data) >= 200:
                    # Try to fit if we have enough data
                    if self.xgboost_model.fit(data):
                        xgb_signal = self.xgboost_model.predict(data)
            except Exception as e:
                log_debug(f"XGBoost prediction failed: {e}")
            
            # Check if we have minimum required models
            available_models = sum([garch_pred is not None, lstm_pred is not None, xgb_signal is not None])
            if available_models < self.min_models_required:
                log_debug(f"Insufficient models available ({available_models}) for ensemble prediction")
                return None
            
            # Calculate model agreement
            model_agreement = self._calculate_model_agreement(garch_pred, lstm_pred, xgb_signal)
            
            # Combine volatility predictions
            combined_volatility, vol_confidence = self._combine_volatility_predictions(garch_pred, lstm_pred)
            
            # Combine signals
            signal_type, signal_strength, reasoning = self._combine_signals(
                garch_pred, lstm_pred, xgb_signal, current_price
            )
            
            # Calculate prediction premium
            if garch_pred:
                prediction_premium = self.garch_model.calculate_prediction_premium(
                    current_price, combined_volatility
                )
            else:
                prediction_premium = current_price * combined_volatility * np.sqrt(1/252)  # Daily premium
            
            # Calculate overall confidence
            confidence = (vol_confidence + model_agreement + signal_strength) / 3
            
            # Create ensemble signal
            ensemble_signal = EnsembleSignal(
                symbol=self.symbol,
                timestamp=datetime.now(),
                signal_type=signal_type,
                signal_strength=signal_strength,
                confidence=confidence,
                predicted_volatility=combined_volatility,
                prediction_premium=prediction_premium,
                garch_prediction=garch_pred,
                lstm_prediction=lstm_pred,
                xgboost_signal=xgb_signal,
                ensemble_weights=self.model_weights.copy(),
                model_agreement=model_agreement,
                ensemble_method=self.ensemble_method.value,
                reasoning=reasoning
            )
            
            # Store prediction
            self.ensemble_history.append(ensemble_signal)
            
            # Keep only last 1000 predictions
            if len(self.ensemble_history) > 1000:
                self.ensemble_history = self.ensemble_history[-1000:]
            
            self.last_update_time = datetime.now()
            
            log_info(f"Ensemble prediction for {self.symbol}: {signal_type} "
                    f"(strength: {signal_strength:.3f}, confidence: {confidence:.3f})")
            
            return ensemble_signal
            
        except Exception as e:
            log_error(f"Error generating ensemble prediction for {self.symbol}: {e}")
            return None
    
    def fit_all_models(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Fit all component models
        
        Args:
            data: Market data for training
            
        Returns:
            Dictionary of fit results
        """
        results = {}
        
        # Fit GARCH model
        try:
            returns = data['close'].pct_change().dropna()
            results['garch'] = self.garch_model.fit(returns)
        except Exception as e:
            log_error(f"Error fitting GARCH model: {e}")
            results['garch'] = False
        
        # Fit LSTM model
        try:
            results['lstm'] = self.lstm_model.fit(data)
        except Exception as e:
            log_error(f"Error fitting LSTM model: {e}")
            results['lstm'] = False
        
        # Fit XGBoost model
        try:
            results['xgboost'] = self.xgboost_model.fit(data)
        except Exception as e:
            log_error(f"Error fitting XGBoost model: {e}")
            results['xgboost'] = False
        
        log_info(f"Ensemble model fitting results for {self.symbol}: {results}")
        return results
    
    def update_model_performance(self, actual_outcome: float):
        """
        Update model performance based on actual outcome
        
        Args:
            actual_outcome: Actual price movement or volatility
        """
        if not self.ensemble_history:
            return
        
        last_prediction = self.ensemble_history[-1]
        
        # Calculate performance for each model
        model_performance = {}
        
        # GARCH performance (volatility prediction accuracy)
        if last_prediction.garch_prediction:
            garch_error = abs(last_prediction.garch_prediction.predicted_volatility - abs(actual_outcome))
            model_performance['garch'] = max(0, 1 - garch_error * 10)  # Scale error
        
        # LSTM performance (volatility prediction accuracy)
        if last_prediction.lstm_prediction:
            lstm_error = abs(last_prediction.lstm_prediction.predicted_volatility - abs(actual_outcome))
            model_performance['lstm'] = max(0, 1 - lstm_error * 10)  # Scale error
        
        # XGBoost performance (signal correctness)
        if last_prediction.xgboost_signal:
            signal_correct = (
                (last_prediction.xgboost_signal.signal_type == 'BUY' and actual_outcome > 0) or
                (last_prediction.xgboost_signal.signal_type == 'SELL' and actual_outcome < 0) or
                (last_prediction.xgboost_signal.signal_type == 'HOLD' and abs(actual_outcome) < 0.005)
            )
            model_performance['xgboost'] = 1.0 if signal_correct else 0.0
        
        # Update adaptive weights
        if self.ensemble_method == EnsembleMethod.ADAPTIVE_WEIGHTS:
            self._adaptive_weight_update(model_performance)
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get ensemble model summary"""
        summary = {
            'symbol': self.symbol,
            'ensemble_method': self.ensemble_method.value,
            'model_weights': self.model_weights,
            'last_update': self.last_update_time,
            'prediction_count': len(self.ensemble_history),
            'available_models': {
                'garch': self.garch_model.fitted_model is not None,
                'lstm': self.lstm_model.model is not None,
                'xgboost': self.xgboost_model.model is not None
            }
        }
        
        if self.ensemble_history:
            # Calculate recent performance metrics
            recent_predictions = self.ensemble_history[-50:]  # Last 50 predictions
            
            signal_distribution = {}
            for pred in recent_predictions:
                signal_type = pred.signal_type
                signal_distribution[signal_type] = signal_distribution.get(signal_type, 0) + 1
            
            avg_confidence = np.mean([pred.confidence for pred in recent_predictions])
            avg_agreement = np.mean([pred.model_agreement for pred in recent_predictions])
            
            summary.update({
                'signal_distribution': signal_distribution,
                'average_confidence': avg_confidence,
                'average_model_agreement': avg_agreement
            })
        
        return summary
    
    def save_ensemble_state(self):
        """Save ensemble model state"""
        try:
            state = {
                'symbol': self.symbol,
                'model_weights': self.model_weights,
                'model_performance_history': self.model_performance_history,
                'ensemble_method': self.ensemble_method.value,
                'last_update_time': self.last_update_time
            }
            
            state_path = self.model_dir / f"{self.symbol}_ensemble_state.pkl"
            joblib.dump(state, state_path)
            
            log_info(f"Ensemble state saved for {self.symbol}")
            
        except Exception as e:
            log_error(f"Error saving ensemble state: {e}")
    
    def load_ensemble_state(self) -> bool:
        """Load ensemble model state"""
        try:
            state_path = self.model_dir / f"{self.symbol}_ensemble_state.pkl"
            if not state_path.exists():
                return False
            
            state = joblib.load(state_path)
            
            self.model_weights = state['model_weights']
            self.model_performance_history = state['model_performance_history']
            self.ensemble_method = EnsembleMethod(state['ensemble_method'])
            self.last_update_time = state['last_update_time']
            
            log_info(f"Ensemble state loaded for {self.symbol}")
            return True
            
        except Exception as e:
            log_error(f"Error loading ensemble state: {e}")
            return False


class EnsembleModelManager:
    """Manager for ensemble models"""
    
    def __init__(self):
        self.models = {}
        log_info("Ensemble model manager initialized")
    
    def get_model(self, symbol: str) -> EnsembleModel:
        """Get or create ensemble model for symbol"""
        if symbol not in self.models:
            self.models[symbol] = EnsembleModel(symbol)
            
            # Try to load existing state
            if not self.models[symbol].load_ensemble_state():
                log_debug(f"No existing ensemble state found for {symbol}")
        
        return self.models[symbol]
    
    def fit_all_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, bool]]:
        """Fit all ensemble models"""
        results = {}
        
        for symbol, data in data_dict.items():
            model = self.get_model(symbol)
            results[symbol] = model.fit_all_models(data)
        
        return results
    
    def predict_all(self, data_dict: Dict[str, pd.DataFrame], 
                   prices: Dict[str, float]) -> Dict[str, Optional[EnsembleSignal]]:
        """Generate ensemble predictions for all symbols"""
        predictions = {}
        
        for symbol in data_dict.keys():
            if symbol in prices:
                model = self.get_model(symbol)
                prediction = model.predict(data_dict[symbol], prices[symbol])
                predictions[symbol] = prediction
        
        return predictions
    
    def get_ensemble_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all ensemble models"""
        performance = {}
        
        for symbol, model in self.models.items():
            performance[symbol] = model.get_ensemble_summary()
        
        return performance
    
    def save_all_states(self):
        """Save state for all ensemble models"""
        for model in self.models.values():
            model.save_ensemble_state()


# Global ensemble model manager instance
ensemble_manager = EnsembleModelManager()