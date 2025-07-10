"""
GARCH Model Implementation for Volatility Forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import warnings
from arch import arch_model
from arch.univariate import GARCH, EGARCH, ConstantMean, ZeroMean, ARX
from arch.univariate.distribution import Normal, StudentsT, SkewStudent
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import joblib

from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug, log_garch


@dataclass
class GarchPrediction:
    """GARCH model prediction result"""
    symbol: str
    timestamp: datetime
    predicted_volatility: float
    variance_forecast: float
    confidence_interval: Tuple[float, float]
    model_params: Dict[str, Any]
    log_likelihood: float
    aic: float
    bic: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GarchModelMetrics:
    """GARCH model performance metrics"""
    symbol: str
    period: str
    log_likelihood: float
    aic: float
    bic: float
    arch_lm_statistic: float
    arch_lm_pvalue: float
    ljung_box_statistic: float
    ljung_box_pvalue: float
    jarque_bera_statistic: float
    jarque_bera_pvalue: float
    model_params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GarchModel:
    """
    Advanced GARCH model implementation for volatility forecasting
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.garch_config = config.get_garch_config()
        
        # Model configuration
        self.model_type = self.garch_config.model_type
        self.p = self.garch_config.p
        self.q = self.garch_config.q
        self.mean_model = self.garch_config.mean_model
        self.distribution = self.garch_config.distribution
        self.rescale = self.garch_config.rescale
        
        # Model objects
        self.model = None
        self.fitted_model = None
        
        # Data storage
        self.returns_data = pd.Series()
        self.training_data = pd.Series()
        self.validation_data = pd.Series()
        
        # Performance tracking
        self.last_fit_time = None
        self.last_prediction_time = None
        self.model_metrics = None
        
        # Prediction history
        self.prediction_history = []
        
        log_info(f"GARCH model initialized for {symbol}")
    
    def _prepare_data(self, returns: pd.Series) -> pd.Series:
        """Prepare returns data for GARCH modeling"""
        
        # Remove missing values
        returns = returns.dropna()
        
        # Remove extreme outliers (beyond 5 standard deviations)
        mean_return = returns.mean()
        std_return = returns.std()
        returns = returns[
            (returns >= mean_return - 5 * std_return) & 
            (returns <= mean_return + 5 * std_return)
        ]
        
        # Scale returns to percentage
        returns = returns * 100
        
        log_debug(f"Prepared {len(returns)} return observations for {self.symbol}")
        return returns
    
    def _get_mean_model(self):
        """Get mean model based on configuration"""
        if self.mean_model == "Zero":
            return ZeroMean
        elif self.mean_model == "Constant":
            return ConstantMean
        elif self.mean_model == "AR":
            return ARX
        else:
            return ZeroMean
    
    def _get_volatility_model(self):
        """Get volatility model based on configuration"""
        if self.model_type == "GARCH":
            return GARCH
        elif self.model_type == "GJR-GARCH":
            # GJR-GARCH not available in this version, use EGARCH as alternative
            return EGARCH
        elif self.model_type == "EGARCH":
            return EGARCH
        else:
            return GARCH
    
    def _get_distribution(self):
        """Get distribution based on configuration"""
        if self.distribution == "normal":
            return Normal
        elif self.distribution == "t":
            return StudentsT
        elif self.distribution == "skewt":
            return SkewStudent
        else:
            return Normal
    
    def fit(self, returns: pd.Series, 
            train_ratio: float = 0.8) -> bool:
        """
        Fit GARCH model to returns data
        
        Args:
            returns: Time series of returns
            train_ratio: Proportion of data to use for training
            
        Returns:
            bool: True if fitting successful
        """
        
        try:
            # Prepare data
            self.returns_data = self._prepare_data(returns)
            
            if len(self.returns_data) < 100:
                log_error(f"Insufficient data for {self.symbol}: {len(self.returns_data)} observations")
                return False
            
            # Split data
            split_idx = int(len(self.returns_data) * train_ratio)
            self.training_data = self.returns_data[:split_idx]
            self.validation_data = self.returns_data[split_idx:]
            
            # Create model
            self.model = arch_model(
                self.training_data,
                mean=self.mean_model,
                vol=self.model_type,
                p=self.p,
                q=self.q,
                dist=self.distribution,
                rescale=self.rescale
            )
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted_model = self.model.fit(disp='off', show_warning=False)
            
            self.last_fit_time = datetime.now()
            
            # Calculate model metrics
            self._calculate_model_metrics()
            
            # Log model parameters
            log_garch(
                self.symbol,
                0.0,  # Will be updated with actual prediction
                (0.0, 0.0),  # Will be updated with actual confidence interval
                self.fitted_model.params.to_dict()
            )
            
            log_info(f"GARCH model fitted for {self.symbol} with {len(self.training_data)} observations")
            return True
            
        except Exception as e:
            log_error(f"Error fitting GARCH model for {self.symbol}: {e}")
            return False
    
    def _calculate_model_metrics(self):
        """Calculate model performance metrics"""
        if self.fitted_model is None:
            return
        
        try:
            # Basic model metrics
            log_likelihood = self.fitted_model.loglikelihood
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic
            
            # Residual diagnostics
            std_resid = self.fitted_model.std_resid
            
            # ARCH LM test
            arch_lm = self.fitted_model.arch_lm_test(lags=5)
            arch_lm_stat = arch_lm.stat
            arch_lm_pval = arch_lm.pvalue
            
            # Ljung-Box test
            ljung_box = self.fitted_model.ljung_box(lags=10)
            ljung_box_stat = ljung_box.stat
            ljung_box_pval = ljung_box.pvalue
            
            # Jarque-Bera test
            jarque_bera = self.fitted_model.jarque_bera()
            jb_stat = jarque_bera.stat
            jb_pval = jarque_bera.pvalue
            
            self.model_metrics = GarchModelMetrics(
                symbol=self.symbol,
                period=f"{self.last_fit_time.strftime('%Y-%m-%d')}",
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                arch_lm_statistic=arch_lm_stat,
                arch_lm_pvalue=arch_lm_pval,
                ljung_box_statistic=ljung_box_stat,
                ljung_box_pvalue=ljung_box_pval,
                jarque_bera_statistic=jb_stat,
                jarque_bera_pvalue=jb_pval,
                model_params=self.fitted_model.params.to_dict()
            )
            
            log_debug(f"Model metrics calculated for {self.symbol}")
            
        except Exception as e:
            log_error(f"Error calculating model metrics: {e}")
    
    def predict(self, horizon: int = 1) -> Optional[GarchPrediction]:
        """
        Generate volatility forecast
        
        Args:
            horizon: Forecast horizon (number of periods)
            
        Returns:
            GarchPrediction object or None if prediction fails
        """
        
        if self.fitted_model is None:
            log_error(f"Model not fitted for {self.symbol}")
            return None
        
        try:
            # Generate forecast
            forecast = self.fitted_model.forecast(horizon=horizon)
            
            # Extract variance forecast
            variance_forecast = forecast.variance.iloc[-1, 0]
            
            # Calculate volatility (standard deviation)
            volatility_forecast = np.sqrt(variance_forecast)
            
            # Calculate confidence interval (assuming normal distribution)
            z_score = 1.96  # 95% confidence
            volatility_std = volatility_forecast * 0.1  # Approximate standard error
            
            conf_lower = volatility_forecast - z_score * volatility_std
            conf_upper = volatility_forecast + z_score * volatility_std
            
            # Create prediction object
            prediction = GarchPrediction(
                symbol=self.symbol,
                timestamp=datetime.now(),
                predicted_volatility=volatility_forecast,
                variance_forecast=variance_forecast,
                confidence_interval=(conf_lower, conf_upper),
                model_params=self.fitted_model.params.to_dict(),
                log_likelihood=self.fitted_model.loglikelihood,
                aic=self.fitted_model.aic,
                bic=self.fitted_model.bic
            )
            
            # Store prediction
            self.prediction_history.append(prediction)
            self.last_prediction_time = datetime.now()
            
            # Log prediction
            log_garch(
                self.symbol,
                volatility_forecast,
                (conf_lower, conf_upper),
                self.fitted_model.params.to_dict()
            )
            
            return prediction
            
        except Exception as e:
            log_error(f"Error generating prediction for {self.symbol}: {e}")
            return None
    
    def rolling_forecast(self, returns: pd.Series, 
                        window_size: int = 250,
                        refit_frequency: int = 20) -> List[GarchPrediction]:
        """
        Generate rolling forecasts
        
        Args:
            returns: Full returns series
            window_size: Rolling window size
            refit_frequency: Refit model every N periods
            
        Returns:
            List of GarchPrediction objects
        """
        
        returns = self._prepare_data(returns)
        predictions = []
        
        if len(returns) < window_size + 50:
            log_error(f"Insufficient data for rolling forecast: {len(returns)} observations")
            return predictions
        
        try:
            for i in range(window_size, len(returns)):
                # Get rolling window
                window_returns = returns.iloc[i-window_size:i]
                
                # Refit model periodically
                if i % refit_frequency == 0 or self.fitted_model is None:
                    success = self.fit(window_returns, train_ratio=1.0)
                    if not success:
                        continue
                
                # Generate prediction
                prediction = self.predict(horizon=1)
                if prediction is not None:
                    # Update timestamp to match the actual prediction time
                    prediction.timestamp = returns.index[i]
                    predictions.append(prediction)
            
            log_info(f"Generated {len(predictions)} rolling forecasts for {self.symbol}")
            return predictions
            
        except Exception as e:
            log_error(f"Error in rolling forecast: {e}")
            return predictions
    
    def calculate_prediction_premium(self, current_price: float, 
                                   predicted_volatility: float,
                                   time_horizon: float = 1.0) -> float:
        """
        Calculate prediction premium based on volatility forecast
        
        Args:
            current_price: Current asset price
            predicted_volatility: Forecasted volatility
            time_horizon: Time horizon in days
            
        Returns:
            Prediction premium as percentage
        """
        
        # Annualize volatility if needed
        if time_horizon < 1.0:
            annualized_vol = predicted_volatility * np.sqrt(252)
        else:
            annualized_vol = predicted_volatility
        
        # Calculate expected price movement
        expected_move = current_price * annualized_vol * np.sqrt(time_horizon / 252)
        
        # Premium as percentage of current price
        premium = expected_move / current_price
        
        return premium
    
    def validate_model(self) -> Dict[str, Any]:
        """
        Validate model performance on validation data
        
        Returns:
            Dictionary with validation metrics
        """
        
        if self.fitted_model is None or len(self.validation_data) == 0:
            return {}
        
        try:
            # Generate forecasts for validation period
            val_forecasts = []
            actual_volatility = []
            
            # Calculate rolling validation forecasts
            for i in range(len(self.validation_data)):
                # Use all training data + validation data up to point i
                all_data = pd.concat([self.training_data, self.validation_data.iloc[:i+1]])
                
                # Fit model on available data
                temp_model = arch_model(
                    all_data[:-1],  # Exclude current observation
                    mean=self.mean_model,
                    vol=self.model_type,
                    p=self.p,
                    q=self.q,
                    dist=self.distribution,
                    rescale=self.rescale
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    temp_fit = temp_model.fit(disp='off', show_warning=False)
                
                # Generate 1-step forecast
                forecast = temp_fit.forecast(horizon=1)
                predicted_vol = np.sqrt(forecast.variance.iloc[-1, 0])
                val_forecasts.append(predicted_vol)
                
                # Calculate actual volatility (realized volatility)
                actual_vol = abs(all_data.iloc[-1])  # Use absolute return as proxy
                actual_volatility.append(actual_vol)
            
            # Calculate validation metrics
            val_forecasts = np.array(val_forecasts)
            actual_volatility = np.array(actual_volatility)
            
            # Mean Absolute Error
            mae = np.mean(np.abs(val_forecasts - actual_volatility))
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((val_forecasts - actual_volatility) ** 2))
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((val_forecasts - actual_volatility) / actual_volatility)) * 100
            
            # Correlation
            correlation = np.corrcoef(val_forecasts, actual_volatility)[0, 1]
            
            validation_metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'correlation': correlation,
                'validation_samples': len(val_forecasts)
            }
            
            log_info(f"Model validation completed for {self.symbol}: MAE={mae:.4f}, RMSE={rmse:.4f}")
            return validation_metrics
            
        except Exception as e:
            log_error(f"Error in model validation: {e}")
            return {}
    
    def plot_forecast(self, save_path: Optional[str] = None):
        """Plot volatility forecast"""
        
        if not self.prediction_history:
            log_error("No predictions available for plotting")
            return
        
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Historical volatility and predictions
            pred_times = [p.timestamp for p in self.prediction_history]
            pred_vols = [p.predicted_volatility for p in self.prediction_history]
            
            ax1.plot(pred_times, pred_vols, 'b-', label='Predicted Volatility', linewidth=2)
            ax1.fill_between(
                pred_times,
                [p.confidence_interval[0] for p in self.prediction_history],
                [p.confidence_interval[1] for p in self.prediction_history],
                alpha=0.3, color='blue', label='95% Confidence Interval'
            )
            
            ax1.set_title(f'GARCH Volatility Forecast - {self.symbol}')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Volatility')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Model residuals
            if self.fitted_model is not None:
                residuals = self.fitted_model.resid
                ax2.plot(residuals.index, residuals, 'r-', alpha=0.7)
                ax2.set_title('Model Residuals')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Residuals')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                log_info(f"Forecast plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            log_error(f"Error plotting forecast: {e}")
    
    def save_model(self, filepath: str):
        """Save fitted model to file"""
        
        if self.fitted_model is None:
            log_error("No fitted model to save")
            return
        
        try:
            model_data = {
                'fitted_model': self.fitted_model,
                'symbol': self.symbol,
                'config': self.garch_config,
                'last_fit_time': self.last_fit_time,
                'model_metrics': self.model_metrics,
                'prediction_history': self.prediction_history[-100:]  # Save last 100 predictions
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            log_info(f"Model saved to {filepath}")
            
        except Exception as e:
            log_error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str) -> bool:
        """Load fitted model from file"""
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.fitted_model = model_data['fitted_model']
            self.symbol = model_data['symbol']
            self.last_fit_time = model_data['last_fit_time']
            self.model_metrics = model_data['model_metrics']
            self.prediction_history = model_data.get('prediction_history', [])
            
            log_info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            log_error(f"Error loading model: {e}")
            return False
    
    def get_model_summary(self) -> str:
        """Get model summary"""
        
        if self.fitted_model is None:
            return "Model not fitted"
        
        return str(self.fitted_model.summary())
    
    def needs_refit(self) -> bool:
        """Check if model needs refitting"""
        
        if self.fitted_model is None:
            return True
        
        if self.last_fit_time is None:
            return True
        
        # Check if model is older than refit frequency
        days_since_fit = (datetime.now() - self.last_fit_time).days
        refit_frequency = self.garch_config.refit_frequency
        
        return days_since_fit >= refit_frequency


class GarchModelManager:
    """
    Manager for multiple GARCH models
    """
    
    def __init__(self):
        self.models = {}
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        log_info("GARCH model manager initialized")
    
    def get_model(self, symbol: str) -> GarchModel:
        """Get or create GARCH model for symbol"""
        
        if symbol not in self.models:
            self.models[symbol] = GarchModel(symbol)
            
            # Try to load existing model
            model_path = self.model_dir / f"{symbol}_garch.pkl"
            if model_path.exists():
                self.models[symbol].load_model(str(model_path))
        
        return self.models[symbol]
    
    def fit_all_models(self, returns_data: Dict[str, pd.Series]) -> Dict[str, bool]:
        """Fit all models with provided data"""
        
        results = {}
        
        for symbol, returns in returns_data.items():
            model = self.get_model(symbol)
            success = model.fit(returns)
            results[symbol] = success
            
            if success:
                # Save model
                model_path = self.model_dir / f"{symbol}_garch.pkl"
                model.save_model(str(model_path))
        
        return results
    
    def predict_all(self, symbols: List[str]) -> Dict[str, Optional[GarchPrediction]]:
        """Generate predictions for all symbols"""
        
        predictions = {}
        
        for symbol in symbols:
            model = self.get_model(symbol)
            prediction = model.predict()
            predictions[symbol] = prediction
        
        return predictions
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        
        performance = {}
        
        for symbol, model in self.models.items():
            if model.model_metrics:
                performance[symbol] = model.model_metrics.to_dict()
        
        return performance


# Global GARCH model manager instance
garch_manager = GarchModelManager()