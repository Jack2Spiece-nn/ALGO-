"""
Enhanced Risk Management System with Dynamic Correlation-Based Position Sizing

This module extends the existing risk management with advanced features:
- Dynamic correlation-based position sizing
- Real-time portfolio risk monitoring
- Stress testing capabilities
- Advanced VaR calculations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.covariance import LedoitWolf
import concurrent.futures

from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug, log_risk
from src.strategy.garch_strategy import TradingSignal, SignalType
from src.execution.alpaca_executor import OrderResult, PositionInfo, OrderExecutionStatus
from src.execution.risk_manager import RiskLevel, RiskEventType, RiskEvent, RiskMetrics


@dataclass
class CorrelationMetrics:
    """Portfolio correlation metrics"""
    correlation_matrix: pd.DataFrame
    max_correlation: float
    diversification_ratio: float
    effective_positions: float
    concentration_risk: float
    cluster_risk: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'max_correlation': self.max_correlation,
            'diversification_ratio': self.diversification_ratio,
            'effective_positions': self.effective_positions,
            'concentration_risk': self.concentration_risk,
            'cluster_risk': self.cluster_risk
        }


@dataclass
class AdvancedRiskMetrics(RiskMetrics):
    """Extended risk metrics with correlation analysis"""
    correlation_metrics: CorrelationMetrics
    stress_test_results: Dict[str, float]
    liquidity_metrics: Dict[str, float]
    regime_indicators: Dict[str, float]
    tail_risk_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'correlation_metrics': self.correlation_metrics.to_dict(),
            'stress_test_results': self.stress_test_results,
            'liquidity_metrics': self.liquidity_metrics,
            'regime_indicators': self.regime_indicators,
            'tail_risk_metrics': self.tail_risk_metrics
        })
        return base_dict


class EnhancedRiskManager:
    """
    Enhanced risk management system with dynamic correlation-based position sizing
    """
    
    def __init__(self):
        self.risk_config = config.get_risk_config()
        
        # Portfolio state
        self.portfolio_value = 100000.0  # Default starting value
        self.positions = {}
        self.pnl_history = []
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 100000.0
        
        # Risk events tracking
        self.risk_events = []
        self.alerts_triggered = set()
        
        # Correlation analysis
        self.price_history = {}
        self.returns_history = {}
        self.correlation_matrix = pd.DataFrame()
        self.lookback_periods = 60  # Days for correlation calculation
        
        # Stress testing scenarios
        self.stress_scenarios = {
            'market_crash': -0.20,      # 20% market drop
            'volatility_spike': 3.0,    # 3x normal volatility
            'correlation_spike': 0.9,   # High correlation regime
            'liquidity_crunch': -0.10   # 10% liquidity discount
        }
        
        # Risk regime detection
        self.regime_indicators = {
            'vix_level': 0.0,
            'correlation_regime': 0.0,
            'volatility_regime': 0.0,
            'trend_strength': 0.0
        }
        
        log_info("Enhanced risk manager initialized")
    
    def update_price_history(self, symbol: str, price: float, timestamp: datetime):
        """Update price history for correlation analysis"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.returns_history[symbol] = []
        
        self.price_history[symbol].append((timestamp, price))
        
        # Calculate return if we have previous price
        if len(self.price_history[symbol]) > 1:
            prev_price = self.price_history[symbol][-2][1]
            return_val = (price - prev_price) / prev_price
            self.returns_history[symbol].append((timestamp, return_val))
        
        # Keep only recent data
        max_history = self.lookback_periods * 24 * 60  # Assume minute data
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.returns_history[symbol] = self.returns_history[symbol][-max_history:]
    
    def _calculate_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix for given symbols"""
        if not symbols or len(symbols) < 2:
            return pd.DataFrame()
        
        # Prepare returns data
        returns_data = {}
        min_length = float('inf')
        
        for symbol in symbols:
            if symbol in self.returns_history and len(self.returns_history[symbol]) > 30:
                returns = [ret[1] for ret in self.returns_history[symbol][-252:]]  # Last 252 observations
                returns_data[symbol] = returns
                min_length = min(min_length, len(returns))
        
        if len(returns_data) < 2 or min_length < 30:
            return pd.DataFrame()
        
        # Truncate all series to same length
        for symbol in returns_data:
            returns_data[symbol] = returns_data[symbol][-min_length:]
        
        # Create DataFrame and calculate correlation
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def _calculate_diversification_metrics(self, correlation_matrix: pd.DataFrame, 
                                         position_weights: Dict[str, float]) -> CorrelationMetrics:
        """Calculate portfolio diversification metrics"""
        if correlation_matrix.empty or not position_weights:
            return CorrelationMetrics(
                correlation_matrix=correlation_matrix,
                max_correlation=0.0,
                diversification_ratio=1.0,
                effective_positions=1.0,
                concentration_risk=0.0,
                cluster_risk={}
            )
        
        # Calculate maximum correlation
        correlation_values = correlation_matrix.values
        np.fill_diagonal(correlation_values, 0)  # Exclude diagonal
        max_correlation = np.max(correlation_values)
        
        # Calculate effective number of positions (inverse of concentration)
        weights = np.array([position_weights.get(symbol, 0) for symbol in correlation_matrix.columns])
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        effective_positions = 1 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 1
        
        # Calculate diversification ratio
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix.values, weights))
        avg_correlation = np.mean(correlation_values[correlation_values != 0])
        diversification_ratio = 1 / (1 + avg_correlation * (len(weights) - 1)) if len(weights) > 1 else 1
        
        # Calculate concentration risk (Herfindahl index)
        concentration_risk = np.sum(weights ** 2)
        
        # Identify correlation clusters
        cluster_risk = {}
        threshold = 0.7
        
        for i, symbol1 in enumerate(correlation_matrix.columns):
            high_corr_symbols = []
            for j, symbol2 in enumerate(correlation_matrix.columns):
                if i != j and abs(correlation_matrix.iloc[i, j]) > threshold:
                    high_corr_symbols.append(symbol2)
            
            if high_corr_symbols:
                cluster_weight = sum(position_weights.get(s, 0) for s in high_corr_symbols + [symbol1])
                cluster_risk[symbol1] = cluster_weight
        
        return CorrelationMetrics(
            correlation_matrix=correlation_matrix,
            max_correlation=max_correlation,
            diversification_ratio=diversification_ratio,
            effective_positions=effective_positions,
            concentration_risk=concentration_risk,
            cluster_risk=cluster_risk
        )
    
    def _perform_stress_tests(self, positions: Dict[str, PositionInfo], 
                            correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Perform stress testing scenarios"""
        stress_results = {}
        
        if not positions or correlation_matrix.empty:
            return stress_results
        
        # Current portfolio value
        current_value = sum(pos.market_value for pos in positions.values())
        
        for scenario_name, scenario_shock in self.stress_scenarios.items():
            scenario_pnl = 0.0
            
            if scenario_name == 'market_crash':
                # Apply uniform shock to all positions
                scenario_pnl = current_value * scenario_shock
            
            elif scenario_name == 'volatility_spike':
                # Increase position risk based on volatility
                for symbol, position in positions.items():
                    vol_impact = abs(position.unrealized_pnl_pct) * scenario_shock
                    scenario_pnl += position.market_value * vol_impact * np.sign(position.unrealized_pnl_pct)
            
            elif scenario_name == 'correlation_spike':
                # Simulate high correlation scenario
                if len(positions) > 1:
                    avg_return = np.mean([pos.unrealized_pnl_pct for pos in positions.values()])
                    for position in positions.values():
                        # In high correlation, all positions move together
                        correlated_return = scenario_shock * avg_return + (1 - scenario_shock) * position.unrealized_pnl_pct
                        scenario_pnl += position.cost_basis * (correlated_return - position.unrealized_pnl_pct)
            
            elif scenario_name == 'liquidity_crunch':
                # Apply liquidity discount to large positions
                for position in positions.values():
                    position_weight = position.market_value / current_value
                    liquidity_impact = max(0, position_weight - 0.1) * abs(scenario_shock)  # Penalty for >10% positions
                    scenario_pnl += position.market_value * liquidity_impact
            
            stress_results[scenario_name] = scenario_pnl / current_value if current_value > 0 else 0
        
        return stress_results
    
    def _calculate_liquidity_metrics(self, positions: Dict[str, PositionInfo]) -> Dict[str, float]:
        """Calculate portfolio liquidity metrics"""
        if not positions:
            return {}
        
        total_value = sum(pos.market_value for pos in positions.values())
        
        # Liquidity score based on position sizes and typical volumes
        # This is simplified - in practice, you'd use actual volume data
        liquidity_score = 0.0
        for position in positions.values():
            position_weight = abs(position.market_value) / total_value
            # Assume larger positions are less liquid
            position_liquidity = max(0, 1 - position_weight * 5)  # Penalty for large positions
            liquidity_score += position_weight * position_liquidity
        
        # Time to liquidate (simplified estimate)
        time_to_liquidate = 0.0
        for position in positions.values():
            position_weight = abs(position.market_value) / total_value
            # Estimate based on position size
            estimated_days = position_weight * 10  # Larger positions take longer
            time_to_liquidate = max(time_to_liquidate, estimated_days)
        
        return {
            'liquidity_score': liquidity_score,
            'time_to_liquidate_days': time_to_liquidate,
            'large_position_count': sum(1 for pos in positions.values() 
                                      if abs(pos.market_value) / total_value > 0.2)
        }
    
    def _calculate_tail_risk_metrics(self, positions: Dict[str, PositionInfo]) -> Dict[str, float]:
        """Calculate tail risk metrics"""
        if not positions:
            return {}
        
        # Collect return data
        all_returns = []
        for symbol in positions.keys():
            if symbol in self.returns_history:
                returns = [ret[1] for ret in self.returns_history[symbol][-252:]]
                all_returns.extend(returns)
        
        if not all_returns:
            return {}
        
        returns_array = np.array(all_returns)
        
        # Calculate tail risk metrics
        skewness = stats.skew(returns_array)
        kurtosis = stats.kurtosis(returns_array)
        
        # Expected Shortfall (Conditional VaR)
        var_95 = np.percentile(returns_array, 5)
        var_99 = np.percentile(returns_array, 1)
        
        tail_returns_95 = returns_array[returns_array <= var_95]
        tail_returns_99 = returns_array[returns_array <= var_99]
        
        expected_shortfall_95 = np.mean(tail_returns_95) if len(tail_returns_95) > 0 else var_95
        expected_shortfall_99 = np.mean(tail_returns_99) if len(tail_returns_99) > 0 else var_99
        
        # Maximum drawdown simulation
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_simulated_drawdown = np.min(drawdowns)
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': expected_shortfall_95,
            'expected_shortfall_99': expected_shortfall_99,
            'max_simulated_drawdown': max_simulated_drawdown
        }
    
    def _update_regime_indicators(self, positions: Dict[str, PositionInfo]):
        """Update market regime indicators"""
        # This is simplified - in practice, you'd use more sophisticated regime detection
        
        # Volatility regime
        if positions:
            recent_vols = []
            for symbol in positions.keys():
                if symbol in self.returns_history and len(self.returns_history[symbol]) > 20:
                    recent_returns = [ret[1] for ret in self.returns_history[symbol][-20:]]
                    vol = np.std(recent_returns) * np.sqrt(252)  # Annualized
                    recent_vols.append(vol)
            
            if recent_vols:
                avg_vol = np.mean(recent_vols)
                # Normalize to 0-1 scale (0.1 = low vol, 0.4+ = high vol)
                self.regime_indicators['volatility_regime'] = min(1.0, avg_vol / 0.4)
        
        # Correlation regime
        if not self.correlation_matrix.empty:
            avg_correlation = np.mean(self.correlation_matrix.values[self.correlation_matrix.values != 1])
            self.regime_indicators['correlation_regime'] = (avg_correlation + 1) / 2  # Scale to 0-1
        
        # Trend strength (simplified)
        trend_strengths = []
        for symbol in positions.keys():
            if symbol in self.returns_history and len(self.returns_history[symbol]) > 10:
                recent_returns = [ret[1] for ret in self.returns_history[symbol][-10:]]
                trend_strength = abs(np.mean(recent_returns)) / (np.std(recent_returns) + 1e-6)
                trend_strengths.append(trend_strength)
        
        if trend_strengths:
            self.regime_indicators['trend_strength'] = min(1.0, np.mean(trend_strengths))
    
    def calculate_dynamic_position_size(self, signal: TradingSignal, 
                                      current_positions: Dict[str, PositionInfo],
                                      portfolio_value: float) -> float:
        """
        Calculate dynamic position size based on correlation and risk metrics
        
        Args:
            signal: Trading signal
            current_positions: Current portfolio positions
            portfolio_value: Total portfolio value
            
        Returns:
            Optimal position size
        """
        try:
            base_size = portfolio_value * self.risk_config.max_position_size
            
            # Get correlation matrix
            symbols = list(current_positions.keys()) + [signal.symbol]
            correlation_matrix = self._calculate_correlation_matrix(symbols)
            
            # Position weights
            position_weights = {}
            for symbol, position in current_positions.items():
                position_weights[symbol] = abs(position.market_value) / portfolio_value
            
            # Calculate diversification metrics
            correlation_metrics = self._calculate_diversification_metrics(correlation_matrix, position_weights)
            
            # Dynamic sizing factors
            size_adjustment = 1.0
            
            # 1. Correlation adjustment
            if signal.symbol in correlation_matrix.columns:
                max_corr_with_existing = 0.0
                for existing_symbol in current_positions.keys():
                    if existing_symbol in correlation_matrix.columns:
                        corr = abs(correlation_matrix.loc[signal.symbol, existing_symbol])
                        max_corr_with_existing = max(max_corr_with_existing, corr)
                
                # Reduce size for highly correlated positions
                correlation_adjustment = 1 - (max_corr_with_existing * 0.5)
                size_adjustment *= correlation_adjustment
                
                log_debug(f"Correlation adjustment for {signal.symbol}: {correlation_adjustment:.3f}")
            
            # 2. Concentration adjustment
            if correlation_metrics.concentration_risk > 0.5:  # High concentration
                concentration_adjustment = 1 - (correlation_metrics.concentration_risk - 0.5)
                size_adjustment *= max(0.5, concentration_adjustment)
                
                log_debug(f"Concentration adjustment for {signal.symbol}: {concentration_adjustment:.3f}")
            
            # 3. Volatility adjustment
            volatility_adjustment = 1.0
            if signal.predicted_volatility > 0.03:  # High volatility
                volatility_adjustment = 0.03 / signal.predicted_volatility
                size_adjustment *= volatility_adjustment
                
                log_debug(f"Volatility adjustment for {signal.symbol}: {volatility_adjustment:.3f}")
            
            # 4. Signal strength adjustment
            signal_adjustment = signal.strength * signal.confidence
            size_adjustment *= signal_adjustment
            
            # 5. Regime adjustment
            self._update_regime_indicators(current_positions)
            
            if self.regime_indicators.get('volatility_regime', 0) > 0.7:  # High vol regime
                regime_adjustment = 0.7
                size_adjustment *= regime_adjustment
                log_debug(f"High volatility regime adjustment: {regime_adjustment:.3f}")
            
            if self.regime_indicators.get('correlation_regime', 0) > 0.8:  # High correlation regime
                regime_adjustment = 0.8
                size_adjustment *= regime_adjustment
                log_debug(f"High correlation regime adjustment: {regime_adjustment:.3f}")
            
            # Apply adjustment
            adjusted_size = base_size * size_adjustment
            
            # Ensure minimum and maximum limits
            min_size = portfolio_value * 0.01  # 1% minimum
            max_size = portfolio_value * self.risk_config.max_position_size
            
            final_size = max(min_size, min(max_size, adjusted_size))
            
            log_info(f"Dynamic position size for {signal.symbol}: ${final_size:.2f} "
                    f"(adjustment: {size_adjustment:.3f})")
            
            return final_size / signal.price  # Convert to shares
            
        except Exception as e:
            log_error(f"Error calculating dynamic position size: {e}")
            # Fallback to basic sizing
            return (portfolio_value * self.risk_config.max_position_size * 0.5) / signal.price
    
    def calculate_enhanced_risk_metrics(self, portfolio_value: float, 
                                      positions: Dict[str, PositionInfo]) -> AdvancedRiskMetrics:
        """
        Calculate comprehensive risk metrics including correlation analysis
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            
        Returns:
            AdvancedRiskMetrics object
        """
        try:
            # Update price history for all positions
            for symbol, position in positions.items():
                self.update_price_history(symbol, position.current_price, datetime.now())
            
            # Calculate base metrics (from original risk manager)
            total_exposure = sum(abs(pos.market_value) for pos in positions.values())
            max_position = max([abs(pos.market_value) for pos in positions.values()]) if positions else 0
            concentration_ratio = max_position / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate P&L metrics
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
            self.daily_pnl = total_unrealized_pnl / portfolio_value if portfolio_value > 0 else 0
            
            # Update drawdown
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Position weights for correlation analysis
            position_weights = {}
            for symbol, position in positions.items():
                position_weights[symbol] = abs(position.market_value) / portfolio_value
            
            # Calculate correlation metrics
            symbols = list(positions.keys())
            correlation_matrix = self._calculate_correlation_matrix(symbols)
            correlation_metrics = self._calculate_diversification_metrics(correlation_matrix, position_weights)
            
            # Perform stress tests
            stress_test_results = self._perform_stress_tests(positions, correlation_matrix)
            
            # Calculate liquidity metrics
            liquidity_metrics = self._calculate_liquidity_metrics(positions)
            
            # Calculate tail risk metrics
            tail_risk_metrics = self._calculate_tail_risk_metrics(positions)
            
            # Update regime indicators
            self._update_regime_indicators(positions)
            
            # VaR calculation (simplified)
            portfolio_returns = []
            if len(self.pnl_history) > 30:
                portfolio_returns = self.pnl_history[-252:]  # Last year
            
            if portfolio_returns:
                var_95 = np.percentile(portfolio_returns, 5) * portfolio_value
                var_99 = np.percentile(portfolio_returns, 1) * portfolio_value
                portfolio_vol = np.std(portfolio_returns)
                sharpe_ratio = np.mean(portfolio_returns) / portfolio_vol if portfolio_vol > 0 else 0
            else:
                var_95 = -portfolio_value * 0.05  # Default 5%
                var_99 = -portfolio_value * 0.10  # Default 10%
                portfolio_vol = 0.15  # Default 15%
                sharpe_ratio = 0.0
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                concentration_ratio, self.daily_pnl, current_drawdown,
                correlation_metrics.max_correlation, stress_test_results
            )
            
            # Create enhanced risk metrics
            enhanced_metrics = AdvancedRiskMetrics(
                total_exposure=total_exposure,
                max_position_size=max_position,
                concentration_ratio=concentration_ratio,
                daily_pnl=self.daily_pnl,
                max_drawdown=self.max_drawdown,
                var_95=var_95,
                var_99=var_99,
                portfolio_volatility=portfolio_vol,
                sharpe_ratio=sharpe_ratio,
                risk_level=risk_level,
                correlation_metrics=correlation_metrics,
                stress_test_results=stress_test_results,
                liquidity_metrics=liquidity_metrics,
                regime_indicators=self.regime_indicators.copy(),
                tail_risk_metrics=tail_risk_metrics
            )
            
            # Log risk summary
            log_risk(
                portfolio_value, total_exposure, self.daily_pnl,
                self.max_drawdown, risk_level.value,
                f"Correlation: {correlation_metrics.max_correlation:.3f}, "
                f"Diversification: {correlation_metrics.diversification_ratio:.3f}"
            )
            
            return enhanced_metrics
            
        except Exception as e:
            log_error(f"Error calculating enhanced risk metrics: {e}")
            # Return basic metrics as fallback
            return AdvancedRiskMetrics(
                total_exposure=0, max_position_size=0, concentration_ratio=0,
                daily_pnl=0, max_drawdown=0, var_95=0, var_99=0,
                portfolio_volatility=0, sharpe_ratio=0, risk_level=RiskLevel.LOW,
                correlation_metrics=CorrelationMetrics(
                    pd.DataFrame(), 0, 1, 1, 0, {}
                ),
                stress_test_results={}, liquidity_metrics={},
                regime_indicators={}, tail_risk_metrics={}
            )
    
    def _determine_risk_level(self, concentration_ratio: float, daily_pnl: float,
                            drawdown: float, max_correlation: float,
                            stress_results: Dict[str, float]) -> RiskLevel:
        """Determine overall portfolio risk level"""
        risk_score = 0
        
        # Concentration risk
        if concentration_ratio > 0.3:
            risk_score += 2
        elif concentration_ratio > 0.2:
            risk_score += 1
        
        # P&L risk
        if daily_pnl < -0.05:  # -5%
            risk_score += 3
        elif daily_pnl < -0.02:  # -2%
            risk_score += 1
        
        # Drawdown risk
        if drawdown > 0.15:  # 15%
            risk_score += 3
        elif drawdown > 0.10:  # 10%
            risk_score += 2
        elif drawdown > 0.05:  # 5%
            risk_score += 1
        
        # Correlation risk
        if max_correlation > 0.8:
            risk_score += 2
        elif max_correlation > 0.6:
            risk_score += 1
        
        # Stress test results
        if stress_results:
            worst_stress = min(stress_results.values())
            if worst_stress < -0.20:  # -20%
                risk_score += 3
            elif worst_stress < -0.10:  # -10%
                risk_score += 1
        
        # Map score to risk level
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def should_allow_trade(self, signal: TradingSignal, 
                          current_positions: Dict[str, PositionInfo],
                          portfolio_value: float) -> Tuple[bool, str]:
        """
        Enhanced trade approval with correlation analysis
        
        Args:
            signal: Trading signal
            current_positions: Current positions
            portfolio_value: Portfolio value
            
        Returns:
            Tuple of (allowed, reason)
        """
        try:
            # Calculate enhanced risk metrics
            risk_metrics = self.calculate_enhanced_risk_metrics(portfolio_value, current_positions)
            
            # Basic risk checks (from original implementation)
            if risk_metrics.risk_level == RiskLevel.CRITICAL:
                return False, "Portfolio risk level is CRITICAL"
            
            if risk_metrics.daily_pnl < -self.risk_config.max_daily_loss:
                return False, f"Daily loss limit exceeded: {risk_metrics.daily_pnl:.2%}"
            
            if risk_metrics.max_drawdown > self.risk_config.max_drawdown:
                return False, f"Maximum drawdown exceeded: {risk_metrics.max_drawdown:.2%}"
            
            # Enhanced correlation checks
            if not risk_metrics.correlation_metrics.correlation_matrix.empty:
                if signal.symbol in risk_metrics.correlation_metrics.correlation_matrix.columns:
                    # Check correlation with existing positions
                    for existing_symbol in current_positions.keys():
                        if existing_symbol in risk_metrics.correlation_metrics.correlation_matrix.columns:
                            correlation = abs(risk_metrics.correlation_metrics.correlation_matrix.loc[
                                signal.symbol, existing_symbol
                            ])
                            
                            if correlation > 0.8:  # High correlation threshold
                                existing_weight = abs(current_positions[existing_symbol].market_value) / portfolio_value
                                if existing_weight > 0.15:  # Significant existing position
                                    return False, f"High correlation ({correlation:.2f}) with existing large position in {existing_symbol}"
            
            # Concentration limits
            if risk_metrics.correlation_metrics.concentration_risk > 0.7:
                return False, f"Portfolio concentration too high: {risk_metrics.correlation_metrics.concentration_risk:.2f}"
            
            # Stress test limits
            if risk_metrics.stress_test_results:
                worst_scenario = min(risk_metrics.stress_test_results.values())
                if worst_scenario < -0.25:  # -25% stress loss
                    return False, f"Stress test failure: worst scenario {worst_scenario:.2%}"
            
            # Regime-based restrictions
            if self.regime_indicators.get('volatility_regime', 0) > 0.8:
                if signal.strength < 0.7:  # Require higher conviction in high vol regime
                    return False, "High volatility regime requires higher signal conviction"
            
            # Position count limit
            if len(current_positions) >= self.risk_config.max_positions:
                return False, f"Maximum positions limit reached: {len(current_positions)}"
            
            # Signal quality checks
            if signal.strength < self.risk_config.get('min_signal_strength', 0.3):
                return False, f"Signal strength too low: {signal.strength:.3f}"
            
            if signal.confidence < 0.5:
                return False, f"Signal confidence too low: {signal.confidence:.3f}"
            
            return True, "Trade approved"
            
        except Exception as e:
            log_error(f"Error in enhanced trade approval: {e}")
            return False, f"Risk calculation error: {str(e)}"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        return {
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'peak_value': self.peak_value,
            'risk_events_count': len(self.risk_events),
            'regime_indicators': self.regime_indicators.copy(),
            'correlation_matrix_shape': self.correlation_matrix.shape if not self.correlation_matrix.empty else (0, 0),
            'price_history_symbols': list(self.price_history.keys()),
            'last_risk_calculation': datetime.now()
        }


# Global enhanced risk manager instance
enhanced_risk_manager = EnhancedRiskManager()