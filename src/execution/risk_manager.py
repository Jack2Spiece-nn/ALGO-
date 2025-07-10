"""
Advanced Risk Management System for GARCH Intraday Trading
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug, log_risk
from src.strategy.garch_strategy import TradingSignal, SignalType
from src.execution.alpaca_executor import OrderResult, PositionInfo, OrderExecutionStatus


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskEventType(Enum):
    """Risk event types"""
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    POSITION_LIMIT = "POSITION_LIMIT"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    STOP_LOSS_TRIGGERED = "STOP_LOSS_TRIGGERED"
    TAKE_PROFIT_TRIGGERED = "TAKE_PROFIT_TRIGGERED"
    POSITION_TIMEOUT = "POSITION_TIMEOUT"
    CORRELATION_LIMIT = "CORRELATION_LIMIT"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"


@dataclass
class RiskEvent:
    """Risk event notification"""
    event_type: RiskEventType
    symbol: str
    risk_level: RiskLevel
    current_value: float
    threshold: float
    timestamp: datetime
    message: str
    action_required: bool
    recommended_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'symbol': self.symbol,
            'risk_level': self.risk_level.value,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'message': self.message,
            'action_required': self.action_required,
            'recommended_action': self.recommended_action
        }


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_exposure: float
    max_position_size: float
    concentration_ratio: float
    daily_pnl: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    portfolio_volatility: float
    sharpe_ratio: float
    risk_level: RiskLevel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_exposure': self.total_exposure,
            'max_position_size': self.max_position_size,
            'concentration_ratio': self.concentration_ratio,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'portfolio_volatility': self.portfolio_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'risk_level': self.risk_level.value
        }


class PositionSizer:
    """Dynamic position sizing based on risk metrics"""
    
    def __init__(self):
        self.strategy_config = config.get_strategy_config()
        self.risk_config = config.get_risk_config()
        
    def calculate_position_size(self, signal: TradingSignal, 
                              account_value: float,
                              current_positions: Dict[str, PositionInfo],
                              risk_metrics: 'RiskMetrics') -> float:
        """
        Calculate optimal position size based on multiple risk factors
        
        Args:
            signal: Trading signal
            account_value: Current account value
            current_positions: Current positions
            risk_metrics: Portfolio risk metrics
            
        Returns:
            Position size in shares
        """
        
        # Base position size
        base_size = self._calculate_base_size(signal, account_value)
        
        # Risk adjustments
        volatility_adjustment = self._calculate_volatility_adjustment(signal)
        correlation_adjustment = self._calculate_correlation_adjustment(signal, current_positions)
        concentration_adjustment = self._calculate_concentration_adjustment(signal, current_positions, account_value)
        drawdown_adjustment = self._calculate_drawdown_adjustment(risk_metrics)
        
        # Apply all adjustments
        final_size = base_size * volatility_adjustment * correlation_adjustment * concentration_adjustment * drawdown_adjustment
        
        # Ensure minimum and maximum limits
        final_size = max(final_size, 1)  # Minimum 1 share
        max_shares = int(account_value * self.strategy_config.max_position_size / signal.price)
        final_size = min(final_size, max_shares)
        
        log_debug(f"Position size calculation for {signal.symbol}: "
                 f"base={base_size:.0f}, vol_adj={volatility_adjustment:.3f}, "
                 f"corr_adj={correlation_adjustment:.3f}, conc_adj={concentration_adjustment:.3f}, "
                 f"dd_adj={drawdown_adjustment:.3f}, final={final_size:.0f}")
        
        return int(final_size)
    
    def _calculate_base_size(self, signal: TradingSignal, account_value: float) -> float:
        """Calculate base position size"""
        
        if self.strategy_config.position_sizing_method == "fixed":
            position_value = account_value * self.strategy_config.max_position_size
        
        elif self.strategy_config.position_sizing_method == "volatility_target":
            target_vol = self.strategy_config.volatility_target
            predicted_vol = signal.predicted_volatility
            
            # Scale position inversely with volatility
            vol_scalar = min(target_vol / max(predicted_vol, 0.001), 3.0)  # Cap at 3x
            position_value = account_value * self.strategy_config.max_position_size * vol_scalar
        
        elif self.strategy_config.position_sizing_method == "kelly":
            # Simplified Kelly criterion
            win_rate = 0.55
            avg_win = 0.025
            avg_loss = 0.015
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            position_value = account_value * kelly_fraction
        
        else:
            position_value = account_value * self.strategy_config.max_position_size
        
        # Adjust for signal strength
        position_value *= signal.strength
        
        return position_value / signal.price
    
    def _calculate_volatility_adjustment(self, signal: TradingSignal) -> float:
        """Adjust position size based on predicted volatility"""
        
        predicted_vol = signal.predicted_volatility
        
        # Reduce position size for high volatility
        if predicted_vol > 0.03:  # 3% daily volatility
            return 0.5
        elif predicted_vol > 0.02:  # 2% daily volatility
            return 0.7
        elif predicted_vol < 0.005:  # Very low volatility
            return 1.2
        else:
            return 1.0
    
    def _calculate_correlation_adjustment(self, signal: TradingSignal, 
                                       current_positions: Dict[str, PositionInfo]) -> float:
        """Adjust position size based on correlation with existing positions"""
        
        # Simplified correlation adjustment
        # In practice, you would calculate actual correlations
        
        if len(current_positions) == 0:
            return 1.0
        
        # Reduce position size if we already have many positions
        if len(current_positions) >= 5:
            return 0.8
        elif len(current_positions) >= 3:
            return 0.9
        else:
            return 1.0
    
    def _calculate_concentration_adjustment(self, signal: TradingSignal,
                                         current_positions: Dict[str, PositionInfo],
                                         account_value: float) -> float:
        """Adjust position size based on portfolio concentration"""
        
        if len(current_positions) == 0:
            return 1.0
        
        # Calculate current concentration
        position_values = [abs(pos.market_value) for pos in current_positions.values()]
        max_position_value = max(position_values) if position_values else 0
        
        concentration = max_position_value / account_value
        
        # Reduce new position size if concentration is high
        if concentration > 0.3:  # 30% concentration
            return 0.6
        elif concentration > 0.2:  # 20% concentration
            return 0.8
        else:
            return 1.0
    
    def _calculate_drawdown_adjustment(self, risk_metrics: 'RiskMetrics') -> float:
        """Adjust position size based on current drawdown"""
        
        drawdown = risk_metrics.max_drawdown
        
        # Reduce position size during drawdown
        if drawdown > 0.15:  # 15% drawdown
            return 0.5
        elif drawdown > 0.10:  # 10% drawdown
            return 0.7
        elif drawdown > 0.05:  # 5% drawdown
            return 0.9
        else:
            return 1.0


class RiskManager:
    """
    Comprehensive risk management system
    """
    
    def __init__(self):
        self.risk_config = config.get_risk_config()
        self.position_sizer = PositionSizer()
        
        # Risk tracking
        self.risk_events = []
        self.daily_pnl_history = []
        self.drawdown_history = []
        
        # Performance tracking
        self.initial_portfolio_value = 100000  # Will be updated with actual value
        self.peak_portfolio_value = 100000
        self.daily_start_value = 100000
        
        # Risk limits
        self.emergency_stop_triggered = False
        self.daily_loss_limit_reached = False
        
        log_info("Risk manager initialized")
    
    def update_portfolio_value(self, current_value: float):
        """Update portfolio value and calculate metrics"""
        
        # Update peak value
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        # Calculate daily P&L
        daily_pnl = current_value - self.daily_start_value
        daily_pnl_pct = daily_pnl / self.daily_start_value
        
        # Store daily P&L
        self.daily_pnl_history.append({
            'timestamp': datetime.now(),
            'value': current_value,
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct
        })
        
        # Keep only recent history
        if len(self.daily_pnl_history) > 252:  # 1 year of daily data
            self.daily_pnl_history = self.daily_pnl_history[-252:]
        
        # Calculate drawdown
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'drawdown': drawdown,
            'peak_value': self.peak_portfolio_value,
            'current_value': current_value
        })
        
        # Check risk limits
        self._check_daily_loss_limit(daily_pnl_pct)
        self._check_drawdown_limit(drawdown)
    
    def _check_daily_loss_limit(self, daily_pnl_pct: float):
        """Check if daily loss limit is exceeded"""
        
        if daily_pnl_pct <= -self.risk_config.max_daily_loss:
            if not self.daily_loss_limit_reached:
                self.daily_loss_limit_reached = True
                
                risk_event = RiskEvent(
                    event_type=RiskEventType.DAILY_LOSS_LIMIT,
                    symbol="PORTFOLIO",
                    risk_level=RiskLevel.CRITICAL,
                    current_value=daily_pnl_pct,
                    threshold=-self.risk_config.max_daily_loss,
                    timestamp=datetime.now(),
                    message=f"Daily loss limit exceeded: {daily_pnl_pct:.2%}",
                    action_required=True,
                    recommended_action="Close all positions and stop trading"
                )
                
                self.risk_events.append(risk_event)
                log_risk(
                    event_type="DAILY_LOSS_LIMIT",
                    symbol="PORTFOLIO",
                    current_value=daily_pnl_pct,
                    threshold=-self.risk_config.max_daily_loss,
                    action_taken="Emergency stop triggered"
                )
    
    def _check_drawdown_limit(self, drawdown: float):
        """Check if maximum drawdown limit is exceeded"""
        
        if drawdown >= self.risk_config.max_drawdown:
            risk_event = RiskEvent(
                event_type=RiskEventType.DRAWDOWN_LIMIT,
                symbol="PORTFOLIO",
                risk_level=RiskLevel.CRITICAL,
                current_value=drawdown,
                threshold=self.risk_config.max_drawdown,
                timestamp=datetime.now(),
                message=f"Maximum drawdown exceeded: {drawdown:.2%}",
                action_required=True,
                recommended_action="Reduce position sizes and review strategy"
            )
            
            self.risk_events.append(risk_event)
            log_risk(
                event_type="DRAWDOWN_LIMIT",
                symbol="PORTFOLIO",
                current_value=drawdown,
                threshold=self.risk_config.max_drawdown,
                action_taken="Drawdown limit warning"
            )
    
    def calculate_risk_metrics(self, current_value: float, 
                             positions: Dict[str, PositionInfo]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Portfolio exposure
        total_exposure = sum(abs(pos.market_value) for pos in positions.values())
        
        # Maximum position size
        max_position_size = max([abs(pos.market_value) for pos in positions.values()], default=0)
        
        # Concentration ratio
        concentration_ratio = max_position_size / current_value if current_value > 0 else 0
        
        # Daily P&L
        daily_pnl = (current_value - self.daily_start_value) / self.daily_start_value if self.daily_start_value > 0 else 0
        
        # Maximum drawdown
        max_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        
        # Value at Risk (simplified)
        var_95, var_99 = self._calculate_var()
        
        # Portfolio volatility
        portfolio_volatility = self._calculate_portfolio_volatility()
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Overall risk level
        risk_level = self._assess_risk_level(max_drawdown, concentration_ratio, portfolio_volatility)
        
        return RiskMetrics(
            total_exposure=total_exposure,
            max_position_size=max_position_size,
            concentration_ratio=concentration_ratio,
            daily_pnl=daily_pnl,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            risk_level=risk_level
        )
    
    def _calculate_var(self) -> Tuple[float, float]:
        """Calculate Value at Risk"""
        
        if len(self.daily_pnl_history) < 30:
            return 0.0, 0.0
        
        # Get daily returns
        returns = [entry['daily_pnl_pct'] for entry in self.daily_pnl_history[-30:]]
        
        # Calculate percentiles
        var_95 = np.percentile(returns, 5)  # 5th percentile
        var_99 = np.percentile(returns, 1)  # 1st percentile
        
        return var_95, var_99
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        
        if len(self.daily_pnl_history) < 30:
            return 0.0
        
        returns = [entry['daily_pnl_pct'] for entry in self.daily_pnl_history[-30:]]
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        
        if len(self.daily_pnl_history) < 30:
            return 0.0
        
        returns = [entry['daily_pnl_pct'] for entry in self.daily_pnl_history[-30:]]
        
        avg_return = np.mean(returns) * 252  # Annualized return
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        if volatility == 0:
            return 0.0
        
        # Assume 2% risk-free rate
        risk_free_rate = 0.02
        
        return (avg_return - risk_free_rate) / volatility
    
    def _assess_risk_level(self, max_drawdown: float, 
                         concentration_ratio: float, 
                         portfolio_volatility: float) -> RiskLevel:
        """Assess overall risk level"""
        
        risk_score = 0
        
        # Drawdown risk
        if max_drawdown > 0.15:
            risk_score += 3
        elif max_drawdown > 0.10:
            risk_score += 2
        elif max_drawdown > 0.05:
            risk_score += 1
        
        # Concentration risk
        if concentration_ratio > 0.4:
            risk_score += 3
        elif concentration_ratio > 0.3:
            risk_score += 2
        elif concentration_ratio > 0.2:
            risk_score += 1
        
        # Volatility risk
        if portfolio_volatility > 0.3:
            risk_score += 3
        elif portfolio_volatility > 0.2:
            risk_score += 2
        elif portfolio_volatility > 0.15:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def should_allow_trade(self, signal: TradingSignal, 
                          current_positions: Dict[str, PositionInfo],
                          account_value: float) -> Tuple[bool, str]:
        """Check if trade should be allowed based on risk limits"""
        
        # Check emergency stop
        if self.emergency_stop_triggered:
            return False, "Emergency stop active"
        
        # Check daily loss limit
        if self.daily_loss_limit_reached:
            return False, "Daily loss limit reached"
        
        # Check position count limit
        if len(current_positions) >= self.risk_config.max_positions:
            return False, f"Maximum positions reached: {len(current_positions)}"
        
        # Check concentration risk
        if signal.symbol in current_positions:
            current_exposure = abs(current_positions[signal.symbol].market_value)
            concentration = current_exposure / account_value
            
            if concentration > self.risk_config.max_position_size:
                return False, f"Position concentration too high: {concentration:.2%}"
        
        # Check volatility risk
        if signal.predicted_volatility > 0.05:  # 5% daily volatility
            return False, f"Predicted volatility too high: {signal.predicted_volatility:.2%}"
        
        # Check signal quality
        if signal.strength < 0.3:
            return False, f"Signal strength too low: {signal.strength:.2f}"
        
        if signal.confidence < 0.5:
            return False, f"Signal confidence too low: {signal.confidence:.2f}"
        
        return True, "Trade approved"
    
    def calculate_stop_loss_price(self, entry_price: float, 
                                 side: str, 
                                 predicted_volatility: float) -> float:
        """Calculate dynamic stop loss price"""
        
        # Base stop loss percentage
        base_stop_pct = self.risk_config.stop_loss_pct
        
        # Adjust based on volatility
        volatility_multiplier = max(0.5, min(2.0, predicted_volatility / 0.02))
        adjusted_stop_pct = base_stop_pct * volatility_multiplier
        
        if side.upper() == "BUY":
            stop_price = entry_price * (1 - adjusted_stop_pct)
        else:
            stop_price = entry_price * (1 + adjusted_stop_pct)
        
        return stop_price
    
    def calculate_take_profit_price(self, entry_price: float, 
                                   side: str, 
                                   predicted_volatility: float) -> float:
        """Calculate dynamic take profit price"""
        
        # Base take profit percentage
        base_tp_pct = self.risk_config.take_profit_pct
        
        # Adjust based on volatility
        volatility_multiplier = max(0.8, min(1.5, predicted_volatility / 0.02))
        adjusted_tp_pct = base_tp_pct * volatility_multiplier
        
        if side.upper() == "BUY":
            tp_price = entry_price * (1 + adjusted_tp_pct)
        else:
            tp_price = entry_price * (1 - adjusted_tp_pct)
        
        return tp_price
    
    def check_position_exit_conditions(self, position: PositionInfo,
                                     current_price: float,
                                     predicted_volatility: float) -> Tuple[bool, str]:
        """Check if position should be exited based on risk conditions"""
        
        # Calculate unrealized P&L percentage
        pnl_pct = position.unrealized_pnl / abs(position.cost_basis)
        
        # Check stop loss
        stop_loss_price = self.calculate_stop_loss_price(
            position.cost_basis / position.quantity,
            "BUY" if position.quantity > 0 else "SELL",
            predicted_volatility
        )
        
        if position.quantity > 0:  # Long position
            if current_price <= stop_loss_price:
                return True, f"Stop loss triggered: {current_price:.2f} <= {stop_loss_price:.2f}"
        else:  # Short position
            if current_price >= stop_loss_price:
                return True, f"Stop loss triggered: {current_price:.2f} >= {stop_loss_price:.2f}"
        
        # Check take profit
        take_profit_price = self.calculate_take_profit_price(
            position.cost_basis / position.quantity,
            "BUY" if position.quantity > 0 else "SELL",
            predicted_volatility
        )
        
        if position.quantity > 0:  # Long position
            if current_price >= take_profit_price:
                return True, f"Take profit triggered: {current_price:.2f} >= {take_profit_price:.2f}"
        else:  # Short position
            if current_price <= take_profit_price:
                return True, f"Take profit triggered: {current_price:.2f} <= {take_profit_price:.2f}"
        
        # Check position timeout
        position_age = datetime.now() - position.entry_time
        max_hold_time = timedelta(minutes=self.risk_config.position_timeout_minutes)
        
        if position_age > max_hold_time:
            return True, f"Position timeout: {position_age} > {max_hold_time}"
        
        return False, "Position within risk limits"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        recent_events = [event.to_dict() for event in self.risk_events[-10:]]
        
        current_drawdown = 0.0
        if self.drawdown_history:
            current_drawdown = self.drawdown_history[-1]['drawdown']
        
        daily_pnl = 0.0
        if self.daily_pnl_history:
            daily_pnl = self.daily_pnl_history[-1]['daily_pnl_pct']
        
        return {
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'daily_loss_limit_reached': self.daily_loss_limit_reached,
            'current_drawdown': current_drawdown,
            'daily_pnl': daily_pnl,
            'recent_events': recent_events,
            'total_risk_events': len(self.risk_events),
            'peak_portfolio_value': self.peak_portfolio_value,
            'risk_limits': {
                'max_daily_loss': self.risk_config.max_daily_loss,
                'max_drawdown': self.risk_config.max_drawdown,
                'max_positions': self.risk_config.max_positions,
                'stop_loss_pct': self.risk_config.stop_loss_pct,
                'take_profit_pct': self.risk_config.take_profit_pct
            }
        }
    
    def reset_daily_limits(self):
        """Reset daily risk limits (call at start of each trading day)"""
        self.daily_loss_limit_reached = False
        self.daily_start_value = self.peak_portfolio_value
        
        log_info("Daily risk limits reset")
    
    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        self.emergency_stop_triggered = True
        
        risk_event = RiskEvent(
            event_type=RiskEventType.DAILY_LOSS_LIMIT,
            symbol="PORTFOLIO",
            risk_level=RiskLevel.CRITICAL,
            current_value=0.0,
            threshold=0.0,
            timestamp=datetime.now(),
            message=f"Emergency stop triggered: {reason}",
            action_required=True,
            recommended_action="Manual intervention required"
        )
        
        self.risk_events.append(risk_event)
        log_risk(
            event_type="EMERGENCY_STOP",
            symbol="PORTFOLIO",
            current_value=0.0,
            threshold=0.0,
            action_taken=f"Emergency stop: {reason}"
        )


# Global risk manager instance
risk_manager = RiskManager()