"""
Alpaca API Integration for Order Execution
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
    TrailingStopOrderRequest, GetOrdersRequest, ClosePositionRequest
)
from alpaca.trading.models import Order, Position, Asset, TradeAccount
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus
from alpaca.common.exceptions import APIError

from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug, log_trade
from src.strategy.garch_strategy import TradingSignal, SignalType


class OrderExecutionStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    status: OrderExecutionStatus
    filled_quantity: float
    filled_price: Optional[float]
    timestamp: datetime
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'timestamp': self.timestamp,
            'error_message': self.error_message
        }


@dataclass
class PositionInfo:
    """Position information"""
    symbol: str
    quantity: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    current_price: float
    entry_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'current_price': self.current_price,
            'entry_time': self.entry_time
        }


class AlpacaExecutor:
    """
    Alpaca API executor for order management and execution
    """
    
    def __init__(self):
        self.config = config.get_broker_config()
        self.risk_config = config.get_risk_config()
        
        # Get API credentials
        credentials = config.get_api_credentials()
        self.api_key = credentials['alpaca_api_key']
        self.secret_key = credentials['alpaca_secret_key']
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not configured")
        
        # Initialize trading client
        self.client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=config.is_paper_trading()
        )
        
        # Order tracking
        self.active_orders = {}
        self.order_history = []
        self.positions = {}
        
        # Account information
        self.account_info = None
        self.buying_power = 0.0
        self.portfolio_value = 0.0
        
        # Rate limiting
        self.last_order_time = {}
        self.min_order_interval = 1.0  # Minimum seconds between orders
        
        # Initialize account info
        self._update_account_info()
        
        log_info(f"Alpaca executor initialized (Paper: {config.is_paper_trading()})")
    
    def _update_account_info(self):
        """Update account information"""
        try:
            account = self.client.get_account()
            self.account_info = account
            self.buying_power = float(account.buying_power)
            self.portfolio_value = float(account.portfolio_value)
            
            log_debug(f"Account updated - Buying Power: ${self.buying_power:.2f}, "
                     f"Portfolio Value: ${self.portfolio_value:.2f}")
            
        except APIError as e:
            log_error(f"Error updating account info: {e}")
        except Exception as e:
            log_error(f"Error updating account info: {e}")
    
    def _update_positions(self):
        """Update current positions"""
        try:
            positions = self.client.get_all_positions()
            
            self.positions = {}
            for position in positions:
                if float(position.qty) != 0:  # Only track non-zero positions
                    self.positions[position.symbol] = PositionInfo(
                        symbol=position.symbol,
                        quantity=float(position.qty),
                        market_value=float(position.market_value),
                        cost_basis=float(position.cost_basis),
                        unrealized_pnl=float(position.unrealized_pl),
                        unrealized_pnl_pct=float(position.unrealized_plpc),
                        current_price=float(position.current_price),
                        entry_time=datetime.now()  # Approximate, as Alpaca doesn't provide exact entry time
                    )
            
            log_debug(f"Updated positions: {list(self.positions.keys())}")
            
        except APIError as e:
            log_error(f"Error updating positions: {e}")
    
    def _check_rate_limit(self, symbol: str) -> bool:
        """Check if rate limit allows new order"""
        current_time = time.time()
        
        if symbol in self.last_order_time:
            time_since_last = current_time - self.last_order_time[symbol]
            if time_since_last < self.min_order_interval:
                return False
        
        self.last_order_time[symbol] = current_time
        return True
    
    def _calculate_position_size(self, symbol: str, signal: TradingSignal) -> float:
        """Calculate position size based on strategy configuration"""
        
        # Update account info
        self._update_account_info()
        
        strategy_config = config.get_strategy_config()
        
        # Base position size calculation
        if strategy_config.position_sizing_method == "fixed":
            # Fixed percentage of portfolio
            position_value = self.portfolio_value * strategy_config.max_position_size
        
        elif strategy_config.position_sizing_method == "volatility_target":
            # Volatility-based position sizing
            target_volatility = strategy_config.volatility_target
            predicted_volatility = signal.predicted_volatility
            
            # Scale position size inversely with volatility
            volatility_scalar = target_volatility / max(predicted_volatility, 0.001)
            position_value = self.portfolio_value * strategy_config.max_position_size * volatility_scalar
        
        elif strategy_config.position_sizing_method == "kelly":
            # Kelly Criterion (simplified)
            win_rate = 0.55  # Assume 55% win rate
            avg_win = 0.02   # Assume 2% average win
            avg_loss = 0.01  # Assume 1% average loss
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            position_value = self.portfolio_value * kelly_fraction
        
        else:
            position_value = self.portfolio_value * strategy_config.max_position_size
        
        # Adjust for signal strength
        position_value *= signal.strength
        
        # Convert to shares
        shares = position_value / signal.price
        
        # Round to whole shares
        shares = int(shares)
        
        # Ensure minimum position size
        shares = max(shares, 1)
        
        log_debug(f"Calculated position size for {symbol}: {shares} shares "
                 f"(${position_value:.2f} value)")
        
        return shares
    
    def _validate_order(self, symbol: str, side: OrderSide, quantity: float) -> Tuple[bool, str]:
        """Validate order before submission"""
        
        # Check rate limiting
        if not self._check_rate_limit(symbol):
            return False, "Rate limit exceeded"
        
        # Check quantity
        if quantity <= 0:
            return False, "Invalid quantity"
        
        # Check buying power for buy orders
        if side == OrderSide.BUY:
            # Get current price estimate
            try:
                quote = self.client.get_latest_quote(symbol)
                estimated_cost = float(quote.ask_price) * quantity
                
                if estimated_cost > self.buying_power:
                    return False, f"Insufficient buying power: ${self.buying_power:.2f} < ${estimated_cost:.2f}"
            
            except APIError:
                # If we can't get quote, proceed with caution
                pass
        
        # Check position limits
        current_positions = len(self.positions)
        max_positions = self.risk_config.max_positions
        
        if side == OrderSide.BUY and current_positions >= max_positions:
            return False, f"Maximum positions reached: {current_positions}/{max_positions}"
        
        # Check if asset is tradeable
        try:
            asset = self.client.get_asset(symbol)
            if not asset.tradable:
                return False, f"Asset {symbol} is not tradeable"
        
        except APIError:
            return False, f"Asset {symbol} not found"
        
        return True, "Order validated"
    
    def submit_market_order(self, symbol: str, side: OrderSide, 
                           quantity: float) -> Optional[OrderResult]:
        """Submit market order"""
        
        # Validate order
        valid, message = self._validate_order(symbol, side, quantity)
        if not valid:
            log_error(f"Order validation failed: {message}")
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=None,
                status=OrderExecutionStatus.REJECTED,
                filled_quantity=0,
                filled_price=None,
                timestamp=datetime.now(),
                error_message=message
            )
        
        try:
            # Create market order request
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = self.client.submit_order(order_request)
            
            # Track order
            self.active_orders[order.id] = order
            
            # Log trade
            log_trade(
                operation_type="MARKET_ORDER",
                symbol=symbol,
                quantity=quantity,
                price=0.0,  # Market price
                order_id=order.id,
                side=side.value
            )
            
            # Create result
            result = OrderResult(
                order_id=order.id,
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=None,
                status=OrderExecutionStatus.PENDING,
                filled_quantity=float(order.filled_qty) if order.filled_qty else 0,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                timestamp=datetime.now()
            )
            
            log_info(f"Market order submitted: {symbol} {side.value} {quantity} shares")
            return result
            
        except APIError as e:
            log_error(f"Error submitting market order: {e}")
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=None,
                status=OrderExecutionStatus.REJECTED,
                filled_quantity=0,
                filled_price=None,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def submit_limit_order(self, symbol: str, side: OrderSide, 
                          quantity: float, limit_price: float) -> Optional[OrderResult]:
        """Submit limit order"""
        
        # Validate order
        valid, message = self._validate_order(symbol, side, quantity)
        if not valid:
            log_error(f"Order validation failed: {message}")
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=limit_price,
                status=OrderExecutionStatus.REJECTED,
                filled_quantity=0,
                filled_price=None,
                timestamp=datetime.now(),
                error_message=message
            )
        
        try:
            # Create limit order request
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
            
            # Submit order
            order = self.client.submit_order(order_request)
            
            # Track order
            self.active_orders[order.id] = order
            
            # Log trade
            log_trade(
                operation_type="LIMIT_ORDER",
                symbol=symbol,
                quantity=quantity,
                price=limit_price,
                order_id=order.id,
                side=side.value
            )
            
            # Create result
            result = OrderResult(
                order_id=order.id,
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=limit_price,
                status=OrderExecutionStatus.PENDING,
                filled_quantity=float(order.filled_qty) if order.filled_qty else 0,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                timestamp=datetime.now()
            )
            
            log_info(f"Limit order submitted: {symbol} {side.value} {quantity} shares @ ${limit_price:.2f}")
            return result
            
        except APIError as e:
            log_error(f"Error submitting limit order: {e}")
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=limit_price,
                status=OrderExecutionStatus.REJECTED,
                filled_quantity=0,
                filled_price=None,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def submit_stop_loss_order(self, symbol: str, quantity: float, 
                              stop_price: float) -> Optional[OrderResult]:
        """Submit stop loss order"""
        
        try:
            # Create stop order request
            order_request = StopOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=stop_price
            )
            
            # Submit order
            order = self.client.submit_order(order_request)
            
            # Track order
            self.active_orders[order.id] = order
            
            # Log trade
            log_trade(
                operation_type="STOP_LOSS",
                symbol=symbol,
                quantity=quantity,
                price=stop_price,
                order_id=order.id,
                side="SELL"
            )
            
            # Create result
            result = OrderResult(
                order_id=order.id,
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=stop_price,
                status=OrderExecutionStatus.PENDING,
                filled_quantity=0,
                filled_price=None,
                timestamp=datetime.now()
            )
            
            log_info(f"Stop loss order submitted: {symbol} SELL {quantity} shares @ ${stop_price:.2f}")
            return result
            
        except APIError as e:
            log_error(f"Error submitting stop loss order: {e}")
            return OrderResult(
                order_id="",
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=stop_price,
                status=OrderExecutionStatus.REJECTED,
                filled_quantity=0,
                filled_price=None,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        
        try:
            self.client.cancel_order_by_id(order_id)
            
            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            
            log_info(f"Order {order_id} cancelled")
            return True
            
        except APIError as e:
            log_error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def close_position(self, symbol: str, percentage: float = 100.0) -> Optional[OrderResult]:
        """Close position (partial or full)"""
        
        # Update positions
        self._update_positions()
        
        if symbol not in self.positions:
            log_error(f"No position found for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Calculate quantity to close
        quantity_to_close = abs(position.quantity) * (percentage / 100.0)
        quantity_to_close = int(quantity_to_close)  # Round down to whole shares
        
        if quantity_to_close <= 0:
            log_error(f"Invalid quantity to close: {quantity_to_close}")
            return None
        
        # Determine side based on current position
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        
        # Submit market order to close position
        result = self.submit_market_order(symbol, side, quantity_to_close)
        
        if result and result.status != OrderExecutionStatus.REJECTED:
            log_info(f"Position close order submitted: {symbol} {side.value} {quantity_to_close} shares")
        
        return result
    
    def close_all_positions(self) -> Dict[str, OrderResult]:
        """Close all positions"""
        
        results = {}
        
        self._update_positions()
        
        for symbol in self.positions.keys():
            result = self.close_position(symbol)
            if result:
                results[symbol] = result
        
        log_info(f"Close all positions initiated for {len(results)} symbols")
        return results
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get order status"""
        
        try:
            order = self.client.get_order_by_id(order_id)
            
            # Map Alpaca status to our status
            status_map = {
                OrderStatus.NEW: OrderExecutionStatus.PENDING,
                OrderStatus.PARTIALLY_FILLED: OrderExecutionStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED: OrderExecutionStatus.FILLED,
                OrderStatus.DONE_FOR_DAY: OrderExecutionStatus.CANCELLED,
                OrderStatus.CANCELED: OrderExecutionStatus.CANCELLED,
                OrderStatus.EXPIRED: OrderExecutionStatus.EXPIRED,
                OrderStatus.REPLACED: OrderExecutionStatus.PENDING,
                OrderStatus.PENDING_CANCEL: OrderExecutionStatus.PENDING,
                OrderStatus.PENDING_REPLACE: OrderExecutionStatus.PENDING,
                OrderStatus.REJECTED: OrderExecutionStatus.REJECTED,
                OrderStatus.SUSPENDED: OrderExecutionStatus.CANCELLED,
                OrderStatus.PENDING_NEW: OrderExecutionStatus.PENDING,
                OrderStatus.CALCULATED: OrderExecutionStatus.PENDING,
                OrderStatus.STOPPED: OrderExecutionStatus.CANCELLED,
                OrderStatus.ACCEPTED: OrderExecutionStatus.PENDING,
                OrderStatus.ACCEPTED_FOR_BIDDING: OrderExecutionStatus.PENDING,
                OrderStatus.ARRIVED: OrderExecutionStatus.PENDING
            }
            
            status = status_map.get(order.status, OrderExecutionStatus.PENDING)
            
            result = OrderResult(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=float(order.qty),
                price=float(order.limit_price) if order.limit_price else None,
                status=status,
                filled_quantity=float(order.filled_qty) if order.filled_qty else 0,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                timestamp=order.created_at
            )
            
            return result
            
        except APIError as e:
            log_error(f"Error getting order status: {e}")
            return None
    
    def update_all_orders(self):
        """Update status of all active orders"""
        
        orders_to_remove = []
        
        for order_id in self.active_orders.keys():
            status = self.get_order_status(order_id)
            
            if status and status.status in [OrderExecutionStatus.FILLED, 
                                          OrderExecutionStatus.CANCELLED,
                                          OrderExecutionStatus.REJECTED,
                                          OrderExecutionStatus.EXPIRED]:
                orders_to_remove.append(order_id)
                self.order_history.append(status)
        
        # Remove completed orders from active tracking
        for order_id in orders_to_remove:
            del self.active_orders[order_id]
    
    def execute_signal(self, signal: TradingSignal) -> Optional[OrderResult]:
        """Execute trading signal"""
        
        if signal.signal_type == SignalType.HOLD:
            return None
        
        # Calculate position size
        quantity = self._calculate_position_size(signal.symbol, signal)
        
        # Determine order side
        side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
        
        # Submit market order for immediate execution
        result = self.submit_market_order(signal.symbol, side, quantity)
        
        if result and result.status != OrderExecutionStatus.REJECTED:
            log_info(f"Signal executed: {signal.symbol} {signal.signal_type.value} "
                    f"strength={signal.strength:.3f} quantity={quantity}")
        
        return result
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        
        self._update_account_info()
        self._update_positions()
        
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'account_value': self.portfolio_value,
            'buying_power': self.buying_power,
            'positions_count': len(self.positions),
            'active_orders': len(self.active_orders),
            'total_unrealized_pnl': total_pnl,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'is_paper_trading': config.is_paper_trading()
        }
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        
        completed_orders = [order for order in self.order_history 
                          if order.status == OrderExecutionStatus.FILLED]
        
        total_orders = len(completed_orders)
        
        if total_orders == 0:
            return {
                'total_orders': 0,
                'fill_rate': 0.0,
                'average_fill_time': 0.0,
                'total_volume': 0.0
            }
        
        buy_orders = [order for order in completed_orders if order.side == 'BUY']
        sell_orders = [order for order in completed_orders if order.side == 'SELL']
        
        total_volume = sum(order.filled_quantity * order.filled_price 
                          for order in completed_orders 
                          if order.filled_price is not None)
        
        return {
            'total_orders': total_orders,
            'buy_orders': len(buy_orders),
            'sell_orders': len(sell_orders),
            'fill_rate': 100.0,  # Market orders have 100% fill rate
            'total_volume': total_volume,
            'average_order_size': sum(order.filled_quantity for order in completed_orders) / total_orders
        }


# Global executor instance
alpaca_executor = AlpacaExecutor()