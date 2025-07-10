"""
MetaTrader 5 Integration for Order Execution
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    # Mock MT5 for development/testing purposes
    class MockMT5:
        def __init__(self):
            self.connected = False
            self.mock_positions = {}
            self.mock_orders = {}
            self.mock_balance = 10000.0
            self.order_counter = 1000
            
        def initialize(self, **kwargs):
            self.connected = True
            return True
            
        def shutdown(self):
            self.connected = False
            
        def login(self, login, password=None, server=None):
            return True
            
        def account_info(self):
            return type('AccountInfo', (), {
                'balance': self.mock_balance,
                'equity': self.mock_balance,
                'margin': 0.0,
                'margin_free': self.mock_balance,
                'margin_level': 0.0,
                'currency': 'USD',
                'profit': 0.0,
                'credit': 0.0,
                'leverage': 100
            })
            
        def positions_get(self, symbol=None):
            if symbol:
                return [pos for pos in self.mock_positions.values() if pos.symbol == symbol]
            return list(self.mock_positions.values())
            
        def orders_get(self, symbol=None):
            if symbol:
                return [order for order in self.mock_orders.values() if order.symbol == symbol]
            return list(self.mock_orders.values())
            
        def order_send(self, request):
            self.order_counter += 1
            order_id = self.order_counter
            
            # Mock successful order
            result = type('OrderResult', (), {
                'retcode': 10009,  # TRADE_RETCODE_DONE
                'order': order_id,
                'deal': order_id,
                'volume': request.get('volume', 0.1),
                'price': request.get('price', 1.0),
                'comment': 'Mock order'
            })
            
            return result
            
        def symbol_info(self, symbol):
            return type('SymbolInfo', (), {
                'name': symbol,
                'digits': 5,
                'point': 0.00001,
                'min_volume': 0.01,
                'max_volume': 100.0,
                'volume_step': 0.01,
                'contract_size': 100000.0,
                'margin_initial': 100.0,
                'margin_maintenance': 50.0,
                'tick_size': 0.00001,
                'tick_value': 1.0,
                'spread': 2,
                'currency_base': symbol[:3],
                'currency_profit': symbol[3:],
                'currency_margin': 'USD'
            })
            
        def symbol_info_tick(self, symbol):
            return type('SymbolTick', (), {
                'time': int(time.time()),
                'bid': 1.0,
                'ask': 1.0002,
                'last': 1.0001,
                'volume': 1.0,
                'time_msc': int(time.time() * 1000),
                'flags': 0,
                'volume_real': 1.0
            })
            
        # Constants
        TRADE_ACTION_DEAL = 1
        TRADE_ACTION_PENDING = 5
        TRADE_ACTION_SLTP = 7
        TRADE_ACTION_MODIFY = 8
        TRADE_ACTION_REMOVE = 9
        
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        ORDER_TYPE_BUY_LIMIT = 2
        ORDER_TYPE_SELL_LIMIT = 3
        ORDER_TYPE_BUY_STOP = 4
        ORDER_TYPE_SELL_STOP = 5
        
        ORDER_FILLING_FOK = 0
        ORDER_FILLING_IOC = 1
        ORDER_FILLING_RETURN = 2
        
        POSITION_TYPE_BUY = 0
        POSITION_TYPE_SELL = 1
        
        TRADE_RETCODE_DONE = 10009
        TRADE_RETCODE_REQUOTE = 10004
        TRADE_RETCODE_REJECT = 10006
        TRADE_RETCODE_CANCEL = 10007
        TRADE_RETCODE_PLACED = 10008
        TRADE_RETCODE_TIMEOUT = 10012
        TRADE_RETCODE_INVALID = 10013
        TRADE_RETCODE_INVALID_VOLUME = 10014
        TRADE_RETCODE_INVALID_PRICE = 10015
        TRADE_RETCODE_INVALID_STOPS = 10016
        TRADE_RETCODE_TRADE_DISABLED = 10017
        TRADE_RETCODE_MARKET_CLOSED = 10018
        TRADE_RETCODE_NO_MONEY = 10019
        TRADE_RETCODE_PRICE_CHANGED = 10020
        TRADE_RETCODE_PRICE_OFF = 10021
        TRADE_RETCODE_INVALID_EXPIRATION = 10022
        TRADE_RETCODE_ORDER_CHANGED = 10023
        TRADE_RETCODE_TOO_MANY_REQUESTS = 10024
        TRADE_RETCODE_NO_CHANGES = 10025
        TRADE_RETCODE_SERVER_DISABLES_AT = 10026
        TRADE_RETCODE_CLIENT_DISABLES_AT = 10027
        TRADE_RETCODE_LOCKED = 10028
        TRADE_RETCODE_FROZEN = 10029
        TRADE_RETCODE_INVALID_FILL = 10030
        TRADE_RETCODE_CONNECTION = 10031
        TRADE_RETCODE_ONLY_REAL = 10032
        TRADE_RETCODE_LIMIT_ORDERS = 10033
        TRADE_RETCODE_LIMIT_VOLUME = 10034
        TRADE_RETCODE_INVALID_ORDER = 10035
        TRADE_RETCODE_POSITION_CLOSED = 10036
        TRADE_RETCODE_INVALID_CLOSE_VOLUME = 10038
        TRADE_RETCODE_CLOSE_ORDER_EXIST = 10039
        TRADE_RETCODE_LIMIT_POSITIONS = 10040
        TRADE_RETCODE_REJECT_CANCEL = 10041
        TRADE_RETCODE_LONG_ONLY = 10042
        TRADE_RETCODE_SHORT_ONLY = 10043
        TRADE_RETCODE_CLOSE_ONLY = 10044
        TRADE_RETCODE_FIFO_CLOSE = 10045
        TRADE_RETCODE_HEDGE_PROHIBITED = 10046
    
    mt5 = MockMT5()

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


class MT5Executor:
    """
    MetaTrader 5 executor for order management and execution
    """
    
    def __init__(self):
        self.config = config.get_broker_config()
        self.risk_config = config.get_risk_config()
        self.mt5_config = config.get('mt5_position_sizing', {})
        
        # MT5 connection parameters
        self.mt5_login = self.config.login or 0
        self.mt5_password = self.config.password or ''
        self.mt5_server = self.config.server or ''
        self.mt5_timeout = self.config.timeout or 60000
        
        # Position sizing configuration
        self.lot_size = self.mt5_config.get('lot_size', 0.1)
        self.max_lot_size = self.mt5_config.get('max_lot_size', 1.0)
        self.min_lot_size = self.mt5_config.get('min_lot_size', 0.01)
        self.lot_step = self.mt5_config.get('lot_step', 0.01)
        self.risk_per_trade = self.mt5_config.get('risk_per_trade', 0.02)
        self.use_fixed_lot = self.mt5_config.get('use_fixed_lot', False)
        
        # State tracking
        self.connected = False
        self.positions = {}
        self.orders = {}
        self.account_info = None
        self.last_order_time = 0
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_pnl = 0.0
        
        # Initialize connection
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to MetaTrader 5 terminal"""
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                log_error("Failed to initialize MT5 connection")
                return False
            
            # Login to trading account
            if self.mt5_login and self.mt5_password:
                if not mt5.login(self.mt5_login, password=self.mt5_password, server=self.mt5_server):
                    log_error(f"Failed to login to MT5 account {self.mt5_login}")
                    return False
            
            self.connected = True
            self.account_info = mt5.account_info()
            
            log_info(f"Connected to MT5 - Account: {self.mt5_login}, Balance: ${self.account_info.balance:.2f}")
            return True
            
        except Exception as e:
            log_error(f"Error connecting to MT5: {e}")
            return False
    
    def _disconnect(self):
        """Disconnect from MetaTrader 5"""
        try:
            mt5.shutdown()
            self.connected = False
            log_info("Disconnected from MT5")
        except Exception as e:
            log_error(f"Error disconnecting from MT5: {e}")
    
    def _ensure_connection(self) -> bool:
        """Ensure MT5 connection is active"""
        if not self.connected:
            return self._connect()
        return True
    
    def _calculate_lot_size(self, symbol: str, signal: TradingSignal, account_balance: float) -> float:
        """Calculate position size in lots based on risk management"""
        try:
            if self.use_fixed_lot:
                return self.lot_size
            
            # Get symbol information
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                log_error(f"Failed to get symbol info for {symbol}")
                return self.lot_size
            
            # Calculate risk amount
            risk_amount = account_balance * self.risk_per_trade
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                log_error(f"Failed to get tick data for {symbol}")
                return self.lot_size
            
            current_price = tick.bid if signal.signal_type == SignalType.SELL else tick.ask
            
            # Calculate position size based on risk
            # For forex: risk_amount / (stop_loss_pips * pip_value * lot_size)
            stop_loss_pct = self.risk_config.get('stop_loss_pct', 0.015)
            stop_loss_pips = (current_price * stop_loss_pct) / symbol_info.point
            
            # Calculate pip value (for standard lot)
            pip_value = symbol_info.point * symbol_info.contract_size
            
            # Calculate lot size
            calculated_lots = risk_amount / (stop_loss_pips * pip_value)
            
            # Apply signal strength adjustment
            calculated_lots *= signal.strength
            
            # Apply constraints
            calculated_lots = max(self.min_lot_size, min(calculated_lots, self.max_lot_size))
            
            # Round to lot step
            calculated_lots = round(calculated_lots / self.lot_step) * self.lot_step
            
            log_debug(f"Calculated lot size for {symbol}: {calculated_lots} (risk: ${risk_amount:.2f})")
            return calculated_lots
            
        except Exception as e:
            log_error(f"Error calculating lot size: {e}")
            return self.lot_size
    
    def _create_order_request(self, symbol: str, order_type: int, volume: float, 
                            price: float = 0.0, sl: float = 0.0, tp: float = 0.0) -> Dict:
        """Create MT5 order request"""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,  # Price deviation in points
            "magic": 12345,   # Magic number for EA identification
            "comment": "GARCH Strategy",
            "type_time": mt5.ORDER_FILLING_IOC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        return request
    
    def _parse_order_result(self, result, symbol: str, side: str, volume: float) -> OrderResult:
        """Parse MT5 order result"""
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            status = OrderExecutionStatus.FILLED
            error_message = None
        else:
            status = OrderExecutionStatus.REJECTED
            error_message = f"MT5 Error: {result.retcode}"
        
        return OrderResult(
            order_id=str(result.order),
            symbol=symbol,
            side=side,
            quantity=volume,
            price=result.price if hasattr(result, 'price') else None,
            status=status,
            filled_quantity=result.volume if hasattr(result, 'volume') else 0,
            filled_price=result.price if hasattr(result, 'price') else None,
            timestamp=datetime.now(),
            error_message=error_message
        )
    
    def execute_signal(self, signal: TradingSignal) -> Optional[OrderResult]:
        """Execute a trading signal"""
        if not self._ensure_connection():
            return None
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_order_time < 1:
            log_debug("Rate limiting: waiting before next order")
            time.sleep(1)
        
        try:
            # Get current account info
            account_info = mt5.account_info()
            if not account_info:
                log_error("Failed to get account information")
                return None
            
            # Calculate position size
            lot_size = self._calculate_lot_size(signal.symbol, signal, account_info.balance)
            
            # Get current price
            tick = mt5.symbol_info_tick(signal.symbol)
            if not tick:
                log_error(f"Failed to get tick data for {signal.symbol}")
                return None
            
            # Determine order type and price
            if signal.signal_type == SignalType.BUY:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                side = "BUY"
            elif signal.signal_type == SignalType.SELL:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                side = "SELL"
            else:
                log_debug(f"No action for HOLD signal for {signal.symbol}")
                return None
            
            # Calculate stop loss and take profit
            stop_loss_pct = self.risk_config.get('stop_loss_pct', 0.015)
            take_profit_pct = self.risk_config.get('take_profit_pct', 0.03)
            
            if signal.signal_type == SignalType.BUY:
                sl = price * (1 - stop_loss_pct)
                tp = price * (1 + take_profit_pct)
            else:
                sl = price * (1 + stop_loss_pct)
                tp = price * (1 - take_profit_pct)
            
            # Create order request
            request = self._create_order_request(signal.symbol, order_type, lot_size, price, sl, tp)
            
            # Execute order
            result = mt5.order_send(request)
            
            # Parse result
            order_result = self._parse_order_result(result, signal.symbol, side, lot_size)
            
            # Update statistics
            self.total_trades += 1
            if order_result.status == OrderExecutionStatus.FILLED:
                self.successful_trades += 1
                log_trade(f"Order executed: {signal.symbol} {side} {lot_size} lots at {price:.5f}")
            else:
                self.failed_trades += 1
                log_error(f"Order failed: {order_result.error_message}")
            
            # Update last order time
            self.last_order_time = current_time
            
            return order_result
            
        except Exception as e:
            log_error(f"Error executing signal: {e}")
            return None
    
    def close_position(self, symbol: str) -> Optional[OrderResult]:
        """Close a position"""
        if not self._ensure_connection():
            return None
        
        try:
            # Get current positions
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                log_debug(f"No positions found for {symbol}")
                return None
            
            # Close the first position (in case of multiple)
            position = positions[0]
            
            # Determine close order type
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                side = "SELL"
            else:
                order_type = mt5.ORDER_TYPE_BUY
                side = "BUY"
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                log_error(f"Failed to get tick data for {symbol}")
                return None
            
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
            
            # Create close order request
            request = self._create_order_request(symbol, order_type, position.volume, price)
            
            # Execute close order
            result = mt5.order_send(request)
            
            # Parse result
            order_result = self._parse_order_result(result, symbol, side, position.volume)
            
            if order_result.status == OrderExecutionStatus.FILLED:
                log_trade(f"Position closed: {symbol} {side} {position.volume} lots at {price:.5f}")
            else:
                log_error(f"Failed to close position: {order_result.error_message}")
            
            return order_result
            
        except Exception as e:
            log_error(f"Error closing position: {e}")
            return None
    
    def close_all_positions(self) -> List[OrderResult]:
        """Close all open positions"""
        results = []
        
        try:
            positions = mt5.positions_get()
            if not positions:
                log_info("No positions to close")
                return results
            
            for position in positions:
                result = self.close_position(position.symbol)
                if result:
                    results.append(result)
            
            log_info(f"Closed {len(results)} positions")
            return results
            
        except Exception as e:
            log_error(f"Error closing all positions: {e}")
            return results
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if not self._ensure_connection():
            return False
        
        try:
            # Get order information
            orders = mt5.orders_get()
            order = None
            for o in orders:
                if str(o.ticket) == order_id:
                    order = o
                    break
            
            if not order:
                log_debug(f"Order {order_id} not found")
                return False
            
            # Create cancel request
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
            }
            
            # Execute cancel
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                log_info(f"Order {order_id} cancelled successfully")
                return True
            else:
                log_error(f"Failed to cancel order {order_id}: {result.retcode}")
                return False
                
        except Exception as e:
            log_error(f"Error cancelling order: {e}")
            return False
    
    def get_positions(self) -> Dict[str, PositionInfo]:
        """Get all open positions"""
        if not self._ensure_connection():
            return {}
        
        try:
            positions = mt5.positions_get()
            if not positions:
                return {}
            
            position_info = {}
            for pos in positions:
                # Get current price
                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue
                
                current_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                
                # Calculate unrealized P&L
                unrealized_pnl = pos.profit
                unrealized_pnl_pct = (unrealized_pnl / abs(pos.volume * pos.price_open)) * 100
                
                position_info[pos.symbol] = PositionInfo(
                    symbol=pos.symbol,
                    quantity=pos.volume if pos.type == mt5.POSITION_TYPE_BUY else -pos.volume,
                    market_value=pos.volume * current_price,
                    cost_basis=pos.volume * pos.price_open,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    current_price=current_price,
                    entry_time=datetime.fromtimestamp(pos.time)
                )
            
            self.positions = position_info
            return position_info
            
        except Exception as e:
            log_error(f"Error getting positions: {e}")
            return {}
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        if not self._ensure_connection():
            return {}
        
        try:
            account_info = mt5.account_info()
            if not account_info:
                return {}
            
            positions = self.get_positions()
            
            return {
                'account_value': account_info.balance,
                'buying_power': account_info.margin_free,
                'equity': account_info.equity,
                'positions_count': len(positions),
                'total_pnl': sum(pos.unrealized_pnl for pos in positions.values()),
                'margin_used': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'currency': account_info.currency,
                'leverage': account_info.leverage
            }
            
        except Exception as e:
            log_error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'success_rate': (self.successful_trades / max(self.total_trades, 1)) * 100,
            'total_pnl': self.total_pnl
        }
    
    def update_all_orders(self):
        """Update status of all orders"""
        if not self._ensure_connection():
            return
        
        try:
            orders = mt5.orders_get()
            if orders:
                self.orders = {str(order.ticket): order for order in orders}
            else:
                self.orders = {}
                
        except Exception as e:
            log_error(f"Error updating orders: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'connected') and self.connected:
            self._disconnect()


# Create global instance
mt5_executor = MT5Executor()