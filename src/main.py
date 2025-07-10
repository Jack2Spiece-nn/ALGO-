"""
Main Trading Engine for GARCH Intraday Strategy
"""

import asyncio
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import traceback
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.market_data import market_data_manager, MarketDataPoint
from src.models.garch_model import garch_manager
from src.strategy.garch_strategy import strategy_manager, TradingSignal
from src.execution.risk_manager import risk_manager
from src.utils.config import config
from src.utils.logger import trading_logger, log_info, log_error, log_debug


class TradingEngine:
    """
    Main trading engine that orchestrates all components
    """
    
    def __init__(self):
        self.is_running = False
        self.is_market_hours = False
        self.last_update_time = None
        
        # Component references
        self.market_data = market_data_manager
        self.garch_models = garch_manager
        self.strategies = strategy_manager
        self.risk_manager = risk_manager
        
        # Initialize executor based on broker configuration
        broker_name = config.get_broker_config().get('name', 'mt5')
        if broker_name == 'mt5':
            from src.execution.mt5_executor import mt5_executor
            self.executor = mt5_executor
        else:
            from src.execution.alpaca_executor import alpaca_executor
            self.executor = alpaca_executor
        
        # Configuration
        self.symbols = config.get_symbols()
        self.primary_symbol = self.symbols.get('primary', 'SPY')
        self.watchlist = self.symbols.get('watchlist', [self.primary_symbol])
        
        # Trading state
        self.current_positions = {}
        self.active_orders = {}
        self.daily_stats = {
            'trades_executed': 0,
            'signals_generated': 0,
            'pnl': 0.0,
            'start_time': None
        }
        
        # Performance tracking
        self.loop_count = 0
        self.average_loop_time = 0.0
        
        log_info("Trading engine initialized")
    
    def _check_market_hours(self) -> bool:
        """Check if market is open"""
        
        now = datetime.now()
        current_time = now.time()
        
        # Get trading hours configuration
        trading_hours = config.get_trading_hours_config()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Parse market hours
        from datetime import time
        market_open = time.fromisoformat(trading_hours.market_open)
        market_close = time.fromisoformat(trading_hours.market_close)
        
        # Check if current time is within market hours
        return market_open <= current_time <= market_close
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            log_info(f"Received signal {signum}, shutting down gracefully...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _initialize_components(self):
        """Initialize all trading components"""
        
        log_info("Initializing trading components...")
        
        # Initialize market data
        self.market_data.add_real_time_callback(self._on_market_data_update)
        
        # Initialize risk manager with current portfolio value
        portfolio_summary = self.executor.get_portfolio_summary()
        self.risk_manager.update_portfolio_value(portfolio_summary['account_value'])
        
        # Load historical data for GARCH models
        self._load_historical_data()
        
        log_info("Trading components initialized successfully")
    
    def _load_historical_data(self):
        """Load historical data for all symbols"""
        
        log_info("Loading historical data...")
        
        returns_data = {}
        
        for symbol in self.watchlist:
            try:
                # Get historical data
                historical_data = self.market_data.get_historical_data(
                    symbol, 
                    timeframe='1Min', 
                    days_back=config.get_data_config().get('lookback_days', 30)
                )
                
                if not historical_data.empty:
                    # Calculate returns
                    returns = self.market_data.calculate_returns(historical_data)
                    returns_data[symbol] = returns
                    
                    # Update strategy with historical data
                    strategy = self.strategies.get_strategy(symbol)
                    for idx, row in historical_data.iterrows():
                        data_point = MarketDataPoint(
                            symbol=symbol,
                            timestamp=idx,
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume'],
                            timeframe='1Min'
                        )
                        strategy.update_market_data(data_point)
                
                log_info(f"Loaded {len(historical_data)} data points for {symbol}")
                
            except Exception as e:
                log_error(f"Error loading historical data for {symbol}: {e}")
        
        # Fit GARCH models
        if returns_data:
            log_info("Fitting GARCH models...")
            fit_results = self.garch_models.fit_all_models(returns_data)
            
            for symbol, success in fit_results.items():
                if success:
                    log_info(f"GARCH model fitted successfully for {symbol}")
                else:
                    log_error(f"Failed to fit GARCH model for {symbol}")
    
    def _on_market_data_update(self, data_point: MarketDataPoint):
        """Handle real-time market data updates"""
        
        try:
            # Update strategy with new data
            strategy = self.strategies.get_strategy(data_point.symbol)
            strategy.update_market_data(data_point)
            
            log_debug(f"Market data updated for {data_point.symbol}: {data_point.close}")
            
        except Exception as e:
            log_error(f"Error processing market data update: {e}")
    
    def _generate_signals(self) -> Dict[str, Optional[TradingSignal]]:
        """Generate trading signals for all symbols"""
        
        signals = {}
        current_prices = {}
        
        # Get current prices
        for symbol in self.watchlist:
            try:
                quote = self.market_data.get_latest_quote(symbol)
                if quote:
                    current_prices[symbol] = (quote['bid'] + quote['ask']) / 2
                else:
                    # Fallback to last known price
                    strategy = self.strategies.get_strategy(symbol)
                    if not strategy.price_data.empty:
                        current_prices[symbol] = strategy.price_data['close'].iloc[-1]
            
            except Exception as e:
                log_error(f"Error getting current price for {symbol}: {e}")
        
        # Generate signals
        if current_prices:
            signals = self.strategies.generate_all_signals(current_prices)
            
            # Count signals generated
            signal_count = sum(1 for s in signals.values() if s is not None)
            self.daily_stats['signals_generated'] += signal_count
            
            log_debug(f"Generated {signal_count} signals")
        
        return signals
    
    def _execute_signals(self, signals: Dict[str, Optional[TradingSignal]]):
        """Execute trading signals"""
        
        # Update current positions
        self.current_positions = self.executor.positions
        
        # Get portfolio summary
        portfolio_summary = self.executor.get_portfolio_summary()
        account_value = portfolio_summary['account_value']
        
        # Update risk manager
        self.risk_manager.update_portfolio_value(account_value)
        
        # Calculate risk metrics
        risk_metrics = self.risk_manager.calculate_risk_metrics(
            account_value, self.current_positions
        )
        
        # Process each signal
        for symbol, signal in signals.items():
            if signal is None:
                continue
            
            try:
                # Check if trade should be allowed
                allowed, reason = self.risk_manager.should_allow_trade(
                    signal, self.current_positions, account_value
                )
                
                if not allowed:
                    log_debug(f"Trade not allowed for {symbol}: {reason}")
                    continue
                
                # Check if we should enter position
                strategy = self.strategies.get_strategy(symbol)
                if not strategy.should_enter_position(signal):
                    log_debug(f"Position entry conditions not met for {symbol}")
                    continue
                
                # Execute the signal
                result = self.executor.execute_signal(signal)
                
                if result and result.status.value != "REJECTED":
                    self.daily_stats['trades_executed'] += 1
                    
                    # Update strategy with position info
                    strategy.update_position({
                        'symbol': symbol,
                        'quantity': result.quantity,
                        'entry_price': result.price,
                        'entry_time': datetime.now(),
                        'order_id': result.order_id
                    })
                    
                    log_info(f"Signal executed: {symbol} {signal.signal_type.value} "
                            f"(strength: {signal.strength:.3f})")
                
            except Exception as e:
                log_error(f"Error executing signal for {symbol}: {e}")
    
    def _check_exit_conditions(self):
        """Check exit conditions for existing positions"""
        
        # Update positions
        self.current_positions = self.executor.positions
        
        for symbol, position in self.current_positions.items():
            try:
                # Get current price
                quote = self.market_data.get_latest_quote(symbol)
                if not quote:
                    continue
                
                current_price = (quote['bid'] + quote['ask']) / 2
                
                # Get GARCH prediction for volatility
                garch_model = self.garch_models.get_model(symbol)
                prediction = garch_model.predict()
                
                predicted_volatility = prediction.predicted_volatility if prediction else 0.02
                
                # Check risk-based exit conditions
                should_exit, reason = self.risk_manager.check_position_exit_conditions(
                    position, current_price, predicted_volatility
                )
                
                if should_exit:
                    log_info(f"Exiting position for {symbol}: {reason}")
                    
                    # Close position
                    result = self.executor.close_position(symbol)
                    
                    if result and result.status.value != "REJECTED":
                        # Update strategy
                        strategy = self.strategies.get_strategy(symbol)
                        strategy.update_position(None)
                        
                        self.daily_stats['trades_executed'] += 1
                        
                        log_info(f"Position closed: {symbol}")
                
                # Check strategy-based exit conditions
                strategy = self.strategies.get_strategy(symbol)
                if strategy.should_exit_position(current_price):
                    log_info(f"Strategy exit signal for {symbol}")
                    
                    result = self.executor.close_position(symbol)
                    
                    if result and result.status.value != "REJECTED":
                        strategy.update_position(None)
                        self.daily_stats['trades_executed'] += 1
                        
                        log_info(f"Position closed (strategy): {symbol}")
            
            except Exception as e:
                log_error(f"Error checking exit conditions for {symbol}: {e}")
    
    def _update_orders(self):
        """Update status of all active orders"""
        
        try:
            self.executor.update_all_orders()
            
        except Exception as e:
            log_error(f"Error updating orders: {e}")
    
    def _log_performance_metrics(self):
        """Log performance metrics"""
        
        try:
            # Portfolio summary
            portfolio_summary = self.executor.get_portfolio_summary()
            
            # Trading statistics
            trading_stats = self.executor.get_trading_statistics()
            
            # Strategy performance
            strategy_performance = self.strategies.get_performance_summary()
            
            # Risk summary
            risk_summary = self.risk_manager.get_risk_summary()
            
            # Log key metrics
            log_info(f"Portfolio Value: ${portfolio_summary['account_value']:.2f} | "
                    f"Daily P&L: {risk_summary['daily_pnl']:.2%} | "
                    f"Positions: {portfolio_summary['positions_count']} | "
                    f"Trades: {self.daily_stats['trades_executed']}")
            
        except Exception as e:
            log_error(f"Error logging performance metrics: {e}")
    
    async def _trading_loop(self):
        """Main trading loop"""
        
        loop_start_time = time.time()
        
        try:
            # Check market hours
            self.is_market_hours = self._check_market_hours()
            
            if not self.is_market_hours:
                log_debug("Market is closed, skipping trading loop")
                return
            
            # Generate signals
            signals = self._generate_signals()
            
            # Execute signals
            self._execute_signals(signals)
            
            # Check exit conditions
            self._check_exit_conditions()
            
            # Update orders
            self._update_orders()
            
            # Log performance metrics (every 10 loops)
            if self.loop_count % 10 == 0:
                self._log_performance_metrics()
            
            # Update loop statistics
            loop_time = time.time() - loop_start_time
            self.average_loop_time = (self.average_loop_time * self.loop_count + loop_time) / (self.loop_count + 1)
            self.loop_count += 1
            
            self.last_update_time = datetime.now()
            
        except Exception as e:
            log_error(f"Error in trading loop: {e}")
            log_error(traceback.format_exc())
    
    async def run(self):
        """Main run method"""
        
        log_info("Starting trading engine...")
        
        try:
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Initialize components
            self._initialize_components()
            
            # Start market data feed
            if self.is_market_hours:
                self.market_data.start_real_time_feed(self.watchlist)
            
            # Set running flag
            self.is_running = True
            self.daily_stats['start_time'] = datetime.now()
            
            log_info("Trading engine started successfully")
            
            # Main trading loop
            while self.is_running:
                await self._trading_loop()
                
                # Sleep for a short interval (e.g., 1 second)
                await asyncio.sleep(1)
        
        except Exception as e:
            log_error(f"Critical error in trading engine: {e}")
            log_error(traceback.format_exc())
        
        finally:
            await self._shutdown()
    
    async def _shutdown(self):
        """Graceful shutdown"""
        
        log_info("Shutting down trading engine...")
        
        try:
            # Stop market data feed
            self.market_data.stop_real_time_feed()
            
            # Close all positions if configured
            if config.get('strategy.close_positions_on_shutdown', True):
                log_info("Closing all positions...")
                self.executor.close_all_positions()
            
            # Cancel all pending orders
            log_info("Cancelling pending orders...")
            for order_id in self.active_orders.keys():
                self.executor.cancel_order(order_id)
            
            # Final performance log
            self._log_performance_metrics()
            
            log_info("Trading engine shutdown complete")
            
        except Exception as e:
            log_error(f"Error during shutdown: {e}")
    
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        log_info("Trading engine stop requested")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the trading engine"""
        
        return {
            'is_running': self.is_running,
            'is_market_hours': self.is_market_hours,
            'last_update_time': self.last_update_time,
            'loop_count': self.loop_count,
            'average_loop_time': self.average_loop_time,
            'daily_stats': self.daily_stats.copy(),
            'watchlist': self.watchlist,
            'current_positions': len(self.current_positions),
            'active_orders': len(self.active_orders)
        }


async def main():
    """Main entry point"""
    
    # Create and run trading engine
    engine = TradingEngine()
    
    try:
        await engine.run()
    
    except KeyboardInterrupt:
        log_info("Received keyboard interrupt")
    
    except Exception as e:
        log_error(f"Unhandled exception: {e}")
        log_error(traceback.format_exc())
    
    finally:
        log_info("Application terminated")


if __name__ == "__main__":
    # Run the trading engine
    asyncio.run(main())