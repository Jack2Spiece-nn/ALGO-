"""
Real-time and historical market data pipeline for GARCH Intraday Trading Strategy
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import sqlite3
from pathlib import Path

# Optional imports for Alpaca (only needed when using Alpaca broker)
try:
    import yfinance as yf
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.models.bars import Bar
    from alpaca.data.models.quotes import Quote
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
try:
    import MetaTrader5 as mt5
except ImportError:
    # Mock MT5 for development/testing
    class MockMT5:
        def __init__(self):
            self.connected = False
            
        def initialize(self, **kwargs):
            return True
            
        def shutdown(self):
            pass
            
        def symbol_info(self, symbol):
            return type('SymbolInfo', (), {
                'name': symbol,
                'digits': 5,
                'point': 0.00001,
                'min_volume': 0.01,
                'max_volume': 100.0,
                'volume_step': 0.01,
                'contract_size': 100000.0,
                'tick_size': 0.00001,
                'tick_value': 1.0,
                'spread': 2,
                'currency_base': symbol[:3],
                'currency_profit': symbol[3:],
                'currency_margin': 'USD'
            })
            
        def symbol_info_tick(self, symbol):
            return type('SymbolTick', (), {
                'time': int(datetime.now().timestamp()),
                'bid': 1.0,
                'ask': 1.0002,
                'last': 1.0001,
                'volume': 1.0,
                'time_msc': int(datetime.now().timestamp() * 1000),
                'flags': 0,
                'volume_real': 1.0
            })
            
        def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
            import numpy as np
            # Generate mock historical data
            dates = pd.date_range(end=datetime.now(), periods=count, freq='1min')
            base_price = 1.0
            prices = base_price + np.cumsum(np.random.normal(0, 0.0001, count))
            
            data = []
            for i, date in enumerate(dates):
                data.append({
                    'time': int(date.timestamp()),
                    'open': prices[i],
                    'high': prices[i] * 1.0002,
                    'low': prices[i] * 0.9998,
                    'close': prices[i],
                    'tick_volume': np.random.randint(100, 1000),
                    'spread': 2,
                    'real_volume': np.random.randint(100, 1000)
                })
            
            return np.array(data, dtype=[
                ('time', 'i8'), ('open', 'f8'), ('high', 'f8'), 
                ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'),
                ('spread', 'i4'), ('real_volume', 'i8')
            ])
            
        # MT5 constants
        TIMEFRAME_M1 = 1
        TIMEFRAME_M5 = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_M30 = 30
        TIMEFRAME_H1 = 16385
        TIMEFRAME_H4 = 16388
        TIMEFRAME_D1 = 16408
        TIMEFRAME_W1 = 32769
        TIMEFRAME_MN1 = 49153
    
    mt5 = MockMT5()

import pytz

try:
    from src.utils.config import config
    from src.utils.logger import trading_logger, log_info, log_error, log_debug
except ImportError:
    try:
        from utils.config import config
        from utils.logger import trading_logger, log_info, log_error, log_debug
    except ImportError:
        # Fallback - create a basic config object with required methods
        class BasicConfig:
            def __init__(self):
                self.data = {
                    'broker': {
                        'name': 'mt5',
                        'login': 94435704,
                        'server': 'MetaQuotes-Demo',
                        'password': 'Z_4uYgWf',
                        'mt5_path': 'C:\\Users\\guest_1\\AppData\\Roaming\\MetaTrader 5\\terminal64.exe'
                    },
                    'symbols': {
                        'primary': 'EURUSD',
                        'watchlist': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
                    },
                    'data': {
                        'timeframes': ['1Min', '5Min', '15Min'],
                        'lookback_days': 252,
                        'min_data_points': 100
                    }
                }
            
            def get_broker_config(self):
                from dataclasses import dataclass
                from typing import Optional
                
                @dataclass
                class BrokerConfig:
                    name: str
                    login: Optional[int] = None
                    server: Optional[str] = None
                    password: Optional[str] = None
                    mt5_path: Optional[str] = None
                
                broker_data = self.data['broker']
                return BrokerConfig(**broker_data)
            
            def get_symbols(self):
                return self.data['symbols']
            
            def get_data_config(self):
                return self.data['data']
        
        config = BasicConfig()
        def log_info(msg): print(f"INFO: {msg}")
        def log_error(msg): print(f"ERROR: {msg}")
        def log_debug(msg): print(f"DEBUG: {msg}")
        trading_logger = None


@dataclass
class MarketDataPoint:
    """Standardized market data point structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe
        }


class MarketDataStorage:
    """Local storage for market data"""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for market data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                timeframe TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, timeframe)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON market_data(symbol, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timeframe 
            ON market_data(timeframe)
        ''')
        
        conn.commit()
        conn.close()
    
    def store_data(self, data_points: List[MarketDataPoint]):
        """Store market data points in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for point in data_points:
            cursor.execute('''
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                point.symbol, point.timestamp, point.open, point.high,
                point.low, point.close, point.volume, point.timeframe
            ))
        
        conn.commit()
        conn.close()
    
    def get_data(self, symbol: str, timeframe: str, 
                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve market data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        '''
        
        df = pd.read_sql_query(
            query, conn, 
            params=(symbol, timeframe, start_date, end_date),
            parse_dates=['timestamp']
        )
        
        conn.close()
        
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df


class AlpacaDataProvider:
    """Alpaca market data provider"""
    
    def __init__(self):
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca packages not available. Install alpaca-py and yfinance.")
            
        credentials = config.get_api_credentials()
        self.api_key = credentials['alpaca_api_key']
        self.secret_key = credentials['alpaca_secret_key']
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not configured")
        
        self.historical_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        self.stream = StockDataStream(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        self.data_callbacks: Dict[str, List[Callable]] = {}
    
    def get_historical_data(self, symbol: str, timeframe: str,
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical market data from Alpaca"""
        
        # Map timeframe to Alpaca TimeFrame
        timeframe_map = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute),
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day
        }
        
        alpaca_timeframe = timeframe_map.get(timeframe, TimeFrame.Minute)
        
        try:
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=alpaca_timeframe,
                start=start_date,
                end=end_date
            )
            
            bars = self.historical_client.get_stock_bars(request)
            
            if symbol not in bars.data:
                log_error(f"No data returned for symbol {symbol}")
                return pd.DataFrame()
            
            data = []
            for bar in bars.data[symbol]:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            log_info(f"Retrieved {len(df)} bars for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            log_error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()
    
    def subscribe_to_real_time_data(self, symbols: List[str], 
                                   callback: Callable[[MarketDataPoint], None]):
        """Subscribe to real-time market data"""
        
        async def handle_bar(bar: Bar):
            """Handle incoming bar data"""
            data_point = MarketDataPoint(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                timeframe='1Min'
            )
            
            callback(data_point)
        
        # Subscribe to minute bars
        self.stream.subscribe_bars(handle_bar, *symbols)
    
    def start_streaming(self):
        """Start real-time data streaming"""
        try:
            self.stream.run()
        except Exception as e:
            log_error(f"Error in data streaming: {e}")
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.stream.stop()


class YFinanceDataProvider:
    """Yahoo Finance data provider (backup/alternative)"""
    
    def __init__(self):
        if not ALPACA_AVAILABLE:
            raise ImportError("yfinance package not available. Install yfinance.")
    
    def get_historical_data(self, symbol: str, timeframe: str,
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        
        # Map timeframe to yfinance intervals
        interval_map = {
            '1Min': '1m',
            '5Min': '5m',
            '15Min': '15m',
            '1Hour': '1h',
            '1Day': '1d'
        }
        
        interval = interval_map.get(timeframe, '1m')
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                df.reset_index(inplace=True)
                df.rename(columns={'datetime': 'timestamp'}, inplace=True)
                df.set_index('timestamp', inplace=True)
            
            log_info(f"Retrieved {len(df)} bars from Yahoo Finance for {symbol}")
            return df
            
        except Exception as e:
            log_error(f"Error retrieving data from Yahoo Finance: {e}")
            return pd.DataFrame()


class MT5DataProvider:
    """MetaTrader 5 data provider"""
    
    def __init__(self):
        self.initialized = False
        self.connection_retry_count = 0
        self.max_retries = 3
        
    def initialize_connection(self) -> bool:
        """Initialize MT5 connection"""
        
        if self.initialized:
            return True
        
        try:
            # Get MT5 configuration
            broker_config = config.get_broker_config()
            
            # Initialize MT5 connection
            if not mt5.initialize(
                path=broker_config.get('mt5_path', ''),
                login=broker_config.get('login', 0),
                password=broker_config.get('password', ''),
                server=broker_config.get('server', ''),
                timeout=broker_config.get('timeout', 60000)
            ):
                log_error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
            
            # Check connection
            account_info = mt5.account_info()
            if account_info is None:
                log_error(f"Failed to get account info: {mt5.last_error()}")
                return False
            
            self.initialized = True
            log_info(f"MT5 connection established. Account: {account_info.login}, Server: {account_info.server}")
            return True
            
        except Exception as e:
            log_error(f"Error initializing MT5 connection: {e}")
            return False
    
    def shutdown_connection(self):
        """Shutdown MT5 connection"""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            log_info("MT5 connection closed")
    
    def get_historical_data(self, symbol: str, timeframe: str,
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical market data from MT5"""
        
        if not self.initialize_connection():
            return pd.DataFrame()
        
        # Map timeframe to MT5 timeframe
        timeframe_map = {
            '1Min': mt5.TIMEFRAME_M1,
            '5Min': mt5.TIMEFRAME_M5,
            '15Min': mt5.TIMEFRAME_M15,
            '30Min': mt5.TIMEFRAME_M30,
            '1Hour': mt5.TIMEFRAME_H1,
            '4Hour': mt5.TIMEFRAME_H4,
            '1Day': mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)
        
        try:
            # Get rates from MT5
            rates = mt5.copy_rates_range(
                symbol, 
                mt5_timeframe, 
                start_date, 
                end_date
            )
            
            if rates is None or len(rates) == 0:
                log_error(f"No data returned for symbol {symbol}: {mt5.last_error()}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to match our standard format
            df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            log_info(f"Retrieved {len(df)} bars from MT5 for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            log_error(f"Error retrieving historical data from MT5: {e}")
            return pd.DataFrame()
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote from MT5"""
        
        if not self.initialize_connection():
            return None
        
        try:
            # Get symbol tick
            tick = mt5.symbol_info_tick(symbol)
            
            if tick is None:
                log_error(f"Failed to get tick for {symbol}: {mt5.last_error()}")
                return None
            
            return {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume
            }
            
        except Exception as e:
            log_error(f"Error getting latest quote from MT5: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information from MT5"""
        
        if not self.initialize_connection():
            return None
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                log_error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
                return None
            
            return {
                'symbol': symbol,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'spread': symbol_info.spread,
                'trade_contract_size': symbol_info.trade_contract_size,
                'trade_tick_value': symbol_info.trade_tick_value,
                'trade_tick_size': symbol_info.trade_tick_size,
                'minimum_volume': symbol_info.volume_min,
                'maximum_volume': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'margin_initial': symbol_info.margin_initial,
                'margin_maintenance': symbol_info.margin_maintenance
            }
            
        except Exception as e:
            log_error(f"Error getting symbol info from MT5: {e}")
            return None
    
    def subscribe_to_real_time_data(self, symbols: List[str], 
                                   callback: Callable[[MarketDataPoint], None]):
        """Subscribe to real-time market data from MT5"""
        
        if not self.initialize_connection():
            return
        
        # Note: MT5 doesn't have streaming API like Alpaca
        # We'll implement a polling mechanism for real-time data
        log_info(f"MT5 real-time data subscription requested for {symbols}")
        log_info("Note: MT5 uses polling mechanism for real-time data")
    
    def get_current_rates(self, symbols: List[str]) -> Dict[str, MarketDataPoint]:
        """Get current rates for multiple symbols"""
        
        if not self.initialize_connection():
            return {}
        
        current_rates = {}
        
        for symbol in symbols:
            try:
                # Get the latest completed bar
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
                
                if rates is not None and len(rates) > 0:
                    rate = rates[0]
                    
                    current_rates[symbol] = MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(rate['time']),
                        open=rate['open'],
                        high=rate['high'],
                        low=rate['low'],
                        close=rate['close'],
                        volume=rate['tick_volume'],
                        timeframe='1Min'
                    )
                
            except Exception as e:
                log_error(f"Error getting current rate for {symbol}: {e}")
        
        return current_rates


class MarketDataManager:
    """Central market data management system"""
    
    def __init__(self):
        self.storage = MarketDataStorage()
        
        # Check broker configuration to determine primary provider
        broker_config = config.get_broker_config()
        if broker_config.name == 'mt5':
            self.primary_provider = MT5DataProvider()
            self.backup_provider = YFinanceDataProvider() if ALPACA_AVAILABLE else None
        else:
            self.primary_provider = AlpacaDataProvider() if ALPACA_AVAILABLE else MT5DataProvider()
            self.backup_provider = YFinanceDataProvider() if ALPACA_AVAILABLE else None
        
        self.real_time_callbacks: List[Callable] = []
        self.is_streaming = False
        
        self.symbols = config.get_symbols()
        self.timeframes = config.get_data_config().get('timeframes', ['1Min'])
        
        log_info("Market data manager initialized")
    
    def add_real_time_callback(self, callback: Callable[[MarketDataPoint], None]):
        """Add callback for real-time data updates"""
        self.real_time_callbacks.append(callback)
    
    def _handle_real_time_data(self, data_point: MarketDataPoint):
        """Handle incoming real-time data"""
        # Store in database
        self.storage.store_data([data_point])
        
        # Call all registered callbacks
        for callback in self.real_time_callbacks:
            try:
                callback(data_point)
            except Exception as e:
                log_error(f"Error in real-time callback: {e}")
    
    def get_historical_data(self, symbol: str, timeframe: str = '1Min',
                          days_back: int = 30) -> pd.DataFrame:
        """Get historical market data with fallback providers"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Try to get from local storage first
        df = self.storage.get_data(symbol, timeframe, start_date, end_date)
        
        # If not enough data, fetch from API
        if df.empty or len(df) < config.get_data_config().get('min_data_points', 100):
            log_info(f"Fetching historical data for {symbol} from API")
            
            # Try primary provider first
            try:
                df = self.primary_provider.get_historical_data(
                    symbol, timeframe, start_date, end_date
                )
                
                if not df.empty:
                    # Store in local database
                    data_points = [
                        MarketDataPoint(
                            symbol=symbol,
                            timestamp=row.Index,
                            open=row.open,
                            high=row.high,
                            low=row.low,
                            close=row.close,
                            volume=row.volume,
                            timeframe=timeframe
                        )
                        for row in df.itertuples()
                    ]
                    self.storage.store_data(data_points)
                
            except Exception as e:
                log_error(f"Primary provider failed: {e}")
                
                # Try backup provider
                try:
                    df = self.backup_provider.get_historical_data(
                        symbol, timeframe, start_date, end_date
                    )
                except Exception as e:
                    log_error(f"Backup provider failed: {e}")
                    return pd.DataFrame()
        
        return df
    
    def start_real_time_feed(self, symbols: Optional[List[str]] = None):
        """Start real-time market data feed"""
        
        if self.is_streaming:
            log_info("Real-time feed already running")
            return
        
        symbols = symbols or self.symbols.get('watchlist', [self.symbols.get('primary')])
        
        self.primary_provider.subscribe_to_real_time_data(
            symbols, self._handle_real_time_data
        )
        
        # Start streaming in background
        asyncio.create_task(self._start_streaming())
        self.is_streaming = True
        
        log_info(f"Started real-time data feed for {symbols}")
    
    async def _start_streaming(self):
        """Start streaming in async context"""
        try:
            self.primary_provider.start_streaming()
        except Exception as e:
            log_error(f"Error in real-time streaming: {e}")
            self.is_streaming = False
    
    def stop_real_time_feed(self):
        """Stop real-time market data feed"""
        if self.is_streaming:
            self.primary_provider.stop_streaming()
            self.is_streaming = False
            log_info("Stopped real-time data feed")
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote for a symbol"""
        try:
            # Use primary provider's get_latest_quote method
            quote = self.primary_provider.get_latest_quote(symbol)
            return quote
            
        except Exception as e:
            log_error(f"Error getting latest quote: {e}")
        
        return None
    
    def get_current_rates(self, symbols: List[str]) -> Dict[str, MarketDataPoint]:
        """Get current rates for multiple symbols (MT5 specific)"""
        if hasattr(self.primary_provider, 'get_current_rates'):
            return self.primary_provider.get_current_rates(symbols)
        else:
            # Fallback for non-MT5 providers
            current_rates = {}
            for symbol in symbols:
                quote = self.get_latest_quote(symbol)
                if quote:
                    # Create a basic MarketDataPoint from quote data
                    current_rates[symbol] = MarketDataPoint(
                        symbol=symbol,
                        timestamp=quote.get('timestamp', datetime.now()),
                        open=quote.get('last', quote.get('bid', 0)),
                        high=quote.get('last', quote.get('bid', 0)),
                        low=quote.get('last', quote.get('bid', 0)),
                        close=quote.get('last', quote.get('bid', 0)),
                        volume=quote.get('volume', 0),
                        timeframe='1Min'
                    )
            return current_rates
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data"""
        if df.empty:
            return pd.Series()
        
        returns = df['close'].pct_change().dropna()
        return returns
    
    def get_market_hours_data(self, symbol: str, timeframe: str = '1Min') -> pd.DataFrame:
        """Get data only for market hours"""
        
        trading_hours = config.get_trading_hours_config()
        
        df = self.get_historical_data(symbol, timeframe)
        
        if df.empty:
            return df
        
        # Filter for market hours
        market_open = trading_hours.market_open
        market_close = trading_hours.market_close
        
        df = df.between_time(market_open, market_close)
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        if df.empty:
            return False
        
        # Check for missing values
        if df.isnull().any().any():
            log_error("Data contains missing values")
            return False
        
        # Check for negative prices
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            log_error("Data contains negative prices")
            return False
        
        # Check for high > low, etc.
        if (df['high'] < df['low']).any():
            log_error("Data contains high < low")
            return False
        
        return True
    
    def get_data_summary(self, symbol: str) -> Dict[str, Any]:
        """Get data summary for a symbol"""
        summary = {}
        
        for timeframe in self.timeframes:
            df = self.get_historical_data(symbol, timeframe)
            
            if not df.empty:
                summary[timeframe] = {
                    'count': len(df),
                    'start_date': df.index.min(),
                    'end_date': df.index.max(),
                    'avg_volume': df['volume'].mean(),
                    'price_range': [df['low'].min(), df['high'].max()]
                }
        
        return summary


# Global market data manager instance
market_data_manager = MarketDataManager()