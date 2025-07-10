"""
Real-time Trading Dashboard for GARCH Intraday Strategy

This module provides a comprehensive web-based dashboard for monitoring
trading performance, risk metrics, and system health in real-time.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from typing import Dict, List, Any, Optional
import asyncio

from src.models.ensemble_model import ensemble_manager
from src.execution.enhanced_risk_manager import enhanced_risk_manager
from src.execution.alpaca_executor import alpaca_executor
from src.strategy.garch_strategy import strategy_manager
from src.data.market_data import market_data_manager
from src.utils.config import config
from src.utils.logger import log_info, log_error


class TradingDashboard:
    """
    Real-time trading dashboard with comprehensive monitoring
    """
    
    def __init__(self, host='localhost', port=8050):
        self.host = host
        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        
        # Data storage for dashboard
        self.performance_data = []
        self.risk_data = []
        self.signal_data = []
        self.portfolio_data = []
        self.model_performance = {}
        
        # Update intervals
        self.update_interval = 5  # seconds
        self.data_retention_hours = 24
        
        # Dashboard state
        self.is_running = False
        self.last_update = None
        
        # Setup dashboard layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        log_info("Trading dashboard initialized")
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("GARCH Intraday Trading Strategy Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Div(id='last-update', style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # in milliseconds
                n_intervals=0
            ),
            
            # Main content tabs
            dcc.Tabs(id='main-tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Performance', value='performance'),
                dcc.Tab(label='Risk Metrics', value='risk'),
                dcc.Tab(label='Signals', value='signals'),
                dcc.Tab(label='Models', value='models'),
                dcc.Tab(label='System Health', value='system')
            ]),
            
            # Tab content
            html.Div(id='tab-content')
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('main-tabs', 'value')
        )
        def render_tab_content(active_tab):
            if active_tab == 'overview':
                return self._render_overview_tab()
            elif active_tab == 'performance':
                return self._render_performance_tab()
            elif active_tab == 'risk':
                return self._render_risk_tab()
            elif active_tab == 'signals':
                return self._render_signals_tab()
            elif active_tab == 'models':
                return self._render_models_tab()
            elif active_tab == 'system':
                return self._render_system_tab()
            
            return html.Div("Select a tab to view content")
        
        @self.app.callback(
            [Output('last-update', 'children'),
             Output('overview-metrics', 'children'),
             Output('portfolio-chart', 'figure'),
             Output('risk-gauge', 'figure'),
             Output('signal-table', 'data')],
            Input('interval-component', 'n_intervals')
        )
        def update_dashboard(n):
            """Update dashboard data"""
            try:
                self._update_data()
                
                # Last update time
                last_update_text = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                # Overview metrics
                overview_metrics = self._get_overview_metrics()
                
                # Portfolio chart
                portfolio_chart = self._create_portfolio_chart()
                
                # Risk gauge
                risk_gauge = self._create_risk_gauge()
                
                # Recent signals
                signal_table_data = self._get_recent_signals()
                
                return (last_update_text, overview_metrics, portfolio_chart, 
                       risk_gauge, signal_table_data)
                
            except Exception as e:
                log_error(f"Error updating dashboard: {e}")
                return ("Update error", html.Div("Error updating data"), {}, {}, [])
    
    def _render_overview_tab(self):
        """Render overview tab content"""
        return html.Div([
            # Key metrics row
            html.Div([
                html.Div([
                    html.H4("Portfolio Overview", style={'textAlign': 'center'}),
                    html.Div(id='overview-metrics')
                ], className='four columns'),
                
                html.Div([
                    html.H4("Portfolio Value", style={'textAlign': 'center'}),
                    dcc.Graph(id='portfolio-chart')
                ], className='four columns'),
                
                html.Div([
                    html.H4("Risk Level", style={'textAlign': 'center'}),
                    dcc.Graph(id='risk-gauge')
                ], className='four columns')
            ], className='row'),
            
            # Recent signals table
            html.Div([
                html.H4("Recent Trading Signals"),
                html.Div([
                    html.Table(id='signal-table')
                ])
            ], style={'marginTop': '30px'}),
            
            # Active positions
            html.Div([
                html.H4("Active Positions"),
                html.Div(id='positions-table')
            ], style={'marginTop': '30px'})
        ])
    
    def _render_performance_tab(self):
        """Render performance analysis tab"""
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Performance Metrics"),
                    dcc.Graph(id='performance-metrics-chart')
                ], className='six columns'),
                
                html.Div([
                    html.H4("Returns Distribution"),
                    dcc.Graph(id='returns-distribution')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.H4("Drawdown Analysis"),
                    dcc.Graph(id='drawdown-chart')
                ], className='six columns'),
                
                html.Div([
                    html.H4("Trading Statistics"),
                    html.Div(id='trading-stats')
                ], className='six columns')
            ], className='row', style={'marginTop': '30px'})
        ])
    
    def _render_risk_tab(self):
        """Render risk metrics tab"""
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Risk Metrics Timeline"),
                    dcc.Graph(id='risk-timeline')
                ], className='six columns'),
                
                html.Div([
                    html.H4("Correlation Matrix"),
                    dcc.Graph(id='correlation-heatmap')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.H4("VaR Analysis"),
                    dcc.Graph(id='var-chart')
                ], className='six columns'),
                
                html.Div([
                    html.H4("Stress Test Results"),
                    html.Div(id='stress-test-table')
                ], className='six columns')
            ], className='row', style={'marginTop': '30px'})
        ])
    
    def _render_signals_tab(self):
        """Render signals analysis tab"""
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Signal Performance"),
                    dcc.Graph(id='signal-performance-chart')
                ], className='six columns'),
                
                html.Div([
                    html.H4("Signal Distribution"),
                    dcc.Graph(id='signal-distribution')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.H4("Detailed Signal History"),
                html.Div(id='signal-history-table')
            ], style={'marginTop': '30px'})
        ])
    
    def _render_models_tab(self):
        """Render model performance tab"""
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Model Performance Comparison"),
                    dcc.Graph(id='model-comparison-chart')
                ], className='six columns'),
                
                html.Div([
                    html.H4("Ensemble Weights"),
                    dcc.Graph(id='ensemble-weights-chart')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.H4("GARCH Model Metrics"),
                    html.Div(id='garch-metrics')
                ], className='four columns'),
                
                html.Div([
                    html.H4("LSTM Model Metrics"),
                    html.Div(id='lstm-metrics')
                ], className='four columns'),
                
                html.Div([
                    html.H4("XGBoost Model Metrics"),
                    html.Div(id='xgboost-metrics')
                ], className='four columns')
            ], className='row', style={'marginTop': '30px'})
        ])
    
    def _render_system_tab(self):
        """Render system health tab"""
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("System Status"),
                    html.Div(id='system-status')
                ], className='six columns'),
                
                html.Div([
                    html.H4("Performance Metrics"),
                    html.Div(id='system-performance')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.H4("Recent Log Entries"),
                html.Div(id='log-entries')
            ], style={'marginTop': '30px'})
        ])
    
    def _update_data(self):
        """Update dashboard data from trading system"""
        try:
            current_time = datetime.now()
            
            # Get portfolio summary
            portfolio_summary = alpaca_executor.get_portfolio_summary()
            
            # Get risk metrics
            if portfolio_summary['positions']:
                risk_metrics = enhanced_risk_manager.calculate_enhanced_risk_metrics(
                    portfolio_summary['account_value'],
                    portfolio_summary['positions']
                )
            else:
                risk_metrics = None
            
            # Get ensemble performance
            ensemble_performance = ensemble_manager.get_ensemble_performance()
            
            # Get strategy performance
            strategy_performance = strategy_manager.get_performance_summary()
            
            # Update performance data
            self.performance_data.append({
                'timestamp': current_time,
                'portfolio_value': portfolio_summary['account_value'],
                'buying_power': portfolio_summary['buying_power'],
                'total_pnl': portfolio_summary['total_unrealized_pnl'],
                'positions_count': portfolio_summary['positions_count']
            })
            
            # Update risk data
            if risk_metrics:
                self.risk_data.append({
                    'timestamp': current_time,
                    'risk_level': risk_metrics.risk_level.value,
                    'daily_pnl': risk_metrics.daily_pnl,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'var_95': risk_metrics.var_95,
                    'portfolio_volatility': risk_metrics.portfolio_volatility,
                    'concentration_ratio': risk_metrics.concentration_ratio,
                    'max_correlation': risk_metrics.correlation_metrics.max_correlation if risk_metrics.correlation_metrics else 0
                })
            
            # Update model performance
            self.model_performance = {
                'ensemble': ensemble_performance,
                'strategy': strategy_performance
            }
            
            # Clean old data
            cutoff_time = current_time - timedelta(hours=self.data_retention_hours)
            self.performance_data = [d for d in self.performance_data if d['timestamp'] > cutoff_time]
            self.risk_data = [d for d in self.risk_data if d['timestamp'] > cutoff_time]
            
            self.last_update = current_time
            
        except Exception as e:
            log_error(f"Error updating dashboard data: {e}")
    
    def _get_overview_metrics(self):
        """Get overview metrics for display"""
        if not self.performance_data:
            return html.Div("No data available")
        
        latest_data = self.performance_data[-1]
        
        return html.Div([
            html.P(f"Portfolio Value: ${latest_data['portfolio_value']:,.2f}"),
            html.P(f"Buying Power: ${latest_data['buying_power']:,.2f}"),
            html.P(f"Total P&L: ${latest_data['total_pnl']:,.2f}"),
            html.P(f"Active Positions: {latest_data['positions_count']}")
        ])
    
    def _create_portfolio_chart(self):
        """Create portfolio value chart"""
        if not self.performance_data:
            return {}
        
        df = pd.DataFrame(self.performance_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2ecc71', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Time",
            yaxis_title="Value ($)",
            template="plotly_white",
            height=300
        )
        
        return fig
    
    def _create_risk_gauge(self):
        """Create risk level gauge"""
        if not self.risk_data:
            risk_value = 0
            risk_level = "LOW"
        else:
            latest_risk = self.risk_data[-1]
            risk_level = latest_risk['risk_level']
            risk_mapping = {'LOW': 25, 'MEDIUM': 50, 'HIGH': 75, 'CRITICAL': 100}
            risk_value = risk_mapping.get(risk_level, 0)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Risk Level: {risk_level}"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def _get_recent_signals(self):
        """Get recent trading signals for table"""
        # This would get actual signal data from the strategy manager
        # For now, return empty list
        return []
    
    def start_dashboard(self, debug=False):
        """Start the dashboard server"""
        try:
            log_info(f"Starting dashboard server on {self.host}:{self.port}")
            self.is_running = True
            
            # Start data update thread
            update_thread = threading.Thread(target=self._periodic_update, daemon=True)
            update_thread.start()
            
            # Run dashboard
            self.app.run_server(host=self.host, port=self.port, debug=debug)
            
        except Exception as e:
            log_error(f"Error starting dashboard: {e}")
            self.is_running = False
    
    def _periodic_update(self):
        """Periodic data update in background thread"""
        while self.is_running:
            try:
                self._update_data()
                time.sleep(self.update_interval)
            except Exception as e:
                log_error(f"Error in periodic update: {e}")
                time.sleep(self.update_interval)
    
    def stop_dashboard(self):
        """Stop the dashboard"""
        self.is_running = False
        log_info("Dashboard stopped")


# Create global dashboard instance
trading_dashboard = TradingDashboard()


def start_monitoring_dashboard(host='localhost', port=8050, debug=False):
    """
    Start the monitoring dashboard
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    trading_dashboard.start_dashboard(debug=debug)


if __name__ == "__main__":
    # Start dashboard if run directly
    start_monitoring_dashboard(debug=True)