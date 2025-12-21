import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from utils import add_journal_entry
from ai_helper import generate_trading_insights

dash.register_page(__name__, path='/', name='Dashboard')

def fetch_stock_data(symbol, period, interval='1d'):
    try:
        ticker_obj = yf.Ticker(symbol)
        company_name = ticker_obj.info.get('longName', symbol)
        if period == '1d_intraday':
            data = ticker_obj.history(period='5d', interval='1m')
            if not data.empty:
                data = data[data.index.normalize() == data.index.normalize().max()]
                if data.empty:
                    data = ticker_obj.history(period='1d', interval='5m')
        else:
            data = ticker_obj.history(period=period, interval=interval)
        if data.empty:
            return None, company_name, f"No data found for {symbol}"
        return data, company_name, None
    except Exception as e:
        return None, symbol, str(e)

def calculate_moving_averages(df):
    if df is None or df.empty: return df
    df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['SMA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    return df

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index"""
    if df is None or df.empty or len(df) < period: return df
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df):
    """Calculate MACD indicator"""
    if df is None or df.empty: return df
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def find_support_resistance(df, window=20):
    """Find support and resistance levels"""
    if df is None or df.empty or len(df) < window: return [], []
    
    # Find local minima (support) and maxima (resistance)
    supports = []
    resistances = []
    
    for i in range(window, len(df) - window):
        # Support: local minimum
        if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window].min():
            supports.append(df['Low'].iloc[i])
        # Resistance: local maximum
        if df['High'].iloc[i] == df['High'].iloc[i-window:i+window].max():
            resistances.append(df['High'].iloc[i])
    
    # Cluster nearby levels
    def cluster_levels(levels, tolerance=0.02):
        if not levels: return []
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level <= current_cluster[-1] * (1 + tolerance):
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        clustered.append(np.mean(current_cluster))
        return clustered
    
    supports = cluster_levels(supports)[-3:]  # Keep top 3
    resistances = cluster_levels(resistances)[-3:]  # Keep top 3
    
    return supports, resistances

def generate_swing_signals(df):
    """Generate swing trading signals using multiple indicators"""
    if df is None or df.empty or len(df) < 50: return df, "HOLD", 0, {}
    
    df['Signal'] = 0
    current = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else current
    
    # Scoring system for signal strength
    buy_score = 0
    sell_score = 0
    reasons = {'buy': [], 'sell': []}
    
    # 1. Moving Average Analysis
    if current['SMA20'] > current['SMA50'] > current['SMA200']:
        buy_score += 2
        reasons['buy'].append("Bullish MA alignment")
    elif current['SMA20'] < current['SMA50'] < current['SMA200']:
        sell_score += 2
        reasons['sell'].append("Bearish MA alignment")
    
    # 2. Golden/Death Cross
    if prev['SMA50'] <= prev['SMA200'] and current['SMA50'] > current['SMA200']:
        buy_score += 3
        reasons['buy'].append("Golden Cross (SMA50 > SMA200)")
        df.loc[df.index[-1], 'Signal'] = 1
    elif prev['SMA50'] >= prev['SMA200'] and current['SMA50'] < current['SMA200']:
        sell_score += 3
        reasons['sell'].append("Death Cross (SMA50 < SMA200)")
        df.loc[df.index[-1], 'Signal'] = -1
    
    # 3. RSI Analysis
    if 'RSI' in df.columns:
        rsi = current['RSI']
        if rsi < 30:
            buy_score += 2
            reasons['buy'].append(f"RSI Oversold ({rsi:.1f})")
        elif rsi > 70:
            sell_score += 2
            reasons['sell'].append(f"RSI Overbought ({rsi:.1f})")
        elif 30 < rsi < 50:
            buy_score += 1
            reasons['buy'].append(f"RSI in buy zone ({rsi:.1f})")
        elif 50 < rsi < 70:
            sell_score += 1
            reasons['sell'].append(f"RSI in sell zone ({rsi:.1f})")
    
    # 4. MACD Analysis
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd = current['MACD']
        macd_signal = current['MACD_Signal']
        prev_macd = prev['MACD']
        prev_signal = prev['MACD_Signal']
        
        if prev_macd <= prev_signal and macd > macd_signal:
            buy_score += 2
            reasons['buy'].append("MACD bullish crossover")
        elif prev_macd >= prev_signal and macd < macd_signal:
            sell_score += 2
            reasons['sell'].append("MACD bearish crossover")
    
    # 5. Price momentum
    price_change_5d = ((current['Close'] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100) if len(df) >= 5 else 0
    if price_change_5d > 5:
        buy_score += 1
        reasons['buy'].append(f"Strong upward momentum ({price_change_5d:.1f}%)")
    elif price_change_5d < -5:
        sell_score += 1
        reasons['sell'].append(f"Strong downward momentum ({price_change_5d:.1f}%)")
    
    # Determine overall signal with improved confidence calculation
    # Max possible score is around 10 (2+3+2+2+1 for each indicator)
    if buy_score > sell_score and buy_score >= 4:
        signal = "STRONG BUY"
        # Strong signals: map 4-10 points to 65-95% confidence
        confidence = min(50 + (buy_score * 8), 95)
    elif buy_score > sell_score and buy_score >= 2:
        signal = "BUY"
        # Regular signals: map 2-3 points to 45-60% confidence
        confidence = min(30 + (buy_score * 15), 65)
    elif sell_score > buy_score and sell_score >= 4:
        signal = "STRONG SELL"
        # Strong signals: map 4-10 points to 65-95% confidence
        confidence = min(50 + (sell_score * 8), 95)
    elif sell_score > buy_score and sell_score >= 2:
        signal = "SELL"
        # Regular signals: map 2-3 points to 45-60% confidence
        confidence = min(30 + (sell_score * 15), 65)
    else:
        signal = "HOLD"
        confidence = 40
    
    # Mark signals on chart
    df['Buy_Signal_Price'] = np.nan
    df['Sell_Signal_Price'] = np.nan
    
    if signal in ["BUY", "STRONG BUY"]:
        df.loc[df.index[-1], 'Buy_Signal_Price'] = current['Low'] * 0.99
    elif signal in ["SELL", "STRONG SELL"]:
        df.loc[df.index[-1], 'Sell_Signal_Price'] = current['High'] * 1.01
    
    return df, signal, confidence, reasons

def calculate_suggested_entry(current_price, signal, supports, resistances, rsi=None):
    """Calculate a suggested entry point based on technical analysis.
    
    Priority for BUY signals:
    1. Check support levels first - suggest nearest support for better entry
    2. Only if very oversold (RSI < 30) AND at/near support, suggest current price
    3. Otherwise suggest waiting for pullback
    
    Priority for SELL signals:
    1. Check resistance levels first - suggest nearest resistance for better exit
    2. Only if very overbought (RSI > 70) AND at/near resistance, suggest current price
    
    Returns: (suggested_price, reason_text)
    """
    if signal in ["BUY", "STRONG BUY"]:
        # For BUY signals, prioritize support levels
        if supports:
            # Find the nearest support below current price
            nearest_support = max([s for s in supports if s < current_price], default=None)
            
            if nearest_support:
                # Calculate distance to support
                distance_to_support = ((current_price - nearest_support) / current_price) * 100
                
                # If we're extremely close to support (within 0.5%) AND RSI is oversold
                if distance_to_support <= 0.5 and rsi and rsi < 30:
                    return current_price, f"BUY near ${current_price:.2f} (at support ${nearest_support:.2f}, RSI {rsi:.1f})"
                # If we're close to support (within 2%), suggest the support level
                elif distance_to_support <= 2.0:
                    if rsi and rsi < 30:
                        return nearest_support, f"BUY near ${nearest_support:.2f} (at support, RSI oversold {rsi:.1f})"
                    else:
                        return nearest_support, f"BUY near ${nearest_support:.2f} (wait for support level)"
                # Otherwise suggest waiting for pullback to support
                else:
                    if rsi and rsi < 25:
                        return nearest_support, f"BUY near ${nearest_support:.2f} (oversold, wait for support)"
                    else:
                        return nearest_support, f"BUY near ${nearest_support:.2f} (wait for pullback to support)"
            else:
                # No support below current price, suggest a pullback target
                # Even if oversold, better to wait for a small dip
                suggested = current_price * 0.98
                if rsi and rsi < 25:
                    return suggested, f"BUY near ${suggested:.2f} (oversold RSI {rsi:.1f}, wait for 2% dip)"
                else:
                    return suggested, f"BUY near ${suggested:.2f} (wait for 2% pullback)"
        else:
            # No support levels identified, always suggest a pullback
            # Never suggest current price even if oversold - better entry discipline
            suggested = current_price * 0.98
            if rsi and rsi < 25:
                return suggested, f"BUY near ${suggested:.2f} (oversold RSI {rsi:.1f}, wait for 2% dip)"
            elif rsi and rsi < 35:
                return suggested, f"BUY near ${suggested:.2f} (RSI {rsi:.1f}, wait for minor dip)"
            else:
                return suggested, f"BUY near ${suggested:.2f} (wait for 2% pullback)"
    
    elif signal in ["SELL", "STRONG SELL"]:
        # For SELL signals (exiting long positions), determine best exit point
        # Priority: Exit at current price OR wait for bounce to resistance for better exit
        
        if resistances:
            # Find the nearest resistance above current price
            nearest_resistance = min([r for r in resistances if r > current_price], default=None)
            
            if nearest_resistance:
                # Calculate distance to resistance
                distance_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
                
                # If very close to resistance (within 1%), exit now
                if distance_to_resistance <= 1.0:
                    if rsi and rsi > 70:
                        return current_price, f"SELL near ${current_price:.2f} (at resistance, RSI {rsi:.1f})"
                    else:
                        return current_price, f"SELL near ${current_price:.2f} (at resistance ${nearest_resistance:.2f})"
                # If resistance is 2-5% above, suggest current price (don't chase)
                elif distance_to_resistance <= 5.0:
                    if rsi and rsi > 70:
                        return current_price, f"SELL near ${current_price:.2f} (exit now, RSI overbought {rsi:.1f})"
                    else:
                        return current_price, f"SELL near ${current_price:.2f} (exit position now)"
                # If resistance is far above (>5%), exit at current or suggest trailing stop
                else:
                    return current_price, f"SELL near ${current_price:.2f} (exit position, use trailing stop)"
            else:
                # No resistance above - exit now
                if rsi and rsi > 70:
                    return current_price, f"SELL near ${current_price:.2f} (exit now, RSI {rsi:.1f})"
                else:
                    return current_price, f"SELL near ${current_price:.2f} (exit position)"
        else:
            # No resistance levels - exit at current price
            if rsi and rsi > 70:
                return current_price, f"SELL near ${current_price:.2f} (exit now, RSI overbought {rsi:.1f})"
            else:
                return current_price, f"SELL near ${current_price:.2f} (exit position)"
    
    else:  # HOLD
        return current_price, f"HOLD - No clear entry point at ${current_price:.2f}"

def calculate_risk_management(current_price, signal, supports, resistances):
    """Calculate stop loss and take profit levels for LONG positions only.
    
    Note: This dashboard is designed for long-only trading. SELL signals indicate
    when to exit existing long positions, not to enter short positions.
    Stop loss is always below entry price.
    """
    risk_reward_ratio = 2.0  # Target 2:1 reward-to-risk
    
    # Always calculate stop loss below entry (long position protection)
    if supports:
        stop_loss = min(supports[-1], current_price * 0.95)
    else:
        stop_loss = current_price * 0.95
    
    risk = current_price - stop_loss
    
    if signal in ["BUY", "STRONG BUY"]:
        # Take profit: above nearest resistance or risk*reward ratio
        if resistances:
            take_profit = max(resistances[0], current_price + (risk * risk_reward_ratio))
        else:
            take_profit = current_price + (risk * risk_reward_ratio)
            
    elif signal in ["SELL", "STRONG SELL"]:
        # For SELL signals, show downside target (where price may go)
        if supports:
            take_profit = min(supports[-1], current_price - (risk * risk_reward_ratio))
        else:
            take_profit = current_price - (risk * risk_reward_ratio)
    else:
        take_profit = current_price * 1.10
    
    return stop_loss, take_profit, risk

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.H5("📊 Stock Analysis Tool", className="mb-0", 
                               style={'color': '#ffffff', 'fontWeight': '600', 'letterSpacing': '0.5px'})
                    ]),
                    style={
                        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        'borderBottom': 'none',
                        'padding': '1rem 1.25rem'
                    }
                ),
                dbc.CardBody([
                    dbc.Label("Stock Symbol:", style={
                        'fontWeight': '600', 
                        'color': '#e0e0e0',
                        'fontSize': '0.875rem',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px',
                        'marginBottom': '0.5rem'
                    }),
                    dbc.Input(
                        id="stock-input", 
                        type="text", 
                        value="AAPL", 
                        placeholder="Enter ticker symbol",
                        style={
                            'backgroundColor': '#1a1a1a', 
                            'color': '#ffffff', 
                            'border': '2px solid #495057',
                            'borderRadius': '8px',
                            'padding': '0.75rem',
                            'fontSize': '1rem',
                            'fontWeight': '500',
                            'transition': 'all 0.3s ease'
                        }
                    ),
                    dbc.Label("Timeframe:", className="mt-3", style={
                        'fontWeight': '600', 
                        'color': '#e0e0e0',
                        'fontSize': '0.875rem',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px',
                        'marginBottom': '0.5rem'
                    }),
                    dcc.Dropdown(
                        id="timeframe-dropdown",
                        options=[
                            {'label': '📅 1 Day (Intraday)', 'value': '1d_intraday'},
                            {'label': '📅 5 Days', 'value': '5d'},
                            {'label': '📅 1 Month', 'value': '1mo'},
                            {'label': '📅 3 Months', 'value': '3mo'},
                            {'label': '📅 6 Months', 'value': '6mo'},
                            {'label': '📅 1 Year', 'value': '1y'},
                            {'label': '📅 2 Years', 'value': '2y'},
                            {'label': '📅 5 Years', 'value': '5y'}
                        ],
                        value='6mo',
                        clearable=False,
                        style={'color': '#000000'}
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-search me-2"), "Analyze Stock"],
                        id="analyze-btn",
                        style={
                            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'border': 'none',
                            'borderRadius': '8px',
                            'padding': '0.75rem 1.5rem',
                            'fontSize': '1rem',
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px',
                            'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.4)',
                            'transition': 'all 0.3s ease'
                        },
                        className="mt-3 w-100", 
                        size="lg"
                    )
                ], style={
                    'backgroundColor': '#2d2d2d',
                    'padding': '1.5rem'
                })
            ], style={
                'border': 'none',
                'borderRadius': '12px',
                'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.4)',
                'overflow': 'hidden'
            })
        ], md=12, lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H5("🎯 Position Size Calculator", className="mb-0", 
                           style={'color': '#ffffff', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
                    style={
                        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        'borderBottom': 'none',
                        'padding': '1rem 1.25rem'
                    }
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Account Size ($):", style={'fontWeight': 'bold', 'color': '#e0e0e0', 'fontSize': '0.9rem'}),
                            dbc.Input(id="account-size", type="number", value=10000, min=100, step=100,
                                    style={'backgroundColor': '#2d2d2d', 'color': '#ffffff', 'border': '1px solid #495057'}),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Risk per Trade (%):", style={'fontWeight': 'bold', 'color': '#e0e0e0', 'fontSize': '0.9rem'}),
                            dbc.Input(id="risk-percent", type="number", value=2, min=0.1, max=10, step=0.1,
                                    style={'backgroundColor': '#2d2d2d', 'color': '#ffffff', 'border': '1px solid #495057'}),
                        ], md=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Entry Price ($):", className="mt-2", style={'fontWeight': 'bold', 'color': '#e0e0e0', 'fontSize': '0.9rem'}),
                            dbc.Input(id="entry-price", type="number", placeholder="Enter price", min=0.01, step=0.01,
                                    style={'backgroundColor': '#2d2d2d', 'color': '#ffffff', 'border': '1px solid #495057'}),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Stop Loss ($):", className="mt-2", style={'fontWeight': 'bold', 'color': '#e0e0e0', 'fontSize': '0.9rem'}),
                            dbc.Input(id="stop-loss-price", type="number", placeholder="Enter stop loss", min=0.01, step=0.01,
                                    style={'backgroundColor': '#2d2d2d', 'color': '#ffffff', 'border': '1px solid #495057'}),
                        ], md=6),
                    ]),
                    html.Hr(style={'borderColor': '#495057', 'margin': '15px 0', 'opacity': '0.3'}),
                    html.Div(id="position-size-output", style={'fontSize': '0.95rem'}),
                    html.Hr(style={'borderColor': '#495057', 'margin': '15px 0', 'opacity': '0.3'}),
                    dbc.Button(
                        [html.I(className="bi bi-journal-plus me-2"), "Add to Journal"],
                        id="add-to-journal-btn",
                        style={
                            'background': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                            'border': 'none',
                            'borderRadius': '8px',
                            'padding': '0.625rem 1.25rem',
                            'fontSize': '0.95rem',
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px',
                            'boxShadow': '0 4px 15px rgba(79, 172, 254, 0.4)',
                            'transition': 'all 0.3s ease'
                        },
                        className="w-100", 
                        size="md"
                    )
                ], style={'backgroundColor': '#2d2d2d', 'padding': '1.5rem'})
            ], style={
                'border': 'none',
                'borderRadius': '12px',
                'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.4)',
                'marginBottom': '20px',
                'overflow': 'hidden'
            })
        ], md=12, lg=5),
        dbc.Col([
            html.Div(id="signal-card", children=[
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("💡 Trading Signal", className="mb-0", 
                               style={'color': '#ffffff', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
                        style={
                            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'borderBottom': 'none',
                            'padding': '0.75rem 1rem'
                        }
                    ),
                    dbc.CardBody([
                        html.H5("Click Analyze", className="text-center", 
                               style={'color': '#adb5bd', 'fontSize': '0.9rem', 'fontWeight': '500'})
                    ], style={'backgroundColor': '#2d2d2d', 'padding': '1.5rem 1rem'})
                ], style={
                    'border': 'none',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.4)',
                    'overflow': 'hidden',
                    'height': '100%'
                })
            ])
        ], md=6, lg=2),
        dbc.Col([
            html.Div(id="entry-point-card", children=[
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("🎯 Entry Point", className="mb-0", 
                               style={'color': '#ffffff', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
                        style={
                            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'borderBottom': 'none',
                            'padding': '0.75rem 1rem'
                        }
                    ),
                    dbc.CardBody([
                        html.H5("Click Analyze", className="text-center", 
                               style={'color': '#adb5bd', 'fontSize': '0.9rem', 'fontWeight': '500'})
                    ], style={'backgroundColor': '#2d2d2d', 'padding': '1.5rem 1rem'})
                ], style={
                    'border': 'none',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.4)',
                    'overflow': 'hidden',
                    'height': '100%'
                })
            ])
        ], md=6, lg=2)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Loading(children=[dcc.Graph(id="stock-chart")], type="default"), width=12)
    ]),
    
    # AI Insights Section
    dbc.Row([
        dbc.Col([
            html.Div(id="ai-insights-container", children=[
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col(
                                html.H6("🤖 AI Trading Insights", className="mb-0", 
                                       style={'color': '#ffffff', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
                                width="auto"
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Generate Insights",
                                    id="generate-ai-insights-btn",
                                    size="sm",
                                    style={
                                        'background': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
                                        'border': 'none',
                                        'borderRadius': '6px',
                                        'padding': '0.4rem 1rem',
                                        'fontSize': '0.85rem',
                                        'fontWeight': '600',
                                        'boxShadow': '0 2px 8px rgba(250, 112, 154, 0.3)'
                                    }
                                ),
                                className="text-end"
                            )
                        ], align="center", justify="between")
                    ], style={
                        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        'borderBottom': 'none',
                        'padding': '1rem 1.25rem'
                    }),
                    dbc.CardBody([
                        dcc.Loading(
                            html.Div(id="ai-insights-output", children=[
                                html.P("Click 'Generate Insights' after analyzing a stock to get AI-powered trading advice, risk assessment, and educational tips.", 
                                      className="text-center text-muted", 
                                      style={'padding': '2rem', 'fontSize': '0.95rem'})
                            ]),
                            type="circle"
                        )
                    ], style={'backgroundColor': '#2d2d2d', 'padding': '1.5rem'})
                ], style={
                    'border': 'none',
                    'borderRadius': '12px',
                    'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.4)',
                    'marginBottom': '20px',
                    'overflow': 'hidden'
                })
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col(html.Div(id="analysis-output", className="mt-3"), width=12)
    ]),
    
    # Journal Entry Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("📔 Add Trade to Journal")),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Date:", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
                    dbc.Input(
                        id="journal-date-input",
                        type="date",
                        value=datetime.now().strftime('%Y-%m-%d'),
                        style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'}
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Action:", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
                    dcc.Dropdown(
                        id="journal-action-dropdown",
                        options=[
                            {'label': '📈 BUY', 'value': 'BUY'},
                            {'label': '📉 SELL', 'value': 'SELL'}
                        ],
                        value='BUY',
                        clearable=False
                    )
                ], md=6)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Ticker:", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
                    dbc.Input(
                        id="journal-ticker-input",
                        type="text",
                        placeholder="Stock symbol",
                        style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'}
                    )
                ], md=6),
                dbc.Col([
                    dbc.Label("Position Size (shares):", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
                    dbc.Input(
                        id="journal-position-size-input",
                        type="number",
                        placeholder="Number of shares",
                        style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'}
                    )
                ], md=6)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Entry Price ($):", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
                    dbc.Input(
                        id="journal-entry-price-input",
                        type="number",
                        placeholder="Entry price",
                        style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'}
                    )
                ], md=4),
                dbc.Col([
                    dbc.Label("Stop Loss ($):", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
                    dbc.Input(
                        id="journal-stop-loss-input",
                        type="number",
                        placeholder="Stop loss",
                        style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'}
                    )
                ], md=4),
                dbc.Col([
                    dbc.Label("Take Profit ($):", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
                    dbc.Input(
                        id="journal-take-profit-input",
                        type="number",
                        placeholder="Take profit",
                        style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'}
                    )
                ], md=4)
            ], className="mb-3"),
            dbc.Label("Notes:", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
            dbc.Textarea(
                id="journal-notes-input",
                placeholder="Trade notes, setup, or reasons...",
                style={'backgroundColor': '#2d2d2d', 'color': '#ffffff', 'minHeight': '100px'}
            ),
            html.Div(id="journal-add-feedback", className="mt-3")
        ]),
        dbc.ModalFooter([
            dbc.Button("💾 Save to Journal", id="journal-save-btn", color="success", className="me-2"),
            dbc.Button("Cancel", id="journal-cancel-btn", color="secondary")
        ])
    ], id="journal-modal", is_open=False, size="lg", style={'color': '#000000'}),
    
    dcc.Store(id="current-ticker-store"),
    dcc.Store(id="analysis-data-store"),  # Store for AI insights data
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(style={
                'borderColor': '#495057', 
                'margin': '40px 0 20px 0',
                'opacity': '0.3'
            }),
            html.P([
                "Made with ",
                html.Span("❤️", style={'color': '#f5576c', 'fontSize': '1.1rem'}),
                " by Aladdin"
            ], 
            className="text-center", 
            style={
                'fontSize': '0.9rem', 
                'paddingBottom': '30px',
                'color': '#9ca3af',
                'fontWeight': '500',
                'letterSpacing': '0.5px'
            })
        ], width=12)
    ])
], fluid=True)

@callback(
    [Output("stock-chart", "figure"),
     Output("analysis-output", "children"),
     Output("signal-card", "children"),
     Output("entry-point-card", "children"),
     Output("entry-price", "value"),
     Output("stop-loss-price", "value"),
     Output("analysis-data-store", "data")],
    Input("analyze-btn", "n_clicks"),
    [State("stock-input", "value"),
     State("timeframe-dropdown", "value")],
    prevent_initial_call=True
)
def update_chart(n_clicks, symbol, timeframe):
    if not n_clicks or not symbol:
        return go.Figure(), "", dbc.Card([
            dbc.CardBody(html.H5("Enter a symbol and click Analyze"))
        ]), dbc.Card([
            dbc.CardBody(html.H5("Enter a symbol and click Analyze"))
        ]), 0, 0, None
    
    interval = {'1d_intraday': '5m', '5d': '15m', '1mo': '1h', '3mo': '1d'}.get(timeframe, '1d')
    df, company_name, error = fetch_stock_data(symbol, timeframe, interval)
    
    if error:
        return go.Figure(), dbc.Alert(error, color="danger"), dbc.Card([
            dbc.CardBody(html.H5("Error loading data", className="text-danger"))
        ]), dbc.Card([
            dbc.CardBody(html.H5("Error loading data", className="text-danger"))
        ]), 0, 0
    
    # Calculate all indicators
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    
    # Find support and resistance
    supports, resistances = find_support_resistance(df)
    
    # Generate swing trading signals
    df, signal, confidence, reasons = generate_swing_signals(df)
    
    # Get current metrics
    try:
        info = yf.Ticker(symbol).info
        rating = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
        price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[0]
        change = ((price - prev_price) / prev_price) * 100
        day_high = df['High'].max()
        day_low = df['Low'].min()
        avg_volume = df['Volume'].mean()
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None
        macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else None
    except:
        rating, price, change, day_high, day_low, avg_volume, rsi, macd = "N/A", 0, 0, 0, 0, 0, None, None
    
    # Calculate risk management
    stop_loss, take_profit, risk_amount = calculate_risk_management(price, signal, supports, resistances)
    
    # Create enhanced subplots with RSI and MACD
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(
            f'{symbol.upper()} - {company_name}',
            'Volume',
            'RSI (14)',
            'MACD'
        )
    )
    
    # === ROW 1: MAIN PRICE CHART ===
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350'
    ), row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA20'], name='SMA 20',
        line=dict(color='#00BCD4', width=1.5), opacity=0.7
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA50'], name='SMA 50',
        line=dict(color='#2196F3', width=2), opacity=0.8
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA200'], name='SMA 200',
        line=dict(color='#FF6F00', width=2), opacity=0.8
    ), row=1, col=1)
    
    # Support levels
    for support in supports:
        fig.add_hline(
            y=support, line_dash="dash", line_color="#4CAF50",
            annotation_text=f"Support: ${support:.2f}",
            annotation_position="right",
            opacity=0.5, row=1, col=1
        )
    
    # Resistance levels
    for resistance in resistances:
        fig.add_hline(
            y=resistance, line_dash="dash", line_color="#F44336",
            annotation_text=f"Resistance: ${resistance:.2f}",
            annotation_position="right",
            opacity=0.5, row=1, col=1
        )
    
    # Buy/Sell signals
    if 'Buy_Signal_Price' in df.columns and df['Buy_Signal_Price'].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Buy_Signal_Price'], mode='markers',
            marker=dict(symbol='triangle-up', size=20, color='#00C853',
                       line=dict(color='white', width=2)),
            name='Buy Signal',
            hovertemplate='<b>BUY</b><br>Price: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
    
    if 'Sell_Signal_Price' in df.columns and df['Sell_Signal_Price'].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Sell_Signal_Price'], mode='markers',
            marker=dict(symbol='triangle-down', size=20, color='#D32F2F',
                       line=dict(color='white', width=2)),
            name='Sell Signal',
            hovertemplate='<b>SELL</b><br>Price: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
    
    # === ROW 2: VOLUME ===
    colors = ['#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef5350'
              for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume',
        marker_color=colors, opacity=0.7, showlegend=False
    ), row=2, col=1)
    
    # === ROW 3: RSI ===
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], name='RSI',
            line=dict(color='#9C27B0', width=2), showlegend=False
        ), row=3, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # === ROW 4: MACD ===
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'], name='MACD',
            line=dict(color='#2196F3', width=2), showlegend=False
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'], name='Signal',
            line=dict(color='#FF9800', width=2), showlegend=False
        ), row=4, col=1)
        
        # MACD histogram
        colors_macd = ['#26a69a' if val >= 0 else '#ef5350' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(
            x=df.index, y=df['MACD_Hist'], name='Histogram',
            marker_color=colors_macd, opacity=0.5, showlegend=False
        ), row=4, col=1)
    
    # Layout styling
    fig.update_layout(
        template='plotly_dark',
        height=1000,
        hovermode='x unified',
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#2d2d2d',
        font=dict(family='Arial, sans-serif', size=11, color='#e0e0e0'),
        xaxis4=dict(title='Date', rangeslider=dict(visible=False), showgrid=True, gridcolor='#404040'),
        yaxis=dict(title='Price (USD)', showgrid=True, gridcolor='#404040'),
        yaxis2=dict(title='Volume', showgrid=True, gridcolor='#404040'),
        yaxis3=dict(title='RSI', range=[0, 100], showgrid=True, gridcolor='#404040'),
        yaxis4=dict(title='MACD', showgrid=True, gridcolor='#404040'),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0.5)', bordercolor='#404040', borderwidth=1
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    # === SIGNAL CARD ===
    signal_color_map = {
        "STRONG BUY": "success",
        "BUY": "info",
        "HOLD": "warning",
        "SELL": "danger",
        "STRONG SELL": "danger"
    }
    signal_icon_map = {
        "STRONG BUY": "🚀",
        "BUY": "📈",
        "HOLD": "⏸️",
        "SELL": "📉",
        "STRONG SELL": "⚠️"
    }
    
    signal_card = dbc.Card([
        dbc.CardHeader(
            html.H6("💡 Trading Signal", className="mb-0", 
                   style={'color': '#ffffff', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
            style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'borderBottom': 'none',
                'padding': '0.75rem 1rem'
            }
        ),
        dbc.CardBody([
            html.H2([
                signal_icon_map.get(signal, ""), " ", signal
            ], className=f"text-{signal_color_map.get(signal, 'secondary')} mb-2 text-center",
               style={'fontWeight': '700', 'letterSpacing': '1px'}),
            html.P(f"Confidence: {confidence}%", className="text-muted mb-0 text-center",
                  style={'fontSize': '0.9rem', 'fontWeight': '500'})
        ], style={'backgroundColor': '#2d2d2d', 'padding': '1.5rem 1rem'})
    ], style={
        'border': 'none',
        'borderRadius': '12px',
        'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.4)',
        'overflow': 'hidden'
    }, className="mb-3")
    
    # Calculate suggested entry point based on technical analysis
    rsi_value = df['RSI'].iloc[-1] if 'RSI' in df.columns and not df['RSI'].isna().iloc[-1] else None
    suggested_entry, entry_reason = calculate_suggested_entry(price, signal, supports, resistances, rsi_value)
    
    # Determine if this is a BUY or SELL action
    action_type = "BUY" if signal in ["BUY", "STRONG BUY"] else "SELL" if signal in ["SELL", "STRONG SELL"] else "HOLD"
    action_color = "success" if action_type == "BUY" else "danger" if action_type == "SELL" else "warning"
    action_icon = "📈" if action_type == "BUY" else "📉" if action_type == "SELL" else "⏸️"
    
    # Build entry point card content based on signal type
    if action_type == "SELL":
        # For SELL signals (exits), show only exit price and downside target
        entry_card_body = [
            # Suggested Entry (inline format)
            html.Div([
                html.P([
                    html.Strong("Suggested Exit:", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                    html.Br(),
                    html.Span(entry_reason, 
                             className=f"text-{action_color}",
                             style={'fontSize': '0.95rem', 'fontWeight': '600'})
                ], className="mb-3 text-center", style={'lineHeight': '1.6'})
            ]),
            
            html.Hr(style={'borderColor': '#495057', 'margin': '10px 0'}),
            
            # Current Price for reference
            html.Div([
                html.Small("Exit Price", className="d-block text-muted text-center", 
                         style={'fontWeight': '600', 'fontSize': '0.7rem'}),
                html.P(f"${price:.2f}", className="text-info mb-3 text-center", 
                       style={'fontWeight': 'bold', 'fontSize': '1.1rem'})
            ]),
            
            # Downside Target (instead of take profit for sells)
            html.Div([
                html.Small("Downside Target", className="d-block text-muted text-center",
                         style={'fontWeight': '600', 'fontSize': '0.7rem'}),
                html.P(f"${take_profit:.2f}", className="text-warning mb-1 text-center",
                       style={'fontWeight': 'bold', 'fontSize': '0.95rem'}),
                html.Small(f"{((take_profit-price)/price*100):.1f}%", 
                         className="text-warning d-block text-center")
            ])
        ]
    else:
        # For BUY/HOLD signals, show full entry setup
        entry_card_body = [
            # Suggested Entry (inline format)
            html.Div([
                html.P([
                    html.Strong("Suggested Entry:", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                    html.Br(),
                    html.Span(entry_reason, 
                             className=f"text-{action_color}",
                             style={'fontSize': '0.95rem', 'fontWeight': '600'})
                ], className="mb-3 text-center", style={'lineHeight': '1.6'})
            ]),
            
            html.Hr(style={'borderColor': '#495057', 'margin': '10px 0'}),
            
            # Current Price for reference
            html.Div([
                html.Small("Current Price", className="d-block text-muted text-center", 
                         style={'fontWeight': '600', 'fontSize': '0.7rem'}),
                html.P(f"${price:.2f}", className="text-info mb-2 text-center", 
                       style={'fontWeight': 'bold', 'fontSize': '1.1rem'})
            ]),
            
            # Stop Loss
            html.Div([
                html.Small("Stop Loss", className="d-block text-muted text-center",
                         style={'fontWeight': '600', 'fontSize': '0.7rem'}),
                html.P(f"${stop_loss:.2f}", className="text-danger mb-1 text-center",
                       style={'fontWeight': 'bold', 'fontSize': '0.95rem'}),
                html.Small(f"{((stop_loss-price)/price*100):.1f}%", 
                         className="text-muted d-block text-center mb-2")
            ]),
            
            # Take Profit
            html.Div([
                html.Small("Take Profit", className="d-block text-muted text-center",
                         style={'fontWeight': '600', 'fontSize': '0.7rem'}),
                html.P(f"${take_profit:.2f}", className="text-success mb-1 text-center",
                       style={'fontWeight': 'bold', 'fontSize': '0.95rem'}),
                html.Small(f"+{((take_profit-price)/price*100):.1f}%", 
                         className="text-success d-block text-center")
            ])
        ]
    
    # Set card header based on signal type
    card_title = "🎯 Recommended Exit" if action_type == "SELL" else "🎯 Recommended Entry"
    
    entry_point_card = dbc.Card([
        dbc.CardHeader(
            html.H6(card_title, className="mb-0", 
                   style={'color': '#ffffff', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
            style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'borderBottom': 'none',
                'padding': '0.75rem 1rem'
            }
        ),
        dbc.CardBody(entry_card_body, style={'backgroundColor': '#2d2d2d', 'padding': '1.5rem 1rem'})
    ], style={
        'border': 'none',
        'borderRadius': '12px',
        'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.4)',
        'overflow': 'hidden'
    })
    
    # === ANALYSIS DASHBOARD ===
    color_class = "success" if change >= 0 else "danger"
    change_icon = "📈" if change >= 0 else "📉"
    
    # RSI interpretation
    rsi_color = "success" if rsi and rsi < 40 else "danger" if rsi and rsi > 60 else "warning"
    rsi_text = "Oversold" if rsi and rsi < 30 else "Overbought" if rsi and rsi > 70 else "Neutral"
    
    analysis = dbc.Card([
        dbc.CardHeader(
            html.H5("📊 Technical Analysis Dashboard", className="mb-0",
                   style={'color': '#ffffff', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
            style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'borderBottom': 'none',
                'padding': '1rem 1.25rem'
            }
        ),
        dbc.CardBody([
            # Top metrics row
            dbc.Row([
                dbc.Col([
                    html.Small("Current Price", className="d-block",
                             style={'color': '#adb5bd', 'fontWeight': '600'}),
                    html.H3(f"${price:.2f}", className=f"text-{color_class} mb-0",
                           style={'fontWeight': 'bold'})
                ], width=2),
                dbc.Col([
                    html.Small("Change", className="d-block",
                             style={'color': '#adb5bd', 'fontWeight': '600'}),
                    html.H3(f"{change_icon} {change:+.2f}%", className=f"text-{color_class} mb-0",
                           style={'fontWeight': 'bold'})
                ], width=2),
                dbc.Col([
                    html.Small("RSI (14)", className="d-block",
                             style={'color': '#adb5bd', 'fontWeight': '600'}),
                    html.H4(f"{rsi:.1f}" if rsi else "N/A", className=f"text-{rsi_color} mb-0",
                           style={'fontWeight': 'bold'}),
                    html.Small(rsi_text, className=f"text-{rsi_color}")
                ], width=2),
                dbc.Col([
                    html.Small("Analyst Rating", className="d-block",
                             style={'color': '#adb5bd', 'fontWeight': '600'}),
                    html.H5(rating, className="mb-0",
                           style={'color': '#26a69a' if rating in ['Strong Buy', 'Buy'] else
                                  '#ef5350' if rating in ['Sell', 'Strong Sell'] else '#FFA726',
                                  'fontWeight': 'bold'})
                ], width=3),
                dbc.Col([
                    html.Small("Timeframe", className="d-block",
                             style={'color': '#adb5bd', 'fontWeight': '600'}),
                    html.H5(timeframe.upper(), className="mb-0",
                           style={'color': '#ffffff', 'fontWeight': 'bold'})
                ], width=3)
            ], className="mb-3"),
            html.Hr(style={'borderColor': '#495057'}),
            
            # Second row - Price metrics
            dbc.Row([
                dbc.Col([
                    html.Small("Period High", className="d-block",
                             style={'color': '#adb5bd', 'fontWeight': '600'}),
                    html.P(f"${day_high:.2f}", className="mb-0",
                          style={'color': '#ffffff', 'fontSize': '1.1rem', 'fontWeight': '500'})
                ], width=2),
                dbc.Col([
                    html.Small("Period Low", className="d-block",
                             style={'color': '#adb5bd', 'fontWeight': '600'}),
                    html.P(f"${day_low:.2f}", className="mb-0",
                          style={'color': '#ffffff', 'fontSize': '1.1rem', 'fontWeight': '500'})
                ], width=2),
                dbc.Col([
                    html.Small("Avg Volume", className="d-block",
                             style={'color': '#adb5bd', 'fontWeight': '600'}),
                    html.P(f"{avg_volume:,.0f}", className="mb-0",
                          style={'color': '#ffffff', 'fontSize': '1.1rem', 'fontWeight': '500'})
                ], width=5),
                dbc.Col([
                    html.Small("Risk Amount", className="d-block",
                             style={'color': '#adb5bd', 'fontWeight': '600'}),
                    html.P(f"${risk_amount:.2f} ({(risk_amount/price*100):.1f}%)", className="mb-0",
                          style={'color': '#ffffff', 'fontSize': '1.1rem', 'fontWeight': '500'})
                ], width=3)
            ], className="mb-3"),
            
            # Signal reasons
            html.Hr(style={'borderColor': '#495057'}),
            dbc.Row([
                dbc.Col([
                    html.H6("📋 Signal Analysis:", style={'color': '#ffffff'}),
                    html.Ul([
                        html.Li(reason, style={'color': '#26a69a'})
                        for reason in reasons.get('buy', [])
                    ] + [
                        html.Li(reason, style={'color': '#ef5350'})
                        for reason in reasons.get('sell', [])
                    ]) if (reasons.get('buy') or reasons.get('sell')) else html.P("No strong signals detected", className="text-muted")
                ], width=12)
            ])
        ], style={'backgroundColor': '#2d2d2d', 'padding': '1.5rem'})
    ], className="mt-3", style={
        'border': 'none',
        'borderRadius': '12px',
        'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.4)',
        'overflow': 'hidden'
    })
    
    # Return current price as entry and calculated stop loss
    entry = round(price, 2) if price else 0
    stop = round(stop_loss, 2) if stop_loss else 0
    
    # Prepare data for AI insights
    analysis_data = {
        'ticker': symbol.upper(),
        'price': float(price),
        'signal': signal,
        'confidence': confidence,
        'rsi': float(rsi) if rsi and not pd.isna(rsi) else None,
        'macd': float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else None,
        'macd_signal': float(df['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in df.columns and not pd.isna(df['MACD_Signal'].iloc[-1]) else None,
        'sma20': float(df['SMA20'].iloc[-1]) if 'SMA20' in df.columns and not pd.isna(df['SMA20'].iloc[-1]) else None,
        'sma50': float(df['SMA50'].iloc[-1]) if 'SMA50' in df.columns and not pd.isna(df['SMA50'].iloc[-1]) else None,
        'sma200': float(df['SMA200'].iloc[-1]) if 'SMA200' in df.columns and not pd.isna(df['SMA200'].iloc[-1]) else None,
        'support_levels': [float(s) for s in supports] if supports else [],
        'resistance_levels': [float(r) for r in resistances] if resistances else [],
        'entry_price': float(entry),
        'stop_loss': float(stop),
        'take_profit': float(take_profit) if take_profit else None
    }
    
    return fig, analysis, signal_card, entry_point_card, entry, stop, analysis_data


@callback(
    Output("ai-insights-output", "children"),
    Input("generate-ai-insights-btn", "n_clicks"),
    State("analysis-data-store", "data"),
    prevent_initial_call=True
)
def generate_ai_insights_callback(n_clicks, analysis_data):
    """Generate AI-powered trading insights"""
    if not n_clicks or not analysis_data:
        return html.P("Please analyze a stock first before generating insights.", 
                     className="text-warning text-center", style={'padding': '2rem'})
    
    try:
        # Generate insights using OpenAI
        insights_text = generate_trading_insights(
            ticker=analysis_data['ticker'],
            price=analysis_data['price'],
            signal=analysis_data['signal'],
            confidence=analysis_data['confidence'],
            technical_data=analysis_data
        )
        
        # Parse and format the insights
        sections = insights_text.split('\n\n')
        formatted_sections = []
        
        for section in sections:
            if section.strip():
                # Check if section has a header (contains **)
                if '**' in section:
                    parts = section.split('**')
                    if len(parts) >= 3:
                        header = parts[1]
                        content = parts[2].strip()
                        
                        # Determine icon and color based on header
                        if 'Signal' in header or 'Analysis' in header:
                            icon = '📊'
                            color = '#4facfe'
                        elif 'Key Factors' in header or 'Factors' in header:
                            icon = '🔑'
                            color = '#fa709a'
                        elif 'Risk' in header:
                            icon = '⚠️'
                            color = '#ffc107'
                        elif 'Action' in header:
                            icon = '🎯'
                            color = '#26a69a'
                        elif 'Learning' in header or 'Tip' in header:
                            icon = '💡'
                            color = '#667eea'
                        else:
                            icon = '📝'
                            color = '#ffffff'
                        
                        formatted_sections.append(
                            html.Div([
                                html.H6([
                                    html.Span(icon, style={'marginRight': '8px'}),
                                    header
                                ], style={'color': color, 'fontWeight': '600', 'marginBottom': '8px'}),
                                dcc.Markdown(content, style={'color': '#e0e0e0', 'lineHeight': '1.6'})
                            ], style={'marginBottom': '20px'})
                        )
                else:
                    formatted_sections.append(
                        html.P(section, style={'color': '#e0e0e0', 'lineHeight': '1.6', 'marginBottom': '15px'})
                    )
        
        return html.Div(formatted_sections)
        
    except Exception as e:
        return dbc.Alert([
            html.H6("⚠️ Error Generating Insights", className="mb-2"),
            html.P(f"Unable to generate AI insights: {str(e)}", className="mb-0 small")
        ], color="warning")


@callback(
    Output("position-size-output", "children"),
    [Input("account-size", "value"),
     Input("risk-percent", "value"),
     Input("entry-price", "value"),
     Input("stop-loss-price", "value")]
)
def calculate_position_size(account_size, risk_percent, entry_price, stop_loss):
    """Calculate position size based on risk management parameters"""
    
    # Check if any value is None (not entered)
    if account_size is None or risk_percent is None or entry_price is None or stop_loss is None:
        return html.Div([
            html.P("⚠️ Enter all values to calculate position size", 
                   style={'color': '#FFA726', 'textAlign': 'center', 'marginBottom': '0'})
        ])
    
    # Validate positive values for entry and stop loss
    if entry_price <= 0 or stop_loss <= 0:
        return html.Div([
            html.P("⚠️ Entry price and stop loss must be greater than 0", 
                   style={'color': '#FFA726', 'textAlign': 'center', 'marginBottom': '0'})
        ])
    
    # Validate account size and risk percent
    if account_size <= 0 or risk_percent <= 0:
        return html.Div([
            html.P("⚠️ Account size and risk percent must be greater than 0", 
                   style={'color': '#FFA726', 'textAlign': 'center', 'marginBottom': '0'})
        ])
    
    # Validate that stop loss is below entry for long positions
    if stop_loss >= entry_price:
        return html.Div([
            html.P("⚠️ Stop loss must be below entry price for long positions", 
                   style={'color': '#ef5350', 'textAlign': 'center', 'marginBottom': '0'})
        ])
    
    # Calculate risk amount in dollars
    risk_amount = account_size * (risk_percent / 100)
    
    # Calculate risk per share
    risk_per_share = entry_price - stop_loss
    
    # Calculate number of shares based on risk
    shares_by_risk = int(risk_amount / risk_per_share)
    
    # Calculate max shares the account can afford
    max_affordable_shares = int(account_size / entry_price)
    
    # Use the lower of the two to ensure position doesn't exceed account size
    shares = min(shares_by_risk, max_affordable_shares)
    
    # Flag if position was limited by account size
    limited_by_account = shares_by_risk > max_affordable_shares
    
    # Calculate position value
    position_value = shares * entry_price
    
    # Calculate actual risk with adjusted shares
    actual_risk = shares * risk_per_share
    
    # Calculate position as percentage of account
    position_percent = (position_value / account_size) * 100
    
    # Determine if position is too large (more than 25% of account)
    alert_color = '#ef5350' if position_percent > 25 else '#26a69a'
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small("SHARES TO BUY", style={'color': '#adb5bd', 'fontSize': '0.75rem'}),
                    html.H4(f"{shares:,}", style={'color': '#26a69a', 'fontWeight': 'bold', 'marginBottom': '0'})
                ], style={'textAlign': 'center'})
            ], width=4),
            dbc.Col([
                html.Div([
                    html.Small("POSITION VALUE", style={'color': '#adb5bd', 'fontSize': '0.75rem'}),
                    html.H4(f"${position_value:,.2f}", style={'color': alert_color, 'fontWeight': 'bold', 'marginBottom': '0'})
                ], style={'textAlign': 'center'})
            ], width=4),
            dbc.Col([
                html.Div([
                    html.Small("ACTUAL RISK", style={'color': '#adb5bd', 'fontSize': '0.75rem'}),
                    html.H4(f"${actual_risk:,.2f}", style={'color': '#FFA726', 'fontWeight': 'bold', 'marginBottom': '0'})
                ], style={'textAlign': 'center'})
            ], width=4),
        ]),
        html.Hr(style={'borderColor': '#495057', 'margin': '10px 0'}),
        dbc.Row([
            dbc.Col([
                html.Small(f"Risk per Share: ${risk_per_share:.2f}", 
                          style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
            ], width=6),
            dbc.Col([
                html.Small(f"Position Size: {position_percent:.1f}% of account", 
                          style={'color': alert_color, 'fontSize': '0.85rem'}),
            ], width=6),
        ]),
        html.Div([
            html.Small(f"ℹ️ Position limited by account size (max {max_affordable_shares} shares affordable)", 
                      style={'color': '#64b5f6', 'fontSize': '0.85rem', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'marginTop': '5px'}) if limited_by_account else html.Div(),
        html.Div([
            html.Small("⚠️ Warning: Position exceeds 25% of account", 
                      style={'color': '#ef5350', 'fontSize': '0.85rem', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'marginTop': '5px'}) if position_percent > 25 else html.Div()
    ])

# Journal Callbacks
@callback(
    [Output("journal-modal", "is_open"),
     Output("journal-ticker-input", "value"),
     Output("journal-entry-price-input", "value"),
     Output("journal-stop-loss-input", "value"),
     Output("journal-take-profit-input", "value")],
    [Input("add-to-journal-btn", "n_clicks"),
     Input("journal-save-btn", "n_clicks"),
     Input("journal-cancel-btn", "n_clicks")],
    [State("journal-modal", "is_open"),
     State("stock-input", "value"),
     State("entry-price", "value"),
     State("stop-loss-price", "value"),
     State("signal-card", "children")],
    prevent_initial_call=True
)
def toggle_journal_modal(add_click, save_click, cancel_click, is_open, ticker, entry, stop, signal_card):
    """Toggle journal modal and pre-fill values from position calculator."""
    if add_click:
        # Try to extract take profit from signal card
        take_profit_value = None
        if signal_card and isinstance(signal_card, dict):
            try:
                # Extract take profit from the signal card structure
                card_body = signal_card.get('props', {}).get('children', [])
                if isinstance(card_body, list):
                    for item in card_body:
                        if isinstance(item, dict):
                            props = item.get('props', {})
                            children = props.get('children', [])
                            if isinstance(children, list):
                                for child in children:
                                    if isinstance(child, dict) and 'Take Profit' in str(child):
                                        # Found take profit, extract value
                                        text = str(child.get('props', {}).get('children', ''))
                                        if '$' in text:
                                            try:
                                                take_profit_value = float(text.split('$')[1].split()[0])
                                            except:
                                                pass
            except:
                pass
        return True, ticker or "", entry or None, stop or None, take_profit_value
    elif save_click or cancel_click:
        return False, no_update, no_update, no_update, no_update
    return is_open, no_update, no_update, no_update, no_update

@callback(
    [Output("journal-add-feedback", "children"),
     Output("journal-modal", "is_open", allow_duplicate=True)],
    Input("journal-save-btn", "n_clicks"),
    [State("journal-date-input", "value"),
     State("journal-ticker-input", "value"),
     State("journal-action-dropdown", "value"),
     State("journal-entry-price-input", "value"),
     State("journal-stop-loss-input", "value"),
     State("journal-take-profit-input", "value"),
     State("journal-position-size-input", "value"),
     State("journal-notes-input", "value")],
    prevent_initial_call=True
)
def save_journal_entry(n_clicks, date, ticker, action, entry_price, stop_loss, 
                       take_profit, position_size, notes):
    """Save journal entry to database."""
    if not n_clicks:
        return no_update, no_update
    
    # Validate required fields
    if not ticker:
        return dbc.Alert("⚠️ Please enter a ticker symbol", color="warning"), True
    
    # Add entry to database
    success, message = add_journal_entry(
        date=date or datetime.now().strftime('%Y-%m-%d'),
        ticker=ticker,
        action=action,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size,
        notes=notes or ""
    )
    
    if success:
        return dbc.Alert("✅ " + message, color="success"), False
    else:
        return dbc.Alert("❌ " + message, color="danger"), True
