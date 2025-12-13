import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

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
    
    # Determine overall signal
    if buy_score > sell_score and buy_score >= 4:
        signal = "STRONG BUY"
        confidence = min(buy_score * 10, 95)
    elif buy_score > sell_score and buy_score >= 2:
        signal = "BUY"
        confidence = min(buy_score * 10, 85)
    elif sell_score > buy_score and sell_score >= 4:
        signal = "STRONG SELL"
        confidence = min(sell_score * 10, 95)
    elif sell_score > buy_score and sell_score >= 2:
        signal = "SELL"
        confidence = min(sell_score * 10, 85)
    else:
        signal = "HOLD"
        confidence = 50
    
    # Mark signals on chart
    df['Buy_Signal_Price'] = np.nan
    df['Sell_Signal_Price'] = np.nan
    
    if signal in ["BUY", "STRONG BUY"]:
        df.loc[df.index[-1], 'Buy_Signal_Price'] = current['Low'] * 0.99
    elif signal in ["SELL", "STRONG SELL"]:
        df.loc[df.index[-1], 'Sell_Signal_Price'] = current['High'] * 1.01
    
    return df, signal, confidence, reasons

def calculate_risk_management(current_price, signal, supports, resistances):
    """Calculate stop loss and take profit levels"""
    risk_reward_ratio = 2.0  # Target 2:1 reward-to-risk
    
    if signal in ["BUY", "STRONG BUY"]:
        # Stop loss: below nearest support or 5% below entry
        if supports:
            stop_loss = min(supports[-1], current_price * 0.95)
        else:
            stop_loss = current_price * 0.95
        
        risk = current_price - stop_loss
        
        # Take profit: above nearest resistance or risk*reward ratio
        if resistances:
            take_profit = max(resistances[0], current_price + (risk * risk_reward_ratio))
        else:
            take_profit = current_price + (risk * risk_reward_ratio)
            
    elif signal in ["SELL", "STRONG SELL"]:
        # Stop loss: above nearest resistance or 5% above entry
        if resistances:
            stop_loss = max(resistances[0], current_price * 1.05)
        else:
            stop_loss = current_price * 1.05
        
        risk = stop_loss - current_price
        
        # Take profit: below nearest support or risk*reward ratio
        if supports:
            take_profit = min(supports[-1], current_price - (risk * risk_reward_ratio))
        else:
            take_profit = current_price - (risk * risk_reward_ratio)
    else:
        stop_loss = current_price * 0.95
        take_profit = current_price * 1.10
        risk = current_price - stop_loss
    
    return stop_loss, take_profit, risk

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📊 Stock Analysis Tool", className="mb-0", style={'color': '#ffffff'}), 
                             style={'backgroundColor': '#212529'}),
                dbc.CardBody([
                    dbc.Label("Stock Symbol:", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
                    dbc.Input(id="stock-input", type="text", value="AAPL", placeholder="Enter ticker symbol",
                            style={'backgroundColor': '#2d2d2d', 'color': '#ffffff', 'border': '1px solid #495057'}),
                    dbc.Label("Timeframe:", className="mt-3", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
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
                    dbc.Button("🔍 Analyze Stock", id="analyze-btn", color="success", 
                             className="mt-3 w-100", size="lg")
                ], style={'backgroundColor': '#2d2d2d'})
            ], style={'border': '1px solid #495057'})
        ], md=12, lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎯 Position Size Calculator", className="mb-0", style={'color': '#ffffff'}), 
                             style={'backgroundColor': '#212529'}),
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
                            dbc.Input(id="entry-price", type="number", value=0, min=0, step=0.01,
                                    style={'backgroundColor': '#2d2d2d', 'color': '#ffffff', 'border': '1px solid #495057'}),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Stop Loss ($):", className="mt-2", style={'fontWeight': 'bold', 'color': '#e0e0e0', 'fontSize': '0.9rem'}),
                            dbc.Input(id="stop-loss-price", type="number", value=0, min=0, step=0.01,
                                    style={'backgroundColor': '#2d2d2d', 'color': '#ffffff', 'border': '1px solid #495057'}),
                        ], md=6),
                    ]),
                    html.Hr(style={'borderColor': '#495057', 'margin': '15px 0'}),
                    html.Div(id="position-size-output", style={'fontSize': '0.95rem'})
                ], style={'backgroundColor': '#2d2d2d', 'padding': '15px'})
            ], style={'border': '1px solid #495057', 'marginBottom': '20px'})
        ], md=12, lg=5),
        dbc.Col([
            html.Div(id="signal-card", children=[
                dbc.Card([
                    dbc.CardHeader(html.H6("💡 Trading Signal", className="mb-0", style={'color': '#ffffff'}),
                                 style={'backgroundColor': '#212529'}),
                    dbc.CardBody([
                        html.H5("Click Analyze to see signals", className="text-center", style={'color': '#adb5bd'})
                    ], style={'backgroundColor': '#2d2d2d'})
                ], style={'border': '1px solid #495057'})
            ])
        ], md=12, lg=4)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Loading(children=[dcc.Graph(id="stock-chart")], type="default"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="analysis-output", className="mt-3"), width=12)
    ])
], fluid=True)

@callback(
    [Output("stock-chart", "figure"),
     Output("analysis-output", "children"),
     Output("signal-card", "children"),
     Output("entry-price", "value"),
     Output("stop-loss-price", "value")],
    Input("analyze-btn", "n_clicks"),
    [State("stock-input", "value"),
     State("timeframe-dropdown", "value")],
    prevent_initial_call=True
)
def update_chart(n_clicks, symbol, timeframe):
    if not n_clicks or not symbol:
        return go.Figure(), "", dbc.Card([
            dbc.CardBody(html.H5("Enter a symbol and click Analyze"))
        ]), 0, 0
    
    interval = {'1d_intraday': '5m', '5d': '15m', '1mo': '1h', '3mo': '1d'}.get(timeframe, '1d')
    df, company_name, error = fetch_stock_data(symbol, timeframe, interval)
    
    if error:
        return go.Figure(), dbc.Alert(error, color="danger"), dbc.Card([
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
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H2([
                        signal_icon_map.get(signal, ""), " ", signal
                    ], className=f"text-{signal_color_map.get(signal, 'secondary')} mb-0"),
                    html.P(f"Confidence: {confidence}%", className="text-muted mb-0")
                ], width=4),
                dbc.Col([
                    html.Small("Stop Loss", className="d-block text-muted"),
                    html.H5(f"${stop_loss:.2f}", className="text-danger mb-0"),
                    html.Small(f"{((stop_loss-price)/price*100):.1f}%", className="text-muted")
                ], width=4),
                dbc.Col([
                    html.Small("Take Profit", className="d-block text-muted"),
                    html.H5(f"${take_profit:.2f}", className="text-success mb-0"),
                    html.Small(f"+{((take_profit-price)/price*100):.1f}%", className="text-muted")
                ], width=4)
            ])
        ])
    ], color=signal_color_map.get(signal, "secondary"), outline=True, className="mb-3")
    
    # === ANALYSIS DASHBOARD ===
    color_class = "success" if change >= 0 else "danger"
    change_icon = "📈" if change >= 0 else "📉"
    
    # RSI interpretation
    rsi_color = "success" if rsi and rsi < 40 else "danger" if rsi and rsi > 60 else "warning"
    rsi_text = "Oversold" if rsi and rsi < 30 else "Overbought" if rsi and rsi > 70 else "Neutral"
    
    analysis = dbc.Card([
        dbc.CardHeader(
            html.H5("📊 Technical Analysis Dashboard", className="mb-0",
                   style={'color': '#ffffff', 'fontWeight': 'bold'}),
            style={'backgroundColor': '#212529'}
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
        ], style={'backgroundColor': '#2d2d2d'})
    ], className="mt-3", style={'border': '1px solid #495057'})
    
    # Return current price as entry and calculated stop loss
    entry = round(price, 2) if price else 0
    stop = round(stop_loss, 2) if stop_loss else 0
    
    return fig, analysis, signal_card, entry, stop


@callback(
    Output("position-size-output", "children"),
    [Input("account-size", "value"),
     Input("risk-percent", "value"),
     Input("entry-price", "value"),
     Input("stop-loss-price", "value")]
)
def calculate_position_size(account_size, risk_percent, entry_price, stop_loss):
    """Calculate position size based on risk management parameters"""
    
    if not all([account_size, risk_percent, entry_price, stop_loss]) or entry_price <= 0 or stop_loss <= 0:
        return html.Div([
            html.P("⚠️ Enter all values to calculate position size", 
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
    
    # Calculate number of shares
    shares = int(risk_amount / risk_per_share)
    
    # Calculate position value
    position_value = shares * entry_price
    
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
                    html.Small("RISK AMOUNT", style={'color': '#adb5bd', 'fontSize': '0.75rem'}),
                    html.H4(f"${risk_amount:,.2f}", style={'color': '#FFA726', 'fontWeight': 'bold', 'marginBottom': '0'})
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
            html.Small("⚠️ Warning: Position exceeds 25% of account", 
                      style={'color': '#ef5350', 'fontSize': '0.85rem', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'marginTop': '5px'}) if position_percent > 25 else html.Div()
    ])
