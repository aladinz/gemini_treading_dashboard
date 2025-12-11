# 📈 Gemini Trading Dashboard

A professional-grade swing trading dashboard built with Python Dash, featuring comprehensive technical analysis, multi-indicator signals, and risk management tools.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/dash-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

### 📊 Technical Analysis Dashboard
- **Real-time Stock Data**: Powered by yfinance for accurate market data
- **Interactive Candlestick Charts**: Professional dark-themed 4-panel layout
- **Multiple Moving Averages**: SMA (20/50/200) and EMA (12/26)
- **RSI Indicator**: 14-period Relative Strength Index with overbought/oversold zones
- **MACD Analysis**: Full MACD histogram with signal line crossovers
- **Support & Resistance**: Automatic detection using price clustering algorithms
- **Volume Analysis**: Color-coded volume bars for trend confirmation

### 🎯 Trading Signals
- **Multi-Factor Scoring System**: Combines 6+ indicators for confidence-based signals
  - Moving Average Alignment
  - RSI Zones (Oversold/Overbought)
  - MACD Crossovers
  - Momentum Analysis
  - Golden/Death Cross Detection
- **Signal Confidence**: 0-95% confidence ratings
- **Clear Recommendations**: BUY, SELL, or HOLD with detailed reasoning

### 💼 Risk Management
- **Automatic Stop Loss Calculation**: Based on support/resistance levels
- **Take Profit Targets**: 2:1 reward-to-risk ratio
- **Position Sizing**: Risk amount calculations
- **Entry Price Optimization**: Support/resistance-based entry points

### 📋 Watchlist Management
- **Persistent Storage**: SQLite database for watchlist tracking
- **Quick Add/Remove**: Easy stock management interface
- **Analyst Ratings Integration**: Real-time analyst recommendations
- **Color-Coded Display**: Visual indicators for Buy/Sell/Hold ratings

### 🎨 Modern UI/UX
- **Dark Theme**: Professional DARKLY Bootstrap theme
- **Responsive Layout**: Optimized for all screen sizes
- **Interactive Components**: Smooth user experience with Dash components
- **Clean Navigation**: Multi-page architecture

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/aladinz/gemini_treading_dashboard.git
cd gemini_treading_dashboard
```

2. **Create a virtual environment**
```bash
python -m venv venv
```

3. **Activate the virtual environment**

Windows:
```bash
.\venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage

1. **Start the application**
```bash
python app.py
```

2. **Open your browser**
Navigate to `http://127.0.0.1:8050`

3. **Analyze stocks**
   - Enter a stock symbol (e.g., AAPL, MSFT, TSLA)
   - Select timeframe (1mo, 3mo, 6mo, 1y, 2y, 5y)
   - Choose interval (1d, 1wk, 1mo)
   - Click "Update Chart" to view analysis

4. **Manage watchlist**
   - Navigate to "Watchlist" page
   - Add stocks to track
   - View analyst ratings
   - Remove stocks as needed

## 🏗️ Project Structure

```
gemini_treading_dashboard/
│
├── app.py                      # Main application entry point
├── utils.py                    # Database operations & utilities
├── requirements.txt            # Python dependencies
│
├── pages/
│   ├── __init__.py            # Package initialization
│   ├── dashboard_page.py      # Main trading analysis dashboard
│   └── watchlist_page.py      # Watchlist management page
│
├── assets/
│   └── custom.css             # Custom styling
│
└── watchlist.db               # SQLite database (auto-created)
```

## 🔧 Technical Stack

- **Framework**: Dash 2.x with Plotly
- **UI Components**: dash-bootstrap-components (DARKLY theme)
- **Data Source**: yfinance API
- **Data Processing**: pandas, numpy
- **Database**: SQLite3
- **Charts**: Plotly.js with dark theme
- **Architecture**: Multi-page Dash application

## 📊 Technical Indicators Explained

### RSI (Relative Strength Index)
- 14-period calculation
- Oversold: < 30
- Overbought: > 70

### MACD (Moving Average Convergence Divergence)
- Fast EMA: 12 periods
- Slow EMA: 26 periods
- Signal Line: 9 periods

### Moving Averages
- SMA 20: Short-term trend
- SMA 50: Medium-term trend
- SMA 200: Long-term trend
- EMA 12/26: MACD components

### Support & Resistance
- Clustering algorithm for level detection
- Minimum 2 touches per level
- Recent price data weighting

## 🎓 Usage Tips

1. **Swing Trading**: Best suited for 1-week to 1-month holding periods
2. **Signal Interpretation**: Use confidence scores above 60% for higher probability trades
3. **Risk Management**: Always respect stop-loss levels
4. **Multi-Timeframe Analysis**: Check multiple timeframes for confirmation
5. **Volume Confirmation**: Look for volume spikes on breakouts

## 📝 Dependencies

```
dash>=2.0.0
dash-bootstrap-components>=1.0.0
plotly>=5.0.0
pandas>=1.3.0
numpy>=1.21.0
yfinance>=0.2.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It should not be considered financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Dash by Plotly](https://dash.plotly.com/) - Web application framework
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance market data
- [Bootstrap](https://getbootstrap.com/) - UI components and theming

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ for swing traders**
