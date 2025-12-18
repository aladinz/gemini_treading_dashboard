# Gemini Trading Dashboard - Copilot Instructions

## Architecture Overview

This is a **multi-page Dash application** for swing trading analysis with a dark-themed UI.

```
app.py              → Entry point, sets up Dash app with DARKLY theme and navbar
utils.py            → SQLite database ops (watchlist CRUD) + yfinance helpers
pages/
  dashboard_page.py → Main analysis: technical indicators, charts, signals, risk calc
  watchlist_page.py → Watchlist management with pattern-matching callbacks
assets/custom.css   → Rating color classes
```

## Key Patterns

### Page Registration (Dash Pages)
Each page must register itself at module level:
```python
dash.register_page(__name__, path='/watchlist', name='Watchlist')
```
Pages export a `layout` variable (not a function).

### Callback Pattern
Use `@callback` decorator (not `@app.callback`) with imports from `dash`:
```python
from dash import callback, Input, Output, State, ctx, no_update, ALL
```
- Use `ctx.triggered_id` for multi-trigger callbacks
- Use pattern-matching callbacks with `ALL` for dynamic components (see watchlist remove buttons)
- Always set `prevent_initial_call=True` for user-triggered actions

### Database Operations (utils.py)
- SQLite database: `watchlist.db` (auto-created on import)
- Functions return `(bool, message)` tuples for success/error handling
- DB init runs once via `WERKZEUG_RUN_MAIN` check to avoid duplicate init in debug mode

### Technical Analysis Pipeline (dashboard_page.py)
Data flow: `fetch_stock_data()` → `calculate_moving_averages()` → `calculate_rsi()` → `calculate_macd()` → `find_support_resistance()` → `generate_swing_signals()` → `calculate_risk_management()`

Indicators:
- SMAs: 20/50/200 periods
- EMAs: 12/26 (for MACD)
- RSI: 14 periods, zones at 30/70
- MACD: 12/26/9 standard

### Chart Construction
Uses Plotly `make_subplots` with 4 rows (heights: 0.5, 0.2, 0.15, 0.15):
1. Candlestick + MAs + support/resistance
2. Volume bars (color-coded by direction)
3. RSI with zones
4. MACD with histogram

### Signal Scoring System
`generate_swing_signals()` uses a point-based system:
- MA alignment: ±2 points
- Golden/Death cross: ±3 points
- RSI zones: ±1-2 points
- MACD crossover: ±2 points
- Momentum: ±1 point

Thresholds: ≥4 points = STRONG, ≥2 points = regular signal

## Styling Conventions

- Theme: Dash Bootstrap Components with `dbc.themes.DARKLY`
- Background colors: `#1a1a1a` (container), `#2d2d2d` (cards), `#212529` (headers)
- Border: `1px solid #495057`
- Bullish/Bearish colors: `#26a69a` / `#ef5350`
- All inputs styled with dark background: `{'backgroundColor': '#2d2d2d', 'color': '#ffffff'}`

## Development Workflow

```bash
# Activate venv (Windows)
.\venv\Scripts\activate

# Run with hot reload
python app.py
# Opens at http://127.0.0.1:8050

# For production-like run
python run_with_errors.py
```

## Dependencies

Core: `dash`, `dash-bootstrap-components`, `plotly`, `pandas`, `numpy`, `yfinance`

yfinance API notes:
- Use `ticker_obj.info` for metadata (can be slow/fail for delisted tickers)
- Use `ticker_obj.history()` for OHLCV data
- Always wrap in try/except - API can be unreliable

## Adding New Features

**New page**: Create `pages/new_page.py`, register with `dash.register_page()`, export `layout`

**New indicator**: Add calculation function to `dashboard_page.py`, integrate into signal scoring, add trace to chart subplot

**New watchlist field**: Update `utils.py` schema, modify CRUD functions, update display in `watchlist_page.py`
