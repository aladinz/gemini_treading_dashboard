import dash
from dash import dcc, html, Input, Output, State, ALL, ctx, callback, no_update
import dash_bootstrap_components as dbc
import yfinance as yf
from utils import add_stock_to_db, remove_stock_from_db, get_watchlist_from_db, get_stock_info_for_watchlist_display

dash.register_page(__name__, path='/watchlist', name='Watchlist')

def build_watchlist_display():
    watchlist = get_watchlist_from_db()
    if not watchlist:
        return dbc.Alert("Your watchlist is empty. Add some tickers!", color="info", 
                        style={'backgroundColor': '#0d6efd20', 'color': '#e0e0e0', 'border': '1px solid #0d6efd'})
    
    rows = []
    for item in watchlist:
        ticker = item['ticker']
        company, rating = get_stock_info_for_watchlist_display(ticker)
        
        # Color code rating
        rating_color = '#26a69a' if rating in ['Strong Buy', 'Buy'] else \
                      '#ef5350' if rating in ['Sell', 'Strong Sell'] else '#FFA726'
        
        rows.append(html.Tr([
            html.Td(ticker, style={'color': '#00BCD4', 'fontWeight': 'bold'}),
            html.Td(company, style={'color': '#e0e0e0'}),
            html.Td(rating or "N/A", style={'color': rating_color, 'fontWeight': '600'}),
            html.Td(dbc.Button("🗑️ Remove", id={"type": "remove-btn", "index": ticker},
                                color="danger", size="sm", outline=True))
        ], style={'backgroundColor': '#2d2d2d'}))
    
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Ticker", style={'color': '#ffffff', 'backgroundColor': '#212529'}), 
            html.Th("Company", style={'color': '#ffffff', 'backgroundColor': '#212529'}), 
            html.Th("Rating", style={'color': '#ffffff', 'backgroundColor': '#212529'}), 
            html.Th("Action", style={'color': '#ffffff', 'backgroundColor': '#212529'})
        ])),
        html.Tbody(rows)
    ], bordered=True, dark=True, hover=True, style={'border': '1px solid #495057'})

layout = dbc.Container([
    dbc.Card([
        dbc.CardHeader(html.H3("📋 My Watchlist", className="mb-0", style={'color': '#ffffff'}),
                      style={'backgroundColor': '#212529'}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Add Stock to Watchlist:", style={'fontWeight': 'bold', 'color': '#e0e0e0', 'fontSize': '1.1rem'}),
                    dbc.InputGroup([
                        dbc.Input(id="add-input", placeholder="Enter ticker symbol (e.g., MSFT, TSLA)",
                                style={'backgroundColor': '#2d2d2d', 'color': '#ffffff', 'border': '1px solid #495057'}),
                        dbc.Button("➕ Add Stock", id="add-btn", color="success", size="lg")
                    ], className="mt-2"),
                    html.Div(id="feedback", className="mt-3")
                ], md=12)
            ]),
            html.Hr(style={'borderColor': '#495057', 'marginTop': '2rem', 'marginBottom': '2rem'}),
            dbc.Row([
                dbc.Col(dcc.Loading(children=html.Div(id="watchlist-display")))
            ])
        ], style={'backgroundColor': '#2d2d2d'})
    ], style={'border': '1px solid #495057'})
], fluid=True)

@callback(
    [Output("feedback", "children"),
     Output("add-input", "value"),
     Output("watchlist-display", "children")],
    [Input("add-btn", "n_clicks"),
     Input({"type": "remove-btn", "index": ALL}, "n_clicks"),
     Input("url", "pathname")],
    [State("add-input", "value"),
     State("url", "pathname")],
    prevent_initial_call=True
)
def handle_watchlist(add_clicks, remove_clicks, url_path, add_value, current_path):
    triggered = ctx.triggered_id
    
    if triggered == 'url' and current_path == '/watchlist':
        return no_update, no_update, build_watchlist_display()
    
    if triggered == "add-btn" and current_path == '/watchlist':
        if not add_value:
            return dbc.Alert("Enter a ticker", color="warning"), "", no_update
        
        ticker = add_value.upper()
        try:
            info = yf.Ticker(ticker).info
            company = info.get('longName', info.get('shortName', ticker))
            if company:
                success, msg = add_stock_to_db(ticker, company)
                return dbc.Alert(msg, color="success" if success else "danger"), \
                       "" if success else ticker, build_watchlist_display()
        except Exception as e:
            return dbc.Alert(f"Error: {e}", color="danger"), ticker, no_update
    
    if isinstance(triggered, dict) and triggered.get('type') == 'remove-btn':
        ticker = triggered.get('index')
        if ticker:
            success, msg = remove_stock_from_db(ticker)
            return dbc.Alert(msg, color="success" if success else "danger"), \
                   no_update, build_watchlist_display()
    
    return no_update, no_update, no_update
