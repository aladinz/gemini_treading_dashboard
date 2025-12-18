import dash
from dash import html, dcc, callback, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
from utils import get_journal_entries, delete_journal_entry, update_journal_entry
from datetime import datetime

dash.register_page(__name__, path='/journal', name='Trading Journal')

layout = dbc.Container([
    html.H2("📔 Trading Journal", className="text-center mb-4", style={'color': '#ffffff'}),
    
    # Stats cards
    html.Div(id="journal-stats", className="mb-4"),
    
    # Advanced stats row
    html.Div(id="journal-advanced-stats", className="mb-4"),
    
    # Filter and sort controls
    dbc.Card([
        dbc.CardHeader(html.H6("🔍 Filters & Controls", className="mb-0", style={'color': '#ffffff'}),
                      style={'backgroundColor': '#212529'}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Filter by Status:", style={'color': '#e0e0e0', 'fontSize': '0.9rem'}),
                    dcc.Dropdown(
                        id="filter-status",
                        options=[
                            {'label': '📊 All Trades', 'value': 'all'},
                            {'label': '⏳ Open Positions', 'value': 'open'},
                            {'label': '✅ Wins', 'value': 'win'},
                            {'label': '❌ Losses', 'value': 'loss'},
                            {'label': '⚖️ Breakeven', 'value': 'breakeven'}
                        ],
                        value='all',
                        clearable=False,
                        className='dark-dropdown'
                    )
                ], md=3),
                dbc.Col([
                    dbc.Label("Filter by Action:", style={'color': '#e0e0e0', 'fontSize': '0.9rem'}),
                    dcc.Dropdown(
                        id="filter-action",
                        options=[
                            {'label': '📈 All Actions', 'value': 'all'},
                            {'label': '🟢 BUY Only', 'value': 'BUY'},
                            {'label': '🔴 SELL Only', 'value': 'SELL'}
                        ],
                        value='all',
                        clearable=False,
                        className='dark-dropdown'
                    )
                ], md=3),
                dbc.Col([
                    dbc.Label("Sort by:", style={'color': '#e0e0e0', 'fontSize': '0.9rem'}),
                    dcc.Dropdown(
                        id="sort-by",
                        options=[
                            {'label': '📅 Date (Newest)', 'value': 'date_desc'},
                            {'label': '📅 Date (Oldest)', 'value': 'date_asc'},
                            {'label': '💰 P&L (Highest)', 'value': 'pnl_desc'},
                            {'label': '💰 P&L (Lowest)', 'value': 'pnl_asc'},
                            {'label': '🔤 Ticker (A-Z)', 'value': 'ticker_asc'}
                        ],
                        value='date_desc',
                        clearable=False,
                        className='dark-dropdown'
                    )
                ], md=3),
                dbc.Col([
                    dbc.Label("Search Ticker:", style={'color': '#e0e0e0', 'fontSize': '0.9rem'}),
                    dbc.Input(
                        id="search-ticker",
                        type="text",
                        placeholder="e.g. AAPL",
                        style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'}
                    )
                ], md=3)
            ])
        ], style={'backgroundColor': '#2d2d2d'})
    ], className="mb-4", style={'border': '1px solid #495057'}),
    
    # Journal entries
    dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.H5("📝 Journal Entries", className="mb-0", style={'color': '#ffffff'}), width="auto"),
                dbc.Col(html.Div(id="entries-count-badge"), width="auto")
            ], justify="between", align="center"),
            style={'backgroundColor': '#212529'}
        ),
        dbc.CardBody([
            html.Div(id="journal-entries-container")
        ], style={'backgroundColor': '#2d2d2d'})
    ], style={'border': '1px solid #495057'}),
    
    # Update result modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Update Trade Result")),
        dbc.ModalBody([
            dbc.Label("Exit Date:", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
            dbc.Input(
                id="update-exit-date-input",
                type="date",
                value=datetime.now().strftime('%Y-%m-%d'),
                style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'},
                className="mb-3"
            ),
            dbc.Label("Exit Price ($):", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
            dbc.Input(
                id="update-exit-price-input",
                type="number",
                placeholder="Enter exit price",
                step="0.01",
                style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'},
                className="mb-3"
            ),
            dbc.Label("Result:", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
            dcc.Dropdown(
                id="update-result-dropdown",
                options=[
                    {'label': '✅ Win', 'value': 'Win'},
                    {'label': '❌ Loss', 'value': 'Loss'},
                    {'label': '⚖️ Breakeven', 'value': 'Breakeven'}
                ],
                className="mb-3 dark-dropdown",
                disabled=True
            ),
            dbc.Label("Profit/Loss ($):", style={'fontWeight': 'bold', 'color': '#e0e0e0'}),
            dbc.Input(
                id="update-profit-loss-input",
                type="number",
                placeholder="Auto-calculated",
                style={'backgroundColor': '#2d2d2d', 'color': '#ffffff'},
                disabled=True
            )
        ]),
        dbc.ModalFooter([
            dbc.Button("Update", id="update-result-confirm-btn", color="success", className="me-2"),
            dbc.Button("Cancel", id="update-result-cancel-btn", color="secondary")
        ])
    ], id="update-result-modal", is_open=False, style={'color': '#000000'}),
    
    dcc.Store(id="update-entry-id-store")
    
], fluid=True, style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'padding': '2rem'})

@callback(
    [Output("journal-stats", "children"),
     Output("journal-advanced-stats", "children"),
     Output("journal-entries-container", "children"),
     Output("entries-count-badge", "children")],
    [Input("journal-entries-container", "id"),
     Input("filter-status", "value"),
     Input("filter-action", "value"),
     Input("sort-by", "value"),
     Input("search-ticker", "value")]
)
def display_journal(_, filter_status, filter_action, sort_by, search_ticker):
    """Display journal stats and entries with filters and sorting."""
    entries = get_journal_entries()
    
    if not entries:
        stats = html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("0", className="text-center mb-0"),
                            html.P("Total Trades", className="text-center mb-0 text-muted")
                        ])
                    ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
                ], md=3)
            ])
        ])
        
        entries_display = html.Div([
            html.P("No journal entries yet. Add your first trade from the Dashboard!", 
                  className="text-center text-muted")
        ])
        
        count_badge = dbc.Badge("0 trades", color="secondary", className="ms-2")
        
        return stats, html.Div(), entries_display, count_badge
    
    # Apply filters
    filtered_entries = entries.copy()
    
    # Filter by status
    if filter_status != 'all':
        if filter_status == 'open':
            filtered_entries = [e for e in filtered_entries if not e['result']]
        elif filter_status == 'win':
            filtered_entries = [e for e in filtered_entries if e['result'] == 'Win']
        elif filter_status == 'loss':
            filtered_entries = [e for e in filtered_entries if e['result'] == 'Loss']
        elif filter_status == 'breakeven':
            filtered_entries = [e for e in filtered_entries if e['result'] == 'Breakeven']
    
    # Filter by action
    if filter_action != 'all':
        filtered_entries = [e for e in filtered_entries if e['action'] == filter_action]
    
    # Filter by ticker search
    if search_ticker:
        search_ticker = search_ticker.upper().strip()
        filtered_entries = [e for e in filtered_entries if search_ticker in e['ticker'].upper()]
    
    # Apply sorting
    if sort_by == 'date_desc':
        filtered_entries.sort(key=lambda x: x['date'], reverse=True)
    elif sort_by == 'date_asc':
        filtered_entries.sort(key=lambda x: x['date'])
    elif sort_by == 'pnl_desc':
        filtered_entries.sort(key=lambda x: x['profit_loss'] if x['profit_loss'] else 0, reverse=True)
    elif sort_by == 'pnl_asc':
        filtered_entries.sort(key=lambda x: x['profit_loss'] if x['profit_loss'] else 0)
    elif sort_by == 'ticker_asc':
        filtered_entries.sort(key=lambda x: x['ticker'])
    
    # Count badge
    count_text = f"{len(filtered_entries)} trade{'s' if len(filtered_entries) != 1 else ''}"
    if len(filtered_entries) != len(entries):
        count_text += f" (of {len(entries)})"
    count_badge = dbc.Badge(count_text, color="info", className="ms-2")
    
    # Calculate stats (using ALL entries for overall stats)
    total_trades = len(entries)
    completed_trades = [e for e in entries if e['result']]
    wins = len([e for e in completed_trades if e['result'] == 'Win'])
    losses = len([e for e in completed_trades if e['result'] == 'Loss'])
    breakevens = len([e for e in completed_trades if e['result'] == 'Breakeven'])
    win_rate = (wins / len(completed_trades) * 100) if completed_trades else 0
    total_pnl = sum([e['profit_loss'] for e in completed_trades if e['profit_loss']])
    
    # Advanced stats
    winning_trades = [e for e in completed_trades if e['result'] == 'Win' and e['profit_loss']]
    losing_trades = [e for e in completed_trades if e['result'] == 'Loss' and e['profit_loss']]
    
    avg_win = sum([e['profit_loss'] for e in winning_trades]) / len(winning_trades) if winning_trades else 0
    avg_loss = sum([e['profit_loss'] for e in losing_trades]) / len(losing_trades) if losing_trades else 0
    largest_win = max([e['profit_loss'] for e in winning_trades]) if winning_trades else 0
    largest_loss = min([e['profit_loss'] for e in losing_trades]) if losing_trades else 0
    profit_factor = abs(sum([e['profit_loss'] for e in winning_trades]) / sum([e['profit_loss'] for e in losing_trades])) if losing_trades and sum([e['profit_loss'] for e in losing_trades]) != 0 else 0
    
    stats = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(str(total_trades), className="text-center mb-0", style={'color': '#ffffff', 'fontSize': '1.8rem'}),
                        html.P("Total Trades", className="text-center mb-0", style={'color': '#adb5bd', 'fontSize': '0.85rem'})
                    ], style={'padding': '1rem'})
                ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
            ], md=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{win_rate:.1f}%", className="text-center mb-0", 
                               style={'color': '#26a69a' if win_rate >= 50 else '#ef5350', 'fontSize': '1.8rem'}),
                        html.P("Win Rate", className="text-center mb-0", style={'color': '#adb5bd', 'fontSize': '0.85rem'})
                    ], style={'padding': '1rem'})
                ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
            ], md=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{wins}W/{losses}L/{breakevens}BE", className="text-center mb-0", 
                               style={'color': '#ffffff', 'fontSize': '1.5rem'}),
                        html.P("W / L / BE", className="text-center mb-0", style={'color': '#adb5bd', 'fontSize': '0.85rem'})
                    ], style={'padding': '1rem'})
                ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
            ], md=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"${total_pnl:.2f}", className="text-center mb-0", 
                               style={'color': '#26a69a' if total_pnl >= 0 else '#ef5350', 'fontSize': '1.8rem', 'fontWeight': 'bold'}),
                        html.P("Total P&L", className="text-center mb-0", style={'color': '#adb5bd', 'fontSize': '0.85rem'})
                    ], style={'padding': '1rem'})
                ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{len([e for e in entries if not e['result']])}", className="text-center mb-0", 
                               style={'color': '#ffc107', 'fontSize': '1.8rem'}),
                        html.P("Open Positions", className="text-center mb-0", style={'color': '#adb5bd', 'fontSize': '0.85rem'})
                    ], style={'padding': '1rem'})
                ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
            ], md=3)
        ])
    ])
    
    # Advanced stats
    advanced_stats = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("💰 Avg Win: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(f"${avg_win:.2f}", style={'color': '#26a69a', 'fontWeight': 'bold'})
                        ], className="mb-1"),
                        html.Div([
                            html.Span("💸 Avg Loss: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(f"${avg_loss:.2f}", style={'color': '#ef5350', 'fontWeight': 'bold'})
                        ])
                    ], style={'padding': '0.75rem'})
                ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("🏆 Best Win: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(f"${largest_win:.2f}", style={'color': '#26a69a', 'fontWeight': 'bold'})
                        ], className="mb-1"),
                        html.Div([
                            html.Span("📉 Worst Loss: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(f"${largest_loss:.2f}", style={'color': '#ef5350', 'fontWeight': 'bold'})
                        ])
                    ], style={'padding': '0.75rem'})
                ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("📊 Profit Factor: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(f"{profit_factor:.2f}", 
                                     style={'color': '#26a69a' if profit_factor >= 2 else '#ffc107' if profit_factor >= 1 else '#ef5350', 
                                            'fontWeight': 'bold'})
                        ], className="mb-1"),
                        html.Div([
                            html.Span("🎯 Risk/Reward: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(f"{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A", 
                                     style={'color': '#ffffff', 'fontWeight': 'bold'})
                        ])
                    ], style={'padding': '0.75rem'})
                ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("📈 Total Volume: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(f"{sum([e['position_size'] for e in entries if e['position_size']])}", 
                                     style={'color': '#ffffff', 'fontWeight': 'bold'})
                        ], className="mb-1"),
                        html.Div([
                            html.Span("🔄 Avg Size: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(f"{sum([e['position_size'] for e in entries if e['position_size']]) / len([e for e in entries if e['position_size']]):.0f}" if [e for e in entries if e['position_size']] else "N/A", 
                                     style={'color': '#ffffff', 'fontWeight': 'bold'})
                        ])
                    ], style={'padding': '0.75rem'})
                ], style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
            ], md=3)
        ])
    ])
    
    # Create entry cards (use filtered entries)
    entry_cards = []
    for entry in filtered_entries:
        # Format values
        entry_price = f"${entry['entry_price']:.2f}" if entry['entry_price'] else "N/A"
        stop_loss = f"${entry['stop_loss']:.2f}" if entry['stop_loss'] else "N/A"
        take_profit = f"${entry['take_profit']:.2f}" if entry['take_profit'] else "N/A"
        position_size = f"{entry['position_size']} shares" if entry['position_size'] else "N/A"
        
        # Calculate days held and risk/reward
        days_held = None
        risk_reward = None
        if entry['exit_date'] and entry['date']:
            try:
                entry_date = datetime.strptime(entry['date'], '%Y-%m-%d')
                exit_date = datetime.strptime(entry['exit_date'], '%Y-%m-%d')
                days_held = (exit_date - entry_date).days
            except:
                days_held = None
        
        # Calculate R:R ratio
        if entry['entry_price'] and entry['stop_loss'] and entry['take_profit']:
            risk = abs(entry['entry_price'] - entry['stop_loss'])
            reward = abs(entry['take_profit'] - entry['entry_price'])
            if risk > 0:
                risk_reward = reward / risk
        
        # Result badge
        if entry['result']:
            if entry['result'] == 'Win':
                result_badge = dbc.Badge("✅ Win", color="success", className="me-2")
            elif entry['result'] == 'Loss':
                result_badge = dbc.Badge("❌ Loss", color="danger", className="me-2")
            else:
                result_badge = dbc.Badge("⚖️ Breakeven", color="warning", className="me-2")
            
            pnl_text = f"${entry['profit_loss']:.2f}" if entry['profit_loss'] else "$0.00"
            pnl_color = '#26a69a' if entry['profit_loss'] and entry['profit_loss'] >= 0 else '#ef5350'
        else:
            result_badge = dbc.Badge("⏳ Open", color="info", className="me-2")
            pnl_text = ""
            pnl_color = '#ffffff'
        
        # Action color
        action_color = '#26a69a' if entry['action'] == 'BUY' else '#ef5350'
        
        card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5([
                            html.Span(entry['ticker'], style={'color': '#ffffff', 'fontWeight': 'bold'}),
                            html.Span(f" • {entry['action']}", 
                                    style={'color': action_color, 'fontSize': '1rem', 'marginLeft': '10px'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Entry: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(entry['date'], style={'color': '#e0e0e0', 'fontSize': '0.85rem'}),
                        ], className="mb-1"),
                        html.Div([
                            html.Span("Exit: ", style={'color': '#adb5bd', 'fontSize': '0.85rem'}),
                            html.Span(entry['exit_date'] if entry['exit_date'] else "—", 
                                    style={'color': '#e0e0e0', 'fontSize': '0.85rem'}),
                        ], className="mb-1") if entry['exit_date'] else None,
                        html.Div([
                            dbc.Badge(f"📅 {days_held} days", color="secondary", className="mt-1")
                        ]) if days_held is not None else None,
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.Span("Entry: ", style={'color': '#adb5bd', 'fontWeight': 'bold', 'fontSize': '0.85rem'}),
                            html.Span(entry_price, style={'color': '#ffffff', 'fontSize': '0.9rem'}),
                        ], className="mb-1"),
                        html.Div([
                            html.Span("Stop: ", style={'color': '#adb5bd', 'fontWeight': 'bold', 'fontSize': '0.85rem'}),
                            html.Span(stop_loss, style={'color': '#ef5350', 'fontSize': '0.9rem'}),
                        ], className="mb-1"),
                        html.Div([
                            html.Span("Target: ", style={'color': '#adb5bd', 'fontWeight': 'bold', 'fontSize': '0.85rem'}),
                            html.Span(take_profit, style={'color': '#26a69a', 'fontSize': '0.9rem'}),
                        ], className="mb-1"),
                        html.Div([
                            dbc.Badge(f"R:R 1:{risk_reward:.2f}", color="secondary", pill=True, className="mt-1")
                        ]) if risk_reward else None,
                    ], md=3),
                    dbc.Col([
                        html.Div([
                            html.Span("Size: ", style={'color': '#adb5bd', 'fontWeight': 'bold', 'fontSize': '0.85rem'}),
                            html.Span(position_size, style={'color': '#ffffff', 'fontSize': '0.9rem'}),
                        ], className="mb-2"),
                        html.Div([
                            result_badge,
                            html.Span(pnl_text, style={'color': pnl_color, 'fontWeight': 'bold', 'fontSize': '1.1rem'}) 
                            if pnl_text else None
                        ]),
                        html.Div([
                            html.Span(f"Return: {(entry['profit_loss'] / (entry['entry_price'] * entry['position_size']) * 100):.2f}%" if entry.get('profit_loss') and entry.get('entry_price') and entry.get('position_size') else "", 
                                     style={'color': '#adb5bd', 'fontSize': '0.75rem'})
                        ], className="mt-1") if entry.get('profit_loss') and entry.get('entry_price') and entry.get('position_size') else None
                    ], md=3),
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button("📊 Update", id={'type': 'update-entry-btn', 'index': entry['id']},
                                      color="primary", size="sm", className="me-2",
                                      disabled=bool(entry['result'])),
                            dbc.Button("🗑️", id={'type': 'delete-entry-btn', 'index': entry['id']},
                                      color="danger", size="sm")
                        ], className="float-end")
                    ], md=3)
                ]),
                html.Hr(style={'borderColor': '#495057', 'margin': '10px 0'}) if entry['notes'] else None,
                html.Div([
                    html.Span("Notes: ", style={'color': '#adb5bd', 'fontWeight': 'bold'}),
                    html.Span(entry['notes'], style={'color': '#e0e0e0'})
                ]) if entry['notes'] else None
            ])
        ], className="mb-3", style={'backgroundColor': '#2d2d2d', 'border': '1px solid #495057'})
        
        entry_cards.append(card)
    
    # Show message if no entries match filters
    if not entry_cards:
        entries_display = html.Div([
            html.P("No trades match the current filters.", 
                  className="text-center text-muted", style={'padding': '2rem'})
        ])
    else:
        entries_display = html.Div(entry_cards)
    
    return stats, advanced_stats, entries_display, count_badge

@callback(
    Output("journal-entries-container", "id", allow_duplicate=True),
    Input({'type': 'delete-entry-btn', 'index': ALL}, "n_clicks"),
    prevent_initial_call=True
)
def delete_entry(n_clicks):
    """Delete a journal entry."""
    if not any(n_clicks):
        return dash.no_update
    
    triggered_id = ctx.triggered_id
    if triggered_id:
        entry_id = triggered_id['index']
        delete_journal_entry(entry_id)
    
    return "journal-entries-container"

@callback(
    [Output("update-result-modal", "is_open"),
     Output("update-entry-id-store", "data")],
    [Input({'type': 'update-entry-btn', 'index': ALL}, "n_clicks"),
     Input("update-result-confirm-btn", "n_clicks"),
     Input("update-result-cancel-btn", "n_clicks")],
    [State("update-result-modal", "is_open"),
     State("update-entry-id-store", "data")],
    prevent_initial_call=True
)
def toggle_update_modal(update_clicks, confirm_click, cancel_click, is_open, stored_id):
    """Toggle update result modal."""
    triggered_id = ctx.triggered_id
    
    # Check if any input actually triggered the callback
    if not triggered_id:
        return dash.no_update, dash.no_update
    
    # For pattern-matching callbacks, also check if any button was actually clicked
    if isinstance(triggered_id, dict) and triggered_id['type'] == 'update-entry-btn':
        # Verify that at least one button has been clicked (not None and > 0)
        if not update_clicks or not any(c for c in update_clicks if c):
            return dash.no_update, dash.no_update
        return True, triggered_id['index']
    elif triggered_id == "update-result-cancel-btn":
        if not cancel_click:
            return dash.no_update, dash.no_update
        return False, None
    elif triggered_id == "update-result-confirm-btn":
        if not confirm_click:
            return dash.no_update, dash.no_update
        return False, stored_id
    
    return dash.no_update, dash.no_update

@callback(
    [Output("update-profit-loss-input", "value"),
     Output("update-result-dropdown", "value")],
    Input("update-exit-price-input", "value"),
    State("update-entry-id-store", "data"),
    prevent_initial_call=True
)
def calculate_pnl_from_exit(exit_price, entry_id):
    """Calculate profit/loss and determine result when exit price is entered."""
    if not exit_price or not entry_id:
        return dash.no_update, dash.no_update
    
    # Get the entry details from the database
    entries = get_journal_entries()
    entry = next((e for e in entries if e['id'] == entry_id), None)
    
    if not entry or not entry['entry_price'] or not entry['position_size']:
        return dash.no_update, dash.no_update
    
    # Calculate profit/loss based on action type
    if entry['action'] == 'BUY':
        # For BUY: profit when exit > entry
        profit_loss = (exit_price - entry['entry_price']) * entry['position_size']
    else:  # SELL
        # For SELL: profit when entry > exit
        profit_loss = (entry['entry_price'] - exit_price) * entry['position_size']
    
    # Determine result based on profit/loss
    if profit_loss > 0.01:  # Small threshold to account for rounding
        result = 'Win'
    elif profit_loss < -0.01:
        result = 'Loss'
    else:
        result = 'Breakeven'
    
    return round(profit_loss, 2), result

@callback(
    Output("journal-entries-container", "id", allow_duplicate=True),
    Input("update-result-confirm-btn", "n_clicks"),
    [State("update-entry-id-store", "data"),
     State("update-result-dropdown", "value"),
     State("update-profit-loss-input", "value"),
     State("update-exit-date-input", "value"),
     State("update-exit-price-input", "value")],
    prevent_initial_call=True
)
def update_entry_result(n_clicks, entry_id, result, profit_loss, exit_date, exit_price):
    """Update journal entry with result."""
    if not n_clicks or not entry_id:
        return dash.no_update
    
    # If exit price was not provided, we can't calculate
    if not exit_price:
        return dash.no_update
    
    # The result and profit_loss should already be calculated by the calculate_pnl_from_exit callback
    # But we'll recalculate here as a safety measure
    if not result or profit_loss is None:
        entries = get_journal_entries()
        entry = next((e for e in entries if e['id'] == entry_id), None)
        
        if entry and entry['entry_price'] and entry['position_size']:
            if entry['action'] == 'BUY':
                profit_loss = (exit_price - entry['entry_price']) * entry['position_size']
            else:
                profit_loss = (entry['entry_price'] - exit_price) * entry['position_size']
            
            profit_loss = round(profit_loss, 2)
            
            if profit_loss > 0.01:
                result = 'Win'
            elif profit_loss < -0.01:
                result = 'Loss'
            else:
                result = 'Breakeven'
    
    update_journal_entry(entry_id, result, profit_loss or 0, exit_date)
    return "journal-entries-container"
