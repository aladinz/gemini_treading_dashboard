import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], use_pages=True, 
                 suppress_callback_exceptions=True)
server = app.server

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Dashboard", href="/", style={'fontSize': '1.1rem'})),
        dbc.NavItem(dbc.NavLink("Watchlist", href="/watchlist", style={'fontSize': '1.1rem'})),
    ],
    brand="📊 Swing Trading Dashboard",
    brand_href="/",
    color="dark",
    dark=True,
    className="mb-4",
    style={'borderBottom': '2px solid #404040'}
)

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    navbar,
    dash.page_container
], fluid=True, style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'paddingBottom': '2rem'})

if __name__ == '__main__':
    app.run(debug=True, dev_tools_ui=False, dev_tools_props_check=False)
