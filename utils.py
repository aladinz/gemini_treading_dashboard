"""
Utility functions for the Gemini Trading Dashboard, including database operations 
and stock information retrieval.
"""
import sqlite3
import os
import yfinance as yf

# --- Database Setup ---
DB_FILE = "watchlist.db"

def init_db():
    """Initialize the database with required tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            company_name TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_stock_to_db(ticker, company_name):
    """Add a stock to the watchlist database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR REPLACE INTO watchlist (ticker, company_name) VALUES (?, ?)", 
                      (ticker.upper(), company_name))
        conn.commit()
        return True, f"{ticker.upper()} added/updated in watchlist."
    except Exception as e:
        return False, f"Error adding {ticker.upper()}: {str(e)}"
    finally:
        conn.close()

def remove_stock_from_db(ticker):
    """Remove a stock from the watchlist database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
        conn.commit()
        return True, f"{ticker.upper()} removed from watchlist."
    except Exception as e:
        return False, f"Error removing {ticker.upper()}: {str(e)}"
    finally:
        conn.close()

def get_watchlist_from_db():
    """Get all stocks in the watchlist database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, company_name FROM watchlist ORDER BY ticker ASC")
    watchlist = [{"ticker": row[0], "company_name": row[1]} for row in cursor.fetchall()]
    conn.close()
    return watchlist

def get_stock_info_for_watchlist_display(ticker_symbol):
    """Fetches company name and recommendation for a ticker for display purposes."""
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        company_name = info.get('longName', info.get('shortName', ticker_symbol))
        recommendation = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
        # Ensure company_name is not None or empty
        if not company_name:
            company_name = ticker_symbol # Fallback to ticker if name is missing
        return company_name, recommendation
    except Exception:
        # If yfinance fails for any reason (e.g. delisted ticker, network issue)
        return ticker_symbol, "N/A"

# Initialize DB on module import (runs only once)
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not os.environ.get("WERKZEUG_RUN_MAIN"):
    init_db()
