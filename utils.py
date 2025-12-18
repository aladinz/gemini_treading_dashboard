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
    # Create journal table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            position_size INTEGER,
            notes TEXT,
            result TEXT,
            profit_loss REAL,
            exit_date TEXT
        )
    """)
    
    # Migrate existing journal table to add exit_date column if it doesn't exist
    try:
        cursor.execute("SELECT exit_date FROM journal LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        cursor.execute("ALTER TABLE journal ADD COLUMN exit_date TEXT")
    
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

def add_journal_entry(date, ticker, action, entry_price, stop_loss, take_profit, position_size, notes):
    """Add a journal entry to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO journal (date, ticker, action, entry_price, stop_loss, take_profit, position_size, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, ticker.upper(), action, entry_price, stop_loss, take_profit, position_size, notes))
        conn.commit()
        return True, f"Journal entry for {ticker.upper()} added successfully."
    except Exception as e:
        return False, f"Error adding journal entry: {str(e)}"
    finally:
        conn.close()

def get_journal_entries():
    """Get all journal entries from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, date, ticker, action, entry_price, stop_loss, take_profit, 
               position_size, notes, result, profit_loss, exit_date 
        FROM journal 
        ORDER BY date DESC, id DESC
    """)
    entries = []
    for row in cursor.fetchall():
        entries.append({
            "id": row[0],
            "date": row[1],
            "ticker": row[2],
            "action": row[3],
            "entry_price": row[4],
            "stop_loss": row[5],
            "take_profit": row[6],
            "position_size": row[7],
            "notes": row[8],
            "result": row[9],
            "profit_loss": row[10],
            "exit_date": row[11]
        })
    conn.close()
    return entries

def delete_journal_entry(entry_id):
    """Delete a journal entry from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM journal WHERE id = ?", (entry_id,))
        conn.commit()
        return True, "Journal entry deleted successfully."
    except Exception as e:
        return False, f"Error deleting journal entry: {str(e)}"
    finally:
        conn.close()

def update_journal_entry(entry_id, result, profit_loss, exit_date=None):
    """Update journal entry with result, profit/loss, and exit date."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE journal 
            SET result = ?, profit_loss = ?, exit_date = ?
            WHERE id = ?
        """, (result, profit_loss, exit_date, entry_id))
        conn.commit()
        return True, "Journal entry updated successfully."
    except Exception as e:
        return False, f"Error updating journal entry: {str(e)}"
    finally:
        conn.close()

# Initialize DB on module import (runs only once)
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not os.environ.get("WERKZEUG_RUN_MAIN"):
    init_db()
