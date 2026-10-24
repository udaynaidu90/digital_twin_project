"""
db_utils.py

Small SQLite helper layer. Keeps table creation and queries in one
place so both simulate_stream.py (writer) and app.py (reader) stay
in sync.
"""

import sqlite3
import os
import pandas as pd

from data_loader import COLUMNS

DB_PATH = os.path.join("database", "sensor_data.db")


def get_connection():
    os.makedirs("database", exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    columns_sql = ", ".join([f'"{c}" REAL' for c in COLUMNS])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS live_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            {columns_sql}
        )
    """)
    conn.commit()
    conn.close()


def insert_reading(row_dict, timestamp):
    conn = get_connection()
    cur = conn.cursor()
    cols = ["timestamp"] + list(row_dict.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join([f'"{c}"' for c in cols])
    values = [timestamp] + list(row_dict.values())
    cur.execute(f'INSERT INTO live_readings ({col_names}) VALUES ({placeholders})', values)
    conn.commit()
    conn.close()


def get_latest_readings(limit_per_machine=50):
    """Return the most recent readings for every machine, as a DataFrame."""
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM live_readings", conn)
    except Exception:
        df = pd.DataFrame(columns=["timestamp"] + COLUMNS)
    conn.close()

    if df.empty:
        return df

    df = df.sort_values("id")
    # Keep only the last N rows per machine for display/inference
    df = df.groupby("unit_number").tail(limit_per_machine)
    return df


def reset_db():
    """Wipe all stored readings (useful when restarting a demo)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS live_readings")
    conn.commit()
    conn.close()
    init_db()
