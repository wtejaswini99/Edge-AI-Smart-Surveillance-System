import sqlite3
from datetime import datetime

# Connect to database
conn = sqlite3.connect("surveillance.db")
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    time TEXT,
    mask TEXT,
    distance REAL,
    image TEXT
)
""")

conn.commit()
conn.close()


def save_detection(name, mask, distance, image=""):
    conn = sqlite3.connect("surveillance.db")
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")

    cursor.execute("""
    INSERT INTO detections
    (name, date, time, mask, distance, image)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        name,
        date,
        time,
        mask,
        float(distance),
        image
    ))

    conn.commit()
    conn.close()
