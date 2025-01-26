import sqlite3
import hashlib

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Admin table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_name TEXT NOT NULL,
        image_path TEXT NOT NULL,
        result TEXT NOT NULL,
        confidence FLOAT,
        uploaded_at TIMESTAMP  DEFAULT CURRENT_TIMESTAMP
    )
''')


# Feedback table
cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    farmer_name TEXT NOT NULL,
    message TEXT NOT NULL
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS admin (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL
)
''')

# Hash the password using bcrypt
password = 'password'
hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()

# Insert a new admin record with the hashed password
cursor.execute("INSERT INTO admin (username, password) VALUES (?, ?)", ('admin', hashed_password))



conn.commit()
conn.close()
