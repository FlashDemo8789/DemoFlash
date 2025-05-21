import os
import sqlite3

# Make sure the directory exists
db_path = "./flashcamp.db"
db_dir = os.path.dirname(os.path.abspath(db_path))
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# Create the database file
conn = sqlite3.connect(db_path)
print(f"Created database at {os.path.abspath(db_path)}")
conn.close() 