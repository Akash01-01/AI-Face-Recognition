#!/usr/bin/env python3
"""
Database migration script to add phone and dob columns to existing Person table.
Run this script to update your existing database safely.
"""

import sqlite3
import os
import sys

def migrate_database():
    # Check both possible database locations
    db_paths = ['people.db', 'instance/people.db']
    db_path = None
    
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    print("Starting database migration...")
    
    if not db_path:
        print("No existing database found.")
        print("Creating new database with updated schema...")
        # Let the app create the new database
        return True
    
    print(f"Found database at: {db_path}")
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(person)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"Current columns: {columns}")
        
        # Add phone column if it doesn't exist
        if 'phone' not in columns:
            cursor.execute("ALTER TABLE person ADD COLUMN phone VARCHAR(20)")
            print("✅ Added 'phone' column")
        else:
            print("ℹ️  'phone' column already exists")
            
        # Add dob column if it doesn't exist  
        if 'dob' not in columns:
            cursor.execute("ALTER TABLE person ADD COLUMN dob DATE")
            print("✅ Added 'dob' column")
        else:
            print("ℹ️  'dob' column already exists")
            
        conn.commit()
        print("✅ Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == '__main__':
    success = migrate_database()
    if not success:
        sys.exit(1)