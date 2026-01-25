from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parents[1]

CONVERSATIONS_DB = BASE_DIR / "data" / "memories" / "conversations.db"

def view_database(db_path, db_name):
    print(f"\n{'='*80}")
    print(f"DATABASE: {db_name}")
    print(f"Path: {db_path}")
    print(f"{'='*80}\n")
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}\n")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in this database.\n")
            return
        
        print(f"Found {len(tables)} table(s):\n")
        
        for (table_name,) in tables:
            print(f"\n📋 Table: {table_name}")
            print("-" * 80)
            
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print(f"Columns: {', '.join([col[1] for col in columns])}")
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"Row count: {count}")
            
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                rows = cursor.fetchall()
                print("\nSample data (first 5 rows):")
                for i, row in enumerate(rows, 1):
                    print(f"  Row {i}: {row}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Error reading database: {e}\n")

if __name__ == "__main__":
    view_database(CONVERSATIONS_DB, "conversations.db")
    print("\n" + "="*80)
    print("Done!")
    print("="*80)
