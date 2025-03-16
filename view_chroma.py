import sqlite3


def view_chroma_db():
    db_path = (
        "/Users/eugenegan/Documents/GitHub/langchain-rag-tutorial/chroma/chroma.sqlite3"
    )
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", tables)

    # Query the first few rows of the 'embeddings' table
    cursor.execute("PRAGMA table_info(embeddings);")
    columns = [column[1] for column in cursor.fetchall()]
    print("Columns:", columns)

    cursor.execute("SELECT * FROM embeddings LIMIT 5;")
    rows = cursor.fetchall()
    for row in rows:
        print(dict(zip(columns, row)))

    conn.close()


if __name__ == "__main__":
    view_chroma_db()
