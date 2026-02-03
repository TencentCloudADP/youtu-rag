"""Inspect SQLite database in rag_data/relational_database.

This script reads and displays:
- All tables in the database
- Table schema (columns and types)
- First 10 rows of each table
"""

import sqlite3
from pathlib import Path
from typing import List, Tuple


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section_header(title: str):
    """Print a section header."""
    print()
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def get_all_tables(cursor: sqlite3.Cursor) -> List[str]:
    """Get all table names in the database.

    Args:
        cursor: SQLite cursor

    Returns:
        List of table names
    """
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]
    return tables


def get_table_info(cursor: sqlite3.Cursor, table_name: str) -> List[Tuple]:
    """Get column information for a table.

    Args:
        cursor: SQLite cursor
        table_name: Name of the table

    Returns:
        List of tuples containing column info (cid, name, type, notnull, default, pk)
    """
    cursor.execute(f"PRAGMA table_info({table_name});")
    return cursor.fetchall()


def get_table_data(cursor: sqlite3.Cursor, table_name: str, limit: int = 10) -> Tuple[List[str], List[Tuple]]:
    """Get first N rows from a table.

    Args:
        cursor: SQLite cursor
        table_name: Name of the table
        limit: Number of rows to fetch

    Returns:
        Tuple of (column_names, rows)
    """
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
    column_names = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    return column_names, rows


def format_value(value, max_length: int = 50) -> str:
    """Format a value for display, truncating if too long.

    Args:
        value: Value to format
        max_length: Maximum length before truncation

    Returns:
        Formatted string
    """
    if value is None:
        return "NULL"

    str_value = str(value)
    if len(str_value) > max_length:
        return str_value[:max_length] + "..."
    return str_value


def print_table_info(cursor: sqlite3.Cursor, table_name: str):
    """Print detailed information about a table.

    Args:
        cursor: SQLite cursor
        table_name: Name of the table
    """
    print_section_header(f"Table: {table_name}")

    # Get table schema
    table_info = get_table_info(cursor, table_name)

    print("📋 Schema:")
    print()
    print(f"{'Column':<20} {'Type':<15} {'Nullable':<10} {'Default':<15} {'PK':<5}")
    print("-" * 75)

    for col in table_info:
        cid, name, col_type, notnull, default_val, pk = col
        nullable = "NO" if notnull else "YES"
        default = format_value(default_val, 15) if default_val else "-"
        is_pk = "✓" if pk else ""

        print(f"{name:<20} {col_type:<15} {nullable:<10} {default:<15} {is_pk:<5}")

    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]

    print()
    print(f"📊 Total rows: {row_count}")

    # Get and print first 10 rows
    if row_count > 0:
        print()
        print("📄 First 10 rows:")
        print()

        column_names, rows = get_table_data(cursor, table_name, limit=10)

        # Calculate column widths
        col_widths = [len(name) for name in column_names]
        for row in rows:
            for i, value in enumerate(row):
                # Special handling for source_etag column - don't truncate
                max_len = 100 if column_names[i] == 'source_etag' else 300
                col_widths[i] = max(col_widths[i], len(format_value(value, max_len)))

        # Limit column width to 30 characters for display (except source_etag)
        col_widths = [
            min(w, 100) if column_names[i] == 'source_etag' else min(w, 300)
            for i, w in enumerate(col_widths)
        ]

        # Print header
        header = " | ".join(name[:w].ljust(w) for name, w in zip(column_names, col_widths))
        print(header)
        print("-" * len(header))

        # Print rows
        for row in rows:
            formatted_row = " | ".join(
                format_value(value, 100 if column_names[i] == 'source_etag' else 300)[:w].ljust(w)
                for i, (value, w) in enumerate(zip(row, col_widths))
            )
            print(formatted_row)
    else:
        print()
        print("⚠️  Table is empty")

    print()


def inspect_database(db_path: str | Path):
    """Inspect a SQLite database and print all tables and their contents.

    Args:
        db_path: Path to the SQLite database file
    """
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"❌ Database file not found: {db_path}")
        print(f"\nThe database will be created automatically when the application starts.")
        return

    print_section_header(f"Database Inspector: {db_path.name}")
    print(f"📁 Location: {db_path.absolute()}")
    print(f"📦 Size: {db_path.stat().st_size / 1024:.2f} KB")

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get all tables
        tables = get_all_tables(cursor)

        if not tables:
            print()
            print("⚠️  No tables found in the database")
            return

        print()
        print(f"📊 Found {len(tables)} table(s): {', '.join(tables)}")

        # Print info for each table
        for table_name in tables:
            print_table_info(cursor, table_name)

        print_separator()
        print(f"✅ Database inspection complete. Total tables: {len(tables)}")
        print_separator()

    finally:
        conn.close()


def main():
    """Main entry point."""
    # Get project root (tests/rag -> project root is 2 levels up)
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "rag_data" / "relational_database" / "rag_demo.sqlite"
    print(db_path)

    print()
    print("🔍 RAG Database Inspector")
    print()

    inspect_database(db_path)


if __name__ == "__main__":
    main()
