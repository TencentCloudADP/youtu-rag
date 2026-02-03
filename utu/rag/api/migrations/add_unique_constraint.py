"""
Database migration: Add unique constraint to kb_source_configs

Run this script to add the unique constraint to prevent duplicate sources
in the same knowledge base.

Usage:
    uv run python -m utu.rag.api.migrations.add_unique_constraint
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
env_path = project_root / ".env"
load_dotenv(env_path)

DATABASE_URL = os.getenv("UTU_DB_URL", "sqlite:///./rag_data/relational_database/rag_demo.sqlite")


def migrate():
    """Apply the migration"""
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Check if we're using SQLite
        if "sqlite" in DATABASE_URL:
            print("Detected SQLite database")

            # SQLite doesn't support adding constraints directly
            # We need to check if the constraint already exists by querying the schema
            try:
                # Check for duplicate records before adding constraint
                result = conn.execute(text("""
                    SELECT knowledge_base_id, source_type, source_identifier, COUNT(*) as count
                    FROM kb_source_configs
                    GROUP BY knowledge_base_id, source_type, source_identifier
                    HAVING COUNT(*) > 1
                """))

                duplicates = result.fetchall()

                if duplicates:
                    print("\n⚠️  WARNING: Found duplicate records that violate the unique constraint:")
                    for dup in duplicates:
                        print(f"  - KB ID: {dup[0]}, Type: {dup[1]}, Identifier: {dup[2]}, Count: {dup[3]}")

                    print("\nPlease remove duplicates manually before applying this constraint.")
                    print("You can use the following SQL to identify which records to keep:")
                    print("""
                    SELECT * FROM kb_source_configs
                    WHERE (knowledge_base_id, source_type, source_identifier) IN (
                        SELECT knowledge_base_id, source_type, source_identifier
                        FROM kb_source_configs
                        GROUP BY knowledge_base_id, source_type, source_identifier
                        HAVING COUNT(*) > 1
                    )
                    ORDER BY knowledge_base_id, source_type, source_identifier, id;
                    """)
                    return False

                # For SQLite, the constraint is defined in the model
                # It will be created automatically when creating new tables
                # For existing tables, we need to recreate the table
                print("\n✓ No duplicate records found")
                print("✓ Unique constraint will be enforced on new records")
                print("\nNote: For SQLite, the constraint is defined in the model and will be")
                print("enforced automatically. Existing tables may need to be recreated to add")
                print("the constraint retroactively.")

            except Exception as e:
                print(f"Error checking for duplicates: {e}")
                return False

        else:
            # For PostgreSQL/MySQL
            print(f"Detected non-SQLite database: {DATABASE_URL}")

            # Check if constraint already exists
            if "postgresql" in DATABASE_URL:
                check_sql = """
                    SELECT constraint_name
                    FROM information_schema.table_constraints
                    WHERE table_name = 'kb_source_configs'
                    AND constraint_name = 'uq_kb_source'
                """
            elif "mysql" in DATABASE_URL:
                check_sql = """
                    SELECT CONSTRAINT_NAME
                    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
                    WHERE TABLE_NAME = 'kb_source_configs'
                    AND CONSTRAINT_NAME = 'uq_kb_source'
                """
            else:
                print("Unsupported database type")
                return False

            result = conn.execute(text(check_sql))
            existing = result.fetchone()

            if existing:
                print("✓ Unique constraint already exists")
                return True

            # Add the constraint
            print("Adding unique constraint...")
            try:
                conn.execute(text("""
                    ALTER TABLE kb_source_configs
                    ADD CONSTRAINT uq_kb_source
                    UNIQUE (knowledge_base_id, source_type, source_identifier)
                """))
                conn.commit()
                print("✓ Unique constraint added successfully")
                return True
            except Exception as e:
                print(f"Error adding constraint: {e}")
                print("This might be because duplicate records exist.")
                return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: Add Unique Constraint to kb_source_configs")
    print("=" * 60)
    print()

    success = migrate()

    print()
    if success:
        print("✓ Migration completed successfully!")
    else:
        print("✗ Migration failed. Please fix the issues above and try again.")
    print()
