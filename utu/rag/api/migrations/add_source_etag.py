"""
Database migration: Add source_etag column to kb_source_configs

This field stores the ETag (content hash) of MinIO files and QA files to detect
when file content has changed, enabling incremental builds.

Usage:
    uv run python -m utu.rag.api.migrations.add_source_etag
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

# Load environment variables
env_path = project_root / ".env"
load_dotenv(env_path)

DATABASE_URL = os.getenv("UTU_DB_URL", "sqlite:///./rag_data/relational_database/rag_demo.sqlite")


def migrate():
    """Apply the migration"""
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Check if column already exists
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('kb_source_configs')]

        if 'source_etag' in columns:
            print("✓ Column 'source_etag' already exists")
            return True

        print("Adding 'source_etag' column to kb_source_configs table...")

        try:
            if "sqlite" in DATABASE_URL:
                # SQLite ADD COLUMN syntax
                conn.execute(text("""
                    ALTER TABLE kb_source_configs
                    ADD COLUMN source_etag VARCHAR(100)
                """))
                conn.commit()
            else:
                # PostgreSQL/MySQL syntax
                conn.execute(text("""
                    ALTER TABLE kb_source_configs
                    ADD COLUMN source_etag VARCHAR(100)
                """))
                conn.commit()

            print("✓ Column 'source_etag' added successfully")
            print("\nNote: Existing records will have NULL for source_etag.")
            print("This is expected - ETags will be populated on the next configuration save.")
            return True

        except Exception as e:
            print(f"✗ Error adding column: {e}")
            conn.rollback()
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: Add source_etag to kb_source_configs")
    print("=" * 60)
    print()

    success = migrate()

    print()
    if success:
        print("✓ Migration completed successfully!")
        print("\nWhat this enables:")
        print("  • Detect when file content changes (even with same filename)")
        print("  • Enable incremental builds (skip unchanged files)")
        print("  • Improve build performance for large knowledge bases")
    else:
        print("✗ Migration failed. Please fix the issues above and try again.")
    print()
