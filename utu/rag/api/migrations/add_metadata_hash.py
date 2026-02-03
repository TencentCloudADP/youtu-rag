"""
Database migration: Add metadata_hash column to kb_source_configs

This field stores the hash of file metadata (char_length, publish_date, key_timepoints,
summary, custom tags) to detect when metadata has changed, enabling rebuilds even when
file content (ETag) is unchanged.

Usage:
    uv run python -m utu.rag.api.migrations.add_metadata_hash
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

        if 'metadata_hash' in columns:
            print("✓ Column 'metadata_hash' already exists")
            return True

        print("Adding 'metadata_hash' column to kb_source_configs table...")

        try:
            if "sqlite" in DATABASE_URL:
                # SQLite ADD COLUMN syntax
                conn.execute(text("""
                    ALTER TABLE kb_source_configs
                    ADD COLUMN metadata_hash VARCHAR(100)
                """))
                conn.commit()
            else:
                # PostgreSQL/MySQL syntax
                conn.execute(text("""
                    ALTER TABLE kb_source_configs
                    ADD COLUMN metadata_hash VARCHAR(100)
                """))
                conn.commit()

            print("✓ Column 'metadata_hash' added successfully")
            print("\nNote: Existing records will have NULL for metadata_hash.")
            print("This is expected - metadata hashes will be populated on the next build.")
            return True

        except Exception as e:
            print(f"✗ Error adding column: {e}")
            conn.rollback()
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: Add metadata_hash to kb_source_configs")
    print("=" * 60)
    print()

    success = migrate()

    print()
    if success:
        print("✓ Migration completed successfully!")
        print("\nWhat this enables:")
        print("  • Detect when file metadata changes (char_length, publish_date, etc.)")
        print("  • Trigger rebuilds for metadata-only changes (even when file content unchanged)")
        print("  • Keep vector embeddings in sync with updated metadata")
        print("  • Improve metadata consistency across ChromaDB and MinIO")
    else:
        print("✗ Migration failed. Please fix the issues above and try again.")
    print()
