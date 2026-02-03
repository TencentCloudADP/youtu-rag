"""Manager for cleaning up knowledge base.

Supports:
1. Cleanup vector data of deleted files
2. Cleanup SQLite tables converted from Excel
3. Cleanup column-level vectors
4. Provide data consistency guarantee
"""

import logging
import os
import sqlite3
from typing import Any

from ..storage.base_storage import BaseVectorStore

logger = logging.getLogger(__name__)


class KnowledgeCleanupManager:
    """Manage cleanup of knowledge base."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        relational_db_path: str = "rag_data/relational_database/rag_demo.sqlite"
    ):
        """Initialize cleanup manager.

        Args:
            vector_store: vector store instance
            relational_db_path: SQLite database path
        """
        self.vector_store = vector_store
        self.relational_db_path = relational_db_path

    async def cleanup_source(
        self,
        source_identifier: str,
        source_type: str,
        tables_created: list[str] | None = None,
        kb_id: int | None = None
    ) -> dict[str, Any]:
        """Cleanup a single source.

        Args:
            source_identifier: source identifier (filename or path)
            source_type: source type ("minio_file", "database", "qa_file")
            tables_created: list of table names created by this source (if any)
            kb_id: knowledge base ID (for precise deletion of mapping records, to avoid accidentally deleting same-name files in other knowledge bases)

        Returns:
            A dictionary of cleanup statistics
        """
        stats = {
            "source_identifier": source_identifier,
            "vector_chunks_deleted": 0,
            "sqlite_tables_deleted": 0,
            "errors": []
        }

        try:
            # 1. Cleanup vector data, including document main vector
            try:
                deleted_count = await self.vector_store.delete_by_document_id(source_identifier)
                stats["vector_chunks_deleted"] += deleted_count
                logger.info(f"🗑️  Deleted {deleted_count} main vector chunks for: {source_identifier}")
            except Exception as e:
                error_msg = f"Failed to delete main vectors for {source_identifier}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

            # 2. Cleanup derived vectors and tables from Excel files (MinIO or QA files)
            if source_type in ["minio_file", "qa_file"]:
                file_ext = source_identifier.split('.')[-1].lower() if '.' in source_identifier else ''

                if file_ext in ['xlsx', 'xls', 'csv', 'excel']:
                    table_vector_count = await self._cleanup_table_vectors(source_identifier, tables_created)
                    stats["vector_chunks_deleted"] += table_vector_count

                    column_vector_count = await self._cleanup_column_vectors(tables_created)
                    stats["vector_chunks_deleted"] += column_vector_count

                    text_vector_count = await self._cleanup_text_vectors(source_identifier)
                    stats["vector_chunks_deleted"] += text_vector_count

                    sqlite_count = await self._cleanup_sqlite_tables(source_identifier, tables_created, kb_id)
                    stats["sqlite_tables_deleted"] = sqlite_count

            # 3. Cleanup QA associations
            if source_type == "qa_file":
                qa_count = await self._cleanup_qa_associations(source_identifier, kb_id)
                if qa_count > 0:
                    logger.info(f"🗑️  Deleted {qa_count} QA associations for: {source_identifier}")

            logger.info(
                f"✅ Cleanup completed for {source_identifier}: "
                f"{stats['vector_chunks_deleted']} vectors, "
                f"{stats['sqlite_tables_deleted']} tables"
            )

            return stats

        except Exception as e:
            error_msg = f"Cleanup failed for {source_identifier}: {e}"
            logger.error(error_msg, exc_info=True)
            stats["errors"].append(error_msg)
            return stats

    async def _cleanup_table_vectors(
        self,
        source_identifier: str,
        tables_created: list[str] | None
    ) -> int:
        """Cleanup all the table vectors (table_schema_*).

        Method 1: Delete by metadata.source (more reliable).
        
        Method 2: Delete by document_id in the vector storage (chunk.id = table_schema_{table_name}).

        Fallback to Method 2 only if Method 1 deletes nothing.

        Args:
            source_identifier: Identifier of the source
            tables_created: List of table names; if not provided, will attempt to query the database or derive from source_identifier

        Returns:
            Number of deleted vectors
        """
        if not tables_created:
            tables_created = self._query_table_names_from_db(source_identifier)
            if not tables_created:
                tables_created = self._infer_table_names(source_identifier)

        total_deleted = 0

        # Method 1: by source metadata
        if hasattr(self.vector_store, 'delete_by_metadata'):
            try:
                deleted = await self.vector_store.delete_by_metadata({
                    "source": source_identifier,
                    "type": "table_schema"
                })
                total_deleted += deleted
                if deleted > 0:
                    logger.info(f"🗑️  Deleted {deleted} table-level vectors by metadata")
            except Exception as e:
                logger.warning(f"Failed to delete table vectors by metadata: {e}")

        # Method 2: by metadata.source, delete each table individually
        if total_deleted == 0 and tables_created:
            for table_name in tables_created:
                try:
                    doc_id = f"table_schema_{table_name}"
                    deleted = await self.vector_store.delete_by_document_id(doc_id)
                    total_deleted += deleted

                    if deleted > 0:
                        logger.debug(f"🗑️  Deleted table vector for: {table_name}")

                except Exception as e:
                    logger.warning(f"Failed to delete table vector for {table_name}: {e}")

            if total_deleted > 0:
                logger.info(f"🗑️  Deleted {total_deleted} table-level vectors")

        return total_deleted

    async def _cleanup_column_vectors(
        self,
        tables_created: list[str] | None
    ) -> int:
        """Cleanup all the column vectors (column_values_*).

        Method 1: Delete by metadata.table_name (recommended).

        Method 2: Delete by document_id (requires querying SQLite, adopted only when Method 1 deletes nothing).

        Args:
            tables_created: List of table names

        Returns:
            Number of deleted vectors
        """
        if not tables_created:
            return 0

        total_deleted = 0

        # Method 1: by metadata.table_name
        # Note: delete both column_value (for individual strategy) and column_values (for concatenate strategy)
        if hasattr(self.vector_store, 'delete_by_metadata'):
            for table_name in tables_created:
                try:
                    deleted = await self.vector_store.delete_by_metadata({
                        "table_name": table_name,
                        "type": "column_value"
                    })
                    total_deleted += deleted
                    if deleted > 0:
                        logger.debug(f"🗑️  Deleted {deleted} column vectors (individual) for table: {table_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete column vectors (individual) for {table_name}: {e}")

                try:
                    deleted = await self.vector_store.delete_by_metadata({
                        "table_name": table_name,
                        "type": "column_values"
                    })
                    total_deleted += deleted
                    if deleted > 0:
                        logger.debug(f"🗑️  Deleted {deleted} column vectors (concatenate) for table: {table_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete column vectors (concatenate) for {table_name}: {e}")

            if total_deleted > 0:
                logger.info(f"🗑️  Deleted {total_deleted} column-level vectors by metadata")
                return total_deleted

        # Method 2: by document_id
        if total_deleted == 0 and os.path.exists(self.relational_db_path):
            conn = sqlite3.connect(self.relational_db_path)
            conn.row_factory = sqlite3.Row

            try:
                for table_name in tables_created:
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table_name,)
                    )
                    if not cursor.fetchone():
                        continue

                    columns_info = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()

                    for col_info in columns_info:
                        col_name = col_info["name"]
                        try:
                            doc_id = f"column_values_{table_name}_{col_name}"
                            deleted = await self.vector_store.delete_by_document_id(doc_id)
                            total_deleted += deleted

                            if deleted > 0:
                                logger.debug(f"🗑️  Deleted column vector: {table_name}.{col_name}")

                        except Exception as e:
                            logger.warning(f"Failed to delete column vector for {table_name}.{col_name}: {e}")

            except Exception as e:
                logger.error(f"Failed to cleanup column vectors: {e}")
            finally:
                conn.close()

            if total_deleted > 0:
                logger.info(f"🗑️  Deleted {total_deleted} column-level vectors")

        return total_deleted

    async def _cleanup_text_vectors(self, source_identifier: str) -> int:
        """Cleanup all the text vectors (*_text_chunk_*).

        Excel documents are text-ified, generating the following vectors:
        - document_id: {source_identifier}_text
        - chunk_id: {source_identifier}_text_chunk_0, _chunk_1, ...

        Args:
            source_identifier: Source file identifier (e.g., "file.xlsx")

        Returns:
            Number of deleted vectors
        """
        total_deleted = 0

        try:
            text_document_id = f"{source_identifier}_text"

            if hasattr(self.vector_store, 'delete_by_document_id'):
                try:
                    deleted = await self.vector_store.delete_by_document_id(text_document_id)
                    total_deleted += deleted
                    if deleted > 0:
                        logger.info(f"🗑️  Deleted {deleted} text vectors for: {source_identifier}")
                except Exception as e:
                    logger.warning(f"Failed to delete text vectors for {source_identifier}: {e}")

        except Exception as e:
            logger.error(f"Failed to cleanup text vectors for {source_identifier}: {e}")

        return total_deleted

    async def _cleanup_sqlite_tables(
        self,
        source_identifier: str,
        tables_created: list[str] | None,
        kb_id: int | None = None
    ) -> int:
        """Cleanup SQLite tables and delete records.

        Args:
            source_identifier: Source identifier
            tables_created: List of table names; if not provided, query from SQLite or infer from source_identifier
            kb_id: Knowledge base ID (for precise deletion of mapping records, to avoid deleting other KBs' same-name files)

        Returns:
            Number of deleted tables
        """
        if not tables_created:
            tables_created = self._query_table_names_from_db(source_identifier)
            if not tables_created:
                tables_created = self._infer_table_names(source_identifier)

        if not tables_created or not os.path.exists(self.relational_db_path):
            return 0

        deleted_count = 0
        conn = sqlite3.connect(self.relational_db_path)

        try:
            for table_name in tables_created:
                try:
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table_name,)
                    )

                    if cursor.fetchone():
                        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                        conn.commit()
                        deleted_count += 1
                        logger.info(f"🗑️  Dropped SQLite table: {table_name}")

                except Exception as e:
                    logger.error(f"Failed to drop table {table_name}: {e}")

        except Exception as e:
            logger.error(f"SQLite cleanup error: {e}")
        finally:
            conn.close()

        # Note: records should be deleted after the SQLite connection is closed
        if deleted_count > 0:
            try:
                from utu.rag.api.database import get_db_session, KBExcelTable

                db = get_db_session()
                try:
                    # Filter files by both source_file and kb_id (if provided) to avoid deleting other KBs' same-name files
                    if kb_id is not None:
                        mapping_deleted = db.query(KBExcelTable).filter(
                            KBExcelTable.source_file == source_identifier,
                            KBExcelTable.kb_id == kb_id
                        ).delete()
                        logger.info(f"🗑️  Deleted {mapping_deleted} table mapping records for {source_identifier} in KB {kb_id}")
                    else:  # Fallback to conventional deletion, which has potential risks to delete same-name files in other KBs
                        mapping_deleted = db.query(KBExcelTable).filter(
                            KBExcelTable.source_file == source_identifier
                        ).delete()
                        logger.warning(f"⚠️  Deleted {mapping_deleted} table mapping records for {source_identifier} (without kb_id filter, may affect other KBs)")

                    db.commit()

                finally:
                    db.close()

            except Exception as e:
                logger.warning(f"Failed to delete table mapping records: {e}")

        return deleted_count

    async def _cleanup_qa_associations(self, source_identifier: str, kb_id: int = None) -> int:
        """Cleanup QA associations.

        Args:
            source_identifier: Source identifier for QA files.
            kb_id: Knowledge base ID (for precise deletion).

        Returns:
            Number of deleted QA associations.
        """
        if not kb_id:
            logger.warning(f"No kb_id provided for QA cleanup, skipping: {source_identifier}")
            return 0

        conn = sqlite3.connect(self.relational_db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM qa_associations WHERE kb_id = ? AND source_file = ?",
                (kb_id, source_identifier)
            )
            deleted_count = cursor.rowcount
            conn.commit()

            logger.info(f"Deleted {deleted_count} QA associations for source: {source_identifier}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup QA associations for {source_identifier}: {e}")
            return 0
        finally:
            conn.close()

    def _query_table_names_from_db(self, source_identifier: str) -> list[str]:
        """Query table names from SQLite database (kb_excel_tables).

        This method is preferred over _infer_table_names.

        Args:
            source_identifier: Source identifier (file path)

        Returns:
            Table name list
        """
        try:
            import sys
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from utu.rag.api.database import get_db_session, KBExcelTable

            db = get_db_session()
            try:
                tables = db.query(KBExcelTable).filter(
                    KBExcelTable.source_file == source_identifier
                ).all()

                table_names = [t.table_name for t in tables]

                if table_names:
                    logger.info(f"Found {len(table_names)} tables from kb_excel_tables for {source_identifier}")

                return table_names

            finally:
                db.close()

        except Exception as e:
            logger.warning(f"Failed to query table names from kb_excel_tables: {e}")
            return []

    def _infer_table_names(self, source_identifier: str) -> list[str]:
        """Infer possible table names from source identifier.

        Note: It is recommended to use _query_table_names_from_db whenever possible.

        Args:
            source_identifier: Source identifier (file path)

        Returns:
            Possible table name list
        """
        from pathlib import Path

        if not source_identifier:
            return []

        filename = Path(source_identifier).stem

        clean_filename = "".join(c if c.isalnum() or c == "_" else "_" for c in filename)

        possible_names = [
            f"excel_{clean_filename}",  # Single sheet
            # May try to match all tables of the form excel_{filename}_*
        ]

        # Find tables in the database that match the pattern excel_*_{filename}_* or excel_{filename}_*
        if os.path.exists(self.relational_db_path):
            conn = sqlite3.connect(self.relational_db_path)
            try:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE ? OR name LIKE ?)",
                    (f"excel_%_{clean_filename}_%", f"excel_{clean_filename}%")
                )
                matched_tables = [row[0] for row in cursor.fetchall()]
                return matched_tables
            except Exception as e:
                logger.error(f"Failed to infer table names: {e}")
            finally:
                conn.close()

        return possible_names

    async def cleanup_knowledge_base(self, kb_id: int, collection_name: str) -> dict[str, Any]:
        """Cleanup all the data in the knowledge base.

        ⚠️ Warning: This will delete all vectors and tables!

        Args:
            kb_id: Knowledge base ID
            collection_name: Collection name

        Returns:
            A dictionary of cleanup statistics
        """
        stats = {
            "kb_id": kb_id,
            "collection_name": collection_name,
            "total_vectors_deleted": 0,
            "total_tables_deleted": 0,
            "errors": []
        }

        try:
            logger.warning(f"⚠️  Starting complete cleanup for KB {kb_id} (collection: {collection_name})")

            # 1. Delete vector collection (all the vector data)
            try:
                logger.info(f"🗑️  Deleting entire vector collection: {collection_name}")

                if hasattr(self.vector_store, 'delete_collection'):
                    self.vector_store.delete_collection()
                    stats["total_vectors_deleted"] = -1  # -1 represents all vectors deleted
                    logger.info(f"✅ Deleted vector collection: {collection_name}")

            except Exception as e:
                error_msg = f"Failed to delete vector collection: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

            # 2. Delete all SQLite tables
            try:
                from utu.rag.api.database import get_db_session, KBExcelTable

                db = get_db_session()
                excel_tables = []
                try:
                    tables = db.query(KBExcelTable).filter(
                        KBExcelTable.kb_id == kb_id
                    ).all()
                    excel_tables = [t.table_name for t in tables]
                    logger.info(f"Found {len(excel_tables)} Excel tables for KB {kb_id}")
                finally:
                    db.close()

                if excel_tables and os.path.exists(self.relational_db_path):
                    conn = sqlite3.connect(self.relational_db_path)
                    try:
                        for table_name in excel_tables:
                            try:
                                cursor = conn.execute(
                                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                                    (table_name,)
                                )
                                if cursor.fetchone():
                                    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                                    stats["total_tables_deleted"] += 1
                                    logger.info(f"🗑️  Dropped table: {table_name}")
                            except Exception as e:
                                logger.error(f"Failed to drop table {table_name}: {e}")

                        conn.commit()
                    finally:
                        conn.close()

            except Exception as e:
                logger.warning(f"Failed to query/delete Excel tables: {e}")

            # 3. Delete table mapping records
            try:
                from utu.rag.api.database import get_db_session, KBExcelTable

                db = get_db_session()
                try:
                    mapping_deleted = db.query(KBExcelTable).filter(
                        KBExcelTable.kb_id == kb_id
                    ).delete()
                    db.commit()

                    if mapping_deleted > 0:
                        logger.info(f"🗑️  Deleted {mapping_deleted} Excel table mapping records")

                finally:
                    db.close()

            except Exception as e:
                logger.warning(f"Failed to delete table mapping records: {e}")

            # 4. Delete QA associations
            if os.path.exists(self.relational_db_path):
                conn = sqlite3.connect(self.relational_db_path)
                try:
                    try:
                        cursor = conn.execute(
                            "DELETE FROM qa_associations WHERE kb_id = ?",
                            (kb_id,)
                        )
                        qa_deleted = cursor.rowcount
                        conn.commit()

                        if qa_deleted > 0:
                            logger.info(f"🗑️  Deleted {qa_deleted} QA associations")

                    except Exception as e:
                        logger.warning(f"Failed to cleanup QA associations: {e}")

                except Exception as e:
                    error_msg = f"SQLite cleanup error: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                finally:
                    conn.close()

            logger.warning(
                f"✅ Complete cleanup finished for KB {kb_id}: "
                f"collection deleted, {stats['total_tables_deleted']} tables dropped"
            )

            return stats

        except Exception as e:
            error_msg = f"Complete KB cleanup failed: {e}"
            logger.error(error_msg, exc_info=True)
            stats["errors"].append(error_msg)
            return stats


async def cleanup_removed_source(
    vector_store: BaseVectorStore,
    source_identifier: str,
    source_type: str,
    tables_created: list[str] | None = None,
    kb_id: int | None = None,
    relational_db_path: str = "rag_data/relational_database/rag_demo.sqlite"
) -> dict[str, Any]:
    """Convenience function to cleanup removed source.
    Clean up removed data source (convenience function)

    Args:
        vector_store: Vector store instance
        source_identifier: Source identifier
        source_type: Source type
        tables_created: Created table name list
        kb_id: Knowledge base ID (for precise deletion of mapping records, to avoid accidentally deleting same-name files from other knowledge bases)
        relational_db_path: SQLite path

    Returns:
        A dictionary of cleanup statistics
    """
    manager = KnowledgeCleanupManager(vector_store, relational_db_path)
    return await manager.cleanup_source(source_identifier, source_type, tables_created, kb_id)
