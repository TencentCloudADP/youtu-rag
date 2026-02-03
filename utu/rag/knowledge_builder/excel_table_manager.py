"""Manage Excel table mappings.

Supports:
1. Generate unique table names with kb_id, avoiding cross-KB conflicts
2. Register Excel-to-SQLite table mappings
3. Query table lists for a specific KB or source file
4. Clean up table mapping records
"""

import logging
import re
from typing import Optional

from sqlalchemy.orm import Session

from ..api.database import KBExcelTable

logger = logging.getLogger(__name__)


class ExcelTableManager:
    @staticmethod
    def generate_table_name(kb_id: int, filename: str, sheet_name: str) -> str:
        """Generate a unique table name: excel_{kb_id}_{filename}_{sheet}

        Args:
            kb_id: Knowledge base ID
            filename: File name (automatically remove extension)
            sheet_name: Sheet name

        Returns:
            Unique table name

        Example:
            >>> generate_table_name(1, "财务数据.xlsx", "Q2报表")
            'excel_1_财务数据_Q2报表'
        """
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]

        clean_filename = re.sub(r'[^\w\u4e00-\u9fff]', '_', filename)
        clean_sheet = re.sub(r'[^\w\u4e00-\u9fff]', '_', sheet_name)

        max_filename_len = 50
        max_sheet_len = 30

        if len(clean_filename) > max_filename_len:
            clean_filename = clean_filename[:max_filename_len]

        if len(clean_sheet) > max_sheet_len:
            clean_sheet = clean_sheet[:max_sheet_len]

        table_name = f"excel_{kb_id}_{clean_filename}_{clean_sheet}"

        logger.debug(f"Generated table name: {table_name}")
        return table_name

    @staticmethod
    def register_table(
        db: Session,
        kb_id: int,
        source_id: Optional[int],
        source_file: str,
        sheet_name: str,
        table_name: str,
        row_count: int = 0,
        column_count: int = 0
    ) -> KBExcelTable:
        """Register an Excel table mapping.

        Args:
            db: Database session
            kb_id: Knowledge base ID
            source_id: Source config ID (optional)
            source_file: Source file path
            sheet_name: Sheet name
            table_name: SQLite table name
            row_count: Table row count
            column_count: Table column count

        Returns:
            Created mapping record

        Raises:
            ValueError: raised if table name already exists
        """
        existing = db.query(KBExcelTable).filter(
            KBExcelTable.table_name == table_name
        ).first()

        if existing:  # Update if exists in the same source
            if (existing.kb_id == kb_id and
                existing.source_file == source_file and
                existing.sheet_name == sheet_name):
                existing.row_count = row_count
                existing.column_count = column_count
                db.commit()
                logger.info(f"Updated existing table mapping: {table_name}")
                return existing
            else:  # Raise error if exists in a different source
                raise ValueError(
                    f"Table name '{table_name}' already exists for a different source. "
                    f"Existing: kb_id={existing.kb_id}, file={existing.source_file}, sheet={existing.sheet_name}"
                )

        mapping = KBExcelTable(
            kb_id=kb_id,
            source_id=source_id,
            source_file=source_file,
            sheet_name=sheet_name,
            table_name=table_name,
            row_count=row_count,
            column_count=column_count
        )

        db.add(mapping)
        db.commit()
        db.refresh(mapping)

        logger.info(
            f"✅ Registered Excel table mapping: {table_name} "
            f"(kb_id={kb_id}, file={source_file}, sheet={sheet_name}, "
            f"rows={row_count}, cols={column_count})"
        )

        return mapping

    @staticmethod
    def get_tables_by_kb(db: Session, kb_id: int) -> list[KBExcelTable]:
        """Get all Excel table mappings for a given knowledge base.

        Args:
            db: Database session
            kb_id: Knowledge base ID

        Returns:
            Table mapping list
        """
        tables = db.query(KBExcelTable).filter(
            KBExcelTable.kb_id == kb_id
        ).all()

        logger.debug(f"Found {len(tables)} Excel tables for KB {kb_id}")
        return tables

    @staticmethod
    def get_tables_by_source(
        db: Session,
        kb_id: int,
        source_file: str
    ) -> list[KBExcelTable]:
        """Get all Excel table mappings for a given source file.

        Args:
            db: Database session
            kb_id: Knowledge base ID
            source_file: Source file path

        Returns:
            Table mapping list
        """
        tables = db.query(KBExcelTable).filter(
            KBExcelTable.kb_id == kb_id,
            KBExcelTable.source_file == source_file
        ).all()

        logger.debug(
            f"Found {len(tables)} Excel tables for source {source_file} in KB {kb_id}"
        )
        return tables

    @staticmethod
    def get_tables_by_source_id(
        db: Session,
        source_id: int
    ) -> list[KBExcelTable]:
        """Get Excel table mappings for a given source ID, used for cleanup.

        Args:
            db: Database session
            source_id: table ID from kb_source_configs

        Returns:
            Table mapping list
        """
        tables = db.query(KBExcelTable).filter(
            KBExcelTable.source_id == source_id
        ).all()

        logger.debug(f"Found {len(tables)} Excel tables for source_id {source_id}")
        return tables

    @staticmethod
    def delete_by_source(
        db: Session,
        kb_id: int,
        source_file: str
    ) -> int:
        """Delete all Excel table mappings for a given source file.

        Args:
            db: Database session
            kb_id: Knowledge base ID
            source_file: Source file path

        Returns:
            Number of deleted records
        """
        deleted_count = db.query(KBExcelTable).filter(
            KBExcelTable.kb_id == kb_id,
            KBExcelTable.source_file == source_file
        ).delete()

        db.commit()

        logger.info(
            f"🗑️  Deleted {deleted_count} Excel table mappings for {source_file} in KB {kb_id}"
        )
        return deleted_count

    @staticmethod
    def delete_by_kb(db: Session, kb_id: int) -> int:
        """Delete all Excel table mappings for a given knowledge base.
        Delete all table mappings for a given knowledge base.

        Args:
            db: Database session
            kb_id: Knowledge base ID

        Returns:
            Number of deleted records
        """
        deleted_count = db.query(KBExcelTable).filter(
            KBExcelTable.kb_id == kb_id
        ).delete()

        db.commit()

        logger.warning(f"🗑️  Deleted {deleted_count} Excel table mappings for KB {kb_id}")
        return deleted_count

    @staticmethod
    def delete_by_table_name(db: Session, table_name: str) -> bool:
        """Delete a specific Excel table mapping by table name.
        Delete a specific Excel table mapping by table name.

        Args:
            db: Database session
            table_name: Table name

        Returns:
            Whether deletion was successful
        """
        deleted_count = db.query(KBExcelTable).filter(
            KBExcelTable.table_name == table_name
        ).delete()

        db.commit()

        if deleted_count > 0:
            logger.info(f"🗑️  Deleted Excel table mapping: {table_name}")
            return True
        else:
            logger.warning(f"⚠️  Table mapping not found: {table_name}")
            return False

    @staticmethod
    def get_table_info(db: Session, table_name: str) -> Optional[KBExcelTable]:
        """Get table information by table name.

        Args:
            db: Database session
            table_name: Table name

        Returns:
            Table mapping info, None if not found
        """
        return db.query(KBExcelTable).filter(
            KBExcelTable.table_name == table_name
        ).first()

    @staticmethod
    def update_table_stats(
        db: Session,
        table_name: str,
        row_count: int,
        column_count: int
    ) -> bool:
        """Update table statistics.

        Args:
            db: Database session
            table_name: Table name
            row_count: New row count
            column_count: New column count

        Returns:
            Whether update was successful
        """
        table = db.query(KBExcelTable).filter(
            KBExcelTable.table_name == table_name
        ).first()

        if table:
            table.row_count = row_count
            table.column_count = column_count
            db.commit()
            logger.debug(f"Updated stats for {table_name}: {row_count} rows, {column_count} cols")
            return True
        else:
            logger.warning(f"Table mapping not found: {table_name}")
            return False
