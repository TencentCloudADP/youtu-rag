"""Knowledge config routes, including tool configuration, building, and validation"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.kb_config import (
    KBConfigurationUpdate,
    KBBuildRequest,
    KBBuildResponse,
    QAValidationResult,
    DBConnectionTestRequest,
    DBConnectionTestResponse,
)
from ..services.kb_config_service import KBConfigService


logger = logging.getLogger(__name__)
router = APIRouter()


@router.put("/{kb_id}/configuration")
async def update_kb_configuration(
    kb_id: int,
    config_update: KBConfigurationUpdate,
    db: Session = Depends(get_db)
):
    """Update knowledge base config (tools, files, and connections)
    
    Args:
        kb_id: Knowledge base ID.
        config_update: Update configuration.
        
    Returns:
        Result of the update.
        
    Example:
        ```
        PUT /api/knowledge/{kb_id}/configuration
        {
            "configuration": {
                "tools": {...},
                "selectedFiles": ["doc1.pdf"],
                "selectedQAFiles": ["qa.xlsx"],
                "dbConnections": [...]
            }
        }
        ```
    """
    try:
        tools_config_dict = {}
        for tool_name, tool_config in config_update.configuration.tools.items():
            tools_config_dict[tool_name] = {
                "enabled": tool_config.enabled,
                "settings": tool_config.settings
            }
        
        result = await KBConfigService.update_configuration(
            kb_id=kb_id,
            tools_config=tools_config_dict,
            selected_files=config_update.configuration.selectedFiles,
            selected_qa_files=config_update.configuration.selectedQAFiles,
            db_connections=config_update.configuration.dbConnections,
            db=db
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error(f"Update configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{kb_id}/build", response_model=KBBuildResponse)
async def build_knowledge_base(
    kb_id: int,
    build_request: KBBuildRequest,
    db: Session = Depends(get_db)
):
    """Build/rebuild knowledge base using configured tools and sources.

    The build process:
    1. Load sources from config (MinIO files, database, QA files);
    2. Process sources in parallel using KnowledgeBuilderAgent;
    3. Store vectors in ChromaDB, structured data in SQLite;
    4. Update build status and logs.
    
    Args:
        kb_id: Knowledge base ID.
        build_request: Build options.
        
    Returns:
        Build results.
        
    Example:
        ```
        POST /api/knowledge/{kb_id}/build
        {
            "force_rebuild": false,
            "file_filter": ["doc1.pdf"]
        }
        ```
    """
    try:
        from ..kb_config_routes import build_knowledge_base as _build_impl
        return await _build_impl(kb_id, build_request, db)
    except Exception as e:
        logger.error(f"Build error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/validate-qa/{filename}", response_model=QAValidationResult)
async def validate_qa_file(filename: str):
    """Validate the format of a QA Excel file.
    
    Expected format:
    - Sheet name: "example"
    - Headers: "question", "answer", "howtofind"
    
    Args:
        filename: QA Excel name in MinIO
        
    Returns:
        Validation results.
        
    Example:
        ```
        GET /api/knowledge/files/validate-qa/qa_examples.xlsx
        ```
    """
    try:
        result = await KBConfigService.validate_qa_file(filename)
        return QAValidationResult(**result)
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/database/test-connection", response_model=DBConnectionTestResponse)
async def test_database_connection(request: DBConnectionTestRequest):
    """Test database connection and retrieve table list.

    Supporting MySQL and SQLite connections.
    - MySQL: requires host, port, database, username, password
    - SQLite: requires file_path
    
    Returns available table list on success.
    
    Args:
        request: Request for database connection test.
        
    Returns:
        Connection test results.
        
    Example:
        ```
        POST /api/knowledge/database/test-connection
        {
            "db_type": "mysql",
            "host": "localhost",
            "port": 3306,
            "database": "mydb",
            "username": "user",
            "password": "pass"
        }
        ```
    """
    try:
        result = await KBConfigService.test_database_connection(
            db_type=request.db_type,
            host=request.host,
            port=request.port,
            database=request.database,
            username=request.username,
            password=request.password,
            file_path=request.file_path
        )
        return DBConnectionTestResponse(**result)
    except Exception as e:
        logger.error(f"Connection test error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

