"""File and knowledge base selection routes"""
import logging
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Body
from sqlalchemy.orm import Session

from ..database import get_db, KnowledgeBase, KBSourceConfig
from ..minio_client import minio_client

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/list")
async def get_knowledge_base_list(db: Session = Depends(get_db)):
    """List all knowledge bases, used for dropdown selection.
    
    Returns:
        List of knowledge bases with id, name, collection_name, and description.
    """
    try:
        knowledge_bases = db.query(KnowledgeBase).all()
        
        result = []
        for kb in knowledge_bases:
            result.append({
                "id": kb.id,
                "name": kb.name,
                "collection_name": kb.collection_name,
                "description": kb.description
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Get knowledge base list error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select/{kb_id}")
async def select_knowledge_base(kb_id: int, db: Session = Depends(get_db)):
    """Select a knowledge base and set KB_NAME and kb_collection_name environment variables.
    
    Args:
        kb_id: The knowledge base ID to select.
        
    Returns:
        Selected knowledge base information.
    """
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        
        # 使用 collection_name 设置环境变量 KB_NAME 和 kb_collection_name
        os.environ["KB_NAME"] = kb.collection_name
        os.environ["kb_collection_name"] = kb.collection_name
        
        logger.info(f"Selected knowledge base: {kb.name} (collection: {kb.collection_name})")
        
        return {
            "id": kb.id,
            "name": kb.name,
            "collection_name": kb.collection_name,
            "description": kb.description,
            "message": f"Knowledge base '{kb.name}' selected successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Select knowledge base error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_knowledge_base(db: Session = Depends(get_db)):
    """Get the currently selected knowledge base.
    
    Returns:
        Current knowledge base information, or None if not selected.
    """
    try:
        kb_name = os.getenv("KB_NAME")
        kb_collection_name = os.getenv("kb_collection_name")
        
        if not kb_name:
            return {
                "selected": False,
                "message": "No knowledge base selected"
            }
        
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.collection_name == kb_name).first()
        
        if not kb:  # KB_NAME exists but not in database
            return {
                "selected": True,
                "kb_name": kb_name,
                "kb_collection_name": kb_collection_name,
                "warning": "Knowledge base not found in database",
                "message": "Selected knowledge base may have been deleted"
            }
        
        return {
            "selected": True,
            "id": kb.id,
            "name": kb.name,
            "collection_name": kb.collection_name,
            "kb_collection_name": kb_collection_name,
            "description": kb.description
        }
    
    except Exception as e:
        logger.error(f"Get current knowledge base error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{kb_id}/files")
async def get_knowledge_base_files(kb_id: int, db: Session = Depends(get_db)):
    """Get all the files associated with a specific knowledge base.
    
    Args:
        kb_id: The knowledge base ID.
        
    Returns:
        A list of files associated with the knowledge base, including source information.
    """
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        
        # Source configuration, including MinIO files and QA files
        sources = db.query(KBSourceConfig).filter(
            KBSourceConfig.knowledge_base_id == kb_id,
            KBSourceConfig.source_type.in_(["minio_file", "qa_file"])
        ).all()
        
        files = []
        for source in sources:
            config = source.config or {}
            
            if source.source_type == "minio_file":
                object_name = config.get("object_name", source.source_identifier)
                display_name = config.get("display_name", object_name.split('/')[-1])
                
                files.append({
                    "id": source.id,
                    "name": display_name,
                    "path": object_name,
                    "type": "minio_file",
                    "file_type": config.get("file_type", "unknown")
                })
            
            elif source.source_type == "qa_file":
                object_name = config.get("object_name", source.source_identifier)
                display_name = config.get("display_name", object_name.split('/')[-1])
                sheet_name = config.get("sheet_name", "")
                
                files.append({
                    "id": source.id,
                    "name": f"{display_name} ({sheet_name})" if sheet_name else display_name,
                    "path": object_name,
                    "type": "qa_file",
                    "sheet_name": sheet_name
                })
        
        return files
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get knowledge base files error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-file")
async def select_file(source_ids: list[int] = Body(..., embed=True), db: Session = Depends(get_db)):
    """Select files (supports multiple selection) and download to local cache.

    Args:
        source_ids: A list of source configuration IDs.

    Returns:
        Selected file information list.
    """
    try:
        if not source_ids:
            logger.info("Cleared file selection")
            return {
                "files": [],
                "message": "File selection cleared",
                "timestamp": datetime.now().isoformat()
            }
        
        selected_files = []
        file_names = []

        for source_id in source_ids:
            source = db.query(KBSourceConfig).filter(KBSourceConfig.id == source_id).first()
            
            if not source:
                logger.warning(f"File with source_id {source_id} not found, skipping")
                continue
            
            config = source.config or {}
            downloaded = False  # A flag to indicate whether the file was downloaded
            
            if source.source_type == "minio_file":
                file_path = config.get("object_name", source.source_identifier)
                display_name = config.get("display_name", file_path.split('/')[-1])

                # Download the file if it is not in local
                if not minio_client.check_file_is_local(file_path):
                    logger.info(f"File not in local, downloading: {file_path}")
                    local_path = minio_client.download_file_to_local(file_path)
                    if not local_path:
                        logger.error(f"Failed to download file from MinIO: {file_path}")
                        continue
                    logger.info(f"File downloaded to: {local_path}")
                    downloaded = True
                
            elif source.source_type == "qa_file":
                file_path = config.get("object_name", source.source_identifier)
                display_name = config.get("display_name", file_path.split('/')[-1])
                
                # Download the file if it is not in local
                if not minio_client.check_file_is_local(file_path):
                    logger.info(f"File not in local, downloading: {file_path}")
                    local_path = minio_client.download_file_to_local(file_path)
                    if not local_path:
                        logger.error(f"Failed to download file from MinIO: {file_path}")
                        continue
                    logger.info(f"File downloaded to: {local_path}")
                    downloaded = True
                
            elif source.source_type == "database":
                file_path = config.get("connection_string", source.source_identifier)
                display_name = f"[DB] {config.get('table_name', '')}"
            else:
                file_path = source.source_identifier
                display_name = file_path
            
            file_names.append(file_path)

            # Locate the local path
            if source.source_type in ["minio_file", "qa_file"]:
                local_path = os.path.join(minio_client.tmp_dir, file_path)
            else:
                local_path = file_path

            selected_files.append({
                "id": source.id,
                "name": display_name,
                "path": file_path,
                "local_path": local_path,
                "type": source.source_type,
                "downloaded": downloaded
            })
        
        if not selected_files:
            raise HTTPException(status_code=404, detail="No valid files found")

        logger.info(f"Selected {len(selected_files)} file(s): {', '.join(file_names)}")

        return {
            "files": selected_files,
            "count": len(selected_files),
            "message": f"{len(selected_files)} file(s) selected successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Select file error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-file")
async def get_current_file(db: Session = Depends(get_db)):
    """Get the selected file (supports multiple selection).
    
    Returns:
        Selected file information list, empty list if none selected.
    """
    try:
        file_names_str = os.getenv("FILE_NAME")
        
        if not file_names_str:
            return {
                "selected": False,
                "files": [],
                "message": "No file selected"
            }
        
        # Split the file names string
        file_paths = [fp.strip() for fp in file_names_str.split(",") if fp.strip()]
        
        if not file_paths:
            return {
                "selected": False,
                "files": [],
                "message": "No file selected"
            }
        
        selected_files = []
        
        for file_path in file_paths:
            # Try to find the source by source_identifier
            source = db.query(KBSourceConfig).filter(
                KBSourceConfig.source_identifier == file_path
            ).first()
            
            # If not found, try to find by config.object_name
            if not source:
                sources = db.query(KBSourceConfig).all()
                for s in sources:
                    config = s.config or {}
                    if config.get("object_name") == file_path or config.get("connection_string") == file_path:
                        source = s
                        break
            
            if not source:
                selected_files.append({
                    "path": file_path,
                    "warning": "File not found in database",
                    "message": "Selected file may have been deleted"
                })
                continue
            
            config = source.config or {}
            display_name = config.get("display_name", file_path.split('/')[-1])
            
            selected_files.append({
                "id": source.id,
                "name": display_name,
                "path": file_path,
                "type": source.source_type
            })
        
        return {
            "selected": True,
            "files": selected_files,
            "count": len(selected_files)
        }
    
    except Exception as e:
        logger.error(f"Get current file error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
