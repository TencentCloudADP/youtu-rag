"""Configuration management routes, including YAML configuration and available config list"""
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from ..database import get_db, KnowledgeBase
from ..services.kb_config_service import KBConfigService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/filemanage-status")
async def check_fm_status():
    """
    Read from configs/rag/file_management.yaml

    Returns:
        Status of file_management configuration

    Example:
        ```
        GET /api/config/filemanage-status
        ```
    """
    try:
        config = KBConfigService.load_yaml_config("file_management")

        if not config:
            return {
                "ocr_enabled": False,
                "metadata_extraction_enabled": False,
                "message": "File management configuration not found"
            }

        ocr_config = config.get("ocr", {})
        # OCR is enabled only if enabled=true and model and base_url are not empty
        ocr_config_enabled = ocr_config.get("enabled", False)
        ocr_model = ocr_config.get("model", "")
        ocr_base_url = ocr_config.get("base_url", "")
        ocr_enabled = ocr_config_enabled and bool(ocr_model) and bool(ocr_base_url)

        metadata_config = config.get("metadata_extraction", {})
        metadata_extraction_enabled = metadata_config.get("enabled", True)

        return {
            "ocr_enabled": ocr_enabled,
            "ocr_config": {
                "enabled": ocr_enabled,
                "model": ocr_config.get("model"),
                "base_url": ocr_config.get("base_url")
            },
            "metadata_extraction_enabled": metadata_extraction_enabled,
            "metadata_config": {
                "enabled": metadata_extraction_enabled,
                "fields": metadata_config.get("fields", [])
            },
            "message": f"OCR: {'enabled' if ocr_enabled else 'disabled'}, Metadata extraction: {'enabled' if metadata_extraction_enabled else 'disabled'}"
        }

    except Exception as e:
        logger.error(f"Error checking OCR status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{kb_id_or_name}")
async def get_config(kb_id_or_name: str, db: Session = Depends(get_db)):
    """Get RAG configuration for a specific knowledge base.

    Supports both KB ID and KB name.
    It reads from configs/rag/{kb_name}.yaml or configs/rag/default.yaml and returns configurations of OCR, chunking, embedding, reranker, etc.
    
    Args:
        kb_id_or_name: knowledge base ID (str of an integer) or name (str)
        
    Returns:
        Config file in YAML format.
        
    Example:
        ```
        GET /api/config/12       # By ID
        GET /api/config/my_kb    # By name
        GET /api/config/default  # Get the default config
        ```
    """
    try:
        kb_name = None
        
        try:
            # Try to parse kb_id_or_name as KB ID, which is an integer.
            # Query the database for the knowledge base name.
            kb_id = int(kb_id_or_name)
            kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
            if kb:
                kb_name = kb.name
                logger.info(f"Found KB name '{kb_name}' for ID {kb_id}")
            else:
                # Fallback to default config if KB ID not found
                logger.warning(f"KB ID {kb_id} not found, using default config")
                kb_name = "default"
        except ValueError:
            # Treat as KB name
            kb_name = kb_id_or_name
        
        # Load YAML config
        config = KBConfigService.load_yaml_config(kb_name)
        
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration not found for KB '{kb_name}' and default config is missing"
            )
        
        # Add metadata
        from ..config import settings
        config_dir = settings.PROJECT_ROOT / "configs" / "rag"
        kb_config_path = config_dir / f"{kb_name}.yaml"
        
        config['_metadata'] = {
            'config_name': kb_name,
            'config_path': str(kb_config_path.relative_to(settings.PROJECT_ROOT)) if kb_config_path.exists() else f"configs/rag/default.yaml",
            'loaded_at': datetime.now(timezone.utc).isoformat(),
            'is_custom': kb_config_path.exists()
        }
        
        return config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading config for '{kb_id_or_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_available_configs():
    """List all the available RAG configurations.
    
    It scans the configs/rag/ directory for all .yaml files.

    Returns:
        List of available configurations.

    Example:
        ```
        GET /api/config
        ```
    """
    try:
        from ..config import settings
        config_dir = settings.PROJECT_ROOT / "configs" / "rag"

        if not config_dir.exists():
            return {
                "available_configs": [],
                "config_directory": str(config_dir.relative_to(settings.PROJECT_ROOT)),
                "message": "Config directory does not exist"
            }

        # Search for all .yaml files
        configs = []
        for yaml_file in config_dir.glob("*.yaml"):
            config_name = yaml_file.stem  # not including .yaml extension

            # skip files starting with underscore, which are private configs
            if config_name.startswith('_'):
                continue

            configs.append({
                "name": config_name,
                "path": str(yaml_file.relative_to(settings.PROJECT_ROOT)),
                "is_default": config_name == "default",
                "size": yaml_file.stat().st_size
            })

        # Sort by name, default first
        configs.sort(key=lambda x: (not x['is_default'], x['name']))

        return {
            "available_configs": configs,
            "config_directory": str(config_dir.relative_to(settings.PROJECT_ROOT)),
            "total_count": len(configs)
        }

    except Exception as e:
        logger.error(f"Error listing configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
