"""Knowledge base management routes"""
import logging
import os
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from ..database import get_db, KnowledgeBase, KBSourceConfig, KBBuildConfig
from ..models.knowledge_base import KnowledgeBaseCreate, KnowledgeBaseUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/list")
async def list_knowledge_bases(db: Session = Depends(get_db)):
    """List all knowledge bases"""
    try:
        knowledge_bases = db.query(KnowledgeBase).all()

        result = []
        for kb in knowledge_bases:
            # Count the number of documents (MinIO files) in this knowledge base
            document_count = db.query(KBSourceConfig).filter(
                KBSourceConfig.knowledge_base_id == kb.id,
                KBSourceConfig.source_type == "minio_file"
            ).count()

            # Count the number of database connections in this knowledge base
            database_count = db.query(KBSourceConfig).filter(
                KBSourceConfig.knowledge_base_id == kb.id,
                KBSourceConfig.source_type == "database"
            ).count()

            result.append({
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "collection_name": kb.collection_name,
                "document_count": document_count,
                "database_count": database_count,
                "created_at": kb.created_at.isoformat(),
                "updated_at": kb.updated_at.isoformat()
            })

        return result

    except Exception as e:
        logger.error(f"List knowledge bases error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
async def create_knowledge_base(
    kb_data: KnowledgeBaseCreate,
    db: Session = Depends(get_db)
):
    """Create a new knowledge base"""
    try:
        # Check if a knowledge base with this name already exists
        existing = db.query(KnowledgeBase).filter(KnowledgeBase.name == kb_data.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Knowledge base with this name already exists")

        # Use UUID to generate a unique collection name, avoiding issues with Chinese or special characters
        unique_id = uuid.uuid4().hex[:12]
        collection_name = f"kb_{unique_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create new knowledge base
        new_kb = KnowledgeBase(
            name=kb_data.name,
            description=kb_data.description,
            collection_name=collection_name
        )

        db.add(new_kb)
        db.commit()
        db.refresh(new_kb)

        logger.info(f"Created knowledge base: {new_kb.name} (ID: {new_kb.id})")

        return {
            "id": new_kb.id,
            "name": new_kb.name,
            "description": new_kb.description,
            "collection_name": new_kb.collection_name,
            "created_at": new_kb.created_at.isoformat(),
            "message": "Knowledge base created successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Create knowledge base error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{kb_id}")
async def get_knowledge_base(kb_id: int, db: Session = Depends(get_db)):
    """Get the detailed information and full configuration of the given knowledge base"""
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()

        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        files = []

        sources = db.query(KBSourceConfig).filter(
            KBSourceConfig.knowledge_base_id == kb_id
        ).all()

        # Separate sources by type
        selected_files = []
        selected_qa_files = []
        db_connections = []

        for source in sources:
            if source.source_type == "minio_file":
                selected_files.append(source.source_identifier)
            elif source.source_type == "qa_file":
                selected_qa_files.append(source.source_identifier)
            elif source.source_type == "database":
                config = source.config or {}
                conn_str = config.get("connection_string", "")

                db_type = config.get("db_type", "mysql")
                table_name = config.get("table_name", "")

                # Try to find existing connection or create new entry.
                # Note: Both connectionString and db_type must match, otherwise different types of databases will be merged.
                existing_conn = None
                for conn in db_connections:
                    if conn.get("connectionString") == conn_str and conn.get("type") == db_type:
                        existing_conn = conn
                        break

                if existing_conn:
                    if table_name and table_name not in existing_conn["tables"]:
                        existing_conn["tables"].append(table_name)
                else:
                    conn_data = {
                        "id": len(db_connections) + 1,
                        "type": db_type,
                        "connectionString": conn_str,
                        "tables": [table_name] if table_name else []
                    }

                    # Add MySQL/PostgreSQL specific fields (read from config)
                    if db_type != "sqlite":
                        conn_data["host"] = config.get("host", "")
                        conn_data["port"] = config.get("port", 3306)
                        conn_data["database"] = config.get("database", "")
                        conn_data["username"] = config.get("username", "")
                        conn_data["password"] = config.get("password", "")  # Include password for editing in the frontend
                    else:
                        conn_data["file_path"] = config.get("file_path", "")

                    db_connections.append(conn_data)

        # Get build config (tools)
        build_config = db.query(KBBuildConfig).filter(
            KBBuildConfig.knowledge_base_id == kb_id
        ).first()

        tools_config = build_config.tools_config if build_config else {}

        configuration = {
            "tools": tools_config,
            "selectedFiles": selected_files,
            "selectedQAFiles": selected_qa_files,
            "dbConnections": db_connections
        }

        return {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "collection_name": kb.collection_name,
            "files": files,  # Legacy field
            "document_count": len(files),
            "vector_count": 0,  # TODO: Get vector count from vector store
            "created_at": kb.created_at.isoformat(),
            "updated_at": kb.updated_at.isoformat(),
            "configuration": configuration
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get knowledge base error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{kb_id}/file-status")
async def get_knowledge_base_file_status(kb_id: int, db: Session = Depends(get_db)):
    """Get the status of all files in the given knowledge base.

    It returns a dictionary mapping file names to their status information, used for file selection interface and database table to display processing status.

    Args:
        kb_id: Knowledge base ID.

    Returns:
        Dictionary mapping file names/table names to status information.

    Example:
        GET /api/knowledge/1/file-status

        Return example:
        ```
        {
            "file1.pdf": {
                "status": "completed",
                "chunks_created": 150,
                "source_id": 15,
                "source_type": "minio_file",
                "error_message": null,
                "updated_at": "2025-12-25T10:30:00"
            },
            "qa_examples.xlsx": {
                "status": "completed",
                "chunks_created": 50,
                "source_id": 17,
                "source_type": "qa_file",
                "error_message": null,
                "updated_at": "2025-12-25T10:32:00"
            },
            "database:table_name": {
                "status": "completed",
                "chunks_created": 150,
                "source_id": 16,
                "source_type": "database",
                "error_message": null,
                "updated_at": "2025-12-25T10:25:00"
            }
        }
        ```
    """
    try:
        # Check if knowledge base exists
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        # Query all source configurations for the given knowledge base, including MinIO files, QA files and databases
        sources = db.query(KBSourceConfig).filter(
            KBSourceConfig.knowledge_base_id == kb_id
        ).all()

        status_map = {}
        for source in sources:
            if source.source_type == 'minio_file':
                # MinIO file: use source_identifier as key
                key = source.source_identifier
            elif source.source_type == 'qa_file':
                # QA file: use source_identifier as key
                key = source.source_identifier
            elif source.source_type == 'database':
                # database: use "database:table_name" as key
                import json
                config = json.loads(source.config) if isinstance(source.config, str) else source.config
                table_name = config.get('table_name', '')
                key = f"database:{table_name}"
            else:
                continue

            status_map[key] = {
                "status": source.status or "pending",
                "chunks_created": source.chunks_created or 0,
                "source_id": source.id,
                "source_type": source.source_type,
                "error_message": source.error_message,
                "updated_at": source.updated_at.isoformat() if source.updated_at else None
            }

        return status_map

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get file status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{kb_id}")
async def update_knowledge_base(
    kb_id: int,
    kb_data: KnowledgeBaseUpdate,
    db: Session = Depends(get_db)
):
    """Apply knowledge base update"""
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()

        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        # Update fields
        if kb_data.name is not None:
            # Check if new name is already in use
            existing = db.query(KnowledgeBase).filter(
                KnowledgeBase.name == kb_data.name,
                KnowledgeBase.id != kb_id
            ).first()
            if existing:
                raise HTTPException(status_code=400, detail="Knowledge base with this name already exists")
            kb.name = kb_data.name

        if kb_data.description is not None:
            kb.description = kb_data.description

        kb.updated_at = datetime.now()

        db.commit()
        db.refresh(kb)

        logger.info(f"Updated knowledge base: {kb.name} (ID: {kb.id})")

        return {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "collection_name": kb.collection_name,
            "updated_at": kb.updated_at.isoformat(),
            "message": "Knowledge base updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Update knowledge base error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{kb_id}")
async def delete_knowledge_base(kb_id: int, db: Session = Depends(get_db)):
    """Delete knowledge base, including vector data, Excel tables, mapping records, etc."""
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()

        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        kb_name = kb.name
        collection_name = kb.collection_name

        # 1. Clean up vector data, SQLite tables, mapping records, etc.
        try:
            from ...knowledge_builder.cleanup_manager import KnowledgeCleanupManager
            from ...storage.implementations.chroma_store import ChromaVectorStore
            from ...config import VectorStoreConfig
            from ..config import settings

            vector_config = VectorStoreConfig(
                collection_name=collection_name,
                persist_directory=settings.chroma_persist_directory
            )
            vector_store = ChromaVectorStore(config=vector_config)

            cleanup_manager = KnowledgeCleanupManager(
                vector_store=vector_store,
                relational_db_path=settings.relational_db_path
            )

            cleanup_stats = await cleanup_manager.cleanup_knowledge_base(kb_id, collection_name)

            logger.info(
                f"Cleanup stats for KB {kb_id}: "
                f"vectors_deleted={cleanup_stats.get('total_vectors_deleted', 0)}, "
                f"tables_deleted={cleanup_stats.get('total_tables_deleted', 0)}"
            )

        except Exception as cleanup_error:
            # Record error when clean up fails, but still delete the record from the database
            logger.error(f"Cleanup error for KB {kb_id}: {cleanup_error}", exc_info=True)

        # 2. Delete the record from the database
        db.query(KBSourceConfig).filter(KBSourceConfig.knowledge_base_id == kb_id).delete()

        db.query(KBBuildConfig).filter(KBBuildConfig.knowledge_base_id == kb_id).delete()

        db.delete(kb)
        db.commit()

        logger.info(f"Deleted knowledge base: {kb_name} (ID: {kb_id})")

        return {
            "message": f"Knowledge base '{kb_name}' deleted successfully",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Delete knowledge base error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{kb_id}/qa/{source_file}")
async def get_qa_associations(
    kb_id: int,
    source_file: str,
    db: Session = Depends(get_db)
):
    """Get the QA associations for a knowledge base and source file."""
    import sqlite3
    from ..config import settings

    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        conn = sqlite3.connect(settings.relational_db_path)
        conn.row_factory = sqlite3.Row  # Allow accessing columns by name

        try:
            cursor = conn.execute(
                """
                SELECT qa_id, kb_id, question, answer, howtofind, source_file,
                       learning_status, memory_status, created_at, updated_at
                FROM qa_associations
                WHERE kb_id = ? AND source_file = ?
                ORDER BY qa_id
                """,
                (kb_id, source_file)
            )

            rows = cursor.fetchall()

            qa_list = []
            for row in rows:
                qa_list.append({
                    "id": row["qa_id"],
                    "kb_id": row["kb_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "howtofind": row["howtofind"],
                    "source_file": row["source_file"],
                    "learning_status": row["learning_status"] or "pending",
                    "memory_status": row["memory_status"] or "pending",
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                })

            return {"qa_list": qa_list}

        finally:
            conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get QA associations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{kb_id}/qa/{qa_id}/status")
async def update_qa_status(
    kb_id: int,
    qa_id: int,
    status_data: dict,
    db: Session = Depends(get_db)
):
    """Update the learning status of a QA."""
    import sqlite3
    from ..config import settings

    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        learning_status = status_data.get("learning_status")
        if learning_status not in ["pending", "learning", "completed", "failed"]:
            raise HTTPException(status_code=400, detail="Invalid learning status")

        conn = sqlite3.connect(settings.relational_db_path)

        try:
            conn.execute(
                """
                UPDATE qa_associations
                SET learning_status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE qa_id = ? AND kb_id = ?
                """,
                (learning_status, qa_id, kb_id)
            )
            conn.commit()

            if conn.total_changes == 0:
                raise HTTPException(status_code=404, detail="QA not found")

            return {"message": "Status updated successfully"}

        finally:
            conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update QA status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{kb_id}/qa/{qa_id}/execute")
async def execute_qa(
    kb_id: int,
    qa_id: int,
    db: Session = Depends(get_db)
):
    """Execute a single QA.
    
    It calls LLM to select appropriate Agents and execute the QA.
    """
    import sqlite3
    import yaml
    from pathlib import Path
    from ..config import settings

    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        conn = sqlite3.connect(settings.relational_db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Get QA data
            cursor = conn.execute(
                """
                SELECT qa_id, question, answer, howtofind
                FROM qa_associations
                WHERE qa_id = ? AND kb_id = ?
                """,
                (qa_id, kb_id)
            )

            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="QA not found")

            question = row["question"]
            expected_answer = row["answer"]
            howtofind = row["howtofind"]

            # Update the status as "learning"
            conn.execute(
                """
                UPDATE qa_associations
                SET learning_status = 'learning', updated_at = CURRENT_TIMESTAMP
                WHERE qa_id = ? AND kb_id = ?
                """,
                (qa_id, kb_id)
            )
            conn.commit()

            # Read Agent config from frontend_agents.yaml
            config_path = Path(settings.PROJECT_ROOT) / "configs" / "rag" / "frontend_agents.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                agents_config = yaml.safe_load(f)

            available_agents = agents_config.get('agents', [])

            # Select Agent using LLM
            selected_agents = await select_agents_for_qa(
                question=question,
                howtofind=howtofind,
                available_agents=available_agents
            )

            # Execute selected Agents
            execution_results = []
            for agent_name in selected_agents:
                try:
                    result = await execute_agent_for_qa(
                        agent_name=agent_name,
                        question=question,
                        kb_id=kb_id,
                        agents_config=available_agents
                    )
                    execution_results.append({
                        "agent": agent_name,
                        "success": True,
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Agent {agent_name} execution failed: {str(e)}")
                    execution_results.append({
                        "agent": agent_name,
                        "success": False,
                        "error": str(e)
                    })

            # Check execution results
            success_count = sum(1 for r in execution_results if r.get("success"))
            final_status = "completed" if success_count > 0 else "failed"

            # Update the final status
            conn.execute(
                """
                UPDATE qa_associations
                SET learning_status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE qa_id = ? AND kb_id = ?
                """,
                (final_status, qa_id, kb_id)
            )
            conn.commit()

            env_memory_enabled = os.environ.get("memoryEnabled", "false").lower() == "true"
            if env_memory_enabled:
                has_answer = expected_answer and expected_answer.strip()
                has_howtofind = howtofind and howtofind.strip()

                if has_answer:
                    try:
                        from ..config import settings
                        from ....tools.memory_toolkit import VectorMemoryToolkit

                        memory_toolkit = VectorMemoryToolkit(
                            persist_directory=settings.memory_store_path,
                            collection_prefix="rag_chat",
                            default_user_id="default_user",
                            max_working_memory_turns=10000,
                        )
                        if has_howtofind:
                            memory_answer = f"answer: {expected_answer.strip()}\nhowtofind: {howtofind.strip()}"
                        else:
                            memory_answer = f"answer: {expected_answer.strip()}"

                        await memory_toolkit.save_conversation_to_episodic(
                            question=question.strip(),
                            answer=memory_answer,
                            importance_score=0.5,
                        )

                        conn.execute(
                            """
                            UPDATE qa_associations
                            SET memory_status = 'memorized', updated_at = CURRENT_TIMESTAMP
                            WHERE qa_id = ? AND kb_id = ?
                            """,
                            (qa_id, kb_id)
                        )
                        conn.commit()
                        logger.info(f"💾 [QA Execute] Saved QA {qa_id} to memory")
                    except Exception as e:
                        # Update memory_status to failed
                        conn.execute(
                            """
                            UPDATE qa_associations
                            SET memory_status = 'failed', updated_at = CURRENT_TIMESTAMP
                            WHERE qa_id = ? AND kb_id = ?
                            """,
                            (qa_id, kb_id)
                        )
                        conn.commit()
                        logger.error(f"❌ Failed to save QA {qa_id} to memory: {str(e)}")
                else:
                    logger.warning(f"⚠️ [QA Execute] QA {qa_id} missing expected_answer or howtofind, skipped memory storage")

            return {
                "message": "QA executed successfully",
                "qa_id": qa_id,
                "selected_agents": selected_agents,
                "execution_results": execution_results,
                "final_status": final_status
            }

        finally:
            conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execute QA error: {str(e)}")
        # Update the status as "failed"
        try:
            conn = sqlite3.connect(settings.relational_db_path)
            conn.execute(
                """
                UPDATE qa_associations
                SET learning_status = 'failed', updated_at = CURRENT_TIMESTAMP
                WHERE qa_id = ? AND kb_id = ?
                """,
                (qa_id, kb_id)
            )
            conn.commit()
            conn.close()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{kb_id}/qa/batch-execute")
async def batch_execute_qa(
    kb_id: int,
    batch_data: dict,
    db: Session = Depends(get_db)
):
    """Execute a batch of QA."""
    import sqlite3
    from ..config import settings

    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        qa_ids = batch_data.get("qa_ids", [])
        if not qa_ids:
            raise HTTPException(status_code=400, detail="No QA IDs provided")

        # Execute one by one
        results = []
        for qa_id in qa_ids:
            try:
                result = await execute_qa(kb_id, qa_id, db)
                results.append({
                    "qa_id": qa_id,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Failed to execute QA {qa_id}: {str(e)}")
                results.append({
                    "qa_id": qa_id,
                    "success": False,
                    "error": str(e)
                })

        success_count = sum(1 for r in results if r.get("success"))

        return {
            "message": f"Batch executed {len(qa_ids)} QA items",
            "total": len(qa_ids),
            "success": success_count,
            "failed": len(qa_ids) - success_count,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch execute QA error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for agent selection and execution

async def select_agents_for_qa(question: str, howtofind: str, available_agents: list) -> list:
    """Select appropriate agents using LLM"""
    from utu.utils.openai_utils.simplified_client import SimplifiedAsyncOpenAI
    from ..config import settings
    import json

    # Construct Agent descriptions
    agents_desc = "\n".join([
        f"- {agent['name']}: {agent['description']}"
        for agent in available_agents
    ])

    # Read Agent selection prompt from YAML
    config_path = Path(settings.PROJECT_ROOT) / "configs" / "rag" / "frontend_agents.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        agents_config = yaml.safe_load(f)

    agent_selection_config = agents_config.get('agent_selection', {})
    prompt_template = agent_selection_config.get('selection_prompt', '')

    prompt = prompt_template.format(
        question=question,
        howtofind=howtofind or '无',
        agents_desc=agents_desc
    )

    try:
        logger.info("=" * 80)
        logger.info("LLM Agent Selection Request:")
        logger.info(f"Question: {question}")
        logger.info(f"HowToFind: {howtofind or '无'}")
        logger.info(f"Prompt:\n{prompt}")
        logger.info("=" * 80)

        client = SimplifiedAsyncOpenAI()

        content = await client.query_one(
            messages=[{"role": "user", "content": prompt}]
        )

        logger.info("=" * 80)
        logger.info("LLM Agent Selection Response:")
        logger.info(f"Raw Response:\n{content}")
        logger.info("=" * 80)

        # Parse the response
        from ..utils import parse_agent_selection_response
        selected_agents = parse_agent_selection_response(content, available_agents)

        logger.info(f"Final Selected Agents: {selected_agents}")
        logger.info("=" * 80)
        return selected_agents

    except Exception as e:
        logger.error(f"Failed to select agents: {str(e)}")
        # Fallback to KB Search
        return ["KB Search"]


async def execute_agent_for_qa(agent_name: str, question: str, kb_id: int, agents_config: list) -> dict:
    """Execute the give Agent."""
    from ..main import get_memory_toolkit
    from ..dependencies import _create_agent_instance
    from utu.config import ConfigLoader

    agent_config = None
    for agent in agents_config:
        if agent['name'] == agent_name:
            agent_config = agent
            break

    if not agent_config:
        raise ValueError(f"Agent {agent_name} not found in configuration")

    memory_toolkit = get_memory_toolkit()
    if not memory_toolkit:
        raise ValueError("Memory toolkit not initialized")

    request_data = {
        "query": question,
        "stream": False,
        "session_id": None,
        "kb_id": kb_id,
        "file_ids": None,
        "use_memory": True
    }

    logger.info(f"Executing agent {agent_name} for question: {question}")

    try:
        agent_object_type = agent_config.get('agent_object', 'SimpleAgent')
        config_path = agent_config.get('config_path')

        logger.info(f"Loading agent: {agent_name} (type: {agent_object_type}, config: {config_path})")

        # Load Agent config and create an instance
        if agent_object_type == "ExcelAgent":
            # ExcelAgent requires the full path to the config file
            from ..config import settings
            full_config_path = str(settings.PROJECT_ROOT / "configs" / "agents" / config_path)
            agent = _create_agent_instance(agent_object_type, full_config_path)
        else:
            # Other agents use ConfigLoader to load config
            config = ConfigLoader.load_agent_config(config_path)
            agent = _create_agent_instance(agent_object_type, config)
            # SimpleAgent needs to build
            if hasattr(agent, 'build'):
                await agent.build()

        # Inject memory toolkit
        agent.set_memory_toolkit(memory_toolkit)

        # Execute the agent using ChatService
        from ..services.chat_service import ChatService

        chat_service = ChatService(agent)

        response = await chat_service.get_response(
            query=question,
            session_id=None,  # session is not used in QA execution
            kb_id=kb_id,
            file_ids=None
        )

        return {
            "agent": agent_name,
            "question": question,
            "answer": response.answer,
            "memory_saved": True
        }

    except Exception as e:
        logger.error(f"Failed to execute agent {agent_name}: {str(e)}", exc_info=True)
        raise

