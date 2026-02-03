"""Chat-related routes"""
import logging
import os
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..models.chat import ChatRequest, ChatResponse
from ..dependencies import get_agent
from ..services.chat_service import ChatService
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("")
async def chat(request: ChatRequest):
    """Endpoint for chat.
    
    Supports both streaming and non-streaming modes, except that auto select mode only supports streaming mode.

    Supported streaming event types:
    - start: start of a session
    - delta: chunk of Agent response
    - thinking: Agent reasoning process
    - tool_call: tool invocation information
    - workflow_update: workflow visualization update
    - done: complete response and final output
    - error: error message
    """
    try:
        logger.info(f"Received chat request: {request.query[:100]}...")

        # Auto select mode
        if request.auto_select:
            logger.info("Using Auto Select mode")
            if request.stream:
                return StreamingResponse(
                    auto_select_and_stream(request),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    }
                )
            else:
                # Non-streaming is not supported for auto select mode
                raise HTTPException(status_code=400, detail="Auto select only supports streaming mode")

        # Default mode: using the currently selected agent
        agent = await get_agent()

        # ADDED: Check environment variable to decide whether to enable Memory
        env_memory_enabled = os.environ.get("memoryEnabled", "false").lower() == "true"
        logger.info(f"Memory enabled from env: {env_memory_enabled}")

        if env_memory_enabled:
            # If enabling memory, create and inject VectorMemoryToolkit
            try:
                from utu.tools.memory_toolkit import VectorMemoryToolkit

                memory_toolkit = VectorMemoryToolkit(
                    persist_directory=settings.memory_store_path,
                    collection_prefix="rag_chat",
                    default_user_id="default_user",
                    max_working_memory_turns=10000,
                )

                # Use session_id from request if provided; otherwise create new.
                # This is to distinguish between multiple users and multiple sessions.
                if request.session_id:
                    memory_toolkit.current_session_id = request.session_id
                else:
                    memory_toolkit.start_session()

                if hasattr(agent, 'set_memory_toolkit'):
                    agent.set_memory_toolkit(memory_toolkit)
                    logger.info(f"Memory toolkit injected, session: {memory_toolkit.current_session_id}")
                else:
                    logger.warning("Agent does not support set_memory_toolkit")

            except ImportError as e:
                logger.warning(f"VectorMemoryToolkit not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to create memory toolkit: {e}")
        chat_service = ChatService(agent)

        if request.stream:
            return StreamingResponse(
                chat_service.stream_response(
                    request.query,
                    request.session_id,
                    request.kb_id,
                    request.file_ids
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            response = await chat_service.get_response(
                request.query,
                request.session_id,
                request.kb_id,
                request.file_ids
            )
            return response

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def auto_select_and_stream(request: ChatRequest):
    """
    Auto select mode -- automatically select agents -- supporting only streaming mode.
    """
    import json
    import yaml
    from pathlib import Path
    from ..config import settings
    from ..dependencies import _create_agent_instance
    from utu.config import ConfigLoader
    from utu.utils.openai_utils.simplified_client import SimplifiedAsyncOpenAI

    try:
        # 1. Read configurations from frontend_agents.yaml
        config_path = Path(settings.PROJECT_ROOT) / "configs" / "rag" / "frontend_agents.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            agents_config = yaml.safe_load(f)

        available_agents = agents_config.get('agents', [])
        agent_selection_config = agents_config.get('agent_selection', {})
        prompt_template = agent_selection_config.get('selection_prompt', '')

        # 2. Create descriptions of Agent
        agents_desc = "\n".join([
            f"- {agent['name']}: {agent['description']}"
            for agent in available_agents
        ])

        # 3. Construct the prompt for selecting Agent
        prompt = prompt_template.format(
            question=request.query,
            howtofind='无',  # howtofind is not supported for chat mode
            agents_desc=agents_desc
        )

        # 4. Send UI feedback
        yield f"data: {json.dumps({'type': 'start', 'content': '🤖 正在智能选择Agent...'}, ensure_ascii=False)}\n\n"

        # 5. Select Agent using LLM
        client = SimplifiedAsyncOpenAI()
        content = await client.query_one(
            messages=[{"role": "user", "content": prompt}]
        )

        # 6. Parse the response
        from ..utils import parse_agent_selection_response
        selected_agents = parse_agent_selection_response(content, available_agents)
        
        logger.info(f"Auto selected agents: {selected_agents}")

        # 7. Notify user of selected agents
        yield f"data: {json.dumps({'type': 'delta', 'content': f'\\n\\n📋 已选择Agent: {', '.join(selected_agents)}\\n\\n'}, ensure_ascii=False)}\n\n"

        # 8. Execute selected agents
        for agent_name in selected_agents:
            agent_config = None
            for agent in available_agents:
                if agent['name'] == agent_name:
                    agent_config = agent
                    break

            if not agent_config:
                logger.warning(f"Agent {agent_name} not found in configuration")
                continue

            yield f"data: {json.dumps({'type': 'delta', 'content': f'\\n🔄 执行 {agent_name}...\\n'}, ensure_ascii=False)}\n\n"

            try:
                # Dynamically create agent instance
                agent_object_type = agent_config.get('agent_object', 'SimpleAgent')
                config_path = agent_config.get('config_path')

                if agent_object_type == "ExcelAgent":
                    full_config_path = str(settings.PROJECT_ROOT / "configs" / "agents" / config_path)
                    agent = _create_agent_instance(agent_object_type, full_config_path)
                else:
                    config = ConfigLoader.load_agent_config(config_path)
                    agent = _create_agent_instance(agent_object_type, config)
                    if hasattr(agent, 'build'):
                        await agent.build()

                # Inject memory toolkit
                env_memory_enabled = os.environ.get("memoryEnabled", "false").lower() == "true"
                if env_memory_enabled:
                    try:
                        from utu.tools.memory_toolkit import VectorMemoryToolkit
                        memory_toolkit = VectorMemoryToolkit(
                            persist_directory=settings.memory_store_path,
                            collection_prefix="rag_chat",
                            default_user_id="default_user",
                            max_working_memory_turns=10000,
                        )
                        if request.session_id:
                            memory_toolkit.current_session_id = request.session_id
                        else:
                            memory_toolkit.start_session()

                        if hasattr(agent, 'set_memory_toolkit'):
                            agent.set_memory_toolkit(memory_toolkit)
                    except Exception as e:
                        logger.warning(f"Failed to inject memory toolkit: {e}")

                chat_service = ChatService(agent)

                # Stream the response
                async for chunk in chat_service.stream_response(
                    request.query,
                    request.session_id,
                    request.kb_id,
                    request.file_ids
                ):
                    yield chunk

                yield f"data: {json.dumps({'type': 'delta', 'content': f'\\n✅ {agent_name} 执行完成\\n'}, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"Failed to execute agent {agent_name}: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'content': f'❌ {agent_name} 执行失败: {str(e)}'}, ensure_ascii=False)}\n\n"

        # 9. Notify user of completion
        yield f"data: {json.dumps({'type': 'done', 'content': ''}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"Auto select error: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': f'Auto select failed: {str(e)}'}, ensure_ascii=False)}\n\n"

