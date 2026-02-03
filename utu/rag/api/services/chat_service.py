"""chat service"""
import logging
import os
import sys
from typing import AsyncGenerator, Optional
from datetime import datetime
from pathlib import Path

from ..models.chat import ChatResponse, StreamEventType, WorkflowStepType
from ..utils.sse_utils import format_sse_event
from ..utils.format_utils import format_content
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from utu.agents.orchestrator.parallel_common import ParallelOrchestratorEvent
from ..database import get_db, KBSourceConfig

# 导入 ExcelAgentStreamEvent（保持与 dependencies.py 一致的导入方式）
project_root = Path(__file__).parent.parent.parent.parent
adg_path = project_root / "integrations" / "ADG" / "src" / "model" / "adg_benchmark"
if str(adg_path) not in sys.path:
    sys.path.insert(0, str(adg_path))

try:
    from excel_agent import ExcelAgentStreamEvent
except ImportError:
    ExcelAgentStreamEvent = None

logger = logging.getLogger(__name__)


class ChatService:
    """Service interface for chat"""

    def __init__(self, agent):
        self.agent = agent

    def _get_file_names_from_ids(self, file_ids: list[str]) -> list[str]:
        """Get list of file names from source config IDs.

        Args:
            file_ids: Source config ID list (integers in string format, e.g. ["15", "16"]).

        Returns:
            File name list (object_name, e.g. ["file1.pdf", "file2.txt"]).
        """
        if not file_ids:
            return []

        db = next(get_db())
        try:
            file_names = []
            for file_id_str in file_ids:
                try:
                    file_id = int(file_id_str)
                    source = db.query(KBSourceConfig).filter(KBSourceConfig.id == file_id).first()
                    if source:
                        config = source.config or {}
                        # Get the name by either object_name or source_identifier
                        file_name = config.get("object_name", source.source_identifier)
                        file_names.append(file_name)
                        logger.debug(f"Resolved file_id {file_id} -> {file_name}")
                    else:
                        logger.warning(f"Source config ID {file_id} not found in database")
                except ValueError:
                    logger.warning(f"Invalid file_id format: {file_id_str}")
            return file_names
        finally:
            db.close()
    
    async def get_response(
        self,
        query: str,
        session_id: Optional[str] = None,
        kb_id: Optional[int] = None,
        file_ids: Optional[list[str]] = None
    ) -> ChatResponse:
        """Get non-streaming response."""
        # Modify the query to inject KB parameters
        query = self.modify_query(query, kb_id=kb_id, file_ids=file_ids)

        result = self.agent.run_streamed(query)

        # Consume all the streaming events
        async for _ in result.stream_events():
            pass

        return ChatResponse(
            answer=result.final_output or "No response generated",
            session_id=session_id,
            workflow_steps=[]
        )
    
    async def stream_response(
        self,
        query: str,
        session_id: Optional[str] = None,
        kb_id: Optional[int] = None,
        file_ids: Optional[list[str]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        try:
            # Send start event
            yield format_sse_event({
                "type": StreamEventType.START,
                "message": "Agent started processing your query",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Workflow state
            workflow_steps = []
            current_step = None
            step_counter = 0
            
            # Content buffers
            reasoning_buffer = ""
            output_buffer = ""
            current_tool_name = None

            # Modify the query according to the agent's function
            query = self.modify_query(query, kb_id=kb_id, file_ids=file_ids)
            
            result = self.agent.run_streamed(query)
            
            async for event in result.stream_events():
                agent_name = getattr(event, 'agent_name', None)

                if isinstance(event, RawResponsesStreamEvent):
                    event_data = event.data

                    if event_data.type == "response.reasoning_text.delta":  # Reasoning or thinking
                        formatted_delta = format_content(event_data.delta)
                        reasoning_buffer += formatted_delta
                        logger.debug(f"[{event_data.type}]: {formatted_delta}")
                        sse_data = {
                            "type": StreamEventType.REASONING,
                            "content": formatted_delta,
                            "timestamp": datetime.now().isoformat()
                        }
                        if agent_name:
                            sse_data["agent_name"] = agent_name
                        yield format_sse_event(sse_data)
                    
                    elif event_data.type == "response.reasoning_text.done":  # Reasoning completion
                        logger.debug(f"[{event_data.type}] Reasoning completed")
                        sse_data = {
                            "type": StreamEventType.DONE,
                            "content": "",
                            "done": True,
                            "terminate_card": True,
                            "timestamp": datetime.now().isoformat()
                        }
                        if agent_name:
                            sse_data["agent_name"] = agent_name
                        yield format_sse_event(sse_data)
                    
                    elif event_data.type == "response.output_item.added":  # Tool call
                        if event_data.item.type == "function_call":
                            current_tool_name = event_data.item.name
                            logger.debug(f"[{event_data.type}] - [{event_data.item.type}]: ({event_data.item.name}) {event_data.item.arguments}")
                            step_counter += 1
                            current_step = {
                                "id": step_counter,
                                "type": WorkflowStepType.TOOL_CALL,
                                "name": event_data.item.name,
                                "status": "running",
                                "start_time": datetime.now().isoformat()
                            }
                            workflow_steps.append(current_step)

                            tool_name = event_data.item.name
                            if "code" in tool_name or "sql" in tool_name:
                                mode = "code"
                            else:
                                mode = "json"
                            
                            sse_data = {
                                "type": StreamEventType.TOOL_CALL,
                                "tool_name": event_data.item.name,
                                "arguments": event_data.item.arguments,
                                "mode": mode,
                                "timestamp": datetime.now().isoformat()
                            }
                            if agent_name:
                                sse_data["agent_name"] = agent_name
                            yield format_sse_event(sse_data)
                            
                            # yield format_sse_event({
                            #     "type": StreamEventType.WORKFLOW_UPDATE,
                            #     "workflow_steps": workflow_steps,
                            #     "timestamp": datetime.now().isoformat()
                            # })
                    
                    elif event_data.type == "response.function_call_arguments.delta":  # Incremental tool call arguments
                        if current_step:
                            formatted_delta = format_content(event_data.delta)
                            logger.debug(f"[{event_data.type}]: {formatted_delta}")
                            if "code" in current_tool_name or "sql" in current_tool_name:
                                mode = "code"
                            else:
                                mode = "json"
                            sse_data = {
                                "type": StreamEventType.TOOL_CALL,
                                "tool_name": current_tool_name,
                                "arguments_delta": formatted_delta,
                                "mode": mode,
                                "timestamp": datetime.now().isoformat()
                            }
                            if agent_name:
                                sse_data["agent_name"] = agent_name
                            yield format_sse_event(sse_data)
                    
                    elif event_data.type == "response.function_call_arguments.done":  # Function call completion
                        if current_step:
                            current_step["status"] = "completed"
                            current_step["end_time"] = datetime.now().isoformat()
                            logger.debug(f"[{event_data.type}] Tool call completed: {current_tool_name}")
                            
                            sse_data = {
                                "type": StreamEventType.TOOL_CALL,
                                "tool_name": current_tool_name,
                                "done": True,
                                "timestamp": datetime.now().isoformat()
                            }
                            if agent_name:
                                sse_data["agent_name"] = agent_name
                            yield format_sse_event(sse_data)
                            
                            yield format_sse_event({
                                "type": StreamEventType.WORKFLOW_UPDATE,
                                "workflow_steps": workflow_steps,
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    elif event_data.type == "response.output_item.done":  # Tool output
                        if hasattr(event_data, 'item') and event_data.item.type == "function_call_output":
                            tool_output = format_content(getattr(event_data.item, 'output', ''))
                            logger.debug(f"[{event_data.type}] Tool output: {tool_output[:100]}...")
                            
                            sse_data = {
                                "type": StreamEventType.TOOL_OUTPUT,
                                "tool_name": current_tool_name,
                                "output": tool_output,
                                "timestamp": datetime.now().isoformat()
                            }
                            if agent_name:
                                sse_data["agent_name"] = agent_name
                            yield format_sse_event(sse_data)
                        else:
                            logger.debug(f"[{event_data.type}] Output item done, terminate current card")
                            sse_data = {
                                "type": StreamEventType.DONE,
                                "content": "",
                                "terminate_card": True,
                                "timestamp": datetime.now().isoformat()
                            }
                            if agent_name:
                                sse_data["agent_name"] = agent_name
                            yield format_sse_event(sse_data)

                    elif event_data.type == "response.output_text.delta":  # Incremental text output (final output)
                        formatted_delta = format_content(event_data.delta)
                        output_buffer += formatted_delta
                        logger.debug(f"[{event_data.type}]: {formatted_delta}")
                        sse_data = {
                            "type": StreamEventType.DELTA,
                            "content": formatted_delta,
                            "timestamp": datetime.now().isoformat()
                        }
                        if agent_name:
                            sse_data["agent_name"] = agent_name
                        yield format_sse_event(sse_data)

                    elif event_data.type == "response.output_text.done":  # Output completion
                        logger.debug(f"[{event_data.type}] Output completed")
                        sse_data = {
                            "type": StreamEventType.DONE,
                            "content": "",
                            "done": True,
                            "terminate_card": True,
                            "timestamp": datetime.now().isoformat()
                        }
                        if agent_name:
                            sse_data["agent_name"] = agent_name
                        yield format_sse_event(sse_data)

                    elif event_data.type == "response.content_part.done":  # Content part completion
                        # Terminate current card and start a new one
                        logger.debug(f"[{event_data.type}] Content part completed, terminate current card")
                        sse_data = {
                            "type": StreamEventType.DONE,
                            "content": "",
                            "terminate_card": True,
                            "timestamp": datetime.now().isoformat()
                        }
                        if agent_name:
                            sse_data["agent_name"] = agent_name
                        yield format_sse_event(sse_data)
                    
                    elif event_data.type == "custom.tool_log":  # Custom tool log
                        from utu.utils import PrintUtils
                        logger.debug(f"[{event_data.type}]: {PrintUtils.truncate_text(event_data.message)}")
                        sse_data = {
                            "type": StreamEventType.TOOL_LOG,
                            "message": event_data.message,
                            "tool_name": getattr(event_data, 'tool_name', 'unknown'),
                            "timestamp": datetime.now().isoformat()
                        }
                        if agent_name:
                            sse_data["agent_name"] = agent_name
                        yield format_sse_event(sse_data)
                
                elif isinstance(event, RunItemStreamEvent):
                    item = event.item
                    item_type = item.type
                    logger.debug(f"[RunItemStreamEvent] item_type: {item_type}")

                    item_data = {
                        "type": StreamEventType.RUN_ITEM,
                        "item_type": item_type,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Add agent_name (if in parallel mode)
                    if agent_name:
                        item_data["agent_name"] = agent_name
                    
                    if item_type == "message_output_item":
                        continue
                    
                    elif item_type == "reasoning_item":
                        if hasattr(item, 'summary'):
                            summary_text = ""
                            if isinstance(item.summary, list):
                                for s in item.summary:
                                    if isinstance(s, dict) and s.get('type') == 'summary_text':
                                        summary_text += s.get('text', '')
                            item_data["reasoning_summary"] = summary_text
                            logger.debug(f"[RunItemStreamEvent] reasoning_item: {summary_text[:100]}...")
                            yield format_sse_event(item_data)
                    
                    elif item_type == "tool_call_item":
                        continue
                    
                    elif item_type == "tool_call_output_item":
                        output = getattr(item, 'output', '')
                        tool_output_text = ''
                        
                        if isinstance(output, dict):
                            tool_output_text = output.get('message', '') or str(output)
                        elif isinstance(output, str):
                            tool_output_text = output
                        else:
                            tool_output_text = str(output)
                        
                        item_data["tool_output"] = tool_output_text
                        logger.debug(f"[RunItemStreamEvent] tool_call_output_item: {tool_output_text[:100]}...")
                        yield format_sse_event(item_data)
                    
                    elif item_type == "handoff_call_item":
                        if hasattr(item, 'raw_item'):
                            item_data["handoff_name"] = getattr(item.raw_item, 'name', 'unknown')
                            item_data["handoff_arguments"] = getattr(item.raw_item, 'arguments', '')
                            logger.debug(f"[RunItemStreamEvent] handoff_call_item: {item_data['handoff_name']}")
                            yield format_sse_event(item_data)
                    
                    elif item_type == "handoff_output_item":
                        if hasattr(item, 'source_agent') and hasattr(item, 'target_agent'):
                            item_data["source_agent"] = item.source_agent.name if item.source_agent else 'unknown'
                            item_data["target_agent"] = item.target_agent.name if item.target_agent else 'unknown'
                            logger.debug(f"[RunItemStreamEvent] handoff_output_item: {item_data['source_agent']} -> {item_data['target_agent']}")
                            yield format_sse_event(item_data)
                    
                    else:
                        item_data["raw_info"] = str(item)
                        logger.debug(f"[RunItemStreamEvent] unknown item_type: {item_type}")
                        yield format_sse_event(item_data)
                
                elif ExcelAgentStreamEvent and isinstance(event, ExcelAgentStreamEvent):
                    logger.debug(f"[ExcelAgentStreamEvent] name: {event.name}")
                    
                    if event.name == "excel_agent.plan.start":  # Analysis starts
                        yield format_sse_event({
                            "type": StreamEventType.DELTA,
                            "content": f"正在解析 Excel 表格 {event.item.get('file_path', '')}\n构建 Meta Graph...",
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    elif event.name == "excel_agent.plan.delta":  # Analysis in progress
                        yield format_sse_event({
                            "type": StreamEventType.DELTA,
                            "content": f"\n{event.item.get("content", "")}",
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    elif event.name == "excel_agent.plan.done":  # Analysis completes
                        yield format_sse_event({
                            "type": StreamEventType.DELTA,
                            "done": True,
                            "timestamp": datetime.now().isoformat()
                        })

                    elif event.name == "excel_agent.task.start":  # Excel Agent starts processing
                        yield format_sse_event({
                            "type": StreamEventType.EXCEL_AGENT_EVENT,
                            "title": f"Task: {event.item.get('operation', '')}",
                            "content": "Generating ...",
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    elif event.name == "excel_agent.task.delta":  # Excel Agent in process
                        if event.item.get("type") == "code_generation":
                            content = f"```python\n{event.item.get("content", "")}```"
                        elif event.item.get("type") == "code_execution":
                            content = f"\n---\n执行结果：\n{event.item.get("content", "")}"
                        else:
                            content = event.item.get("content", "")
                        yield format_sse_event({
                            "type": StreamEventType.EXCEL_AGENT_EVENT,
                            "title": f"Task: {event.item.get('operation', '')}",
                            "content": content,
                            "clean": event.item.get("clean", False),
                            "mode": event.item.get("mode", "text"),
                            "timestamp": datetime.now().isoformat()
                        })

                    elif event.name == "excel_agent.task.done":  # Excel Agent processing completes
                        yield format_sse_event({
                            "type": StreamEventType.EXCEL_AGENT_EVENT,
                            "title": f"Task: {event.item.get('operation', '')}",
                            "done": True,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    elif event.name == "excel_agent.answer.start":  # Excel Agent starts to respond
                        yield format_sse_event({
                            "type": StreamEventType.DELTA,
                            "content": f"正在生成回答...\n",
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    elif event.name == "excel_agent.answer.delta":  # Excel Agent in response
                        yield format_sse_event({
                            "type": StreamEventType.DELTA,
                            "content": f"{event.item.get("content", "")}",
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    elif event.name == "excel_agent.answer.done":  # Excel Agent response completes
                        yield format_sse_event({
                            "type": StreamEventType.DELTA,
                            "done": True,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    else:
                        raise ValueError(f"Unknown event name: {event.name}")

                elif isinstance(event, ParallelOrchestratorEvent):
                    logger.debug(f"[ParallelOrchestratorEvent] name: {event.name}")

                    # Serialize the event and send it to the frontend
                    event_dict = event.to_dict()
                    logger.debug(f"[ParallelOrchestratorEvent] Forwarding event: {event_dict}")
                    yield format_sse_event(event_dict)

            # Send the final response
            final_output = result.final_output or "No response generated"
            
            yield format_sse_event({
                "type": StreamEventType.DONE,
                "final_output": final_output,
                "workflow_steps": workflow_steps,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
        
        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            yield format_sse_event({
                "type": StreamEventType.ERROR,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    def modify_query(
        self,
        query: str,
        kb_id: Optional[int] = None,
        file_ids: Optional[list[str]] = None
    ) -> str:
        """Modify the query based on the agent's functionality, injecting necessary context information."""

        if hasattr(self.agent, '__class__') and self.agent.__class__.__name__ == "ExcelAgent":
            # For Excel Agent, load the files and set the environment variable FILE_PATH.
            # Excel Agent reads file path from environment variable FILE_PATH.
            if kb_id is not None and file_ids:
                # Download files from knowledge base into temporary directory
                file_names = self._get_file_names_from_ids(file_ids)
                if file_names:
                    from ..minio_client import minio_client

                    downloaded_paths = []
                    for file_name in file_names:
                        try:
                            # Reusing cached file if available; otherwise download
                            if minio_client.check_file_is_local(file_name):
                                local_path = os.path.join(minio_client.tmp_dir, file_name)
                                downloaded_paths.append(local_path)
                                logger.info(f"Using cached file: {local_path}")
                            else:
                                local_path = minio_client.download_file_to_local(file_name)
                                if local_path:
                                    downloaded_paths.append(local_path)
                                    logger.info(f"Downloaded KB file to: {local_path}")
                                else:
                                    logger.error(f"Failed to download file: {file_name}")
                        except Exception as e:
                            logger.error(f"Error downloading {file_name}: {e}")

                    if downloaded_paths:
                        file_path_str = ",".join(downloaded_paths)
                        os.environ["FILE_PATH"] = file_path_str
                        logger.info(f"ExcelAgent: Set FILE_PATH={file_path_str}")
                        return query
                    else:
                        raise ValueError("⚠️ Failed to download any files from knowledge base.")
            else:
                raise ValueError("⚠️ ExcelAgent requires kb_id and file_ids parameters.")


        agent_name = self.agent.config.agent.name

        if agent_name in ["kb-search-agent"]:
            # For KB Search Agent, inject kb_id and file_ids
            if kb_id is None:
                raise ValueError("⚠️ KB Search Agent requires kb_id parameter. Please select a knowledge base.")

            # Construct the prompt
            kb_context = f"[Knowledge Base ID: {kb_id}]"
            if file_ids:
                file_names = self._get_file_names_from_ids(file_ids)  # From knowledge base

                if file_names:
                    # Prompt Agent to use metadata_filters to filter by these files -- using file names, not IDs
                    if len(file_names) == 1:
                        filter_hint = f'{{"source": "{file_names[0]}"}}'
                    else:
                        files_str = ', '.join([f'"{f}"' for f in file_names])
                        filter_hint = f'{{"source": {{"$in": [{files_str}]}}}}'
                    kb_context += f"\n[IMPORTANT: When calling kb_embedding_search or kb_file_search, use metadata_filters={filter_hint} to filter by these files]"
                    logger.info(f"Resolved {len(file_ids)} file IDs to {len(file_names)} file names: {file_names}")

            modified_query = f"{kb_context}\n\nUser Question: {query}"
            logger.info(f"Agent [{agent_name}] Modified query with KB context (kb_id={kb_id}, files={len(file_ids) if file_ids else 0})")
            return modified_query

        elif agent_name in ["file_qa"]:
            # For File QA Agent, inject file_names of the selected files.
            # File QA Agent will automatically discover files if not provided.
            context_parts = []

            if kb_id is not None:
                context_parts.append(f"[Knowledge Base ID: {kb_id}]")

                if file_ids:
                    file_names = self._get_file_names_from_ids(file_ids)  # From knowledge base
                    if file_names:
                        context_parts.append(f"[Selected Files: {', '.join(file_names)}]")
                        logger.info(f"File QA: Using KB files - kb_id={kb_id}, files={file_names}")
                else:
                    # Fallback to auto file discovery using kb_file_search
                    logger.info(f"File QA: Using KB with auto file discovery - kb_id={kb_id}")
            else:
                raise ValueError("⚠️ File QA Agent requires kb_id parameter.")

            modified_query = "\n".join(context_parts) + f"\n\nUser Question: {query}"
            logger.info(f"Agent [{agent_name}] Modified query: {modified_query[:250]}...")
            return modified_query

        elif agent_name in ["ParallelOrchestratorAgent"]:
            # For Parallel Orchestrator Agent, inject knowledge base (kb_id) and files (file_names).
            context_parts = []

            if kb_id is not None and file_ids:
                file_names = self._get_file_names_from_ids(file_ids)  # From knowledge base
                if file_names:
                    context_parts.append(f"[Knowledge Base ID: {kb_id}]")
                    context_parts.append(f"[Selected Files: {', '.join(file_names)}]")
                    logger.info(f"ParallelOrchestrator: Using KB files - kb_id={kb_id}, files={file_names}")

            if context_parts:
                modified_query = "\n".join(context_parts) + f"\n\nUser Question: {query}"
                logger.info(f"Agent [{agent_name}] Modified query with context")
                return modified_query
            else:
                # Returns original query if no context is available
                return query

        else:
            return query

