"""Agent-related utility functions."""
import json
import logging

logger = logging.getLogger(__name__)


def parse_agent_selection_response(content: str, available_agents: list, fallback_agent: str = "KB Search") -> list:
    """Parse LLM response for agent selection.
    
    This function handles the parsing of LLM responses that contain agent selections,
    typically returned as JSON arrays. It cleans up markdown code blocks and validates
    the selected agents against available agents.
    
    Args:
        content: Raw LLM response string
        available_agents: List of available agent configs, each containing 'name' key
        fallback_agent: Default agent to use if parsing fails or no valid agents found
        
    Returns:
        List of validated agent names
        
    Examples:
        >>> available = [{'name': 'KB Search'}, {'name': 'SQL Agent'}]
        >>> parse_agent_selection_response('["KB Search", "SQL Agent"]', available)
        ['KB Search', 'SQL Agent']
        
        >>> parse_agent_selection_response('```json\\n["KB Search"]\\n```', available)
        ['KB Search']
    """
    try:
        # Strip whitespace
        content = content.strip()
        
        # Remove markdown code block markers
        if content.startswith("```"):
            # Remove opening ``` and first line
            content = content.split("\n", 1)[1]
            # Remove closing ```
            content = content.rsplit("```", 1)[0]
        
        # Remove JSON code block markers
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        selected_agents = json.loads(content)
        
        # Validate selected agents exist in available agents
        valid_agent_names = [agent['name'] for agent in available_agents]
        selected_agents = [name for name in selected_agents if name in valid_agent_names]
        
        # Fallback if no valid agents selected
        if not selected_agents:
            logger.warning(f"No valid agents selected, falling back to {fallback_agent}")
            selected_agents = [fallback_agent]
        
        return selected_agents
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse agent selection response: {str(e)}, content: {content}")
        # Fallback to default agent
        return [fallback_agent]
