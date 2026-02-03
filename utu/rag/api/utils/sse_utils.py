"""Server-Sent Events utils."""
import json


def format_sse_event(data: dict) -> str:
    """Format data as a Server-Sent Event (SSE).
    
    Args:
        data: Event data dictionary to send.
        
    Returns:
        SSE formatted string.
    """
    return f"data: {json.dumps(data)}\n\n"
