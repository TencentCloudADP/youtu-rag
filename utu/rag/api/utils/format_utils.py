"""Utils for formatting content."""


def format_content(content: str) -> str:
    """Ensure at most one trailing newline in the content.
    
    Args:
        content: Original content.
        
    Returns:
        Formatted content.
    """
    if not content:
        return content
    
    formatted = content.strip('\n')
    if content.endswith('\n'):
        formatted += '\n'
    
    return formatted
