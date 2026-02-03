import logging
import sys
from datetime import datetime
from typing import Optional

__all__ = ["logger", "setup_logger", "progress"]

# ANSI color codes for colored output
COLORS = {
    'DEBUG': '\033[0;36m',    # Cyan
    'INFO': '\033[0;32m',     # Green
    'WARNING': '\033[0;33m',  # Yellow
    'ERROR': '\033[0;31m',    # Red
    'CRITICAL': '\033[0;35m', # Magenta
    'RESET': '\033[0m'        # Reset color
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter that colors the entire log line based on level."""
    def format(self, record):
        formatted = super().format(record)
        color = COLORS.get(record.levelname)
        if color:
            return f"{color}{formatted}{COLORS['RESET']}"
        return formatted

def setup_logger(name: str = "youtu-graphrag", 
                level: int = logging.INFO,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup and return a logger instance with colored output
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path to save logs
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicate logs
    logger.handlers.clear()
    
    # Avoid duplicate logs from propagating to root logger
    # This prevents an extra line like "INFO:logger-name:message" from root handlers
    logger.propagate = False

    # Create console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format: [Time] LevelName Module:Line - Message
    formatter = ColoredFormatter(
        fmt='[%(asctime)s] %(levelname)-8s %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        # File handler without colors
        file_formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)-8s %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger instance
logger = setup_logger()

def progress(stage: str, message: str, *, done: Optional[bool] = None):
    """Unified progress logging helper.
    Args:
        stage: Short stage/category name
        message: Detail message
        done: Optional flag mark completion (prints ✅/❌)
    """
    suffix = ""
    if done is True:
        suffix = " ✅"
    elif done is False:
        suffix = " ❌"
    logger.info(f"[{stage}] {message}{suffix}")

# Usage example:
if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")