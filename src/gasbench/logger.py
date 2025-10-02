"""Shared logging utilities for standalone benchmark execution."""

import logging
from typing import Optional, Dict, Any

# Initialize logging configuration once
_logging_initialized = False

def init_logging():
    """Initialize Python logging configuration."""
    global _logging_initialized
    if not _logging_initialized:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        _logging_initialized = True

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    # Ensure logging is initialized
    init_logging()
    return logging.getLogger(name)

class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that adds context to all log messages."""
    
    def __init__(self, logger, extra: Optional[Dict[str, Any]] = None):
        """Initialize with logger and optional extra context."""
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        # Get context from extra dict
        context_parts = []
        
        # Common context fields
        for field in ['miner_uid', 'media_type', 'request_id', 'model_hash']:
            value = self.extra.get(field)
            if value:
                context_parts.append(f"[{value.upper() if isinstance(value, str) else value}]")
        
        # Build prefix
        prefix = "".join(context_parts) if context_parts else ""
        
        if prefix:
            return f"{prefix} {msg}", kwargs
        return msg, kwargs
    
    def update_context(self, **kwargs):
        """Update the logger context."""
        self.extra.update(kwargs)
