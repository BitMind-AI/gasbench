"""Shared logging utilities for standalone benchmark execution."""

import logging

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
