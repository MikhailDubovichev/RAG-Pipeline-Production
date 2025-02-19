"""Common logging utilities for the RAG pipeline."""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging(
    log_folder: Path,
    log_file: str,
    max_bytes: int,
    backup_count: int,
    log_level: str,
    log_format: str,
    logger_name: str = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_folder: Directory where log files will be stored
        log_file: Name of the log file
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        logger_name: Optional name for the logger (defaults to root logger if None)
    
    Returns:
        Logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)
    
    # Get or create logger
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Setup file handler with rotation
    log_path = log_folder / log_file
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging setup completed for {logger_name if logger_name else 'root'}")
    
    return logger 