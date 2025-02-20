"""Common logging utilities for the RAG pipeline."""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logging(
    log_folder: Path,
    log_file: str = "app.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s",
    logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration with rotation capability.
    
    Args:
        log_folder: Directory where log files will be stored
        log_file: Name of the log file (default: app.log)
        max_bytes: Maximum size of each log file before rotation (default: 5MB)
        backup_count: Number of backup log files to keep (default: 3)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        logger_name: Optional name for the logger (defaults to root logger if None)
    
    Returns:
        Logger: Configured logger instance
    """
    try:
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
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging setup completed for {logger_name if logger_name else 'root'}")
        
        return logger
        
    except Exception as e:
        # Fallback to basic console logging if file-based logging fails
        print(f"Failed to configure file-based logging: {e}")
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        logger.handlers.clear()
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            force=True
        )
        logger.error(f"Falling back to basic console logging due to error: {e}")
        return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger, typically __name__ of the module
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 