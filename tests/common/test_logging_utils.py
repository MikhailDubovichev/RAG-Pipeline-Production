"""
Unit tests for the logging_utils module.

This module tests the functionality of the logging_utils module, which is responsible for:
1. Setting up logging with appropriate handlers
2. Configuring log rotation
3. Setting log levels
4. Formatting log messages

The tests use pytest and include:
- Logger setup
- Log file creation
- Log rotation
- Log level configuration
"""

import os
import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import time

# Import the module to test
from common import logging_utils

class TestLoggingUtils:
    """Test suite for the logging_utils module."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for log files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)
        
        # Reset the root logger before each test
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Close all log handlers to release file locks
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
            
        # Close any other loggers that might have been created
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
                
        # Give a small delay to ensure file handles are released
        time.sleep(0.1)
        
        # Clean up the temporary directory
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")
            # Try to clean up individual files that might be causing issues
            import shutil
            try:
                for root, dirs, files in os.walk(self.log_dir, topdown=False):
                    for file in files:
                        try:
                            os.unlink(os.path.join(root, file))
                        except:
                            pass
                    for dir in dirs:
                        try:
                            shutil.rmtree(os.path.join(root, dir))
                        except:
                            pass
            except:
                pass
    
    def test_setup_logging_file_creation(self):
        """Test that setup_logging creates the log file."""
        # Set up parameters
        log_file = "test.log"
        log_path = self.log_dir / log_file
        
        # Call the function
        logger = logging_utils.setup_logging(
            log_folder=self.log_dir,
            log_file=log_file,
            max_bytes=1024,
            backup_count=3,
            log_level="INFO",
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            logger_name="test_logger"
        )
        
        # Log a message to ensure the file is created
        logger.info("Test message")
        
        # Check that the log file was created
        assert log_path.exists()
        
        # Check that the log file contains the test message
        with open(log_path, "r") as f:
            log_content = f.read()
            assert "Test message" in log_content
    
    def test_setup_logging_log_level(self):
        """Test that setup_logging sets the correct log level."""
        # Set up parameters
        log_file = "test.log"
        
        # Call the function with DEBUG level
        logger = logging_utils.setup_logging(
            log_folder=self.log_dir,
            log_file=log_file,
            max_bytes=1024,
            backup_count=3,
            log_level="DEBUG",
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            logger_name="test_logger"
        )
        
        # Check that the logger has the correct level
        assert logger.level == logging.DEBUG
        
        # Call the function with INFO level
        logger = logging_utils.setup_logging(
            log_folder=self.log_dir,
            log_file=log_file,
            max_bytes=1024,
            backup_count=3,
            log_level="INFO",
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            logger_name="test_logger_info"
        )
        
        # Check that the logger has the correct level
        assert logger.level == logging.INFO
    
    def test_setup_logging_rotation(self):
        """Test that setup_logging configures log rotation correctly."""
        # Set up parameters
        log_file = "test.log"
        log_path = self.log_dir / log_file
        
        # Call the function with small max_bytes to trigger rotation
        logger = logging_utils.setup_logging(
            log_folder=self.log_dir,
            log_file=log_file,
            max_bytes=50,  # Very small to trigger rotation
            backup_count=3,
            log_level="INFO",
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            logger_name="test_logger"
        )
        
        # Log multiple messages to trigger rotation
        for i in range(10):
            logger.info(f"Test message {i} with some extra text to make it longer")
        
        # Check that backup files were created
        assert log_path.exists()
        assert (self.log_dir / f"{log_file}.1").exists()
        
        # Check that we don't have more than backup_count + 1 files
        log_files = list(self.log_dir.glob(f"{log_file}*"))
        assert len(log_files) <= 4  # Original + 3 backups
    
    def test_setup_logging_format(self):
        """Test that setup_logging sets the correct log format."""
        # Set up parameters
        log_file = "test.log"
        log_path = self.log_dir / log_file
        test_format = "%(levelname)s: %(message)s"
        
        # Call the function with a custom format
        logger = logging_utils.setup_logging(
            log_folder=self.log_dir,
            log_file=log_file,
            max_bytes=1024,
            backup_count=3,
            log_level="INFO",
            log_format=test_format,
            logger_name="test_logger"
        )
        
        # Log a message
        logger.info("Test message")
        
        # Check that the log file contains the message in the correct format
        with open(log_path, "r") as f:
            log_content = f.read()
            assert "INFO: Test message" in log_content
    
    def test_setup_logging_console_handler(self):
        """Test that setup_logging adds a console handler."""
        # Call the function
        logger = logging_utils.setup_logging(
            log_folder=self.log_dir,
            log_file="test.log",
            max_bytes=1024,
            backup_count=3,
            log_level="INFO",
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            logger_name="test_logger"
        )
        
        # Check that the logger has at least one StreamHandler
        has_stream_handler = any(
            isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
            for handler in logger.handlers
        )
        assert has_stream_handler
    
    def test_setup_logging_file_handler(self):
        """Test that setup_logging adds a file handler."""
        # Call the function
        logger = logging_utils.setup_logging(
            log_folder=self.log_dir,
            log_file="test.log",
            max_bytes=1024,
            backup_count=3,
            log_level="INFO",
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            logger_name="test_logger"
        )
        
        # Check that the logger has at least one RotatingFileHandler
        has_file_handler = any(
            isinstance(handler, logging.handlers.RotatingFileHandler)
            for handler in logger.handlers
        )
        assert has_file_handler
    
    def test_setup_logging_existing_logger(self):
        """Test that setup_logging works with an existing logger."""
        # Create a logger
        existing_logger = logging.getLogger("existing_logger")
        
        # Call the function with the existing logger name
        logger = logging_utils.setup_logging(
            log_folder=self.log_dir,
            log_file="test.log",
            max_bytes=1024,
            backup_count=3,
            log_level="INFO",
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            logger_name="existing_logger"
        )
        
        # Check that the returned logger is the same as the existing one
        assert logger is existing_logger
        
        # Check that the logger has handlers
        assert len(logger.handlers) > 0 