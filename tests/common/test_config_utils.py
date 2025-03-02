"""
Unit tests for the config_utils module.

This module tests the functionality of the config_utils module, which is responsible for:
1. Loading configuration from JSON files
2. Loading environment variables
3. Setting up directories
4. Validating configuration

The tests use pytest and include:
- Configuration loading
- Environment variable handling
- Directory setup
- Error handling
"""

import os
import json
import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Import the module to test
from common import config_utils

class TestConfigUtils:
    """Test cases for the config_utils module."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir_path = Path(self.temp_dir)
        
        # Create a test config file
        self.config_path = self.test_dir_path / "config.json"
        self.test_config = {
            "directories": {
                "to_process_dir": str(self.test_dir_path / "input"),
                "processed_dir": str(self.test_dir_path / "processed"),
                "log_folder": str(self.test_dir_path / "logs"),
                "whoosh_index_path": str(self.test_dir_path / "whoosh"),
                "faiss_index_path": str(self.test_dir_path / "faiss"),
                "data_directory": str(self.test_dir_path / "data")
            },
            "chunking": {
                "max_tokens": 1000,
                "overlap_tokens": 200
            },
            "embedding_model": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "llm": {
                "model_name": "gpt-3.5-turbo"
            },
            "reranking": {
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "threshold": 0.2
            },
            "logging": {
                "data_preparation_log": "data_prep.log",
                "inference_log": "inference.log",
                "max_bytes": 10485760,
                "backup_count": 3,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # Create the directories
        os.makedirs(self.test_dir_path / "input", exist_ok=True)
        os.makedirs(self.test_dir_path / "processed", exist_ok=True)
        os.makedirs(self.test_dir_path / "logs", exist_ok=True)
        os.makedirs(self.test_dir_path / "whoosh", exist_ok=True)
        os.makedirs(self.test_dir_path / "faiss", exist_ok=True)
        os.makedirs(self.test_dir_path / "data", exist_ok=True)
        
        # Write the config file
        with open(self.config_path, "w") as f:
            json.dump(self.test_config, f)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Close all loggers
        import logging
        for logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        
        # Close root logger handlers
        for handler in list(logging.getLogger().handlers):
            handler.close()
            logging.getLogger().removeHandler(handler)
        
        # Small delay to ensure file handles are released
        time.sleep(0.1)
        
        # Clean up the temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except (PermissionError, OSError) as e:
            # If we can't remove the whole directory, try to remove files individually
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except:
                        pass
                for dir in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir))
                    except:
                        pass
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
    
    def test_load_config(self):
        """Test loading configuration from a file."""
        # Call the function
        config = config_utils.load_config(self.config_path)
        
        # Check that the config was loaded correctly
        assert config["directories"]["to_process_dir"] == str(self.test_dir_path / "input")
        assert config["chunking"]["max_tokens"] == 1000
        assert config["embedding_model"]["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    
    @patch("builtins.open", side_effect=Exception("Test exception"))
    def test_load_config_error(self, mock_file):
        """Test handling of errors when loading configuration."""
        # Check that an exception is raised
        with pytest.raises(config_utils.ConfigurationError) as excinfo:
            config_utils.load_config(self.config_path)
        
        # Check the error message
        assert "Error loading config" in str(excinfo.value)
    
    @patch.dict(os.environ, {"API_KEY": "test_key", "API_BASE": "test_base"})
    def test_load_environment(self):
        """Test loading environment variables."""
        # Call the function
        api_key, api_base = config_utils.load_environment()
        
        # Check that the environment variables were loaded correctly
        assert api_key == "test_key"
        assert api_base == "test_base"
    
    @patch.dict(os.environ, {"API_KEY": "test_key"}, clear=True)
    @patch('common.config_utils.load_dotenv')
    def test_load_environment_missing(self, mock_load_dotenv):
        """Test handling of missing environment variables."""
        # Mock load_dotenv to do nothing
        mock_load_dotenv.return_value = None
        
        # Check that an exception is raised when API_BASE is missing
        with pytest.raises(config_utils.ConfigurationError) as excinfo:
            config_utils.load_environment()
        
        # Check the error message
        assert "API_KEY and API_BASE" in str(excinfo.value)
    
    def test_setup_directories(self):
        """Test setting up directories from configuration."""
        # Call the function
        directories = config_utils.setup_directories(self.test_config)
        
        # Check that the directories were created and returned
        assert directories["to_process_dir"] == Path(self.test_dir_path / "input")
        assert directories["processed_dir"] == Path(self.test_dir_path / "processed")
        assert directories["log_folder"] == Path(self.test_dir_path / "logs")
        assert directories["whoosh_index_path"] == Path(self.test_dir_path / "whoosh")
        assert directories["faiss_index_path"] == Path(self.test_dir_path / "faiss")
        assert directories["data_directory"] == Path(self.test_dir_path / "data") 