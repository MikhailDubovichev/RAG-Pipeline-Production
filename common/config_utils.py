"""Common configuration utilities for the RAG pipeline.

This module provides centralized configuration management for both data preparation
and inference pipelines. It handles:
1. Loading and validating JSON configuration
2. Managing environment variables
3. Setting up required directories
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        ConfigurationError: If configuration is invalid or missing required fields
    """
    try:
        with open(config_path, encoding='utf-8-sig') as config_file:
            config = json.load(config_file)
            
        # Validate required sections
        required_sections = [
            'directories', 'chunking', 'embedding_model', 
            'llm', 'reranking', 'logging'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required section: {section}")
                
        # Validate directories section
        required_dirs = [
            'data_directory', 'to_process_dir', 'processed_dir',
            'whoosh_index_path', 'faiss_index_path', 'log_folder'
        ]
        
        for dir_name in required_dirs:
            if dir_name not in config['directories']:
                raise ConfigurationError(f"Missing required directory config: {dir_name}")
                
        return config
        
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading config: {e}")

def load_environment() -> Tuple[str, str]:
    """
    Load and validate environment variables.
    
    Returns:
        Tuple of (api_key, api_base)
        
    Raises:
        ConfigurationError: If required environment variables are missing
    """
    # Load environment variables from .env file
    load_dotenv(dotenv_path=Path(".env"))
    
    # Get required environment variables
    api_key = os.getenv("API_KEY")
    api_base = os.getenv("API_BASE")
    
    if not api_key or not api_base:
        raise ConfigurationError("API_KEY and API_BASE must be set as environment variables")
        
    return api_key, api_base

def setup_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Create necessary directories from configuration.
    
    Args:
        config: Configuration dictionary containing directory paths
        
    Returns:
        Dictionary of Path objects for each directory
    """
    directories = {}
    
    for dir_name, dir_path in config['directories'].items():
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        directories[dir_name] = path
        
    return directories 