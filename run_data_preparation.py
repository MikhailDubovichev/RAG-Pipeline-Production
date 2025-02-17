#!/usr/bin/env python3
"""
Data Preparation Pipeline Runner

This script runs the data preparation pipeline which:
1. Processes documents from data/to_process/
2. Creates/updates search indices
3. Moves processed files to data/processed/
"""

import sys
import os
from pathlib import Path

def check_environment():
    """Check if all required directories and files exist."""
    required_files = [
        Path("config/config.sample.json"),
        Path(".env"),
    ]
    
    required_dirs = [
        Path("data/to_process"),
        Path("data/processed"),
        Path("logs"),
    ]
    
    # Check files
    for file_path in required_files:
        if not file_path.is_file():
            print(f"Error: Required file {file_path} not found!")
            return False
            
    # Create directories if they don't exist
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return True

def main():
    """Main entry point for the data preparation pipeline."""
    print("Starting Data Preparation Pipeline...")
    
    # Add the current directory to Python path
    current_dir = str(Path(__file__).parent.absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    try:
        from data_preparation.main import main as pipeline_main
        pipeline_main()
    except ImportError as e:
        print(f"Error: Failed to import pipeline components: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
        print("Also ensure you're running the script from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 