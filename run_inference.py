#!/usr/bin/env python3
"""
Inference Pipeline Runner

This script runs the inference pipeline which:
1. Loads pre-built search indices
2. Starts the FastAPI server
3. Launches the Gradio interface
"""

import sys
import os
from pathlib import Path

def check_environment():
    """Check if all required files and indices exist."""
    required_files = [
        Path("config/config.sample.json"),
        Path(".env"),
    ]
    
    required_indices = [
        Path("data/whoosh_index"),
        Path("data/faiss_index/default__vector_store.faiss"),
    ]
    
    # Check files
    for file_path in required_files:
        if not file_path.is_file():
            print(f"Error: Required file {file_path} not found!")
            return False
            
    # Check indices
    for index_path in required_indices:
        if not index_path.exists():
            print(f"Error: Required index {index_path} not found!")
            print("Run the data preparation pipeline first: python run_data_preparation.py")
            return False
            
    return True

def main():
    """Main entry point for the inference pipeline."""
    print("Starting Inference Pipeline...")
    
    # Add the current directory to Python path
    current_dir = str(Path(__file__).parent.absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    try:
        from inference.main import main as pipeline_main
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