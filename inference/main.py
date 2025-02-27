"""
Main entry point for the inference (question-answering) pipeline.

This module implements a RAG (Retrieval-Augmented Generation) chatbot that:
1. Accepts user questions about documents in the knowledge base
2. Retrieves relevant context using both semantic (FAISS) and keyword (Whoosh) search
3. Uses an LLM (Large Language Model) to generate accurate answers
4. Provides source references for transparency

The system combines:
- Vector search (FAISS) for semantic similarity
- Keyword search (Whoosh) for exact matches
- Cross-encoder for result reranking
- LLM for natural language generation

Available interfaces:
1. REST API (via FastAPI) for programmatic access
2. Web UI (via Gradio) for interactive chat
"""

import logging  # Python's built-in logging facility
from pathlib import Path  # Object-oriented filesystem paths
import os  # Operating system interface
import threading  # Thread-based parallelism
import subprocess  # For running system commands
import sys  # For system-specific parameters and functions
import time  # For time-related functions

# External dependencies
import httpx  # Modern HTTP client with timeout support
import uvicorn  # ASGI server implementation
from fastapi import FastAPI  # Framework for building REST APIs
from sentence_transformers import CrossEncoder  # Model for reranking search results

# LlamaIndex components
from llama_index.core import (
    Settings,  # Global settings
    StorageContext,  # Manages index storage
    load_index_from_storage  # Loads saved indices
)
from llama_index.embeddings.nebius import NebiusEmbedding  # Text to vector conversion
from llama_index.llms.nebius import NebiusLLM  # Large Language Model interface
from llama_index.vector_stores.faiss import FaissVectorStore  # Vector similarity search

from whoosh.index import open_dir  # Keyword search engine

# Try package-level import first (for installed package)
# Fall back to local import (for development)
try:
    from inference.services import SearchService, LLMService  # Core services
    from inference.api.routes import router, initialize_routes  # API endpoints
    from inference.ui.gradio_interface import GradioInterface  # Gradio interface
    from common.config_utils import load_config, load_environment, setup_directories  # Configuration utilities
except ImportError:
    from services import SearchService, LLMService
    from api.routes import router, initialize_routes
    from ui.gradio_interface import GradioInterface
    from common.config_utils import load_config, load_environment, setup_directories

from common.logging_utils import setup_logging  # Logging configuration

# Initialize FastAPI app and include routes
app = FastAPI(title="Inference Pipeline API")
app.include_router(router)

def initialize_services(config, api_key, api_base):
    """
    Initialize all required services for the inference pipeline.
    
    This function sets up four main components:
    1. Embedding Model: Converts text to numerical vectors for semantic search
    2. Language Model (LLM): Generates natural language responses
    3. Search Indices: Both FAISS (vector) and Whoosh (keyword) indices
    4. Cross-Encoder: Reranks search results for better accuracy
    
    The function expects pre-existing search indices created by the
    data preparation pipeline. If indices are not found, it raises
    an error.
    
    Process Flow:
    1. Setup embedding model and LLM with API credentials
    2. Load and verify search indices
    3. Initialize cross-encoder for result reranking
    4. Create service instances
    
    Args:
        config (dict): Configuration settings including model parameters
            and directory paths
        api_key (str): API key for Nebius services (embedding and LLM)
        api_base (str): Base URL for Nebius API
        
    Returns:
        tuple: (SearchService, LLMService) - Initialized service instances
        
    Raises:
        FileNotFoundError: If required indices are not found
        Exception: For any other initialization errors
    """
    try:
        # Setup embedding model with extended timeout for large texts
        custom_http_client = httpx.Client(timeout=60.0)  # 60-second timeout
        embedding_model = NebiusEmbedding(
            api_key=api_key,
            model_name=config['embedding_model']['model_name'],
            http_client=custom_http_client,
            api_base=api_base,
        )
        Settings.embed_model = embedding_model  # Set as global embedding model

        # Setup Language Model with temperature control
        # Temperature controls randomness in generation (0.0 = deterministic)
        llm = NebiusLLM(
            api_key=api_key,
            model=config['llm']['model_name'],
            temperature=config['llm']['temperature'],
        )
        Settings.llm = llm  # Set as global LLM

        # Get paths to pre-built search indices
        whoosh_index_path = Path(config['directories']['whoosh_index_path'])
        faiss_index_path = Path(config['directories']['faiss_index_path'])

        # Verify that required indices exist
        if not whoosh_index_path.exists():
            raise FileNotFoundError(f"Whoosh index not found at {whoosh_index_path}")
        if not (faiss_index_path / "default__vector_store.faiss").exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")

        # Load Whoosh index for keyword search
        whoosh_index = open_dir(str(whoosh_index_path))
        logging.info("Whoosh index loaded successfully")

        # Load FAISS index for semantic search
        vector_store = FaissVectorStore.from_persist_path(
            str(faiss_index_path / "default__vector_store.faiss")
        )
        storage_context = StorageContext.from_defaults(
            persist_dir=str(faiss_index_path),
            vector_store=vector_store,
        )
        faiss_index = load_index_from_storage(storage_context, embedding=embedding_model)
        logging.info("FAISS index loaded successfully")

        # Initialize cross-encoder model for reranking search results
        cross_encoder = CrossEncoder(config['reranking']['cross_encoder_model'])

        # Create service instances that will handle business logic
        search_service = SearchService(
            faiss_index=faiss_index,  # For semantic search
            whoosh_index=whoosh_index,  # For keyword search
            cross_encoder=cross_encoder  # For result reranking
        )
        
        llm_service = LLMService(llm)  # For answer generation

        return search_service, llm_service

    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        raise

def check_and_kill_process_on_port(port):
    """
    Check if a process is using the specified port and kill it if found.
    
    Args:
        port (int): The port number to check
        
    Returns:
        bool: True if a process was killed, False otherwise
    """
    try:
        # Different commands for different operating systems
        if sys.platform.startswith('win'):
            # Windows command to find process using the port
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, 
                capture_output=True, 
                text=True
            )
            
            if result.stdout:
                # Extract PID from the output
                for line in result.stdout.strip().split('\n'):
                    if f':{port}' in line and ('LISTENING' in line or 'ESTABLISHED' in line):
                        parts = line.strip().split()
                        if len(parts) > 4:
                            pid = parts[-1]
                            logging.info(f"Found process with PID {pid} using port {port}")
                            
                            # Kill the process
                            kill_result = subprocess.run(
                                f'taskkill /F /PID {pid}',
                                shell=True,
                                capture_output=True,
                                text=True
                            )
                            
                            if kill_result.returncode == 0:
                                logging.info(f"Successfully terminated process with PID {pid}")
                                # Wait a moment for the port to be released
                                time.sleep(1)
                                return True
                            else:
                                logging.warning(f"Failed to terminate process: {kill_result.stderr}")
        else:
            # Linux/Mac command to find and kill process
            result = subprocess.run(
                f"lsof -i :{port} -t", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            
            if result.stdout:
                pid = result.stdout.strip()
                logging.info(f"Found process with PID {pid} using port {port}")
                
                # Kill the process
                kill_result = subprocess.run(
                    f"kill -9 {pid}",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                if kill_result.returncode == 0:
                    logging.info(f"Successfully terminated process with PID {pid}")
                    # Wait a moment for the port to be released
                    time.sleep(1)
                    return True
                else:
                    logging.warning(f"Failed to terminate process: {kill_result.stderr}")
                    
        return False
    except Exception as e:
        logging.error(f"Error checking/killing process on port {port}: {e}")
        return False

def main():
    """
    Main entry point for the inference pipeline application.
    
    This function:
    1. Loads configuration and environment settings
    2. Sets up logging infrastructure
    3. Initializes required directories
    4. Creates service instances
    5. Launches API and UI interfaces
    
    The application provides two interfaces:
    - FastAPI REST API on port 8082
    - Gradio web UI on port 7862
    
    Special handling:
    - In CI/CD environments (detected via CI env var), interfaces are not launched
    - FastAPI runs in a separate thread to not block the Gradio interface
    - All errors are logged before being re-raised
    
    Configuration:
    - Loads from config/config.json
    - Environment variables from .env file
    - Logging settings from config
    
    Error Handling:
    - All exceptions are caught, logged, and re-raised
    - Detailed error information is saved to log files
    """
    try:
        # Load configuration and environment settings
        config = load_config(Path("config/config.json"))
        api_key, api_base = load_environment()
        
        # Log API configuration (without exposing the key)
        logging.info(f"API Base URL: {api_base}")
        logging.info("API Key loaded: %s", "Yes" if api_key else "No")
        if not api_key:
            raise ValueError("API key not found in environment")
        if not api_base:
            raise ValueError("API base URL not found in environment")
        
        # Setup logging infrastructure with rotation
        logger = setup_logging(
            log_folder=Path(config['directories']['log_folder']),
            log_file=config['logging']['inference_log'],
            max_bytes=config['logging']['max_bytes'],
            backup_count=config['logging']['backup_count'],
            log_level=config['logging']['level'],
            log_format=config['logging']['format'],
            logger_name='inference'
        )
        
        # Create necessary directories if they don't exist
        setup_directories(config)

        # Initialize core services
        search_service, llm_service = initialize_services(config, api_key, api_base)

        # Setup FastAPI routes with service instances
        initialize_routes(search_service, llm_service)

        # Check if we're in a CI environment
        if os.getenv("CI") is None:
            # Preferred ports
            preferred_fastapi_port = 8082
            preferred_gradio_port = 7862
            
            # Try to kill any existing processes on our preferred ports
            check_and_kill_process_on_port(preferred_fastapi_port)
            check_and_kill_process_on_port(preferred_gradio_port)
            
            # Launch FastAPI with port retry
            fastapi_port = None
            for port in range(preferred_fastapi_port, preferred_fastapi_port + 10):
                try:
                    # Create the server without starting it
                    uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=port)
                    server = uvicorn.Server(uvicorn_config)
                    
                    # Start the server in a thread
                    uvicorn_thread = threading.Thread(
                        target=server.run
                    )
                    uvicorn_thread.daemon = True  # Make thread daemon so it exits when main thread exits
                    uvicorn_thread.start()
                    
                    fastapi_port = port
                    logging.info(f"FastAPI interface launched on port {port}")
                    break
                except Exception as e:
                    logging.warning(f"Port {port} is in use, trying next port")
                    if port == preferred_fastapi_port + 9:  # Last attempt
                        raise Exception(f"Could not find available port for FastAPI: {e}")
                    continue

            # Launch Gradio interface with port retry
            gradio_ui = GradioInterface(search_service, llm_service)
            gradio_port = None
            
            for port in range(preferred_gradio_port, preferred_gradio_port + 10):
                try:
                    gradio_ui.launch(
                        server_name="0.0.0.0",
                        server_port=port,
                        share=False
                    )
                    gradio_port = port
                    logging.info(f"Gradio interface launched on port {port}")
                    break
                except Exception as e:
                    logging.warning(f"Port {port} is in use, trying next port")
                    if port == preferred_gradio_port + 9:  # Last attempt
                        raise Exception(f"Could not find available port for Gradio: {e}")
                    continue
                    
            logging.info(f"Services running - FastAPI on port {fastapi_port}, Gradio on port {gradio_port}")
        else:
            logging.info("CI/CD environment detected. Skipping interface launch.")

    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        logging.exception("Detailed error information:")
        raise

if __name__ == "__main__":
    main() 