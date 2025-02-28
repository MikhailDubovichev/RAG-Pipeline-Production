"""
Main entry point for the data preparation pipeline.

This module serves as the core of the data preparation process in the RAG pipeline.
It provides both API endpoints (via FastAPI) and a user interface (via Gradio) for:
1. Uploading documents (PDF, Excel, PowerPoint, Word)
2. Processing these documents into chunks
3. Creating searchable indices (Whoosh for keyword search, FAISS for semantic search)

The pipeline follows these main steps:
1. Document upload (via API or UI)
2. Document processing (chunking and text extraction)
3. Index creation/updating
4. Moving processed files to archive
"""

import logging  # Python's built-in logging facility for tracking events
import os  # Operating system interface for file and path operations
import shutil  # Shell utilities for high-level file operations (copy, move)
from pathlib import Path  # Object-oriented filesystem paths
import httpx  # Modern HTTP client with timeout support

# Web framework and file handling
from fastapi import FastAPI, UploadFile  # FastAPI for creating REST APIs
from fastapi.responses import JSONResponse  # JSON response formatting
import uvicorn  # ASGI server implementation
import gradio as gr  # Framework for creating web UI
import threading  # Thread-based parallelism

# LlamaIndex components for document indexing
from llama_index.embeddings.nebius import NebiusEmbedding  # Nebius embedding model
from llama_index.core import Settings  # Global settings for LlamaIndex

# Local imports from our pipeline
from .services import DocumentProcessor, IndexingService, FileService  # Core services
from .ui.gradio_interface import GradioInterface  # Gradio interface
from .api.routes import router, initialize_routes  # API routes
from common.config_utils import load_config, load_environment, setup_directories  # Configuration utilities
from common.logging_utils import setup_logging  # Logging configuration

# Initialize FastAPI app with title
app = FastAPI(title="Data Preparation Pipeline API")
app.include_router(router)

# Global service instances
# These will be initialized when the application starts
config = None  # Configuration settings
doc_processor = None  # Handles document loading and chunking
indexing_service = None  # Manages search indices (Whoosh and FAISS)
file_service = None  # Handles file operations (moving, tracking)

def initialize_services(config_data, api_key, api_base):
    """
    Initialize all required services for the data preparation pipeline.
    
    This function sets up three core services:
    1. Document Processor: Handles document loading and text chunking
    2. Indexing Service: Creates and updates search indices
    3. File Service: Manages file operations and tracking
    
    It also configures the embedding model (Nebius) which converts text into
    numerical vectors for semantic search.
    
    Args:
        config_data (dict): Configuration settings including model parameters and directories
        api_key (str): API key for accessing the Nebius embedding service
        api_base (str): Base URL for the Nebius API
        
    Returns:
        tuple: (DocumentProcessor, IndexingService, FileService) - Initialized service instances
        
    Example config_data structure:
    {
        'embedding_model': {'model_name': 'model_name'},
        'chunking': {'max_tokens': 500, 'overlap_tokens': 50},
        'directories': {
            'to_process_dir': 'path/to/input',
            'processed_dir': 'path/to/output'
        }
    }
    """
    # Setup embedding model with extended timeout
    custom_http_client = httpx.Client(timeout=60.0)  # This timeout is just an "ugly hack" - without argument httpx.client() didn't
    embedding_model = NebiusEmbedding(
        api_key=api_key,
        model_name=config_data['embedding_model']['model_name'],
        http_client=custom_http_client,
        api_base=api_base,
    )
    Settings.embed_model = embedding_model  # Set as global embedding model

    # Initialize document processor with chunking parameters
    document_processor = DocumentProcessor(
        max_tokens=config_data['chunking']['max_tokens'],  # Maximum tokens per chunk
        overlap_tokens=config_data['chunking']['overlap_tokens']  # Overlap between chunks
    )
    
    # Initialize indexing service with the configured embedding model
    indexing_service = IndexingService(embedding_model)
    
    # Initialize file service with input/output directories
    file_service = FileService(
        to_process_dir=Path(config_data['directories']['to_process_dir']),  # Source directory
        processed_dir=Path(config_data['directories']['processed_dir'])  # Archive directory
    )
    
    return document_processor, indexing_service, file_service

def main():
    """Main entry point for the application."""
    global config, doc_processor, indexing_service, file_service
    
    # Track resources that need cleanup
    server = None
    uvicorn_thread = None
    
    try:
        # Load configuration
        config = load_config(Path("config/config.json"))
        api_key, api_base = load_environment()
        
        # Setup logging
        logger = setup_logging(
            log_folder=Path(config['directories']['log_folder']),
            log_file=config['logging']['data_preparation_log'],
            max_bytes=config['logging']['max_bytes'],
            backup_count=config['logging']['backup_count'],
            log_level=config['logging']['level'],
            log_format=config['logging']['format'],
            logger_name='data_preparation'
        )
        
        # Setup directories
        setup_directories(config)
        
        # Initialize services
        doc_processor, indexing_service, file_service = initialize_services(config, api_key, api_base)
        
        # Initialize API routes
        initialize_routes(doc_processor, indexing_service, file_service, config)
        
        if os.getenv("CI") is None:
            # Launch FastAPI with port retry
            fastapi_port = None
            for port in range(8080, 8090):
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
                    if port == 8089:  # Last attempt
                        raise Exception(f"Could not find available port for FastAPI: {e}")
                    continue

            # Launch Gradio interface
            gradio_ui = GradioInterface(doc_processor, indexing_service, file_service, config)
            gradio_port = None
            
            for port in range(7860, 7870):
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
                    if port == 7869:  # Last attempt
                        raise Exception(f"Could not find available port for Gradio: {e}")
                    continue
                    
            logging.info(f"Services running - FastAPI on port {fastapi_port}, Gradio on port {gradio_port}")
            
        else:
            logging.info("CI/CD environment detected. Running processing only.")
            from .api.routes import data_prep_service
            try:
                num_processed = data_prep_service.process_documents()
                logging.info(f"Processed {num_processed} documents in CI/CD mode")
            except Exception as e:
                logging.error(f"Document processing failed in CI/CD mode: {e}")
                raise
            finally:
                # Force cleanup in CI/CD mode
                import gc
                gc.collect()
            
    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        
        # Cleanup on error
        if uvicorn_thread and uvicorn_thread.is_alive():
            logging.info("Shutting down FastAPI server thread")
            if server:
                server.should_exit = True
        
        # Force garbage collection
        import gc
        gc.collect()
        
        raise

if __name__ == "__main__":
    main() 