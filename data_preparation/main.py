"""
Main entry point for the data preparation pipeline.

This module serves as the core of the data preparation process in the RAG (Retrieval-Augmented Generation) pipeline.
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
from common.config_utils import load_config, load_environment, setup_directories  # Configuration utilities
from common.logging_utils import setup_logging  # Logging configuration

# Initialize FastAPI app with title
app = FastAPI(title="Data Preparation Pipeline API")

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

def process_documents(doc_processor: DocumentProcessor, 
                     indexing_service: IndexingService,
                     file_service: FileService,
                     config: dict):
    """
    Core function that processes new documents and updates search indices.
    
    This function performs the main document processing workflow:
    1. Loads records of previously processed files
    2. Identifies new files that need processing
    3. Processes documents by type (PDF, Excel, PowerPoint, Word)
    4. Updates search indices with new content
    5. Moves processed files to archive
    6. Updates processing records
    
    The function maintains separate records for each file type to allow
    for type-specific processing and better error isolation.
    
    Processing Flow:
    1. Load records -> Find new files -> Process files -> Update indices
    2. After successful processing: Move files -> Update records
    
    Args:
        doc_processor (DocumentProcessor): Service for loading and chunking documents
        indexing_service (IndexingService): Service for managing search indices
        file_service (FileService): Service for file operations
        config (dict): Configuration containing directory paths
        
    Returns:
        int: Number of new documents processed
        
    Raises:
        Exception: Any error during processing is logged and re-raised
    """
    try:
        # Get paths for record keeping - separate records for each file type
        processed_files_record_pdf = Path(config['directories']['processed_dir']) / "processed_files_pdf.json"
        processed_files_record_excel = Path(config['directories']['processed_dir']) / "processed_files_excel.json"
        processed_files_record_ppt = Path(config['directories']['processed_dir']) / "processed_files_ppt.json"
        processed_files_record_doc = Path(config['directories']['processed_dir']) / "processed_files_doc.json"

        # Load records of previously processed files for each type
        processed_files_pdf = file_service.load_processed_files(processed_files_record_pdf)
        processed_files_excel = file_service.load_processed_files(processed_files_record_excel)
        processed_files_ppt = file_service.load_processed_files(processed_files_record_ppt)
        processed_files_doc = file_service.load_processed_files(processed_files_record_doc)

        # Get all files by type from the input directory
        pdf_files, excel_files, ppt_files, doc_files = file_service.get_files_by_type()

        # Identify new files that haven't been processed yet
        new_pdf_files = file_service.get_new_files(pdf_files, processed_files_pdf)
        new_excel_files = file_service.get_new_files(excel_files, processed_files_excel)
        new_ppt_files = file_service.get_new_files(ppt_files, processed_files_ppt)
        new_doc_files = file_service.get_new_files(doc_files, processed_files_doc)

        # Process new documents by type and collect all processed documents
        all_new_docs = []
        
        # Process PDFs if any new ones exist
        if new_pdf_files:
            pdf_docs = doc_processor.load_pdf_docs(new_pdf_files)
            all_new_docs.extend(pdf_docs)
            
        # Process Excel files if any new ones exist
        if new_excel_files:
            excel_docs = doc_processor.load_excel_docs(new_excel_files)
            all_new_docs.extend(excel_docs)
            
        # Process PowerPoint files if any new ones exist
        if new_ppt_files:
            ppt_docs = doc_processor.load_ppt_docs(new_ppt_files)
            all_new_docs.extend(ppt_docs)
            
        # Process Word documents if any new ones exist
        if new_doc_files:
            doc_docs = doc_processor.load_doc_docs(new_doc_files)
            all_new_docs.extend(doc_docs)

        # If we have any new documents, update indices and move files
        if all_new_docs:
            # Update both Whoosh (keyword) and FAISS (semantic) indices
            indexing_service.create_or_update_whoosh_index(
                all_new_docs, 
                Path(config['directories']['whoosh_index_path'])
            )
            indexing_service.create_or_update_faiss_index(
                all_new_docs, 
                Path(config['directories']['faiss_index_path'])
            )

            # Move all processed files to the archive directory
            file_service.move_to_processed(new_pdf_files)
            file_service.move_to_processed(new_excel_files)
            file_service.move_to_processed(new_ppt_files)
            file_service.move_to_processed(new_doc_files)

            # Update processing records for each file type
            file_service.update_processed_files_record(processed_files_record_pdf, new_pdf_files)
            file_service.update_processed_files_record(processed_files_record_excel, new_excel_files)
            file_service.update_processed_files_record(processed_files_record_ppt, new_ppt_files)
            file_service.update_processed_files_record(processed_files_record_doc, new_doc_files)

        return len(all_new_docs)
    except Exception as e:
        logging.error(f"Error in document processing: {e}")
        raise

# FastAPI endpoints for REST API access
@app.post("/upload")
async def upload_file(file: UploadFile):
    """
    FastAPI endpoint for uploading a single file to the processing pipeline.
    
    This endpoint:
    1. Receives a file through HTTP POST request
    2. Saves it to the to_process directory
    3. Returns a success/error message
    
    The file will be processed later when the /process endpoint is called.
    
    Args:
        file (UploadFile): File object from the HTTP request
            Supported formats: PDF, Excel (.xls, .xlsx), 
            PowerPoint (.ppt, .pptx), Word (.doc, .docx)
            
    Returns:
        dict: Message indicating success or failure
        
    HTTP Status Codes:
        200: File uploaded successfully
        500: Upload failed (with error details)
    """
    try:
        # Create full path for saving the file
        file_path = Path(config['directories']['to_process_dir']) / file.filename
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        logging.info(f"File {file.filename} uploaded successfully.")
        return {"message": f"File {file.filename} uploaded successfully."}
        
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"File upload failed: {str(e)}"}
        )

@app.post("/process")
async def process_files():
    """
    FastAPI endpoint that triggers the document processing pipeline.
    
    This endpoint:
    1. Processes all new files in the to_process directory
    2. Updates search indices with new content
    3. Moves processed files to the archive
    
    The processing includes:
    - Text extraction from documents
    - Chunking the text into smaller pieces
    - Creating/updating search indices
    - Moving processed files to archive
    
    Returns:
        dict: Message indicating number of documents processed
        
    HTTP Status Codes:
        200: Processing completed successfully
        500: Processing failed (with error details)
    """
    try:
        # Process all new documents and get count
        num_processed = process_documents(doc_processor, indexing_service, file_service, config)
        return {"message": f"Processed {num_processed} new documents."}
        
    except Exception as e:
        logging.error(f"Error processing documents: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Document processing failed: {str(e)}"}
        )

def main():
    global config, doc_processor, indexing_service, file_service
    
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
            process_documents(doc_processor, indexing_service, file_service, config)
            
    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        raise

if __name__ == "__main__":
    main() 