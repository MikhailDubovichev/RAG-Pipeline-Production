"""
API routes for the data preparation pipeline.

This module handles:
1. File upload endpoint
2. Document processing endpoint
3. Service initialization and management
"""

import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

class DataPrepService:
    """
    Service class for handling data preparation operations.
    
    This class encapsulates:
    1. Document processing logic
    2. File handling
    3. Index management
    """
    
    def __init__(self, doc_processor, indexing_service, file_service, config):
        """
        Initialize service with required components.
        
        Args:
            doc_processor: Service for processing documents
            indexing_service: Service for managing indices
            file_service: Service for file operations
            config: Configuration settings
        """
        self.doc_processor = doc_processor
        self.indexing_service = indexing_service
        self.file_service = file_service
        self.config = config

    async def upload_file(self, file: UploadFile) -> Dict[str, str]:
        """Handle file upload request."""
        try:
            # Verify file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext != '.pdf':
                raise ValueError(f"Unsupported file type: {file_ext}. Only PDF files are supported.")
                
            file_path = Path(self.config['directories']['to_process_dir']) / file.filename
            
            with open(file_path, "wb") as f:
                f.write(await file.read())
                
            logger.info(f"File {file.filename} uploaded successfully.")
            return {"message": f"File {file.filename} uploaded successfully."}
            
        except ValueError as ve:
            logger.error(f"Invalid file type: {str(ve)}")
            raise HTTPException(
                status_code=400,
                detail=str(ve)
            )
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"File upload failed: {str(e)}"
            )

    def process_documents(self) -> int:
        """
        Process new documents and update search indices.
        
        This method performs the main document processing workflow:
        1. Loads records of previously processed files
        2. Identifies new files that need processing
        3. Processes PDF documents
        4. Updates search indices with new content
        5. Moves processed files to archive
        6. Updates processing records
        
        Returns:
            int: Number of new documents processed
            
        Raises:
            Exception: Any error during processing is logged and re-raised
        """
        try:
            # Get path for record keeping
            processed_files_record_pdf = Path(self.config['directories']['processed_dir']) / "processed_files_pdf.json"

            # Load records of previously processed files
            processed_files_pdf = self.file_service.load_processed_files(processed_files_record_pdf)

            # Get all PDF files
            pdf_files = self.file_service.get_files_by_type()

            # Identify new files
            new_pdf_files = self.file_service.get_new_files(pdf_files, processed_files_pdf)

            # Process new documents
            all_new_docs = []
            
            if new_pdf_files:
                pdf_docs = self.doc_processor.load_pdf_docs(new_pdf_files)
                all_new_docs.extend(pdf_docs)

            # Update indices if we have new documents
            if all_new_docs:
                self.indexing_service.create_or_update_whoosh_index(
                    all_new_docs, 
                    Path(self.config['directories']['whoosh_index_path'])
                )
                self.indexing_service.create_or_update_faiss_index(
                    all_new_docs, 
                    Path(self.config['directories']['faiss_index_path'])
                )

                # Move processed files and update records
                self.file_service.move_to_processed(new_pdf_files)
                self.file_service.update_processed_files_record(processed_files_record_pdf, new_pdf_files)

            return len(all_new_docs)
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            raise

    def process_documents_with_response(self) -> Dict[str, str]:
        """Handle document processing request with formatted response."""
        try:
            num_processed = self.process_documents()
            return {"message": f"Processed {num_processed} new documents."}
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(e)}"
            )

# Initialize service (to be done in main.py)
data_prep_service = None

def initialize_routes(doc_proc, idx_svc, file_svc, config):
    """Initialize the data preparation service with required dependencies."""
    global data_prep_service
    data_prep_service = DataPrepService(doc_proc, idx_svc, file_svc, config)

@router.post("/upload")
async def upload_file(file: UploadFile):
    """
    Upload a file for processing.
    
    Accepts:
    - PDF files only
    """
    if not data_prep_service:
        raise HTTPException(
            status_code=500,
            detail="Data preparation service not initialized"
        )
    return await data_prep_service.upload_file(file)

@router.post("/process")
async def process_files():
    """
    Process all uploaded files.
    
    This endpoint:
    1. Processes new files in the to_process directory
    2. Updates search indices
    3. Moves processed files to archive
    """
    if not data_prep_service:
        raise HTTPException(
            status_code=500,
            detail="Data preparation service not initialized"
        )
    return data_prep_service.process_documents_with_response()

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 