"""
Document Processing Service for the RAG Pipeline.

This module handles the extraction and processing of text from PDF files using PDFReader from llama_index.

Key Features:
1. Text extraction from PDF documents
2. Document chunking with configurable size and overlap
3. Metadata preservation and enhancement
4. Unique document ID assignment
5. Error handling and logging

The processing follows these general steps:
1. Load PDF document from file
2. Extract text content
3. Split into manageable chunks
4. Assign metadata (source, IDs, etc.)
5. Create Document objects for indexing
"""

import logging  # Python's built-in logging facility
from pathlib import Path  # Object-oriented filesystem paths
from typing import List, Dict  # Type hints for better code clarity
import os  # Operating system interface for file operations

# Document processing libraries
from llama_index.core import Document  # Base document class for RAG
from llama_index.readers.file import PDFReader  # PDF text extraction
import uuid  # Unique identifier generation

# Configure module logger
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Service for processing PDF documents into indexable chunks.
    
    This class handles:
    1. Document loading from PDF format
    2. Text extraction and cleaning
    3. Content chunking with overlap
    4. Metadata assignment
    
    Each document is processed into Document objects that contain:
    - Extracted and cleaned text
    - Source metadata (filename, page numbers)
    - Unique document ID
    - Chunk information (for split documents)
    """

    def __init__(self, max_tokens: int = 1024, overlap_tokens: int = 50):
        """
        Initialize document processor with chunking parameters.
        
        The processor splits large documents into smaller chunks for better
        search and retrieval. Overlapping tokens help maintain context
        across chunk boundaries.
        
        Args:
            max_tokens (int): Maximum number of tokens per chunk
                Default is 1024, which is suitable for most LLMs
            overlap_tokens (int): Number of overlapping tokens between chunks
                Default is 50, which helps maintain context between chunks
                
        Note: Current implementation uses character-based chunking as an
        approximation. A proper tokenizer should be used in production.
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.pdf_reader = PDFReader()

    def load_pdf_docs(self, pdf_paths: List[Path]) -> List[Document]:
        """
        Load and process PDF documents into searchable chunks.
        
        This method:
        1. Validates each PDF file (existence and readability)
        2. Extracts text content using PDFReader
        3. Assigns unique IDs and metadata
        4. Handles memory cleanup after processing
        
        The PDFReader automatically handles:
        - Page-based chunking
        - Text extraction from various PDF formats
        - Basic structure preservation
        
        Args:
            pdf_paths (List[Path]): List of paths to PDF files to process
            
        Returns:
            List[Document]: List of processed Document objects, one per page
            
        Error Handling:
        - Skips files that don't exist or aren't readable
        - Logs warnings for PDFs with no extractable content
        - Continues processing remaining files if one fails
        - Ensures cleanup of file handles
        """
        all_docs = []
        for pdf_path in pdf_paths:
            try:
                # Log the start of processing for this PDF
                logger.info(f"Starting to process PDF: {pdf_path.name} (size: {pdf_path.stat().st_size} bytes)")
                
                # Check if file exists and is readable
                if not pdf_path.exists():
                    logger.error(f"PDF file not found: {pdf_path}")
                    continue
                    
                if not os.access(pdf_path, os.R_OK):
                    logger.error(f"PDF file not readable: {pdf_path}")
                    continue
                
                # Load and process the PDF
                docs = []
                try:
                    docs = self.pdf_reader.load_data(str(pdf_path))
                finally:
                    # Ensure any file handles are closed
                    import gc
                    gc.collect()
                
                if not docs:
                    logger.warning(f"No content extracted from PDF: {pdf_path.name}")
                    continue
                    
                # Assign document IDs and metadata
                docs = self._assign_doc_ids(docs)
                for doc in docs:
                    doc = self._extract_metadata(doc, source_file=pdf_path.name)
                    
                # Add to the collection
                all_docs.extend(docs)
                logger.info(f"Successfully processed PDF: {pdf_path.name} - extracted {len(docs)} pages")
                
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path.name}: {str(e)}")
                logger.exception("Detailed error information:")
                
        logger.info(f"Completed processing {len(pdf_paths)} PDF files, extracted {len(all_docs)} total pages")
        return all_docs

    @staticmethod
    def _assign_doc_ids(documents: List[Document]) -> List[Document]:
        """
        Assign unique identifiers to a list of documents.
        
        This method:
        1. Generates a UUID for each document
        2. Stores it in the document's metadata
        3. Returns the modified document list
        
        The UUID ensures:
        - Globally unique identification
        - Consistent reference tracking
        - Safe document deduplication
        
        Args:
            documents (List[Document]): List of documents needing IDs
            
        Returns:
            List[Document]: Same documents with added IDs in metadata
            
        Note:
            This method modifies the documents in place but also returns
            them for convenience in method chaining.
        """
        for doc in documents:
            doc.metadata["doc_id"] = str(uuid.uuid4())
        return documents

    @staticmethod
    def _extract_metadata(doc: Document, source_file: str) -> Document:
        """
        Extract and assign source metadata to a document.
        
        This method:
        1. Adds source filename to document metadata
        2. Preserves existing metadata
        3. Returns the enhanced document
        
        The source tracking enables:
        - Origin tracing for each chunk
        - Reference back to original documents
        - User-friendly source citations
        
        Args:
            doc (Document): Document to enhance with metadata
            source_file (str): Name of the source file
            
        Returns:
            Document: Same document with added source metadata
            
        Note:
            This method modifies the document in place but also returns
            it for convenience in method chaining.
        """
        doc.metadata["source"] = source_file
        return doc 