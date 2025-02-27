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

    def load_pdf_docs(self, pdf_paths: List[Path]) -> tuple[List[Document], List[tuple[Path, str]]]:
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
            tuple: (List[Document], List[tuple[Path, str]]) - Successful and failed documents
                - First element is a list of processed Document objects, one per page
                - Second element is a list of tuples containing (file_path, error_message) for failed files
            
        Error Handling:
        - Skips files that don't exist or aren't readable
        - Logs warnings for PDFs with no extractable content
        - Continues processing remaining files if one fails
        - Ensures cleanup of file handles
        """
        all_docs = []
        failed_files = []
        
        if not pdf_paths:
            logger.info("No PDF files to process")
            return all_docs, failed_files
        
        # Process files one at a time to manage memory usage
        for pdf_path in pdf_paths:
            try:
                # Log the start of processing for this PDF
                logger.info(f"Starting to process PDF: {pdf_path.name} (size: {pdf_path.stat().st_size} bytes)")
                
                # Check if file exists and is readable
                if not pdf_path.exists():
                    error_msg = f"PDF file not found: {pdf_path}"
                    logger.error(error_msg)
                    failed_files.append((pdf_path, error_msg))
                    continue
                    
                if not os.access(pdf_path, os.R_OK):
                    error_msg = f"PDF file not readable: {pdf_path}"
                    logger.error(error_msg)
                    failed_files.append((pdf_path, error_msg))
                    continue
                
                # Check file size to prevent memory issues with very large files
                file_size_mb = pdf_path.stat().st_size / (1024 * 1024)  # Convert to MB
                if file_size_mb > 100:  # 100MB threshold
                    logger.warning(f"Large PDF detected: {pdf_path.name} ({file_size_mb:.2f} MB). This may require significant memory.")
                
                # Load and process the PDF
                docs = []
                try:
                    # Use a separate function call to isolate memory usage
                    docs = self._load_single_pdf(pdf_path)
                except Exception as e:
                    error_msg = f"Error extracting text from PDF: {e}"
                    logger.error(error_msg)
                    logger.exception("Detailed error information:")
                    failed_files.append((pdf_path, error_msg))
                    continue
                
                if not docs:
                    error_msg = f"No content extracted from PDF: {pdf_path.name}"
                    logger.warning(error_msg)
                    failed_files.append((pdf_path, error_msg))
                    continue
                    
                # Assign document IDs and metadata
                try:
                    docs = self._assign_doc_ids(docs)
                    for doc in docs:
                        doc = self._extract_metadata(doc, source_file=pdf_path.name)
                        
                    # Add to the collection
                    all_docs.extend(docs)
                    logger.info(f"Successfully processed PDF: {pdf_path.name} - extracted {len(docs)} pages")
                except Exception as e:
                    error_msg = f"Error assigning metadata to documents: {e}"
                    logger.error(error_msg)
                    logger.exception("Detailed error information:")
                    failed_files.append((pdf_path, error_msg))
                
            except Exception as e:
                error_msg = f"Failed to process PDF {pdf_path.name}: {str(e)}"
                logger.error(error_msg)
                logger.exception("Detailed error information:")
                failed_files.append((pdf_path, error_msg))
            
            # Force garbage collection after each file to free memory
            import gc
            gc.collect()
                
        logger.info(f"Completed processing {len(pdf_paths)} PDF files, extracted {len(all_docs)} total pages")
        logger.info(f"Successfully processed {len(pdf_paths) - len(failed_files)} files, failed to process {len(failed_files)} files")
        return all_docs, failed_files
        
    def _load_single_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load a single PDF file and extract its content.
        
        This helper method isolates the memory-intensive PDF loading operation
        to better manage resources and enable garbage collection between files.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            List[Document]: List of document objects, one per page
            
        Raises:
            Exception: If PDF loading fails
        """
        try:
            docs = self.pdf_reader.load_data(str(pdf_path))
            return docs
        finally:
            # Ensure any file handles are closed
            import gc
            gc.collect()

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
        
        # Extract section information from the content if possible
        DocumentProcessor._extract_section_info(doc)
        
        return doc
        
    @staticmethod
    def _extract_section_info(doc: Document) -> None:
        """
        Extract section information from document content.
        
        This method analyzes the document content to identify section headings
        and distinguishes them from document titles. It adds this information
        to the document's metadata.
        
        The method:
        1. Examines the first few lines of text
        2. Identifies potential section headings based on formatting patterns
        3. Distinguishes between document titles and actual section headings
        4. Adds the section information to document metadata
        
        Args:
            doc (Document): Document to extract section information from
            
        Note:
            This method modifies the document in place by adding section
            information to its metadata.
        """
        if not doc.text:
            return
            
        lines = doc.text.split('\n')
        if not lines:
            return
            
        # Skip empty lines at the beginning
        start_idx = 0
        while start_idx < len(lines) and not lines[start_idx].strip():
            start_idx += 1
            
        if start_idx >= len(lines):
            return
            
        # First non-empty line is often the document title or section heading
        first_line = lines[start_idx].strip()
        
        # Check if this is likely a document title rather than a section
        is_likely_doc_title = False
        
        # Document titles are often short and may be followed by metadata like dates
        if len(first_line) < 30 and len(lines) > start_idx + 1:
            # If next line is a page number, this is likely a header/title pattern
            if lines[start_idx + 1].strip().isdigit():
                is_likely_doc_title = True
                
            # If there's a blank line after the title followed by actual content
            elif (start_idx + 2 < len(lines) and 
                  not lines[start_idx + 1].strip() and 
                  lines[start_idx + 2].strip()):
                is_likely_doc_title = True
        
        # If we believe this is a document title, look for actual section headings
        if is_likely_doc_title:
            # Store the document title
            doc.metadata["document_title"] = first_line
            
            # Look for actual section headings in the content
            for i in range(start_idx + 2, min(start_idx + 10, len(lines))):
                line = lines[i].strip()
                if line and len(line) < 50:
                    # Potential section heading - look for patterns
                    if (line.isupper() or  # ALL CAPS
                        line.endswith(':') or  # Ends with colon
                        (i+1 < len(lines) and not lines[i+1].strip())):  # Followed by blank line
                        doc.metadata["section"] = line
                        return
        else:
            # If not a document title, treat as section
            doc.metadata["section"] = first_line 