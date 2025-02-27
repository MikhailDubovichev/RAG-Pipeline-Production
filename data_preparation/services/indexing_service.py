"""
Indexing Service for the RAG Pipeline.

This module creates and maintains two types of search indices:
1. Whoosh: For keyword-based (full-text) search
2. FAISS: For semantic (vector) search

Key Features:
1. Dual index maintenance (Whoosh + FAISS)
2. Automatic dimension detection for embeddings
3. Rich metadata preservation for PDF documents
4. Incremental index updates
5. Error handling and logging

The service handles:
- Index creation and initialization
- PDF document addition and updates
- Metadata field management (page numbers, sections)
- Index persistence
- Embedding generation

Technical Details:
- Whoosh provides inverted index for fast keyword search
- FAISS (Facebook AI Similarity Search) enables efficient vector search
- Both indices are persisted to disk for reuse
- Supports incremental updates without full reindexing
"""

import logging  # Python's built-in logging facility
from pathlib import Path  # Object-oriented filesystem paths
from typing import List  # Type hints for better code clarity
import os  # Operating system interface for file operations

# Vector search components
import faiss  # Facebook AI Similarity Search library
from llama_index.core import (
    Document,  # Base document class
    StorageContext,  # Index storage management
    VectorStoreIndex  # High-level vector index interface
)
from llama_index.vector_stores.faiss import FaissVectorStore  # FAISS integration
from llama_index.core.storage.docstore import SimpleDocumentStore  # Document storage
from llama_index.core.storage.index_store import SimpleIndexStore  # Index metadata storage

# Keyword search components
from whoosh.index import (
    create_in,  # Create new index
    open_dir,  # Open existing index
    exists_in  # Check index existence
)
from whoosh.fields import (
    Schema,  # Index schema definition
    TEXT,  # Full-text searchable field
    ID,  # Unique identifier field
    NUMERIC  # Number field (for sorting)
)

# Configure module logger
logger = logging.getLogger(__name__)

class IndexingService:
    """
    Service for creating and maintaining search indices for PDF documents.
    
    This service manages two types of indices:
    1. Whoosh Index:
       - Full-text search capability
       - PDF metadata indexing (page numbers, sections)
       - Field-specific queries
       - Sorted results
       
    2. FAISS Index:
       - Semantic similarity search
       - Fast vector operations
       - Approximate nearest neighbors
       - Scalable to millions of documents
       
    The service ensures:
    - Consistent PDF document processing
    - Safe concurrent access
    - Proper persistence
    - Error recovery
    """

    def __init__(self, embedding_model):
        """
        Initialize indexing service with an embedding model.
        
        The service needs an embedding model to:
        1. Convert text to vectors for FAISS
        2. Determine vector dimensions automatically
        3. Ensure consistent vector representations
        
        Args:
            embedding_model: Model instance (e.g., NebiusEmbedding) that:
                - Has get_text_embedding method
                - Returns consistent vector dimensions
                - Handles text preprocessing
                
        Note:
            The embedding dimension is determined automatically by
            running a test embedding during initialization.
        """
        self.embedding_model = embedding_model
        self.embedding_dimension = self._get_embedding_dimension()

    def _get_embedding_dimension(self) -> int:
        """
        Determine the dimension of embeddings produced by the model.
        
        This method:
        1. Creates a sample embedding from test text
        2. Measures the vector dimension
        3. Returns the dimension for FAISS initialization
        
        The dimension is critical for:
        - Initializing FAISS index correctly
        - Ensuring vector compatibility
        - Optimizing memory usage
        
        Returns:
            int: Number of dimensions in the embedding vectors
            
        Note:
            This is called during initialization to automatically
            configure FAISS for the specific embedding model.
        """
        sample_text = "test text to create an embedding"
        sample_embedding = self.embedding_model.get_text_embedding(sample_text)
        return len(sample_embedding)

    def create_or_update_whoosh_index(self, documents: List[Document], index_dir: Path) -> tuple[bool, str, List[str]]:
        """
        Create or update the Whoosh index for keyword search.
        
        This method handles:
        1. Schema definition with rich metadata
        2. Index creation or opening
        3. Document addition with metadata
        4. Safe updates and persistence
        
        Schema Fields:
        - content: Full text content (searchable)
        - source: Document source/filename
        - section: Document section/heading
        - chunk_number: Position in chunked document
        - total_chunks_in_section: Total chunks in section
        - page_number: Page number in PDFs
        - doc_id: Unique document identifier
        
        Args:
            documents (List[Document]): Documents to index
            index_dir (Path): Directory for index storage
            
        Returns:
            tuple: (bool, str, List[str]) - Success status, error message, and failed document IDs
                - First element is True if indexing was successful, False otherwise
                - Second element is an error message if indexing failed, empty string otherwise
                - Third element is a list of document IDs that failed to be indexed
            
        Note:
            - Creates directory if it doesn't exist
            - Updates existing documents based on doc_id
            - Preserves all metadata fields
            - Handles missing metadata gracefully
        """
        # Define the index schema with all possible fields
        schema = Schema(
            content=TEXT(stored=True),  # Main searchable content
            source=ID(stored=True),  # Source filename
            section=TEXT(stored=True),  # Document section
            chunk_number=NUMERIC(stored=True, sortable=True),  # Chunk sequence
            total_chunks_in_section=NUMERIC(stored=True, sortable=True),  # Total chunks
            page_number=NUMERIC(stored=True, sortable=True),  # Page in PDF
            doc_id=ID(stored=True, unique=True),  # Unique identifier
        )

        failed_docs = []
        
        try:
            # Ensure index directory exists
            try:
                os.makedirs(index_dir, exist_ok=True)
                logger.info(f"Ensured index directory exists: {index_dir}")
            except PermissionError as e:
                error_msg = f"Permission denied when creating index directory: {e}"
                logger.error(error_msg)
                return False, error_msg, []
            except OSError as e:
                error_msg = f"OS error when creating index directory: {e}"
                logger.error(error_msg)
                return False, error_msg, []

            # Create new index or open existing
            try:
                if not exists_in(index_dir):
                    idx = create_in(index_dir, schema)
                    logger.info(f"Created new Whoosh index at {index_dir}")
                else:
                    idx = open_dir(index_dir)
                    logger.info(f"Opened existing Whoosh index at {index_dir}")
            except Exception as e:
                error_msg = f"Failed to create/open Whoosh index: {e}"
                logger.error(error_msg)
                logger.exception("Detailed error information:")
                return False, error_msg, []

            # Get index writer for updates
            try:
                writer = idx.writer()
            except Exception as e:
                error_msg = f"Failed to get index writer: {e}"
                logger.error(error_msg)
                logger.exception("Detailed error information:")
                return False, error_msg, []

            # Process each document
            for doc in documents:
                try:
                    # Prepare base document fields
                    doc_fields = {
                        "content": doc.text,
                        "source": doc.metadata.get("source", "unknown_source"),
                        "doc_id": doc.metadata.get("doc_id", ""),
                    }

                    # Add optional metadata fields if present
                    for field in ["section", "chunk_number", "total_chunks_in_section", "page_number"]:
                        if field in doc.metadata:
                            doc_fields[field] = doc.metadata[field]

                    # Update document in index
                    writer.update_document(**doc_fields)
                except Exception as e:
                    doc_id = doc.metadata.get('doc_id', 'unknown')
                    error_msg = f"Failed to index document {doc_id}: {e}"
                    logger.error(error_msg)
                    failed_docs.append(doc_id)

            # Commit changes to index
            try:
                writer.commit()
                logger.info("Whoosh index updated successfully")
            except Exception as e:
                error_msg = f"Failed to commit changes to Whoosh index: {e}"
                logger.error(error_msg)
                logger.exception("Detailed error information:")
                return False, error_msg, failed_docs

            if failed_docs:
                return True, f"Indexed successfully with {len(failed_docs)} document failures", failed_docs
            else:
                return True, "", []

        except Exception as e:
            error_msg = f"Failed to create/update Whoosh index: {e}"
            logger.error(error_msg)
            logger.exception("Detailed error information:")
            return False, error_msg, failed_docs

    def create_or_update_faiss_index(self, documents: List[Document], index_path: Path) -> tuple[bool, str]:
        """
        Create or update the FAISS index for semantic search.
        
        Args:
            documents (List[Document]): Documents to index
            index_path (Path): Directory for index storage
            
        Returns:
            tuple: (bool, str) - Success status and error message if any
        """
        try:
            faiss_file = "default__vector_store.faiss"
            logging.info(f"Attempting to create/update FAISS index at {index_path / faiss_file}")
            
            # Create directory if it doesn't exist
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Process documents in batches to manage memory
            batch_size = 100  # Adjust based on document size and available memory
            
            # Check for existing index
            if (index_path / faiss_file).exists():
                logging.info(f"Found existing FAISS index at {index_path / faiss_file}")
                try:
                    # Load existing index
                    vector_store = FaissVectorStore.from_persist_path(
                        str(index_path / faiss_file)
                    )
                    storage_context = StorageContext.from_defaults(
                        vector_store=vector_store,
                        docstore=SimpleDocumentStore(),
                        index_store=SimpleIndexStore(),
                        persist_dir=str(index_path)
                    )
                    
                    # Process documents in batches
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        index = VectorStoreIndex.from_documents(
                            batch,
                            storage_context=storage_context,
                            embedding=self.embedding_model
                        )
                        # Force garbage collection after each batch
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    logging.error(f"Failed to load existing index: {e}")
                    # Fall through to create new index
            else:
                logging.info("No existing index found, creating new one")
                
                # Create new index
                vector_store = FaissVectorStore(
                    faiss.IndexFlatIP(self.embedding_dimension)
                )
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    docstore=SimpleDocumentStore(),
                    index_store=SimpleIndexStore(),
                    persist_dir=str(index_path)
                )
                
                # Process documents in batches
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    index = VectorStoreIndex.from_documents(
                        batch,
                        storage_context=storage_context,
                        embedding=self.embedding_model
                    )
                    # Force garbage collection after each batch
                    import gc
                    gc.collect()
            
            # Persist the index
            if vector_store:
                vector_store.persist(persist_path=str(index_path / faiss_file))
            if storage_context:
                storage_context.persist(persist_dir=str(index_path))
                
            # Clean up any duplicate vector store in JSON format if it exists
            json_file = index_path / "default__vector_store.json"
            if json_file.exists():
                json_file.unlink()
                
            return True, ""

        except Exception as e:
            error_msg = f"Failed to create/update FAISS index: {str(e)}"
            logging.error(error_msg)
            
            # Ensure cleanup even on error
            import gc
            gc.collect()
            
            return False, error_msg 